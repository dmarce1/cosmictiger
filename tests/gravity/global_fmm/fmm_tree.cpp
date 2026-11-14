/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <hpx/serialization/array.hpp>
#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/fmm_types.hpp>
#include <cosmictiger/gravity/global_fmm/distributed_tree.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/gravity/tree_build.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/particles/particles.hpp>

namespace detail {
constexpr int kFmmTreeParticlesPerRank = 8;

void requireFmmTree(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(std::string("FMM tree regression: ") + message);
    }
}

Vector3<Position> fmmTreePosition(int rank, int particle) {
    Vector3<Position> position;
    position[kXi] = 0.25 + 0.125 * (rank + (particle & 1)) / hpxSize();
    position[kYi] = 0.25 + 0.0625 * ((particle >> 1) & 1);
    position[kZi] = 0.25 + 0.0625 * ((particle >> 2) & 1);
    if (particle == 7) {
        position[kYi] = 0.25 + 0.03125;
    }
    // X/Z extrema fall between stored box coordinates; Y remains exactly aligned.
    position[kXi].raw() += 1;
    position[kZi].raw() += 3;
    return position;
}

template <class SpatialValue>
std::vector<char> vectorArchiveBytes(SpatialValue const& value, uint32_t flags) {
    std::vector<char> bytes;
    hpx::serialization::output_archive archive(bytes, flags);
    archive << value;
    return bytes;
}

template <class LegacyVector, class Real>
void verifyVectorArchiveCompatibility(Vector3<Real> const& expected, uint32_t flags) {
    LegacyVector legacy;
    for (int dim = 0; dim < kDimensions; ++dim) {
        legacy[dim] = expected[dim];
    }
    auto const legacyBytes = vectorArchiveBytes(legacy, flags);
    auto const vectorBytes = vectorArchiveBytes(expected, flags);
    requireFmmTree(legacyBytes == vectorBytes, "Vector3 changed the legacy vector archive bytes");
    Vector3<Real> vectorFromLegacy{};
    LegacyVector legacyFromVector{};
    {
        hpx::serialization::input_archive archive(legacyBytes);
        archive >> vectorFromLegacy;
    }
    {
        hpx::serialization::input_archive archive(vectorBytes);
        archive >> legacyFromVector;
    }
    for (int dim = 0; dim < kDimensions; ++dim) {
        requireFmmTree(
            vectorFromLegacy[dim] == expected[dim] && legacyFromVector[dim] == expected[dim],
            "Vector3 and legacy vectors cannot deserialize each other's archives");
    }
}

void verifyVectorArchives() {
    Vector3<double> const values(0.125, -2.75, 19.5);
    Vector3<Position> positions(Position(0.125), Position(0.5), Position(0.875));
    positions[0].raw() += 1;
    positions[2].raw() -= 1;
    for (uint32_t flags :
        {uint32_t(0), uint32_t(hpx::serialization::archive_flags::disable_array_optimization)}) {
        verifyVectorArchiveCompatibility<Array<double, kDimensions>>(values, flags);
        verifyVectorArchiveCompatibility<std::array<double, kDimensions>>(values, flags);
        verifyVectorArchiveCompatibility<Array<Position, kDimensions>>(positions, flags);
    }
}

void verifyTreeReturnArchive() {
    TreeCreateReturn expected;
    expected.id.proc = 3;
    expected.id.index = 17;
    expected.radius = 0.125f;
    expected.centerOfMassOffset = {1.e-11, -2.e-11, 3.e-11};
    // Source moments retain double precision across locality boundaries.
    int const coefficientCount = fmmMultipoleCoefficientCount(kFmmOrder);
    for (int i = 0; i < coefficientCount; i++) {
        expected.multi[i] = double(i + 1) + 1.0e-10;
    }
    for (int dim = 0; dim < kDimensions; dim++) {
        expected.pos[dim] = 0.25 + 0.0625 * dim;
        expected.box.begin[dim] = 0.125 + 0.0625 * dim;
        expected.box.end[dim] = 0.5 + 0.0625 * dim;
    }
    std::vector<char> bytes;
    {
        hpx::serialization::output_archive archive(bytes);
        archive << expected;
    }
    TreeCreateReturn actual;
    for (int dim = 0; dim < kDimensions; dim++) {
        actual.box.begin[dim] = actual.box.end[dim] = -1.0;
    }
    {
        hpx::serialization::input_archive archive(bytes);
        archive >> actual;
    }
    requireFmmTree(actual.id == expected.id && actual.radius == expected.radius,
        "tree-return metadata did not survive the HPX archive");
    if constexpr (cosmictiger::gravity::cartesian::kDipoleFree) {
        for (int dim = 0; dim < kDimensions; ++dim) {
            requireFmmTree(actual.centerOfMassOffset[dim] == expected.centerOfMassOffset[dim],
                "COM rounding residual did not survive the HPX archive");
        }
    }
    for (int i = 0; i < coefficientCount; i++) {
        requireFmmTree(
            actual.multi[i] == expected.multi[i], "multipole did not survive the HPX archive");
    }
    for (int dim = 0; dim < kDimensions; dim++) {
        requireFmmTree(actual.pos[dim] == expected.pos[dim] &&
                actual.box.begin[dim] == expected.box.begin[dim] &&
                actual.box.end[dim] == expected.box.end[dim],
            "tree-return position or bounds did not survive the HPX archive");
    }
}

void verifyTreeNodeArchive(TreeId id) {
    // Node loading allocates a multipole in the live tree arena, so this check
    // runs after tree construction. Do not alter the production node itself.
    TreeNode const expected = *treeGetNode(id);
    std::vector<char> bytes;
    {
        hpx::serialization::output_archive archive(bytes);
        archive << expected;
    }
    TreeNode actual;
    {
        hpx::serialization::input_archive archive(bytes);
        archive >> actual;
    }
    requireFmmTree(actual.radius == expected.radius &&
            actual.particleRange.first == expected.particleRange.first &&
            actual.particleRange.second == expected.particleRange.second &&
            actual.processRange.first == expected.processRange.first &&
            actual.processRange.second == expected.processRange.second &&
            actual.depth == expected.depth && actual.localRoot == expected.localRoot &&
            actual.leaf == expected.leaf,
        "tree-node metadata did not survive the HPX archive");
    for (int child = 0; child < kChildCount; ++child) {
        requireFmmTree(actual.children[child] == expected.children[child],
            "tree-node child did not survive the HPX archive");
    }
    for (int dim = 0; dim < kDimensions; ++dim) {
        requireFmmTree(actual.box.begin[dim] == expected.box.begin[dim] &&
                actual.box.end[dim] == expected.box.end[dim] &&
                actual.pos[dim] == expected.pos[dim] &&
                actual.mpos->pos[dim] == expected.mpos->pos[dim],
            "tree-node bounds or position did not survive the HPX archive");
    }
    MultipoleBuffer actualMultipole;
    MultipoleBuffer expectedMultipole;
    treeReadMultipole(actual.mpos, actualMultipole);
    treeReadMultipole(expected.mpos, expectedMultipole);
    for (int i = 0; i < fmmMultipoleCoefficientCount(kFmmOrder); ++i) {
        requireFmmTree(actualMultipole[i] == expectedMultipole[i],
            "tree-node multipole did not survive the HPX archive");
    }

    // The remote cache sends vectors, whose optimized serialization path can
    // bypass a trivially copyable element's own save/load implementation.
    // Use a nonempty node and the ordinary archive flags so raw mpos transport
    // cannot hide behind the shared empty-multipole record or disabled fast path.
    Vector<TreeNode> const expectedLine{expected, expected};
    bytes.clear();
    {
        hpx::serialization::output_archive archive(bytes);
        archive << expectedLine;
    }
    Vector<TreeNode> actualLine;
    {
        hpx::serialization::input_archive archive(bytes);
        archive >> actualLine;
    }
    requireFmmTree(actualLine.size() == expectedLine.size(),
        "tree cache-line size did not survive the HPX archive");
    for (auto const& received : actualLine) {
        requireFmmTree(received.mpos != expected.mpos,
            "tree cache-line archive copied a process-local multipole pointer");
        treeReadMultipole(received.mpos, actualMultipole);
        for (int i = 0; i < fmmMultipoleCoefficientCount(kFmmOrder); ++i) {
            requireFmmTree(actualMultipole[i] == expectedMultipole[i],
                "tree cache-line multipole did not survive the HPX archive");
        }
    }
}

void verifyRemoteMultipoleStorage() {
    auto const& opts = getOptions();
    // Grow past a remote chunk, then reserve fresh local allocator lines. A
    // remote record must survive both operations without sharing local slots.
    constexpr size_t kCount = 4101;
    Vector3<Position> position{};
    position[0] = 0.375;
    MultipoleBuffer multipole;
    multipole.zero();
    std::vector<MultipolePosition*> remote;
    remote.reserve(kCount);
    for (size_t index = 0; index < kCount; ++index) {
        multipole[0] = float(index + 1);
        remote.push_back(treeAddMultipole(position, multipole));
    }
    TreeNode node{};
    multipole.zero();
    size_t const localCount = 2 * size_t(opts.compute.treeAllocLineSize) + kCount;
    for (size_t index = 0; index < localCount; ++index) {
        treeStoreNode(treeAllocateNode(), node, multipole);
    }
    for (size_t index = 0; index < kCount; ++index) {
        treeReadMultipole(remote[index], multipole);
        requireFmmTree(multipole[0] == float(index + 1),
            "local arena growth overwrote a deserialized remote multipole");
        requireFmmTree(remote[index]->pos[0] == position[0],
            "remote multipole position did not survive chunk growth");
    }
}

struct TreeCloud {
    std::vector<Vector3<Position>> positions;
};

/// Check every stored center and radius against the particles below it.
TreeCloud verifyMassCenters(TreeId id) {
    auto const* node = treeGetNode(id);
    TreeCloud cloud;
    if (node->leaf) {
        // This fixture is distributed by disjoint kd boxes; exact leaf bounds
        // identify its known particles without relying on remote particle arrays.
        for (int rank = 0; rank < hpxSize(); ++rank) {
            for (int i = 0; i < kFmmTreeParticlesPerRank; ++i) {
                auto const position = fmmTreePosition(rank, i);
                bool inside = true;
                for (int dim = 0; dim < kDimensions; ++dim) {
                    inside = inside &&
                        position[dim].toDouble() >= node->box.begin[dim].toDouble() &&
                        position[dim].toDouble() <= node->box.end[dim].toDouble();
                }
                if (inside) {
                    cloud.positions.push_back(position);
                }
            }
        }
    } else {
        for (auto const child : node->children) {
            auto subset = verifyMassCenters(child);
            cloud.positions.insert(
                cloud.positions.end(), subset.positions.begin(), subset.positions.end());
        }
    }
    auto const mass = fmmMultiPos(node->mpos).multi[0];
    requireFmmTree(
        cloud.positions.size() == size_t(mass), "COM fixture particle ownership mismatch");
    if (!cloud.positions.empty()) {
        Vector3<long double> moment{};
        for (auto const& position : cloud.positions) {
            Vector3<double> displacement{};
            for (int dim = 0; dim < kDimensions; ++dim) {
                displacement[dim] = unwrappedDoubleDistance(position[dim], node->pos[dim]);
                moment[dim] += displacement[dim];
            }
            requireFmmTree(displacement.norm() <= node->radius, "COM sphere misses a particle");
        }
        for (int dim = 0; dim < kDimensions; ++dim) {
            requireFmmTree(std::abs(moment[dim] / mass) <= 2 * Position::kToFloatFactor_ + 1.e-16,
                "tree node is not centered at its particle center of mass");
        }
    }
    return cloud;
}

void verifyTreeGeometry(TreeId id, bool empty) {
    TreeNode const* node = treeGetNode(id);
    requireFmmTree(
        std::isfinite(node->radius) && node->radius >= 0.0f, "node has an invalid bounding radius");
    for (int dim = 0; dim < kDimensions; dim++) {
        requireFmmTree(
            node->box.begin[dim] <= node->box.end[dim], "node has reversed particle bounds");
    }
    [&]() {
        auto const& multipole = fmmMultiPos(node->mpos).multi;
        for (int i = 0; i < FmmOrder<kFmmOrder>::kMultipoleCoefficients_; i++) {
            requireFmmTree(std::isfinite(multipole[i]), "node has a nonfinite multipole");
            if (empty) {
                requireFmmTree(multipole[i] == 0.0f, "empty node has a nonzero multipole");
            }
        }
    }();
    if constexpr (cosmictiger::gravity::cartesian::kDipoleFree) {
        auto const& multipole = fmmMultiPos(node->mpos).multi;
        for (int i = 1; i < 4; ++i) {
            requireFmmTree(multipole[i] == 0.0f, "mass-centered node retained a dipole");
        }
    }
    if (empty) {
        requireFmmTree(node->radius == 0.0f, "empty node has a nonzero radius");
    }
    if (!node->leaf) {
        auto const children = node->children;
        for (int child = 0; child < kChildCount; child++) {
            verifyTreeGeometry(children[child], empty);
        }
    }
}
}    // namespace detail

void initializeFmmTreeRegression(bool empty) {
    particlesResize(empty ? 0 : detail::kFmmTreeParticlesPerRank);
    for (ParticleInteger i = 0; i < particlesSize(); i++) {
        auto const position = detail::fmmTreePosition(hpxRank(), i);
        for (int dim = 0; dim < kDimensions; dim++) {
            particlesPos(dim, i) = position[dim];
            particlesVel(dim, i) = 0.0f;
        }
        particlesRung(i) = 0;
    }
}
HPX_PLAIN_ACTION(initializeFmmTreeRegression, InitializeFmmTreeRegressionAction);

void fmmTreeTest() {
    detail::verifyVectorArchives();
    detail::verifyTreeReturnArchive();
    auto options = getOptions();
    options.hydro.enabled = false;

    options.analysis.groups = false;
    options.analysis.tracers = false;
    options.compute.bucketSize = 4;
    options.initial.numberOfParticles = detail::kFmmTreeParticlesPerRank * hpxSize();
    options.gravity.smoothingLength = 0.001;
    setOptions(options);
    auto const initialize = [](bool empty) {
        Vector<hpx::future<void>> pending;
        for (auto const& locality : hpxLocalities()) {
            pending.push_back(hpx::async<InitializeFmmTreeRegressionAction>(locality, empty));
        }
        for (auto& future : pending) {
            future.get();
        }
        particlesSortByRung(0);
    };
    initialize(false);
    domainsRebound();
    domainsBegin(0);
    domainsEnd();
    auto const built = treeCreate(TreeCreateParameters(0, 0.5));
    detail::verifyTreeGeometry(built.id, false);
    if constexpr (cosmictiger::gravity::cartesian::kDipoleFree) {
        detail::verifyMassCenters(built.id);
    }
    detail::verifyTreeNodeArchive(built.id);
    detail::verifyRemoteMultipoleStorage();
    detail::requireFmmTree(built.multi[0] == float(options.initial.numberOfParticles),
        "clustered tree lost particle mass");
    detail::requireFmmTree(built.radius > 0.0f, "clustered global node has zero radius");
    auto const* root = treeGetNode(built.id);
    for (int dim = 0; dim < kDimensions; dim++) {
        double const offset = dim == kYi ? 0.0 : (dim + 1) * Position::kToFloatFactor_;
        double const lower = 0.25 + offset;
        double const upper = (dim == kXi ? 0.375 : 0.3125) + offset;
        detail::requireFmmTree(built.box.begin[dim] == lower && built.box.end[dim] == upper,
            "clustered global bounds differ from the particle extrema");
        detail::requireFmmTree(
            root->box.begin[dim].toDouble() <= lower && root->box.end[dim].toDouble() >= upper,
            "stored global bounds do not enclose the exact particle extrema");
        detail::requireFmmTree(
            lower - root->box.begin[dim].toDouble() < RangeFixed::kToFloatFactor_ &&
                root->box.end[dim].toDouble() - upper < RangeFixed::kToFloatFactor_,
            "stored global bounds extend beyond the adjacent fixed-point grid value");
    }
    for (int rank = 0; rank < hpxSize(); rank++) {
        for (int i = 0; i < detail::kFmmTreeParticlesPerRank; i++) {
            auto const position = detail::fmmTreePosition(rank, i);
            double radiusSquared = 0.0;
            for (int dim = 0; dim < kDimensions; dim++) {
                double const displacement = position[dim].toDouble() - built.pos[dim].toDouble();
                radiusSquared += displacement * displacement;
                detail::requireFmmTree(
                    root->box.begin[dim].toDouble() <= position[dim].toDouble() &&
                        position[dim].toDouble() <= root->box.end[dim].toDouble(),
                    "stored global box does not enclose every particle");
            }
            detail::requireFmmTree(std::sqrt(radiusSquared) <= built.radius + 2.0e-7,
                "clustered global sphere does not enclose every particle");
        }
    }
    treeDestroy(true);

    if (hpxSize() == 1) {
        // More coincident particles than the bucket must terminate while the
        // enclosing global cells still satisfy the ordinary Ewald criterion.
        initialize(false);
        for (ParticleInteger particle = 0; particle < particlesSize(); ++particle) {
            for (int dim = 0; dim < kDimensions; ++dim) {
                particlesPos(dim, particle) = 0.375;
            }
        }
        auto const coincident = treeCreate(TreeCreateParameters(0, 0.5));
        detail::verifyTreeGeometry(coincident.id, false);
        detail::requireFmmTree(coincident.multi[0] == float(options.initial.numberOfParticles),
            "coincident tree lost particle mass");
        detail::requireFmmTree(coincident.radius == 0.0f, "coincident tree has a nonzero radius");
        treeDestroy(true);
    }

    // Preserve the distributed domain tree while removing every active particle.
    // With multiple localities this constructs an empty, multi-rank parent.
    initialize(true);
    auto const empty = treeCreate(TreeCreateParameters(0, 0.5));
    detail::verifyTreeGeometry(empty.id, true);
    treeDestroy(true);
    std::cout << std::format("FMM tree regression passed ({:d} localities)\n", hpxSize())
              << std::flush;
}
