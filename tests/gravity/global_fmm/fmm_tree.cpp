/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

// Test tree geometry, leaf limits, multipoles, serialization and arena lifetime invariants.

#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <limits>
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
#include <cosmictiger/gravity/geometric_tree_geometry.hpp>
#include <cosmictiger/gravity/global_fmm/distributed_tree.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/gravity/tree_build.hpp>
#include <cosmictiger/gravity/tree_multipole.hpp>
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

void verifyGeometricNode(TreeId id, int depth) {
    auto const* node = treeGetNode(id);
    requireFmmTree(node->depth == depth, "geometric node depth mismatch");
    long double maximumDistance = 0;
    for (ParticleInteger i = node->particleRange.first; i < node->particleRange.second; ++i) {
        long double squared = 0;
        for (int axis = 0; axis < kDimensions; ++axis) {
            auto const a = particlesPos(axis, i).raw();
            auto const b = node->pos[axis].raw();
            long double const delta =
                std::ldexp(static_cast<long double>(a >= b ? a - b : b - a), -kPositionBits);
            squared += delta * delta;
        }
        maximumDistance = std::max(maximumDistance, std::sqrt(squared));
        requireFmmTree(
            maximumDistance <= node->radius, "geometric radius does not enclose a source");
    }
    if (node->numberOfParticles() == 0) {
        requireFmmTree(node->leaf && node->radius == 0, "empty geometric node is not terminal");
    } else if (node->leaf) {
        float const upper =
            std::nextafter(float(maximumDistance), std::numeric_limits<float>::infinity());
        requireFmmTree(node->radius <= upper, "geometric leaf radius is not measured tightly");
    }
    if (!node->leaf) {
        auto const* left = treeGetNode(node->children[kLeftChild]);
        auto const* right = treeGetNode(node->children[kRightChild]);
        requireFmmTree(left->particleRange.first == node->particleRange.first &&
                left->particleRange.second == right->particleRange.first &&
                right->particleRange.second == node->particleRange.second,
            "geometric children do not partition the parent range");
        int const axis = kGeometricSplitAxes[depth % kGeometricSplitAxes.size()];
        long double childBound = 0;
        for (auto const* child : {left, right}) {
            if (child->numberOfParticles() != 0) {
                auto const a = child->pos[axis].raw();
                auto const b = node->pos[axis].raw();
                long double const offset =
                    std::ldexp(static_cast<long double>(a >= b ? a - b : b - a), -kPositionBits);
                childBound = std::max(childBound, offset + child->radius);
            }
        }
        long double const factor =
            depth % 3 == 0 ? std::sqrt(3.0L) / 2 : (depth % 3 == 1 ? 0.75L : std::sqrt(6.0L) / 4);
        long double const expectedRadius = std::min(childBound, std::ldexp(factor, -depth / 3));
        requireFmmTree(node->radius >= expectedRadius &&
                node->radius <=
                    std::nextafter(float(expectedRadius), std::numeric_limits<float>::infinity()),
            "geometric parent radius differs from the capped child recurrence");
        for (ParticleInteger i = left->particleRange.first; i < left->particleRange.second; ++i) {
            requireFmmTree(particlesPos(axis, i).raw() < node->pos[axis].raw(),
                "left particle is not below the split");
        }
        for (ParticleInteger i = right->particleRange.first; i < right->particleRange.second; ++i) {
            requireFmmTree(particlesPos(axis, i).raw() >= node->pos[axis].raw(),
                "right particle is below the split");
        }
        verifyGeometricNode(node->children[kLeftChild], depth + 1);
        verifyGeometricNode(node->children[kRightChild], depth + 1);
    }
}

void verifyGeometricBuilder() {
    auto options = getOptions();
    options.compute.bucketSize = 4;
    options.gravity.cuda = false;
    options.gravity.smoothingLength = 0.001;
    setOptions(options);
    constexpr std::array<int, 18> kExpectedAxes{
        kXi, kYi, kZi, kZi, kYi, kXi, kZi, kXi, kYi, kYi, kXi, kZi, kYi, kZi, kXi, kXi, kZi, kYi};
    GeometricTreeCell cell{};
    std::array<double, kDimensions> lower{};
    std::array<double, kDimensions> upper{1, 1, 1};
    for (size_t depth = 0; depth < kExpectedAxes.size(); ++depth) {
        requireFmmTree(
            kGeometricSplitAxes[depth] == kExpectedAxes[depth], "split schedule changed");
        int const axis = kExpectedAxes[depth];
        double const midpoint = (lower[axis] + upper[axis]) / 2;
        bool const right = depth % 3 == 1;
        if (right) {
            lower[axis] = midpoint;
        } else {
            upper[axis] = midpoint;
        }
        cell = geometricTreeChild(cell, right);
        auto const box = geometricTreeBox(cell);
        for (int dim = 0; dim < kDimensions; ++dim) {
            requireFmmTree(
                geometricTreeCenter(cell, dim).toDouble() == (lower[dim] + upper[dim]) / 2,
                "geometric center disagrees with midpoint construction");
            if (!box.full(dim)) {
                requireFmmTree(box.begin[dim].toDouble() == lower[dim] &&
                        (upper[dim] == 1 ? box.end[dim].raw() == 0 :
                                           box.end[dim].toDouble() == upper[dim]),
                    "geometric box disagrees with midpoint construction");
            }
        }
    }
    GeometricTreeCell lastCell{};
    for (int depth = 0; depth < kGeometricTreeMaxDepth; ++depth) {
        lastCell = geometricTreeChild(lastCell, true);
    }
    auto const lastCuts = geometricTreeAxisCuts(kGeometricTreeMaxDepth);
    auto const lastBox = geometricTreeBox(lastCell);
    for (int axis = 0; axis < kDimensions; ++axis) {
        PositionInteger const expected =
            PositionInteger{0} - (PositionInteger{1} << (kPositionBits - lastCuts[axis] - 1));
        requireFmmTree(geometricTreeCenter(lastCell, axis).raw() == expected,
            "deep geometric center lost fixed-point bits");
        requireFmmTree(lastBox.end[axis].raw() == 0 && lastBox.extentTicks(axis) >= 1,
            "deep geometric box does not cover the seam-side cell");
    }
    if constexpr (kPositionBits == 64) {
        auto adjacentCell = lastCell;
        --adjacentCell.coordinate[kXi];
        auto const adjacentBox = geometricTreeBox(adjacentCell);
        requireFmmTree(adjacentBox.begin[kXi] == lastBox.begin[kXi] &&
                adjacentBox.end[kXi] == lastBox.end[kXi] &&
                geometricTreeCenter(adjacentCell, kXi) != geometricTreeCenter(lastCell, kXi),
            "sub-Fixed32 cells lost their distinct native identities");
    }
    constexpr ParticleInteger kGeometricFixtureParticles = 8;
    particlesResize(kGeometricFixtureParticles);
    for (ParticleInteger i = 0; i < kGeometricFixtureParticles; ++i) {
        particlesPos(kXi, i) = i == 0 ? 0.5 : 0.125 + 0.03125 * i;
        particlesPos(kYi, i) = 0.25 + 0.015625 * i;
        particlesPos(kZi, i) = 0.375;
        particlesRung(i) = char(i % 2);
    }
    auto result =
        treeBuildGeometricLocal(TreeCreateParameters(1, 0.5), {0, kGeometricFixtureParticles});
    verifyGeometricNode(result.root.id, 0);
    requireFmmTree(result.root.multi[0] == double(kGeometricFixtureParticles),
        "inactive owned sources were omitted from geometric moments");
    auto const* root = treeGetNode(result.root.id);
    for (int axis = 0; axis < kDimensions; ++axis) {
        requireFmmTree(root->pos[axis].raw() == Position(0.5).raw() && root->box.full(axis),
            "geometric root is not the unit box");
    }
    SourceMultipoleBuffer direct;
    treeComputeMultipole<kFmmOrder>(
        direct, {0, kGeometricFixtureParticles}, result.root.pos, fmmSourceView());
    for (int i = 0; i < fmmMultipoleCoefficientCount(kFmmOrder); ++i) {
        requireFmmTree(
            std::abs(direct[i] - result.root.multi[i]) < 1.0e-9 * (1 + std::abs(direct[i])),
            "geometric upward moments differ from direct P2M");
    }
    requireFmmTree(std::abs(direct[1]) + std::abs(direct[2]) + std::abs(direct[3]) > 0,
        "geometric fixture has no dipole");

    for (ParticleInteger i = 0; i < kGeometricFixtureParticles; ++i) {
        for (int axis = 0; axis < kDimensions; ++axis) {
            particlesPos(axis, i) = 0.375;
        }
    }
    result = treeBuildGeometricLocal(TreeCreateParameters(0, 0.5), {0, kGeometricFixtureParticles});
    requireFmmTree(result.coincidentOverflowLeaves == 1, "coincident overflow was not recorded");
    verifyGeometricNode(result.root.id, 0);
    particlesResize(1);
    result = treeBuildGeometricLocal(TreeCreateParameters(0, 0.5), {0, 1});
    requireFmmTree(result.root.nodeCount > 1 && result.root.multi[0] > 0,
        "single source did not reach the required periodic depth");
    verifyGeometricNode(result.root.id, 0);
    particlesResize(2);
    for (int axis = 0; axis < kDimensions; ++axis) {
        particlesPos(axis, 1) = particlesPos(axis, 0);
    }
    particlesPos(kXi, 1).raw() += 1;
    options.compute.bucketSize = 2;
    setOptions(options);
    result = treeBuildGeometricLocal(TreeCreateParameters(0, 0.5), {0, 2});
    requireFmmTree(result.coincidentOverflowLeaves == 0 && result.depthOverflowLeaves == 0,
        "a leaf at the bucket limit was classified as overflow");
    requireFmmTree(result.root.nodeCount == size_t(2 * result.ewaldDepth + 1),
        "a leaf at the bucket limit did not stop at the required periodic depth");
    verifyGeometricNode(result.root.id, 0);
    options.compute.bucketSize = 1;
    setOptions(options);
    result = treeBuildGeometricLocal(TreeCreateParameters(0, 0.5), {0, 2});
    requireFmmTree(result.depthOverflowLeaves == 1, "depth overflow was not recorded");
    verifyGeometricNode(result.root.id, 0);
    particlesResize(0);
    result = treeBuildGeometricLocal(TreeCreateParameters(0, 0.5), {0, 0});
    requireFmmTree(
        result.root.nodeCount == 1 && result.root.radius == 0, "empty geometric root is incorrect");
}

Vector3<Position> fmmTreePosition(int rank, int particle) {
    Vector3<Position> position;
    position[kXi] = 0.25 + 0.125 * (rank + (particle & 1)) / hpxSize();
    position[kYi] = 0.25 + 0.0625 * ((particle >> 1) & 1);
    position[kZi] = 0.25 + 0.0625 * ((particle >> 2) & 1);
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
    if (hpxSize() == 1) {
        detail::verifyGeometricBuilder();
    }
    std::cout << std::format("FMM tree regression passed ({:d} localities)\n", hpxSize())
              << std::flush;
}
