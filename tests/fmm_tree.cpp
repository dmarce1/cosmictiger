/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>

#include <cosmictiger/domain.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/tree.hpp>

namespace detail {
constexpr int kFmmTreeParticlesPerRank = 8;

void requireFmmTree(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(std::string("FMM tree regression: ") + message);
    }
}

Array<Fixed32, NDIM> fmmTreePosition(int rank, int particle) {
    Array<Fixed32, NDIM> position;
    position[XDIM] = 0.25 + 0.125 * (rank + (particle & 1)) / hpxSize();
    position[YDIM] = 0.25 + 0.0625 * ((particle >> 1) & 1);
    position[ZDIM] = 0.25 + 0.0625 * ((particle >> 2) & 1);
    // X/Z extrema fall between stored box coordinates; Y remains exactly aligned.
    position[XDIM].raw() += 1;
    position[ZDIM].raw() += 3;
    return position;
}

void verifyTreeReturnArchive() {
    TreeCreateReturn expected;
    expected.id.proc = 3;
    expected.id.index = 17;
    expected.radius = 0.125f;
    expected.maxGravitySupportRadius = 0.21875f;
    for (int i = 0; i < MULTIPOLE_SIZE; i++) {
        expected.multi[i] = float(i + 1);
    }
    for (int dim = 0; dim < NDIM; dim++) {
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
    actual.maxGravitySupportRadius = -1.0f;
    for (int dim = 0; dim < NDIM; dim++) {
        actual.box.begin[dim] = actual.box.end[dim] = -1.0;
    }
    {
        hpx::serialization::input_archive archive(bytes);
        archive >> actual;
    }
    requireFmmTree(actual.id == expected.id && actual.radius == expected.radius &&
            actual.maxGravitySupportRadius == expected.maxGravitySupportRadius,
        "tree-return metadata did not survive the HPX archive");
    for (int i = 0; i < MULTIPOLE_SIZE; i++) {
        requireFmmTree(
            actual.multi[i] == expected.multi[i], "multipole did not survive the HPX archive");
    }
    for (int dim = 0; dim < NDIM; dim++) {
        requireFmmTree(actual.pos[dim] == expected.pos[dim] &&
                actual.box.begin[dim] == expected.box.begin[dim] &&
                actual.box.end[dim] == expected.box.end[dim],
            "tree-return position or bounds did not survive the HPX archive");
    }
}

void verifyTreeNodeArchive(TreeId id) {
    // Node loading allocates a multipole in the live tree arena, so this check
    // runs after tree construction. Do not alter the production node itself.
    TreeNode expected = *treeGetNode(id);
    expected.maxGravitySupportRadius = 0.34375f;
    std::vector<char> bytes;
    {
        hpx::serialization::output_archive archive(bytes);
        archive << expected;
    }
    TreeNode actual;
    actual.maxGravitySupportRadius = -1.0f;
    {
        hpx::serialization::input_archive archive(bytes);
        archive >> actual;
    }
    requireFmmTree(actual.radius == expected.radius &&
            actual.maxGravitySupportRadius == expected.maxGravitySupportRadius &&
            actual.particleRange.first == expected.particleRange.first &&
            actual.particleRange.second == expected.particleRange.second &&
            actual.processRange.first == expected.processRange.first &&
            actual.processRange.second == expected.processRange.second &&
            actual.depth == expected.depth && actual.localRoot == expected.localRoot &&
            actual.leaf == expected.leaf,
        "tree-node metadata did not survive the HPX archive");
    for (int child = 0; child < NCHILD; ++child) {
        requireFmmTree(actual.children[child] == expected.children[child],
            "tree-node child did not survive the HPX archive");
    }
    for (int dim = 0; dim < NDIM; ++dim) {
        requireFmmTree(actual.box.begin[dim] == expected.box.begin[dim] &&
                actual.box.end[dim] == expected.box.end[dim] &&
                actual.pos[dim] == expected.pos[dim] &&
                actual.mpos->pos[dim] == expected.mpos->pos[dim],
            "tree-node bounds or position did not survive the HPX archive");
    }
    for (int i = 0; i < MULTIPOLE_SIZE; ++i) {
        requireFmmTree(actual.mpos->multi[i] == expected.mpos->multi[i],
            "tree-node multipole did not survive the HPX archive");
    }
}

float verifyTreeGeometry(TreeId id, bool empty, float supportRadius, int distinctSupportRank = -1,
    float distinctSupportRadius = 0.0f) {
    TreeNode const* node = treeGetNode(id);
    requireFmmTree(
        std::isfinite(node->radius) && node->radius >= 0.0f, "node has an invalid bounding radius");
    for (int dim = 0; dim < NDIM; dim++) {
        requireFmmTree(
            node->box.begin[dim] <= node->box.end[dim], "node has reversed particle bounds");
    }
    for (int i = 0; i < MULTIPOLE_SIZE; i++) {
        requireFmmTree(std::isfinite(node->mpos->multi[i]), "node has a nonfinite multipole");
        if (empty) {
            requireFmmTree(node->mpos->multi[i] == 0.0f, "empty node has a nonzero multipole");
        }
    }
    if (empty) {
        requireFmmTree(node->radius == 0.0f, "empty node has a nonzero radius");
    }
    requireFmmTree(
        std::isfinite(node->maxGravitySupportRadius) && node->maxGravitySupportRadius >= 0.0f,
        "node has an invalid maximum gravity support");
    float expectedSupport = 0.0f;
    if (!node->leaf) {
        auto const children = node->children;
        for (int child = 0; child < NCHILD; child++) {
            float const childSupport = verifyTreeGeometry(
                children[child], empty, supportRadius, distinctSupportRank, distinctSupportRadius);
            expectedSupport = std::max(expectedSupport, childSupport);
        }
    } else if (node->numberOfParticles() > 0) {
        expectedSupport =
            node->processRange.first == distinctSupportRank ? distinctSupportRadius : supportRadius;
    }
    requireFmmTree(node->maxGravitySupportRadius == expectedSupport,
        "node maximum gravity support disagrees with its represented particles");
    return expectedSupport;
}
}    // namespace detail

void initializeFmmTreeRegression(bool empty) {
    particlesResize(empty ? 0 : detail::kFmmTreeParticlesPerRank);
    for (ParticleInteger i = 0; i < particlesSize(); i++) {
        auto const position = detail::fmmTreePosition(hpxRank(), i);
        for (int dim = 0; dim < NDIM; dim++) {
            particlesPos(dim, i) = position[dim];
            particlesVel(dim, i) = 0.0f;
        }
        particlesRung(i) = 0;
    }
}
HPX_PLAIN_ACTION(initializeFmmTreeRegression, InitializeFmmTreeRegressionAction);

ParticleInteger setFmmTreeRegressionSupport(float supportRadius) {
    detail::requireFmmTree(hpxChildren().empty(), "support fixture requires a terminal locality");
    auto options = getOptions();
    options.gravity.supportRadius = supportRadius;
    setOptions(options);
    return particlesSize();
}
HPX_PLAIN_ACTION(setFmmTreeRegressionSupport, SetFmmTreeRegressionSupportAction);

void fmmTreeTest() {
    detail::verifyTreeReturnArchive();
    auto options = getOptions();
    options.hydro.enabled = false;

    options.analysis.groups = false;
    options.analysis.tracers = false;
    options.compute.bucketSize = 4;
    options.initial.numberOfParticles = detail::kFmmTreeParticlesPerRank * hpxSize();
    options.gravity.supportRadius = 0.001;
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
    auto const built = treeCreate(TreeCreateParameters(0, 0.5, options.gravity.supportRadius));
    detail::verifyTreeGeometry(built.id, false, float(options.gravity.supportRadius));
    detail::verifyTreeNodeArchive(built.id);
    detail::requireFmmTree(built.maxGravitySupportRadius == float(options.gravity.supportRadius),
        "clustered tree return has the wrong fixed gravity support");
    detail::requireFmmTree(built.multi[0] == float(options.initial.numberOfParticles),
        "clustered tree lost particle mass");
    detail::requireFmmTree(built.radius > 0.0f, "clustered global node has zero radius");
    auto const* root = treeGetNode(built.id);
    for (int dim = 0; dim < NDIM; dim++) {
        double const offset = dim == YDIM ? 0.0 : (dim + 1) * Fixed32::kToFloatFactor_;
        double const lower = 0.25 + offset;
        double const upper = (dim == XDIM ? 0.375 : 0.3125) + offset;
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
            for (int dim = 0; dim < NDIM; dim++) {
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

    if (hpxSize() > 1) {
        // Exercise the real distributed max reduction with heterogeneous leaf
        // metadata. Each locality still builds from a fixed support; this test
        // performs no gravity kick and does not enable adaptive-softening physics.
        int const distinctRank = hpxSize() - 1;
        constexpr float kDistinctSupportRadius = 0.00390625f;
        try {
            auto const count = hpx::async<SetFmmTreeRegressionSupportAction>(
                hpxLocalities()[distinctRank], kDistinctSupportRadius)
                                   .get();
            detail::requireFmmTree(count > 0, "distinct-support locality has no particles");
            auto const heterogeneous =
                treeCreate(TreeCreateParameters(0, 0.5, options.gravity.supportRadius));
            detail::verifyTreeGeometry(heterogeneous.id, false,
                float(options.gravity.supportRadius), distinctRank, kDistinctSupportRadius);
            detail::requireFmmTree(heterogeneous.maxGravitySupportRadius == kDistinctSupportRadius,
                "distributed parent did not receive the remote maximum gravity support");
            treeDestroy(true);
        } catch (...) {
            setOptions(options);
            throw;
        }
        setOptions(options);
    }

    // Preserve the distributed domain tree while removing every active particle.
    // With multiple localities this constructs an empty, multi-rank parent.
    initialize(true);
    auto const empty = treeCreate(TreeCreateParameters(0, 0.5, options.gravity.supportRadius));
    detail::verifyTreeGeometry(empty.id, true, float(options.gravity.supportRadius));
    detail::requireFmmTree(empty.maxGravitySupportRadius == 0.0f,
        "empty tree return has a nonzero maximum gravity support");
    treeDestroy(true);
    PRINT("FMM tree regression passed (%i localities)\n", hpxSize());
}
