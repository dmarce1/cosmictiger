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
    return position;
}

void verifyTreeReturnArchive() {
    TreeCreateReturn expected;
    expected.id.proc = 3;
    expected.id.index = 17;
    expected.radius = 0.125f;
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
    for (int dim = 0; dim < NDIM; dim++) {
        actual.box.begin[dim] = actual.box.end[dim] = -1.0;
    }
    {
        hpx::serialization::input_archive archive(bytes);
        archive >> actual;
    }
    requireFmmTree(actual.id == expected.id && actual.radius == expected.radius,
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

void verifyTreeGeometry(TreeId id, bool empty) {
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
    if (!node->leaf) {
        auto const children = node->children;
        for (int child = 0; child < NCHILD; child++) {
            verifyTreeGeometry(children[child], empty);
        }
    }
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

void fmmTreeTest() {
    detail::verifyTreeReturnArchive();
    auto options = getOptions();
    options.sph = false;
    options.vsoft = false;
    options.stars = false;
    options.doGroups = false;
    options.doTracers = false;
    options.bucketSize = 4;
    options.numberOfParticles = detail::kFmmTreeParticlesPerRank * hpxSize();
    options.gravitySupportRadius = 0.001;
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
    auto const built = treeCreate(TreeCreateParameters(0, 0.5, options.gravitySupportRadius));
    detail::verifyTreeGeometry(built.id, false);
    detail::requireFmmTree(
        built.multi[0] == float(options.numberOfParticles), "clustered tree lost particle mass");
    detail::requireFmmTree(built.radius > 0.0f, "clustered global node has zero radius");
    for (int dim = 0; dim < NDIM; dim++) {
        double const upper = dim == XDIM ? 0.375 : 0.3125;
        detail::requireFmmTree(built.box.begin[dim] == 0.25 && built.box.end[dim] == upper,
            "clustered global bounds differ from the particle extrema");
    }
    for (int rank = 0; rank < hpxSize(); rank++) {
        for (int i = 0; i < detail::kFmmTreeParticlesPerRank; i++) {
            auto const position = detail::fmmTreePosition(rank, i);
            double radiusSquared = 0.0;
            for (int dim = 0; dim < NDIM; dim++) {
                double const displacement = position[dim].toDouble() - built.pos[dim].toDouble();
                radiusSquared += displacement * displacement;
            }
            detail::requireFmmTree(std::sqrt(radiusSquared) <= built.radius + 2.0e-7,
                "clustered global sphere does not enclose every particle");
        }
    }
    treeDestroy(true);

    // Preserve the distributed domain tree while removing every active particle.
    // With multiple localities this constructs an empty, multi-rank parent.
    initialize(true);
    auto const empty = treeCreate(TreeCreateParameters(0, 0.5, options.gravitySupportRadius));
    detail::verifyTreeGeometry(empty.id, true);
    treeDestroy(true);
    PRINT("FMM tree regression passed (%i localities)\n", hpxSize());
}
