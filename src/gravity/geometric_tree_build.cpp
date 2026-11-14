/* CosmicTiger - staged local geometric FMM tree; BSL-1.0. */

// Build local geometric FMM trees with exact node counts, bounded radii and multipoles.

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/gravity/fmm_acceptance.hpp>
#include <cosmictiger/gravity/geometric_tree_geometry.hpp>
#include <cosmictiger/gravity/tree_build.hpp>
#include <cosmictiger/gravity/tree_multipole.hpp>
#include <cosmictiger/particles/particle_partition.hpp>

namespace detail {

// This staged builder is serial: the counting pass both partitions the source
// range and determines the exact stable-arena capacity, including empty siblings.
struct GeometricBuildState {
    int ewaldDepth;                 // Minimum depth of nonempty leaves required by periodic FMM.
    int bucketSize;                 // Ordinary leaves must contain no more sources than this.
    FmmSourceView sources;          // Owned source arrays used for geometry and multipoles.
    size_t coincidentLeaves = 0;    // Leaves above the bucket limit with identical positions.
    size_t depthLeaves = 0;         // Leaves above the bucket limit stopped by maximum depth.
};

enum class GeometricTerminal {
    no,            // Split further: no stopping condition has been met.
    empty,         // No sources; an empty leaf is allowed before the required Ewald depth.
    ordinary,      // Required Ewald depth reached and source count is at or below the bucket limit.
    coincident,    // Required depth reached; all positions coincide, so splitting cannot help.
    depth          // Too many sources, but another split would exceed the depth limit.
};

GeometricTerminal geometricTerminal(
    GeometricTreeCell const& cell, Pair<ParticleInteger> range, GeometricBuildState const& state) {
    auto const count = range.second - range.first;
    if (count == 0) {
        return GeometricTerminal::empty;
    }
    if (int(cell.depth) < state.ewaldDepth) {
        return GeometricTerminal::no;
    }
    if (count <= state.bucketSize) {
        return GeometricTerminal::ordinary;
    }
    bool coincident = true;
    for (ParticleInteger i = range.first + 1; i < range.second && coincident; ++i) {
        for (int axis = 0; axis < kDimensions; ++axis) {
            if (state.sources.positions[axis][i].raw() !=
                state.sources.positions[axis][range.first].raw()) {
                coincident = false;
                break;
            }
        }
    }
    if (coincident) {
        return GeometricTerminal::coincident;
    }
    return cell.depth == kGeometricTreeMaxDepth ? GeometricTerminal::depth : GeometricTerminal::no;
}

ParticleInteger geometricPartition(GeometricTreeCell const& cell, Pair<ParticleInteger> range) {
    int const axis = kGeometricSplitAxes[cell.depth % kGeometricSplitAxes.size()];
    return particlesPartition(range, geometricTreeCenter(cell, axis), axis);
}

size_t geometricCountNodes(
    GeometricTreeCell const& cell, Pair<ParticleInteger> range, GeometricBuildState const& state) {
    if (geometricTerminal(cell, range, state) != GeometricTerminal::no) {
        return 1;
    }
    ParticleInteger const middle = geometricPartition(cell, range);
    size_t const left =
        geometricCountNodes(geometricTreeChild(cell, false), {range.first, middle}, state);
    size_t const right =
        geometricCountNodes(geometricTreeChild(cell, true), {middle, range.second}, state);
    if (right == std::numeric_limits<size_t>::max() ||
        left > std::numeric_limits<size_t>::max() - right - 1) {
        throw std::overflow_error("Geometric tree node count overflows size_t");
    }
    return 1 + left + right;
}

long double geometricUp(long double value) {
    return std::nextafter(value, std::numeric_limits<long double>::infinity());
}

float geometricNarrowUp(long double value) {
    float result = float(value);
    if (static_cast<long double>(result) < value) {
        result = std::nextafter(result, std::numeric_limits<float>::infinity());
    }
    return result;
}

// Raw integer subtraction precedes conversion, preserving sub-double Q64
// separations. Each conversion and nonnegative operation is bounded upward.
float geometricLeafRadius(
    Pair<ParticleInteger> range, Vector3<Position> const& center, FmmSourceView const& sources) {
    long double largest = 0;
    for (ParticleInteger i = range.first; i < range.second; ++i) {
        long double squared = 0;
        for (int axis = 0; axis < kDimensions; ++axis) {
            PositionInteger const a = sources.positions[axis][i].raw();
            PositionInteger const b = center[axis].raw();
            PositionInteger const magnitude = a >= b ? a - b : b - a;
            if (magnitude == 0) {
                continue;
            }
            long double const delta =
                std::ldexp(geometricUp(static_cast<long double>(magnitude)), -kPositionBits);
            squared = geometricUp(squared + geometricUp(delta * delta));
        }
        largest = std::max(largest, geometricTreeSqrtUp(squared));
    }
    return range.first == range.second ? 0.0f : geometricNarrowUp(largest);
}

TreeCreateReturn geometricBuildNode(
    GeometricTreeCell const& cell, Pair<ParticleInteger> range, GeometricBuildState& state) {
    TreeCreateReturn result;
    TreeNode node{};
    int const index = treeAllocateNode();
    auto const terminal = geometricTerminal(cell, range, state);
    auto const cuts = geometricTreeAxisCuts(cell.depth);
    for (int axis = 0; axis < kDimensions; ++axis) {
        result.pos[axis] = geometricTreeCenter(cell, axis);
        result.box.begin[axis] = std::ldexp(double(cell.coordinate[axis]), -cuts[axis]);
        result.box.end[axis] = std::ldexp(double(cell.coordinate[axis] + 1), -cuts[axis]);
    }
    node.box = geometricTreeBox(cell);
    node.pos = result.pos;
    node.particleRange = range;
    node.processRange = {hpxRank(), hpxRank() + 1};
    node.depth = cell.depth;
    node.localRoot = cell.depth == 0;
    node.leaf = terminal != GeometricTerminal::no;
    FlopCounts momentFlops{};
    FlopCount geometryFlops = 5;    // Geometric radius limit and narrowing.
    if (node.leaf) {
        node.children[kLeftChild].index = node.children[kRightChild].index = -1;
        result.radius = std::min(geometricTreeRadiusLimit(cell.depth),
            geometricLeafRadius(range, result.pos, state.sources));
        momentFlops +=
            treeComputeMultipole<kFmmOrder>(result.multi, range, result.pos, state.sources);
        state.coincidentLeaves += terminal == GeometricTerminal::coincident;
        state.depthLeaves += terminal == GeometricTerminal::depth;
        geometryFlops += FlopCount(range.second - range.first) * 13;
    } else {
        ParticleInteger const middle = geometricPartition(cell, range);
        auto const left =
            geometricBuildNode(geometricTreeChild(cell, false), {range.first, middle}, state);
        auto const right =
            geometricBuildNode(geometricTreeChild(cell, true), {middle, range.second}, state);
        node.children[kLeftChild] = left.id;
        node.children[kRightChild] = right.id;
        result.nodeCount += left.nodeCount + right.nodeCount;
        result.leafCount = left.leafCount + right.leafCount;
        int const axis = kGeometricSplitAxes[cell.depth % kGeometricSplitAxes.size()];
        long double const offset = std::ldexp(1.0L, -cuts[axis] - 2);
        // Derived fixed-center adaptation of Dehnen (2002), equation 9:
        // r = min(R, maxⱼ(|cⱼ − c| + rⱼ)), excluding empty children.
        // Symbol directory: r → cloud radius; R → cell half-diagonal;
        // c, cⱼ → parent and child centers; rⱼ → child cloud radius.
        // Code names: r → result.radius; R → geometricTreeRadiusLimit;
        // |cⱼ − c| → offset; maxⱼ(|cⱼ − c| + rⱼ) → bound.
        long double bound = 0;
        if (middle != range.first) {
            bound = geometricUp(offset + static_cast<long double>(left.radius));
        }
        if (middle != range.second) {
            bound = std::max(bound, geometricUp(offset + static_cast<long double>(right.radius)));
        }
        result.radius = std::min(geometricTreeRadiusLimit(cell.depth), geometricNarrowUp(bound));
        geometryFlops += (middle != range.first) + (middle != range.second) + 1;
        momentFlops += treeComputeMultipole<kFmmOrder>(
            result.multi, left.multi, right.multi, left.pos, right.pos, result.pos);
    }
    node.radius = result.radius;
    treeStoreNode(index, node, result.multi.narrow());
    result.id = {hpxRank(), index};
    addCpuFlops(momentFlops);
    addCpuFlops(geometryFlops, FlopPrecision::Double);
    return result;
}

}    // namespace detail

GeometricTreeBuildResult treeBuildGeometricLocal(
    TreeCreateParameters const& params, Pair<ParticleInteger> particleRange) {
    if constexpr (kDimensions != 3) {
        throw std::invalid_argument("Geometric FMM trees require three dimensions");
    }
    auto const& opts = getOptions();
    float const requestedTheta = float(params.theta);
    float const inverseTheta = fmmInverseOpeningAngle(requestedTheta);
    double const smoothing = double(float(opts.gravity.smoothingLength));
    if (!(std::isfinite(requestedTheta) && requestedTheta > 0 && requestedTheta <= 1 &&
            std::isfinite(inverseTheta) && inverseTheta > 0 && std::isfinite(smoothing) &&
            smoothing > 0 && smoothing < 0.5) ||
        opts.compute.bucketSize <= 0) {
        throw std::invalid_argument("Invalid geometric tree opening, support, or bucket size");
    }
    if (particleRange.first < 0 || particleRange.second < particleRange.first ||
        particleRange.second > particlesOwnedSize() ||
        particleRange.second > std::numeric_limits<int>::max() - std::max(32, kSimdFloatSize)) {
        throw std::length_error("Geometric tree source range must be owned and within int range");
    }
    long double const infinity = std::numeric_limits<long double>::infinity();
    long double const theta =
        std::nextafter(1.0L / static_cast<long double>(inverseTheta), -infinity);
    long double const denominator = detail::geometricUp(1 + theta);
    long double const firstLimit = std::nextafter(theta / (4 * denominator), -infinity);
    long double const secondLimit =
        std::nextafter(0.125L - 0.25L * static_cast<long double>(smoothing), -infinity);
    int ewaldDepth = 0;
    while (ewaldDepth <= kGeometricTreeMaxDepth &&
        !(geometricTreeRadiusLimit(ewaldDepth) < firstLimit &&
            geometricTreeRadiusLimit(ewaldDepth) < secondLimit)) {
        ++ewaldDepth;
    }
    if (ewaldDepth > kGeometricTreeMaxDepth) {
        throw std::length_error("Geometric tree cannot satisfy the Ewald depth");
    }
    detail::GeometricBuildState state{ewaldDepth, opts.compute.bucketSize, fmmSourceView()};
    GeometricTreeCell const root{};
    size_t const nodes = detail::geometricCountNodes(root, particleRange, state);
    treeAllocateNodesLocal(nodes);
    GeometricTreeBuildResult result;
    result.root = detail::geometricBuildNode(root, particleRange, state);
    result.coincidentOverflowLeaves = state.coincidentLeaves;
    result.depthOverflowLeaves = state.depthLeaves;
    result.ewaldDepth = ewaldDepth;
    return result;
}
