/* CosmicTiger - independent geometric FMM trees; BSL-1.0. */

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/gravity/global_fmm/distributed_tree.hpp>
#include <cosmictiger/gravity/global_fmm/measurement.hpp>
#include <cosmictiger/gravity/global_fmm/particle_exchange.hpp>
#include <cosmictiger/gravity/global_fmm/remote_cache.hpp>
#include <cosmictiger/gravity/source_access.hpp>
#include <cosmictiger/gravity/sources.hpp>
#include <cosmictiger/gravity/tree_build.hpp>

std::atomic<uint64_t> nextGeneration{1};
Vector<SourceRootDescriptor> geometricRoots;

Range<double> localSourceCover(
    int ewaldDepth, FmmSourceView const& sources, Pair<ParticleInteger> range) {
    Range<double> cover = unitBox<double>();
    if (range.first == range.second) {
        return cover;
    }
    auto const cuts = geometricTreeAxisCuts(ewaldDepth);
    for (int axis = 0; axis < kDimensions; ++axis) {
        PositionInteger lowest = std::numeric_limits<PositionInteger>::max();
        PositionInteger highest = 0;
        for (ParticleInteger index = range.first; index < range.second; ++index) {
            PositionInteger const value = sources.positions[axis][index].raw();
            lowest = std::min(lowest, value);
            highest = std::max(highest, value);
        }
        if (cuts[axis] == 0) {
            continue;
        }
        int const shift = kPositionBits - cuts[axis];
        uint64_t const lowerCell = lowest >> shift;
        uint64_t const upperCell = highest >> shift;
        long double const lower = std::ldexp(static_cast<long double>(lowerCell), -cuts[axis]);
        long double const upper = std::ldexp(static_cast<long double>(upperCell) + 1, -cuts[axis]);
        cover.begin[axis] =
            std::max(0.0, std::nextafter(double(lower), -std::numeric_limits<double>::infinity()));
        cover.end[axis] =
            std::min(1.0, std::nextafter(double(upper), std::numeric_limits<double>::infinity()));
    }
    return cover;
}

SourceRootDescriptor createGeometricLocal(
    TreeCreateParameters params, uint64_t generation, GeometricSourceSelection selection) {
    // The hierarchical Hamiltonian correction is F(S_L <- S_L). Full-source
    // solves instead retain all owned sources even when sinks are rung-filtered.
    auto const currentRange = particlesCurrentRange();
    if (selection == GeometricSourceSelection::AllOwned &&
        currentRange != Pair<ParticleInteger>{0, particlesOwnedSize()}) {
        throw std::logic_error("A full-source geometric build requires the full current range");
    }
    auto const range = selection == GeometricSourceSelection::CurrentRungRange ?
        currentRange :
        Pair<ParticleInteger>{0, particlesOwnedSize()};
    treeResetLocal();
    treeEnableRemoteCache();
    gravitySetRemoteSourceReader(globalFmmReadSourcePositions);
    FmmMeasurementScope buildTimer(FmmMeasurementPhase::LocalBuild);
    auto const build = treeBuildGeometricLocal(params, range);
    SourceRootDescriptor descriptor;
    descriptor.root = build.root.id;
    descriptor.multi = build.root.multi;
    descriptor.cover = localSourceCover(build.ewaldDepth, fmmSourceView(), range);
    descriptor.generation = generation;
    descriptor.sourceCount = range.second - range.first;
    descriptor.nodeCount = build.root.nodeCount;
    descriptor.leafCount = build.root.leafCount;
    descriptor.ewaldDepth = build.ewaldDepth;
    descriptor.theta = float(params.theta);
    descriptor.smoothingLength = float(getOptions().gravity.smoothingLength);
    descriptor.gpu = getOptions().gravity.cuda;
    treeSourceForestRegisterOwner(descriptor);
    return descriptor;
}
HPX_PLAIN_ACTION(createGeometricLocal, CreateGeometricLocalAction);

TreeCreateReturn treeCreateGeometric(
    TreeCreateParameters params, GeometricSourceSelection selection, bool exchange) {
    if (hpxRank() != 0) {
        throw std::logic_error("Geometric tree coordination must start on locality zero");
    }
    uint64_t const generation = nextGeneration.fetch_add(1, std::memory_order_relaxed);
    Vector<SourceRootDescriptor> roots;
    roots.reserve(hpxLocalities().size());
    std::exception_ptr failure;
    try {
        auto futures = detail::geometricLaunchAndDrain<SourceRootDescriptor>(
            hpxLocalities().size(), [&](size_t rank) {
                return hpx::async<CreateGeometricLocalAction>(
                    hpxLocalities()[rank], params, generation, selection);
            });
        for (auto& future : futures) {
            roots.push_back(future.get());
        }
    } catch (...) {
        failure = std::current_exception();
    }
    if (failure) {
        treeRetireGeometric();
        std::rethrow_exception(failure);
    }
    for (size_t rank = 0; rank < roots.size(); ++rank) {
        if (roots[rank].root.proc != int(rank) || roots[rank].generation != generation ||
            roots[rank].theta != roots[0].theta ||
            roots[rank].smoothingLength != roots[0].smoothingLength ||
            roots[rank].gpu != roots[0].gpu) {
            treeRetireGeometric();
            throw std::runtime_error("Inconsistent geometric source-root descriptors");
        }
    }
    geometricRoots = std::move(roots);
    try {
        if (exchange) {
            treeExchangeSourceForest(geometricRoots);
        }
    } catch (...) {
        failure = std::current_exception();
    }
    if (failure) {
        treeRetireGeometric();
        std::rethrow_exception(failure);
    }
    TreeCreateReturn result;
    TreeNode const* root = treeLocalNode(geometricRoots[0].root);
    result.id = geometricRoots[0].root;
    result.multi = geometricRoots[0].multi;
    result.pos = root->pos;
    result.radius = root->radius;
    result.box = unitBox<double>();
    result.nodeCount = 0;
    result.leafCount = 0;
    for (auto const& owner : geometricRoots) {
        result.nodeCount += owner.nodeCount;
        result.leafCount += owner.leafCount;
    }
    return result;
}

Vector<SourceRootDescriptor> const& treeGeometricRoots() {
    return geometricRoots;
}

void treeRetireGeometric(bool freeTree) {
    treeDestroy(freeTree);
    geometricRoots.clear();
}
