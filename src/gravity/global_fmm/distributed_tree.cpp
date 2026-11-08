/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cmath>

#include <hpx/threading_base/annotated_function.hpp>

#include <cosmictiger/core/profiler.hpp>
#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/core/timer.hpp>
#include <cosmictiger/gravity/global_fmm/distributed_tree.hpp>
#include <cosmictiger/gravity/global_fmm/particle_exchange.hpp>
#include <cosmictiger/gravity/global_fmm/remote_cache.hpp>
#include <cosmictiger/gravity/source_access.hpp>
#include <cosmictiger/gravity/tree_build.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/particles/domain.hpp>

constexpr bool kVerbose = true;

static void treeAllocateNodes();
HPX_PLAIN_ACTION(treeAllocateNodes, TreeAllocateNodesAction);
HPX_PLAIN_ACTION(treeCreate, TreeCreateAction);

static void treeAllocateNodes() {
    Vector<hpx::future<void>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<TreeAllocateNodesAction>(child));
    }
    try {
        treeResetLocal();
        treeEnableRemoteCache();
        gravitySetRemoteSourceReader(globalFmmReadSourcePositions);
        treeResetRemoteCache();
        treeAllocateNodesLocal(treeNodeCapacity(size_t(particlesSize())));
    } catch (...) {
        hpx::wait_all(futures.begin(), futures.end());
        throw;
    }
    hpx::wait_all(futures.begin(), futures.end());
    for (auto& future : futures) {
        future.get();
    }
}

static hpx::future<TreeCreateReturn> treeCreateFork(TreeBuildRequest request, bool threadme) {
    bool const remote = request.processRange.first != hpxRank();
    if (remote) {
        return hpx::async<TreeCreateAction>(hpxLocalities()[request.processRange.first],
            request.params, request.key, request.processRange, request.particleRange, request.box,
            request.depth, request.localRoot);
    }
    auto const build = [request] {
        return treeCreate(request.params, request.key, request.processRange, request.particleRange,
            request.box, request.depth, request.localRoot);
    };
    threadme = threadme &&
        request.particleRange.second - request.particleRange.first >=
            request.params.parallelParticleThreshold;
    return threadme ? hpx::async(hpx::annotated_function(build, "gravity.buildTree")) :
                      hpx::make_ready_future(build());
}

TreeCreateReturn treeCreate(TreeCreateParameters params, size_t key, Pair<int, int> processRange,
    Pair<ParticleInteger> particleRange, Range<double> box, int depth, bool localRoot) {
    bool const coordinator = depth == 0 && hpxRank() == 0;
    if (depth == 0) {
        profile::ScopedPhase phase("treeCreate.allocate", coordinator);
        treeAllocateNodes();
    }
    if (localRoot) {
        particleRange = particlesCurrentRange();
    }
    TreeBuildPolicy policy;
    policy.representedProcessRange = processRange;
    float boxR = 0.0f;
    for (int dim = 0; dim < kDimensions; dim++) {
        boxR += sqr(0.5 * (box.end[dim] - box.begin[dim]));
    }
    boxR = sqrt(boxR);
    bool const distanceSatisfied = boxR < 0.25 * (params.theta / (1.0 + params.theta));
    bool const ewaldSatisfied =
        distanceSatisfied && boxR < 0.125 - 0.25 * getOptions().gravity.smoothingLength;
    policy.extraSubdivision = !ewaldSatisfied && particleRange.second > particleRange.first;
    // Box extents/accumulation and opening arithmetic are double. The
    // float radius accumulator widens and narrows in each dimension.
    policy.subdivisionFlops.singlePrecision =
        std::is_same_v<decltype(sqrt(boxR)), float> ? 1ULL : 0ULL;
    policy.subdivisionFlops.doublePrecision = 18 + 5 + (distanceSatisfied ? 2ULL : 0ULL) +
        (std::is_same_v<decltype(sqrt(boxR)), float> ? 0ULL : 3ULL);
    policy.subdivisionFlops.unclassified = 0;
    policy.splitDomains = [](void*, TreeBuildRequest& left, TreeBuildRequest& right) {
        int const mid = (left.processRange.first + left.processRange.second) / 2;
        left.box = domainsRange(left.key);
        right.box = domainsRange(right.key);
        left.processRange.second = right.processRange.first = mid;
        left.localRoot = left.processRange.second - left.processRange.first == 1;
        right.localRoot = right.processRange.second - right.processRange.first == 1;
    };
    policy.fork = [](void*, TreeBuildRequest request, bool threadme, bool) {
        return treeCreateFork(request, threadme);
    };
    auto result = [&]() {
        // Only the root call owns a phase; recursively forked HPX tasks share the stack.
        profile::ScopedPhase phase("treeCreate.build", coordinator);
        return treeBuild({params, key, processRange, particleRange, box, depth, localRoot}, policy);
    }();
#if COSMICTIGER_WITH_SPH
    // Hierarchical swaps maintain inverse gas indices without scanning slow rungs.
    if (localRoot && getOptions().hydro.enabled && !usesHierarchicalIntegration(getOptions().run)) {
        particlesResolveWithSphParticles();
    }
#endif
    return result;
}
