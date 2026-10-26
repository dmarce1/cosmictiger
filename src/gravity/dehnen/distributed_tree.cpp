/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cmath>

#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/core/timer.hpp>
#include <cosmictiger/gravity/dehnen/distributed_tree.hpp>
#include <cosmictiger/gravity/dehnen/particle_exchange.hpp>
#include <cosmictiger/gravity/dehnen/remote_cache.hpp>
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
        gravitySetRemoteSourceReader(dehnenReadSourcePositions);
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
    return threadme ? hpx::async(build) : hpx::make_ready_future(build());
}

TreeCreateReturn treeCreate(TreeCreateParameters params, size_t key, Pair<int, int> processRange,
    Pair<ParticleInteger> particleRange, Range<double> box, int depth, bool localRoot) {
    if (depth == 0) {
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
        distanceSatisfied && boxR < 0.125 - 0.25 * getOptions().gravity.supportRadius;
    policy.extraSubdivision = !ewaldSatisfied && particleRange.second > particleRange.first;
    policy.subdivisionFlops = 16 + (distanceSatisfied ? 2 : 0);
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
    auto result =
        treeBuild({params, key, processRange, particleRange, box, depth, localRoot}, policy);
    if (localRoot && getOptions().hydro.enabled) {
        particlesResolveWithSphParticles();
    }
    return result;
}
