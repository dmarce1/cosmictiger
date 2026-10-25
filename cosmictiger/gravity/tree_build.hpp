/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/gravity/tree.hpp>

// Solver-neutral build input. Distributed domain routing and bounded-cell
// geometry are supplied by the caller; the local kd and multipole arithmetic
// are implemented once.
struct TreeBuildRequest {
    TreeCreateParameters params;
    size_t key;
    Pair<int, int> processRange;
    Pair<ParticleInteger> particleRange;
    Range<double> box;
    int depth;
    bool localRoot;
};

struct TreeBuildPolicy {
    FmmParticleStore sourceStore = FmmParticleStore::Owned;
    bool boundedGeometry = false;
    bool extraSubdivision = false;
    FlopCount subdivisionFlops = 0;
    Pair<int, int> representedProcessRange;
    void* context = nullptr;
    void (*constrainCenter)(void*, Array<double, NDIM>&, Array<Position, NDIM>&) = nullptr;
    void (*splitDomains)(void*, TreeBuildRequest&, TreeBuildRequest&) = nullptr;
    hpx::future<TreeCreateReturn> (*fork)(void*, TreeBuildRequest, bool, bool) = nullptr;
};

TreeCreateReturn treeBuild(TreeBuildRequest request, TreeBuildPolicy const& policy);
// Arena addresses remain fixed from preparation until all tree readers finish.
size_t treeNodeCapacity(size_t sourceParticleCount);
void treeAllocateNodesLocal(size_t capacity);
int treeAllocateNode();
void treeStoreNode(int index, TreeNode node, MultipoleBuffer const& multipole);
TreeNode const* treeLocalNode(TreeId id);
bool treeNodeIsPopulated(int index);
void treeRecordRemoteFetch();
// The active solver registers its remote access and epoch-reset policy.
void treeRegisterRemoteAccess(TreeNode const* (*reader)(TreeId), void (*reset)());
void treeResetLocal();
