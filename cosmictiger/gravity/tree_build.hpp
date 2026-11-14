/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cosmictiger/core/flops.hpp>
#include <cosmictiger/gravity/tree.hpp>
#include <cosmictiger/math/vector3.hpp>

// Build input for the distributed global FMM kd-tree.
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
    bool extraSubdivision = false;
    FlopCounts subdivisionFlops{};
    Pair<int, int> representedProcessRange;
    void (*splitDomains)(TreeBuildRequest&, TreeBuildRequest&) = nullptr;
    hpx::future<TreeCreateReturn> (*fork)(TreeBuildRequest, bool) = nullptr;
};

TreeCreateReturn treeBuild(TreeBuildRequest request, TreeBuildPolicy const& policy);

// Local host builder for CPU and CUDA walks. The range is a contiguous owned source
// range: all owned particles for a full-source solve, the current rung suffix for
// hierarchical integration. Callers must retire all readers of the previous local
// tree arena. This does not install the resulting root into the distributed solver.
// It retains the source range's final permutation, preserving suffix membership.
struct GeometricTreeBuildResult {
    TreeCreateReturn root;
    size_t coincidentOverflowLeaves = 0;
    size_t depthOverflowLeaves = 0;
    int ewaldDepth = 0;
};
GeometricTreeBuildResult treeBuildGeometricLocal(
    TreeCreateParameters const& params, Pair<ParticleInteger> particleRange);
// Arena addresses remain fixed from preparation until all tree readers finish.
size_t treeNodeCapacity(size_t sourceParticleCount);
void treeAllocateNodesLocal(size_t capacity);

// A launch-owned contiguous traversal array. Local indices and positioned multipole
// pointers remain unchanged; remote nodes use independent capacity. The owned tree
// and its multipoles must remain alive until the staging arena is destroyed.
class TreeStagingArena {
    TreeNode* nodes_;
    size_t remoteBegin_;
    size_t size_;
    size_t capacity_;
    bool ownsNodes_;

public:
    explicit TreeStagingArena(size_t remoteNodeCount);
    ~TreeStagingArena();
    TreeStagingArena(TreeStagingArena const&) = delete;
    TreeStagingArena& operator=(TreeStagingArena const&) = delete;
    TreeNode* data() const;
    size_t remoteBegin() const;
    size_t size() const;
    size_t appendRemote(TreeNode const& node);
    void adviseGpu() const;
};

int treeAllocateNode();
void treeStoreNode(int index, TreeNode node, MultipoleBuffer const& multipole);
TreeNode const* treeLocalNode(TreeId id);
bool treeNodeIsPopulated(int index);
void treeRecordRemoteFetch();
// The active solver registers its remote access and epoch-reset policy.
void treeRegisterRemoteAccess(TreeNode const* (*reader)(TreeId), void (*reset)());
void treeResetLocal();

// Allocated remote multipole chunks, excluding allocator metadata. Readers joined.
size_t treeRemoteMultipoleBytes();
