/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

// Tree build requests/results and stable GPU staging storage.

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

// Arena addresses remain fixed from preparation until all tree readers finish.
size_t treeNodeCapacity(size_t sourceParticleCount);
void treeAllocateNodesLocal(size_t capacity);

// A launch-owned contiguous traversal array. Local indices and positioned multipole
// pointers remain unchanged; remote nodes use independent capacity. The owned tree
// and its multipoles must remain alive until the staging arena is destroyed.
class TreeStagingArena {
    TreeNode* nodes_;       // Contiguous local and imported node records used by this GPU launch.
    size_t remoteBegin_;    // First imported-node slot; preceding slots preserve local indices.
    size_t size_;           // End of the staged index range, including any unused local slots.
    size_t capacity_;       // Total node slots available without moving the storage.
    bool ownsNodes_;        // True for a separate allocation; false when borrowing the owned arena.

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
