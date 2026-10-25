/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/gravity/dehnen/remote_cache.hpp>
#include <cosmictiger/gravity/tree_build.hpp>

constexpr bool kVerbose = true;

static Vector<TreeNode> treeFetchCacheLine(int index);
HPX_PLAIN_ACTION(treeFetchCacheLine, TreeFetchCacheLineAction);

static Array<
    std::unordered_map<TreeId, std::shared_ptr<hpx::shared_future<Vector<TreeNode>>>, TreeIdHashHi>,
    TREE_CACHE_SIZE>
    treeCache;
static Array<SpinlockType, TREE_CACHE_SIZE> mutex;
struct LastCacheEntryT;

static std::unordered_set<LastCacheEntryT*> lastCacheEntries;
static std::atomic<int> lastCacheEntryMtx(0);

struct LastCacheEntryT {
    TreeId line;
    TreeNode const* ptr;
    void reset() {
        line.proc = line.index = -1;
        ptr = nullptr;
    }
    LastCacheEntryT() {
        line.proc = line.index = -1;
        ptr = nullptr;
        while (lastCacheEntryMtx++ != 0) {
            lastCacheEntryMtx--;
        }
        lastCacheEntries.insert(this);
        lastCacheEntryMtx--;
    }
    ~LastCacheEntryT() {
        while (lastCacheEntryMtx++ != 0) {
            lastCacheEntryMtx--;
        }
        lastCacheEntries.erase(this);
        lastCacheEntryMtx--;
    }
};

static thread_local LastCacheEntryT lastCacheEntry;

static void resetLastCacheEntries() {
    for (auto i = lastCacheEntries.begin(); i != lastCacheEntries.end(); i++) {
        (*i)->reset();
    }
}

static TreeNode const* treeCacheRead(TreeId id);

void treeResetRemoteCache() {
    treeCache = decltype(treeCache)();
    resetLastCacheEntries();
}

void treeEnableRemoteCache() {
    treeRegisterRemoteAccess(treeCacheRead, treeResetRemoteCache);
}

static TreeNode const* treeCacheRead(TreeId id) {
    int const lineSize = getOptions().compute.treeCacheLineSize;
    TreeId lineId;
    TreeNode const* ptr;
    lineId.proc = id.proc;
    ASSERT(lineId.proc >= 0 && lineId.proc < hpxSize());
    lineId.index = (id.index / lineSize) * lineSize;
    if (lineId != lastCacheEntry.line) {
        size_t const bin = TreeIdHashLo()(id);
        std::shared_ptr<hpx::shared_future<Vector<TreeNode>>> future;
        {
            std::lock_guard<SpinlockType> lock(mutex[bin]);
            auto const entry = treeCache[bin].find(lineId);
            if (entry != treeCache[bin].end()) {
                future = entry->second;
            }
        }
        if (!future) {
            auto promise = std::make_shared<hpx::promise<Vector<TreeNode>>>();
            auto candidate = std::make_shared<hpx::shared_future<Vector<TreeNode>>>(
                promise->get_future().share());
            bool inserted;
            {
                std::lock_guard<SpinlockType> lock(mutex[bin]);
                auto const entry = treeCache[bin].emplace(lineId, candidate);
                inserted = entry.second;
                future = entry.first->second;
            }
            if (inserted) {
                try {
                    hpx::post([promise, lineId]() {
                        try {
                            treeRecordRemoteFetch();
                            promise->set_value(hpx::async<TreeFetchCacheLineAction>(
                                hpxLocalities()[lineId.proc], lineId.index)
                                    .get());
                        } catch (...) {
                            promise->set_exception(std::current_exception());
                        }
                    });
                } catch (...) {
                    promise->set_exception(std::current_exception());
                }
            }
        }
        ptr = future->get().data();
    } else {
        ptr = lastCacheEntry.ptr;
    }
    lastCacheEntry.ptr = ptr;
    lastCacheEntry.line = lineId;
    return ptr + id.index - lineId.index;
}

static Vector<TreeNode> treeFetchCacheLine(int index) {
    size_t const nodesSize = treeNodesSize();
    int const lineSize = getOptions().compute.treeCacheLineSize;
    ALWAYS_ASSERT(index >= 0 && size_t(index) < nodesSize);
    Vector<TreeNode> line;
    line.reserve(lineSize);
    int const begin = (index / lineSize) * lineSize;
    int const end = std::min(size_t(begin) + size_t(lineSize), nodesSize);
    // A zeroed record sized for the largest order is a zeroed record of the
    // runtime order: every per-order view reads only its leading coefficients.
    static FmmMultiPos<kFmmMaxOrder> emptyMultipole = [] {
        FmmMultiPos<kFmmMaxOrder> empty;
        empty.multi = 0.0f;
        for (int dim = 0; dim < NDIM; dim++) {
            empty.pos[dim] = 0.0;
        }
        return empty;
    }();
    for (int i = begin; i < end; i++) {
        if (treeNodeIsPopulated(i)) {
            line.push_back(*treeLocalNode({hpxRank(), i}));
        } else {
            // Unused slots preserve cache offsets but never dereference arena garbage.
            TreeNode empty{};
            empty.mpos = reinterpret_cast<MultipolePosition*>(&emptyMultipole);
            empty.leaf = true;
            empty.processRange = Pair<int>(hpxRank(), hpxRank() + 1);
            for (int child = 0; child < NCHILD; child++) {
                empty.children[child].proc = hpxRank();
                empty.children[child].index = -1;
            }
            line.push_back(empty);
        }
    }
    return line;
}
