/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2021  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include <unordered_set>

#include <cosmictiger/domain.hpp>
#include <cosmictiger/fast_future.hpp>
#include <cosmictiger/group_tree.hpp>

static Vector<GroupTreeNode> groupTreeFetchCacheLine(int index);
static GroupTreeNode const* groupTreeCacheRead(TreeId id);
static Vector<unsigned> groupTreeRefreshCacheLine(int index);

HPX_PLAIN_ACTION(groupTreeCreate, GroupTreeCreateAction);
HPX_PLAIN_ACTION(groupTreeDestroy, GroupTreeDestroyAction);
HPX_PLAIN_ACTION(groupTreeFetchCacheLine, GroupTreeFetchCacheLineAction);
HPX_PLAIN_ACTION(groupTreeRefreshCacheLine, GroupTreeRefreshCacheLineAction);
HPX_PLAIN_ACTION(groupTreeIncCacheEpoch, GroupTreeIncCacheEpochAction);

class GroupTreeAllocator {
    int next_;
    int last_;

public:
    bool ready_;
    void reset();
    bool isReady();
    GroupTreeAllocator();
    ~GroupTreeAllocator();
    GroupTreeAllocator(GroupTreeAllocator const&) = default;
    GroupTreeAllocator& operator=(GroupTreeAllocator const&) = default;
    GroupTreeAllocator(GroupTreeAllocator&&) = default;
    GroupTreeAllocator& operator=(GroupTreeAllocator&&) = default;
    int allocate();
};

static Vector<GroupTreeAllocator*> allocatorList;
static Array<SpinlockType, TREE_CACHE_SIZE> mutex;
static thread_local GroupTreeAllocator allocator;
static std::atomic<int> nextId;
static Vector<GroupTreeNode> nodes;

struct TreeCacheEntry {
    hpx::shared_future<Vector<GroupTreeNode>> data;
    int epoch;
};

static int treeCacheEpoch;

static Array<std::unordered_map<TreeId, TreeCacheEntry, TreeIdHashHi>, TREE_CACHE_SIZE> treeCache;

struct LastCacheEntryT;

static std::unordered_set<LastCacheEntryT*> lastCacheEntries;
static std::atomic<int> lastCacheEntryMtx;
static std::atomic<int> allocatorMtx(0);

struct LastCacheEntryT {
    TreeId line;
    GroupTreeNode const* ptr;
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

FastFuture<GroupTreeReturn> groupTreeCreateFork(size_t key, Pair<int, int> procRange,
    Pair<ParticleInteger> partRange, GroupRange box, int depth, bool localRoot, bool threadme) {
    static std::atomic<int> numberThreads(0);
    FastFuture<GroupTreeReturn> rc;
    bool remote = false;
    if (procRange.first != hpxRank()) {
        threadme = true;
        remote = true;
    } else if (threadme) {
        threadme = partRange.second - partRange.first > MIN_SORT_THREAD_PARTS;
        if (threadme) {
            if (numberThreads++ < SORT_OVERSUBSCRIPTION * hpxHardwareConcurrency() ||
                procRange.second - procRange.first > 1) {
                threadme = true;
            } else {
                threadme = false;
                numberThreads--;
            }
        }
    }
    if (!threadme) {
        rc.setValue(groupTreeCreate(key, procRange, partRange, box, depth, localRoot));
    } else if (remote) {
        rc = hpx::async<GroupTreeCreateAction>(
            hpxLocalities()[procRange.first], key, procRange, partRange, box, depth, localRoot);
    } else {
        rc = hpx::async([procRange, partRange, depth, localRoot, box, key]() {
            auto rc = groupTreeCreate(key, procRange, partRange, box, depth, localRoot);
            numberThreads--;
            return rc;
        });
    }
    return rc;
}

GroupTreeReturn groupTreeCreate(size_t key, Pair<int, int> procRange,
    Pair<ParticleInteger> partRange, GroupRange box, int depth, bool localRoot) {
    int const treeCacheLineSize = getOptions().treeCacheLineSize;
    if (depth >= MAX_DEPTH) {
        THROW_ERROR("%s\n", "Maximum depth exceeded\n");
    }
    if (nodes.size() == 0) {
        nextId = -treeCacheLineSize;
        nodes.resize(std::max(
            size_t(size_t(TREE_NODE_ALLOCATION_SIZE) * particlesSize() / GROUP_BUCKET_SIZE),
            (size_t) 1));
        while (allocatorMtx++ != 0) {
            allocatorMtx--;
        }
        for (int i = 0; i < allocatorList.size(); i++) {
            allocatorList[i]->ready_ = false;
        }
        allocatorMtx--;
    }
    if (localRoot) {
        partRange.first = 0;
        partRange.second = particlesSize();
        treeCacheEpoch = 0;
    }
    if (!allocator.ready_) {
        allocator.reset();
        allocator.ready_ = true;
    }
    int const index = allocator.allocate();
    GroupTreeNode node;
    GroupRange partBox;
    if (procRange.second - procRange.first > 1 ||
        partRange.second - partRange.first > GROUP_BUCKET_SIZE) {
        auto leftBox = box;
        auto rightBox = box;
        auto leftRange = procRange;
        auto rightRange = procRange;
        auto leftParts = partRange;
        auto rightParts = partRange;
        bool leftLocalRoot = false;
        bool rightLocalRoot = false;
        if (procRange.second - procRange.first > 1) {
            int const mid = (procRange.first + procRange.second) / 2;
            double const wt =
                double(procRange.second - mid) / double(procRange.second - procRange.first);
            leftRange.second = rightRange.first = mid;
            leftBox = domainsRange(key << 1);
            rightBox = domainsRange((key << 1) + 1);
            leftLocalRoot = leftRange.second - leftRange.first == 1;
            rightLocalRoot = rightRange.second - rightRange.first == 1;
        } else {
            int const xdim = box.longestDim();
            double const xmid = (box.begin[xdim] + box.end[xdim]) * 0.5;
            ParticleInteger const mid = particlesSort(partRange, xmid, xdim);
            leftParts.second = rightParts.first = mid;
            leftBox.end[xdim] = rightBox.begin[xdim] = xmid;
        }
        auto futr = groupTreeCreateFork(
            (key << 1) + 1, rightRange, rightParts, rightBox, depth + 1, rightLocalRoot, true);
        auto futl = groupTreeCreateFork(
            (key << 1), leftRange, leftParts, leftBox, depth + 1, leftLocalRoot, false);
        auto const rcl = futl.get();
        auto const rcr = futr.get();
        node.children[LEFT] = rcl.id;
        node.children[RIGHT] = rcr.id;
        for (int dim = 0; dim < NDIM; dim++) {
            partBox.begin[dim] = std::min(rcr.box.begin[dim], rcl.box.begin[dim]);
            partBox.end[dim] = std::max(rcr.box.end[dim], rcl.box.end[dim]);
        }
    } else {
        for (int dim = 0; dim < NDIM; dim++) {
            partBox.begin[dim] = 1.0;
            partBox.end[dim] = 0.0;
        }
        for (ParticleInteger i = partRange.first; i < partRange.second; i++) {
            for (int dim = 0; dim < NDIM; dim++) {
                double const x = particlesPos(dim, i).toDouble();
                partBox.begin[dim] = std::min(partBox.begin[dim], x);
                partBox.end[dim] = std::max(partBox.end[dim], x);
            }
        }
        node.children[LEFT].index = node.children[RIGHT].index = -1;
    }
    node.procRange = procRange;
    node.partRange = partRange;
    node.box = partBox;
    node.active = true;
    node.lastActive = true;
    node.localRoot = localRoot;
    nodes[index] = node;
    TreeId myid;
    myid.index = index;
    myid.proc = hpxRank();
    GroupTreeReturn rc;
    rc.id = myid;
    rc.box = partBox;
    return rc;
}

void groupTreeSetActive(TreeId id, bool b) {
    nodes[id.index].active = b;
}

void GroupTreeAllocator::reset() {
    int const treeCacheLineSize = getOptions().treeCacheLineSize;
    next_ = (nextId += treeCacheLineSize);
    last_ = std::min(next_ + treeCacheLineSize, (int) nodes.size());
    if (next_ >= nodes.size()) {
        THROW_ERROR("%s\n", "Tree arena full");
    }
}

GroupTreeAllocator::GroupTreeAllocator() {
    ready_ = false;
    while (allocatorMtx++ != 0) {
        allocatorMtx--;
    }
    allocatorList.push_back(this);
    allocatorMtx--;
}

int GroupTreeAllocator::allocate() {
    if (next_ == last_) {
        reset();
    }
    return next_++;
}

GroupTreeAllocator::~GroupTreeAllocator() {
    while (allocatorMtx++ != 0) {
        allocatorMtx--;
    }
    for (int i = 0; i < allocatorList.size(); i++) {
        if (allocatorList[i] == this) {
            allocatorList[i] = allocatorList.back();
            allocatorList.pop_back();
            break;
        }
    }
    allocatorMtx--;
}

void groupTreeDestroy() {
    Vector<hpx::future<void>> futs;
    auto const children = hpxChildren();
    for (auto const& c : children) {
        futs.push_back(hpx::async<GroupTreeDestroyAction>(c));
    }
    nodes = decltype(nodes)();
    treeCache = decltype(treeCache)();
    resetLastCacheEntries();
    hpx::wait_all(futs.begin(), futs.end());
}

GroupTreeNode const* groupTreeGetNode(TreeId id) {
    GroupTreeNode const* ptr;
    if (id.proc == hpxRank()) {
        ptr = &nodes[id.index];
        //		PRINT( "%i %i\n", id.index, ptr->children[LEFT].index);
    } else {
        ptr = groupTreeCacheRead(id);
    }
    return ptr;
}

static GroupTreeNode const* groupTreeCacheRead(TreeId id) {
    int const lineSize = getOptions().treeCacheLineSize;
    TreeId lineId;
    GroupTreeNode const* ptr;
    lineId.proc = id.proc;
    ASSERT(lineId.proc >= 0 && lineId.proc < hpxSize());
    lineId.index = (id.index / lineSize) * lineSize;
    if (lineId != lastCacheEntry.line) {
        size_t const bin = TreeIdHashLo()(id);
        std::unique_lock<SpinlockType> lock(mutex[bin]);
        auto iter = treeCache[bin].find(lineId);
        if (iter == treeCache[bin].end()) {
            auto prms = std::make_shared<hpx::promise<Vector<GroupTreeNode>>>();
            auto& entry = treeCache[bin][lineId];
            entry.data = prms->get_future();
            entry.epoch = treeCacheEpoch;
            lock.unlock();
            hpx::async(HPX_PRIORITY_HI, [prms, lineId]() {
                auto lineFut = hpx::async<GroupTreeFetchCacheLineAction>(
                    hpxLocalities()[lineId.proc], lineId.index);
                prms->set_value(lineFut.get());
                return 'a';
            });
            lock.lock();
            iter = treeCache[bin].find(lineId);
        } else if (iter->second.epoch < treeCacheEpoch) {
            auto prms = std::make_shared<hpx::promise<Vector<GroupTreeNode>>>();
            auto oldFut = std::move(iter->second.data);
            auto& entry = treeCache[bin][lineId];
            entry.data = prms->get_future();
            entry.epoch = treeCacheEpoch;
            lock.unlock();
            auto oldData = oldFut.get();
            hpx::async(
                HPX_PRIORITY_HI,
                [prms, lineId](Vector<GroupTreeNode> data) {
                    auto lineFut = hpx::async<GroupTreeRefreshCacheLineAction>(
                        hpxLocalities()[lineId.proc], lineId.index);
                    auto newData = lineFut.get();
                    for (int i = 0; i < newData.size(); i++) {
                        data[i].lastActive = newData[i];
                    }
                    prms->set_value(std::move(data));
                    return 'a';
                },
                std::move(oldData));
            lock.lock();
            iter = treeCache[bin].find(lineId);
        }
        auto fut = iter->second.data;
        lock.unlock();
        ptr = fut.get().data();
    } else {
        ptr = lastCacheEntry.ptr;
    }
    lastCacheEntry.ptr = ptr;
    lastCacheEntry.line = lineId;
    return ptr + id.index - lineId.index;
}

static Vector<GroupTreeNode> groupTreeFetchCacheLine(int index) {
    int const lineSize = getOptions().treeCacheLineSize;
    Vector<GroupTreeNode> line;
    line.reserve(lineSize);
    int const begin = (index / lineSize) * lineSize;
    int const end = begin + lineSize;
    for (int i = begin; i < end; i++) {
        line.push_back(nodes[i]);
    }
    return std::move(line);
}

static Vector<unsigned> groupTreeRefreshCacheLine(int index) {
    int const lineSize = getOptions().treeCacheLineSize;
    Vector<unsigned> line;
    line.reserve(lineSize);
    int const begin = (index / lineSize) * lineSize;
    int const end = begin + lineSize;
    for (int i = begin; i < end; i++) {
        line.push_back(nodes[i].lastActive);
    }
    return std::move(line);
}

void groupTreeIncCacheEpoch() {
    ParticleInteger const lineSize = getOptions().partCacheLineSize;
    Vector<hpx::future<void>> futs;
    auto const children = hpxChildren();
    for (auto const& c : children) {
        futs.push_back(hpx::async<GroupTreeIncCacheEpochAction>(c));
    }
    for (auto& node : nodes) {
        node.lastActive = node.active;
    }
    treeCacheEpoch++;
    resetLastCacheEntries();
    hpx::wait_all(futs.begin(), futs.end());
}
