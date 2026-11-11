/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <format>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/core/timer.hpp>
#include <cosmictiger/cuda/cuda_memory.hpp>
#include <cosmictiger/cuda/cuda_memory_advice.hpp>
#include <cosmictiger/gravity/fmm_order.hpp>
#include <cosmictiger/gravity/fmm_types.hpp>
#include <cosmictiger/gravity/tree_build.hpp>
#include <cosmictiger/math/vector3.hpp>

constexpr bool kVerbose = true;

class TreeAllocator {
    int next_;
    int last_;

public:
    bool ready_;
    void reset();
    bool isReady();
    TreeAllocator();
    ~TreeAllocator();
    TreeAllocator(TreeAllocator const&) = default;
    TreeAllocator& operator=(TreeAllocator const&) = default;
    TreeAllocator(TreeAllocator&&) = default;
    TreeAllocator& operator=(TreeAllocator&&) = default;
    int allocate();
};

static Vector<TreeAllocator*> allocatorList;
static thread_local TreeAllocator allocator;
static TreeNode* nodes = nullptr;
// Positioned multipole arena: one record of multipoleStride bytes per node slot,
// sized for the build FMM order when the arena is (re)allocated. The stride
// is the record size rounded up to 16 bytes (fmmPositionedMultipoleStride), so
// the CUDA kick stages whole records with 16-byte asynchronous copies.
static char* multis = nullptr;
static size_t multipoleStride = 0;
// Cache lines span reserved allocator slots, including holes between threads.
static std::vector<unsigned char> populatedNodes;
static Vector<Pair<ParticleInteger>> leafParticleRanges;
static MutexType leafParticleRangeMutex;
static std::atomic<int> treeThreadCount(0);
static std::atomic<int> nextId;
// Remote deserialization must neither consume nor overlap local node slots.
// Chunks remain stable and device accessible until the next tree epoch.
struct RemoteMultipoleChunk {
    char* data;
    size_t used;
};
static std::vector<RemoteMultipoleChunk> remoteMultipoleChunks;
static MutexType remoteMultipoleMutex;
static MultipolePosition* emptyRemoteMultipole = nullptr;
constexpr size_t kRemoteMultipoleChunkSize = 4096;
static std::atomic<int> allocatorMtx(0);
static size_t nodesSize;
static std::atomic<uint64_t> remoteTreeFetchCount{0};
static Vector<short> leafSizes;
static SpinlockType leafSizesMutex;

/// Return the allocated node-arena size.
long long treeNodesSize() {
    return nodesSize;
}
// The coefficients follow the position header without padding for every order.
static_assert(offsetof(FmmMultiPos, multi) == sizeof(MultipolePosition),
    "Positioned multipole coefficients must directly follow the position");

/// Locate a node's variable-stride multipole record in the coefficient arena.
static MultipolePosition* treeMultipoleRecord(size_t index) {
    return reinterpret_cast<MultipolePosition*>(multis + index * multipoleStride);
}

/// Locate the packed coefficients immediately after a multipole position record.
static float* treeMultipoleCoefficients(MultipolePosition* record) {
    return reinterpret_cast<float*>(reinterpret_cast<char*>(record) + sizeof(MultipolePosition));
}

/// Locate the packed coefficients immediately after a multipole position record.
static float const* treeMultipoleCoefficients(MultipolePosition const* record) {
    return reinterpret_cast<float const*>(
        reinterpret_cast<char const*>(record) + sizeof(MultipolePosition));
}

/// Copy the active-order coefficients and zero transport padding in the destination record.
static void treeWriteMultipole(MultipolePosition* record, MultipoleBuffer const& multipole) {
    int const count = fmmMultipoleCoefficientCount(kFmmOrder);
    int const slots = int((multipoleStride - sizeof(MultipolePosition)) / sizeof(float));
    float* coefficients = treeMultipoleCoefficients(record);
    for (int i = 0; i < count; i++) {
        coefficients[i] = multipole[i];
    }
    // The padding up to the record stride travels to the GPU with the record;
    // keep it defined.
    for (int i = count; i < slots; i++) {
        coefficients[i] = 0.0f;
    }
}

/// Cache a remote position/multipole record, sharing a singleton record for an entirely
/// zero source.
MultipolePosition* treeAddMultipole(
    Vector3<Position> const& pos, MultipoleBuffer const& multipole) {
    bool empty = true;
    for (int dim = 0; dim < kDimensions; ++dim) {
        empty = empty && pos[dim].raw() == 0;
    }
    for (int coefficient = 0; coefficient < fmmMultipoleCoefficientCount(kFmmOrder);
        ++coefficient) {
        empty = empty && multipole[coefficient] == 0.0f;
    }
    std::lock_guard<MutexType> lock(remoteMultipoleMutex);
    if (empty && emptyRemoteMultipole != nullptr) {
        return emptyRemoteMultipole;
    }
    if (remoteMultipoleChunks.empty() ||
        remoteMultipoleChunks.back().used == kRemoteMultipoleChunkSize) {
        char* data =
            static_cast<char*>(allocateManagedMemory(kRemoteMultipoleChunkSize * multipoleStride));
        try {
            remoteMultipoleChunks.push_back({data, 0});
        } catch (...) {
            freeManagedMemory(data);
            throw;
        }
    }
    auto& chunk = remoteMultipoleChunks.back();
    auto* ptr = reinterpret_cast<MultipolePosition*>(chunk.data + chunk.used++ * multipoleStride);
    if (empty) {
        emptyRemoteMultipole = ptr;
    }
    ptr->pos = pos;
    treeWriteMultipole(ptr, multipole);
    return ptr;
}

/// Copy the active-order packed coefficients into a multipole buffer.
void treeReadMultipole(MultipolePosition const* record, MultipoleBuffer& multipole) {
    int const count = fmmMultipoleCoefficientCount(kFmmOrder);
    float const* coefficients = treeMultipoleCoefficients(record);
    for (int i = 0; i < count; i++) {
        multipole[i] = coefficients[i];
    }
}

/// Reserve the next allocator block from the shared node arena, failing on exhaustion.
void TreeAllocator::reset() {
    int const treeAllocLineSize = getOptions().compute.treeAllocLineSize;
    next_ = (nextId += treeAllocLineSize);
    last_ = std::min(next_ + treeAllocLineSize, (int) nodesSize);
    if (next_ >= nodesSize) {
        throw std::length_error("Tree arena full");
    }
}

/// Register a thread-local block allocator for coordinated arena resets.
TreeAllocator::TreeAllocator() {
    ready_ = false;
    while (allocatorMtx++ != 0) {
        allocatorMtx--;
    }
    allocatorList.push_back(this);
    allocatorMtx--;
}

/// Return the next node index, reserving another block when the current one is exhausted.
int TreeAllocator::allocate() {
    if (next_ == last_) {
        reset();
    }
    ASSERT(next_ < nodesSize);
    return next_++;
}

/// Unregister this allocator from the shared reset list.
TreeAllocator::~TreeAllocator() {
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

/// Initialize tree-building controls from the opening angle and minimum active rung.
TreeCreateParameters::TreeCreateParameters(int minimumRung, double openingAngle) {
    theta = openingAngle;
    minRung = minimumRung;
    minLevel = 9;
    leafPushed = false;
    doLeafSizes = false;
    parallelParticleThreshold = 0;
}

/// Store a remote node copy in the arena and mark its slot populated.
size_t treeAddRemote(TreeNode const& remote) {
    int const treeAllocLineSize = getOptions().compute.treeAllocLineSize;
    if (nextId + treeAllocLineSize >= nodesSize) {
        std::cout << "TREE ALLOCATION EXCEEDED\n" << std::flush;
        abort();
    }
    size_t index = nextId++ + treeAllocLineSize;
    nodes[index] = remote;
    populatedNodes[index] = true;
    return index;
}

/// Expose the contiguous node arena for CPU or CUDA tree traversal.
TreeNode* treeData() {
    return nodes;
}

static bool setGpu = false;
static bool setCpu = false;

/// Remove GPU read-mostly advice before host mutation of the node arena.
void tree2Cpu() {
    if (!setCpu) {
#ifdef USE_CUDA
        if (setGpu && getOptions().gravity.cuda) {
            cudaSetDevice();
            CUDA_WARN(cudaAdviseMemory(nodes, nodesSize * sizeof(TreeNode),
                cudaMemAdviseUnsetReadMostly, cudaGetDevice()));
        }
#endif
        setGpu = false;
        setCpu = true;
    }
}

/// Apply GPU read-mostly advice to the node arena after checking the selected backend.
void tree2Gpu() {
#ifdef USE_CUDA
    if (!getOptions().gravity.cuda) {
        throw std::invalid_argument("GPU tree access requires gravity.cuda=true");
    }
    if (!setGpu) {
        cudaSetDevice();
        CUDA_WARN(cudaAdviseMemory(
            nodes, nodesSize * sizeof(TreeNode), cudaMemAdviseSetReadMostly, cudaGetDevice()));
        setGpu = true;
        setCpu = false;
    }
#else
    throw std::invalid_argument("GPU tree access requested in a build without CUDA");
#endif
}

/// Estimate node capacity from source count and bucket size with overflow protection.
size_t treeNodeCapacity(size_t sourceParticleCount) {
    size_t const multiplier = kTreeNodeAllocationSize;
    if (sourceParticleCount > std::numeric_limits<size_t>::max() / multiplier) {
        throw std::overflow_error("Tree arena capacity overflows size_t");
    }
    return multiplier * sourceParticleCount / getOptions().compute.bucketSize;
}

/// Allocate or reuse node and multipole arenas and reset registered allocator blocks.
void treeAllocateNodesLocal(size_t capacity) {
    int const treeAllocLineSize = getOptions().compute.treeAllocLineSize;
    auto const multiply = [](size_t first, size_t second) {
        if (second != 0 && first > std::numeric_limits<size_t>::max() / second) {
            throw std::overflow_error("Tree arena capacity overflows size_t");
        }
        return first * second;
    };
    size_t const sz = std::max(capacity, size_t(kMinTrees));
    if (sz > size_t(std::numeric_limits<int>::max() - treeAllocLineSize)) {
        throw std::overflow_error("Tree arena exceeds TreeId index range");
    }
    size_t const stride = fmmPositionedMultipoleStride(kFmmOrder);
    std::vector<unsigned char> freshPopulation(sz, 0);
    if (nodesSize < sz || multipoleStride != stride) {
        TreeNode* freshNodes =
            static_cast<TreeNode*>(allocateManagedMemory(multiply(sz, sizeof(TreeNode))));
        char* freshMultipoles;
        try {
            freshMultipoles = static_cast<char*>(allocateManagedMemory(multiply(sz, stride)));
        } catch (...) {
            freeManagedMemory(freshNodes);
            throw;
        }
        freeManagedMemory(nodes);
        freeManagedMemory(multis);
        nodes = freshNodes;
        multis = freshMultipoles;
        multipoleStride = stride;
    }
    {
        std::lock_guard<MutexType> lock(remoteMultipoleMutex);
        for (auto const& chunk : remoteMultipoleChunks) {
            freeManagedMemory(chunk.data);
        }
        remoteMultipoleChunks.clear();
        emptyRemoteMultipole = nullptr;
    }
    nodesSize = sz;
    nextId = -treeAllocLineSize;
    populatedNodes.swap(freshPopulation);
    while (allocatorMtx++ != 0) {
        allocatorMtx--;
    }
    for (int i = 0; i < allocatorList.size(); i++) {
        allocatorList[i]->ready_ = false;
    }
    allocatorMtx--;
    tree2Cpu();
}

/// Return the next shared node index including the allocator block reservation offset.
long long treeNodesNextIndex() {
    int const treeAllocLineSize = getOptions().compute.treeAllocLineSize;
    return nextId + treeAllocLineSize;
}

namespace detail {
static TreeNode const* (*remoteReader)(TreeId) = nullptr;
static void (*remoteReset)() = nullptr;
}    // namespace detail

/// Install solver-specific callbacks for remote node reads and cache resets.
void treeRegisterRemoteAccess(TreeNode const* (*reader)(TreeId), void (*reset)()) {
    detail::remoteReader = reader;
    detail::remoteReset = reset;
}

/// Invoke the active solver's local remote-cache reset callback.
void treeResetLocal() {
    if (detail::remoteReset != nullptr) {
        detail::remoteReset();
    }
}

/// Allocate a local node slot using this thread's arena block.
int treeAllocateNode() {
    if (!allocator.ready_) {
        allocator.reset();
        allocator.ready_ = true;
    }
    return allocator.allocate();
}

/// Store a node and its multipole coefficients and mark the arena slot populated.
void treeStoreNode(int index, TreeNode node, MultipoleBuffer const& multipole) {
    if (size_t(index) >= nodesSize) {
        throw std::length_error("Tree arena full");
    }
    node.mpos = treeMultipoleRecord(index);
    node.mpos->pos = node.pos;
    treeWriteMultipole(node.mpos, multipole);
    nodes[index] = node;
    populatedNodes[index] = true;
}

HPX_PLAIN_ACTION(treeReset, TreeResetAction);

/// Reset solver-specific tree caches across the locality tree.
void treeReset() {
    Vector<hpx::future<void>> futs;
    auto const children = hpxChildren();
    for (auto const& c : children) {
        futs.push_back(hpx::async<TreeResetAction>(c));
    }
    treeResetLocal();
    hpx::wait_all(futs.begin(), futs.end());
}
Vector<double> treeGetLeafSizes();
HPX_PLAIN_ACTION(treeGetLeafSizes, TreeGetLeafSizesAction);

/// Collect each locality's mean recorded leaf size.
Vector<double> treeGetLeafSizes() {
    Vector<hpx::future<Vector<double>>> futs;
    for (auto& c : hpxChildren()) {
        futs.push_back(hpx::async<TreeGetLeafSizesAction>(c));
    }
    double myAvg = 0.0;
    for (int i = 0; i < leafSizes.size(); i++) {
        myAvg += leafSizes[i];
    }
    myAvg /= leafSizes.size();
    Vector<double> avgs;
    avgs.push_back(myAvg);
    for (auto& f : futs) {
        auto tmp = f.get();
        for (auto const& m : tmp) {
            avgs.push_back(m);
        }
    }
    return avgs;
}

/// Average the per-locality mean leaf sizes.
int treeAvgLeafSize() {
    auto sizes = treeGetLeafSizes();
    double avg = 0.0;
    for (int i = 0; i < sizes.size(); i++) {
        avg += sizes[i];
    }
    avg /= sizes.size();
    return avg;
}

HPX_PLAIN_ACTION(treeDestroy, TreeDestroyAction);

/// Invoke tree-cache cleanup across the locality tree; this implementation retains arena
/// allocations.
void treeDestroy(bool freeTree) {
    Vector<hpx::future<void>> futs;
    auto const children = hpxChildren();
    for (auto const& c : children) {
        futs.push_back(hpx::async<TreeDestroyAction>(c, freeTree));
    }
    treeResetLocal();
    hpx::wait_all(futs.begin(), futs.end());
}

/// Return a local node or obtain a remote node through the registered solver adapter.
TreeNode const* treeGetNode(TreeId id) {
    if (id.proc == hpxRank()) {
        ASSERT(id.index >= 0);
#ifndef NDEBUG
        if (id.index >= nodesSize) {
            throw std::out_of_range(
                std::format("id.index is {} but nodes.size() is {}", id.index, nodesSize));
        }
#endif
        return &nodes[id.index];
    } else {
        if (detail::remoteReader == nullptr) {
            throw std::logic_error("Remote tree access requires an active solver adapter");
        }
        return detail::remoteReader(id);
    }
}

/// Require local ownership and return the corresponding node pointer.
TreeNode const* treeLocalNode(TreeId id) {
    ALWAYS_ASSERT(id.proc == hpxRank());
    return treeGetNode(id);
}

/// Report whether a valid arena slot has been filled with a node.
bool treeNodeIsPopulated(int index) {
    ALWAYS_ASSERT(index >= 0 && size_t(index) < nodesSize);
    return populatedNodes[index] != 0;
}

/// Read the process-local count of remote tree-node fetches.
uint64_t treeRemoteFetchCount() {
    return remoteTreeFetchCount.load(std::memory_order_relaxed);
}

/// Increment the process-local remote tree-fetch counter.
void treeRecordRemoteFetch() {
    remoteTreeFetchCount.fetch_add(1, std::memory_order_relaxed);
}
