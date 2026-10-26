/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>

#include <cosmictiger/gravity/nitadori/cell_tree.hpp>
#include <cosmictiger/gravity/tree_build.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

constexpr bool kVerbose = true;

namespace detail {

struct TreeCellBuild {
    int meshSize = 0;
    uint64_t cellKey = 0;
    FmmParticleStore sourceStore = FmmParticleStore::Owned;
    Vector<TreeId> roots;
    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & meshSize & cellKey & sourceStore & roots;
    }
};

}    // namespace detail

static std::unordered_map<int, TreeCreateReturn> cellRootReturns;
static MutexType cellRootMutex;
static TreeCreateReturn treeReadCellRoot(TreeId id);
static TreeCreateReturn treeCreateCellImpl(TreeCreateParameters params, size_t key,
    Pair<int, int> processRange, Pair<ParticleInteger> particleRange, Range<double> box, int depth,
    bool localRoot, detail::TreeCellBuild cellBuild);
HPX_PLAIN_ACTION(treeCreateCellImpl, TreeCreateCellImplAction);
HPX_PLAIN_ACTION(treeReadCellRoot, TreeReadCellRootAction);

static hpx::future<TreeCreateReturn> treeCreateFork(TreeCreateParameters params, size_t key,
    Pair<int, int> const& processRange, Pair<ParticleInteger> const& particleRange,
    Range<double> const& box, int const depth, bool const localRoot, bool threadme,
    detail::TreeCellBuild const& cellBuild) {
    hpx::future<TreeCreateReturn> rc;
    bool remote = false;
    int const owner = cellBuild.roots.empty() ? processRange.first : cellBuild.roots.front().proc;
    if (owner != hpxRank()) {
        threadme = true;
        remote = true;
    } else if (threadme) {
        threadme = particleRange.second - particleRange.first >= params.parallelParticleThreshold;
    }
    if (!threadme) {
        rc = hpx::make_ready_future(treeCreateCellImpl(
            params, key, processRange, particleRange, box, depth, localRoot, cellBuild));
    } else if (remote) {
        rc = hpx::async<TreeCreateCellImplAction>(hpxLocalities()[owner], params, key, processRange,
            particleRange, box, depth, localRoot, cellBuild);
    } else {
        rc = hpx::async(
            [params, processRange, key, particleRange, depth, localRoot, box, cellBuild]() {
                return treeCreateCellImpl(
                    params, key, processRange, particleRange, box, depth, localRoot, cellBuild);
            });
    }
    return rc;
}

namespace detail {

Array<uint64_t, kDimensions> treeCellCoordinates(uint64_t key, int meshSize) {
    uint64_t const count = particleMeshCellCount(meshSize);
    if (key >= count) {
        throw std::invalid_argument("Tree cell key is outside the mesh");
    }
    Array<uint64_t, kDimensions> coordinate;
    for (int dim = kDimensions - 1; dim >= 0; --dim) {
        coordinate[dim] = key % meshSize;
        key /= meshSize;
    }
    return coordinate;
}

Range<double> treeCellBox(uint64_t key, int meshSize) {
    auto const coordinate = treeCellCoordinates(key, meshSize);
    Range<double> box;
    for (int dim = 0; dim < kDimensions; ++dim) {
        box.begin[dim] = double(coordinate[dim]) / meshSize;
        box.end[dim] = double(coordinate[dim] + 1) / meshSize;
    }
    return box;
}

// The center is used both as a fixed-point record and as a floating-point M2M
// origin. Choose a double whose conversion remains in the exact half-open cell,
// including odd meshes and fixed64 values whose toDouble() rounds to one.
void treeClampCellCenter(Array<double, kDimensions>& center, Array<Position, kDimensions>& position,
    TreeCellBuild const& cellBuild) {
    auto const coordinate = treeCellCoordinates(cellBuild.cellKey, cellBuild.meshSize);
    unsigned __int128 const scale = static_cast<unsigned __int128>(1) << kPositionBits;
    auto const meshSize = static_cast<uint64_t>(cellBuild.meshSize);
    for (int dim = 0; dim < kDimensions; ++dim) {
        auto const lower =
            static_cast<PositionInteger>((coordinate[dim] * scale + meshSize - 1) / meshSize);
        auto const upper = static_cast<PositionInteger>(
            ((coordinate[dim] + 1) * scale + meshSize - 1) / meshSize - 1);
        Position lowerPosition;
        Position upperPosition;
        lowerPosition.raw() = lower;
        upperPosition.raw() = upper;
        double lowerValue = lowerPosition.toDouble();
        double upperValue = upperPosition.toDouble();
        if (Position(lowerValue).raw() < lower) {
            lowerValue = std::nextafter(lowerValue, 1.0);
        }
        if (upperValue >= 1.0 || Position(upperValue).raw() > upper) {
            upperValue = std::nextafter(upperValue, 0.0);
        }
        position[dim] = std::clamp(center[dim], lowerValue, upperValue);
        center[dim] = position[dim].toDouble();
        if (particleMeshCoordinate(position[dim].raw(), cellBuild.meshSize) != coordinate[dim]) {
            throw std::runtime_error("Tree multipole center left its mesh cell");
        }
    }
}

}    // namespace detail

static TreeCreateReturn treeCreateCellImpl(TreeCreateParameters params, size_t key,
    Pair<int, int> processRange, Pair<ParticleInteger> particleRange, Range<double> box, int depth,
    bool localRoot, detail::TreeCellBuild cellBuild) {
    if (!cellBuild.roots.empty()) {
        if (cellBuild.roots.size() == 1) {
            return treeReadCellRoot(cellBuild.roots.front());
        }
        int const mid = (processRange.first + processRange.second) / 2;
        bool const onlyLeft = cellBuild.roots.back().proc < mid;
        bool const onlyRight = cellBuild.roots.front().proc >= mid;
        if (onlyLeft || onlyRight) {
            key = (key << 1) + int(onlyRight);
            if (onlyLeft) {
                processRange.second = mid;
            } else {
                processRange.first = mid;
            }
            box = box.intersection(domainsRange(key));
            return treeCreateFork(
                params, key, processRange, particleRange, box, depth + 1, false, false, cellBuild)
                .get();
        }
    }
    TreeBuildPolicy policy;
    policy.sourceStore = cellBuild.sourceStore;
    policy.boundedGeometry = true;
    policy.representedProcessRange = cellBuild.roots.empty() ?
        processRange :
        Pair<int>(cellBuild.roots.front().proc, cellBuild.roots.back().proc + 1);
    struct BuildContext {
        detail::TreeCellBuild const& build;
        Range<double> const& box;
        detail::TreeCellBuild left;
        detail::TreeCellBuild right;
    } context{cellBuild, box, cellBuild, cellBuild};
    policy.context = &context;
    policy.constrainCenter = [](void* pointer, auto& center, auto& position) {
        auto const& context = *static_cast<BuildContext*>(pointer);
        detail::treeClampCellCenter(center, position, context.build);
    };
    policy.splitDomains = [](void* pointer, TreeBuildRequest& left, TreeBuildRequest& right) {
        auto& context = *static_cast<BuildContext*>(pointer);
        int const mid = (left.processRange.first + left.processRange.second) / 2;
        left.box = domainsRange(left.key).intersection(context.box);
        right.box = domainsRange(right.key).intersection(context.box);
        auto const& roots = context.build.roots;
        auto const split = std::lower_bound(
            roots.begin(), roots.end(), mid, [](TreeId id, int rank) { return id.proc < rank; });
        context.left.roots.assign(roots.begin(), split);
        context.right.roots.assign(split, roots.end());
        left.processRange.second = right.processRange.first = mid;
        left.localRoot = left.processRange.second - left.processRange.first == 1;
        right.localRoot = right.processRange.second - right.processRange.first == 1;
    };
    policy.fork = [](void* pointer, TreeBuildRequest request, bool threadme, bool right) {
        auto const& context = *static_cast<BuildContext*>(pointer);
        return treeCreateFork(request.params, request.key, request.processRange,
            request.particleRange, request.box, request.depth, request.localRoot, threadme,
            right ? context.right : context.left);
    };
    return treeBuild({params, key, processRange, particleRange, box, depth, localRoot}, policy);
}

// Compatibility distributed cell-forest diagnostics can inspect remote roots.
// Production whole-cell exchange builds every source tree locally and never
// invokes this adapter.
static TreeNode readCellNode(TreeId id) {
    return *treeLocalNode(id);
}
HPX_PLAIN_ACTION(readCellNode, ReadCellNodeAction);
static std::unordered_map<int64_t, std::unique_ptr<TreeNode>> remoteCellNodes;
static MutexType remoteCellMutex;
static TreeNode const* readRemoteCellNode(TreeId id) {
    std::unique_lock<MutexType> lock(remoteCellMutex);
    int64_t const key = (int64_t(id.proc) << 32) | uint32_t(id.index);
    auto found = remoteCellNodes.find(key);
    if (found == remoteCellNodes.end()) {
        treeRecordRemoteFetch();
        auto value = hpx::async<ReadCellNodeAction>(hpxLocalities()[id.proc], id).get();
        found = remoteCellNodes.emplace(key, std::make_unique<TreeNode>(value)).first;
    }
    return found->second.get();
}
static void resetCellTreeState() {
    remoteCellNodes.clear();
    cellRootReturns.clear();
}

static TreeCreateReturn treeReadCellRoot(TreeId id) {
    if (id.proc != hpxRank()) {
        return hpx::async<TreeReadCellRootAction>(hpxLocalities()[id.proc], id).get();
    }
    std::unique_lock<MutexType> lock(cellRootMutex);
    auto const entry = cellRootReturns.find(id.index);
    if (entry == cellRootReturns.end()) {
        throw std::runtime_error("Cell tree root does not belong to the current forest");
    }
    return entry->second;
}

HPX_PLAIN_ACTION(treePrepareCellForest, TreePrepareCellForestAction);

void treePrepareCellForest() {
    Vector<hpx::future<void>> futures;
    for (auto const& child : hpxChildren()) {
        futures.push_back(hpx::async<TreePrepareCellForestAction>(child));
    }
    try {
        size_t occupiedCells = 0;
        uint64_t previous = std::numeric_limits<uint64_t>::max();
        int const meshSize = getOptions().gravity.meshSize;
        for (ParticleInteger i = 0; i < particlesSize(); ++i) {
            uint64_t key = 0;
            for (int dim = 0; dim < kDimensions; ++dim) {
                key = key * meshSize +
                    detail::particleMeshCoordinate(particlesPos(dim, i).raw(), meshSize);
            }
            occupiedCells += key != previous;
            previous = key;
        }
        treePrepareCellForestLocal(size_t(particlesSize()), occupiedCells);
    } catch (...) {
        hpx::wait_all(futures.begin(), futures.end());
        throw;
    }
    hpx::wait_all(futures.begin(), futures.end());
    for (auto& future : futures) {
        future.get();
    }
}

void treePrepareCellForestLocal(size_t sourceParticleCount, size_t sourceRootCount) {
    treeResetLocal();
    treeRegisterRemoteAccess(readRemoteCellNode, resetCellTreeState);
    resetCellTreeState();
    auto const multiply = [](size_t first, size_t second) {
        if (second != 0 && first > std::numeric_limits<size_t>::max() / second) {
            throw std::overflow_error("Cell tree capacity overflows size_t");
        }
        return first * second;
    };
    auto const add = [](size_t first, size_t second) {
        if (first > std::numeric_limits<size_t>::max() - second) {
            throw std::overflow_error("Cell tree capacity overflows size_t");
        }
        return first + second;
    };
    size_t levels = 0;
    for (size_t ranks = hpxSize(); ranks > 1; ranks = (ranks + 1) / 2) {
        ++levels;
    }
    // Local kd nodes, sparse domain joins, allocator slack and cache allowance.
    // No capacity term scales as mesh cells times ranks.
    size_t capacity = treeNodeCapacity(sourceParticleCount);
    capacity = add(capacity, multiply(4, sourceParticleCount));
    capacity = add(capacity, multiply(sourceRootCount, levels + 2));
    capacity = add(capacity,
        multiply(size_t(getOptions().compute.treeAllocLineSize),
            4 * size_t(hpxHardwareConcurrency()) + 2));
    treeAllocateNodesLocal(capacity);
}

TreeCellRoot treeCreateCell(ParticleMeshCell const& cell, int meshSize, FmmParticleStore store,
    Range<double> const* sourceBox) {
    uint64_t const cellCount = detail::particleMeshCellCount(meshSize);
    auto const sources = fmmSourceView(store);
    if (cell.key >= cellCount || cell.begin < 0 || cell.end <= cell.begin ||
        cell.end > sources.count) {
        throw std::invalid_argument("Cell tree range is outside its source store");
    }
    size_t domainKey = 1;
    int domainDepth = 0;
    Pair<int> processRange(0, hpxSize());
    while (processRange.second - processRange.first > 1) {
        int const mid = (processRange.first + processRange.second) / 2;
        if (hpxRank() < mid) {
            processRange.second = mid;
            domainKey <<= 1;
        } else {
            processRange.first = mid;
            domainKey = (domainKey << 1) + 1;
        }
        ++domainDepth;
    }
    detail::TreeCellBuild context;
    context.meshSize = meshSize;
    context.cellKey = cell.key;
    context.sourceStore = store;
    auto const cellBox = detail::treeCellBox(cell.key, meshSize);
    auto box = cellBox;
    if (sourceBox != nullptr) {
        box = box.intersection(*sourceBox);
    } else if (store == FmmParticleStore::Owned) {
        box = box.intersection(hpxSize() == 1 ? unitBox<double>() : domainsRange(domainKey));
    }
    for (int dim = 0; dim < kDimensions; ++dim) {
        if (box.end[dim] < box.begin[dim]) {
            throw std::runtime_error("Occupied mesh cell does not intersect its source domain");
        }
    }
    auto const& gravity = getOptions().gravity;
    TreeCreateParameters params(0, gravity.openingAngle, gravity.supportRadius);
    auto const built = treeCreateCellImpl(params, domainKey, processRange,
        Pair<ParticleInteger>(cell.begin, cell.end), box, domainDepth, true, context);
    if (store == FmmParticleStore::Owned && getOptions().hydro.enabled) {
        // The completed cell owns these particle/category pairs. Repair only
        // their reverse links while other disjoint cells may still be sorting.
        for (ParticleInteger i = cell.begin; i < cell.end; ++i) {
            if (particlesType(i) == kSphType) {
                ParticleInteger const gasIndex = particlesCatIndex(i);
                if (gasIndex < 0 || gasIndex >= sphParticlesSize()) {
                    throw std::runtime_error("Invalid gas index after cell tree construction");
                }
                sphParticlesParticleIndex(gasIndex) = i;
            }
        }
    }
    {
        std::unique_lock<MutexType> lock(cellRootMutex);
        cellRootReturns.emplace(built.id.index, built);
    }
    return {cell.key, built.id, store == FmmParticleStore::Ghost};
}

Vector<TreeCellRoot> treeCreateCellForest(Vector<ParticleMeshCell> const& cells, int meshSize) {
    std::unique_lock<SharedMutexType> particleLock(particlesSharedMutex());
    for (size_t i = 1; i < cells.size(); ++i) {
        auto const& previous = cells[i - 1];
        auto const& cell = cells[i];
        if (previous.ghost > cell.ghost ||
            (previous.ghost == cell.ghost &&
                (previous.end > cell.begin || previous.key >= cell.key))) {
            throw std::invalid_argument(
                "Cell tree ranges must be disjoint and sorted by store then key");
        }
    }
    Vector<TreeCellRoot> result(cells.size());
    size_t const workers = std::min(cells.size(), size_t(std::max(1, hpxHardwareConcurrency())));
    Vector<hpx::future<void>> futures;
    try {
        for (size_t worker = 0; worker < workers; ++worker) {
            futures.push_back(hpx::async([&, worker]() {
                for (size_t i = worker; i < cells.size(); i += workers) {
                    result[i] = treeCreateCell(cells[i], meshSize,
                        cells[i].ghost ? FmmParticleStore::Ghost : FmmParticleStore::Owned);
                }
            }));
        }
    } catch (...) {
        hpx::wait_all(futures.begin(), futures.end());
        throw;
    }
    // Drain every worker even on failure before releasing the reorder lock.
    hpx::wait_all(futures.begin(), futures.end());
    for (auto& future : futures) {
        future.get();
    }
    particlesCacheFreeLocal();
    return result;
}

TreeId treeCombineCellRoots(uint64_t cellKey, int meshSize, Vector<TreeId> contributors) {
    auto const box = detail::treeCellBox(cellKey, meshSize);
    if (contributors.empty()) {
        throw std::invalid_argument("An occupied cell must have at least one tree root");
    }
    std::sort(contributors.begin(), contributors.end());
    for (size_t i = 0; i < contributors.size(); ++i) {
        auto const root = contributors[i];
        if (root.proc < 0 || root.proc >= hpxSize() || root.index < 0 ||
            (i != 0 && contributors[i - 1].proc == root.proc)) {
            throw std::invalid_argument(
                "Cell roots must identify distinct participating localities");
        }
    }
    if (contributors.size() == 1) {
        return contributors.front();
    }
    detail::TreeCellBuild context;
    context.meshSize = meshSize;
    context.cellKey = cellKey;
    context.roots = std::move(contributors);
    auto const& gravity = getOptions().gravity;
    TreeCreateParameters params(0, gravity.openingAngle, gravity.supportRadius);
    return treeCreateFork(params, 1, Pair<int>(0, hpxSize()), Pair<ParticleInteger>(-1, -1), box, 0,
        false, false, context)
        .get()
        .id;
}
