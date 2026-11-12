/* CosmicTiger - Nitadori PMMM mesh exchange; BSL-1.0. */

#include <cosmictiger/gravity/tree_multipole.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/domain.hpp>

#include "actions.hpp"

namespace pmmm::detail {

/// Wrap a signed mesh coordinate into the current mesh's nonnegative index range.
int periodicIndex(int index) {
    int const n = cache.meshSize;
    return (index % n + n) % n;
}

/// Convert an unsigned periodic coordinate to its signed minimum-image representative.
int signedIndex(int index) {
    return index > cache.meshSize / 2 ? index - cache.meshSize : index;
}

/// Decode a linear mesh key into three cell coordinates.
std::array<int, 3> coordinates(uint64_t key) {
    uint64_t const n = cache.meshSize;
    return {int(key / (n * n)), int(key / n % n), int(key % n)};
}

// Particle domains cut only on coarse-mesh planes and every stage side is a
// multiple of the coarse side, so one locality owns all the
// particles of any cell. The cell center identifies that locality.
int cellOwner(uint64_t key) {
    auto const cell = coordinates(key);
    Vector3<double> center;
    for (int axis = 0; axis < kDimensions; ++axis) {
        center[axis] = (cell[axis] + 0.5) / cache.meshSize;
    }
    return domainsFindRank(center);
}

/// Pack a near-field particle with cell-centered mesh offsets, mass, and stable owner/index
/// identity.
/// Add received cell moments and radial trace to local FFT-slab storage under the
/// accumulation lock.
void accumulate(std::vector<CellData> const& cells) {
    std::unique_lock<MutexType> lock(state.accumulationMutex);
    for (auto const& cell : cells) {
        size_t const begin = cache.cellIndex(cell.key) * cache.sourceCount;
        if (cell.coefficients.size() != size_t(cache.sourceCount)) {
            throw std::runtime_error("PMMM moment packet has the wrong order");
        }
        for (int component = 0; component < cache.sourceCount; ++component) {
            state.moments[begin + component] += cell.coefficients[component];
        }
        state.radialTrace += cell.radialTrace;
        state.radialTraces[cache.cellIndex(cell.key)] += cell.radialTrace;
        observe(PmmmCellEvent::MomentsDeposited, cell.key);
    }
    addCpuFlops(FlopCount(cells.size()) * (cache.sourceCount + 1), FlopPrecision::Double);
}

/// Find a locally owned cell for the current solve generation, rejecting stale requests.
static ParticleMeshCell const* ownedCell(uint64_t generation, uint64_t key) {
    if (generation != state.request.generation) {
        throw std::runtime_error("PMMM cell request belongs to another solve");
    }
    auto const found = state.ownedCells.find(key);
    return found == state.ownedCells.end() ? nullptr : &state.cells[found->second];
}

// Occupied-cell particle counts; an unoccupied cell reports zero. Every cell
// requested here lies in this locality's domain, so zero means empty.
std::vector<ParticleInteger> readCellCounts(
    uint64_t generation, std::vector<uint64_t> const& keys) {
    std::vector<ParticleInteger> counts;
    counts.reserve(keys.size());
    for (auto key : keys) {
        auto const* cell = ownedCell(generation, key);
        counts.push_back(cell == nullptr ? 0 : cell->end - cell->begin);
    }
    return counts;
}

std::vector<double> readExpansion(uint64_t key) {
    state.farReady.get();
    int const n = cache.meshSize;
    double const normalization = 1.0 / (double(n) * n * n);
    size_t const index = cache.cellIndex(key);
    std::vector<double> coefficients(cache.localStride);
    size_t const padded = ((index / n) * (2 * (n / 2 + 1)) + index % n) * cache.localStride;
    for (int component = 0; component < cache.localStride; ++component) {
        // One float-to-double conversion and one double multiplication.
        coefficients[component] = normalization * cache.localReal[padded + component];
    }
    addCpuFlops(3 + 2 * cache.localStride, FlopPrecision::Double);
    return coefficients;
}

/// Group cell moments by FFT-slab owner and complete all remote deposition requests.
void depositCells(std::vector<CellData> cells) {
    std::unordered_map<int, std::vector<CellData>> packets;
    for (auto& cell : cells) {
        int const x = coordinates(cell.key)[0];
        packets[cache.slabOwner[x]].push_back(std::move(cell));
    }
    std::vector<hpx::future<void>> futures;
    futures.reserve(packets.size());
    std::exception_ptr failure;
    try {
        for (auto& [owner, packet] : packets) {
            futures.push_back(
                hpx::async<PmmmAccumulateAction>(cache.owners[owner], std::move(packet)));
        }
    } catch (...) {
        failure = std::current_exception();
    }
    for (auto& future : futures) {
        future.wait();
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
    for (auto& future : futures) {
        future.get();
    }
}

/// Convert an unrounded endpoint tree-root multipole to mesh-cell moments and record the source
/// rung snapshot.
CellData rootMoments(size_t cellIndex, TreeCreateReturn const& root) {
    auto const& range = state.cells[cellIndex];
    if (!root.multi.isDouble()) {
        throw std::runtime_error("Owned mesh deposition requires an unrounded double tree root");
    }
    auto const cellCoordinates = coordinates(range.key);
    Vector3<Position> center;
    Vector3<double> centerCorrection;
    for (int axis = 0; axis < kDimensions; ++axis) {
        double const coordinate = (cellCoordinates[axis] + 0.5) / cache.meshSize;
        center[axis] = coordinate;
        centerCorrection[axis] = center[axis].toDouble() - coordinate;
    }
    CellData cell;
    cell.key = range.key;
    cell.coefficients.resize(cache.sourceCount);
    FlopCount flops = treeSourceToMesh<kFmmOrder>(cell.coefficients.data(), root.multi.precise(),
        root.pos, center, cache.meshSize, centerCorrection);
    // Independent radial summary about the mesh center. No source trace is
    // inferred from the STF moments, and no second harmonic P2M is performed.
    for (ParticleInteger i = range.begin; i < range.end; ++i) {
        double const mass = particlesMass(i);
        for (int axis = 0; axis < kDimensions; ++axis) {
            double const offset = (unwrappedDoubleDistance(particlesPos(axis, i), center[axis]) +
                                      centerCorrection[axis]) *
                cache.meshSize;
            cell.radialTrace += mass * offset * offset;
        }
        state.stageRungs[i - state.activeRange.first] = particlesRung(i);
        flops += 1 + 7 * kDimensions;
    }
    addCpuFlops(flops + kDimensions * (kPositionBits == 64 ? 9 : 6), FlopPrecision::Double);
    return cell;
}

/// Deposit all owned-cell source moments, reusing prepared roots when the solve permits it.
void deposit() {
    // These futures contain no near consumers and can complete before FFT.
    for (auto& task : state.ownedTasks) {
        try {
            task.get();
        } catch (...) {
            recordFailure(std::current_exception());
        }
    }
    std::exception_ptr failure;
    {
        std::lock_guard<MutexType> lock(state.queueMutex);
        failure = state.failure;
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
}

std::vector<double> localExpansion(uint64_t key) {
    auto const owner = cache.slabOwner[key / (uint64_t(cache.meshSize) * cache.meshSize)];
    auto coefficients = cache.owners[owner] == hpx::find_here() ?
        readExpansion(key) :
        hpx::async<PmmmReadExpansionAction>(cache.owners[owner], key).get();
    if (coefficients.size() != size_t(cache.localStride)) {
        throw std::runtime_error("PMMM local expansion is missing");
    }
    return coefficients;
}

}    // namespace pmmm::detail

HPX_REGISTER_ACTION(PmmmAccumulateAction, PmmmAccumulateAction);
HPX_REGISTER_ACTION(PmmmReadExpansionAction, PmmmReadExpansionAction);
HPX_REGISTER_ACTION(PmmmReadCellCountsAction, PmmmReadCellCountsAction);
HPX_REGISTER_ACTION(PmmmDepositAction, PmmmDepositAction);
