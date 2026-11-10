/* CosmicTiger - Nitadori PMMM mesh exchange; BSL-1.0. */

#include <cosmictiger/gravity/tree_multipole.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/domain.hpp>

#include "actions.hpp"

namespace pmmm::detail {

int periodicIndex(int index) {
    int const n = cache.meshSize;
    return (index % n + n) % n;
}

int signedIndex(int index) {
    return index > cache.meshSize / 2 ? index - cache.meshSize : index;
}

std::array<int, 3> coordinates(uint64_t key) {
    uint64_t const n = cache.meshSize;
    return {int(key / (n * n)), int(key / n % n), int(key % n)};
}

// Particle domains cut only on coarse-mesh planes and every stage mesh is a
// power of two no coarser than the coarse mesh, so one locality owns all the
// particles of any cell. The cell center identifies that locality.
int cellOwner(uint64_t key) {
    auto const cell = coordinates(key);
    Vector3<double> center;
    for (int axis = 0; axis < kDimensions; ++axis) {
        center[axis] = (cell[axis] + 0.5) / cache.meshSize;
    }
    return domainsFindRank(center);
}

PmmmNearParticle particle(ParticleInteger index, std::array<int, 3> const& cell, FlopCount& flops) {
    PmmmNearParticle result{};
    for (int axis = 0; axis < kDimensions; ++axis) {
        result.offset[axis] = positionDomainCoordinate(particlesPos(axis, index)) * cache.meshSize -
            (cell[axis] + 0.5);
        // Fixed conversion, mesh scaling, center construction, and subtraction.
        flops += 4;
        if constexpr (kPositionBits == 64) {
            // Mirror the half-open rounding branches in positionDomainCoordinate.
            // These repeated predicates are accounting overhead, not solver work.
            auto const position = particlesPos(axis, index);
            double const coordinate = position.toDouble();
            if (coordinate < 1.0) {
                flops += 4;    // Fixed64 limb split in Position(coordinate).
            }
            if (coordinate >= 1.0 || Position(coordinate).raw() > position.raw()) {
                ++flops;    // nextafter.
            }
        }
    }
    if (frozenNearRequest(state.request)) {
        // Frozen membership: the particle may have drifted out of its cell or
        // across the periodic seam. Its offset is taken from the image nearest
        // the frozen cell center, which is exact while no particle travels half
        // a box within one outer interval (docs/pmmm.md).
        double const n = cache.meshSize;
        for (int axis = 0; axis < kDimensions; ++axis) {
            result.offset[axis] -= n * std::nearbyint(result.offset[axis] / n);
        }
        flops += 3 * kDimensions;
    }
    result.mass = particlesMass(index);
    ++flops;    // Promote the stored float particle mass to double.
    result.rank = hpxRank();
    result.index = index;
    return result;
}

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
        observe(PmmmCellEvent::MomentsDeposited, cell.key);
    }
    addCpuFlops(FlopCount(cells.size()) * (cache.sourceCount + 1), FlopPrecision::Double);
}

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

std::vector<CellParticles> readCellParticles(
    uint64_t generation, std::vector<uint64_t> const& keys) {
    std::vector<CellParticles> result;
    result.reserve(keys.size());
    FlopCount flops = 0;
    for (auto key : keys) {
        auto const* cell = ownedCell(generation, key);
        if (cell == nullptr) {
            continue;
        }
        CellParticles particles;
        particles.key = key;
        particles.particles.reserve(cell->end - cell->begin);
        auto const center = coordinates(key);
        for (ParticleInteger i = cell->begin; i < cell->end; ++i) {
            particles.particles.push_back(particle(i, center, flops));
        }
        result.push_back(std::move(particles));
    }
    addCpuFlops(flops, FlopPrecision::Double);
    return result;
}

std::vector<double> readExpansion(uint64_t key) {
    state.farReady.get();
    int const n = cache.meshSize;
    double const normalization = 1.0 / (double(n) * n * n);
    size_t const index = cache.cellIndex(key);
    std::vector<double> coefficients(cache.localCount);
    size_t const padded = ((index / n) * (2 * (n / 2 + 1)) + index % n) * cache.localCount;
    for (int component = 0; component < cache.localCount; ++component) {
        // One float-to-double conversion and one double multiplication.
        coefficients[component] = normalization * cache.localReal[padded + component];
    }
    double const traceFactor = (4 * std::numbers::pi / 3) / (double(n) * n);
    if (cache.order >= 4) {
        coefficients[0] -= 0.5 * traceFactor * state.background[4];
    }
    coefficients[3] -= 2 * traceFactor * state.background[1];
    coefficients[1] -= 2 * traceFactor * state.background[2];
    coefficients[2] -= traceFactor * state.background[3];
    addCpuFlops(3 + 2 * cache.localCount + 10 + (cache.order >= 4 ? 3 : 0), FlopPrecision::Double);
    return coefficients;
}

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

CellData rootMoments(size_t cellIndex, TreeCreateReturn const& root) {
    auto const& range = state.cells[cellIndex];
    if (!root.multi.isDouble()) {
        throw std::runtime_error("Owned mesh deposition requires an unrounded double tree root");
    }
    auto const cellCoordinates = coordinates(range.key);
    Vector3<Position> center;
    for (int axis = 0; axis < kDimensions; ++axis) {
        center[axis] = (cellCoordinates[axis] + 0.5) / cache.meshSize;
    }
    CellData cell;
    cell.key = range.key;
    cell.coefficients.resize(cache.sourceCount);
    FlopCount flops = fmmDispatch([&](auto tag) {
        return treeSourceToMesh<decltype(tag)::value>(
            cell.coefficients.data(), root.multi.precise(), root.pos, center, cache.meshSize);
    });
    // Independent radial summary about the mesh center. No source trace is
    // inferred from the STF moments, and no second harmonic P2M is performed.
    for (ParticleInteger i = range.begin; i < range.end; ++i) {
        double const mass = particlesMass(i);
        for (int axis = 0; axis < kDimensions; ++axis) {
            double const offset =
                unwrappedDoubleDistance(particlesPos(axis, i), center[axis]) * cache.meshSize;
            cell.radialTrace += mass * offset * offset;
        }
        state.stageRungs[i - state.activeRange.first] = particlesRung(i);
        flops += 1 + 6 * kDimensions;
    }
    addCpuFlops(flops + kDimensions * (kPositionBits == 64 ? 6 : 3), FlopPrecision::Double);
    return cell;
}

void deposit() {
    if (reuseOwnedRoots()) {
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
        return;
    }
    // Bound packet sizes and outstanding remote writes independently of N.
    constexpr size_t kCellsPerPacket = 32;
    parallelFor((state.cells.size() + kCellsPerPacket - 1) / kCellsPerPacket,
        "gravity.pmmm.depositMoments", [&](size_t batch) {
            FlopCount flops = 0;
            std::vector<CellData> cells;
            size_t const end = std::min(state.cells.size(), (batch + 1) * kCellsPerPacket);
            for (size_t cellIndex = batch * kCellsPerPacket; cellIndex < end; ++cellIndex) {
                auto const& range = state.cells[cellIndex];
                auto const cellCoordinates = coordinates(range.key);
                CellData cell;
                cell.key = range.key;
                cell.coefficients.assign(cache.sourceCount, 0.0);
                for (ParticleInteger i = range.begin; i < range.end; ++i) {
                    auto const source = particle(i, cellCoordinates, flops);
                    flops += addMoments(
                        cache.order, cell.coefficients.data(), source.mass, source.offset);
                    for (int axis = 0; axis < kDimensions; ++axis) {
                        cell.radialTrace += source.mass * source.offset[axis] * source.offset[axis];
                    }
                    flops += 3 * kDimensions;
                }
                cells.push_back(std::move(cell));
            }
            addCpuFlops(flops, FlopPrecision::Double);
            depositCells(std::move(cells));
        });
}

std::vector<double> localExpansion(uint64_t key) {
    auto const owner = cache.slabOwner[key / (uint64_t(cache.meshSize) * cache.meshSize)];
    auto coefficients = cache.owners[owner] == hpx::find_here() ?
        readExpansion(key) :
        hpx::async<PmmmReadExpansionAction>(cache.owners[owner], key).get();
    if (coefficients.size() != size_t(cache.localCount)) {
        throw std::runtime_error("PMMM local expansion is missing");
    }
    return coefficients;
}

}    // namespace pmmm::detail

HPX_REGISTER_ACTION(PmmmAccumulateAction, PmmmAccumulateAction);
HPX_REGISTER_ACTION(PmmmReadExpansionAction, PmmmReadExpansionAction);
HPX_REGISTER_ACTION(PmmmReadCellCountsAction, PmmmReadCellCountsAction);
HPX_REGISTER_ACTION(PmmmReadCellParticlesAction, PmmmReadCellParticlesAction);
HPX_REGISTER_ACTION(PmmmDepositAction, PmmmDepositAction);
