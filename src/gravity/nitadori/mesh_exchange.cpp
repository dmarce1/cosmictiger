/* CosmicTiger - Nitadori PMMM mesh exchange; BSL-1.0. */

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
    Array<double, kDimensions> center;
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

void deposit() {
    // Bound packet sizes and outstanding remote writes independently of N.
    constexpr size_t kCellsPerPacket = 32;
    parallelFor((state.cells.size() + kCellsPerPacket - 1) / kCellsPerPacket,
        "gravity.pmmm.depositMoments", [&](size_t batch) {
            FlopCount flops = 0;
            std::unordered_map<int, std::vector<CellData>> packets;
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
                packets[cache.slabOwner[cellCoordinates[0]]].push_back(std::move(cell));
            }
            addCpuFlops(flops, FlopPrecision::Double);
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
