/* CosmicTiger - Nitadori PMMM mesh exchange; BSL-1.0. */

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

PmmmNearParticle particle(ParticleInteger index, std::array<int, 3> const& cell) {
    PmmmNearParticle result{};
    for (int axis = 0; axis < kDimensions; ++axis) {
        result.offset[axis] = positionDomainCoordinate(particlesPos(axis, index)) * cache.meshSize -
            (cell[axis] + 0.5);
    }
    result.mass = particlesMass(index);
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
        if (cell.sourceRank >= 0) {
            state.sourceOwners[cell.key].push_back({cell.sourceRank, cell.particleCount});
        } else {
            auto& sources = state.sources[cell.key];
            sources.insert(sources.end(), cell.sources.begin(), cell.sources.end());
        }
    }
}

std::vector<CellOwners> readOwners(std::vector<uint64_t> const& keys) {
    std::vector<CellOwners> result;
    for (auto key : keys) {
        auto const found = state.sourceOwners.find(key);
        if (found != state.sourceOwners.end()) {
            result.push_back({key, found->second});
        }
    }
    return result;
}

std::vector<CellData> readCells(std::vector<uint64_t> const& keys, uint64_t expansionKey) {
    state.farReady.get();
    std::vector<CellData> result;
    result.reserve(keys.size());
    int const n = cache.meshSize;
    double const normalization = 1.0 / (double(n) * n * n);
    for (auto key : keys) {
        CellData cell;
        cell.key = key;
        size_t const index = cache.cellIndex(key);
        auto const found = state.sources.find(key);
        if (found != state.sources.end()) {
            cell.sources = found->second;
        }
        if (key == expansionKey) {
            cell.coefficients.resize(cache.localCount);
            size_t const padded = ((index / n) * (2 * (n / 2 + 1)) + index % n) * cache.localCount;
            for (int component = 0; component < cache.localCount; ++component) {
                cell.coefficients[component] = normalization * cache.localReal[padded + component];
            }
            double const traceFactor = (4 * std::numbers::pi / 3) / (double(n) * n);
            if (cache.order >= 4) {
                cell.coefficients[0] -= 0.5 * traceFactor * state.background[4];
            }
            cell.coefficients[3] -= 2 * traceFactor * state.background[1];
            cell.coefficients[1] -= 2 * traceFactor * state.background[2];
            cell.coefficients[2] -= traceFactor * state.background[3];
        }
        result.push_back(std::move(cell));
    }
    return result;
}

void deposit() {
    // Bound packet sizes and outstanding remote writes independently of N.
    constexpr size_t kCellsPerPacket = 32;
    parallelFor((state.cells.size() + kCellsPerPacket - 1) / kCellsPerPacket, [&](size_t batch) {
        std::unordered_map<int, std::vector<CellData>> packets;
        size_t const end = std::min(state.cells.size(), (batch + 1) * kCellsPerPacket);
        for (size_t cellIndex = batch * kCellsPerPacket; cellIndex < end; ++cellIndex) {
            auto const& range = state.cells[cellIndex];
            auto const cellCoordinates = coordinates(range.key);
            CellData cell;
            cell.key = range.key;
            cell.coefficients.assign(cache.sourceCount, 0.0);
            bool const useFmm = getOptions().gravity.pmmmNearMethod == "fmm";
            if (useFmm) {
                cell.sourceRank = hpxRank();
                cell.particleCount = range.end - range.begin;
            } else {
                cell.sources.reserve(range.end - range.begin);
            }
            for (ParticleInteger i = range.begin; i < range.end; ++i) {
                auto const source = particle(i, cellCoordinates);
                addMoments(cache.order, cell.coefficients.data(), source.mass, source.offset);
                for (int axis = 0; axis < kDimensions; ++axis) {
                    cell.radialTrace += source.mass * source.offset[axis] * source.offset[axis];
                }
                if (!useFmm) {
                    cell.sources.push_back(source);
                }
            }
            packets[cache.slabOwner[cellCoordinates[0]]].push_back(std::move(cell));
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
    });
}

std::vector<double> localExpansion(uint64_t key) {
    auto const owner = cache.slabOwner[key / (uint64_t(cache.meshSize) * cache.meshSize)];
    auto packet =
        hpx::async<PmmmReadCellsAction>(cache.owners[owner], std::vector<uint64_t>{key}, key).get();
    if (packet.size() != 1 || packet[0].coefficients.size() != size_t(cache.localCount)) {
        throw std::runtime_error("PMMM local expansion is missing");
    }
    return std::move(packet[0].coefficients);
}

}    // namespace pmmm::detail

HPX_REGISTER_ACTION(PmmmAccumulateAction, PmmmAccumulateAction);
HPX_REGISTER_ACTION(PmmmReadCellsAction, PmmmReadCellsAction);
HPX_REGISTER_ACTION(PmmmReadOwnersAction, PmmmReadOwnersAction);
HPX_REGISTER_ACTION(PmmmDepositAction, PmmmDepositAction);
