/* CosmicTiger - Nitadori PMMM ghost exchange; BSL-1.0. */

#include "actions.hpp"

namespace pmmm::detail {

Vector<ParticleGhost> readGhostParticles(uint64_t generation, uint64_t key) {
    if (generation != state.request.generation) {
        throw std::runtime_error("PMMM ghost request belongs to another solve");
    }
    auto const found = state.ownedCells.find(key);
    if (found == state.ownedCells.end()) {
        throw std::runtime_error("PMMM ghost request reached the wrong particle owner");
    }
    auto const& source = state.sourceCells[found->second];
    // Tree construction performs the final in-cell permutation. Kicks can now
    // change live rungs, but outgoing membership uses the frozen stage snapshot.
    source->ready.get();
    auto const& cell = source->range;
    Vector<ParticleGhost> result(cell.end - cell.begin);
    for (ParticleInteger i = cell.begin; i < cell.end; ++i) {
        auto& particle = result[i - cell.begin];
        for (int dim = 0; dim < kDimensions; ++dim) {
            particle.x[dim] = particlesPos(dim, i);
        }
        particle.type = getOptions().hydro.enabled ? particlesType(i) : kDarkMatterType;
        particle.rung = state.stageRungs[i - state.activeRange.first];
    }
    return result;
}

std::vector<uint64_t> neighbors(uint64_t key) {
    std::vector<uint64_t> result;
    auto const center = coordinates(key);
    int const n = cache.meshSize;
    int const width = cache.nearWidth;
    for (int x = -width; x <= width; ++x) {
        for (int y = -width; y <= width; ++y) {
            for (int z = -width; z <= width; ++z) {
                uint64_t const sx = periodicIndex(center[0] + x);
                uint64_t const sy = periodicIndex(center[1] + y);
                uint64_t const sz = periodicIndex(center[2] + z);
                result.push_back((sx * n + sy) * n + sz);
            }
        }
    }
    return result;
}

void planGhosts() {
    // Every neighbor cell has exactly one owner, known from the particle-domain
    // tree. Ask each remote owner how many particles its cells hold so the ghost
    // arena can be reserved before any record arrives; an empty cell needs no
    // ghost. Owned neighbors are already source cells.
    std::unordered_map<int, std::vector<uint64_t>> requests;
    {
        std::unordered_set<uint64_t> keys;
        for (auto const& cell : state.cells) {
            for (auto key : neighbors(cell.key)) {
                if (keys.insert(key).second) {
                    int const owner = cellOwner(key);
                    if (owner != hpxRank()) {
                        requests[owner].push_back(key);
                    }
                }
            }
        }
    }
    std::vector<std::pair<int, hpx::future<std::vector<ParticleInteger>>>> futures;
    futures.reserve(requests.size());
    try {
        for (auto& [owner, request] : requests) {
            std::sort(request.begin(), request.end());
            futures.emplace_back(owner,
                hpx::async<PmmmReadCellCountsAction>(
                    hpxLocalities()[owner], state.request.generation, request));
        }
    } catch (...) {
        for (auto& [owner, future] : futures) {
            future.wait();
        }
        throw;
    }
    struct GhostCell {
        uint64_t key;
        int owner;
        ParticleInteger count;
    };
    std::vector<GhostCell> directory;
    for (auto& [owner, future] : futures) {
        auto const counts = future.get();
        auto const& request = requests[owner];
        if (counts.size() != request.size()) {
            throw std::runtime_error("PMMM cell count reply has the wrong length");
        }
        for (size_t i = 0; i < request.size(); ++i) {
            if (counts[i] > 0) {
                directory.push_back({request[i], owner, counts[i]});
            }
        }
    }
    // Arrival order must not change source traversal or floating-point sums.
    std::sort(directory.begin(), directory.end(),
        [](auto const& first, auto const& second) { return first.key < second.key; });
    ParticleInteger ghostCount = 0;
    for (auto const& cell : directory) {
        if (cell.count > std::numeric_limits<ParticleInteger>::max() - ghostCount) {
            throw std::runtime_error("PMMM ghost count exceeds its index range");
        }
        auto source = std::make_shared<SourceCell>();
        source->range = {cell.key, ghostCount, ghostCount + cell.count, true};
        source->owner = cell.owner;
        ghostCount += cell.count;
        state.sourceCellsByKey.emplace(cell.key, state.sourceCells.size());
        state.sourceCells.push_back(std::move(source));
    }
    fmmGhostReserve(ghostCount);
    state.pendingGhosts = state.sourceCells.size() - state.cells.size();
    state.diagnostics.ghostParticles = ghostCount;
    state.diagnostics.ghostCells = state.pendingGhosts;
    state.diagnostics.ghostBytes =
        uint64_t(ghostCount) * (kDimensions * sizeof(PositionInteger) + 2 * sizeof(char));
    state.sinkSources.resize(state.cells.size());
    state.unresolvedSources.resize(state.cells.size());
    for (size_t sink = 0; sink < state.cells.size(); ++sink) {
        auto& dependencies = state.sinkSources[sink];
        for (auto key : neighbors(state.cells[sink].key)) {
            auto const found = state.sourceCellsByKey.find(key);
            if (found != state.sourceCellsByKey.end()) {
                dependencies.push_back(found->second);
                state.sourceCells[found->second]->dependents.push_back(sink);
            }
        }
        state.unresolvedSources[sink] = dependencies.size();
    }
    treePrepareCellForestLocal(
        size_t(state.activeRange.second - state.activeRange.first) + ghostCount,
        state.sourceCells.size());
}

void receiveGhostCell(size_t index) {
    auto const view = fmmGhostView();
    auto const& source = state.sourceCells[index];
    auto const& cell = source->range;
    bool pending = true;
    try {
        observe(PmmmCellEvent::GhostRequested, cell.key);
        auto packet = hpx::async<PmmmReadGhostParticlesAction>(
            hpxLocalities()[source->owner], state.request.generation, cell.key)
                          .get();
        if (packet.size() != size_t(cell.end - cell.begin)) {
            throw std::runtime_error("PMMM ghost packet has the wrong particle count");
        }
        for (ParticleInteger i = cell.begin; i < cell.end; ++i) {
            auto const& particle = packet[i - cell.begin];
            for (int dim = 0; dim < kDimensions; ++dim) {
                view.positions[dim][i] = particle.x[dim];
            }
            view.types[i] = particle.type;
            view.rungs[i] = particle.rung;
        }
        auto root = treeCreateCell(cell, cache.meshSize, FmmParticleStore::Ghost);
        --state.pendingGhosts;
        pending = false;
        publishSource(index, root);
        observe(PmmmCellEvent::GhostReady, cell.key);
    } catch (...) {
        failSource(index, std::current_exception());
    }
    if (pending) {
        --state.pendingGhosts;
    }
}

}    // namespace pmmm::detail

HPX_REGISTER_ACTION(PmmmReadGhostParticlesAction, PmmmReadGhostParticlesAction);
HPX_REGISTER_ACTION(PmmmPlanGhostsAction, PmmmPlanGhostsAction);
