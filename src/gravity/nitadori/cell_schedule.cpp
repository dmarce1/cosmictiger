/* CosmicTiger - Nitadori PMMM cell schedule; BSL-1.0. */

#include "actions.hpp"

namespace pmmm::detail {

void recordFailure(std::exception_ptr failure) {
    {
        std::lock_guard<MutexType> lock(state.queueMutex);
        if (!state.failure) {
            state.failure = std::move(failure);
        }
    }
    state.queueChanged.notify_all();
}

void publishSource(size_t index, TreeCellRoot root) {
    auto const& source = state.sourceCells[index];
    if (!source->published.exchange(true)) {
        source->promise.set_value(root);
        {
            std::lock_guard<MutexType> lock(state.queueMutex);
            for (size_t sink : source->dependents) {
                if (--state.unresolvedSources[sink] == 0) {
                    state.readySinks.push_back(sink);
                }
            }
        }
        state.queueChanged.notify_all();
    }
}

void failSource(size_t index, std::exception_ptr failure) {
    auto const& source = state.sourceCells[index];
    if (!source->published.exchange(true)) {
        source->promise.set_exception(failure);
    }
    recordFailure(std::move(failure));
}

void clearGhosts() {
    fmmGhostClear();
    state.sourceCells.clear();
    state.sourceCellsByKey.clear();
    state.sinkSources.clear();
    state.tasks.clear();
    state.stageRungs.clear();
}

void computeReadyCells() {
    size_t const targetParticles = getOptions().gravity.pmmmBatchParticles;
    for (;;) {
        std::vector<size_t> sinks;
        size_t particleCount = 0;
        {
            std::unique_lock<MutexType> lock(state.queueMutex);
            state.queueChanged.wait(lock, []() {
                return state.failure || !state.readySinks.empty() || state.remainingSinks == 0;
            });
            if (state.failure || state.remainingSinks == 0) {
                return;
            }
            // Bound batches by particles so sparsely populated mesh cells can
            // still fill one shared CPU/GPU work packet.
            while (!state.readySinks.empty()) {
                size_t const index = state.readySinks.front();
                auto const& cell = state.cells[index];
                size_t const count = cell.end - cell.begin;
                if (!sinks.empty() && particleCount + count > targetParticles) {
                    break;
                }
                state.readySinks.pop_front();
                sinks.push_back(index);
                particleCount += count;
            }
        }
        try {
            std::sort(sinks.begin(), sinks.end());
            Vector<KickWorkitem> workitems;
            workitems.reserve(sinks.size());
            for (size_t sink : sinks) {
                auto const root = state.sourceCells[sink]->ready.get();
                KickWorkitem item;
                item.self = root.id;
                item.pos = treeGetNode(root.id)->pos;
                item.l.zero();
                item.geometry.meshSize = cache.meshSize;
                auto const center = coordinates(root.key);
                for (int dim = 0; dim < kDimensions; ++dim) {
                    item.geometry.sinkCell[dim] = center[dim];
                }
                for (size_t source : state.sinkSources[sink]) {
                    item.dchecklist.push_back(state.sourceCells[source]->ready.get().id);
                }
                workitems.push_back(std::move(item));
            }
            // Only this bounded batch retains particle fields while the FFT or
            // remote local-expansion reads finish. Production has no N-sized
            // acceleration or potential result arrays.
            auto fields = kickCellBatch(std::move(workitems), state.request.doPotential);
            bool const overlapped = state.pendingGhosts.load() != 0;
            for (size_t sink : sinks) {
                observe(PmmmCellEvent::NearCompleted, state.cells[sink].key);
            }
            state.farReady.get();
            double const h = 1.0 / cache.meshSize;
            double const localTrace = (4 * std::numbers::pi / 3) * h * h * state.background[0];
            FlopCount flops = 4;
            KickReturn result;
            size_t compact = 0;
            for (size_t sink : sinks) {
                auto const& cell = state.cells[sink];
                auto const center = coordinates(cell.key);
                auto const coefficients = localExpansion(cell.key);
                for (ParticleInteger index = cell.begin; index < cell.end; ++index, ++compact) {
                    if (compact >= fields.particleIndices.size() ||
                        fields.particleIndices[compact] != index) {
                        throw std::runtime_error("PMMM batch output lost its sink mapping");
                    }
                    std::array<double, 4> field{};
                    auto const value = particle(index, center, flops);
                    flops += evaluate(
                        cache.order, field, coefficients.data(), localTrace, value.offset, h);
                    field[0] += fields.potential[compact];
                    for (int dim = 0; dim < kDimensions; ++dim) {
                        field[dim + 1] += fields.acceleration[dim][compact];
                    }
                    flops += 4;
                    consumeField(index, field, result, flops);
                }
            }
            addCpuFlops(flops);
            {
                std::lock_guard<MutexType> lock(state.queueMutex);
                state.kickResult += result;
                auto const& counts = fields.interactionCounts;
                state.diagnostics.nearCpuAcceptedMultipoles +=
                    counts.cellCellDirect + counts.cellParticle + counts.particleCell;
                state.diagnostics.nearCpuDirectPairs += counts.particleParticlePairs;
                ++state.diagnostics.nearBatches;
                state.diagnostics.peakBatchParticles =
                    std::max<uint64_t>(state.diagnostics.peakBatchParticles, particleCount);
                state.diagnostics.completedWhileGhostsPending += overlapped ? sinks.size() : 0;
                state.remainingSinks -= sinks.size();
            }
            state.queueChanged.notify_all();
        } catch (...) {
            recordFailure(std::current_exception());
            return;
        }
    }
}

void startPipeline() {
    state.treeFetchStart = treeRemoteFetchCount();
    state.particleFetchStart = particlesRemoteFetchCount();
    size_t const ownedCount = state.cells.size();
    size_t const workers = std::max(size_t(1), hpx::get_os_thread_count());
    state.tasks.reserve(2 * workers + state.sourceCells.size() - ownedCount);
    try {
        for (size_t worker = 0; worker < workers; ++worker) {
            state.tasks.push_back(hpx::async([worker, workers, ownedCount]() {
                for (size_t index = worker; index < ownedCount; index += workers) {
                    try {
                        auto const& cell = state.cells[index];
                        auto root = treeCreateCell(cell, cache.meshSize, FmmParticleStore::Owned);
                        for (ParticleInteger i = cell.begin; i < cell.end; ++i) {
                            state.stageRungs[i - state.activeRange.first] = particlesRung(i);
                        }
                        publishSource(index, root);
                    } catch (...) {
                        failSource(index, std::current_exception());
                    }
                }
            }));
        }
        // Each receive suspends independently: a delayed cell must not stop
        // requests for other cells that can release ready boundary work.
        for (size_t index = ownedCount; index < state.sourceCells.size(); ++index) {
            state.tasks.push_back(hpx::async(receiveGhostCell, index));
        }
        for (size_t worker = 0; worker < workers; ++worker) {
            state.tasks.push_back(hpx::async(computeReadyCells));
        }
    } catch (...) {
        auto failure = std::current_exception();
        for (size_t index = 0; index < state.sourceCells.size(); ++index) {
            failSource(index, failure);
        }
        throw;
    }
}

void finishPipeline() {
    for (auto& task : state.tasks) {
        if (task.valid()) {
            task.wait();
        }
    }
    for (auto& task : state.tasks) {
        if (!task.valid()) {
            continue;
        }
        try {
            task.get();
        } catch (...) {
            recordFailure(std::current_exception());
        }
    }
    if (state.failure) {
        std::rethrow_exception(state.failure);
    }
    state.diagnostics.nearRemoteTreeFetches = treeRemoteFetchCount() - state.treeFetchStart;
    state.diagnostics.nearRemoteParticleFetches =
        particlesRemoteFetchCount() - state.particleFetchStart;
    ++state.diagnostics.solveCount;
}

void abortPipeline() {
    auto failure = std::make_exception_ptr(
        std::runtime_error("PMMM solve was cancelled after a phase failure"));
    recordFailure(failure);
    for (size_t index = 0; index < state.sourceCells.size(); ++index) {
        failSource(index, failure);
    }
    if (!state.farReady.is_ready()) {
        state.farPromise->set_exception(failure);
    }
}

}    // namespace pmmm::detail

HPX_REGISTER_ACTION(PmmmStartPipelineAction, PmmmStartPipelineAction);
HPX_REGISTER_ACTION(PmmmFinishPipelineAction, PmmmFinishPipelineAction);
HPX_REGISTER_ACTION(PmmmAbortPipelineAction, PmmmAbortPipelineAction);
HPX_REGISTER_ACTION(PmmmClearGhostsAction, PmmmClearGhostsAction);
