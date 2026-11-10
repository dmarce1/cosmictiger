/* CosmicTiger - Nitadori PMMM cell schedule; BSL-1.0. */

#include <hpx/threading_base/annotated_function.hpp>
#include <hpx/threading_base/scoped_annotation.hpp>

#include <cosmictiger/gravity/fmm/tree_local.hpp>
#include "../actions.hpp"

namespace pmmm::detail {

// Reuse the near batch's immutable, sorted selection. In particular, do not
// invoke selectBatchSinks again: mixed leaves have already reserved sink-only
// nodes, and another selection would duplicate them. This descent only reads
// the original owned source tree and keeps one order-sized local per depth.
template <int order>
void descendFar(TreeId id,
    cosmictiger::gravity::cartesian::Expansion<float, order> const& expansion, float curvature,
    FmmForceResult const& near, size_t& compact, KickReturn& result, FlopCount& flops,
    FlopCount& singleFlops, FlopCounts& kickFlops) {
    namespace local = cosmictiger::gravity::tree_local;
    auto const* node = treeGetNode(id);
    if (compact == near.particleIndices.size() ||
        near.particleIndices[compact] >= node->particleRange.second) {
        return;
    }
    ALWAYS_ASSERT(id.proc == hpxRank() && node->sourceStore == FmmParticleStore::Owned);
    if (near.particleIndices[compact] < node->particleRange.first) {
        throw std::runtime_error("PMMM far descent lost its selected sink mapping");
    }
    if (!node->leaf) {
        for (auto const child : node->children) {
            auto const* childNode = treeGetNode(child);
            if (compact == near.particleIndices.size() ||
                near.particleIndices[compact] >= childNode->particleRange.second) {
                continue;
            }
            cosmictiger::gravity::cartesian::Expansion<float, order> translated{};
            singleFlops += local::translate(translated, expansion,
                               local::displacement<float>(childNode->pos, node->pos), curvature,
                               state.request.doPotential) +
                6;
            descendFar(
                child, translated, curvature, near, compact, result, flops, singleFlops, kickFlops);
        }
        return;
    }
    for (; compact < near.particleIndices.size() &&
        near.particleIndices[compact] < node->particleRange.second;
        ++compact) {
        ParticleInteger const index = near.particleIndices[compact];
        Vector3<Position> position;
        for (int axis = 0; axis < kDimensions; ++axis) {
            position[axis] = particlesPos(axis, index);
        }
        cosmictiger::gravity::Field<float> far{};
        singleFlops +=
            local::evaluate(far, expansion, local::displacement<float>(position, node->pos),
                curvature, state.request.doPotential) +
            6;
        std::array<double, 4> field{};
        if (state.request.doPotential) {
            field[0] = double(far[0]) + near.potential[compact];
        }
        for (int axis = 0; axis < kDimensions; ++axis) {
            field[axis + 1] = double(far[axis + 1]) + near.acceleration[axis][compact];
        }
        flops += 3 * (kDimensions + int(state.request.doPotential));
        consumeField(index, field, result, flops, kickFlops);
    }
}

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
    state.ownedTasks.clear();
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
            int const minimumRung = state.request.applyKick ? state.request.parameters.minRung : 0;
            auto fields =
                kickCellBatch(std::move(workitems), state.request.doPotential, minimumRung);
            bool const overlapped = state.pendingGhosts.load() != 0;
            for (size_t sink : sinks) {
                observe(PmmmCellEvent::NearCompleted, state.cells[sink].key);
            }
            state.farReady.get();
            double const h = 1.0 / cache.meshSize;
            double const localTrace = (4 * std::numbers::pi / 3) * h * h * state.background[0];
            double const curvature =
                cosmictiger::gravity::tree_local::backgroundCurvature(localTrace, cache.meshSize);
            float const treeCurvature = curvature;
            // Width conversion/division (2), trace products (3), and curvature
            // products/division plus two mesh-size conversions (6).
            FlopCount flops = 12;    // Includes narrowing the solve's curvature once per batch.
            FlopCount singleFlops = 0;
            FlopCounts kickFlops;
            KickReturn result;
            size_t compact = 0;
            for (size_t sink : sinks) {
                auto const& cell = state.cells[sink];
                if (compact == fields.particleIndices.size() ||
                    fields.particleIndices[compact] >= cell.end) {
                    continue;
                }
                auto const center = coordinates(cell.key);
                auto const coefficients = localExpansion(cell.key);
                hpx::scoped_annotation farWork("gravity.pmmm.cartesianFarDescent");
                Vector3<Position> meshCenter;
                Vector3<double> centerCorrection;
                for (int axis = 0; axis < kDimensions; ++axis) {
                    double const coordinate = (center[axis] + .5) * h;
                    meshCenter[axis] = coordinate;
                    centerCorrection[axis] = meshCenter[axis].toDouble() - coordinate;
                }
                // Cell-index promotion/add/product and native fixed conversion;
                // the host Fixed64 conversion separates two safe 32-bit limbs.
                flops += kDimensions * (kPositionBits == 64 ? 13 : 8);
                auto const root = state.sourceCells[sink]->ready.get();
                fmmDispatch([&](auto tag) {
                    constexpr int kOrder = decltype(tag)::value;
                    cosmictiger::gravity::cartesian::Expansion<double, kOrder> local;
                    flops += cosmictiger::gravity::tree_local::fromMesh(
                        local, coefficients.data(), localTrace, cache.meshSize);
                    auto const rootPosition = treeGetNode(root.id)->pos;
                    cosmictiger::gravity::cartesian::Expansion<double, kOrder> atRoot{};
                    Vector3<double> displacement;
                    for (int axis = 0; axis < kDimensions; ++axis) {
                        displacement[axis] = double(kScaleFactor) *
                            (unwrappedDoubleDistance(rootPosition[axis], meshCenter[axis]) +
                                centerCorrection[axis]);
                    }
                    flops += cosmictiger::gravity::tree_local::translate(atRoot, local,
                                 displacement, curvature, state.request.doPotential) +
                        12;
                    cosmictiger::gravity::cartesian::Expansion<float, kOrder> treeLocal;
                    for (int coefficient = 0; coefficient < treeLocal.size(); ++coefficient) {
                        treeLocal[coefficient] = atRoot[coefficient];
                    }
                    flops += treeLocal.size();
                    observe(PmmmCellEvent::FarLocalReady, cell.key);
                    descendFar(root.id, treeLocal, treeCurvature, fields, compact, result, flops,
                        singleFlops, kickFlops);
                });
            }
            if (compact != fields.particleIndices.size()) {
                throw std::runtime_error("PMMM batch output contains an unknown sink");
            }
            addCpuFlops(flops + kickFlops.doublePrecision, FlopPrecision::Double);
            addCpuFlops(singleFlops + kickFlops.singlePrecision, FlopPrecision::Single);
            {
                std::lock_guard<MutexType> lock(state.queueMutex);
                state.kickResult += result;
                state.diagnostics.evaluatedParticles += fields.particleIndices.size();
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
    state.tasks.reserve(workers + state.sourceCells.size() - ownedCount);
    state.ownedTasks.reserve(workers);
    try {
        for (size_t worker = 0; worker < workers; ++worker) {
            state.ownedTasks.push_back(hpx::async(
                hpx::annotated_function(
                    [worker, workers, ownedCount]() {
                        // Retain at most 32 converted cells per producer; the double
                        // Cartesian root is released after each conversion. Slab
                        // writes are acknowledged before the next packet is built.
                        constexpr size_t kPacketCells = 32;
                        for (size_t begin = worker; begin < ownedCount;
                            begin += workers * kPacketCells) {
                            std::vector<CellData> packet;
                            for (size_t index = begin;
                                index < std::min(ownedCount, begin + workers * kPacketCells);
                                index += workers) {
                                try {
                                    auto const& cell = state.cells[index];
                                    TreeCreateReturn completed;
                                    auto const root = treeCreateCell(cell, cache.meshSize,
                                        FmmParticleStore::Owned, nullptr, &completed);
                                    if (reuseOwnedRoots()) {
                                        packet.push_back(rootMoments(index, completed));
                                    } else {
                                        for (ParticleInteger i = cell.begin; i < cell.end; ++i) {
                                            state.stageRungs[i - state.activeRange.first] =
                                                particlesRung(i);
                                        }
                                    }
                                    observe(PmmmCellEvent::OwnedRootReady, cell.key);
                                    publishSource(index, root);
                                } catch (...) {
                                    failSource(index, std::current_exception());
                                }
                            }
                            if (!packet.empty()) {
                                try {
                                    depositCells(std::move(packet));
                                } catch (...) {
                                    recordFailure(std::current_exception());
                                }
                            }
                        }
                    },
                    "gravity.pmmm.buildOwnedSourceCells"))
                    .share());
        }
        // Each receive suspends independently: a delayed cell must not stop
        // requests for other cells that can release ready boundary work.
        for (size_t index = ownedCount; index < state.sourceCells.size(); ++index) {
            state.tasks.push_back(hpx::async(
                hpx::annotated_function(receiveGhostCell, "gravity.pmmm.receiveGhostCell"), index));
        }
        for (size_t worker = 0; worker < workers; ++worker) {
            state.tasks.push_back(hpx::async(
                hpx::annotated_function(computeReadyCells, "gravity.pmmm.computeReadyCells")));
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
    for (auto& task : state.ownedTasks) {
        try {
            task.get();
        } catch (...) {
            recordFailure(std::current_exception());
        }
    }
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
