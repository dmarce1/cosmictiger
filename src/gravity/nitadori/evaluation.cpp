/* CosmicTiger - Nitadori PMMM evaluation; BSL-1.0. */

#include <cosmictiger/math/vector3.hpp>
#include "actions.hpp"

namespace pmmm::detail {

void consumeField(ParticleInteger index, std::array<double, 4> const& field, KickReturn& result,
    FlopCount& doubleFlops, FlopCounts& kickFlops) {
    Vector3<float> acceleration;
    double const gm = getOptions().gravity.gm;
    for (int component = 0; component < 4; ++component) {
        if (!std::isfinite(field[component]) ||
            std::abs(gm * field[component]) > std::numeric_limits<float>::max()) {
            throw std::runtime_error("PMMM combined particle field is not representable");
        }
        if (component != 0) {
            acceleration[component - 1] = gm * field[component];
        }
    }
    float const potential = state.request.doPotential ? gm * field[0] : 0.0f;
    // Each retained component is scaled in double and narrowed to float.
    doubleFlops += 2 * (kDimensions + (state.request.doPotential ? 1 : 0));
    if (state.request.applyKick) {
        result += particleMeshApplyField(
            index, acceleration, potential, state.request.parameters, nullptr, &kickFlops);
    } else {
        state.acceleration[index] = acceleration;
        if (!state.potential.empty()) {
            state.potential[index] = potential;
        }
    }
}

#if COSMICTIGER_ENABLE_PMMM_PP
void evaluateParticles(bool doPotential) {
    int const n = cache.meshSize;
    int const width = cache.nearWidth;
    double const h = 1.0 / n;
    double const localTrace = (4 * std::numbers::pi / 3) * h * h * state.background[0];
    addCpuFlops(4, FlopPrecision::Double);
    // Bundle whole cells by sink count. Tiny cells share one backend submission;
    // each sink still traverses only its own cell's shifted neighbor sources.
    size_t const targetParticles = getOptions().gravity.pmmmBatchParticles;
    std::vector<std::pair<size_t, size_t>> batches;
    size_t begin = 0;
    size_t particleCount = 0;
    for (size_t index = 0; index < state.cells.size(); ++index) {
        auto const& cell = state.cells[index];
        size_t const count = cell.end - cell.begin;
        if (index != begin && count > targetParticles - std::min(targetParticles, particleCount)) {
            batches.emplace_back(begin, index);
            begin = index;
            particleCount = 0;
        }
        particleCount += count;
    }
    if (begin != state.cells.size()) {
        batches.emplace_back(begin, state.cells.size());
    }
    parallelFor(batches.size(), "gravity.pmmm.evaluateParticles", [&](size_t batchIndex) {
        struct BatchCell {
            size_t sinkBegin;
            std::vector<double> coefficients;
        };
        // A dense neighboring source cell must not be duplicated without a
        // bound across many sparse sinks. An oversized single cell runs whole.
        constexpr size_t kMaximumBatchSources = 64 * 1024 * 1024 / sizeof(PmmmNearParticle);
        std::vector<BatchCell> cells;
        std::vector<PmmmNearParticle> sinks;
        std::vector<PmmmNearParticle> batchSources;
        std::vector<PmmmNearSourceRange> sourceRanges;
        FlopCount flops = 0;
        FlopCounts kickFlops;
        auto submit = [&]() {
            if (cells.empty()) {
                return;
            }
            std::vector<std::array<double, 4>> nearFields;
            pmmmNearFieldBatch(sinks, batchSources, sourceRanges, h, doPotential, nearFields);
            KickReturn result;
            for (size_t cellIndex = 0; cellIndex < cells.size(); ++cellIndex) {
                auto const& cell = cells[cellIndex];
                size_t const end =
                    cellIndex + 1 < cells.size() ? cells[cellIndex + 1].sinkBegin : sinks.size();
                for (size_t sink = cell.sinkBegin; sink < end; ++sink) {
                    std::array<double, 4> field{};
                    flops += evaluate(cache.order, field, cell.coefficients.data(), localTrace,
                        sinks[sink].offset, h);
                    auto const index = ParticleInteger(sinks[sink].index);
                    for (int component = 0; component < 4; ++component) {
                        field[component] += nearFields[sink][component];
                    }
                    flops += 4;
                    consumeField(index, field, result, flops, kickFlops);
                }
            }
            addCpuFlops(flops + kickFlops.doublePrecision, FlopPrecision::Double);
            addCpuFlops(kickFlops.singlePrecision, FlopPrecision::Single);
            flops = 0;
            kickFlops = {};
            {
                std::lock_guard<MutexType> lock(state.queueMutex);
                state.kickResult += result;
                state.diagnostics.evaluatedParticles += sinks.size();
                ++state.diagnostics.nearBatches;
                state.diagnostics.peakBatchParticles =
                    std::max<uint64_t>(state.diagnostics.peakBatchParticles, sinks.size());
            }
            cells.clear();
            sinks.clear();
            batchSources.clear();
            sourceRanges.clear();
        };
        auto const [cellBegin, cellEnd] = batches[batchIndex];
        for (size_t cellIndex = cellBegin; cellIndex < cellEnd; ++cellIndex) {
            auto const& cell = state.cells[cellIndex];
            auto active = [&](ParticleInteger index) {
                return !state.request.applyKick ||
                    particlesRung(index) >= state.request.parameters.minRung;
            };
            bool hasActive = false;
            for (ParticleInteger i = cell.begin; i < cell.end; ++i) {
                hasActive = hasActive || active(i);
            }
            if (!hasActive) {
                continue;
            }
            auto const center = coordinates(cell.key);
            // The slab owner serves the sink cell's local expansion; each
            // neighbor cell's owner serves that cell's particles directly.
            auto coefficients = localExpansion(cell.key);
            struct Request {
                std::vector<uint64_t> keys;
                hpx::future<std::vector<CellParticles>> future;
            };
            std::unordered_map<int, Request> requests;
            std::unordered_map<uint64_t, std::array<int, 3>> displacements;
            for (int x = -width; x <= width; ++x) {
                for (int y = -width; y <= width; ++y) {
                    for (int z = -width; z <= width; ++z) {
                        int const sx = periodicIndex(center[0] - x);
                        int const sy = periodicIndex(center[1] - y);
                        int const sz = periodicIndex(center[2] - z);
                        uint64_t const key = (uint64_t(sx) * n + sy) * n + sz;
                        requests[cellOwner(key)].keys.push_back(key);
                        displacements.emplace(key, std::array<int, 3>{x, y, z});
                    }
                }
            }
            std::exception_ptr failure;
            try {
                for (auto& [owner, request] : requests) {
                    request.future = owner == hpxRank() ?
                        hpx::make_ready_future(
                            readCellParticles(state.request.generation, request.keys)) :
                        hpx::async<PmmmReadCellParticlesAction>(
                            hpxLocalities()[owner], state.request.generation, request.keys);
                }
            } catch (...) {
                failure = std::current_exception();
            }
            // Drain remote reads even if one launch or unpacking operation fails.
            for (auto& [owner, request] : requests) {
                if (request.future.valid()) {
                    request.future.wait();
                }
            }
            if (failure) {
                std::rethrow_exception(failure);
            }
            std::vector<PmmmNearParticle> sources;
            for (auto& [owner, request] : requests) {
                for (auto& result : request.future.get()) {
                    auto const displacement = displacements.at(result.key);
                    for (auto source : result.particles) {
                        for (int axis = 0; axis < kDimensions; ++axis) {
                            source.offset[axis] -= displacement[axis];
                        }
                        sources.push_back(source);
                        flops += kDimensions;
                    }
                }
            }
            if (coefficients.size() != size_t(cache.localCount)) {
                throw std::runtime_error("PMMM local expansion is missing");
            }
            if (!cells.empty() &&
                sources.size() >
                    kMaximumBatchSources - std::min(kMaximumBatchSources, batchSources.size())) {
                submit();
            }
            PmmmNearSourceRange const sourceRange{
                batchSources.size(), batchSources.size() + sources.size()};
            batchSources.insert(batchSources.end(), sources.begin(), sources.end());
            cells.push_back({sinks.size(), std::move(coefficients)});
            for (ParticleInteger i = cell.begin; i < cell.end; ++i) {
                if (active(i)) {
                    sinks.push_back(particle(i, center, flops));
                    sourceRanges.push_back(sourceRange);
                }
            }
        }
        submit();
    });
    ++state.diagnostics.solveCount;
}

#endif

}    // namespace pmmm::detail

#if COSMICTIGER_ENABLE_PMMM_PP
HPX_REGISTER_ACTION(PmmmEvaluateAction, PmmmEvaluateAction);
#endif
