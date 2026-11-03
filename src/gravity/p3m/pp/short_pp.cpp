/* CosmicTiger - P3M direct short-range pairs (PP-PM); BSL-1.0. */

#include <algorithm>
#include <limits>
#include <stdexcept>

#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/p3m/pp/short_pp.hpp>
#include <cosmictiger/gravity/pm/geometry.hpp>
#include <cosmictiger/math/kernel.hpp>

#include "../state.hpp"

namespace cosmictiger::gravity::p3m::pp {

bool shortFieldBatch(std::vector<ShortParticle> const& sinks,
    std::vector<ShortParticle> const& sources, std::vector<ShortSourceRange> const& sourceRanges,
    ShortPairParameters const& parameters, bool doPotential,
    std::vector<std::array<double, 4>>& fields) {
    if (sourceRanges.size() != sinks.size()) {
        throw std::invalid_argument("P3M short batch requires one source range per sink");
    }
    size_t candidatePairs = 0;
    [[maybe_unused]] size_t usefulSinkCount = 0;
    for (auto const& range : sourceRanges) {
        if (range.begin > range.end || range.end > sources.size()) {
            throw std::invalid_argument("P3M short source range is outside the source array");
        }
        size_t const count = range.end - range.begin;
        usefulSinkCount += count != 0;
        candidatePairs += std::min(count, std::numeric_limits<size_t>::max() - candidatePairs);
    }
    fields.assign(sinks.size(), {});
    if (candidatePairs == 0) {
        return false;
    }
#ifdef USE_CUDA
    if (getOptions().gravity.cuda && shouldUseShortCuda(usefulSinkCount, candidatePairs)) {
        shortFieldCuda(sinks, sources, sourceRanges, parameters, doPotential, fields);
        return true;
    }
#endif
    auto const& kernel = selectedGravityKernel();
    FlopCounter<> flops = 0;
    for (size_t sink = 0; sink < sinks.size(); ++sink) {
        auto const& range = sourceRanges[sink];
        for (size_t source = range.begin; source < range.end; ++source) {
            flops += accumulateShortPair(
                sinks[sink], sources[source], parameters, doPotential, kernel, fields[sink]);
        }
    }
    addCpuFlops(flops, FlopPrecision::Double);
    return false;
}

}    // namespace cosmictiger::gravity::p3m::pp

namespace p3m::detail {

using namespace cosmictiger::gravity::p3m;

void evaluatePairs() {
    auto const& gravity = getOptions().gravity;
    int const n = state.meshSize;
    int const width = gravity.pmmmNearWidth;
    auto const pairParameters = makeShortPairParameters(
        gravity.p3mSplitRadius, gravity.p3mCutoffRadius, gravity.supportRadius);
    bool const doPotential = state.request.doPotential;
    size_t const targetParticles = gravity.pmmmBatchParticles;
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
    parallelFor(batches.size(), "gravity.p3m.evaluatePairs", [&](size_t batchIndex) {
        // A dense neighboring source cell must not be duplicated without a
        // bound across many sparse sinks. An oversized single cell runs whole.
        constexpr size_t kMaximumBatchSources = 64 * 1024 * 1024 / sizeof(ShortParticle);
        std::vector<ShortParticle> sinks;
        std::vector<ShortParticle> batchSources;
        std::vector<ShortSourceRange> sourceRanges;
        FlopCount flops = 0;
        FlopCounts kickFlops;
        uint64_t pairs = 0;
        auto submit = [&]() {
            if (sinks.empty()) {
                return;
            }
            std::vector<std::array<double, 4>> fields;
            bool const device = pp::shortFieldBatch(
                sinks, batchSources, sourceRanges, pairParameters, doPotential, fields);
            KickReturn result;
            for (size_t sink = 0; sink < sinks.size(); ++sink) {
                consumeField(
                    ParticleInteger(sinks[sink].index), fields[sink], result, flops, kickFlops);
                pairs += sourceRanges[sink].end - sourceRanges[sink].begin;
            }
            addCpuFlops(flops + kickFlops.doublePrecision, FlopPrecision::Double);
            addCpuFlops(kickFlops.singlePrecision, FlopPrecision::Single);
            flops = 0;
            kickFlops = {};
            {
                std::lock_guard<MutexType> lock(state.mutex);
                state.kickResult += result;
                state.diagnostics.evaluatedParticles += sinks.size();
                state.diagnostics.directPairs += pairs;
                ++state.diagnostics.batches;
                state.diagnostics.cudaBatches += device ? 1 : 0;
            }
            pairs = 0;
            sinks.clear();
            batchSources.clear();
            sourceRanges.clear();
        };
        auto const [cellBegin, cellEnd] = batches[batchIndex];
        for (size_t cellIndex = cellBegin; cellIndex < cellEnd; ++cellIndex) {
            auto const& cell = state.cells[cellIndex];
            bool hasActive = false;
            for (ParticleInteger i = cell.begin; i < cell.end && !hasActive; ++i) {
                hasActive = activeSink(i);
            }
            if (!hasActive) {
                continue;
            }
            auto const center = coordinates(cell.key);
            std::vector<ShortParticle> sources;
            for (int x = -width; x <= width; ++x) {
                for (int y = -width; y <= width; ++y) {
                    for (int z = -width; z <= width; ++z) {
                        int const raw[3] = {center[0] + x, center[1] + y, center[2] + z};
                        Vector3<double> shift;
                        uint64_t key = 0;
                        for (int axis = 0; axis < kDimensions; ++axis) {
                            int const wrapped = periodicIndex(raw[axis]);
                            shift[axis] = double(raw[axis] - wrapped) / n;
                            key = key * n + wrapped;
                        }
                        auto const found = state.sourceByKey.find(key);
                        if (found == state.sourceByKey.end()) {
                            continue;
                        }
                        for (auto source : state.sourceCells[found->second].particles) {
                            for (int axis = 0; axis < kDimensions; ++axis) {
                                source.position[axis] += shift[axis];
                            }
                            sources.push_back(source);
                        }
                    }
                }
            }
            flops += 3 * sources.size();
            if (!sinks.empty() &&
                sources.size() >
                    kMaximumBatchSources - std::min(kMaximumBatchSources, batchSources.size())) {
                submit();
            }
            ShortSourceRange const range{batchSources.size(), batchSources.size() + sources.size()};
            batchSources.insert(batchSources.end(), sources.begin(), sources.end());
            for (ParticleInteger i = cell.begin; i < cell.end; ++i) {
                if (activeSink(i)) {
                    sinks.push_back(particle(i));
                    sourceRanges.push_back(range);
                }
            }
        }
        submit();
    });
}

}    // namespace p3m::detail
