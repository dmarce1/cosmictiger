/* CosmicTiger - shared CPU/CUDA all-neighbor rung regressions; BSL-1.0. */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include <cosmictiger/sph/run_data.hpp>

namespace detail {

inline void rungLimiterRequire(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Fixture, typename RunPass>
void testInactiveRungCandidateClosure(Fixture& fixture, RunPass&& runPass) {
    constexpr int kCount = 7;
    auto& data = fixture.data_;
    fixture.enableRungCandidates();
    fixture.params_.minimumRung = fixture.params_.maximumRung = 7;
    std::array<int, kCount> previous{7, 0, 0, 0, 0, 0, 0};
    for (int index = 0; index < kCount; ++index) {
        int const source = fixture.sourceIndex(index);
        int const sink = fixture.sinkIndex(index);
        int const particle = fixture.particleIndex(index);
        double const position = 0.9 + 0.1 * index;
        data.sourcePositionX[source] = position - std::floor(position);
        data.sourcePositionY[source] = data.sourcePositionZ[source] = 0.5;
        data.sourceSupportRadius[source] = data.sinkSupportRadius[sink] = index % 2 ? 0.11f : 0.06f;
        data.sourceRungs[source] = data.sinkParticleRungs[particle] = previous[index];
        data.sinkRungCandidates[particle] = previous[index];
    }
    // This periodic chain has only nearest-neighbor edges. Alternating support
    // sizes make each edge belong to just one endpoint's kernel. Six initially
    // inactive targets need six immutable-source passes before a final no-op pass.
    int changedPasses = 0;
    bool converged = false;
    for (int pass = 0; pass < kCount + 1; ++pass) {
        auto expected = previous;
        for (int index = 0; index < kCount; ++index) {
            if (index > 0) {
                expected[index] = std::max(expected[index], previous[index - 1] - 1);
            }
            if (index + 1 < kCount) {
                expected[index] = std::max(expected[index], previous[index + 1] - 1);
            }
        }
        bool const changed = runPass();
        rungLimiterRequire(changed == (expected != previous),
            "candidate closure reported the wrong global repeat condition");
        for (int index = 0; index < kCount; ++index) {
            int const source = fixture.sourceIndex(index);
            int const particle = fixture.particleIndex(index);
            rungLimiterRequire(data.sourceRungs[source] == previous[index],
                "candidate closure changed its frozen source snapshot");
            rungLimiterRequire(data.sinkParticleRungs[particle] == (index == 0 ? 7 : 0),
                "neighbor closure committed an inactive live rung before scheduling");
            rungLimiterRequire(data.sinkRungCandidates[particle] == expected[index],
                "candidate closure missed an inactive endpoint or advanced multiple edges");
        }
        if (!changed) {
            converged = true;
            break;
        }
        ++changedPasses;
        previous = expected;
        for (int index = 0; index < kCount; ++index) {
            data.sourceRungs[fixture.sourceIndex(index)] =
                data.sinkRungCandidates[fixture.particleIndex(index)];
        }
    }
    rungLimiterRequire(converged && changedPasses == 6,
        "inactive periodic chain did not require six passes and reach its fixed point");
    for (int index = 0; index < kCount; ++index) {
        rungLimiterRequire(data.sinkRungCandidates[fixture.particleIndex(index)] == 7 - index,
            "touching candidate rungs differ by more than one at closure");
    }
}

template <typename Fixture, typename RunPass>
void testSupportBoundaryRungCandidates(Fixture& fixture, RunPass&& runPass) {
    auto& data = fixture.data_;
    fixture.enableRungCandidates();
    fixture.params_.minimumRung = fixture.params_.maximumRung = 5;
    for (bool largeTargetSupport : {false, true}) {
        for (double separation : {0.125, 0.140625}) {
            for (int index = 0; index < 2; ++index) {
                int const source = fixture.sourceIndex(index);
                int const sink = fixture.sinkIndex(index);
                int const particle = fixture.particleIndex(index);
                data.sourcePositionX[source] = 0.125 + index * separation;
                data.sourcePositionY[source] = data.sourcePositionZ[source] = 0.5;
                data.sourceSupportRadius[source] = data.sinkSupportRadius[sink] =
                    ((index == 0) == largeTargetSupport) ? 0.125f : 0.03125f;
                data.sourceRungs[source] = data.sinkParticleRungs[particle] = index ? 5 : 0;
                data.sinkRungCandidates[particle] = index ? 5 : 0;
            }
            bool const touching = separation == 0.125;
            rungLimiterRequire(runPass() == touching,
                "rung limiter misclassified exact support contact or a separated pair");
            rungLimiterRequire(
                data.sinkRungCandidates[fixture.particleIndex(0)] == (touching ? 4 : 0),
                "either-kernel support contact did not constrain the inactive endpoint");
            rungLimiterRequire(data.sinkParticleRungs[fixture.particleIndex(0)] == 0 &&
                    data.sourceRungs[fixture.sourceIndex(0)] == 0,
                "support contact committed a live rung or changed a staged source");
        }
    }
}

}    // namespace detail
