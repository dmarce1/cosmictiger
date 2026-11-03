/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cstddef>
#include <span>

// The coarse solve set of the lowest occupied hierarchical rung and its finer
// subset, with the GADGET-4 time-bin merging decision for that rung.
struct RungMergeDecision {
    size_t activeCount{};
    size_t fineCount{};
    bool merge{};
};

// GADGET-4 (Springel et al. 2021, section 4.2, eqs. 51-52) counts N_k, the
// particles on timestep bin k or finer, so N_0 is the whole coarse solve set.
// A hierarchical solve over a set costs its particle count. Per coarse step the
// set on rung n + 1 is solved three times: twice at the coarse synchronization
// (closing and opening sweeps on either side of the top kick) and once at the
// middle one. Rung n's set is solved once; deeper sets cost the same either way.
// Hierarchy N_0 + 3 N_1 against merged 2 N_0 favours merging rung n into rung
// n + 1 when N_1 > N_0 / 3, whether or not n + 1 is the finest rung. The
// fraction is the threshold N_1 / N_0 above which the rung is merged.
inline RungMergeDecision rungMergeDecision(
    std::span<size_t const> rungCounts, int lowestRung, double fraction) {
    RungMergeDecision decision;
    for (size_t rung = static_cast<size_t>(lowestRung); rung < rungCounts.size(); ++rung) {
        decision.activeCount += rungCounts[rung];
        if (rung > static_cast<size_t>(lowestRung)) {
            decision.fineCount += rungCounts[rung];
        }
    }
    decision.merge = decision.fineCount > 0 &&
        static_cast<double>(decision.fineCount) >
            fraction * static_cast<double>(decision.activeCount);
    return decision;
}
