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
// subset, with the inclusive time-bin merging decision for that rung.
struct RungMergeDecision {
    size_t activeCount{};
    size_t fineCount{};
    bool merge{};
};

/*
 * GADGET-4 (Springel et al. 2021, §4.2, equation 52):
 *   K_hierarchical = N₀ + 3 ∑_{k≥1} 2ᵏ⁻¹ Nₖ.
 * https://wwwmpa.mpa-garching.mpg.de/gadget4/gadget4-code-paper.pdf
 * Nₖ counts the particles on rung n+k or finer, not only that rung.
 *
 * Derived comparison after cancelling the unchanged deeper solves:
 *   C_hierarchy = Nₙ + 3 Nₙ₊₁, C_merged = 2 Nₙ.
 *   C_merged ≤ C_hierarchy ⇔ Nₙ ≤ 3 Nₙ₊₁.
 * The paper uses strict cost saving. The selected policy also promotes on an
 * equal-cost tie. An empty finer set never triggers promotion.
 *
 * Symbol directory:
 *   Nₙ -> activeCount; Nₙ₊₁ -> fineCount;
 *   configurable threshold Nₙ₊₁/Nₙ -> fraction (default 1/3).
 */
inline RungMergeDecision rungMergeDecision(
    std::span<size_t const> rungCounts, int lowestRung, double fraction) {
    RungMergeDecision decision;
    for (size_t rung = static_cast<size_t>(lowestRung); rung < rungCounts.size(); ++rung) {
        decision.activeCount += rungCounts[rung];
        if (rung > static_cast<size_t>(lowestRung)) {
            decision.fineCount += rungCounts[rung];
        }
    }
    if (fraction == 1.0 / 3.0) {
        // fineCount ≥ ceil(activeCount / 3), without multiplying counts or
        // rounding a large population through double precision.
        size_t const minimumFineCount =
            decision.activeCount / 3 + size_t(decision.activeCount % 3 != 0);
        decision.merge = decision.fineCount > 0 && decision.fineCount >= minimumFineCount;
    } else if (fraction == 1.0) {
        decision.merge = decision.fineCount > 0 && decision.fineCount == decision.activeCount;
    } else {
        decision.merge = decision.fineCount > 0 &&
            static_cast<double>(decision.fineCount) >=
                fraction * static_cast<double>(decision.activeCount);
    }
    return decision;
}
