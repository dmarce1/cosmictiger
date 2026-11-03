// Copyright (C) 2026 Dominic C. Marcello
// Distributed under the Boost Software License, Version 1.0.
// See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <cosmictiger/core/rung_merge.hpp>

namespace {

constexpr double kGadgetFraction = 1.0 / 3.0;

bool merges(std::vector<size_t> const& counts, int lowestRung) {
    return rungMergeDecision(counts, lowestRung, kGadgetFraction).merge;
}

// Particles in the solve set of rung `level`: every particle on it or finer.
size_t solveSet(std::vector<size_t> const& counts, int level) {
    size_t total = 0;
    for (size_t rung = static_cast<size_t>(level); rung < counts.size(); ++rung) {
        total += counts[rung];
    }
    return total;
}

// Gravity solve work of kickStepHierarchical over one step of rung `lowestRung`.
// A synchronization whose lowest rung is k solves rung k's set once (the top
// kick) and every finer set twice (the closing ascent and the opening descent).
size_t ladderCost(std::vector<size_t> const& counts, int lowestRung) {
    int const finestRung = static_cast<int>(counts.size()) - 1;
    size_t cost = 0;
    for (size_t sync = 0; sync < (size_t{1} << (finestRung - lowestRung)); ++sync) {
        int syncRung = lowestRung;
        if (sync != 0) {
            syncRung = finestRung;
            for (size_t index = sync; (index & 1) == 0; index >>= 1) {
                --syncRung;
            }
        }
        cost += solveSet(counts, syncRung);
        for (int level = syncRung + 1; level <= finestRung; ++level) {
            cost += 2 * solveSet(counts, level);
        }
    }
    return cost;
}

}    // namespace

TEST(RungMerge, FinestPairMergesOnlyAboveOneThirdOfTheSolveSet) {
    // N0 = 301 and N1 = 101 > N0 / 3.
    EXPECT_TRUE(merges({0, 200, 101}, 1));
    // N1 = N0 / 3 exactly saves nothing; the hierarchy is kept.
    EXPECT_FALSE(merges({0, 200, 100}, 1));
    EXPECT_FALSE(merges({0, 200, 99}, 1));
}

TEST(RungMerge, FinerRungsCountTogetherWhetherOrNotTheNextRungIsFinest) {
    EXPECT_TRUE(merges({200, 50, 30, 21}, 0));
    EXPECT_FALSE(merges({200, 50, 30, 20}, 0));
    EXPECT_TRUE(merges({200, 1, 0, 100}, 0));
    EXPECT_FALSE(merges({200, 0, 0, 100}, 0));
}

TEST(RungMerge, LowestBinAloneIsNotTheThreshold) {
    // The former rule 3 N1 > (N0 - N1) merged here although the hierarchy costs
    // 300 + 3 * 76 = 528 solves per coarse step against 2 * 376 = 752 merged.
    auto const decision = rungMergeDecision(std::vector<size_t>{300, 76}, 0, kGadgetFraction);
    EXPECT_EQ(decision.activeCount, 376u);
    EXPECT_EQ(decision.fineCount, 76u);
    EXPECT_FALSE(decision.merge);
}

TEST(RungMerge, BoundsOfTheFraction) {
    EXPECT_FALSE(rungMergeDecision(std::vector<size_t>{0, 0, 10}, 0, 1.0).merge);
    EXPECT_TRUE(rungMergeDecision(std::vector<size_t>{1000, 1}, 0, 0.0).merge);
    EXPECT_FALSE(rungMergeDecision(std::vector<size_t>{10}, 0, 0.0).merge);
    EXPECT_FALSE(rungMergeDecision(std::vector<size_t>{5}, 3, 0.0).merge);
    // An empty lowest rung is merged into its occupied successor.
    EXPECT_TRUE(merges({0, 5, 5}, 0));
}

TEST(RungMerge, DecisionMatchesTheLadderSolveCost) {
    // Merging rung n into n + 1 replaces one step of rung n with two of rung
    // n + 1 over the same interval; merge exactly when that is cheaper.
    int cases = 0;
    for (size_t coarse = 0; coarse <= 12; ++coarse) {
        for (size_t next = 0; next <= 8; ++next) {
            for (size_t deeper = 0; deeper <= 6; ++deeper) {
                for (size_t deepest = 0; deepest <= 4; ++deepest) {
                    std::vector<size_t> const counts{coarse, next, deeper, deepest};
                    std::vector<size_t> const merged{0, coarse + next, deeper, deepest};
                    size_t const hierarchy = ladderCost(counts, 0);
                    size_t const coarsened = 2 * ladderCost(merged, 1);
                    EXPECT_EQ(merges(counts, 0), coarsened < hierarchy)
                        << coarse << " " << next << " " << deeper << " " << deepest;
                    // The ladder reproduces GADGET-4 eq. (52): N0 + 3 N1 against 2 N0.
                    EXPECT_EQ(hierarchy + solveSet(counts, 0), coarsened + 3 * solveSet(counts, 1));
                    ++cases;
                }
            }
        }
    }
    EXPECT_EQ(cases, 13 * 9 * 7 * 5);
}
