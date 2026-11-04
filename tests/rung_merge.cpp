// Copyright (C) 2026 Dominic C. Marcello
// Distributed under the Boost Software License, Version 1.0.
// See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.

#include <cstddef>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include <cosmictiger/core/rung_merge.hpp>

namespace {

constexpr double kDefaultFraction = 1.0 / 3.0;

bool merges(std::vector<size_t> const& counts, int lowestRung) {
    return rungMergeDecision(counts, lowestRung, kDefaultFraction).merge;
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

TEST(RungMerge, FinestPairIncludesTheEqualCostBoundary) {
    EXPECT_TRUE(merges({0, 200, 101}, 1));
    // Nn = 300 and Nnp1 = 100: both schedules cost 600. Promote on this tie.
    EXPECT_TRUE(merges({0, 200, 100}, 1));
    EXPECT_FALSE(merges({0, 200, 99}, 1));
    // An odd/nondivisible population rounds its required finer count upward.
    EXPECT_FALSE(merges({201, 100}, 0));
    EXPECT_TRUE(merges({201, 101}, 0));
}

TEST(RungMerge, FinerRungsCountTogetherWhetherOrNotTheNextRungIsFinest) {
    EXPECT_TRUE(merges({200, 50, 30, 21}, 0));
    EXPECT_TRUE(merges({200, 50, 30, 20}, 0));
    EXPECT_FALSE(merges({200, 50, 30, 19}, 0));
    EXPECT_TRUE(merges({200, 0, 0, 100}, 0));
    EXPECT_FALSE(merges({200, 0, 0, 99}, 0));
}

TEST(RungMerge, LowestBinAloneIsNotTheThreshold) {
    auto const decision = rungMergeDecision(std::vector<size_t>{300, 76}, 0, kDefaultFraction);
    EXPECT_EQ(decision.activeCount, 376u);
    EXPECT_EQ(decision.fineCount, 76u);
    EXPECT_FALSE(decision.merge);
    // Counts below the selected floor never enter its recursive solve sets.
    auto const offset =
        rungMergeDecision(std::vector<size_t>{10000, 200, 100}, 1, kDefaultFraction);
    EXPECT_EQ(offset.activeCount, 300u);
    EXPECT_EQ(offset.fineCount, 100u);
    EXPECT_TRUE(offset.merge);
}

TEST(RungMerge, ConfigurableFractionIsInclusiveAndRequiresFinerParticles) {
    EXPECT_TRUE(rungMergeDecision(std::vector<size_t>{0, 0, 10}, 0, 1.0).merge);
    EXPECT_FALSE(rungMergeDecision(std::vector<size_t>{1, 0, 10}, 0, 1.0).merge);
    EXPECT_TRUE(rungMergeDecision(std::vector<size_t>{1000, 1}, 0, 0.0).merge);
    EXPECT_TRUE(rungMergeDecision(std::vector<size_t>{100, 100}, 0, 0.5).merge);
    EXPECT_FALSE(rungMergeDecision(std::vector<size_t>{101, 100}, 0, 0.5).merge);
    EXPECT_FALSE(rungMergeDecision(std::vector<size_t>{10}, 0, 0.0).merge);
    EXPECT_FALSE(rungMergeDecision(std::vector<size_t>{5}, 3, 0.0).merge);
    EXPECT_FALSE(merges({}, 0));
    EXPECT_FALSE(merges({0, 0, 0}, 0));
    EXPECT_TRUE(merges({0, 5, 5}, 0));
}

TEST(RungMerge, DefaultComparisonDoesNotOverflowOrRoundLargeCounts) {
    size_t const population = std::numeric_limits<size_t>::max();
    size_t const threshold = population / 3 + size_t(population % 3 != 0);
    EXPECT_TRUE(merges({population - threshold, threshold}, 0));
    EXPECT_FALSE(merges({population - threshold + 1, threshold - 1}, 0));
    // Multiplying this finer count by three would overflow size_t.
    size_t const largerFineSet = population / 2 + 1;
    EXPECT_TRUE(merges({population - largerFineSet, largerFineSet}, 0));
    EXPECT_FALSE(rungMergeDecision(std::vector<size_t>{1, population - 1}, 0, 1.0).merge);
    EXPECT_TRUE(rungMergeDecision(std::vector<size_t>{0, population}, 0, 1.0).merge);
    // Adjacent large integer populations can map to the same double value.
    size_t const adjacentPopulation = population / 2 + 2;
    size_t const adjacentThreshold = adjacentPopulation / 3 + size_t(adjacentPopulation % 3 != 0);
    EXPECT_TRUE(merges({adjacentPopulation - adjacentThreshold, adjacentThreshold}, 0));
    EXPECT_FALSE(merges({adjacentPopulation - adjacentThreshold + 1, adjacentThreshold - 1}, 0));
}

TEST(RungMerge, DecisionMatchesTheLadderSolveCostIncludingTies) {
    // Merging replaces one step of rung n with two of rung n+1 over the same
    // interval. Equal work promotes too, but an empty finer set never does.
    int cases = 0;
    for (size_t coarse = 0; coarse <= 12; ++coarse) {
        for (size_t next = 0; next <= 8; ++next) {
            for (size_t deeper = 0; deeper <= 6; ++deeper) {
                for (size_t deepest = 0; deepest <= 4; ++deepest) {
                    std::vector<size_t> const counts{coarse, next, deeper, deepest};
                    std::vector<size_t> const merged{0, coarse + next, deeper, deepest};
                    size_t const hierarchy = ladderCost(counts, 0);
                    size_t const promoted = 2 * ladderCost(merged, 1);
                    size_t const fineCount = solveSet(counts, 1);
                    EXPECT_EQ(merges(counts, 0), fineCount > 0 && promoted <= hierarchy)
                        << coarse << " " << next << " " << deeper << " " << deepest;
                    EXPECT_EQ(hierarchy + solveSet(counts, 0), promoted + 3 * fineCount);
                    ++cases;
                }
            }
        }
    }
    EXPECT_EQ(cases, 13 * 9 * 7 * 5);
}
