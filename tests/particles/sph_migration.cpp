/* CosmicTiger - active-suffix gas migration regressions; BSL-1.0. */

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include <cosmictiger/particles/sph_migration.hpp>

TEST(SphMigrationSuffix, VisitsOnlyTheSelectedSuffix) {
    constexpr int kPrefix = 1000000;
    constexpr int kCount = kPrefix + 4;
    std::vector<int> visited;
    auto const plan = detail::planSphMigrationSuffix(kPrefix, kCount, kCount, 10, [&](int index) {
        visited.push_back(index);
        std::array<int, 4> const slots{2, -1, 8, 5};
        return slots.at(index - kPrefix);
    });
    EXPECT_EQ(visited, (std::vector<int>{kPrefix, kPrefix + 1, kPrefix + 2, kPrefix + 3}));
    EXPECT_EQ(plan.begin, kPrefix);
    EXPECT_EQ(plan.end, kCount);
    EXPECT_EQ(plan.gasIndices, (std::vector<int>{8, 5, 2}));
}

TEST(SphMigrationSuffix, DescendingRemovalPreservesInactiveGasIdentities) {
    // Gas storage is independent of the main-particle order: active gas slots
    // are interleaved with inactive gas, including the last slots moved by erase.
    std::array<int, 8> mainGasSlot{0, 4, 2, 6, 7, 1, 5, 3};
    std::vector<int> gasOwner{0, 5, 2, 7, 1, 6, 3, 4};
    auto const plan = detail::planSphMigrationSuffix(
        5, 8, 8, 8, [&](int particle) { return mainGasSlot.at(particle); });
    for (int const slot : plan.gasIndices) {
        ASSERT_LT(slot, int(gasOwner.size()));
        if (slot != int(gasOwner.size()) - 1) {
            gasOwner[slot] = gasOwner.back();
            mainGasSlot[gasOwner[slot]] = slot;
        }
        gasOwner.pop_back();
    }
    ASSERT_EQ(gasOwner.size(), size_t{5});
    for (int particle = 0; particle < 5; ++particle) {
        ASSERT_GE(mainGasSlot[particle], 0);
        ASSERT_LT(mainGasSlot[particle], int(gasOwner.size()));
        EXPECT_EQ(gasOwner[mainGasSlot[particle]], particle);
    }
    std::sort(gasOwner.begin(), gasOwner.end());
    EXPECT_EQ(gasOwner, (std::vector<int>{0, 1, 2, 3, 4}));
}

TEST(SphMigrationSuffix, EmptySuffixDoesNotReadInactiveParticles) {
    int calls = 0;
    auto const plan = detail::planSphMigrationSuffix(1000000, 1000000, 1000000, 100, [&](int) {
        ++calls;
        return 0;
    });
    EXPECT_EQ(calls, 0);
    EXPECT_TRUE(plan.gasIndices.empty());
}

TEST(SphMigrationSuffix, RejectsInvalidMappingsBeforeCompaction) {
    int calls = 0;
    auto const read = [&](int) {
        ++calls;
        return 1;
    };
    EXPECT_THROW(detail::planSphMigrationSuffix(3, 2, 2, 3, read), std::invalid_argument);
    EXPECT_THROW(detail::planSphMigrationSuffix(0, 2, 3, 3, read), std::invalid_argument);
    EXPECT_EQ(calls, 0);
    EXPECT_THROW(detail::planSphMigrationSuffix(1, 2, 2, 1, read), std::invalid_argument);
    EXPECT_THROW(detail::planSphMigrationSuffix(0, 2, 2, 3, read), std::invalid_argument);
}
