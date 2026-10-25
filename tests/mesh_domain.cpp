#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include <cosmictiger/particles/mesh_domain.hpp>

namespace detail {

template <class Integer>
void checkFixedBoundaries() {
    unsigned __int128 const period = static_cast<unsigned __int128>(1)
        << std::numeric_limits<Integer>::digits;
    for (uint32_t meshSize : {7U, 8U, 31U}) {
        for (uint32_t plane = 1; plane < meshSize; ++plane) {
            Integer const firstRight =
                static_cast<Integer>((plane * period + meshSize - 1) / meshSize);
            Integer const lastLeft = firstRight - 1;
            EXPECT_TRUE(mesh_domain::belowBoundary(lastLeft, plane, meshSize));
            EXPECT_FALSE(mesh_domain::belowBoundary(firstRight, plane, meshSize));
            EXPECT_TRUE(mesh_domain::contains(lastLeft, plane - 1, plane, meshSize));
            EXPECT_FALSE(mesh_domain::contains(firstRight, plane - 1, plane, meshSize));
            EXPECT_TRUE(mesh_domain::contains(firstRight, plane, plane + 1, meshSize));
        }
        Integer const last = std::numeric_limits<Integer>::max();
        EXPECT_FALSE(mesh_domain::belowBoundary(Integer(0), 0, meshSize));
        EXPECT_FALSE(mesh_domain::belowBoundary(last, 0, meshSize));
        EXPECT_TRUE(mesh_domain::belowBoundary(Integer(0), meshSize, meshSize));
        EXPECT_TRUE(mesh_domain::belowBoundary(last, meshSize, meshSize));
        EXPECT_TRUE(mesh_domain::contains(Integer(0), 0, 1, meshSize));
        EXPECT_TRUE(mesh_domain::contains(last, meshSize - 1, meshSize, meshSize));
        EXPECT_FALSE(mesh_domain::contains(last, meshSize, meshSize, meshSize));
        EXPECT_FALSE(mesh_domain::contains(Integer(0), 0, 0, meshSize));
    }
}

}    // namespace detail

TEST(MeshDomain, Fixed32Boundaries) {
    detail::checkFixedBoundaries<uint32_t>();
}

TEST(MeshDomain, Fixed64Boundaries) {
    detail::checkFixedBoundaries<uint64_t>();
    unsigned __int128 const period = static_cast<unsigned __int128>(1) << 64;
    uint64_t const lastLeft = static_cast<uint64_t>(period / 7);
    // These two positions collapse to the same double, but belong to different
    // cells and must migrate to opposite sides of the domain boundary.
    EXPECT_EQ(double(lastLeft), double(lastLeft + 1));
    EXPECT_TRUE(mesh_domain::belowBoundary(lastLeft, 1, 7));
    EXPECT_FALSE(mesh_domain::belowBoundary(lastLeft + 1, 1, 7));
}

TEST(MeshDomain, BoundaryRoundTrip) {
    for (uint32_t meshSize : {1U, 7U, 8U, 31U, 65535U}) {
        for (uint32_t plane = 0; plane <= meshSize; ++plane) {
            EXPECT_EQ(mesh_domain::boundaryIndex(
                          mesh_domain::boundaryCoordinate(plane, meshSize), meshSize),
                plane);
        }
    }
    uint32_t const meshSize = std::numeric_limits<uint32_t>::max();
    for (uint32_t plane : {0U, 1U, meshSize / 2, meshSize - 1, meshSize}) {
        EXPECT_EQ(
            mesh_domain::boundaryIndex(mesh_domain::boundaryCoordinate(plane, meshSize), meshSize),
            plane);
    }
}

TEST(MeshDomain, UnequalRankGroupsChooseParticleBalance) {
    // The left child owns one of three ranks, so its target is 10/3 particles.
    // Rounding the geometrical division alone would choose plane 2 with zero.
    std::array<size_t, 8> const below{0, 0, 0, 5, 6, 9, 10, 10};
    uint32_t best = 0;
    for (uint32_t candidate = 1; candidate < below.size(); ++candidate) {
        if (mesh_domain::betterSplit(
                candidate, below[candidate], best, below[best], 10, 1, 3, 0, 7)) {
            best = candidate;
        }
    }
    EXPECT_EQ(best, 3);
    EXPECT_EQ(mesh_domain::compareTarget(3, 10, 1, 3), -1);
    EXPECT_EQ(mesh_domain::compareTarget(4, 10, 1, 3), 1);
    EXPECT_EQ(mesh_domain::compareTarget(4, 12, 1, 3), 0);
}

TEST(MeshDomain, ParticleBalanceBeatsGeometricDistance) {
    EXPECT_TRUE(mesh_domain::betterSplit(1, 49, 4, 80, 100, 1, 2, 0, 8));
    EXPECT_FALSE(mesh_domain::betterSplit(4, 80, 1, 49, 100, 1, 2, 0, 8));
    EXPECT_TRUE(mesh_domain::betterSplit(3, 40, 1, 60, 100, 1, 2, 0, 8));
}

TEST(MeshDomain, EmptyRegionsChooseGeometricSplit) {
    uint32_t best = 2;
    for (uint32_t candidate = 3; candidate <= 9; ++candidate) {
        if (mesh_domain::betterSplit(candidate, 0, best, 0, 0, 1, 3, 2, 9)) {
            best = candidate;
        }
    }
    EXPECT_EQ(best, 4);
    // Equidistant boundaries resolve toward the smaller plane.
    EXPECT_TRUE(mesh_domain::betterSplit(3, 0, 4, 0, 0, 1, 2, 0, 7));
    EXPECT_FALSE(mesh_domain::betterSplit(4, 0, 3, 0, 0, 1, 2, 0, 7));
    EXPECT_FALSE(mesh_domain::betterSplit(3, 0, 3, 0, 0, 1, 2, 3, 3));
}

TEST(MeshDomain, LargeCountsDoNotOverflow) {
    size_t const total = std::numeric_limits<size_t>::max();
    size_t const nearThird = total / 3;
    uint32_t const rankCount = std::numeric_limits<uint32_t>::max();
    uint32_t const leftRanks = rankCount / 3;
    EXPECT_EQ(mesh_domain::compareTarget(nearThird, total, leftRanks, rankCount), 0);
    EXPECT_EQ(mesh_domain::compareTarget(nearThird + 1, total, leftRanks, rankCount), 1);
    EXPECT_EQ(mesh_domain::compareTarget(nearThird - 1, total, leftRanks, rankCount), -1);
    EXPECT_TRUE(mesh_domain::betterSplit(
        2, nearThird, 3, nearThird + 1, total, leftRanks, rankCount, 0, 7));
}

TEST(MeshDomain, InvalidInputsRejected) {
    EXPECT_THROW(mesh_domain::boundaryCoordinate(1, 0), std::invalid_argument);
    EXPECT_THROW(mesh_domain::boundaryCoordinate(8, 7), std::invalid_argument);
    EXPECT_THROW(mesh_domain::boundaryIndex(-0.1, 7), std::invalid_argument);
    EXPECT_THROW(mesh_domain::boundaryIndex(1.1, 7), std::invalid_argument);
    EXPECT_THROW(mesh_domain::boundaryIndex(std::nan(""), 7), std::invalid_argument);
    EXPECT_THROW(mesh_domain::belowBoundary(uint64_t(0), 8, 7), std::invalid_argument);
    EXPECT_THROW(mesh_domain::contains(uint64_t(0), 3, 2, 7), std::invalid_argument);
    EXPECT_THROW(mesh_domain::compareTarget(11, 10, 1, 2), std::invalid_argument);
    EXPECT_THROW(mesh_domain::compareTarget(1, 10, 1, 0), std::invalid_argument);
    EXPECT_THROW(mesh_domain::betterSplit(8, 5, 4, 5, 10, 1, 2, 0, 7), std::invalid_argument);
}
