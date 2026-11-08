/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <cosmictiger/gravity/fmm_acceptance.hpp>
#if defined(FMM_ACCEPTANCE_TEST_SIMD)
#include <cosmictiger/math/simd.hpp>
#endif

namespace detail {

struct AcceptanceCase {
    char const* label;
    float centerDistanceSquared;
    float sinkRadius;
    float sourceRadius;
    float sinkBoxToSourceCenterSquared;
    float sourceBoxToSinkCenterSquared;
    float pairSmoothingLength;
    float inverseTheta;
    bool sinkLeaf;
    bool sourceLeaf;
    bool supportSeparated;
    FmmInteraction expected;
};

void require(bool condition, std::string const& label) {
    if (!condition) {
        throw std::runtime_error(label);
    }
}

FmmInteraction reciprocalInteraction(FmmInteraction interaction) {
    if (interaction == FmmInteraction::ParticleCell) {
        return FmmInteraction::CellParticle;
    }
    if (interaction == FmmInteraction::CellParticle) {
        return FmmInteraction::ParticleCell;
    }
    return interaction;
}

AcceptanceCase reciprocalCase(AcceptanceCase value) {
    std::swap(value.sinkRadius, value.sourceRadius);
    std::swap(value.sinkBoxToSourceCenterSquared, value.sourceBoxToSinkCenterSquared);
    std::swap(value.sinkLeaf, value.sourceLeaf);
    value.expected = reciprocalInteraction(value.expected);
    return value;
}

std::vector<AcceptanceCase> acceptanceCases() {
    std::vector<AcceptanceCase> cases;
    AcceptanceCase next;
    for (bool sinkLeaf : {false, true}) {
        for (bool sourceLeaf : {false, true}) {
            next.label = "CC takes precedence at every tree level";
            next.centerDistanceSquared = 16.0f;
            next.sinkRadius = 1.0f;
            next.sourceRadius = 1.0f;
            next.sinkBoxToSourceCenterSquared = 9.0f;
            next.sourceBoxToSinkCenterSquared = 9.0f;
            next.pairSmoothingLength = 0.5f;
            next.inverseTheta = 1.5f;
            next.sinkLeaf = sinkLeaf;
            next.sourceLeaf = sourceLeaf;
            next.supportSeparated = true;
            next.expected = FmmInteraction::CellCell;
            cases.push_back(next);
        }
    }

    // The source-center distance to the sink box resolves this anisotropic pair
    // even though the enclosing-sphere PC angular bound is exactly tangent.
    next.label = "PC between two leaves";
    next.centerDistanceSquared = 1.0f;
    next.sinkRadius = 0.5f;
    next.sourceRadius = 0.25f;
    next.sinkBoxToSourceCenterSquared = 0.36f;
    next.sourceBoxToSinkCenterSquared = 0.5625f;
    next.pairSmoothingLength = 0.1f;
    next.inverseTheta = 2.0f;
    next.sinkLeaf = true;
    next.sourceLeaf = true;
    next.supportSeparated = true;
    next.expected = FmmInteraction::ParticleCell;
    cases.push_back(next);
    next.label = "PC from an internal source into a sink leaf";
    next.centerDistanceSquared = 1.0f;
    next.sinkRadius = 0.5f;
    next.sourceRadius = 0.25f;
    next.sinkBoxToSourceCenterSquared = 0.36f;
    next.sourceBoxToSinkCenterSquared = 0.5625f;
    next.pairSmoothingLength = 0.1f;
    next.inverseTheta = 2.0f;
    next.sinkLeaf = true;
    next.sourceLeaf = false;
    next.supportSeparated = true;
    next.expected = FmmInteraction::ParticleCell;
    cases.push_back(next);
    next.label = "PC cannot expose particles from an internal sink";
    next.centerDistanceSquared = 1.0f;
    next.sinkRadius = 0.5f;
    next.sourceRadius = 0.25f;
    next.sinkBoxToSourceCenterSquared = 0.36f;
    next.sourceBoxToSinkCenterSquared = 0.5625f;
    next.pairSmoothingLength = 0.1f;
    next.inverseTheta = 2.0f;
    next.sinkLeaf = false;
    next.sourceLeaf = true;
    next.supportSeparated = true;
    next.expected = FmmInteraction::ParticleParticle;
    cases.push_back(next);
    next.label = "internal pair must descend when CC fails";
    next.centerDistanceSquared = 1.0f;
    next.sinkRadius = 0.5f;
    next.sourceRadius = 0.25f;
    next.sinkBoxToSourceCenterSquared = 0.36f;
    next.sourceBoxToSinkCenterSquared = 0.5625f;
    next.pairSmoothingLength = 0.1f;
    next.inverseTheta = 2.0f;
    next.sinkLeaf = false;
    next.sourceLeaf = false;
    next.supportSeparated = true;
    next.expected = FmmInteraction::ParticleParticle;
    cases.push_back(next);

    // Both hybrid geometries pass while CC fails. A leaf can interact with an
    // internal cloud even when the reciprocal particle side is unavailable.
    next.label = "PC is the eligible hybrid for a sink leaf";
    next.centerDistanceSquared = 1.69f;
    next.sinkRadius = 0.5f;
    next.sourceRadius = 0.5f;
    next.sinkBoxToSourceCenterSquared = 1.21f;
    next.sourceBoxToSinkCenterSquared = 1.21f;
    next.pairSmoothingLength = 0.1f;
    next.inverseTheta = 2.0f;
    next.sinkLeaf = true;
    next.sourceLeaf = false;
    next.supportSeparated = true;
    next.expected = FmmInteraction::ParticleCell;
    cases.push_back(next);
    next.label = "CP is the eligible hybrid for a source leaf";
    next.centerDistanceSquared = 1.69f;
    next.sinkRadius = 0.5f;
    next.sourceRadius = 0.5f;
    next.sinkBoxToSourceCenterSquared = 1.21f;
    next.sourceBoxToSinkCenterSquared = 1.21f;
    next.pairSmoothingLength = 0.1f;
    next.inverseTheta = 2.0f;
    next.sinkLeaf = false;
    next.sourceLeaf = true;
    next.supportSeparated = true;
    next.expected = FmmInteraction::CellParticle;
    cases.push_back(next);
    next.label = "two internal clouds cannot use either hybrid";
    next.centerDistanceSquared = 1.69f;
    next.sinkRadius = 0.5f;
    next.sourceRadius = 0.5f;
    next.sinkBoxToSourceCenterSquared = 1.21f;
    next.sourceBoxToSinkCenterSquared = 1.21f;
    next.pairSmoothingLength = 0.1f;
    next.inverseTheta = 2.0f;
    next.sinkLeaf = false;
    next.sourceLeaf = false;
    next.supportSeparated = true;
    next.expected = FmmInteraction::ParticleParticle;
    cases.push_back(next);
    next.label = "two leaves prefer the smaller source cloud";
    next.centerDistanceSquared = 1.69f;
    next.sinkRadius = 0.5f;
    next.sourceRadius = 0.25f;
    next.sinkBoxToSourceCenterSquared = 1.21f;
    next.sourceBoxToSinkCenterSquared = 1.21f;
    next.pairSmoothingLength = 0.1f;
    next.inverseTheta = 2.0f;
    next.sinkLeaf = true;
    next.sourceLeaf = true;
    next.supportSeparated = true;
    next.expected = FmmInteraction::ParticleCell;
    cases.push_back(next);
    auto reverseRadius = reciprocalCase(cases.back());
    reverseRadius.label = "two leaves prefer the smaller sink cloud";
    cases.push_back(reverseRadius);
    next.label = "leaf eligibility overrides the smaller cloud preference";
    next.centerDistanceSquared = 1.69f;
    next.sinkRadius = 0.25f;
    next.sourceRadius = 0.5f;
    next.sinkBoxToSourceCenterSquared = 1.21f;
    next.sourceBoxToSinkCenterSquared = 1.21f;
    next.pairSmoothingLength = 0.1f;
    next.inverseTheta = 2.0f;
    next.sinkLeaf = true;
    next.sourceLeaf = false;
    next.supportSeparated = true;
    next.expected = FmmInteraction::ParticleCell;
    cases.push_back(next);
    float const lowerRadius = std::nextafter(0.5f, 0.0f);
    float const upperRadius = std::nextafter(0.5f, std::numeric_limits<float>::infinity());
    next.label = "adjacent smaller source radius selects PC";
    next.centerDistanceSquared = 1.69f;
    next.sinkRadius = 0.5f;
    next.sourceRadius = lowerRadius;
    next.sinkBoxToSourceCenterSquared = 1.21f;
    next.sourceBoxToSinkCenterSquared = 1.21f;
    next.pairSmoothingLength = 0.1f;
    next.inverseTheta = 2.0f;
    next.sinkLeaf = true;
    next.sourceLeaf = true;
    next.supportSeparated = true;
    next.expected = FmmInteraction::ParticleCell;
    cases.push_back(next);
    next.label = "equal radii retain reciprocal PP fallback";
    next.centerDistanceSquared = 1.69f;
    next.sinkRadius = 0.5f;
    next.sourceRadius = 0.5f;
    next.sinkBoxToSourceCenterSquared = 1.21f;
    next.sourceBoxToSinkCenterSquared = 1.21f;
    next.pairSmoothingLength = 0.1f;
    next.inverseTheta = 2.0f;
    next.sinkLeaf = true;
    next.sourceLeaf = true;
    next.supportSeparated = true;
    next.expected = FmmInteraction::ParticleParticle;
    cases.push_back(next);
    next.label = "adjacent larger source radius selects CP";
    next.centerDistanceSquared = 1.69f;
    next.sinkRadius = 0.5f;
    next.sourceRadius = upperRadius;
    next.sinkBoxToSourceCenterSquared = 1.21f;
    next.sourceBoxToSinkCenterSquared = 1.21f;
    next.pairSmoothingLength = 0.1f;
    next.inverseTheta = 2.0f;
    next.sinkLeaf = true;
    next.sourceLeaf = true;
    next.supportSeparated = true;
    next.expected = FmmInteraction::CellParticle;
    cases.push_back(next);

    // Internal-cloud eligibility must retain strict angular opening boundaries.
    for (float center : {std::nextafter(0.390625f, 0.0f), 0.390625f,
             std::nextafter(0.390625f, std::numeric_limits<float>::infinity())}) {
        next.label = "strict center opening for an internal source";
        next.centerDistanceSquared = center;
        next.sinkRadius = 0.125f;
        next.sourceRadius = 0.25f;
        next.sinkBoxToSourceCenterSquared = 0.0f;
        next.sourceBoxToSinkCenterSquared = 0.0f;
        next.pairSmoothingLength = 0.0625f;
        next.inverseTheta = 2.0f;
        next.sinkLeaf = true;
        next.sourceLeaf = false;
        next.supportSeparated = true;
        next.expected =
            center > 0.390625f ? FmmInteraction::ParticleCell : FmmInteraction::ParticleParticle;
        cases.push_back(next);
    }
    for (float boxDistance : {std::nextafter(0.25f, 0.0f), 0.25f,
             std::nextafter(0.25f, std::numeric_limits<float>::infinity())}) {
        next.label = "strict box opening for an internal source";
        next.centerDistanceSquared = 0.49f;
        next.sinkRadius = 0.5f;
        next.sourceRadius = 0.25f;
        next.sinkBoxToSourceCenterSquared = boxDistance;
        next.sourceBoxToSinkCenterSquared = 0.0f;
        next.pairSmoothingLength = 0.1f;
        next.inverseTheta = 2.0f;
        next.sinkLeaf = true;
        next.sourceLeaf = false;
        next.supportSeparated = true;
        next.expected =
            boxDistance > 0.25f ? FmmInteraction::ParticleCell : FmmInteraction::ParticleParticle;
        cases.push_back(next);
    }

    // Every separation bound is exactly tangent to support. Angular acceptance
    // alone must not permit either an expansion or a box shortcut.
    float const lowerCenter = std::nextafter(0.25f, 0.0f);
    float const upperCenter = std::nextafter(0.25f, std::numeric_limits<float>::infinity());
    for (float center : {lowerCenter, 0.25f, upperCenter}) {
        bool const separated = center > 0.25f;
        next.label = "strict center support boundary";
        next.centerDistanceSquared = center;
        next.sinkRadius = 0.125f;
        next.sourceRadius = 0.25f;
        next.sinkBoxToSourceCenterSquared = 0.140625f;
        next.sourceBoxToSinkCenterSquared = 0.0625f;
        next.pairSmoothingLength = 0.125f;
        next.inverseTheta = 1.25f;
        next.sinkLeaf = true;
        next.sourceLeaf = false;
        next.supportSeparated = separated;
        next.expected = separated ? FmmInteraction::CellCell : FmmInteraction::ParticleParticle;
        cases.push_back(next);
    }
    float const lowerBox = std::nextafter(0.0625f, 0.0f);
    float const upperBox = std::nextafter(0.0625f, std::numeric_limits<float>::infinity());
    for (float boxDistance : {lowerBox, 0.0625f, upperBox}) {
        bool const separated = boxDistance > 0.0625f;
        next.label = "strict box support boundary";
        next.centerDistanceSquared = 0.49f;
        next.sinkRadius = 0.5f;
        next.sourceRadius = 0.125f;
        next.sinkBoxToSourceCenterSquared = boxDistance;
        next.sourceBoxToSinkCenterSquared = 0.330625f;
        next.pairSmoothingLength = 0.125f;
        next.inverseTheta = 1.5f;
        next.sinkLeaf = true;
        next.sourceLeaf = false;
        next.supportSeparated = separated;
        next.expected = separated ? FmmInteraction::ParticleCell : FmmInteraction::ParticleParticle;
        cases.push_back(next);
    }

    // Centers are 7/16 apart, both radii are 1/8, and the closest particle pair
    // can be 3/16 apart. The old CUDA max(R/theta,H) shortcut accepts H=1/4,
    // because each center is 5/16 from the other box. That pair is softened.
    next.label = "CUDA shortcut must include source extent";
    next.centerDistanceSquared = 0.19140625f;
    next.sinkRadius = 0.125f;
    next.sourceRadius = 0.125f;
    next.sinkBoxToSourceCenterSquared = 0.09765625f;
    next.sourceBoxToSinkCenterSquared = 0.09765625f;
    next.pairSmoothingLength = 0.25f;
    next.inverseTheta = 2.0f;
    next.sinkLeaf = true;
    next.sourceLeaf = false;
    next.supportSeparated = false;
    next.expected = FmmInteraction::ParticleParticle;
    cases.push_back(next);
    next.label = "same geometry with smaller pair support";
    next.centerDistanceSquared = 0.19140625f;
    next.sinkRadius = 0.125f;
    next.sourceRadius = 0.125f;
    next.sinkBoxToSourceCenterSquared = 0.09765625f;
    next.sourceBoxToSinkCenterSquared = 0.09765625f;
    next.pairSmoothingLength = 0.125f;
    next.inverseTheta = 2.0f;
    next.sinkLeaf = true;
    next.sourceLeaf = false;
    next.supportSeparated = true;
    next.expected = FmmInteraction::ParticleCell;
    cases.push_back(next);
    next.label = "coincident cells remain PP";
    next.centerDistanceSquared = 0.0f;
    next.sinkRadius = 0.125f;
    next.sourceRadius = 0.125f;
    next.sinkBoxToSourceCenterSquared = 0.0f;
    next.sourceBoxToSinkCenterSquared = 0.0f;
    next.pairSmoothingLength = 0.125f;
    next.inverseTheta = 2.0f;
    next.sinkLeaf = true;
    next.sourceLeaf = true;
    next.supportSeparated = false;
    next.expected = FmmInteraction::ParticleParticle;
    cases.push_back(next);
    // Include both orientations in the SIMD packs as well as in the scalar
    // reciprocity assertions, especially CP contributions to internal sinks.
    size_t const originalCaseCount = cases.size();
    for (size_t i = 0; i < originalCaseCount; ++i) {
        cases.push_back(reciprocalCase(cases[i]));
    }
    return cases;
}

template <class T>
FmmAcceptance<T> evaluate(AcceptanceCase const& value) {
    return computeFmmAcceptance(T(value.centerDistanceSquared), T(value.sinkRadius),
        T(value.sourceRadius), T(value.sinkBoxToSourceCenterSquared),
        T(value.sourceBoxToSinkCenterSquared), T(value.pairSmoothingLength), T(value.inverseTheta));
}

template <class T>
void testScalarCases(std::vector<AcceptanceCase> const& cases) {
    for (auto const& value : cases) {
        auto const acceptance = evaluate<T>(value);
        auto const interaction = selectFmmInteraction(acceptance.cellCell != T(0),
            acceptance.particleCell != T(0), acceptance.cellParticle != T(0), value.sinkLeaf,
            value.sourceLeaf, value.sinkRadius, value.sourceRadius);
        require(interaction == value.expected, std::string(value.label) + ": interaction");
        require((acceptance.supportSeparated != T(0)) == value.supportSeparated,
            std::string(value.label) + ": support separation");
        for (T mask : {acceptance.cellCell, acceptance.particleCell, acceptance.cellParticle,
                 acceptance.supportSeparated}) {
            require(mask == T(0) || mask == T(1), std::string(value.label) + ": non-boolean mask");
        }
        auto const reverse = reciprocalCase(value);
        auto const reverseAcceptance = evaluate<T>(reverse);
        require(acceptance.cellCell == reverseAcceptance.cellCell &&
                acceptance.particleCell == reverseAcceptance.cellParticle &&
                acceptance.cellParticle == reverseAcceptance.particleCell &&
                acceptance.supportSeparated == reverseAcceptance.supportSeparated,
            std::string(value.label) + ": geometry is not reciprocal");
        auto const reverseInteraction = selectFmmInteraction(reverseAcceptance.cellCell != T(0),
            reverseAcceptance.particleCell != T(0), reverseAcceptance.cellParticle != T(0),
            reverse.sinkLeaf, reverse.sourceLeaf, reverse.sinkRadius, reverse.sourceRadius);
        require(reverseInteraction == reciprocalInteraction(interaction),
            std::string(value.label) + ": selection is not reciprocal");
    }
}

#if defined(FMM_ACCEPTANCE_TEST_SIMD)
void testSimdCases(std::vector<AcceptanceCase> const& cases) {
    for (size_t start = 0; start < cases.size(); start += kSimdFloatSize) {
        SimdFloat centerDistanceSquared{};
        SimdFloat sinkRadius{};
        SimdFloat sourceRadius{};
        SimdFloat sinkBoxToSourceCenterSquared{};
        SimdFloat sourceBoxToSinkCenterSquared{};
        SimdFloat pairSmoothingLength{};
        SimdFloat inverseTheta{};
        for (int lane = 0; lane < kSimdFloatSize; lane++) {
            auto const& value = cases[(start + lane) % cases.size()];
            simdSetLane(centerDistanceSquared, lane, value.centerDistanceSquared);
            simdSetLane(sinkRadius, lane, value.sinkRadius);
            simdSetLane(sourceRadius, lane, value.sourceRadius);
            simdSetLane(sinkBoxToSourceCenterSquared, lane, value.sinkBoxToSourceCenterSquared);
            simdSetLane(sourceBoxToSinkCenterSquared, lane, value.sourceBoxToSinkCenterSquared);
            simdSetLane(pairSmoothingLength, lane, value.pairSmoothingLength);
            simdSetLane(inverseTheta, lane, value.inverseTheta);
        }
        auto const acceptance = computeFmmAcceptance(centerDistanceSquared, sinkRadius,
            sourceRadius, sinkBoxToSourceCenterSquared, sourceBoxToSinkCenterSquared,
            pairSmoothingLength, inverseTheta);
        for (int lane = 0; lane < kSimdFloatSize; lane++) {
            auto const& value = cases[(start + lane) % cases.size()];
            auto const scalar = evaluate<float>(value);
            require(acceptance.cellCell[lane] == scalar.cellCell &&
                    acceptance.particleCell[lane] == scalar.particleCell &&
                    acceptance.cellParticle[lane] == scalar.cellParticle &&
                    acceptance.supportSeparated[lane] == scalar.supportSeparated,
                std::string(value.label) + ": SIMD/scalar mismatch");
            auto const interaction = selectFmmInteraction(acceptance.cellCell[lane],
                acceptance.particleCell[lane], acceptance.cellParticle[lane], value.sinkLeaf,
                value.sourceLeaf, value.sinkRadius, value.sourceRadius);
            require(interaction == value.expected, std::string(value.label) + ": SIMD selection");
        }
    }
}
#endif

// For a leaf pair the cheapest of the valid hybrids and direct summation is
// taken, from the particle counts and the per-order costs; equal counts keep
// the radius rule, and the decision mirrors between the sides.
void testHybridCosts() {
    using Interaction = FmmInteraction;
    float const hybrid = 300.0f;
    float const pair = 38.0f;
    auto const select = [&](bool particleCell, bool cellParticle, int sinkCount, int sourceCount,
                            float sinkRadius = 1.0f, float sourceRadius = 1.0f) {
        return selectFmmInteraction(false, particleCell, cellParticle, true, true, sinkRadius,
            sourceRadius, sinkCount, sourceCount, hybrid, pair);
    };
    require(select(true, true, 2, 2) == Interaction::ParticleParticle, "tiny leaves sum directly");
    require(select(true, true, 64, 40) == Interaction::CellParticle,
        "the leaf with fewer particles is evaluated explicitly");
    require(select(true, true, 40, 64) == Interaction::ParticleCell,
        "the leaf with fewer particles is evaluated explicitly, mirrored");
    require(select(true, false, 64, 40) == Interaction::ParticleCell,
        "a lone valid hybrid beats direct summation");
    require(select(true, false, 1, 7) == Interaction::ParticleParticle,
        "direct summation beats a costlier lone hybrid");
    require(select(true, true, 32, 32, 2.0f, 1.0f) == Interaction::ParticleCell,
        "equal counts keep the smaller cloud");
    require(select(true, true, 32, 32) == Interaction::ParticleParticle,
        "equal counts and radii fall back to direct summation");
    require(selectFmmInteraction(false, true, true, true, true, 2.0f, 1.0f, 64, 40) ==
            Interaction::ParticleCell,
        "without costs the radius rule decides");
    require(selectFmmInteraction(false, true, true, true, false, 1.0f, 1.0f, 1, 1000, hybrid,
                pair) == Interaction::ParticleCell,
        "a leaf against an internal cell is not costed");
    for (int a : {1, 4, 40, 64}) {
        for (int b : {1, 4, 40, 64}) {
            auto const forward = select(true, true, a, b, 1.0f, 1.5f);
            auto const reverse = select(true, true, b, a, 1.5f, 1.0f);
            require(reverse == reciprocalInteraction(forward), "cost choice is not reciprocal");
        }
    }
}

}    // namespace detail

TEST(FmmAcceptance, ScalarFloat) {
    detail::testScalarCases<float>(detail::acceptanceCases());
}

TEST(FmmAcceptance, ScalarDouble) {
    detail::testScalarCases<double>(detail::acceptanceCases());
}

#if defined(FMM_ACCEPTANCE_TEST_SIMD)
TEST(FmmAcceptance, Simd) {
    detail::testSimdCases(detail::acceptanceCases());
}
#endif

TEST(FmmAcceptance, InverseOpeningAngle) {
    EXPECT_EQ(fmmInverseOpeningAngle(0.5f), 2.0f);
}

TEST(FmmAcceptance, HybridCosts) {
    detail::testHybridCosts();
}

TEST(FmmAcceptance, AcceptedLeafCellCosts) {
    using Interaction = FmmInteraction;
    for (auto const counts : {std::pair{2, 8}, std::pair{8, 2}, std::pair{8, 8}}) {
        EXPECT_EQ(selectFmmInteraction(true, true, true, true, true, 1.0f, 1.0f, counts.first,
                      counts.second, 300.0f, 38.0f, 4000.0f),
            Interaction::ParticleParticle);
    }
    EXPECT_EQ(selectFmmInteraction(
                  true, true, true, true, true, 1.0f, 1.0f, 64, 64, 300.0f, 38.0f, 4000.0f),
        Interaction::CellCell);
    EXPECT_EQ(selectFmmInteraction(
                  true, true, true, true, true, 1.0f, 1.0f, 8, 8, 300.0f, 38.0f, 2432.0f),
        Interaction::CellCell);
    for (auto const leaves :
        {std::pair{false, false}, std::pair{false, true}, std::pair{true, false}}) {
        EXPECT_EQ(selectFmmInteraction(true, true, true, leaves.first, leaves.second, 1.0f, 1.0f, 2,
                      2, 300.0f, 38.0f, 4000.0f),
            Interaction::CellCell);
    }
    EXPECT_EQ(selectFmmInteraction(true, true, true, true, true, 1.0f, 1.0f, 2, 2, 300.0f, 38.0f),
        Interaction::CellCell);
    EXPECT_EQ(
        selectFmmInteraction(true, true, true, true, true, 1.0f, 1.0f, 2, 2, 300.0f, 0.0f, 4000.0f),
        Interaction::CellCell);
}
