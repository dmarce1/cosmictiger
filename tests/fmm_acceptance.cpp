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

#include <cosmictiger/fmm/fmm_acceptance.hpp>
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
    float pairSupportRadius;
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
    for (bool sinkLeaf : {false, true}) {
        for (bool sourceLeaf : {false, true}) {
            cases.push_back({"CC takes precedence at every tree level", 16.0f, 1.0f, 1.0f, 9.0f,
                9.0f, 0.5f, 1.5f, sinkLeaf, sourceLeaf, true, FmmInteraction::CellCell});
        }
    }

    // The source-center distance to the sink box resolves this anisotropic pair
    // even though the enclosing-sphere PC angular bound is exactly tangent.
    cases.push_back({"PC between two leaves", 1.0f, 0.5f, 0.25f, 0.36f, 0.5625f, 0.1f, 2.0f, true,
        true, true, FmmInteraction::ParticleCell});
    cases.push_back({"PC from an internal source into a sink leaf", 1.0f, 0.5f, 0.25f, 0.36f,
        0.5625f, 0.1f, 2.0f, true, false, true, FmmInteraction::ParticleCell});
    cases.push_back({"PC cannot expose particles from an internal sink", 1.0f, 0.5f, 0.25f, 0.36f,
        0.5625f, 0.1f, 2.0f, false, true, true, FmmInteraction::ParticleParticle});
    cases.push_back({"internal pair must descend when CC fails", 1.0f, 0.5f, 0.25f, 0.36f, 0.5625f,
        0.1f, 2.0f, false, false, true, FmmInteraction::ParticleParticle});

    // Both hybrid geometries pass while CC fails. A leaf can interact with an
    // internal cloud even when the reciprocal particle side is unavailable.
    cases.push_back({"PC is the eligible hybrid for a sink leaf", 1.69f, 0.5f, 0.5f, 1.21f, 1.21f,
        0.1f, 2.0f, true, false, true, FmmInteraction::ParticleCell});
    cases.push_back({"CP is the eligible hybrid for a source leaf", 1.69f, 0.5f, 0.5f, 1.21f, 1.21f,
        0.1f, 2.0f, false, true, true, FmmInteraction::CellParticle});
    cases.push_back({"two internal clouds cannot use either hybrid", 1.69f, 0.5f, 0.5f, 1.21f,
        1.21f, 0.1f, 2.0f, false, false, true, FmmInteraction::ParticleParticle});
    cases.push_back({"two leaves prefer the smaller source cloud", 1.69f, 0.5f, 0.25f, 1.21f, 1.21f,
        0.1f, 2.0f, true, true, true, FmmInteraction::ParticleCell});
    auto reverseRadius = reciprocalCase(cases.back());
    reverseRadius.label = "two leaves prefer the smaller sink cloud";
    cases.push_back(reverseRadius);
    cases.push_back({"leaf eligibility overrides the smaller cloud preference", 1.69f, 0.25f, 0.5f,
        1.21f, 1.21f, 0.1f, 2.0f, true, false, true, FmmInteraction::ParticleCell});
    float const lowerRadius = std::nextafter(0.5f, 0.0f);
    float const upperRadius = std::nextafter(0.5f, std::numeric_limits<float>::infinity());
    cases.push_back({"adjacent smaller source radius selects PC", 1.69f, 0.5f, lowerRadius, 1.21f,
        1.21f, 0.1f, 2.0f, true, true, true, FmmInteraction::ParticleCell});
    cases.push_back({"equal radii retain reciprocal PP fallback", 1.69f, 0.5f, 0.5f, 1.21f, 1.21f,
        0.1f, 2.0f, true, true, true, FmmInteraction::ParticleParticle});
    cases.push_back({"adjacent larger source radius selects CP", 1.69f, 0.5f, upperRadius, 1.21f,
        1.21f, 0.1f, 2.0f, true, true, true, FmmInteraction::CellParticle});

    // Internal-cloud eligibility must retain strict angular opening boundaries.
    for (float center : {std::nextafter(0.390625f, 0.0f), 0.390625f,
             std::nextafter(0.390625f, std::numeric_limits<float>::infinity())}) {
        cases.push_back({"strict center opening for an internal source", center, 0.125f, 0.25f,
            0.0f, 0.0f, 0.0625f, 2.0f, true, false, true,
            center > 0.390625f ? FmmInteraction::ParticleCell : FmmInteraction::ParticleParticle});
    }
    for (float boxDistance : {std::nextafter(0.25f, 0.0f), 0.25f,
             std::nextafter(0.25f, std::numeric_limits<float>::infinity())}) {
        cases.push_back({"strict box opening for an internal source", 0.49f, 0.5f, 0.25f,
            boxDistance, 0.0f, 0.1f, 2.0f, true, false, true,
            boxDistance > 0.25f ? FmmInteraction::ParticleCell : FmmInteraction::ParticleParticle});
    }

    // Every separation bound is exactly tangent to support. Angular acceptance
    // alone must not permit either an expansion or a box shortcut.
    float const lowerCenter = std::nextafter(0.25f, 0.0f);
    float const upperCenter = std::nextafter(0.25f, std::numeric_limits<float>::infinity());
    for (float center : {lowerCenter, 0.25f, upperCenter}) {
        bool const separated = center > 0.25f;
        cases.push_back({"strict center support boundary", center, 0.125f, 0.25f, 0.140625f,
            0.0625f, 0.125f, 1.25f, true, false, separated,
            separated ? FmmInteraction::CellCell : FmmInteraction::ParticleParticle});
    }
    float const lowerBox = std::nextafter(0.0625f, 0.0f);
    float const upperBox = std::nextafter(0.0625f, std::numeric_limits<float>::infinity());
    for (float boxDistance : {lowerBox, 0.0625f, upperBox}) {
        bool const separated = boxDistance > 0.0625f;
        cases.push_back({"strict box support boundary", 0.49f, 0.5f, 0.125f, boxDistance, 0.330625f,
            0.125f, 1.5f, true, false, separated,
            separated ? FmmInteraction::ParticleCell : FmmInteraction::ParticleParticle});
    }

    // Centers are 7/16 apart, both radii are 1/8, and the closest particle pair
    // can be 3/16 apart. The old CUDA max(R/theta,H) shortcut accepts H=1/4,
    // because each center is 5/16 from the other box. That pair is softened.
    cases.push_back(
        {"CUDA shortcut must include source extent", 0.19140625f, 0.125f, 0.125f, 0.09765625f,
            0.09765625f, 0.25f, 2.0f, true, false, false, FmmInteraction::ParticleParticle});
    cases.push_back({"same geometry with smaller pair support", 0.19140625f, 0.125f, 0.125f,
        0.09765625f, 0.09765625f, 0.125f, 2.0f, true, false, true, FmmInteraction::ParticleCell});
    cases.push_back({"coincident cells remain PP", 0.0f, 0.125f, 0.125f, 0.0f, 0.0f, 0.125f, 2.0f,
        true, true, false, FmmInteraction::ParticleParticle});
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
        T(value.sourceBoxToSinkCenterSquared), T(value.pairSupportRadius), T(value.inverseTheta));
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
    for (size_t start = 0; start < cases.size(); start += SIMD_FLOAT_SIZE) {
        SimdFloat centerDistanceSquared{};
        SimdFloat sinkRadius{};
        SimdFloat sourceRadius{};
        SimdFloat sinkBoxToSourceCenterSquared{};
        SimdFloat sourceBoxToSinkCenterSquared{};
        SimdFloat pairSupportRadius{};
        SimdFloat inverseTheta{};
        for (int lane = 0; lane < SIMD_FLOAT_SIZE; lane++) {
            auto const& value = cases[(start + lane) % cases.size()];
            simdSetLane(centerDistanceSquared, lane, value.centerDistanceSquared);
            simdSetLane(sinkRadius, lane, value.sinkRadius);
            simdSetLane(sourceRadius, lane, value.sourceRadius);
            simdSetLane(sinkBoxToSourceCenterSquared, lane, value.sinkBoxToSourceCenterSquared);
            simdSetLane(sourceBoxToSinkCenterSquared, lane, value.sourceBoxToSinkCenterSquared);
            simdSetLane(pairSupportRadius, lane, value.pairSupportRadius);
            simdSetLane(inverseTheta, lane, value.inverseTheta);
        }
        auto const acceptance = computeFmmAcceptance(centerDistanceSquared, sinkRadius,
            sourceRadius, sinkBoxToSourceCenterSquared, sourceBoxToSinkCenterSquared,
            pairSupportRadius, inverseTheta);
        for (int lane = 0; lane < SIMD_FLOAT_SIZE; lane++) {
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

int main() {
    try {
        auto const cases = detail::acceptanceCases();
        detail::testScalarCases<float>(cases);
        detail::testScalarCases<double>(cases);
#if defined(FMM_ACCEPTANCE_TEST_SIMD)
        detail::testSimdCases(cases);
#endif
        detail::require(fmmInverseOpeningAngle(0.5f) == 2.0f, "inverse opening angle");
        detail::testHybridCosts();
    } catch (std::exception const& error) {
        std::fprintf(stderr, "FMM acceptance test failed: %s\n", error.what());
        return 1;
    }
    std::puts("FMM acceptance tests passed");
    return 0;
}
