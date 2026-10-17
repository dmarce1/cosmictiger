/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the GNU General Public License, version 2 or later.
 */

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cosmictiger/fmm_acceptance.hpp>
#if defined(FMM_ACCEPTANCE_TEST_SIMD)
#include <cosmictiger/simd.hpp>
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
    std::int64_t sinkParticleCount;
    std::int64_t sourceParticleCount;
    bool bothLeaves;
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
    std::swap(value.sinkParticleCount, value.sourceParticleCount);
    value.expected = reciprocalInteraction(value.expected);
    return value;
}

std::vector<AcceptanceCase> acceptanceCases() {
    std::vector<AcceptanceCase> cases;
    cases.push_back({"CC takes precedence, including internal cells", 16.0f, 1.0f, 1.0f, 9.0f, 9.0f,
        0.5f, 1.5f, 37, 39, false, true, FmmInteraction::CellCell});

    // The source-center distance to the sink box resolves this anisotropic pair
    // even though the enclosing-sphere PC angular bound is exactly tangent.
    for (std::int64_t count : {37, 38, 39}) {
        cases.push_back({"PC particle threshold", 1.0f, 0.5f, 0.25f, 0.36f, 0.5625f, 0.1f, 2.0f,
            count, 91, true, true,
            count >= 38 ? FmmInteraction::ParticleCell : FmmInteraction::ParticleParticle});
        auto reverse = reciprocalCase(cases.back());
        reverse.label = "CP particle threshold";
        cases.push_back(reverse);
    }
    cases.push_back({"hybrid interactions require two leaves", 1.0f, 0.5f, 0.25f, 0.36f, 0.5625f,
        0.1f, 2.0f, 38, 91, false, true, FmmInteraction::ParticleParticle});

    // Both angular tests pass through the boxes, while CC fails. Choosing the
    // smaller explicit-particle side must reverse PC/CP when the pair is swapped.
    for (auto const counts : std::array<std::array<std::int64_t, 2>, 6>{
             {{38, 39}, {39, 38}, {38, 38}, {37, 39}, {39, 37}, {37, 37}}}) {
        FmmInteraction expected = FmmInteraction::ParticleParticle;
        if (counts[0] == 38 && counts[1] == 39) {
            expected = FmmInteraction::ParticleCell;
        } else if (counts[0] == 39 && counts[1] == 38) {
            expected = FmmInteraction::CellParticle;
        } else if (counts[0] == 37 && counts[1] == 39) {
            expected = FmmInteraction::CellParticle;
        } else if (counts[0] == 39 && counts[1] == 37) {
            expected = FmmInteraction::ParticleCell;
        }
        cases.push_back({"reciprocal hybrid tie selection", 1.69f, 0.5f, 0.5f, 1.21f, 1.21f, 0.1f,
            2.0f, counts[0], counts[1], true, true, expected});
    }
    cases.push_back({"counts above float integer precision", 1.69f, 0.5f, 0.5f, 1.21f, 1.21f, 0.1f,
        2.0f, 16777216, 16777217, true, true, FmmInteraction::ParticleCell});

    cases.push_back({"equal counts prefer the smaller source cell", 1.69f, 0.5f, 0.25f, 1.21f,
        1.21f, 0.1f, 2.0f, 38, 38, true, true, FmmInteraction::ParticleCell});
    auto reverseRadiusTie = reciprocalCase(cases.back());
    reverseRadiusTie.label = "equal counts prefer the smaller sink cell";
    cases.push_back(reverseRadiusTie);
    cases.push_back({"count preference overrides radius preference", 1.69f, 0.25f, 0.5f, 1.21f,
        1.21f, 0.1f, 2.0f, 38, 39, true, true, FmmInteraction::ParticleCell});
    cases.push_back({"precise integer counts override radius preference", 1.69f, 0.25f, 0.5f, 1.21f,
        1.21f, 0.1f, 2.0f, 16777216, 16777217, true, true, FmmInteraction::ParticleCell});
    cases.push_back({"radius preference does not bypass particle threshold", 1.69f, 0.5f, 0.25f,
        1.21f, 1.21f, 0.1f, 2.0f, 37, 37, true, true, FmmInteraction::ParticleParticle});
    float const lowerRadius = std::nextafter(0.5f, 0.0f);
    float const upperRadius = std::nextafter(0.5f, std::numeric_limits<float>::infinity());
    cases.push_back({"adjacent smaller source radius breaks count tie", 1.69f, 0.5f, lowerRadius,
        1.21f, 1.21f, 0.1f, 2.0f, 39, 39, true, true, FmmInteraction::ParticleCell});
    cases.push_back({"exactly equal radii preserve PP count tie", 1.69f, 0.5f, 0.5f, 1.21f, 1.21f,
        0.1f, 2.0f, 39, 39, true, true, FmmInteraction::ParticleParticle});
    cases.push_back({"adjacent larger source radius breaks count tie", 1.69f, 0.5f, upperRadius,
        1.21f, 1.21f, 0.1f, 2.0f, 39, 39, true, true, FmmInteraction::CellParticle});

    // Every separation bound is exactly tangent to support. Angular acceptance
    // alone must not permit either an expansion or a box shortcut.
    float const lowerCenter = std::nextafter(0.25f, 0.0f);
    float const upperCenter = std::nextafter(0.25f, std::numeric_limits<float>::infinity());
    for (float center : {lowerCenter, 0.25f, upperCenter}) {
        bool const separated = center > 0.25f;
        cases.push_back({"strict center support boundary", center, 0.125f, 0.25f, 0.140625f,
            0.0625f, 0.125f, 1.25f, 38, 39, true, separated,
            separated ? FmmInteraction::CellCell : FmmInteraction::ParticleParticle});
    }
    float const lowerBox = std::nextafter(0.0625f, 0.0f);
    float const upperBox = std::nextafter(0.0625f, std::numeric_limits<float>::infinity());
    for (float boxDistance : {lowerBox, 0.0625f, upperBox}) {
        bool const separated = boxDistance > 0.0625f;
        cases.push_back({"strict box support boundary", 0.49f, 0.5f, 0.125f, boxDistance, 0.330625f,
            0.125f, 1.5f, 38, 39, true, separated,
            separated ? FmmInteraction::ParticleCell : FmmInteraction::ParticleParticle});
    }

    // Centers are 7/16 apart, both radii are 1/8, and the closest particle pair
    // can be 3/16 apart. The old CUDA max(R/theta,H) shortcut accepts H=1/4,
    // because each center is 5/16 from the other box. That pair is softened.
    cases.push_back(
        {"CUDA shortcut must include source extent", 0.19140625f, 0.125f, 0.125f, 0.09765625f,
            0.09765625f, 0.25f, 2.0f, 38, 39, true, false, FmmInteraction::ParticleParticle});
    cases.push_back({"same geometry with smaller pair support", 0.19140625f, 0.125f, 0.125f,
        0.09765625f, 0.09765625f, 0.125f, 2.0f, 38, 39, true, true, FmmInteraction::ParticleCell});
    cases.push_back({"coincident cells remain PP", 0.0f, 0.125f, 0.125f, 0.0f, 0.0f, 0.125f, 2.0f,
        38, 39, true, false, FmmInteraction::ParticleParticle});
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
        auto const interaction =
            selectFmmInteraction(acceptance.cellCell != T(0), acceptance.particleCell != T(0),
                acceptance.cellParticle != T(0), value.bothLeaves, value.sinkParticleCount,
                value.sourceParticleCount, value.sinkRadius, value.sourceRadius);
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
            reverse.bothLeaves, reverse.sinkParticleCount, reverse.sourceParticleCount,
            reverse.sinkRadius, reverse.sourceRadius);
        require(reverseInteraction == reciprocalInteraction(interaction),
            std::string(value.label) + ": selection is not reciprocal");
    }
}

#if defined(FMM_ACCEPTANCE_TEST_SIMD)
void testSimdCases(std::vector<AcceptanceCase> const& cases) {
    for (size_t start = 0; start < cases.size(); start += SIMD_FLOAT_SIZE) {
        SimdFloat centerDistanceSquared;
        SimdFloat sinkRadius;
        SimdFloat sourceRadius;
        SimdFloat sinkBoxToSourceCenterSquared;
        SimdFloat sourceBoxToSinkCenterSquared;
        SimdFloat pairSupportRadius;
        SimdFloat inverseTheta;
        for (int lane = 0; lane < SIMD_FLOAT_SIZE; lane++) {
            auto const& value = cases[(start + lane) % cases.size()];
            centerDistanceSquared[lane] = value.centerDistanceSquared;
            sinkRadius[lane] = value.sinkRadius;
            sourceRadius[lane] = value.sourceRadius;
            sinkBoxToSourceCenterSquared[lane] = value.sinkBoxToSourceCenterSquared;
            sourceBoxToSinkCenterSquared[lane] = value.sourceBoxToSinkCenterSquared;
            pairSupportRadius[lane] = value.pairSupportRadius;
            inverseTheta[lane] = value.inverseTheta;
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
            auto const interaction = selectFmmInteraction(acceptance.cellCell[lane] != 0.0f,
                acceptance.particleCell[lane] != 0.0f, acceptance.cellParticle[lane] != 0.0f,
                value.bothLeaves, value.sinkParticleCount, value.sourceParticleCount,
                value.sinkRadius, value.sourceRadius);
            require(interaction == value.expected, std::string(value.label) + ": SIMD selection");
        }
    }
}
#endif

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
    } catch (std::exception const& error) {
        std::fprintf(stderr, "FMM acceptance test failed: %s\n", error.what());
        return 1;
    }
    std::puts("FMM acceptance tests passed");
    return 0;
}
