/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

// Adapted from dmarce1/experimental, commit
// 7ba8f1829fdede7348a1418b7cb36c0e7a955302 (2026 Friedmann solver).
namespace cosmology {

enum class CosmologicalTime { physical, conformal };
enum class ExpansionBranch { expanding, contracting };

// Parameters at a=1, with H0 in inverse caller time units. Densities must
// sum to one; radiation and matter are nonnegative, curvature/Lambda signed.
struct BackgroundCosmologyParameters {
    double hubbleConstant = 1.0;
    double omegaRadiation = 0.0;
    double omegaMatter = 0.3;
    double omegaDarkEnergy = 0.7;
    double omegaCurvature = 0.0;
};

// Scale factor uses relative error. The signed expansion variable also uses
// absolute error normalized by the local Friedmann component amplitude.
struct FriedmannOptions {
    double relativeTolerance = 1.0e-12;
    double absoluteTolerance = 1.0e-14;
    size_t maxSteps = 100000;
};

struct FriedmannState {
    double scaleFactor;
    double physicalDerivative;
    double conformalDerivative;
};

struct FriedmannAdvance {
    FriedmannState state;
    double physicalTime = 0.0;
    double conformalTime = 0.0;
    // Integral of d eta/a = dt/a^2, in the same time units as the input.
    double driftIntegral = 0.0;
    // Source-operation estimate, including rejected adaptive steps.
    uint64_t flops = 0;
};

namespace detail {
    constexpr size_t kFriedmannVariables = 5;
    using FriedmannVariables = std::array<double, kFriedmannVariables>;
    struct FriedmannDenseSegment {
        double begin;
        double step;
        FriedmannVariables initial;
        std::array<std::array<double, 4>, kFriedmannVariables> coefficients;
    };
}    // namespace detail

// Immutable Dormand-Prince quartic dense output. Times are elapsed from the
// initial state, in the clock selected when constructing this trajectory.
class FriedmannTrajectory {
public:
    FriedmannAdvance sample(double elapsedTime) const;
    FriedmannAdvance const& final() const noexcept;

private:
    friend class FriedmannCosmology;
    BackgroundCosmologyParameters parameters_;
    FriedmannState initial_{};
    FriedmannAdvance final_{};
    CosmologicalTime time_ = CosmologicalTime::conformal;
    double elapsedTime_ = 0.0;
    std::vector<detail::FriedmannDenseSegment> segments_;
};

// Elapsed times are measured from the supplied state, with either sign.
// Signed acceleration evolution follows expansion through turnaround.
class FriedmannCosmology {
public:
    explicit FriedmannCosmology(
        BackgroundCosmologyParameters parameters = {}, FriedmannOptions options = {});
    BackgroundCosmologyParameters const& parameters() const noexcept;
    FriedmannOptions const& options() const noexcept;
    FriedmannState state(
        double scaleFactor, ExpansionBranch branch = ExpansionBranch::expanding) const;
    FriedmannState advance(FriedmannState const& initialState, double elapsedTime,
        CosmologicalTime time = CosmologicalTime::physical) const;
    FriedmannAdvance advanceWithIntegrals(FriedmannState const& initialState, double elapsedTime,
        CosmologicalTime time = CosmologicalTime::conformal) const;
    FriedmannState evolve(double initialScaleFactor, double elapsedTime,
        CosmologicalTime time = CosmologicalTime::physical,
        ExpansionBranch branch = ExpansionBranch::expanding) const;
    FriedmannTrajectory trajectory(FriedmannState const& initialState, double elapsedTime,
        CosmologicalTime time = CosmologicalTime::conformal) const;
    // Signed intervals on the expanding branch only. A simple turnaround may
    // be an endpoint; inaccessible intervals and infinite-time roots throw.
    FriedmannAdvance intervalsBetweenScales(double initialScale, double finalScale) const;

private:
    FriedmannAdvance integrate(FriedmannState const& initialState, double elapsedTime,
        CosmologicalTime time, std::vector<detail::FriedmannDenseSegment>* segments) const;
    BackgroundCosmologyParameters parameters_;
    FriedmannOptions options_;
};

}    // namespace cosmology
