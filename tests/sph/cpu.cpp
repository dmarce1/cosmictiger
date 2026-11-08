/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/sph/cpu.hpp>
#include <cosmictiger/sph/viscosity/balsara.hpp>
#include <cosmictiger/sph/viscosity/monaghan.hpp>
#include <tests/sph/conduction_cases.hpp>
#include <tests/sph/global_timestep_cases.hpp>
#include <tests/sph/midpoint_cases.hpp>
#include <tests/sph/pair_kick_cases.hpp>
#include <tests/sph/rung_limiter_cases.hpp>
#include <tests/sph/source_completion_cases.hpp>

namespace detail {
FlopCounts submittedSphFlops;
}

void addCpuFlops(FlopCount count, FlopPrecision precision) {
    if (precision == FlopPrecision::Single) {
        detail::submittedSphFlops.singlePrecision += count;
    } else if (precision == FlopPrecision::Double) {
        detail::submittedSphFlops.doublePrecision += count;
    } else {
        detail::submittedSphFlops.unclassified += count;
    }
}

// The fixture owns ordinary host arrays and calls the production SIMD kernels.
// It does not start an HPX runtime or alias staged source snapshots with sinks.
Options const& getOptions() {
    static Options const options{};
    return options;
}

void cosmictigerAssert(char const* condition, bool value, char const* file, int line) {
    if (!value) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + condition);
    }
}

namespace detail {

constexpr double kPi = 3.141592653589793238462643383279502884;

using sph_geometry::SpatialVector;

void require(bool condition, char const* label) {
    if (!condition) {
        throw std::runtime_error(label);
    }
}

// Integer exponent checks remain effective when the production compiler flags
// include -ffast-math, which permits floating-point isfinite checks to disappear.
bool isFinite(float value) {
    static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == sizeof(uint32_t));
    return (std::bit_cast<uint32_t>(value) & 0x7f800000U) != 0x7f800000U;
}

bool isFinite(double value) {
    static_assert(std::numeric_limits<double>::is_iec559 && sizeof(double) == sizeof(uint64_t));
    return (std::bit_cast<uint64_t>(value) & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}

void requireClose(char const* label, double actual, double expected, double tolerance) {
    if (!isFinite(actual) || !isFinite(expected) || !isFinite(tolerance) ||
        std::abs(actual - expected) > tolerance * std::max(1.0, std::abs(expected))) {
        throw std::runtime_error(std::string(label) + ": got " + std::to_string(actual) +
            ", expected " + std::to_string(expected));
    }
}

class Fixture {
    std::vector<std::function<void()>> allocations_;
    SphRunReturn reductionValue_{};

    template <typename Value>
    Value* allocate(size_t count) {
        Value* pointer = new Value[count]{};
        allocations_.push_back([pointer] { delete[] pointer; });
        return pointer;
    }

public:
    int count_;
    int sourceOffset_;
    int sinkOffset_;
    int particleOffset_;
    SphRunParameters params_;
    SphRunData data_{};
    SphRunReturn* reduction_ = &reductionValue_;
    SphPairSourceId* pairSourceIds_ = nullptr;

    explicit Fixture(int count, int treeCount = 1, int sourceOffset = 0, int sinkOffset = 0,
        int particleOffset = 0)
      : count_(count)
      , sourceOffset_(sourceOffset)
      , sinkOffset_(sinkOffset)
      , particleOffset_(particleOffset) {
        size_t const sourceSize = count + sourceOffset + 3;
        size_t const sinkSize = count + sinkOffset + 3;
        size_t const particleSize = count + particleOffset + 3;
        data_.sourcePositionX = allocate<Position>(sourceSize);
        data_.sourcePositionY = allocate<Position>(sourceSize);
        data_.sourcePositionZ = allocate<Position>(sourceSize);
        data_.sourceVelocityX = allocate<float>(sourceSize);
        data_.sourceVelocityY = allocate<float>(sourceSize);
        data_.sourceVelocityZ = allocate<float>(sourceSize);
        data_.sourceInternalEnergy = allocate<float>(sourceSize);
        data_.sourceSmoothingLength = allocate<float>(sourceSize);
        data_.sourceDensity = allocate<float>(sourceSize);
        data_.sourceGradHCorrection = allocate<float>(sourceSize);
        data_.sinkGradHCorrection = allocate<float>(sinkSize);
        data_.sourceBalsaraFactor = allocate<float>(sourceSize);
        data_.sourceRungs = allocate<char>(sourceSize);
        data_.sinkDensity = allocate<float>(sinkSize);
        data_.sinkBalsaraFactor = allocate<uint8_t>(sinkSize);
        data_.sinkGravityAccelerationX = allocate<float>(sinkSize);
        data_.sinkGravityAccelerationY = allocate<float>(sinkSize);
        data_.sinkGravityAccelerationZ = allocate<float>(sinkSize);
        data_.sinkSmoothingLength = allocate<float>(sinkSize);
        data_.sinkParticleRungs = allocate<char>(particleSize);
        data_.sinkFlags = allocate<SphParticleFlags>(sinkSize);
        data_.sinkParticleIndices = allocate<ParticleInteger>(sinkSize);
        data_.treeNodes = allocate<SphTreeNode>(treeCount);
        data_.targetNodeIndices = allocate<int>(treeCount);
        data_.neighborNodeIndices = allocate<int>(treeCount);
        data_.numberOfTargetNodes = treeCount;
        data_.gasMass = 1.0f / count;
        data_.targetNeighborCount = 32.0f;
        data_.adiabaticIndex = 5.0f / 3.0f;
        data_.accelerationTimestepFactor = 0.25f;
        data_.gravityEffectiveRadius = 0.05f;
        params_.scaleFactor = 1.0f;
        params_.expansionRate = 0.0f;
        params_.minimumRung = 1;
        params_.baseTimestep = 1.0f;
        params_.maximumTimestep = 1.0f;
        params_.phase = 0;
        params_.courantFactor = 0.25f;
        params_.vonNeumannRichtmyerCoefficient = 2.0f;
        params_.viscosityEpsilon = 0.01f;
        params_.viscosityDecay = 0.2f;
        data_.sourceBulkViscosityCoefficient = allocate<float>(sourceSize);
        data_.sinkBulkViscosityCoefficient = allocate<Float16Storage>(sinkSize);
        params_.minimumSmoothingLength = 0.1f;
        // The initial isolated support is held at the upper bound.
        params_.maximumSmoothingLength = 0.2f;
        params_.uniformAccelerationY = 0.0f;
        params_.zPlaneAcceleration = 0.0f;
        params_.velocityDampingRate = 0.0f;
        for (int i = 0; i < count; i++) {
            int const source = sourceIndex(i);
            int const sink = sinkIndex(i);
            data_.sinkParticleIndices[sink] = particleIndex(i);
            data_.sourceBalsaraFactor[source] = 1.0f;
            data_.sinkBalsaraFactor[sink] = 255;
            data_.sourceRungs[source] = data_.sinkParticleRungs[particleIndex(i)] = 1;
            data_.sourceInternalEnergy[source] = 1.0f;
            data_.sourceSmoothingLength[source] = data_.sinkSmoothingLength[sink] = 0.2f;
        }
        setBulkViscosityCoefficient(1.0f);
        for (int tree = 0; tree < treeCount; tree++) {
            // Reverse the target order to exercise targetNodeIndices indirection.
            data_.targetNodeIndices[tree] = treeCount - tree - 1;
            data_.neighborNodeIndices[tree] = tree;
            auto& node = data_.treeNodes[tree];
            node.particleRange.first = sourceIndex(tree * count / treeCount);
            node.particleRange.second = sourceIndex((tree + 1) * count / treeCount);
            node.sinkParticleRange.first = sinkIndex(tree * count / treeCount);
            node.sinkParticleRange.second = sinkIndex((tree + 1) * count / treeCount);
            node.neighborRange = {0, treeCount};
            for (int dim = 0; dim < kDimensions; dim++) {
                node.innerBox.begin[dim] = 0.0;
                node.innerBox.end[dim] = 1.0;
                node.outerBox.begin[dim] = -1.0;
                node.outerBox.end[dim] = 2.0;
            }
        }
    }

    Fixture(Fixture const&) = delete;
    Fixture& operator=(Fixture const&) = delete;

    ~Fixture() {
        for (auto const& allocation : allocations_) {
            allocation();
        }
    }

    int sourceIndex(int index) const {
        return sourceOffset_ + index;
    }

    int sinkIndex(int index) const {
        return sinkOffset_ + index;
    }

    int particleIndex(int index) const {
        // The full-particle index is deliberately unrelated to the sink index.
        return particleOffset_ + count_ - index - 1;
    }

    // Equal bounds hold every particle's alpha fixed, the constant-coefficient limit.
    void setBulkViscosityCoefficient(float α) {
        params_.minimumBulkViscosityCoefficient = params_.maximumBulkViscosityCoefficient = α;
        for (int i = 0; i < count_; ++i) {
            data_.sourceBulkViscosityCoefficient[sourceIndex(i)] = α;
            data_.sinkBulkViscosityCoefficient[sinkIndex(i)] = α;
        }
    }

    void enableVariableViscosity() {
        setBulkViscosityCoefficient(0.7f);
        params_.minimumBulkViscosityCoefficient = 0.1f;
        params_.maximumBulkViscosityCoefficient = 1.5f;
    }

    void densityPass() {
        run(kSphRunDensityAndGradients);
    }

    void smoothingLengthsPass() {
        run(kSphRunSmoothingLengths);
    }

    void enableEntropyRateOutput() {
        if (!data_.sinkEntropyRate) {
            data_.sinkEntropyRate = allocate<float>(count_ + sinkOffset_ + 3);
        }
    }

    void enableMidpointOutputs() {
        enableEntropyRateOutput();
        data_.proposeRungs = true;
        for (int i = 0; i < count_ + sinkOffset_ + 3; i++) {
            data_.sinkFlags[i].setProposedRung(kMaxRung - 1);
        }
    }

    SphPairSourceId* enablePairKickOutputs() {
        pairSourceIds_ = allocate<SphPairSourceId>(count_ + sourceOffset_ + 3);
        data_.sourceIds = pairSourceIds_;
        data_.sourcePairUpdates = allocate<SphPairUpdate>(count_ + sourceOffset_ + 3);
        return pairSourceIds_;
    }

    // Pair kicks are the force output production integrates. With every sink active on one
    // rung, each pair is kicked exactly once per pass, so Δv/Δτ recovers the hydro
    // acceleration: pair forces plus the external kick, never self-gravity. The power-of-two
    // Δτ keeps the float products Δτ a𞁑𞁓 exact; only the compensated sums round.
    static constexpr double kKickTimestep = 0x1p-10;

    SphPairSourceId* enableKickObservation() {
        if (!data_.sourcePairUpdates) {
            enablePairKickOutputs();
            for (int i = 0; i < count_; ++i) {
                pairSourceIds_[sourceIndex(i)] = SphPairSourceId{0, ParticleInteger(i)};
            }
        }
        enableEntropyRateOutput();
        params_.midpointHydro = true;
        params_.hydroKickTimestep = kKickTimestep;
        return pairSourceIds_;
    }

    // Closing passes that need the non-midpoint gravity bound leave the kick arrays idle.
    void disableKickObservation() {
        params_.midpointHydro = false;
        params_.hydroKickTimestep = 0.0;
    }

    void clearPairUpdates() {
        for (int i = 0; i < count_; ++i) {
            data_.sourcePairUpdates[sourceIndex(i)] = {};
        }
    }

    // Δv/Δτ of one sink from the compensated kicks of the last pass.
    SpatialVector<float> kickAcceleration(int index) const {
        require(data_.sourcePairUpdates && params_.hydroKickTimestep > 0.0,
            "kick acceleration requires an enabled kick observation");
        auto const& update = data_.sourcePairUpdates[sourceIndex(index)];
        SpatialVector<float> acceleration;
        for (int dim = 0; dim < kSpatialDimensions; ++dim) {
            acceleration[dim] =
                float(compensatedKickSum(update.velocity[dim]) / params_.hydroKickTimestep);
        }
        return acceleration;
    }

    // (du𞁑/dτ)ₕₑₐₜ of one sink, inverting the kernel's Ȧ𞁑 = (du𞁑/dτ)ₕₑₐₜ / (E/A)(ρ𞁑, γ, a).
    double dissipativeHeating(int index) const {
        require(data_.sinkEntropyRate, "dissipative heating requires the entropy-rate output");
        return sph_entropy::storedEnergy(data_.sinkEntropyRate[sinkIndex(index)],
            data_.sourceDensity[sourceIndex(index)], data_.adiabaticIndex, params_.scaleFactor);
    }

    void enableRungCandidates() {
        data_.sinkRungCandidates = allocate<char>(count_ + particleOffset_ + 3);
    }

    void publishSupport() {
        for (int i = 0; i < count_; i++) {
            data_.sourceSmoothingLength[sourceIndex(i)] = data_.sinkSmoothingLength[sinkIndex(i)];
        }
    }

    void publishDensity() {
        for (int i = 0; i < count_; i++) {
            data_.sourceDensity[sourceIndex(i)] = data_.sinkDensity[sinkIndex(i)];
            data_.sourceBalsaraFactor[sourceIndex(i)] =
                balsara::decodeFactor(data_.sinkBalsaraFactor[sinkIndex(i)]);
            if (data_.sinkGradHCorrection != nullptr) {
                data_.sourceGradHCorrection[sourceIndex(i)] =
                    data_.sinkGradHCorrection[sinkIndex(i)];
            }
        }
    }

    void publishRungs() {
        for (int i = 0; i < count_; i++) {
            data_.sourceRungs[sourceIndex(i)] = data_.sinkParticleRungs[particleIndex(i)];
        }
    }

    SphRunReturn run(int runType) {
        params_.runType = runType;
        *reduction_ = {};
        for (int node = 0; node < data_.numberOfTargetNodes; node++) {
            *reduction_ += sphRunCpuNode(params_, data_, node);
        }
        return *reduction_;
    }

    void solveSmoothing(float minimum, float maximum, float initial) {
        params_.minimumSmoothingLength = minimum;
        params_.maximumSmoothingLength = maximum;
        for (int i = 0; i < count_; i++) {
            data_.sinkFlags[sinkIndex(i)].setConverged(false);
            data_.sourceSmoothingLength[sourceIndex(i)] = data_.sinkSmoothingLength[sinkIndex(i)] =
                initial;
        }
        run(kSphRunSmoothingLengths);
        require(!reduction_->needsAnotherPass,
            "complete neighbor fixture must converge without refresh");
        for (int i = 0; i < count_; i++) {
            require(data_.sinkFlags[sinkIndex(i)].converged(),
                "smoothing solve did not mark convergence");
            requireClose("smoothing preserves the staged source snapshot",
                data_.sourceSmoothingLength[sourceIndex(i)], initial, 0.0);
            float const support = data_.sinkSmoothingLength[sinkIndex(i)];
            require(support >= minimum && support <= maximum,
                "smoothing solve exceeded configured bounds");
        }
        publishSupport();
    }

    void density() {
        run(kSphRunDensityAndGradients);
        require(
            !reduction_->needsAnotherPass, "density fixture unexpectedly needs neighbor refresh");
        for (int i = 0; i < count_; i++) {
            float const density = data_.sinkDensity[sinkIndex(i)];
            require(density > 0.0f && isFinite(density), "positive finite density");
        }
        publishSupport();
        publishDensity();
    }

    void hydro(int phase = 0) {
        params_.phase = phase;
        if (data_.sourcePairUpdates) {
            clearPairUpdates();    // Each call reports the kicks of one pass.
        }
        run(kSphRunHydro);
        for (int i = 0; i < count_; i++) {
            int const sink = sinkIndex(i);
            if (data_.sinkEntropyRate) {
                require(isFinite(data_.sinkEntropyRate[sink]), "finite active-kernel entropy rate");
            }
            if (data_.sourcePairUpdates) {
                auto const& update = data_.sourcePairUpdates[sourceIndex(i)];
                require(isFinite(float(update.entropy)), "finite active-kernel entropy kick");
                for (int dim = 0; dim < kSpatialDimensions; dim++) {
                    require(
                        isFinite(float(update.velocity[dim])), "finite active-kernel pair kick");
                }
            }
            if (phase == 1) {
                // A midpoint pass proposes from minimumProposedRung; a closing pass keeps rung 1.
                int const lowestRung = params_.globalTimestep || params_.midpointHydro ? 0 : 1;
                require(data_.sinkParticleRungs[particleIndex(i)] >= lowestRung &&
                        data_.sinkParticleRungs[particleIndex(i)] < kMaxRung,
                    "valid active-kernel timestep rung");
            }
        }
        publishRungs();
    }

    // Every kicked pair deposits ±Δv once, so the pair kicks of one pass carry no net
    // momentum. The reversible work has no production observable, so the former
    // Σ m v·a + Σ m du/dτ budget is not checked.
    void checkConservation() const {
        require(data_.sourcePairUpdates && params_.hydroKickTimestep > 0.0,
            "momentum check requires an enabled kick observation");
        Vector3<double> momentum{};
        double absoluteForce = 0.0;
        for (int i = 0; i < count_; i++) {
            auto const acceleration = kickAcceleration(i);
            for (int dim = 0; dim < kSpatialDimensions; dim++) {
                momentum[dim] += data_.gasMass * acceleration[dim];
                absoluteForce += std::abs(data_.gasMass * acceleration[dim]);
            }
        }
        for (int dim = 0; dim < kSpatialDimensions; dim++) {
            requireClose("active-kernel total momentum rate", momentum[dim], 0.0,
                3.0e-5 * std::max(1.0, absoluteForce));
        }
    }
};

void initializeViscosityStencil(Fixture& fixture) {
    auto& data = fixture.data_;
    for (int i = 0; i < fixture.count_; ++i) {
        int const source = fixture.sourceIndex(i);
        int const sink = fixture.sinkIndex(i);
        data.sourcePositionX[source] = 0.5 + double(i / 9 - 1) * 0.0625;
        data.sourcePositionY[source] = 0.5 + double(i / 3 % 3 - 1) * 0.0625;
        data.sourcePositionZ[source] = 0.5 + double(i % 3 - 1) * 0.0625;
        data.sourceSmoothingLength[source] = data.sinkSmoothingLength[sink] = 0.25f;
        data.sourceDensity[source] = 1.0f;
        data.sourceInternalEnergy[source] = 0.9f;
        data.sinkFlags[sink].setConverged(true);
    }
}

void testMorrisMonaghanDensity() {
    Fixture fixture(27, 3, 5, 3, 7);
    fixture.enableVariableViscosity();
    initializeViscosityStencil(fixture);
    auto& data = fixture.data_;
    int const center = fixture.sinkIndex(13);
    for (int i = 0; i < fixture.count_; ++i) {
        data.sourceRungs[fixture.sourceIndex(i)] =
            data.sinkParticleRungs[fixture.particleIndex(i)] = i == 13 ? 1 : 0;
    }
    fixture.params_.viscosityTimestep = 0.08;
    // The peculiar compression and shear source is independent of background expansion. The scalar
    // sum retains the finite stencil's SPH consistency error.
    for (float const scaleFactor : {1.0f, 0.4f}) {
        for (bool const globalTimestep : {false, true}) {
            fixture.params_.globalTimestep = globalTimestep;
            for (float const compression : {0.0f, 0.5f, -0.5f}) {
                for (float const expansion : {0.0f, 0.2f}) {
                    fixture.params_.scaleFactor = scaleFactor;
                    fixture.params_.expansionRate = expansion;
                    double density = 0.0;
                    double densityDivergence = 0.0;
                    for (int i = 0; i < fixture.count_; ++i) {
                        int const source = fixture.sourceIndex(i);
                        double const x = data.sourcePositionX[source].toDouble() - 0.5;
                        double const y = data.sourcePositionY[source].toDouble() - 0.5;
                        double const z = data.sourcePositionZ[source].toDouble() - 0.5;
                        data.sourceVelocityX[source] = float(0.25 - compression * x + 0.3 * y);
                        data.sourceVelocityY[source] = float(-0.5 - compression * y);
                        data.sourceVelocityZ[source] = float(0.125 - compression * z);
                        double const radius = std::sqrt(x * x + y * y + z * z);
                        density += data.gasMass * kernelW(radius / 0.25) / std::pow(0.25, 3);
                        if (radius > 0.0) {
                            double const velocityDotRadius =
                                -compression * radius * radius + 0.3 * x * y;
                            densityDivergence -= data.gasMass * velocityDotRadius *
                                dkernelWDq(radius / 0.25) / (std::pow(0.25, 4) * radius);
                        }
                    }
                    double const soundSpeed =
                        std::sqrt(double(data.adiabaticIndex) * (data.adiabaticIndex - 1.0) * 0.9);
                    double const inverseDecayTime = 0.2 * soundSpeed / (scaleFactor * 0.125);
                    double const timeStep = globalTimestep ? 0.08 : 0.04;
                    double const source =
                        std::max(-densityDivergence / (density * scaleFactor), 0.0);
                    double const expected = 0.1 +
                        (double(float(Float16Storage(0.7f))) - 0.1 +
                            timeStep * source * (1.5 - 0.1)) /
                            (1.0 + timeStep * inverseDecayTime + timeStep * source);
                    data.sinkBulkViscosityCoefficient[center] = 0.7f;
                    data.sinkFlags[center].setConverged(false);
                    fixture.densityPass();
                    requireClose("Rosswog-limited M&M backward Euler source and conformal decay",
                        data.sinkBulkViscosityCoefficient[center],
                        float(Float16Storage(float(expected))), 2.0e-5);
                    float const bulkViscosityCoefficient =
                        data.sinkBulkViscosityCoefficient[center];
                    fixture.densityPass();
                    requireClose("repeated density passes advance alpha only once",
                        data.sinkBulkViscosityCoefficient[center], bulkViscosityCoefficient, 0.0);
                    fixture.params_.viscosityTimestep = 0.0;
                    data.sinkFlags[center].setConverged(false);
                    fixture.densityPass();
                    requireClose("initial-time and source refresh passes preserve alpha",
                        data.sinkBulkViscosityCoefficient[center], bulkViscosityCoefficient, 0.0);
                    fixture.params_.viscosityTimestep = 0.08;
                    for (int i = 0; i < fixture.count_; ++i) {
                        require(data.sourceBulkViscosityCoefficient[fixture.sourceIndex(i)] == 0.7f,
                            "density preserves the viscosity source snapshot");
                        if (i != 13) {
                            int const sink = fixture.sinkIndex(i);
                            require(data.sinkFlags[sink].semiactive() &&
                                    float(data.sinkBulkViscosityCoefficient[sink]) ==
                                        float(Float16Storage(0.7f)),
                                "semiactive M&M neighbors retain their viscosity coefficient");
                        }
                    }
                }
            }
        }
    }
}

void testVariableViscosityPairs() {
    Fixture fixture(2, 2);
    fixture.enableVariableViscosity();
    fixture.enableKickObservation();
    auto& data = fixture.data_;
    for (int i = 0; i < 2; ++i) {
        int const source = fixture.sourceIndex(i);
        int const sink = fixture.sinkIndex(i);
        data.sourcePositionX[source] = i == 0 ? 0.35 : 0.65;
        data.sourcePositionY[source] = data.sourcePositionZ[source] = 0.5;
        data.sourceVelocityX[source] = i == 0 ? 0.1f : -0.1f;
        data.sourceDensity[source] = 1.0f + i;
        data.sourceSmoothingLength[source] = i == 0 ? 0.2f : 0.45f;
        data.sourceBulkViscosityCoefficient[source] = i == 0 ? 0.2f : 1.4f;
        data.sinkBulkViscosityCoefficient[sink] = data.sourceBulkViscosityCoefficient[source];
    }
    // Pairs use beta_ab = vonNeumannRichtmyerCoefficient alpha_ab with the mean alpha_ab of the
    // endpoints.
    for (float const vonNeumannRichtmyerCoefficient : {0.0f, 2.0f, 3.5f}) {
        for (int i = 0; i < 2; ++i) {
            data.sourceBulkViscosityCoefficient[fixture.sourceIndex(i)] = i == 0 ? 0.2f : 1.4f;
        }
        fixture.params_.vonNeumannRichtmyerCoefficient = vonNeumannRichtmyerCoefficient;
        fixture.hydro();
        fixture.checkConservation();
        std::array<double, 2> heating;
        std::array<SpatialVector<float>, 2> force;
        for (int i = 0; i < 2; ++i) {
            heating[i] = fixture.dissipativeHeating(i);
            force[i] = fixture.kickAcceleration(i);
        }
        // Both endpoints at the pair mean 0.8 reproduce the same pair coefficients.
        fixture.setBulkViscosityCoefficient(0.8f);
        fixture.hydro();
        fixture.checkConservation();
        for (int i = 0; i < 2; ++i) {
            requireClose("variable pair alpha and beta heating", heating[i],
                fixture.dissipativeHeating(i), 1.0e-5);
            auto const acceleration = fixture.kickAcceleration(i);
            for (int dim = 0; dim < kSpatialDimensions; ++dim) {
                requireClose("symmetric variable pair alpha and beta force", force[i][dim],
                    acceleration[dim], 1.0e-5);
            }
        }
    }
}

void initializeLattice(Fixture& fixture, int side) {
    int i = 0;
    for (int ix = 0; ix < side; ix++) {
        for (int iy = 0; iy < side; iy++) {
            for (int iz = 0; iz < side; iz++) {
                fixture.data_.sourcePositionX[i] = (ix + 0.5) / side;
                fixture.data_.sourcePositionY[i] = (iy + 0.5) / side;
                fixture.data_.sourcePositionZ[i] = (iz + 0.5) / side;
                i++;
            }
        }
    }
}

void testPeriodicSupport() {
    Fixed32Range box;
    Vector3<Position> source;
    for (int dim = 0; dim < kDimensions; dim++) {
        box.begin[dim] = 0.49;
        box.end[dim] = 0.51;
        source[dim] = 0.5;
    }
    box.begin[kXi] = 0.90;
    box.end[kXi] = 0.95;
    source[kXi] = 0.40;
    require(sphSupportIntersects(box, source, 0.49f),
        "periodic support reaches the far end of a sink box");
    require(!sphSupportIntersects(box, source, 0.40f),
        "periodic support excludes a genuinely separated box");
    source[kXi] = 0.02;
    require(sphSupportIntersects(box, source, 0.10f), "periodic support crosses the box boundary");
}

// Bound-layer helpers of the variational reference below.
double targetKernelSum(Fixture const& fixture);
enum class Layer { Lower, Upper };
void requireLayerSupports(Fixture const& fixture, Layer layer);
double referenceStiffness(Fixture const& fixture, double support);

// A 6³ lattice: the free support of 32 neighbors, 0.33, lies below the upper layer of the
// half-box bound 0.49 (which starts at 0.392) and above bounds of 0.24 and below.
void testLattice() {
    constexpr int kSide = 6;
    Fixture fixture(kSide * kSide * kSide);
    initializeLattice(fixture, kSide);
    fixture.solveSmoothing(0.1f, 0.24f, 0.2f);
    requireLayerSupports(fixture, Layer::Upper);
    fixture.solveSmoothing(0.43f, 0.49f, 0.48f);
    requireLayerSupports(fixture, Layer::Lower);
    fixture.solveSmoothing(0.3f, 0.3f, 0.17f);
    for (int i = 0; i < fixture.count_; i++) {
        requireClose("fixed support bound", fixture.data_.sourceSmoothingLength[i], 0.3f, 1.0e-7);
    }
    fixture.solveSmoothing(0.1f, 0.49f, 0.2f);
    fixture.density();
    for (int i = 0; i < fixture.count_; i++) {
        double const support = fixture.data_.sourceSmoothingLength[i];
        double const count = 4.0 * kPi / 3.0 * fixture.data_.sourceDensity[i] * support * support *
            support / fixture.data_.gasMass;
        requireClose("weighted neighbor target", count, fixture.data_.targetNeighborCount, 2.0e-4);
        requireClose("uniform lattice density", fixture.data_.sourceDensity[i],
            fixture.data_.sourceDensity[0], 2.0e-5);
        requireClose(
            "uniform lattice continuum density", fixture.data_.sourceDensity[i], 1.0, 0.03);
    }
    fixture.enableKickObservation();
    fixture.hydro(1);
    for (int i = 0; i < fixture.count_; i++) {
        requireClose(
            "uniform gas at rest generates no entropy", fixture.dissipativeHeating(i), 0.0, 2.0e-5);
        auto const acceleration = fixture.kickAcceleration(i);
        for (int dim = 0; dim < kSpatialDimensions; dim++) {
            requireClose("uniform gas force", acceleration[dim], 0.0, 2.0e-4);
        }
    }
    require(fixture.reduction_->inverseSoundTimestep > 0.0f &&
            fixture.reduction_->inverseViscosityTimestep > 0.0f,
        "active timestep computes sound and viscosity restrictions");

    // A smooth periodic perturbation exercises nonzero work and force for every target.
    for (int i = 0; i < fixture.count_; i++) {
        double const angle = 2.0 * kPi * fixture.data_.sourcePositionX[i].toDouble();
        fixture.data_.sourceVelocityX[i] = 0.02 * std::sin(angle);
        fixture.data_.sourceInternalEnergy[i] = 1.0 + 0.03 * std::cos(angle);
    }
    fixture.hydro();
    fixture.checkConservation();
    double forceNorm = 0.0;
    for (int i = 0; i < fixture.count_; i++) {
        forceNorm += std::abs(fixture.kickAcceleration(i)[kXi]);
    }
    require(forceNorm > 0.1, "perturbed gas must exercise nonzero acceleration");

    // Homogeneous expansion at nonunit a and nonmonatomic gamma changes E, never A.
    fixture.data_.adiabaticIndex = 1.4f;
    fixture.params_.scaleFactor = 0.6f;
    fixture.params_.expansionRate = 0.09f;
    for (int i = 0; i < fixture.count_; i++) {
        fixture.data_.sourceVelocityX[i] = 0.0f;
        fixture.data_.sourceInternalEnergy[i] = 2.0f;
    }
    fixture.density();
    fixture.hydro();
    for (int i = 0; i < fixture.count_; i++) {
        requireClose("homogeneous expansion generates no entropy", fixture.dissipativeHeating(i),
            0.0, 2.0e-6);
    }
}

void testUnequalSupports() {
    Fixture fixture(2, 2);
    auto& data = fixture.data_;
    data.sourcePositionX[0] = 0.35;
    data.sourcePositionX[1] = 0.65;
    for (int i = 0; i < 2; i++) {
        data.sourcePositionY[i] = data.sourcePositionZ[i] = 0.5;
        data.sinkFlags[i].setConverged(true);
        data.sourceInternalEnergy[i] = i == 0 ? 0.7f : 1.2f;
        data.sourceSmoothingLength[i] = data.sinkSmoothingLength[i] = i == 0 ? 0.2f : 0.45f;
        auto& node = data.treeNodes[i];
        for (int dim = 0; dim < kDimensions; dim++) {
            double const position = dim == kXi ? data.sourcePositionX[i].toDouble() : 0.5;
            node.innerBox.begin[dim] = node.innerBox.end[dim] = position;
            node.outerBox.begin[dim] = position - data.sourceSmoothingLength[i];
            node.outerBox.end[dim] = position + data.sourceSmoothingLength[i];
        }
    }
    // Approach before the density pass: its Balsara factor sees pure compression.
    data.sourceVelocityX[0] = 0.1f;
    data.sourceVelocityX[1] = -0.1f;
    fixture.density();
    require(data.sourceDensity[0] != data.sourceDensity[1],
        "unequal supports must produce unequal densities");
    fixture.enableKickObservation();
    fixture.hydro();
    fixture.checkConservation();
    require(fixture.kickAcceleration(0)[kXi] < 0.0f && fixture.kickAcceleration(1)[kXi] > 0.0f,
        "union support must include both directions of repulsive force");
    for (int i = 0; i < 2; i++) {
        require(fixture.dissipativeHeating(i) > 0.0, "approach must add viscosity heating");
    }
    fixture.setBulkViscosityCoefficient(0.0f);
    fixture.params_.vonNeumannRichtmyerCoefficient = 0.0f;
    fixture.hydro();
    for (int i = 0; i < 2; i++) {
        require(fixture.dissipativeHeating(i) == 0.0, "inviscid approach generates no entropy");
    }
    data.sourceVelocityX[0] = -0.1f;
    data.sourceVelocityX[1] = 0.1f;
    fixture.setBulkViscosityCoefficient(1.0f);
    fixture.params_.vonNeumannRichtmyerCoefficient = 2.0f;
    fixture.hydro();
    fixture.checkConservation();
    for (int i = 0; i < 2; i++) {
        requireClose(
            "separating pair has no viscosity heating", fixture.dissipativeHeating(i), 0.0, 2.0e-6);
    }
}

// Exercise the production density and pressure kernels using the selected host
// table through scalar double arithmetic. Core tests also check the table mathematics.
void testSelectedKernel() {
    Fixture fixture(2);
    fixture.params_.maximumSmoothingLength = 0.49f;
    auto& data = fixture.data_;
    data.sourcePositionX[0] = 0.98;
    data.sourcePositionX[1] = 0.16;
    for (int i = 0; i < 2; i++) {
        data.sourcePositionY[i] = data.sourcePositionZ[i] = 0.5;
        data.sinkFlags[i].setConverged(true);
        data.sourceSmoothingLength[i] = data.sinkSmoothingLength[i] = i == 0 ? 0.35f : 0.42f;
        data.sourceInternalEnergy[i] = i == 0 ? 0.7f : 1.2f;
    }
    fixture.setBulkViscosityCoefficient(0.0f);
    fixture.params_.vonNeumannRichtmyerCoefficient = 0.0f;
    fixture.density();
    double const radius = std::abs(distance(data.sourcePositionX[0], data.sourcePositionX[1]));
    double pressureForce = 0.0;
    double inverseCfl = 0.0;
    for (int i = 0; i < 2; i++) {
        double const support = data.sourceSmoothingLength[i];
        double const expectedDensity =
            data.gasMass * (kernelW(0.0) + kernelW(radius / support)) / std::pow(support, 3);
        requireClose(
            "selected kernel scalar/SIMD density", data.sourceDensity[i], expectedDensity, 3.0e-6);
        double const normalizedRadius = radius / support;
        double const derivative = dkernelWDq(normalizedRadius);
        // The second support, 0.42, lies in the upper layer of the 0.49 bound: Omega blends
        // the free value -q w'/(3 sum w) with one by the stiffness psi.
        auto const sums = sph_volume::equalWeightSupportSums<double>(
            kernelW(0.0) + kernelW(normalizedRadius), normalizedRadius * derivative, data.gasMass);
        double const supportJacobian =
            sph_volume::gradHCorrection(sums, referenceStiffness(fixture, support)).supportJacobian;
        double const pressureFactor =
            (data.adiabaticIndex - 1.0f) * data.sourceInternalEnergy[i] / data.sourceDensity[i];
        pressureForce +=
            data.gasMass * pressureFactor / supportJacobian * derivative / std::pow(support, 4);
        double const soundSpeed = std::sqrt(
            data.adiabaticIndex * (data.adiabaticIndex - 1.0f) * data.sourceInternalEnergy[i]);
        inverseCfl = std::max(inverseCfl,
            soundSpeed * selectedHydroKernel().smoothingLengthPerEffectiveRadius / support);
    }
    fixture.enableKickObservation();
    fixture.hydro(1);
    requireClose("selected kernel scalar/SIMD pressure force", fixture.kickAcceleration(0)[kXi],
        pressureForce, 8.0e-6);
    requireClose("selected kernel effective-width CFL", fixture.reduction_->inverseSoundTimestep,
        inverseCfl, 3.0e-6);
    fixture.checkConservation();

    data.sourceVelocityX[0] = 0.1f;
    data.sourceVelocityX[1] = -0.1f;
    fixture.setBulkViscosityCoefficient(1.0f);
    fixture.params_.vonNeumannRichtmyerCoefficient = 2.0f;
    fixture.hydro();
    fixture.checkConservation();
}

void testUnequalPairEos() {
    Fixture fixture(2);
    fixture.params_.maximumSmoothingLength = 0.49f;

    auto& data = fixture.data_;
    // A non-5/3 gamma keeps the cosmological work term nonzero.
    data.adiabaticIndex = 1.4f;
    double const pressureEnergyRatio = data.adiabaticIndex - 1.0;
    data.sourcePositionX[0] = 0.40;
    data.sourcePositionX[1] = 0.58;
    fixture.setBulkViscosityCoefficient(0.0f);
    fixture.params_.vonNeumannRichtmyerCoefficient = 0.0f;
    for (int i = 0; i < 2; ++i) {
        data.sourcePositionY[i] = data.sourcePositionZ[i] = 0.5;
        data.sinkFlags[i].setConverged(true);
        data.sourceSmoothingLength[i] = data.sinkSmoothingLength[i] = i == 0 ? 0.35f : 0.42f;
        data.sourceInternalEnergy[i] = i == 0 ? 0.7f : 1.2f;
    }
    fixture.density();
    double const radius = std::abs(distance(data.sourcePositionX[0], data.sourcePositionX[1]));
    double inverseCfl = 0.0;
    std::array<double, 2> directedGradient{};
    for (int i = 0; i < 2; ++i) {
        double const support = data.sourceSmoothingLength[i];
        double const normalizedRadius = radius / support;
        double const derivative = dkernelWDq(normalizedRadius);
        double const radialGradient = derivative / std::pow(support, 4);
        double const pressureFactor =
            pressureEnergyRatio * data.sourceInternalEnergy[i] / data.sourceDensity[i];
        // Equal masses: zeta = m (1 - 1/Omega), with Omega blended toward one by the
        // stiffness of the 0.42 support inside the upper layer of the 0.49 bound.
        // These unequal supports give distinct, nonzero corrections and gradients.
        auto const sums = sph_volume::equalWeightSupportSums<double>(
            kernelW(0.0) + kernelW(normalizedRadius), normalizedRadius * derivative, data.gasMass);
        double const weightedSupportCorrection =
            sph_volume::gradHCorrection(sums, referenceStiffness(fixture, support))
                .weightedSupportCorrection;
        require(std::abs(weightedSupportCorrection) > 0.1 * data.gasMass,
            "unequal pair exercises a nonzero grad-H correction");
        requireClose("unequal-pair grad-H correction", data.sourceGradHCorrection[i],
            weightedSupportCorrection, 8.0e-6);
        directedGradient[i] =
            pressureFactor * (data.gasMass - weightedSupportCorrection) * radialGradient;
        double const soundSpeed =
            std::sqrt(data.adiabaticIndex * pressureEnergyRatio * data.sourceInternalEnergy[i]);
        inverseCfl = std::max(inverseCfl,
            soundSpeed * selectedHydroKernel().smoothingLengthPerEffectiveRadius / support);
    }
    double const pressureForce = directedGradient[0] + directedGradient[1];
    fixture.enableKickObservation();
    fixture.hydro(1);
    for (int i = 0; i < 2; ++i) {
        requireClose("unequal-pair pressure force", fixture.kickAcceleration(i)[kXi],
            i == 0 ? pressureForce : -pressureForce, 8.0e-6);
    }
    requireClose(
        "unequal-pair sound speed", fixture.reduction_->inverseSoundTimestep, inverseCfl, 3.0e-6);
    fixture.checkConservation();

    // The directed pair work of this approach has no production observable.
    data.sourceVelocityX[0] = 0.1f;
    data.sourceVelocityX[1] = -0.07f;
    fixture.hydro();
    fixture.checkConservation();
    for (int i = 0; i < 2; ++i) {
        data.sourceVelocityX[i] = 0.0f;
    }
    fixture.params_.scaleFactor = 0.6f;
    fixture.params_.expansionRate = 0.09f;
    fixture.hydro();
    for (int i = 0; i < 2; ++i) {
        requireClose("unequal-pair cosmological pressure force", fixture.kickAcceleration(i)[kXi],
            (i == 0 ? pressureForce : -pressureForce) / fixture.params_.scaleFactor, 8.0e-6);
    }
}

void testSelectedCentralGravity() {
    Fixture fixture(1);
    fixture.params_.minimumSmoothingLength = fixture.params_.maximumSmoothingLength = 0.2f;
    auto& data = fixture.data_;
    data.sourcePositionX[0] = 0.6;
    data.sourcePositionY[0] = data.sourcePositionZ[0] = 0.5;
    data.sinkFlags[0].setConverged(true);
    data.centralAcceleration = 0.07f;
    data.centralSmoothingLength = 0.3f;
    fixture.density();
    fixture.enableKickObservation();
    fixture.hydro();
    double const radius = distance(data.sourcePositionX[0], Position(0.5));
    double const expected = -data.centralAcceleration * radius /
        std::pow(double(data.centralSmoothingLength), 3) *
        radial_kernel::forceFactor(selectedGravityKernel(), radius / data.centralSmoothingLength);
    requireClose("selected gravity kernel central force", fixture.kickAcceleration(0)[kXi],
        expected, 5.0e-6);
}

void testFixedGravityTimestep() {
    Fixture fixture(1);
    fixture.params_.minimumSmoothingLength = fixture.params_.maximumSmoothingLength = 0.2f;
    auto& data = fixture.data_;
    data.sourcePositionX[0] = data.sourcePositionY[0] = data.sourcePositionZ[0] = 0.5;
    data.sinkFlags[0].setConverged(true);
    data.sourceInternalEnergy[0] = 1.0e-6f;
    data.selfGravityEnabled = true;
    data.gravityEffectiveRadius = 0.08f / selectedGravityKernel().smoothingLengthPerEffectiveRadius;
    data.sinkGravityAccelerationX[0] = 0.1f;
    fixture.density();
    fixture.hydro(1);
#if COSMICTIGER_WITH_GRAVITY
    double const gravityStep = data.accelerationTimestepFactor *
        std::sqrt(fixture.params_.scaleFactor * data.gravityEffectiveRadius /
            data.sinkGravityAccelerationX[0]);
    int const expectedGravityRung =
        std::max(1, int(std::ceil(std::log2(fixture.params_.baseTimestep / gravityStep))));
    require(fixture.reduction_->maximumGravityRung == expectedGravityRung,
        "gas gravity timestep must use the shared fixed gravity radius");
    data.sourceSmoothingLength[0] = data.sinkSmoothingLength[0] = 0.3f;
    fixture.params_.minimumSmoothingLength = fixture.params_.maximumSmoothingLength = 0.3f;
    fixture.density();
    fixture.hydro(1);
    require(fixture.reduction_->maximumGravityRung == expectedGravityRung,
        "changing gas smoothing must not change the gravity timestep");
#else
    require(fixture.reduction_->maximumGravityRung == 0,
        "SPH-only builds must not report a self-gravity timestep");
    // The isolated particle has no pair or external force, so the closing pass's acceleration
    // bound is exactly zero only when the stored gravity stays out of the acceleration.
    requireClose("SPH-only ignores stored self-gravity",
        fixture.reduction_->inverseAccelerationTimestep, 0.0, 0.0);
    int const hydroRung = fixture.reduction_->maximumHydroRung;
    data.sinkGravityAccelerationX[0] = 1.0e20f;
    data.gravityEffectiveRadius = 1.0e-20f;
    fixture.hydro(1);
    require(fixture.reduction_->maximumGravityRung == 0 &&
            fixture.reduction_->maximumHydroRung == hydroRung &&
            fixture.reduction_->maximumRung == hydroRung,
        "stored self-gravity must not constrain SPH-only timesteps");
    requireClose("SPH-only ignores changed stored self-gravity",
        fixture.reduction_->inverseAccelerationTimestep, 0.0, 0.0);
#endif
}

void testZPlaneAcceleration() {
    // Separate source, sink, and full-particle offsets expose incorrect indexing.
    Fixture fixture(1, 1, 3, 5, 7);
    int const source = fixture.sourceIndex(0);
    int const sink = fixture.sinkIndex(0);
    auto& data = fixture.data_;
    data.sourcePositionX[source] = 0.25;
    data.sourcePositionY[source] = 0.75;
    data.sourcePositionZ[source] = 0.5;
    data.sourceInternalEnergy[source] = 1.0e-6f;
    data.sinkFlags[sink].setConverged(true);
    fixture.density();
    fixture.enableKickObservation();

    // A force with the same magnitude must select the same acceleration timestep
    // regardless of direction. The cold isolated particle has no pressure force.
    fixture.params_.uniformAccelerationY = 0.5f;
    fixture.hydro(1);
    float const referenceInverseTimestep = fixture.reduction_->inverseAccelerationTimestep;
    int const referenceRung = fixture.reduction_->maximumHydroRung;
    require(referenceInverseTimestep > fixture.reduction_->inverseSoundTimestep +
                    fixture.reduction_->inverseViscosityTimestep &&
            referenceRung > 1,
        "external-force fixture must activate acceleration timestep restriction");
    fixture.params_.uniformAccelerationY = 0.0f;
    fixture.params_.zPlaneAcceleration = 0.5f;

    double const tick = std::ldexp(1.0, -32);
    double const positions[] = {0.0, tick, 0.125, 0.5 - tick, 0.5, 0.5 + tick, 0.875, 1.0 - tick};
    double const expectedAccelerations[] = {0.5, 0.5, 0.5, 0.5, 0.0, -0.5, -0.5, -0.5};
    for (size_t i = 0; i < sizeof(positions) / sizeof(positions[0]); i++) {
        data.sourcePositionZ[source] = positions[i];
        fixture.hydro(1);
        auto const acceleration = fixture.kickAcceleration(0);
        requireClose("midplane force has no x component", acceleration[kXi], 0.0, 1.0e-7);
        requireClose("midplane force has no y component", acceleration[kYi], 0.0, 1.0e-7);
        requireClose("midplane force points inward including adjacent fixed coordinates",
            acceleration[kZi], expectedAccelerations[i], 1.0e-7);
        if (expectedAccelerations[i] != 0.0) {
            requireClose("midplane force contributes to acceleration timestep",
                fixture.reduction_->inverseAccelerationTimestep, referenceInverseTimestep, 2.0e-7);
            require(fixture.reduction_->maximumHydroRung == referenceRung,
                "midplane force selects the same rung as equal uniform acceleration");
        } else {
            requireClose("midplane itself has no imposed acceleration timestep restriction",
                fixture.reduction_->inverseAccelerationTimestep, 0.0, 1.0e-7);
        }
    }

    data.sourcePositionZ[source] = 0.75;
    fixture.params_.zPlaneAcceleration = 0.0f;
    fixture.hydro();
    requireClose(
        "zero magnitude disables midplane force", fixture.kickAcceleration(0)[kZi], 0.0, 1.0e-7);

    fixture.params_.zPlaneAcceleration = 0.5f;
    fixture.params_.uniformAccelerationY = 0.25f;
    data.sinkGravityAccelerationX[sink] = 0.125f;
    data.sinkGravityAccelerationY[sink] = -0.0625f;
    data.sinkGravityAccelerationZ[sink] = 0.25f;
    // The external kick never carries self-gravity, which the gravity kick integrates.
    for (bool const selfGravity : {false, true}) {
        data.selfGravityEnabled = selfGravity;
        fixture.hydro();
        auto const acceleration = fixture.kickAcceleration(0);
        requireClose("external kick excludes self gravity in x", acceleration[kXi], 0.0, 1.0e-7);
        requireClose("uniform y force remains independent of midplane force", acceleration[kYi],
            0.25, 1.0e-7);
        requireClose(
            "midplane force adds to the external kick in z", acceleration[kZi], -0.5, 1.0e-7);
    }
    // Self-gravity enters the closing pass's acceleration bound δt_f⁻¹ ∝ √|f𞁑|, so the fourth
    // power of the bound ratio with and without gravity is |f𞁑|² with gravity over |f𞁑|²
    // without: (0.125² + 0.1875² + 0.25²)/(0.25² + 0.5²) = 0.3625.
    fixture.disableKickObservation();
    data.selfGravityEnabled = false;
    fixture.hydro(1);
    double const externalBound = fixture.reduction_->inverseAccelerationTimestep;
    data.selfGravityEnabled = true;
    fixture.hydro(1);
    double const gravityBound = fixture.reduction_->inverseAccelerationTimestep;
    bool const expectSelfGravity = COSMICTIGER_WITH_GRAVITY;
    requireClose("midplane force preserves enabled self gravity",
        std::pow(gravityBound / externalBound, 4), expectSelfGravity ? 0.3625 : 1.0, 4.0e-6);
}

void testDefaultTimestepCalibration() {
    Fixture fixture(1);
    auto& data = fixture.data_;
    data.sourcePositionX[0] = data.sourcePositionY[0] = data.sourcePositionZ[0] = 0.5;
    data.sinkFlags[0].setConverged(true);
    fixture.params_.uniformAccelerationY = 2.0f;
    fixture.params_.courantFactor = 0.25 * std::sqrt(5.0 / 6.0);
    double const oldCentralPotential = 693.0 / 256.0;
    data.accelerationTimestepFactor = 0.25 *
        std::sqrt(selectedGravityKernel().smoothingLengthPerEffectiveRadius / oldCentralPotential);
    fixture.density();
    fixture.hydro(1);
    double const soundSpeed = std::sqrt(
        data.adiabaticIndex * (data.adiabaticIndex - 1.0f) * data.sourceInternalEnergy[0]);
    double const oldSmoothingLength = 0.5 * data.sourceSmoothingLength[0];
    double const expectedSoundStep = 0.25 * oldSmoothingLength /
        (soundSpeed * (1.0 + 0.6 * fixture.params_.maximumBulkViscosityCoefficient));
    double const soundStep = fixture.params_.courantFactor /
        (fixture.reduction_->inverseSoundTimestep + fixture.reduction_->inverseViscosityTimestep);
    requireClose("default cubic sound and viscosity timestep calibration", soundStep,
        expectedSoundStep, 2.0e-7);
    double const expectedAccelerationStep = 0.25 / std::sqrt(oldCentralPotential) *
        std::sqrt(oldSmoothingLength / fixture.params_.uniformAccelerationY);
    double const accelerationStep =
        fixture.params_.courantFactor / fixture.reduction_->inverseAccelerationTimestep;
    requireClose("default cubic acceleration timestep calibration", accelerationStep,
        expectedAccelerationStep, 2.0e-7);
}

void testDefaultThermalTimestep() {
    constexpr float kAdiabaticIndex = 5.0f / 3.0f;
    Fixture fixture(2);
    auto& data = fixture.data_;
    data.adiabaticIndex = kAdiabaticIndex;
    data.sourcePositionX[0] = 0.4;
    data.sourcePositionX[1] = 0.6;
    for (int i = 0; i < 2; i++) {
        data.sourcePositionY[i] = data.sourcePositionZ[i] = 0.5;
        data.sinkFlags[i].setConverged(true);
        data.sourceSmoothingLength[i] = data.sinkSmoothingLength[i] = 0.35f;
        data.sourceInternalEnergy[i] = 1.0e-6f;
        data.sourceVelocityX[i] = i == 0 ? 1.0f : -1.0f;
    }
    fixture.setBulkViscosityCoefficient(0.0f);
    fixture.params_.vonNeumannRichtmyerCoefficient = 0.0f;
    fixture.params_.courantFactor = 0.25 * std::sqrt(5.0 / 6.0);
    double const oldEta = 0.25 / std::sqrt(693.0 / 256.0);
    data.accelerationTimestepFactor =
        oldEta * std::sqrt(selectedGravityKernel().smoothingLengthPerEffectiveRadius);
    fixture.density();
    fixture.enableKickObservation();
    fixture.hydro(1);
    // The reversible work has no output. The scalar pair work of the variational operator,
    // du𞁑/dτ = (Q𞁑𞁓/m𞁑) F𞁑 (v𞁑𞁓 · r𞁑𞁓) with the production ζ𞁑 and ρ𞁑, is the reference
    // for the fractional thermal bound that the closing pass must select.
    double const radius = std::abs(distance(data.sourcePositionX[0], data.sourcePositionX[1]));
    double const velocityDifference = double(data.sourceVelocityX[0]) - data.sourceVelocityX[1];
    for (int i = 0; i < 2; i++) {
        double const support = data.sourceSmoothingLength[i];
        double const oldWidth = 0.5 * support;
        double const soundSpeed =
            std::sqrt(kAdiabaticIndex * (kAdiabaticIndex - 1.0f) * data.sourceInternalEnergy[i]);
        double const acceleration = std::abs(fixture.kickAcceleration(i)[kXi]);
        double const soundStep = 0.25 * oldWidth / soundSpeed;
        double const accelerationStep = oldEta * std::sqrt(oldWidth / acceleration);
        double const directedGradient = (kAdiabaticIndex - 1.0) * data.sourceInternalEnergy[i] /
            data.sourceDensity[i] * (data.gasMass - data.sourceGradHCorrection[i]) *
            dkernelWDq(radius / support) / std::pow(support, 4);
        double const thermalRate = -velocityDifference * directedGradient;
        double const thermalStep = 0.25 * data.sourceInternalEnergy[i] / std::abs(thermalRate);
        require(thermalStep < soundStep && thermalStep < accelerationStep,
            "compression fixture must activate the fractional thermal timestep restriction");
        requireClose("fractional thermal bound selects the hydro timestep",
            fixture.reduction_->minimumTimestep, thermalStep, 1.0e-5);
        int const expectedRung = int(std::ceil(
            std::log2(fixture.params_.baseTimestep / fixture.reduction_->minimumTimestep)));
        require(data.sourceRungs[i] == expectedRung,
            "default fractional thermal bound must preserve the historical timestep rung");
    }
}

void testColdShockTimestep() {
    Fixture fixture(2);
    auto& data = fixture.data_;
    data.sourcePositionX[0] = 0.4;
    data.sourcePositionX[1] = 0.6;
    for (int i = 0; i < 2; ++i) {
        data.sourcePositionY[i] = data.sourcePositionZ[i] = 0.5;
        data.sinkFlags[i].setConverged(true);
        data.sourceSmoothingLength[i] = data.sinkSmoothingLength[i] = 0.35f;
        data.sourceInternalEnergy[i] = 1.0e-20f;
        data.sourceVelocityX[i] = i == 0 ? 1.0f : -1.0f;
    }
    fixture.setBulkViscosityCoefficient(1.0f);
    fixture.params_.vonNeumannRichtmyerCoefficient = 2.0f;
    fixture.density();
    fixture.enableEntropyRateOutput();
    fixture.hydro(1);
    require(fixture.reduction_->minimumTimestep > 1.0e-5,
        "cold viscous shock must remain on a finite CFL timestep");
    for (int i = 0; i < 2; ++i) {
        require(fixture.dissipativeHeating(i) > 0.0,
            "excluding viscous heating from timestep selection must retain actual heating");
    }
}

// A nonmultiple of the SIMD width exercises every masked tail, while unrelated
// staged, SPH-sink, and full-particle offsets exercise all three index spaces.
void testPeriodicTailsAndOffsets() {
    int const count = 2 * kSimdFloatSize + 3;
    Fixture shifted(count, 3, 5, 11, 7);
    Fixture compact(count, 3);
    for (Fixture* fixture : {&shifted, &compact}) {
        auto& data = fixture->data_;
        fixture->enableKickObservation();
        for (int i = 0; i < count; i++) {
            int const source = fixture->sourceIndex(i);
            int const sink = fixture->sinkIndex(i);
            data.sourcePositionX[source] = std::fmod(0.96 + 0.006 * i, 1.0);
            data.sourcePositionY[source] = 0.46 + 0.004 * (i % 5);
            data.sourcePositionZ[source] = 0.48 + 0.003 * (i % 3);
            data.sourceVelocityX[source] = 0.01f * std::sin(float(i));
            data.sourceVelocityY[source] = 0.02f * std::cos(float(i));
            data.sourceInternalEnergy[source] = 0.8f + 0.03f * (i % 7);
            data.sourceSmoothingLength[source] = data.sinkSmoothingLength[sink] =
                0.14f + 0.01f * (i % 3);
            data.sinkFlags[sink].setConverged(true);
        }
        for (int i = 0; i < fixture->sinkOffset_; i++) {
            data.sinkDensity[i] = -91.0f;
            data.sinkEntropyRate[i] = -92.0f;
        }
        data.sinkDensity[fixture->sinkIndex(count)] = -91.0f;
        data.sinkEntropyRate[fixture->sinkIndex(count)] = -92.0f;
        // Pair updates are indexed by the staged source; guard both ends of that range.
        for (int i = 0; i < fixture->sourceOffset_; i++) {
            data.sourcePairUpdates[i].velocity[kXi].add(-93.0f);
        }
        data.sourcePairUpdates[fixture->sourceIndex(count)].velocity[kXi].add(-93.0f);
        fixture->run(kSphRunDensityAndGradients);
        for (int i = 0; i < count; i++) {
            int const source = fixture->sourceIndex(i);
            double const support = data.sourceSmoothingLength[source];
            double expected = 0.0;
            for (int j = 0; j < count; j++) {
                int const neighbor = fixture->sourceIndex(j);
                double const dx =
                    distance(data.sourcePositionX[source], data.sourcePositionX[neighbor]);
                double const dy =
                    distance(data.sourcePositionY[source], data.sourcePositionY[neighbor]);
                double const dz =
                    distance(data.sourcePositionZ[source], data.sourcePositionZ[neighbor]);
                double const radius = std::sqrt(dx * dx + dy * dy + dz * dz);
                expected += data.gasMass * kernelW(radius / support) / std::pow(support, 3);
            }
            requireClose("SIMD tail density agrees with scalar radial table",
                data.sinkDensity[fixture->sinkIndex(i)], expected, 5.0e-6);
            requireClose(
                "density leaves source snapshots unchanged", data.sourceDensity[source], 0.0, 0.0);
        }
        fixture->publishDensity();
        fixture->params_.phase = 1;
        fixture->run(kSphRunHydro);
        for (int i = 0; i < count; i++) {
            require(data.sourceRungs[fixture->sourceIndex(i)] == 1,
                "hydro leaves source rung snapshots unchanged");
        }
        fixture->checkConservation();
        for (int i = 0; i < fixture->sinkOffset_; i++) {
            requireClose("density guard before sinks", data.sinkDensity[i], -91.0, 0.0);
            requireClose("entropy guard before sinks", data.sinkEntropyRate[i], -92.0, 0.0);
        }
        requireClose(
            "density guard after sinks", data.sinkDensity[fixture->sinkIndex(count)], -91.0, 0.0);
        requireClose("entropy guard after sinks", data.sinkEntropyRate[fixture->sinkIndex(count)],
            -92.0, 0.0);
        for (int i = 0; i < fixture->sourceOffset_; i++) {
            requireClose("kick guard before sources",
                float(data.sourcePairUpdates[i].velocity[kXi]), -93.0, 0.0);
        }
        requireClose("kick guard after sources",
            float(data.sourcePairUpdates[fixture->sourceIndex(count)].velocity[kXi]), -93.0, 0.0);
    }
    for (int i = 0; i < count; i++) {
        int const sink = shifted.sinkIndex(i);
        requireClose("offset-invariant density", shifted.data_.sinkDensity[sink],
            compact.data_.sinkDensity[i], 1.0e-6);
        requireClose("offset-invariant heating", shifted.dissipativeHeating(i),
            compact.dissipativeHeating(i), 1.0e-6);
        auto const shiftedAcceleration = shifted.kickAcceleration(i);
        auto const compactAcceleration = compact.kickAcceleration(i);
        for (int dim = 0; dim < kSpatialDimensions; dim++) {
            requireClose("offset-invariant force", shiftedAcceleration[dim],
                compactAcceleration[dim], 1.0e-6);
        }
        require(shifted.data_.sinkParticleRungs[shifted.particleIndex(i)] ==
                compact.data_.sinkParticleRungs[compact.particleIndex(i)],
            "offset-invariant full-particle timestep rung");
    }
}

void testActivity() {
    Fixture fixture(3, 1, 4, 7, 2);
    auto& data = fixture.data_;
    fixture.params_.minimumRung = 4;
    fixture.params_.minimumSmoothingLength = fixture.params_.maximumSmoothingLength = 0.25f;
    fixture.enableKickObservation();
    for (int i = 0; i < 3; i++) {
        int const source = fixture.sourceIndex(i);
        int const sink = fixture.sinkIndex(i);
        data.sourcePositionX[source] = i == 0 ? 0.4 : (i == 1 ? 0.6 : 0.85);
        data.sourcePositionY[source] = data.sourcePositionZ[source] = i == 2 ? 0.85 : 0.5;
        data.sourceRungs[source] = data.sinkParticleRungs[fixture.particleIndex(i)] =
            i == 0 ? 4 : 1;
        data.sinkDensity[sink] = -11.0f;
        data.sinkEntropyRate[sink] = -12.0f;
    }
    // The out-of-range inactive source must receive no recoil either.
    for (int dim = 0; dim < kSpatialDimensions; dim++) {
        data.sourcePairUpdates[fixture.sourceIndex(2)].velocity[dim].add(-13.0f);
    }
    auto result = sphCpuSmoothingLengths(fixture.params_, data, 0);
    require(!result.needsAnotherPass, "active fixed support converges");
    require(data.sinkFlags[fixture.sinkIndex(0)].converged(), "active smoothing is solved");
    require(!data.sinkFlags[fixture.sinkIndex(1)].converged() &&
            !data.sinkFlags[fixture.sinkIndex(2)].converged(),
        "inactive smoothing waits for semiactive classification");
    fixture.publishSupport();
    result = sphCpuDensityAndGradients(fixture.params_, data, 0);
    require(!result.needsAnotherPass, "semiactive fixed support converges");
    require(data.sinkFlags[fixture.sinkIndex(0)].semiactive() &&
            data.sinkFlags[fixture.sinkIndex(1)].semiactive() &&
            !data.sinkFlags[fixture.sinkIndex(2)].semiactive(),
        "only active and interacting inactive particles are semiactive");
    require(data.sinkDensity[fixture.sinkIndex(0)] > 0.0f &&
            data.sinkDensity[fixture.sinkIndex(1)] > 0.0f,
        "active and semiactive density is computed");
    require(data.sinkFlags[fixture.sinkIndex(1)].converged(), "semiactive smoothing is solved");
    requireClose("unrelated inactive density is preserved", data.sinkDensity[fixture.sinkIndex(2)],
        -11.0, 0.0);
    fixture.publishSupport();
    fixture.publishDensity();
    // This source is staged in the same neighbor list as the interacting pair,
    // but its density was not computed because it is inactive and out of range.
    // A mask applied only after pressure division would trap on its zero density.
    data.sourceDensity[fixture.sourceIndex(2)] = 0.0f;
    data.sourceInternalEnergy[fixture.sourceIndex(2)] = 0.0f;
    fixture.params_.phase = 1;
    sphCpuHydro(fixture.params_, data, 0);
    auto const activeAcceleration = fixture.kickAcceleration(0);
    require(activeAcceleration[kXi] < 0.0f, "active particle feels its semiactive neighbor");
    require(isFinite(data.sinkEntropyRate[fixture.sinkIndex(0)]),
        "excluded zero-density neighbor leaves finite heating");
    for (int dim = 0; dim < kSpatialDimensions; dim++) {
        require(isFinite(activeAcceleration[dim]),
            "excluded zero-density neighbor leaves finite force");
    }
    // The active owner kicks its one pair: the semiactive endpoint receives the opposite
    // impulse and the out-of-range source none.
    for (int dim = 0; dim < kSpatialDimensions; dim++) {
        require(compensatedKickSum(data.sourcePairUpdates[fixture.sourceIndex(1)].velocity[dim]) ==
                -compensatedKickSum(data.sourcePairUpdates[fixture.sourceIndex(0)].velocity[dim]),
            "semiactive neighbor misses the owner's recoil");
        requireClose("unrelated inactive kick is preserved",
            float(data.sourcePairUpdates[fixture.sourceIndex(2)].velocity[dim]), -13.0, 0.0);
    }
    for (int i = 1; i < 3; i++) {
        int const sink = fixture.sinkIndex(i);
        requireClose("inactive entropy rate is preserved", data.sinkEntropyRate[sink], -12.0, 0.0);
        require(data.sinkParticleRungs[fixture.particleIndex(i)] == 1,
            "inactive timestep rung is preserved");
    }
}

void testSupportRefresh() {
    Fixture active(1);
    auto& data = active.data_;
    data.sourcePositionX[0] = data.sourcePositionY[0] = data.sourcePositionZ[0] = 0.5;
    active.params_.minimumSmoothingLength = 0.1f;
    active.params_.maximumSmoothingLength = 0.3f;
    data.sourceSmoothingLength[0] = data.sinkSmoothingLength[0] = 0.1f;
    for (int dim = 0; dim < kDimensions; dim++) {
        data.treeNodes[0].outerBox.begin[dim] = 0.39;
        data.treeNodes[0].outerBox.end[dim] = 0.61;
    }
    auto result = sphCpuSmoothingLengths(active.params_, data, 0);
    require(result.needsAnotherPass && !data.sinkFlags[0].converged(),
        "active smoothing requests a neighbor refresh when support grows beyond the search box");
    require(data.sinkSmoothingLength[0] > 0.1f,
        "retry retains the enlarged support for the next search");
    requireClose(
        "retry preserves the source support snapshot", data.sourceSmoothingLength[0], 0.1f, 0.0);
    for (int dim = 0; dim < kDimensions; dim++) {
        data.treeNodes[0].outerBox.begin[dim] = -1.0;
        data.treeNodes[0].outerBox.end[dim] = 2.0;
    }
    result = sphCpuSmoothingLengths(active.params_, data, 0);
    require(!result.needsAnotherPass && data.sinkFlags[0].converged(),
        "active smoothing converges after neighbor refresh");
    // An isolated particle sums only its self term w(0), so its support sits in the upper
    // layer where the bounded target N₀ (1 − t³) equals that sum: t³ = 1 − w(0)/(3N₀/4π).
    double const selfTerm = kernelW(0.0);
    double const layerCoordinate = std::cbrt(1.0 - selfTerm / targetKernelSum(active));
    requireClose("sparse gas settles inside the upper layer of the support ceiling",
        data.sinkSmoothingLength[0], 0.24 + 0.06 * layerCoordinate, 2.0e-4);
    require(data.sinkSmoothingLength[0] < 0.3f, "sparse gas stays below the support ceiling");

    Fixture semiactive(2);
    auto& semiData = semiactive.data_;
    semiactive.params_.minimumRung = 4;
    semiactive.params_.minimumSmoothingLength = 0.1f;
    semiactive.params_.maximumSmoothingLength = 0.3f;
    for (int i = 0; i < 2; i++) {
        semiData.sourcePositionX[i] = 0.5 + 0.05 * i;
        semiData.sourcePositionY[i] = semiData.sourcePositionZ[i] = 0.55;
        semiData.sourceSmoothingLength[i] = semiData.sinkSmoothingLength[i] = 0.1f;
        semiData.sourceRungs[i] = semiData.sinkParticleRungs[semiactive.particleIndex(i)] =
            i == 0 ? 4 : 1;
        semiData.sinkFlags[i].setConverged(i == 0);
        semiData.sinkDensity[i] = -21.0f;
    }
    for (int dim = 0; dim < kDimensions; dim++) {
        semiData.treeNodes[0].outerBox.begin[dim] = 0.43;
        semiData.treeNodes[0].outerBox.end[dim] = 0.67;
    }
    result = sphCpuDensityAndGradients(semiactive.params_, semiData, 0);
    require(result.needsAnotherPass && semiData.sinkFlags[1].semiactive() &&
            !semiData.sinkFlags[1].converged(),
        "semiactive smoothing requests a neighbor refresh");
    requireClose("failed semiactive solve preserves density", semiData.sinkDensity[1], -21.0, 0.0);
    for (int dim = 0; dim < kDimensions; dim++) {
        semiData.treeNodes[0].outerBox.begin[dim] = -1.0;
        semiData.treeNodes[0].outerBox.end[dim] = 2.0;
    }
    result = sphCpuDensityAndGradients(semiactive.params_, semiData, 0);
    require(!result.needsAnotherPass && semiData.sinkFlags[1].converged() &&
            semiData.sinkDensity[1] > 0.0f,
        "semiactive density completes after neighbor refresh");
}

void testRungPropagation() {
    constexpr int kCount = 7;
    Fixture fixture(kCount, kCount, 5, 11, 3);
    auto& data = fixture.data_;
    std::array<int, kCount> previous = {7, 1, 1, 1, 1, 1, 1};
    for (int i = 0; i < kCount; i++) {
        int const source = fixture.sourceIndex(i);
        int const sink = fixture.sinkIndex(i);
        data.sourcePositionX[source] = 0.05 + 0.1 * i;
        data.sourcePositionY[source] = data.sourcePositionZ[source] = 0.5;
        data.sourceSmoothingLength[source] = data.sinkSmoothingLength[sink] = 0.11f;
        data.sourceRungs[source] = data.sinkParticleRungs[fixture.particleIndex(i)] = previous[i];
    }
    int changedPasses = 0;
    bool converged = false;
    for (int pass = 0; pass < kCount + 1; pass++) {
        std::array<int, kCount> expected = previous;
        bool expectedChange = false;
        for (int i = 0; i < kCount; i++) {
            if (i > 0) {
                expected[i] = std::max(expected[i], previous[i - 1] - 1);
            }
            if (i + 1 < kCount) {
                expected[i] = std::max(expected[i], previous[i + 1] - 1);
            }
            expectedChange = expectedChange || expected[i] != previous[i];
        }
        auto const result = fixture.run(kSphRunRungs);
        require(result.needsAnotherPass == expectedChange,
            "rung reduction accurately reports pending changes");
        for (int i = 0; i < kCount; i++) {
            require(data.sourceRungs[fixture.sourceIndex(i)] == previous[i],
                "rung pass preserves its staged source snapshot");
            require(data.sinkParticleRungs[fixture.particleIndex(i)] == expected[i],
                "rungs propagate exactly one neighbor edge per staged pass");
        }
        if (!result.needsAnotherPass) {
            converged = true;
            break;
        }
        changedPasses++;
        fixture.publishRungs();
        previous = expected;
    }
    require(
        converged && changedPasses == 5, "rung chain reaches a fixed point across staged passes");
    for (int i = 0; i < kCount; i++) {
        require(data.sinkParticleRungs[fixture.particleIndex(i)] == 7 - i,
            "converged adjacent rungs differ by at most one");
    }
}

// Variational pressure operator (cosmictiger/sph/variable_smoothing_lengths.hpp). A
// double-precision periodic reference evaluates the same equations at given supports;
// tests/sph/volume.cpp shows that
// those equations are the exact gradient of the discrete thermal energy.
struct VariationalReference {
    std::vector<double> density;
    std::vector<double> weightedSupportCorrection;
    std::vector<double> residual;                 // (sum w - target)/target at the given support
    std::vector<Vector3<double>> acceleration;    // pressure part of dv/dtau
    // Pressure work and homogeneous expansion of dE/dtau. Production exposes no reversible
    // rate, so only the reference integrator below consumes it.
    std::vector<double> energyRate;
};

// A reproducible uniform deviate in [0, 1).
double nextDeviate(uint64_t& state) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return double(state >> 11) * 0x1.0p-53;
}

Vector3<double> periodicSeparation(Fixture const& fixture, int a, int b) {
    Position const* positions[kDimensions] = {fixture.data_.sourcePositionX,
        fixture.data_.sourcePositionY, fixture.data_.sourcePositionZ};
    Vector3<double> separation{};
    for (int dim = 0; dim < kDimensions; dim++) {
        double const difference = positions[dim][fixture.sourceIndex(a)].toDouble() -
            positions[dim][fixture.sourceIndex(b)].toDouble();
        separation[dim] = difference - std::round(difference);
    }
    return separation;
}

double kernelSumAt(Fixture const& fixture, int i, double support) {
    double sum = 0.0;
    for (int j = 0; j < fixture.count_; j++) {
        auto const separation = periodicSeparation(fixture, i, j);
        double const smoothingNormalizedRadius =
            std::sqrt(separation[0] * separation[0] + separation[1] * separation[1] +
                separation[2] * separation[2]) /
            support;
        sum += smoothingNormalizedRadius < 1.0 ?
            radial_kernel::value(selectedHydroKernel(), smoothingNormalizedRadius) :
            0.0;
    }
    return sum;
}

// The interior target 3 N₀/(4π).
double targetKernelSum(Fixture const& fixture) {
    return fixture.data_.targetNeighborCount * 3.0 / (4.0 * kPi);
}

// The production solve's open interval: a margin of each bound layer inside the bounds.
struct SolveLimits {
    sph_volume::SupportLayers<double> layers;
    double lower;
    double upper;
};

SolveLimits solveLimits(Fixture const& fixture) {
    double const minimum = fixture.params_.minimumSmoothingLength;
    double const maximum = fixture.params_.maximumSmoothingLength;
    SolveLimits limits;
    limits.layers = sph_volume::supportLayers(minimum, maximum);
    limits.lower = minimum + sph_volume::kSupportMargin * (limits.layers.lowerJunction - minimum);
    limits.upper = maximum - sph_volume::kSupportMargin * (maximum - limits.layers.upperJunction);
    return limits;
}

// Whether the fixture's grad-H evaluation treats a support as having a constraint:
// strictly inside distinct bounds. Mirrors sphGradHEvaluation.
bool referenceConstrained(Fixture const& fixture, double support) {
    return fixture.params_.minimumSmoothingLength < fixture.params_.maximumSmoothingLength &&
        support > fixture.params_.minimumSmoothingLength &&
        support < fixture.params_.maximumSmoothingLength;
}

// The bounded target 3 N(h)/(4π) at a support, the interior target for a support without
// a constraint.
double referenceTargetSum(Fixture const& fixture, double support) {
    if (!referenceConstrained(fixture, support)) {
        return targetKernelSum(fixture);
    }
    auto const limits = solveLimits(fixture);
    return targetKernelSum(fixture) *
        sph_volume::boundedTarget<double>(support, fixture.params_.minimumSmoothingLength,
            fixture.params_.maximumSmoothingLength, limits.layers)
            .factor;
}

// The support stiffness psi of the grad-H correction: zero without a constraint.
double referenceStiffness(Fixture const& fixture, double support) {
    if (!referenceConstrained(fixture, support)) {
        return 0.0;
    }
    auto const limits = solveLimits(fixture);
    return sph_volume::boundedTarget<double>(support, fixture.params_.minimumSmoothingLength,
        fixture.params_.maximumSmoothingLength, limits.layers)
        .stiffness;
}

// Supports solving sum_j w(r_ij/h) = bounded target to roundoff on the production solve's
// open interval; an iterate pinned at a limit with the bound's residual sign stays there.
std::vector<double> exactSupports(Fixture const& fixture) {
    auto const limits = solveLimits(fixture);
    std::vector<double> supports(fixture.count_);
    for (int i = 0; i < fixture.count_; i++) {
        auto const residual = [&](double support) {
            return kernelSumAt(fixture, i, support) - referenceTargetSum(fixture, support);
        };
        double lower = limits.lower;
        double upper = limits.upper;
        if (residual(lower) >= 0.0) {
            supports[i] = lower;
            continue;
        }
        if (residual(upper) <= 0.0) {
            supports[i] = upper;
            continue;
        }
        for (int iteration = 0; iteration < 200 && upper - lower > 1.0e-15 * upper; iteration++) {
            double const middle = 0.5 * (lower + upper);
            (residual(middle) < 0.0 ? lower : upper) = middle;
        }
        supports[i] = 0.5 * (lower + upper);
    }
    return supports;
}

// Every solved support lies strictly inside the bounds, in the expected layer, and meets
// the bounded target to the solve tolerance unless it is pinned at the open interval's
// limit with the bound's residual sign.
void requireLayerSupports(Fixture const& fixture, Layer layer) {
    auto const limits = solveLimits(fixture);
    for (int i = 0; i < fixture.count_; i++) {
        double const support = fixture.data_.sourceSmoothingLength[fixture.sourceIndex(i)];
        require(support > fixture.params_.minimumSmoothingLength &&
                support < fixture.params_.maximumSmoothingLength,
            "bounded support stays strictly inside the bounds");
        require(layer == Layer::Lower ? support < limits.layers.lowerJunction :
                                        support > limits.layers.upperJunction,
            "bounded support lies in the expected layer");
        double const target = referenceTargetSum(fixture, support);
        double const residual = kernelSumAt(fixture, i, support) - target;
        bool const pinned = (support <= limits.lower && residual >= 0.0) ||
            (support >= limits.upper && residual <= 0.0);
        require(pinned || std::abs(residual) <= 3.0 * kSphSmoothingLengthTolerance * target,
            "bounded support meets the bounded target");
    }
}

VariationalReference variationalReference(
    Fixture const& fixture, std::vector<double> const& support) {
    auto const& data = fixture.data_;
    auto const& table = selectedHydroKernel();
    int const count = fixture.count_;
    double const mass = data.gasMass;
    double const gamma = data.adiabaticIndex;
    double const inverseScaleFactor = 1.0 / fixture.params_.scaleFactor;
    VariationalReference result;
    result.density.resize(count);
    result.weightedSupportCorrection.resize(count);
    result.residual.resize(count);
    result.acceleration.assign(count, {});
    result.energyRate.assign(count, 0.0);
    std::vector<double> pressureFactor(count);
    auto const gradient = [&](double radius, double smoothingLength) {
        double const smoothingNormalizedRadius = radius / smoothingLength;
        return smoothingNormalizedRadius < 1.0 ?
            sph_volume::kernelGradientFactor(
                radial_kernel::derivative(table, smoothingNormalizedRadius), 1.0 / smoothingLength,
                radius) :
            0.0;
    };
    for (int i = 0; i < count; i++) {
        auto sums = sph_volume::emptySupportSums<double>();
        for (int j = 0; j < count; j++) {
            auto const separation = periodicSeparation(fixture, i, j);
            double const smoothingNormalizedRadius =
                std::sqrt(separation[0] * separation[0] + separation[1] * separation[1] +
                    separation[2] * separation[2]) /
                support[i];
            if (smoothingNormalizedRadius < 1.0) {
                sph_volume::accumulateSupportSums(sums, mass,
                    radial_kernel::value(table, smoothingNormalizedRadius),
                    radial_kernel::derivative(table, smoothingNormalizedRadius),
                    smoothingNormalizedRadius);
            }
        }
        result.density[i] = sph_volume::weightedDensity(sums, 1.0 / support[i]);
        double const target = referenceTargetSum(fixture, support[i]);
        result.residual[i] = (sums.numberSum - target) / target;
        result.weightedSupportCorrection[i] =
            sph_volume::gradHCorrection(sums, referenceStiffness(fixture, support[i]))
                .weightedSupportCorrection;
        // (1/mᵢ) Pᵢ xᵢ/yᵢ^2 with x = m.
        pressureFactor[i] =
            (gamma - 1.0) * data.sourceInternalEnergy[fixture.sourceIndex(i)] / result.density[i];
    }
    // The kernels' single-precision expansion coefficient, exactly zero for gamma = 5/3.
    float const expansionCoefficient = 5.0f - 3.0f * data.adiabaticIndex;
    for (int i = 0; i < count; i++) {
        int const sourceI = fixture.sourceIndex(i);
        for (int j = 0; j < count; j++) {
            if (i == j) {
                continue;
            }
            int const sourceJ = fixture.sourceIndex(j);
            auto const separation = periodicSeparation(fixture, i, j);
            double const radius = std::sqrt(separation[0] * separation[0] +
                separation[1] * separation[1] + separation[2] * separation[2]);
            double const gradientI = gradient(radius, support[i]);
            double const gradientJ = gradient(radius, support[j]);
            double const coefficientIJ =
                pressureFactor[i] * (mass - result.weightedSupportCorrection[i]);
            double const coefficientJI =
                pressureFactor[j] * (mass - result.weightedSupportCorrection[j]);
            double const velocityDotSeparation =
                (data.sourceVelocityX[sourceI] - data.sourceVelocityX[sourceJ]) * separation[0] +
                (data.sourceVelocityY[sourceI] - data.sourceVelocityY[sourceJ]) * separation[1] +
                (data.sourceVelocityZ[sourceI] - data.sourceVelocityZ[sourceJ]) * separation[2];
            for (int dim = 0; dim < kDimensions; dim++) {
                result.acceleration[i][dim] -= inverseScaleFactor *
                    (coefficientIJ * gradientI + coefficientJI * gradientJ) * separation[dim];
            }
            result.energyRate[i] +=
                inverseScaleFactor * coefficientIJ * gradientI * velocityDotSeparation;
        }
        result.energyRate[i] += expansionCoefficient * fixture.params_.expansionRate *
            inverseScaleFactor * data.sourceInternalEnergy[sourceI];
    }
    return result;
}

// Largest production/reference acceleration difference over the largest reference
// acceleration, through the pair kicks of the last pass.
double compareAccelerations(Fixture const& fixture, VariationalReference const& reference) {
    double accelerationScale = 0.0;
    double accelerationError = 0.0;
    for (int i = 0; i < fixture.count_; i++) {
        auto const acceleration = fixture.kickAcceleration(i);
        for (int dim = 0; dim < kSpatialDimensions; dim++) {
            accelerationScale =
                std::max(accelerationScale, std::abs(reference.acceleration[i][dim]));
            accelerationError = std::max(
                accelerationError, std::abs(acceleration[dim] - reference.acceleration[i][dim]));
        }
    }
    return accelerationError / accelerationScale;
}

void perturbLattice(Fixture& fixture, int side, uint64_t seed) {
    initializeLattice(fixture, side);
    uint64_t state = seed;
    for (int i = 0; i < fixture.count_; i++) {
        int const source = fixture.sourceIndex(i);
        double const spacing = 1.0 / side;
        fixture.data_.sourcePositionX[source] = fixture.data_.sourcePositionX[source].toDouble() +
            0.3 * spacing * (nextDeviate(state) - 0.5);
        fixture.data_.sourcePositionY[source] = fixture.data_.sourcePositionY[source].toDouble() +
            0.3 * spacing * (nextDeviate(state) - 0.5);
        fixture.data_.sourcePositionZ[source] = fixture.data_.sourcePositionZ[source].toDouble() +
            0.3 * spacing * (nextDeviate(state) - 0.5);
        fixture.data_.sourceVelocityX[source] = 0.1 * (nextDeviate(state) - 0.5);
        fixture.data_.sourceVelocityY[source] = 0.1 * (nextDeviate(state) - 0.5);
        fixture.data_.sourceVelocityZ[source] = 0.1 * (nextDeviate(state) - 0.5);
        fixture.data_.sourceInternalEnergy[source] = 1.0 + 0.4 * nextDeviate(state);
    }
}

// Production float rates against the double reference: at the production supports the
// difference is roundoff; at exactly solved supports it measures the 1e-4 solve tolerance.
// The comoving rates carry 1/a and the homogeneous (5 - 3 gamma) term at a = 0.7.
void testVariationalAgainstReference() {
    constexpr int kSide = 6;
    Fixture fixture(kSide * kSide * kSide);
    perturbLattice(fixture, kSide, 2026);
    fixture.setBulkViscosityCoefficient(0.0f);
    fixture.params_.vonNeumannRichtmyerCoefficient = 0.0f;
    fixture.data_.adiabaticIndex = 1.4f;
    fixture.params_.scaleFactor = 0.7f;
    fixture.params_.expansionRate = 0.05f;
    fixture.solveSmoothing(0.05f, 0.49f, 0.3f);
    fixture.density();
    SphRunReturn const densityPass = *fixture.reduction_;
    require(densityPass.heldSupports == 0, "perturbed lattice supports are free");
    require(densityPass.maximumSupportResidual <= 1.2f * kSphSmoothingLengthTolerance,
        "density pass reports the support-solve residual");
    require(densityPass.minimumGradHOmega > 0.5f && densityPass.illConditionedSupports == 0,
        "perturbed lattice corrections are well conditioned");
    fixture.enableKickObservation();
    fixture.hydro();
    std::vector<double> production(fixture.count_);
    for (int i = 0; i < fixture.count_; i++) {
        production[i] = fixture.data_.sourceSmoothingLength[fixture.sourceIndex(i)];
    }
    auto const reference = variationalReference(fixture, production);
    for (int i = 0; i < fixture.count_; i++) {
        requireClose("variational zeta", fixture.data_.sinkGradHCorrection[fixture.sinkIndex(i)],
            reference.weightedSupportCorrection[i], 2.0e-5 * fixture.data_.gasMass);
        requireClose("variational density", fixture.data_.sinkDensity[fixture.sinkIndex(i)],
            reference.density[i], 2.0e-6);
    }
    double const roundoff = compareAccelerations(fixture, reference);
    std::printf("variational float/double acceleration at production supports: %.3e\n", roundoff);
    require(roundoff < 2.0e-5,
        "production variational acceleration matches the double reference at the same supports");
    double const exact =
        compareAccelerations(fixture, variationalReference(fixture, exactSupports(fixture)));
    std::printf("variational float acceleration against exactly solved supports: %.3e (solve "
                "tolerance %.1e)\n",
        exact, double(kSphSmoothingLengthTolerance));
    require(exact < 2.0e-3, "support-solve tolerance bounds the variational acceleration error");
}

// An isolated cluster in the periodic box: linear and angular momentum of the pair kicks with
// viscosity, and held supports at hₘₐₓ. The Hubble-flow work identity of free supports has no
// production observable since the reversible rate left the kernel outputs.
void testVariationalIsolatedCluster() {
    constexpr int kSide = 4;
    Fixture fixture(kSide * kSide * kSide);
    uint64_t state = 77;
    int i = 0;
    for (int ix = 0; ix < kSide; ix++) {
        for (int iy = 0; iy < kSide; iy++) {
            for (int iz = 0; iz < kSide; iz++) {
                int const source = fixture.sourceIndex(i++);
                fixture.data_.sourcePositionX[source] =
                    0.36 + 0.09 * ix + 0.02 * nextDeviate(state);
                fixture.data_.sourcePositionY[source] =
                    0.36 + 0.09 * iy + 0.02 * nextDeviate(state);
                fixture.data_.sourcePositionZ[source] =
                    0.36 + 0.09 * iz + 0.02 * nextDeviate(state);
                fixture.data_.sourceInternalEnergy[source] = 1.0 + 0.5 * nextDeviate(state);
            }
        }
    }
    // The interior supports of the cluster, 0.18, stay below the upper layer of the 0.25
    // bound (which starts at 0.2); the edge and corner supports enter it.
    fixture.solveSmoothing(0.05f, 0.25f, 0.15f);
    fixture.density();
    require(
        fixture.reduction_->heldSupports > 0 && fixture.reduction_->heldSupports < fixture.count_,
        "cluster edge supports lie in the upper layer of the maximum");
    std::vector<double> supports(fixture.count_);
    for (int k = 0; k < fixture.count_; k++) {
        supports[k] = fixture.data_.sourceSmoothingLength[fixture.sourceIndex(k)];
        require(
            supports[k] < fixture.params_.maximumSmoothingLength, "no support reaches the maximum");
    }
    auto const reference = variationalReference(fixture, supports);
    int referenceHeld = 0;
    for (int k = 0; k < fixture.count_; k++) {
        float const productionZeta = fixture.data_.sinkGradHCorrection[fixture.sinkIndex(k)];
        if (referenceStiffness(fixture, supports[k]) < 1.0) {
            referenceHeld++;
        }
        require(productionZeta != 0.0f, "no support is held");
        requireClose("bounded-support zeta", productionZeta, reference.weightedSupportCorrection[k],
            2.0e-5 * fixture.data_.gasMass);
    }
    require(referenceHeld == fixture.reduction_->heldSupports,
        "production and reference count the same layer supports");
    // Random velocities with viscosity: momentum and angular momentum of the pair kicks.
    fixture.setBulkViscosityCoefficient(1.0f);
    fixture.params_.vonNeumannRichtmyerCoefficient = 2.0f;
    for (int k = 0; k < fixture.count_; k++) {
        int const source = fixture.sourceIndex(k);
        fixture.data_.sourceVelocityX[source] = 0.3 * (nextDeviate(state) - 0.5);
        fixture.data_.sourceVelocityY[source] = 0.3 * (nextDeviate(state) - 0.5);
        fixture.data_.sourceVelocityZ[source] = 0.3 * (nextDeviate(state) - 0.5);
    }
    fixture.enableKickObservation();
    fixture.hydro();
    fixture.checkConservation();
    Vector3<double> torque{};
    double torqueScale = 0.0;
    for (int k = 0; k < fixture.count_; k++) {
        int const source = fixture.sourceIndex(k);
        auto const a = fixture.kickAcceleration(k);
        Vector3<double> const pairDistance{fixture.data_.sourcePositionX[source].toDouble() - 0.5,
            fixture.data_.sourcePositionY[source].toDouble() - 0.5,
            fixture.data_.sourcePositionZ[source].toDouble() - 0.5};
        Vector3<double> const moment{pairDistance[1] * a[2] - pairDistance[2] * a[1],
            pairDistance[2] * a[0] - pairDistance[0] * a[2],
            pairDistance[0] * a[1] - pairDistance[1] * a[0]};
        for (int dim = 0; dim < kDimensions; dim++) {
            torque[dim] += fixture.data_.gasMass * moment[dim];
            torqueScale += std::abs(fixture.data_.gasMass * moment[dim]);
        }
    }
    for (int dim = 0; dim < kDimensions; dim++) {
        requireClose("variational angular momentum rate", torque[dim], 0.0, 3.0e-5 * torqueScale);
    }
}

// Bounded supports: every lattice support in the lower layer, in the upper layer, and
// fixed. Layer supports carry the stiffness-blended zeta of the reference; fixed supports
// use zeta = 0.
void testVariationalHeldSupports() {
    constexpr int kSide = 6;
    Fixture fixture(kSide * kSide * kSide);
    initializeLattice(fixture, kSide);
    struct Bounds {
        float minimum;
        float maximum;
        float initial;
    };
    for (auto const bounds :
        {Bounds{0.43f, 0.49f, 0.48f}, Bounds{0.1f, 0.24f, 0.2f}, Bounds{0.3f, 0.3f, 0.17f}}) {
        fixture.solveSmoothing(bounds.minimum, bounds.maximum, bounds.initial);
        for (int i = 0; i < fixture.count_; i++) {
            fixture.data_.sinkGradHCorrection[fixture.sinkIndex(i)] = -7.0f;
        }
        fixture.density();
        require(fixture.reduction_->heldSupports == fixture.count_,
            "every support is in a layer or fixed");
        bool const fixed = bounds.minimum == bounds.maximum;
        std::vector<double> supports(fixture.count_);
        for (int i = 0; i < fixture.count_; i++) {
            supports[i] = fixture.data_.sourceSmoothingLength[fixture.sourceIndex(i)];
        }
        if (!fixed) {
            requireLayerSupports(fixture, bounds.minimum > 0.4f ? Layer::Lower : Layer::Upper);
        }
        auto const reference = variationalReference(fixture, supports);
        for (int i = 0; i < fixture.count_; i++) {
            float const zeta = fixture.data_.sinkGradHCorrection[fixture.sinkIndex(i)];
            if (fixed) {
                require(zeta == 0.0f, "fixed supports use zeta = 0");
            } else {
                require(zeta != 0.0f, "layer supports keep a grad-H correction");
                requireClose("layer-support zeta", zeta, reference.weightedSupportCorrection[i],
                    2.0e-5 * fixture.data_.gasMass);
            }
        }
    }
    uint64_t state = 5;
    for (int i = 0; i < fixture.count_; i++) {
        int const source = fixture.sourceIndex(i);
        fixture.data_.sourceVelocityX[source] = 0.2 * (nextDeviate(state) - 0.5);
        fixture.data_.sourceVelocityY[source] = 0.2 * (nextDeviate(state) - 0.5);
        fixture.data_.sourceInternalEnergy[source] = 1.0 + 0.5 * nextDeviate(state);
    }
    fixture.density();
    fixture.enableKickObservation();
    fixture.hydro();
    fixture.checkConservation();
    std::vector<double> support(fixture.count_, fixture.params_.minimumSmoothingLength);
    fixture.setBulkViscosityCoefficient(0.0f);
    fixture.params_.vonNeumannRichtmyerCoefficient = 0.0f;
    fixture.hydro();
    double const difference = compareAccelerations(fixture, variationalReference(fixture, support));
    require(difference < 2.0e-5, "fixed supports reproduce the variational pressure force");
}

// A uniform free lattice has no pressure force, and the homogeneous expansion of E leaves A
// unchanged.
void testVariationalUniformLattice() {
    constexpr int kSide = 6;
    Fixture fixture(kSide * kSide * kSide);
    initializeLattice(fixture, kSide);
    fixture.enableKickObservation();
    for (float const gamma : {5.0f / 3.0f, 1.4f}) {
        fixture.data_.adiabaticIndex = gamma;
        fixture.params_.scaleFactor = 0.6f;
        fixture.params_.expansionRate = 0.09f;
        for (int i = 0; i < fixture.count_; i++) {
            fixture.data_.sourceInternalEnergy[fixture.sourceIndex(i)] = 2.0f;
        }
        fixture.solveSmoothing(0.1f, 0.49f, 0.2f);
        fixture.density();
        require(fixture.reduction_->heldSupports == 0, "uniform lattice supports are free");
        fixture.hydro(1);
        for (int i = 0; i < fixture.count_; i++) {
            requireClose("homogeneous expansion generates no entropy",
                fixture.dissipativeHeating(i), 0.0, 2.0e-5);
            auto const acceleration = fixture.kickAcceleration(i);
            for (int dim = 0; dim < kSpatialDimensions; dim++) {
                requireClose("uniform variational force", acceleration[dim], 0.0, 2.0e-4);
            }
        }
    }
}

// Synchronous kick-drift-kick stepping over the production kernels, as in the global
// integrator: every particle is active and the supports, densities and corrections are
// rebuilt before every force pass. The state is kept in double so that time-integration
// error is not hidden by float accumulation; the kernels read float snapshots. Rates are
// conformal-time rates of the stored w = a v_pec and E = a^2 u, and the comoving drift is
// dx/dtau = w/a. The acceleration is the production pair kick Δv/Δτ; production integrates
// entropy and publishes no reversible work rate, so dE/dtau comes from the double reference of
// the same variational equations at the production supports.
class GlobalStepper {
    Fixture& fixture_;
    std::vector<Vector3<double>> acceleration_;
    std::vector<double> energyRate_;

    double scaleFactor(double time) const {
        return cosmological_ ? std::pow(time / initialTime_, 2) : 1.0;
    }

    double expansionRate(double time) const {
        // a = (tau/tau0)^2, the Einstein-de Sitter law in conformal time.
        return cosmological_ ? 2.0 * scaleFactor(time) / time : 0.0;
    }

    void publish() {
        auto& data = fixture_.data_;
        for (int i = 0; i < fixture_.count_; i++) {
            int const source = fixture_.sourceIndex(i);
            Position* positions[kDimensions] = {
                data.sourcePositionX, data.sourcePositionY, data.sourcePositionZ};
            float* velocities[kDimensions] = {
                data.sourceVelocityX, data.sourceVelocityY, data.sourceVelocityZ};
            for (int dim = 0; dim < kDimensions; dim++) {
                double const wrapped = position_[i][dim] - std::floor(position_[i][dim]);
                positions[dim][source] = wrapped;
                velocities[dim][source] = velocity_[i][dim];
            }
            data.sourceInternalEnergy[source] = energy_[i];
        }
    }

public:
    std::vector<Vector3<double>> position_;
    std::vector<Vector3<double>> velocity_;
    std::vector<double> energy_;
    double time_;
    double initialTime_;
    bool cosmological_;
    double expansionWork_ = 0.0;    // integral of (5 - 3 gamma)(da/dtau)/a sum m E dtau

    explicit GlobalStepper(Fixture& fixture, bool cosmological = false, double initialTime = 1.0)
      : fixture_(fixture)
      , acceleration_(fixture.count_)
      , energyRate_(fixture.count_)
      , position_(fixture.count_)
      , velocity_(fixture.count_)
      , energy_(fixture.count_)
      , time_(initialTime)
      , initialTime_(initialTime)
      , cosmological_(cosmological) {
        auto const& data = fixture.data_;
        for (int i = 0; i < fixture.count_; i++) {
            int const source = fixture.sourceIndex(i);
            position_[i] = {data.sourcePositionX[source].toDouble(),
                data.sourcePositionY[source].toDouble(), data.sourcePositionZ[source].toDouble()};
            velocity_[i] = {data.sourceVelocityX[source], data.sourceVelocityY[source],
                data.sourceVelocityZ[source]};
            energy_[i] = data.sourceInternalEnergy[source];
        }
        fixture_.enableKickObservation();
        evaluate();
    }

    void evaluate() {
        auto& data = fixture_.data_;
        fixture_.params_.scaleFactor = scaleFactor(time_);
        fixture_.params_.expansionRate = expansionRate(time_);
        publish();
        for (int i = 0; i < fixture_.count_; i++) {
            data.sinkFlags[fixture_.sinkIndex(i)].setConverged(false);
        }
        fixture_.run(kSphRunSmoothingLengths);
        require(!fixture_.reduction_->needsAnotherPass, "stepping fixture supports converge");
        fixture_.publishSupport();
        fixture_.density();
        fixture_.hydro();
        // The energy budgets of the tests below therefore compare the production force with
        // the reference work.
        std::vector<double> supports(fixture_.count_);
        for (int i = 0; i < fixture_.count_; i++) {
            supports[i] = data.sourceSmoothingLength[fixture_.sourceIndex(i)];
        }
        auto const reference = variationalReference(fixture_, supports);
        for (int i = 0; i < fixture_.count_; i++) {
            auto const a = fixture_.kickAcceleration(i);
            acceleration_[i] = {a[kXi], a[kYi], a[kZi]};
            energyRate_[i] = reference.energyRate[i];
        }
    }

    // The kernels' single-precision coefficient, which is exactly zero for gamma = 5/3.
    double expansionPower() const {
        float const coefficient = 5.0f - 3.0f * fixture_.data_.adiabaticIndex;
        double thermal = 0.0;
        for (double const energy : energy_) {
            thermal += fixture_.data_.gasMass * energy;
        }
        return coefficient * expansionRate(time_) / scaleFactor(time_) * thermal;
    }

    void kick(double interval) {
        for (int i = 0; i < fixture_.count_; i++) {
            for (int dim = 0; dim < kDimensions; dim++) {
                velocity_[i][dim] += acceleration_[i][dim] * interval;
            }
            energy_[i] += energyRate_[i] * interval;
            require(energy_[i] > 0.0, "stepping keeps internal energy positive");
        }
    }

    // The closing half-kick is first predicted with the old rates, the velocity-dependent
    // rates are evaluated at the predicted state, and the corrector replaces the predicted
    // kick. This reference integrator tests the force formulation independently
    // of the production pair-midpoint driver.
    void step(double interval) {
        double const powerBefore = expansionPower();
        kick(0.5 * interval);
        double const drift = interval / scaleFactor(time_ + 0.5 * interval);
        for (int i = 0; i < fixture_.count_; i++) {
            for (int dim = 0; dim < kDimensions; dim++) {
                position_[i][dim] += velocity_[i][dim] * drift;
            }
        }
        time_ += interval;
        auto const predictedAcceleration = acceleration_;
        auto const predictedEnergyRate = energyRate_;
        kick(0.5 * interval);
        evaluate();
        for (int i = 0; i < fixture_.count_; i++) {
            for (int dim = 0; dim < kDimensions; dim++) {
                velocity_[i][dim] +=
                    (acceleration_[i][dim] - predictedAcceleration[i][dim]) * 0.5 * interval;
            }
            energy_[i] += (energyRate_[i] - predictedEnergyRate[i]) * 0.5 * interval;
            require(energy_[i] > 0.0, "stepping keeps internal energy positive");
        }
        expansionWork_ += 0.5 * interval * (powerBefore + expansionPower());
    }

    // sum_i mᵢ (wᵢ^2/2 + Eᵢ) of the stored variables.
    double storedEnergy() const {
        double total = 0.0;
        for (int i = 0; i < fixture_.count_; i++) {
            total += fixture_.data_.gasMass *
                (0.5 *
                        (velocity_[i][0] * velocity_[i][0] + velocity_[i][1] * velocity_[i][1] +
                            velocity_[i][2] * velocity_[i][2]) +
                    energy_[i]);
        }
        return total;
    }

    double kineticEnergy() const {
        double total = 0.0;
        for (int i = 0; i < fixture_.count_; i++) {
            total += 0.5 * fixture_.data_.gasMass *
                (velocity_[i][0] * velocity_[i][0] + velocity_[i][1] * velocity_[i][1] +
                    velocity_[i][2] * velocity_[i][2]);
        }
        return total;
    }

    // Aᵢ = (gamma - 1) uᵢ / rho_i^(gamma - 1) from the last density pass.
    double entropy(int i) const {
        double const gamma = fixture_.data_.adiabaticIndex;
        return (gamma - 1.0) * energy_[i] /
            std::pow(double(fixture_.data_.sourceDensity[fixture_.sourceIndex(i)]), gamma - 1.0);
    }
};

// A plane sound wave rho'/rho = v/c = delta sin(k x) on an 8^3 lattice, inviscid, with c = 1.
void initializeSoundWave(Fixture& fixture, int side, double amplitude, double gamma) {
    initializeLattice(fixture, side);
    double const k = 2.0 * kPi;
    double const internalEnergy = 1.0 / (gamma * (gamma - 1.0));
    fixture.data_.adiabaticIndex = float(gamma);
    fixture.setBulkViscosityCoefficient(0.0f);
    fixture.params_.vonNeumannRichtmyerCoefficient = 0.0f;
    for (int i = 0; i < fixture.count_; i++) {
        int const source = fixture.sourceIndex(i);
        double const x = fixture.data_.sourcePositionX[source].toDouble();
        fixture.data_.sourcePositionX[source] = x + amplitude / k * std::cos(k * x);
        fixture.data_.sourceVelocityX[source] = amplitude * std::sin(k * x);
        fixture.data_.sourceInternalEnergy[source] =
            internalEnergy * std::pow(1.0 + amplitude * std::sin(k * x), gamma - 1.0);
    }
    fixture.solveSmoothing(0.05f, 0.49f, 0.25f);
}

struct WaveRun {
    double energyError;     // max |Delta sum m(v^2/2 + u)| over the wave's kinetic energy
    double entropyError;    // max_i |Delta Aᵢ| / Aᵢ at the end
    double profileError;    // L2 error of vₓ after half a period, over the amplitude
};

WaveRun runSoundWave(int steps) {
    constexpr int kSide = 8;
    constexpr double kAmplitude = 0.05;
    double const gamma = 5.0 / 3.0;
    Fixture fixture(kSide * kSide * kSide);
    initializeSoundWave(fixture, kSide, kAmplitude, gamma);
    GlobalStepper stepper(fixture);
    std::vector<double> initialEntropy(fixture.count_);
    for (int i = 0; i < fixture.count_; i++) {
        initialEntropy[i] = stepper.entropy(i);
    }
    double const initialEnergy = stepper.storedEnergy();
    double const waveEnergy = stepper.kineticEnergy();
    double const halfPeriod = 0.5;
    WaveRun run{};
    for (int step = 0; step < steps; step++) {
        stepper.step(halfPeriod / steps);
        run.energyError = std::max(
            run.energyError, std::abs(stepper.storedEnergy() - initialEnergy) / waveEnergy);
    }
    double squared = 0.0;
    for (int i = 0; i < fixture.count_; i++) {
        run.entropyError =
            std::max(run.entropyError, std::abs(stepper.entropy(i) / initialEntropy[i] - 1.0));
        // After half a period the right-moving wave is the Eulerian field v = -delta sin(k x),
        // evaluated at each particle's current position.
        double const expected = -kAmplitude * std::sin(2.0 * kPi * stepper.position_[i][0]);
        squared += std::pow(stepper.velocity_[i][0] - expected, 2);
    }
    run.profileError = std::sqrt(squared / fixture.count_) / kAmplitude;
    return run;
}

// Kick-drift-kick energy error falls as dt^2. The wave profile error is
// the spatial error at eight particles per wavelength; the entropy error is limited by the
// support-solve tolerance once the time error is small.
void testSoundWaveTimestepRefinement() {
    std::array<WaveRun, 3> runs{};
    int const steps[] = {12, 24, 48};
    for (int k = 0; k < 3; k++) {
        runs[k] = runSoundWave(steps[k]);
        std::printf("variational sound wave, %d steps per half period: energy %.3e, entropy %.3e, "
                    "profile %.3e\n",
            steps[k], runs[k].energyError, runs[k].entropyError, runs[k].profileError);
    }
    for (int k = 1; k < 3; k++) {
        require(runs[k].energyError < 0.35 * runs[k - 1].energyError,
            "kick-drift-kick energy error converges at second order");
    }
    require(runs[2].energyError < 2.0e-3, "refined sound-wave energy error");
    require(runs[2].profileError < 0.15, "sound wave propagates at the sound speed");
    // Each particle's work is its own adiabatic rate, so the entropy error is a
    // time error until the support-solve tolerance is reached.
    require(runs[2].entropyError < 0.5 * runs[1].entropyError && runs[2].entropyError < 2.0e-4,
        "variational sound-wave entropy error converges with the timestep");
}

// A large-amplitude smooth compression, stopped before a shock forms: without viscosity the
// variational particle entropies change only through the time and support-solve errors.
void testAdiabaticCompression() {
    constexpr int kSide = 8;
    std::array<double, 2> entropyErrors{};
    std::array<double, 2> energyErrors{};
    int const steps[] = {20, 40};
    for (int k = 0; k < 2; k++) {
        Fixture fixture(kSide * kSide * kSide);
        initializeSoundWave(fixture, kSide, 0.0, 1.4);
        for (int i = 0; i < fixture.count_; i++) {
            int const source = fixture.sourceIndex(i);
            fixture.data_.sourceVelocityX[source] =
                -0.3 * std::sin(2.0 * kPi * fixture.data_.sourcePositionX[source].toDouble());
        }
        GlobalStepper stepper(fixture);
        std::vector<double> initialEntropy(fixture.count_);
        double maximumDensity = 0.0;
        for (int i = 0; i < fixture.count_; i++) {
            initialEntropy[i] = stepper.entropy(i);
        }
        double const initialEnergy = stepper.storedEnergy();
        double const kinetic = stepper.kineticEnergy();
        for (int step = 0; step < steps[k]; step++) {
            stepper.step(0.2 / steps[k]);
            energyErrors[k] = std::max(
                energyErrors[k], std::abs(stepper.storedEnergy() - initialEnergy) / kinetic);
        }
        for (int i = 0; i < fixture.count_; i++) {
            entropyErrors[k] =
                std::max(entropyErrors[k], std::abs(stepper.entropy(i) / initialEntropy[i] - 1.0));
            maximumDensity = std::max(
                maximumDensity, double(fixture.data_.sourceDensity[fixture.sourceIndex(i)]));
        }
        std::printf("variational compression, %d steps: maximum density %.3f, entropy %.3e, "
                    "energy %.3e\n",
            steps[k], maximumDensity, entropyErrors[k], energyErrors[k]);
        require(maximumDensity > 1.2, "the compression is nonlinear");
    }
    require(energyErrors[1] < 0.35 * energyErrors[0], "compression energy error converges");
    require(entropyErrors[1] < 3.0e-3, "variational compression is adiabatic");
}

// Comoving sound waves in an Einstein-de Sitter background, a = (tau/tau0)^2, from a = 1 to
// a = 4. The pair rates exchange stored kinetic and thermal energy exactly, so
// sum m (w^2/2 + E) changes only by the integral of (5 - 3 gamma)(da/dtau)/a sum m E,
// which vanishes for gamma = 5/3. The residual converges at second order. With gamma = 5/3
// these variables evolve as a static gas in ds = dtau/a, so the early, large steps in s
// dominate the error.
void testCosmologicalExpansion() {
    constexpr int kSide = 8;
    for (double const gamma : {5.0 / 3.0, 1.4}) {
        std::array<double, 3> residuals{};
        std::array<double, 3> uniformErrors{};
        int const steps[] = {40, 80, 160};
        for (int k = 0; k < 3; k++) {
            Fixture fixture(kSide * kSide * kSide);
            initializeSoundWave(fixture, kSide, 0.05, gamma);
            GlobalStepper stepper(fixture, true, 1.0);
            double const initialEnergy = stepper.storedEnergy();
            double const kinetic = stepper.kineticEnergy();
            // The kick-drift-kick energy error oscillates; its maximum over the run converges.
            // It is measured against the wave energy or the homogeneous expansion work.
            for (int step = 0; step < steps[k]; step++) {
                stepper.step(1.0 / steps[k]);
                residuals[k] = std::max(residuals[k],
                    std::abs(stepper.storedEnergy() - initialEnergy - stepper.expansionWork_));
            }
            residuals[k] /= std::max(kinetic, std::abs(stepper.expansionWork_));
            // A uniform lattice follows E proportional to a^(5 - 3 gamma).
            Fixture uniform(kSide * kSide * kSide);

            initializeSoundWave(uniform, kSide, 0.0, gamma);
            GlobalStepper background(uniform, true, 1.0);
            double const initialInternal = background.energy_[0];
            for (int step = 0; step < steps[k]; step++) {
                background.step(1.0 / steps[k]);
            }
            double const expected = initialInternal * std::pow(4.0, 5.0 - 3.0 * gamma);
            uniformErrors[k] = std::abs(background.energy_[0] / expected - 1.0);
            std::printf("variational EdS expansion gamma %.4f, %d steps: energy residual %.3e, "
                        "uniform E error %.3e\n",
                gamma, steps[k], residuals[k], uniformErrors[k]);
        }
        for (int k = 1; k < 3; k++) {
            require(residuals[k] < 0.35 * residuals[k - 1] || residuals[k] < 1.0e-6,
                "cosmological energy budget converges at second order");
            if (gamma != 5.0 / 3.0) {
                require(uniformErrors[k] < 0.35 * uniformErrors[k - 1],
                    "homogeneous expansion converges at second order");
            }
        }
        require(residuals[2] < 2.0e-3, "refined cosmological energy budget");
        require(uniformErrors[2] < (gamma == 5.0 / 3.0 ? 1.0e-6 : 2.0e-3),
            "uniform lattice follows the adiabatic expansion law");
    }
}

// Corrections are rebuilt by each density pass and never carried over: a NaN left on a
// source outside every support is masked before arithmetic, even with FP traps; a NaN on an
// interacting source stops the force pass.
void testVariationalStaleCorrections() {
    Fixture fixture(3);

    fixture.data_.targetNeighborCount = 1.5f;
    double const positions[3][3] = {{0.5, 0.5, 0.5}, {0.56, 0.5, 0.5}, {0.9, 0.1, 0.2}};
    for (int i = 0; i < 3; i++) {
        int const source = fixture.sourceIndex(i);
        fixture.data_.sourcePositionX[source] = positions[i][0];
        fixture.data_.sourcePositionY[source] = positions[i][1];
        fixture.data_.sourcePositionZ[source] = positions[i][2];
        fixture.data_.sourceSmoothingLength[source] =
            fixture.data_.sinkSmoothingLength[fixture.sinkIndex(i)] = 0.1f;
        fixture.data_.sinkFlags[fixture.sinkIndex(i)].setConverged(true);
    }
    fixture.params_.minimumSmoothingLength = fixture.params_.maximumSmoothingLength = 0.1f;
    fixture.density();
    float const notANumber = std::numeric_limits<float>::quiet_NaN();
    fixture.data_.sourceGradHCorrection[fixture.sourceIndex(2)] = notANumber;
    fixture.enableKickObservation();
    fixture.hydro();
    fixture.checkConservation();
    fixture.data_.sourceGradHCorrection[fixture.sourceIndex(1)] = notANumber;
    bool stopped = false;
    try {
        fixture.run(kSphRunHydro);
    } catch (std::runtime_error const&) {
        stopped = true;
    }
    require(stopped, "a stale interacting correction stops the variational force pass");
}

}    // namespace detail

namespace detail {
class SphCpuTest : public ::testing::Test {
protected:
    void SetUp() override {
        captured_ = std::fegetenv(&savedEnvironment_) == 0;
        ASSERT_TRUE(captured_);
        std::feclearexcept(FE_ALL_EXCEPT);
#if defined(__linux__) && defined(__GLIBC__)
        ASSERT_NE(feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW), -1);
#endif
        setKernelSelections(
            {radial_kernel::Family::Ferrers, 8}, {radial_kernel::Family::Spline, 3}, false);
    }

    void TearDown() override {
        if (captured_) {
            std::fesetenv(&savedEnvironment_);
        }
    }

private:
    std::fenv_t savedEnvironment_{};
    bool captured_{};
};
}    // namespace detail

using detail::SphCpuTest;

TEST_F(SphCpuTest, MorrisMonaghanDensity) {
    detail::testMorrisMonaghanDensity();
}

TEST_F(SphCpuTest, VariableViscosityPairs) {
    detail::testVariableViscosityPairs();
}

TEST_F(SphCpuTest, BalsaraVelocityGradients) {
    detail::Fixture fixture(27, 3, 5, 3, 7);
    fixture.enableVariableViscosity();
    auto& data = fixture.data_;
    struct Flow {
        float compression;
        float rotation;
        float shear;
        float expansion;
        float expected;
    };
    Flow const flows[] = {
        {0, 0, 0, 0, 0},       // Uniform translation, including the zero-gradient case.
        {0, 0, 1, 0, 0},       // Pure shear.
        {0, 1, 0, 0, 0},       // Solid-body rotation.
        {1, 0, 0, 0, 1},       // Compression retains full viscosity.
        {1, 1, 0, 0, 0.6f},    // |div| = 3, |curl| = 2.
        {0, 0, 0, -1, 0},      // Homogeneous expansion does not change peculiar gradients.
        {1, 1, 0, 0.5f, 0.6f},
    };
    for (double const viscosityTimestep : {0.0, 0.08}) {
        fixture.params_.viscosityTimestep = viscosityTimestep;
        for (auto const& flow : flows) {
            fixture.params_.expansionRate = flow.expansion;
            for (int i = 0; i < fixture.count_; ++i) {
                int const source = fixture.sourceIndex(i);
                int const sink = fixture.sinkIndex(i);
                float const x = float(i / 9 - 1) * 0.0625f;
                float const y = float(i / 3 % 3 - 1) * 0.0625f;
                float const z = float(i % 3 - 1) * 0.0625f;
                data.sourcePositionX[source] = 0.5 + x;
                data.sourcePositionY[source] = 0.5 + y;
                data.sourcePositionZ[source] = 0.5 + z;
                data.sourceVelocityX[source] =
                    0.25f - flow.compression * x - flow.rotation * y + flow.shear * y;
                data.sourceVelocityY[source] = -0.5f - flow.compression * y + flow.rotation * x;
                data.sourceVelocityZ[source] = 0.125f - flow.compression * z;
                data.sourceSmoothingLength[source] = data.sinkSmoothingLength[sink] = 0.25f;
                data.sinkFlags[sink].setConverged(true);
                data.sourceBalsaraFactor[source] = -7.0f;
            }
            fixture.run(kSphRunDensityAndGradients);
            for (int i = 0; i < fixture.count_; ++i) {
                float const factor =
                    balsara::decodeFactor(data.sinkBalsaraFactor[fixture.sinkIndex(i)]);
                EXPECT_GE(factor, 0.0f);
                EXPECT_LE(factor, 1.0f);
                EXPECT_EQ(data.sourceBalsaraFactor[fixture.sourceIndex(i)], -7.0f);
            }
            EXPECT_EQ(data.sinkBalsaraFactor[fixture.sinkIndex(13)],
                balsara::encodeFactor(flow.expected));
            EXPECT_EQ(data.sinkBalsaraFactor[fixture.sinkOffset_ - 1], 0);
            EXPECT_EQ(data.sinkBalsaraFactor[fixture.sinkIndex(fixture.count_)], 0);
        }
    }
}

TEST_F(SphCpuTest, BalsaraSymmetricForceAndHeating) {
    detail::Fixture fixture(2, 2);
    auto& data = fixture.data_;
    for (int i = 0; i < 2; ++i) {
        data.sourcePositionX[i] = i == 0 ? 0.35 : 0.65;
        data.sourcePositionY[i] = data.sourcePositionZ[i] = 0.5;
        data.sourceVelocityX[i] = i == 0 ? 0.1f : -0.1f;
        data.sourceDensity[i] = 1.0f + i;
        data.sourceSmoothingLength[i] = i == 0 ? 0.2f : 0.45f;
    }
    fixture.enableKickObservation();
    fixture.hydro();
    std::array<double, 2> const fullHeating = {
        fixture.dissipativeHeating(0), fixture.dissipativeHeating(1)};
    std::array<sph_geometry::SpatialVector<float>, 2> const fullForce = {
        fixture.kickAcceleration(0), fixture.kickAcceleration(1)};
    fixture.setBulkViscosityCoefficient(0.0f);
    fixture.params_.vonNeumannRichtmyerCoefficient = 0.0f;
    fixture.hydro();
    std::array<sph_geometry::SpatialVector<float>, 2> const pressureForce = {
        fixture.kickAcceleration(0), fixture.kickAcceleration(1)};
    for (int i = 0; i < 2; ++i) {
        EXPECT_EQ(fixture.dissipativeHeating(i), 0.0);    // the pressure force is reversible
    }
    fixture.setBulkViscosityCoefficient(1.0f);
    fixture.params_.vonNeumannRichtmyerCoefficient = 2.0f;
    for (int i = 0; i < 2; ++i) {
        data.sourceBalsaraFactor[i] = i == 0 ? 0.2f : 0.8f;
    }
    fixture.hydro();
    fixture.checkConservation();
    // The mean Balsara factor 0.5 halves the viscous force and the viscous heating.
    for (int i = 0; i < 2; ++i) {
        EXPECT_GT(fullHeating[i], 0.0);
        EXPECT_NEAR(fixture.dissipativeHeating(i), 0.5 * fullHeating[i], 2.0e-6);
        auto const acceleration = fixture.kickAcceleration(i);
        for (int dim = 0; dim < kSpatialDimensions; ++dim) {
            EXPECT_NEAR(acceleration[dim],
                pressureForce[i][dim] + 0.5f * (fullForce[i][dim] - pressureForce[i][dim]),
                2.0e-6f);
        }
    }
}

TEST_F(SphCpuTest, EveryPhaseSubmitsItsArithmeticCount) {
    // The partial final packet verifies that actual SIMD work is counted, even
    // when only one lane in that packet contains a particle.
    detail::Fixture fixture(SimdFloat::size() + 1);
    fixture.params_.minimumSmoothingLength = 0.2f;
    fixture.params_.maximumSmoothingLength = 0.2f;
    fixture.params_.phase = 1;
    for (int i = 0; i < fixture.count_; ++i) {
        auto const source = fixture.sourceIndex(i);
        fixture.data_.sourcePositionX[source] = 0.4f + 0.01f * i;
        fixture.data_.sourcePositionY[source] = 0.5f;
        fixture.data_.sourcePositionZ[source] = 0.5f;
        fixture.data_.sourceDensity[source] = 1.0f;
    }
    for (int runType :
        {kSphRunSmoothingLengths, kSphRunDensityAndGradients, kSphRunHydro, kSphRunRungs}) {
        detail::submittedSphFlops = {};
        auto const result = fixture.run(runType);
        EXPECT_GT(detail::submittedSphFlops.singlePrecision, 0) << "phase " << runType;
        EXPECT_GT(detail::submittedSphFlops.doublePrecision, 0) << "phase " << runType;
        EXPECT_EQ(detail::submittedSphFlops.unclassified, 0) << "phase " << runType;
        EXPECT_EQ(result.floatingPointOperations, double(detail::submittedSphFlops.total()))
            << "phase " << runType;
        fixture.publishSupport();
        fixture.publishDensity();
        fixture.publishRungs();
    }
}

TEST_F(SphCpuTest, DensityAndRungsCountPaddedPackets) {
    int const width = SimdFloat::size();
    detail::Fixture fixture(width + 1);
    fixture.params_.minimumSmoothingLength = fixture.params_.maximumSmoothingLength = 0.2f;
    // Every neighbor passes all three dimensions of the first periodic image:
    // six fixed-point image conversions per source. Two SIMD packets execute
    // per sink, including the unused lanes of the final packet.
    detail::submittedSphFlops = {};
    auto const rungs = fixture.run(kSphRunRungs);
    EXPECT_EQ(rungs.floatingPointOperations, double(fixture.count_) * (7 + 18 * width));
    EXPECT_EQ(detail::submittedSphFlops.singlePrecision, fixture.count_ * (1 + 18 * width));
    EXPECT_EQ(detail::submittedSphFlops.doublePrecision, 6 * fixture.count_);
    EXPECT_EQ(detail::submittedSphFlops.unclassified, 0);
    // Cubic density evaluates both Bernstein spans in every lane: each span
    // has six interpolation steps (18 operations), four weights, one scalar
    // width subtraction. Include distance/normalization, density accumulation,
    // four scalar setup operations and a width-1 additive reduction per sink.
    // Grad-h adds both quadratic derivative spans, three sums per lane, two
    // reductions, and twenty-four scalar operations for its layers, bounded target,
    // residual and correction.
    // Float tables add four scalar FP32 width subtractions per packet;
    // there are no table precision conversions. Geometry still contributes FP64.
    // The single target node also computes gamma - 1 once.
    // The always-on velocity derivatives add per packet the derivative spans again (26 per
    // lane, two scalar subtractions), the gradient coefficient and relative velocity (6 per
    // lane), the gradient vector (3), divergence (6) and curl (12); per sink they add the
    // sound speed (3), four width-1 reductions and the Balsara factor (12): 110 W + 15.
    // The per-sink mass times inverse gradient scale (1 + kInverseGradientScale = 3) saves
    // one product per lane in both packets, and h_M/h is formed once per call.
    detail::submittedSphFlops = {};
    auto const density = fixture.run(kSphRunDensityAndGradients);
    EXPECT_EQ(density.floatingPointOperations, double(fixture.count_) * (57 + 281 * width) + 2);
    EXPECT_EQ(detail::submittedSphFlops.singlePrecision, fixture.count_ * (51 + 281 * width) + 2);
    EXPECT_EQ(detail::submittedSphFlops.doublePrecision, 6 * fixture.count_);
    EXPECT_EQ(detail::submittedSphFlops.unclassified, 0);
}

TEST_F(SphCpuTest, DisabledGravityTimestepDiagnosticsAreExcluded) {
    detail::Fixture fixture(2);
    fixture.params_.phase = 1;
    for (int i = 0; i < fixture.count_; ++i) {
        fixture.data_.sourceDensity[fixture.sourceIndex(i)] = 1.0f;
    }
    auto const withoutGravity = fixture.run(kSphRunHydro);
    fixture.data_.selfGravityEnabled = true;
    auto const withGravity = fixture.run(kSphRunHydro);
#if COSMICTIGER_WITH_GRAVITY
    // The same gravity limit and rung expressions execute in both cases, but
    // constrain the evolution only when self-gravity is enabled. Otherwise
    // their 13 arithmetic operations produce diagnostics and must not enter
    // the count. The enabled minimum also promotes its float timestep to double.
    EXPECT_EQ(withGravity.floatingPointOperations - withoutGravity.floatingPointOperations,
        14.0 * fixture.count_);
#else
    EXPECT_EQ(withGravity.floatingPointOperations, withoutGravity.floatingPointOperations);
    EXPECT_EQ(withGravity.maximumGravityRung, 0);
    EXPECT_EQ(withoutGravity.maximumGravityRung, 0);
#endif
}

TEST_F(SphCpuTest, PeriodicSupport) {
    detail::testPeriodicSupport();
}

TEST_F(SphCpuTest, PeriodicTailsAndOffsets) {
    detail::testPeriodicTailsAndOffsets();
}

TEST_F(SphCpuTest, Activity) {
    detail::testActivity();
}

TEST_F(SphCpuTest, SupportRefresh) {
    detail::testSupportRefresh();
}

TEST_F(SphCpuTest, RungPropagation) {
    detail::testRungPropagation();
}

TEST_F(SphCpuTest, InactiveRungCandidateClosure) {
    detail::Fixture fixture(7, 7, 5, 11, 3);
    detail::testInactiveRungCandidateClosure(
        fixture, [&] { return fixture.run(kSphRunRungs).needsAnotherPass; });
}

TEST_F(SphCpuTest, SupportBoundaryRungCandidates) {
    detail::Fixture fixture(2, 2, 5, 11, 3);
    detail::testSupportBoundaryRungCandidates(
        fixture, [&] { return fixture.run(kSphRunRungs).needsAnotherPass; });
}

TEST_F(SphCpuTest, Lattice) {
    detail::testLattice();
}

TEST_F(SphCpuTest, InitialSmoothingDensityMatchesKernelSum) {
    constexpr int kSide = 5;
    for (auto const bounds : {std::array<float, 2>{0.1f, 0.49f}, std::array<float, 2>{0.1f, 0.24f},
             std::array<float, 2>{0.43f, 0.49f}, std::array<float, 2>{0.3f, 0.3f}}) {
        detail::Fixture fixture(kSide * kSide * kSide);
        detail::initializeLattice(fixture, kSide);
        fixture.params_.initializeDensity = true;
        fixture.solveSmoothing(bounds[0], bounds[1], 0.2f);
        std::vector<float> initialDensities(fixture.count_);
        for (int i = 0; i < fixture.count_; ++i) {
            initialDensities[i] = fixture.data_.sinkDensity[fixture.sinkIndex(i)];
            EXPECT_GT(initialDensities[i], 0.0f);
        }
        // The independent production density pass sums m W rather than using
        // the target neighbor count. Agreement must also hold at either bound.
        fixture.params_.initializeDensity = false;
        fixture.density();
        for (int i = 0; i < fixture.count_; ++i) {
            EXPECT_NEAR(initialDensities[i], fixture.data_.sinkDensity[fixture.sinkIndex(i)],
                2.0e-6f * initialDensities[i]);
        }
        fixture.solveSmoothing(bounds[0], bounds[1], 0.2f);
        for (int i = 0; i < fixture.count_; ++i) {
            // Ordinary evolution must keep the previous density until its
            // density/gradient pass; initializing density is strictly opt-in.
            EXPECT_EQ(fixture.data_.sinkDensity[fixture.sinkIndex(i)],
                fixture.data_.sourceDensity[fixture.sourceIndex(i)]);
        }
    }
}

TEST_F(SphCpuTest, UnequalSupports) {
    detail::testUnequalSupports();
}

TEST_F(SphCpuTest, ZPlaneAcceleration) {
    detail::testZPlaneAcceleration();
}

TEST_F(SphCpuTest, DefaultTimestepCalibration) {
    detail::testDefaultTimestepCalibration();
}

TEST_F(SphCpuTest, DefaultThermalTimestep) {
    detail::testDefaultThermalTimestep();
}

TEST_F(SphCpuTest, UnequalPairEos) {
    detail::testUnequalPairEos();
}

TEST_F(SphCpuTest, SelectedKernels) {
    radial_kernel::KernelSpec const specifications[] = {
        {radial_kernel::Family::Spline, 2},
        {radial_kernel::Family::Spline, 3},
        {radial_kernel::Family::Spline, 4},
        {radial_kernel::Family::Spline, 5},
        {radial_kernel::Family::Spline, 6},
        {radial_kernel::Family::Spline, 7},
        {radial_kernel::Family::Spline, 8},
        {radial_kernel::Family::Spline, 9},
        {radial_kernel::Family::Wendland, 2},
        {radial_kernel::Family::Wendland, 5},
        {radial_kernel::Family::Wendland, 8},
        {radial_kernel::Family::Ferrers, 2},
        {radial_kernel::Family::Ferrers, 4},
        {radial_kernel::Family::Ferrers, 6},
        {radial_kernel::Family::Ferrers, 8},
    };
    size_t const count = sizeof(specifications) / sizeof(specifications[0]);
    for (size_t i = 0; i < count; i++) {
        auto const hydro = specifications[i];
        auto const gravity = specifications[count - i - 1];
        setKernelSelections(gravity, hydro, false);
        detail::testSelectedKernel();
        detail::testSelectedCentralGravity();
        detail::testFixedGravityTimestep();
    }
}

TEST_F(SphCpuTest, VariationalAgainstReference) {
    detail::testVariationalAgainstReference();
}

TEST_F(SphCpuTest, VariationalIsolatedCluster) {
    detail::testVariationalIsolatedCluster();
}

TEST_F(SphCpuTest, VariationalHeldSupports) {
    detail::testVariationalHeldSupports();
}

TEST_F(SphCpuTest, VariationalUniformLattice) {
    detail::testVariationalUniformLattice();
}

TEST_F(SphCpuTest, VariationalStaleCorrections) {
    detail::testVariationalStaleCorrections();
}

TEST_F(SphCpuTest, SoundWaveTimestepRefinement) {
    detail::testSoundWaveTimestepRefinement();
}

TEST_F(SphCpuTest, AdiabaticCompression) {
    detail::testAdiabaticCompression();
}

TEST_F(SphCpuTest, VariationalCosmologicalExpansion) {
    detail::testCosmologicalExpansion();
}

TEST_F(SphCpuTest, GlobalRawTimestep) {
    detail::testGlobalRawTimestep<detail::Fixture>();
}

TEST_F(SphCpuTest, DeferredSourceCompletion) {
    detail::testDeferredSourceCompletion<detail::Fixture>();
}

TEST_F(SphCpuTest, MidpointHydroUsesExactRungWithoutGravityStorage) {
    detail::Fixture fixture(4, 2, 2, 3, 4);
    detail::testMidpointHydroTargets(fixture);
}

TEST_F(SphCpuTest, EndpointMorrisMonaghanUsesCompletedIntervalsAndClosingRungs) {
    detail::Fixture fixture(3, 1, 2, 3, 4);
    detail::testEndpointMorrisInterval(fixture);
}

TEST_F(SphCpuTest, ConservativeMixedRungPairKicks) {
    detail::Fixture fixture(2, 2, 2, 4, 6);
    detail::testConservativePairKick(fixture, false);
}

TEST_F(SphCpuTest, ConservativeMorrisMonaghanPairKicks) {
    detail::Fixture fixture(2, 2, 2, 4, 6);
    detail::testConservativePairKick(fixture, true);
}

TEST_F(SphCpuTest, ColdShockTimestep) {
    detail::testColdShockTimestep();
}

TEST_F(SphCpuTest, ColdShockThermalTimestep) {
    detail::Fixture fixture(2);
    auto& data = fixture.data_;
    data.sourcePositionX[0] = 0.4;
    data.sourcePositionX[1] = 0.6;
    for (int i = 0; i < 2; ++i) {
        data.sourcePositionY[i] = data.sourcePositionZ[i] = 0.5;
        data.sinkFlags[i].setConverged(true);
        data.sourceSmoothingLength[i] = data.sinkSmoothingLength[i] = 0.35f;
        data.sourceInternalEnergy[i] = 1.0e-20f;
        data.sourceVelocityX[i] = i == 0 ? 1.0f : -1.0f;
    }
    fixture.setBulkViscosityCoefficient(1.0f);
    fixture.params_.vonNeumannRichtmyerCoefficient = 2.0f;
    fixture.params_.globalTimestep = true;
    fixture.params_.minimumRung = 0;
    fixture.density();
    fixture.enableEntropyRateOutput();
    fixture.hydro(1);
    double const coldStep = fixture.reduction_->minimumTimestep;
    ASSERT_GT(coldStep, 1.0e-5);
    ASSERT_LE(coldStep, fixture.params_.maximumTimestep);
    for (int i = 0; i < 2; ++i) {
        EXPECT_GT(fixture.dissipativeHeating(i), 0.0);
        data.sourceInternalEnergy[i] = 1.0e-10f;
    }
    fixture.hydro(1);
    double const warmStep = fixture.reduction_->minimumTimestep;
    ASSERT_LE(warmStep, fixture.params_.maximumTimestep);
    EXPECT_NEAR(warmStep, coldStep, 1.0e-3 * coldStep);
}

TEST_F(SphCpuTest, EntropyThermodynamics) {
    for (double gamma : {5.0 / 3.0, 1.4}) {
        double const entropy = 0.7;
        for (double scale : {0.1, 0.5, 1.0}) {
            double const density = 2.3;
            double const energy = sph_entropy::storedEnergy(entropy, density, gamma, scale);
            EXPECT_NEAR(sph_entropy::fromStoredEnergy(energy, density, gamma, scale), entropy,
                8 * std::numeric_limits<float>::epsilon());
            double const physicalDensity = density / (scale * scale * scale);
            double const pressure = (gamma - 1) * physicalDensity * energy / (scale * scale);
            EXPECT_NEAR(pressure / std::pow(physicalDensity, gamma), entropy,
                8 * std::numeric_limits<float>::epsilon());
            double const compressed = sph_entropy::storedEnergy(entropy, 8 * density, gamma, scale);
            EXPECT_NEAR(compressed / energy, std::pow(8.0, gamma - 1),
                8 * std::numeric_limits<float>::epsilon());
        }
    }
}

TEST_F(SphCpuTest, CompensatedFloatPairAccumulation) {
    // Each increment is below half an ULP of the starting sum. Naive float
    // summation loses every increment; Kahan must retain their exact total.
    SphCompensatedSum sum;
    EXPECT_EQ(sum.add(1.0f), 4);
    for (int i = 0; i < 4096; ++i) {
        EXPECT_EQ(sum.add(0x1p-25f), 4);
    }
    EXPECT_EQ(float(sum), 1.0f + 0x1p-13f);
    SphCompensatedSum partial;
    partial.add(1.0f);
    partial.add(0x1p-25f);
    SphCompensatedSum merged;
    merged.add(-1.0f);
    EXPECT_EQ(merged.merge(partial), 8);
    EXPECT_EQ(float(merged), 0x1p-25f);    // The residual must cross leaf/rank merges.

    SphCompensatedSum concurrent;
    concurrent.add(1.0f);
    std::array<std::thread, 4> workers;
    std::array<FlopCount, 4> counts{};
    for (size_t worker = 0; worker < workers.size(); ++worker) {
        workers[worker] = std::thread([&, worker] {
            for (int i = 0; i < 4096; ++i) {
                counts[worker] += concurrent.add(0x1p-25f);
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    EXPECT_EQ(float(concurrent), 1.0f + 0x1p-11f);
    for (auto count : counts) {
        EXPECT_GE(count, 4 * 4096);
        EXPECT_EQ(count % 4, 0);
    }
    SphPairUpdate target, neighbor;
    sph_geometry::SpatialVector<float> const acceleration(1.0f);
    EXPECT_EQ(sphAccumulatePairKick(target, neighbor, acceleration, 2.0f, 3.0f, 0.5f),
        37);    // Five products and eight four-operation compensated additions.
    EXPECT_EQ(sphAccumulateExternalKick(target, acceleration, 0.5f), 15);
}

TEST_F(SphCpuTest, EntropyProduction) {
    detail::Fixture fixture(2);
    auto& data = fixture.data_;
    std::array<float, 2> rates{};
    data.sinkEntropyRate = rates.data();
    data.sourcePositionX[0] = 0.4;
    data.sourcePositionX[1] = 0.6;
    for (int i = 0; i < 2; ++i) {
        data.sourcePositionY[i] = data.sourcePositionZ[i] = 0.5;
        data.sinkFlags[i].setConverged(true);
        data.sourceSmoothingLength[i] = data.sinkSmoothingLength[i] = 0.35f;
        data.sourceInternalEnergy[i] = 1.0f;
        data.sourceVelocityX[i] = i == 0 ? 1.0f : -1.0f;
    }
    fixture.params_.scaleFactor = 0.4f;
    fixture.params_.expansionRate = 0.2f;
    data.adiabaticIndex = 1.4f;
    fixture.setBulkViscosityCoefficient(0.0f);
    fixture.params_.vonNeumannRichtmyerCoefficient = 0.0f;
    fixture.density();
    fixture.hydro();
    for (int i = 0; i < 2; ++i) {
        EXPECT_EQ(rates[i], 0.0f);    // compression and cosmological expansion conserve A
    }
    fixture.setBulkViscosityCoefficient(1.0f);
    fixture.params_.vonNeumannRichtmyerCoefficient = 2.0f;
    fixture.hydro();
    for (int i = 0; i < 2; ++i) {
        EXPECT_GT(rates[i], 0.0f);    // viscous heating of the approaching pair raises A
    }
}

TEST_F(SphCpuTest, ArtificialConduction) {
    detail::Fixture fixture(2, 2);
    detail::testConductionPair(fixture);
}

TEST_F(SphCpuTest, DissipativeRateSuperposition) {
    for (bool const morrisAndMonaghan : {false, true}) {
        detail::Fixture fixture(2, 2);
        detail::testDissipativeRateSuperposition(fixture, morrisAndMonaghan);
    }
}
