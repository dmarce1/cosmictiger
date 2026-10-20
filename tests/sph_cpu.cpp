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
#include <vector>

#include <cosmictiger/sph/sph_cpu.hpp>

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

    explicit Fixture(int count, int treeCount = 1, int sourceOffset = 0, int sinkOffset = 0,
        int particleOffset = 0)
      : count_(count)
      , sourceOffset_(sourceOffset)
      , sinkOffset_(sinkOffset)
      , particleOffset_(particleOffset) {
        size_t const sourceSize = count + sourceOffset + 3;
        size_t const sinkSize = count + sinkOffset + 3;
        size_t const particleSize = count + particleOffset + 3;
        data_.sourcePositionX = allocate<Fixed32>(sourceSize);
        data_.sourcePositionY = allocate<Fixed32>(sourceSize);
        data_.sourcePositionZ = allocate<Fixed32>(sourceSize);
        data_.sourceVelocityX = allocate<float>(sourceSize);
        data_.sourceVelocityY = allocate<float>(sourceSize);
        data_.sourceVelocityZ = allocate<float>(sourceSize);
        data_.sourceInternalEnergy = allocate<float>(sourceSize);
        data_.sourceSupportRadius = allocate<float>(sourceSize);
        data_.sourceDensity = allocate<float>(sourceSize);
        data_.sourceRungs = allocate<char>(sourceSize);
        data_.sinkDensity = allocate<float>(sinkSize);
        data_.sinkGravityAccelerationX = allocate<float>(sinkSize);
        data_.sinkGravityAccelerationY = allocate<float>(sinkSize);
        data_.sinkGravityAccelerationZ = allocate<float>(sinkSize);
        data_.sinkInternalEnergyRate = allocate<float>(sinkSize);
        data_.sinkThermodynamicStates = allocate<SphThermodynamicState>(sinkSize);
        data_.sinkAccelerations = allocate<SphAcceleration>(sinkSize);
        data_.sinkParticleRungs = allocate<char>(particleSize);
        data_.sinkSmoothingConverged = allocate<char>(sinkSize);
        data_.sinkSemiactive = allocate<char>(sinkSize);
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
        params_.viscosityAlpha = 1.0f;
        params_.viscosityBeta = 2.0f;
        params_.viscosityEpsilon = 0.01f;
        params_.minimumSupportRadius = 0.1f;
        params_.maximumSupportRadius = 0.49f;
        params_.uniformAccelerationY = 0.0f;
        params_.velocityDampingRate = 0.0f;
        for (int i = 0; i < count; i++) {
            int const source = sourceIndex(i);
            int const sink = sinkIndex(i);
            data_.sinkParticleIndices[sink] = particleIndex(i);
            data_.sourceRungs[source] = data_.sinkParticleRungs[particleIndex(i)] = 1;
            data_.sourceInternalEnergy[source] = 1.0f;
            data_.sourceSupportRadius[source] = data_.sinkThermodynamicStates[sink].supportRadius =
                0.2f;
        }
        for (int tree = 0; tree < treeCount; tree++) {
            // Reverse the target order to exercise targetNodeIndices indirection.
            data_.targetNodeIndices[tree] = treeCount - tree - 1;
            data_.neighborNodeIndices[tree] = tree;
            auto& node = data_.treeNodes[tree];
            node.particleRange = {
                sourceIndex(tree * count / treeCount), sourceIndex((tree + 1) * count / treeCount)};
            node.sinkParticleRange = {
                sinkIndex(tree * count / treeCount), sinkIndex((tree + 1) * count / treeCount)};
            node.neighborRange = {0, treeCount};
            for (int dim = 0; dim < NDIM; dim++) {
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

    void enableCompositionEos() {
        data_.sourcePressureEnergyRatio = allocate<float>(count_ + sourceOffset_ + 3);
        data_.sourceAdiabaticIndex = allocate<float>(count_ + sourceOffset_ + 3);
    }

    void publishSupport() {
        for (int i = 0; i < count_; i++) {
            data_.sourceSupportRadius[sourceIndex(i)] =
                data_.sinkThermodynamicStates[sinkIndex(i)].supportRadius;
        }
    }

    void publishDensity() {
        for (int i = 0; i < count_; i++) {
            data_.sourceDensity[sourceIndex(i)] = data_.sinkDensity[sinkIndex(i)];
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
        params_.minimumSupportRadius = minimum;
        params_.maximumSupportRadius = maximum;
        for (int i = 0; i < count_; i++) {
            data_.sinkSmoothingConverged[sinkIndex(i)] = false;
            data_.sourceSupportRadius[sourceIndex(i)] =
                data_.sinkThermodynamicStates[sinkIndex(i)].supportRadius = initial;
        }
        run(SPH_RUN_PREHYDRO1);
        require(!reduction_->needsAnotherPass,
            "complete neighbor fixture must converge without refresh");
        for (int i = 0; i < count_; i++) {
            require(data_.sinkSmoothingConverged[sinkIndex(i)],
                "smoothing solve did not mark convergence");
            requireClose("smoothing preserves the staged source snapshot",
                data_.sourceSupportRadius[sourceIndex(i)], initial, 0.0);
            float const support = data_.sinkThermodynamicStates[sinkIndex(i)].supportRadius;
            require(support >= minimum && support <= maximum,
                "smoothing solve exceeded configured bounds");
        }
        publishSupport();
    }

    void density() {
        run(SPH_RUN_PREHYDRO2);
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
        run(SPH_RUN_HYDRO);
        for (int i = 0; i < count_; i++) {
            int const sink = sinkIndex(i);
            require(
                isFinite(data_.sinkInternalEnergyRate[sink]), "finite active-kernel thermal rate");
            for (int dim = 0; dim < NDIM; dim++) {
                require(isFinite(data_.sinkAccelerations[sink].acceleration[dim]),
                    "finite active-kernel acceleration");
            }
            if (phase == 1) {
                require(data_.sinkParticleRungs[particleIndex(i)] >= 1 &&
                        data_.sinkParticleRungs[particleIndex(i)] < MAX_RUNG,
                    "valid active-kernel timestep rung");
            }
        }
        publishRungs();
    }

    void checkConservation() const {
        std::array<double, NDIM> momentum{};
        double energy = 0.0;
        double absoluteEnergy = 0.0;
        double absoluteForce = 0.0;
        for (int i = 0; i < count_; i++) {
            int const source = sourceIndex(i);
            int const sink = sinkIndex(i);
            auto const& acceleration = data_.sinkAccelerations[sink].acceleration;
            double const kinetic = data_.gasMass *
                (data_.sourceVelocityX[source] * acceleration[XDIM] +
                    data_.sourceVelocityY[source] * acceleration[YDIM] +
                    data_.sourceVelocityZ[source] * acceleration[ZDIM]);
            double const thermal = data_.gasMass * data_.sinkInternalEnergyRate[sink];
            energy += kinetic + thermal;
            absoluteEnergy += std::abs(kinetic) + std::abs(thermal);
            for (int dim = 0; dim < NDIM; dim++) {
                momentum[dim] += data_.gasMass * acceleration[dim];
                absoluteForce += std::abs(data_.gasMass * acceleration[dim]);
            }
        }
        for (int dim = 0; dim < NDIM; dim++) {
            requireClose("active-kernel total momentum rate", momentum[dim], 0.0,
                3.0e-5 * std::max(1.0, absoluteForce));
        }
        requireClose(
            "active-kernel total energy rate", energy, 0.0, 3.0e-5 * std::max(1.0, absoluteEnergy));
    }
};

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
    Array<Fixed32, NDIM> source;
    for (int dim = 0; dim < NDIM; dim++) {
        box.begin[dim] = 0.49;
        box.end[dim] = 0.51;
        source[dim] = 0.5;
    }
    box.begin[XDIM] = 0.90;
    box.end[XDIM] = 0.95;
    source[XDIM] = 0.40;
    require(sphSupportIntersects(box, source, 0.49f),
        "periodic support reaches the far end of a sink box");
    require(!sphSupportIntersects(box, source, 0.40f),
        "periodic support excludes a genuinely separated box");
    source[XDIM] = 0.02;
    require(sphSupportIntersects(box, source, 0.10f), "periodic support crosses the box boundary");
}

void testLattice() {
    constexpr int kSide = 5;
    Fixture fixture(kSide * kSide * kSide);
    initializeLattice(fixture, kSide);
    fixture.solveSmoothing(0.1f, 0.24f, 0.2f);
    for (int i = 0; i < fixture.count_; i++) {
        requireClose("upper support bound", fixture.data_.sourceSupportRadius[i], 0.24f, 1.0e-7);
    }
    fixture.solveSmoothing(0.43f, 0.49f, 0.48f);
    for (int i = 0; i < fixture.count_; i++) {
        requireClose("lower support bound", fixture.data_.sourceSupportRadius[i], 0.43f, 1.0e-7);
    }
    fixture.solveSmoothing(0.3f, 0.3f, 0.17f);
    for (int i = 0; i < fixture.count_; i++) {
        requireClose("fixed support bound", fixture.data_.sourceSupportRadius[i], 0.3f, 1.0e-7);
    }
    fixture.solveSmoothing(0.1f, 0.49f, 0.2f);
    fixture.density();
    for (int i = 0; i < fixture.count_; i++) {
        double const support = fixture.data_.sourceSupportRadius[i];
        double const count = 4.0 * kPi / 3.0 * fixture.data_.sourceDensity[i] * support * support *
            support / fixture.data_.gasMass;
        requireClose("weighted neighbor target", count, fixture.data_.targetNeighborCount, 2.0e-4);
        requireClose("uniform lattice density", fixture.data_.sourceDensity[i],
            fixture.data_.sourceDensity[0], 2.0e-5);
        requireClose(
            "uniform lattice continuum density", fixture.data_.sourceDensity[i], 1.0, 0.03);
    }
    fixture.hydro(1);
    for (int i = 0; i < fixture.count_; i++) {
        requireClose(
            "uniform gas thermal rate", fixture.data_.sinkInternalEnergyRate[i], 0.0, 2.0e-5);
        for (int dim = 0; dim < NDIM; dim++) {
            requireClose("uniform gas force", fixture.data_.sinkAccelerations[i].acceleration[dim],
                0.0, 2.0e-4);
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
        forceNorm += std::abs(fixture.data_.sinkAccelerations[i].acceleration[XDIM]);
    }
    require(forceNorm > 0.1, "perturbed gas must exercise nonzero acceleration");

    // The physical expansion term is tested at nonunit a and nonmonatomic gamma.
    fixture.data_.adiabaticIndex = 1.4f;
    fixture.params_.scaleFactor = 0.6f;
    fixture.params_.expansionRate = 0.09f;
    for (int i = 0; i < fixture.count_; i++) {
        fixture.data_.sourceVelocityX[i] = 0.0f;
        fixture.data_.sourceInternalEnergy[i] = 2.0f;
    }
    fixture.density();
    fixture.hydro();
    double const background = (5.0f - 3.0f * fixture.data_.adiabaticIndex) *
        fixture.params_.expansionRate / fixture.params_.scaleFactor * 2.0f;
    for (int i = 0; i < fixture.count_; i++) {
        requireClose("homogeneous cosmological energy source",
            fixture.data_.sinkInternalEnergyRate[i], background, 2.0e-6);
    }
}

void testUnequalSupports() {
    Fixture fixture(2, 2);
    auto& data = fixture.data_;
    data.sourcePositionX[0] = 0.35;
    data.sourcePositionX[1] = 0.65;
    for (int i = 0; i < 2; i++) {
        data.sourcePositionY[i] = data.sourcePositionZ[i] = 0.5;
        data.sinkSmoothingConverged[i] = true;
        data.sourceInternalEnergy[i] = i == 0 ? 0.7f : 1.2f;
        data.sourceSupportRadius[i] = data.sinkThermodynamicStates[i].supportRadius =
            i == 0 ? 0.2f : 0.45f;
        auto& node = data.treeNodes[i];
        for (int dim = 0; dim < NDIM; dim++) {
            double const position = dim == XDIM ? data.sourcePositionX[i].toDouble() : 0.5;
            node.innerBox.begin[dim] = node.innerBox.end[dim] = position;
            node.outerBox.begin[dim] = position - data.sourceSupportRadius[i];
            node.outerBox.end[dim] = position + data.sourceSupportRadius[i];
        }
    }
    fixture.density();
    require(data.sourceDensity[0] != data.sourceDensity[1],
        "unequal supports must produce unequal densities");
    data.sourceVelocityX[0] = 0.1f;
    data.sourceVelocityX[1] = -0.1f;
    fixture.hydro();
    fixture.checkConservation();
    require(data.sinkAccelerations[0].acceleration[XDIM] < 0.0f &&
            data.sinkAccelerations[1].acceleration[XDIM] > 0.0f,
        "union support must include both directions of repulsive force");
    std::array<float, 2> const viscousEnergy = {
        data.sinkInternalEnergyRate[0], data.sinkInternalEnergyRate[1]};
    fixture.params_.viscosityAlpha = fixture.params_.viscosityBeta = 0.0f;
    fixture.hydro();
    for (int i = 0; i < 2; i++) {
        require(viscousEnergy[i] > data.sinkInternalEnergyRate[i],
            "approach must add viscosity heating");
    }
    data.sourceVelocityX[0] = -0.1f;
    data.sourceVelocityX[1] = 0.1f;
    fixture.hydro();
    std::array<float, 2> const inviscidEnergy = {
        data.sinkInternalEnergyRate[0], data.sinkInternalEnergyRate[1]};
    fixture.params_.viscosityAlpha = 1.0f;
    fixture.params_.viscosityBeta = 2.0f;
    fixture.hydro();
    fixture.checkConservation();
    for (int i = 0; i < 2; i++) {
        requireClose("separating pair has no viscosity heating", data.sinkInternalEnergyRate[i],
            inviscidEnergy[i], 2.0e-6);
    }
}

// Exercise the production density and pressure kernels using the selected host
// table through scalar double arithmetic. Core tests also check the table mathematics.
void testSelectedKernel() {
    Fixture fixture(2);
    auto& data = fixture.data_;
    data.sourcePositionX[0] = 0.98;
    data.sourcePositionX[1] = 0.16;
    for (int i = 0; i < 2; i++) {
        data.sourcePositionY[i] = data.sourcePositionZ[i] = 0.5;
        data.sinkSmoothingConverged[i] = true;
        data.sourceSupportRadius[i] = data.sinkThermodynamicStates[i].supportRadius =
            i == 0 ? 0.35f : 0.42f;
        data.sourceInternalEnergy[i] = i == 0 ? 0.7f : 1.2f;
    }
    fixture.params_.viscosityAlpha = fixture.params_.viscosityBeta = 0.0f;
    fixture.density();
    double const radius = std::abs(distance(data.sourcePositionX[0], data.sourcePositionX[1]));
    double gradient = 0.0;
    double pressurePair = 0.0;
    double inverseCfl = 0.0;
    for (int i = 0; i < 2; i++) {
        double const support = data.sourceSupportRadius[i];
        double const expectedDensity =
            data.gasMass * (kernelW(0.0) + kernelW(radius / support)) / std::pow(support, 3);
        requireClose(
            "selected kernel scalar/SIMD density", data.sourceDensity[i], expectedDensity, 3.0e-6);
        gradient += 0.5 * dkernelWDq(radius / support) / std::pow(support, 4);
        pressurePair +=
            (data.adiabaticIndex - 1.0f) * data.sourceInternalEnergy[i] / data.sourceDensity[i];
        double const sound = std::sqrt(
            data.adiabaticIndex * (data.adiabaticIndex - 1.0f) * data.sourceInternalEnergy[i]);
        inverseCfl = std::max(
            inverseCfl, sound * selectedHydroKernel().supportRadiusPerEffectiveRadius / support);
    }
    fixture.hydro(1);
    requireClose("selected kernel scalar/SIMD pressure force",
        data.sinkAccelerations[0].acceleration[XDIM], data.gasMass * pressurePair * gradient,
        8.0e-6);
    requireClose("selected kernel effective-width CFL", fixture.reduction_->inverseSoundTimestep,
        inverseCfl, 3.0e-6);
    fixture.checkConservation();

    data.sourceVelocityX[0] = 0.1f;
    data.sourceVelocityX[1] = -0.1f;
    fixture.params_.viscosityAlpha = 1.0f;
    fixture.params_.viscosityBeta = 2.0f;
    fixture.hydro();
    fixture.checkConservation();
}

void testCompositionEos() {
    Fixture fixture(2);
    fixture.enableCompositionEos();
    auto& data = fixture.data_;
    data.sourcePositionX[0] = 0.40;
    data.sourcePositionX[1] = 0.58;
    fixture.params_.viscosityAlpha = fixture.params_.viscosityBeta = 0.0f;
    for (int i = 0; i < 2; ++i) {
        data.sourcePositionY[i] = data.sourcePositionZ[i] = 0.5;
        data.sinkSmoothingConverged[i] = true;
        data.sourceSupportRadius[i] = data.sinkThermodynamicStates[i].supportRadius = 0.4f;
        data.sourceInternalEnergy[i] = i == 0 ? 0.7f : 1.2f;
        data.sourcePressureEnergyRatio[i] = i == 0 ? 0.4f : 0.3f;
        data.sourceAdiabaticIndex[i] = i == 0 ? 1.3f : 1.6f;
    }
    fixture.density();
    double const radius = std::abs(distance(data.sourcePositionX[0], data.sourcePositionX[1]));
    double gradient = 0.0;
    double pressurePair = 0.0;
    double inverseCfl = 0.0;
    for (int i = 0; i < 2; ++i) {
        double const support = data.sourceSupportRadius[i];
        gradient += 0.5 * dkernelWDq(radius / support) / std::pow(support, 4);
        pressurePair += data.sourcePressureEnergyRatio[i] * data.sourceInternalEnergy[i] /
            data.sourceDensity[i];
        double const sound = std::sqrt(data.sourceAdiabaticIndex[i] *
            data.sourcePressureEnergyRatio[i] * data.sourceInternalEnergy[i]);
        inverseCfl = std::max(
            inverseCfl, sound * selectedHydroKernel().supportRadiusPerEffectiveRadius / support);
    }
    fixture.hydro(1);
    requireClose("composition-dependent pressure force",
        data.sinkAccelerations[0].acceleration[XDIM], data.gasMass * pressurePair * gradient,
        8.0e-6);
    requireClose("composition-dependent sound speed", fixture.reduction_->inverseSoundTimestep,
        inverseCfl, 3.0e-6);
    fixture.checkConservation();
    fixture.params_.scaleFactor = 0.6f;
    fixture.params_.expansionRate = 0.09f;
    fixture.hydro();
    for (int i = 0; i < 2; ++i) {
        double const expected = (2.0 - 3.0 * data.sourcePressureEnergyRatio[i]) *
            fixture.params_.expansionRate / fixture.params_.scaleFactor *
            data.sourceInternalEnergy[i];
        requireClose("composition-dependent cosmological work", data.sinkInternalEnergyRate[i],
            expected, 2.0e-6);
    }
}

void testSelectedCentralGravity() {
    Fixture fixture(1);
    auto& data = fixture.data_;
    data.sourcePositionX[0] = 0.6;
    data.sourcePositionY[0] = data.sourcePositionZ[0] = 0.5;
    data.sinkSmoothingConverged[0] = true;
    data.centralAcceleration = 0.07f;
    data.centralSupportRadius = 0.3f;
    fixture.density();
    fixture.hydro();
    double const radius = distance(data.sourcePositionX[0], Fixed32(0.5));
    double const expected = -data.centralAcceleration * radius /
        std::pow(double(data.centralSupportRadius), 3) *
        radial_kernel::forceFactor(selectedGravityKernel(), radius / data.centralSupportRadius);
    requireClose("selected gravity kernel central force",
        data.sinkAccelerations[0].acceleration[XDIM], expected, 5.0e-6);
}

void testFixedGravityTimestep() {
    Fixture fixture(1);
    auto& data = fixture.data_;
    data.sourcePositionX[0] = data.sourcePositionY[0] = data.sourcePositionZ[0] = 0.5;
    data.sinkSmoothingConverged[0] = true;
    data.sourceInternalEnergy[0] = 1.0e-6f;
    data.selfGravityEnabled = true;
    data.gravityEffectiveRadius = 0.08f / selectedGravityKernel().supportRadiusPerEffectiveRadius;
    data.sinkGravityAccelerationX[0] = 0.1f;
    fixture.density();
    fixture.hydro(1);
    double const gravityStep = data.accelerationTimestepFactor *
        std::sqrt(fixture.params_.scaleFactor * data.gravityEffectiveRadius /
            data.sinkGravityAccelerationX[0]);
    int const expectedGravityRung =
        std::max(1, int(std::ceil(std::log2(fixture.params_.baseTimestep / gravityStep))));
    require(fixture.reduction_->maximumGravityRung == expectedGravityRung,
        "gas gravity timestep must use the shared fixed gravity radius");
    data.sourceSupportRadius[0] = data.sinkThermodynamicStates[0].supportRadius = 0.3f;
    fixture.density();
    fixture.hydro(1);
    require(fixture.reduction_->maximumGravityRung == expectedGravityRung,
        "changing gas smoothing must not change the gravity timestep");
}

void testDefaultTimestepCalibration() {
    Fixture fixture(1);
    auto& data = fixture.data_;
    data.sourcePositionX[0] = data.sourcePositionY[0] = data.sourcePositionZ[0] = 0.5;
    data.sinkSmoothingConverged[0] = true;
    fixture.params_.uniformAccelerationY = 2.0f;
    fixture.params_.courantFactor = 0.25 * std::sqrt(5.0 / 6.0);
    double const oldCentralPotential = 693.0 / 256.0;
    data.accelerationTimestepFactor = 0.25 *
        std::sqrt(selectedGravityKernel().supportRadiusPerEffectiveRadius / oldCentralPotential);
    fixture.density();
    fixture.hydro(1);
    double const sound = std::sqrt(
        data.adiabaticIndex * (data.adiabaticIndex - 1.0f) * data.sourceInternalEnergy[0]);
    double const oldSmoothingLength = 0.5 * data.sourceSupportRadius[0];
    double const expectedSoundStep =
        0.25 * oldSmoothingLength / (sound * (1.0 + 0.6 * fixture.params_.viscosityAlpha));
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
        data.sinkSmoothingConverged[i] = true;
        data.sourceSupportRadius[i] = data.sinkThermodynamicStates[i].supportRadius = 0.35f;
        data.sourceInternalEnergy[i] = 1.0e-6f;
        data.sourceVelocityX[i] = i == 0 ? 1.0f : -1.0f;
    }
    fixture.params_.viscosityAlpha = fixture.params_.viscosityBeta = 0.0f;
    fixture.params_.courantFactor = 0.25 * std::sqrt(5.0 / 6.0);
    double const oldEta = 0.25 / std::sqrt(693.0 / 256.0);
    data.accelerationTimestepFactor =
        oldEta * std::sqrt(selectedGravityKernel().supportRadiusPerEffectiveRadius);
    fixture.density();
    fixture.hydro(1);
    for (int i = 0; i < 2; i++) {
        double const oldWidth = 0.5 * data.sourceSupportRadius[i];
        double const sound =
            std::sqrt(kAdiabaticIndex * (kAdiabaticIndex - 1.0f) * data.sourceInternalEnergy[i]);
        double const acceleration = std::abs(data.sinkAccelerations[i].acceleration[XDIM]);
        double const soundStep = 0.25 * oldWidth / sound;
        double const accelerationStep = oldEta * std::sqrt(oldWidth / acceleration);
        double const thermalStep =
            0.25 * data.sourceInternalEnergy[i] / std::abs(data.sinkInternalEnergyRate[i]);
        require(thermalStep < soundStep && thermalStep < accelerationStep,
            "compression fixture must activate the fractional thermal timestep restriction");
        int const expectedRung =
            int(std::ceil(std::log2(fixture.params_.baseTimestep / thermalStep)));
        require(data.sourceRungs[i] == expectedRung,
            "default fractional thermal bound must preserve the historical timestep rung");
    }
}

// A nonmultiple of the SIMD width exercises every masked tail, while unrelated
// staged, SPH-sink, and full-particle offsets exercise all three index spaces.
void testPeriodicTailsAndOffsets() {
    int const count = 2 * SIMD_FLOAT_SIZE + 3;
    Fixture shifted(count, 3, 5, 11, 7);
    Fixture compact(count, 3);
    for (Fixture* fixture : {&shifted, &compact}) {
        auto& data = fixture->data_;
        for (int i = 0; i < count; i++) {
            int const source = fixture->sourceIndex(i);
            int const sink = fixture->sinkIndex(i);
            data.sourcePositionX[source] = std::fmod(0.96 + 0.006 * i, 1.0);
            data.sourcePositionY[source] = 0.46 + 0.004 * (i % 5);
            data.sourcePositionZ[source] = 0.48 + 0.003 * (i % 3);
            data.sourceVelocityX[source] = 0.01f * std::sin(float(i));
            data.sourceVelocityY[source] = 0.02f * std::cos(float(i));
            data.sourceInternalEnergy[source] = 0.8f + 0.03f * (i % 7);
            data.sourceSupportRadius[source] = data.sinkThermodynamicStates[sink].supportRadius =
                0.14f + 0.01f * (i % 3);
            data.sinkSmoothingConverged[sink] = true;
        }
        for (int i = 0; i < fixture->sinkOffset_; i++) {
            data.sinkDensity[i] = -91.0f;
            data.sinkInternalEnergyRate[i] = -92.0f;
            data.sinkAccelerations[i].acceleration[XDIM] = -93.0f;
        }
        data.sinkDensity[fixture->sinkIndex(count)] = -91.0f;
        data.sinkInternalEnergyRate[fixture->sinkIndex(count)] = -92.0f;
        data.sinkAccelerations[fixture->sinkIndex(count)].acceleration[XDIM] = -93.0f;
        fixture->run(SPH_RUN_PREHYDRO2);
        for (int i = 0; i < count; i++) {
            int const source = fixture->sourceIndex(i);
            double const support = data.sourceSupportRadius[source];
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
        fixture->run(SPH_RUN_HYDRO);
        for (int i = 0; i < count; i++) {
            require(data.sourceRungs[fixture->sourceIndex(i)] == 1,
                "hydro leaves source rung snapshots unchanged");
        }
        fixture->checkConservation();
        for (int i = 0; i < fixture->sinkOffset_; i++) {
            requireClose("density guard before sinks", data.sinkDensity[i], -91.0, 0.0);
            requireClose("heating guard before sinks", data.sinkInternalEnergyRate[i], -92.0, 0.0);
            requireClose("force guard before sinks", data.sinkAccelerations[i].acceleration[XDIM],
                -93.0, 0.0);
        }
        requireClose(
            "density guard after sinks", data.sinkDensity[fixture->sinkIndex(count)], -91.0, 0.0);
        requireClose("heating guard after sinks",
            data.sinkInternalEnergyRate[fixture->sinkIndex(count)], -92.0, 0.0);
        requireClose("force guard after sinks",
            data.sinkAccelerations[fixture->sinkIndex(count)].acceleration[XDIM], -93.0, 0.0);
    }
    for (int i = 0; i < count; i++) {
        int const sink = shifted.sinkIndex(i);
        requireClose("offset-invariant density", shifted.data_.sinkDensity[sink],
            compact.data_.sinkDensity[i], 1.0e-6);
        requireClose("offset-invariant heating", shifted.data_.sinkInternalEnergyRate[sink],
            compact.data_.sinkInternalEnergyRate[i], 1.0e-6);
        for (int dim = 0; dim < NDIM; dim++) {
            requireClose("offset-invariant force",
                shifted.data_.sinkAccelerations[sink].acceleration[dim],
                compact.data_.sinkAccelerations[i].acceleration[dim], 1.0e-6);
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
    fixture.params_.minimumSupportRadius = fixture.params_.maximumSupportRadius = 0.25f;
    for (int i = 0; i < 3; i++) {
        int const source = fixture.sourceIndex(i);
        int const sink = fixture.sinkIndex(i);
        data.sourcePositionX[source] = i == 0 ? 0.4 : (i == 1 ? 0.6 : 0.85);
        data.sourcePositionY[source] = data.sourcePositionZ[source] = i == 2 ? 0.85 : 0.5;
        data.sourceRungs[source] = data.sinkParticleRungs[fixture.particleIndex(i)] =
            i == 0 ? 4 : 1;
        data.sinkDensity[sink] = -11.0f;
        data.sinkInternalEnergyRate[sink] = -12.0f;
        for (int dim = 0; dim < NDIM; dim++) {
            data.sinkAccelerations[sink].acceleration[dim] = -13.0f;
        }
    }
    auto result = sphCpuPrehydro1(fixture.params_, data, 0);
    require(!result.needsAnotherPass, "active fixed support converges");
    require(data.sinkSmoothingConverged[fixture.sinkIndex(0)], "active smoothing is solved");
    require(!data.sinkSmoothingConverged[fixture.sinkIndex(1)] &&
            !data.sinkSmoothingConverged[fixture.sinkIndex(2)],
        "inactive smoothing waits for semiactive classification");
    fixture.publishSupport();
    result = sphCpuPrehydro2(fixture.params_, data, 0);
    require(!result.needsAnotherPass, "semiactive fixed support converges");
    require(data.sinkSemiactive[fixture.sinkIndex(0)] &&
            data.sinkSemiactive[fixture.sinkIndex(1)] && !data.sinkSemiactive[fixture.sinkIndex(2)],
        "only active and interacting inactive particles are semiactive");
    require(data.sinkDensity[fixture.sinkIndex(0)] > 0.0f &&
            data.sinkDensity[fixture.sinkIndex(1)] > 0.0f,
        "active and semiactive density is computed");
    require(data.sinkSmoothingConverged[fixture.sinkIndex(1)], "semiactive smoothing is solved");
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
    require(data.sinkAccelerations[fixture.sinkIndex(0)].acceleration[XDIM] < 0.0f,
        "active particle feels its semiactive neighbor");
    require(isFinite(data.sinkInternalEnergyRate[fixture.sinkIndex(0)]),
        "excluded zero-density neighbor leaves finite heating");
    for (int dim = 0; dim < NDIM; dim++) {
        require(isFinite(data.sinkAccelerations[fixture.sinkIndex(0)].acceleration[dim]),
            "excluded zero-density neighbor leaves finite force");
    }
    for (int i = 1; i < 3; i++) {
        int const sink = fixture.sinkIndex(i);
        requireClose(
            "inactive heating is preserved", data.sinkInternalEnergyRate[sink], -12.0, 0.0);
        for (int dim = 0; dim < NDIM; dim++) {
            requireClose("inactive force is preserved",
                data.sinkAccelerations[sink].acceleration[dim], -13.0, 0.0);
        }
        require(data.sinkParticleRungs[fixture.particleIndex(i)] == 1,
            "inactive timestep rung is preserved");
    }
}

void testSupportRefresh() {
    Fixture active(1);
    auto& data = active.data_;
    data.sourcePositionX[0] = data.sourcePositionY[0] = data.sourcePositionZ[0] = 0.5;
    active.params_.minimumSupportRadius = 0.1f;
    active.params_.maximumSupportRadius = 0.3f;
    data.sourceSupportRadius[0] = data.sinkThermodynamicStates[0].supportRadius = 0.1f;
    for (int dim = 0; dim < NDIM; dim++) {
        data.treeNodes[0].outerBox.begin[dim] = 0.39;
        data.treeNodes[0].outerBox.end[dim] = 0.61;
    }
    auto result = sphCpuPrehydro1(active.params_, data, 0);
    require(result.needsAnotherPass && !data.sinkSmoothingConverged[0],
        "active smoothing requests a neighbor refresh when support grows beyond the search box");
    require(data.sinkThermodynamicStates[0].supportRadius > 0.1f,
        "retry retains the enlarged support for the next search");
    requireClose(
        "retry preserves the source support snapshot", data.sourceSupportRadius[0], 0.1f, 0.0);
    for (int dim = 0; dim < NDIM; dim++) {
        data.treeNodes[0].outerBox.begin[dim] = -1.0;
        data.treeNodes[0].outerBox.end[dim] = 2.0;
    }
    result = sphCpuPrehydro1(active.params_, data, 0);
    require(!result.needsAnotherPass && data.sinkSmoothingConverged[0],
        "active smoothing converges after neighbor refresh");
    requireClose("sparse gas reaches the configured support ceiling",
        data.sinkThermodynamicStates[0].supportRadius, 0.3f, 1.0e-7);

    Fixture semiactive(2);
    auto& semiData = semiactive.data_;
    semiactive.params_.minimumRung = 4;
    semiactive.params_.minimumSupportRadius = 0.1f;
    semiactive.params_.maximumSupportRadius = 0.3f;
    for (int i = 0; i < 2; i++) {
        semiData.sourcePositionX[i] = 0.5 + 0.05 * i;
        semiData.sourcePositionY[i] = semiData.sourcePositionZ[i] = 0.55;
        semiData.sourceSupportRadius[i] = semiData.sinkThermodynamicStates[i].supportRadius = 0.1f;
        semiData.sourceRungs[i] = semiData.sinkParticleRungs[semiactive.particleIndex(i)] =
            i == 0 ? 4 : 1;
        semiData.sinkSmoothingConverged[i] = i == 0;
        semiData.sinkDensity[i] = -21.0f;
    }
    for (int dim = 0; dim < NDIM; dim++) {
        semiData.treeNodes[0].outerBox.begin[dim] = 0.43;
        semiData.treeNodes[0].outerBox.end[dim] = 0.67;
    }
    result = sphCpuPrehydro2(semiactive.params_, semiData, 0);
    require(result.needsAnotherPass && semiData.sinkSemiactive[1] &&
            !semiData.sinkSmoothingConverged[1],
        "semiactive smoothing requests a neighbor refresh");
    requireClose("failed semiactive solve preserves density", semiData.sinkDensity[1], -21.0, 0.0);
    for (int dim = 0; dim < NDIM; dim++) {
        semiData.treeNodes[0].outerBox.begin[dim] = -1.0;
        semiData.treeNodes[0].outerBox.end[dim] = 2.0;
    }
    result = sphCpuPrehydro2(semiactive.params_, semiData, 0);
    require(!result.needsAnotherPass && semiData.sinkSmoothingConverged[1] &&
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
        data.sourceSupportRadius[source] = data.sinkThermodynamicStates[sink].supportRadius = 0.11f;
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
        auto const result = fixture.run(SPH_RUN_RUNGS);
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

}    // namespace detail

int main() {
    try {
#if defined(__linux__) && defined(__GLIBC__)
        std::feclearexcept(FE_ALL_EXCEPT);
        detail::require(feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW) != -1,
            "enable floating-point traps for masked SIMD arithmetic");
#endif
        setKernelSelections(
            {radial_kernel::Family::Ferrers, 8}, {radial_kernel::Family::Spline, 3}, false);
        detail::testPeriodicSupport();
        detail::testPeriodicTailsAndOffsets();
        detail::testActivity();
        detail::testSupportRefresh();
        detail::testRungPropagation();
        detail::testLattice();
        detail::testUnequalSupports();
        detail::testDefaultTimestepCalibration();
        detail::testDefaultThermalTimestep();
        detail::testCompositionEos();
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
        std::puts("SPH SIMD CPU regressions passed");
    } catch (std::exception const& error) {
        std::fprintf(stderr, "SPH SIMD CPU regression failed: %s\n", error.what());
        return 1;
    }
    return 0;
}
