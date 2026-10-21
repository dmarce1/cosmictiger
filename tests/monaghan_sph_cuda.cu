/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <cosmictiger/sph/sph_cpu.hpp>
#include <cosmictiger/sph/sph_cuda.hpp>
#include "cuda_test.hpp"

// Only the options-constructor and host assertion services are replaced here.
// Density, smoothing, forces, heating, device assertions, and allocation use the
// production translation units. The test does not start an HPX runtime.
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

void checkCuda(cudaError_t status, char const* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void require(bool condition, char const* label) {
    if (!condition) {
        throw std::runtime_error(label);
    }
}

void requireClose(char const* label, double actual, double expected, double tolerance) {
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > tolerance * std::max(1.0, std::abs(expected))) {
        throw std::runtime_error(std::string(label) + ": got " + std::to_string(actual) +
            ", expected " + std::to_string(expected));
    }
}

class Fixture {
    std::vector<void*> allocations_;

    template <typename Value>
    Value* allocate(size_t count) {
        Value* pointer = nullptr;
        checkCuda(cudaMallocManaged(&pointer, count * sizeof(Value)), "managed test allocation");
        allocations_.push_back(pointer);
        for (size_t i = 0; i < count; i++) {
            new (pointer + i) Value{};
        }
        return pointer;
    }

    void resetReduction() {
        *reduction_ = {};
        reduction_->minimumSupportRadius = std::numeric_limits<float>::max();
    }

public:
    int count_;
    SphRunParameters params_;
    SphRunCudaData data_{};
    SphReduction* reduction_;

    explicit Fixture(int count, int treeCount = 1, bool separateSources = false)
      : count_(count) {
        data_.sourcePositionX = allocate<Fixed32>(count);
        data_.sourcePositionY = allocate<Fixed32>(count);
        data_.sourcePositionZ = allocate<Fixed32>(count);
        data_.sourceVelocityX = allocate<float>(count);
        data_.sourceVelocityY = allocate<float>(count);
        data_.sourceVelocityZ = allocate<float>(count);
        data_.sourceInternalEnergy = allocate<float>(count);
        data_.sourceSupportRadius = allocate<float>(count);
        data_.sourceDensity = allocate<float>(count);
        data_.sinkDensity = separateSources ? allocate<float>(count) : data_.sourceDensity;
        data_.sinkGravityAccelerationX = allocate<float>(count);
        data_.sinkGravityAccelerationY = allocate<float>(count);
        data_.sinkGravityAccelerationZ = allocate<float>(count);
        data_.sinkInternalEnergyRate = allocate<float>(count);
        data_.sinkThermodynamicStates = allocate<SphThermodynamicState>(count);
        data_.sinkAccelerations = allocate<SphAcceleration>(count);
        data_.sourceRungs = allocate<char>(count);
        data_.sinkParticleRungs = separateSources ? allocate<char>(count) : data_.sourceRungs;
        data_.sinkSmoothingConverged = allocate<char>(count);
        data_.sinkSemiactive = allocate<char>(count);
        data_.sinkParticleIndices = allocate<ParticleInteger>(count);
        data_.treeNodes = allocate<SphTreeNode>(treeCount);
        data_.targetNodeIndices = allocate<int>(treeCount);
        data_.neighborNodeIndices = allocate<int>(treeCount);
        reduction_ = allocate<SphReduction>(1);
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
        for (int i = 0; i < count; i++) {
            data_.sinkParticleIndices[i] = i;
            data_.sourceRungs[i] = data_.sinkParticleRungs[i] = 1;
            data_.sourceInternalEnergy[i] = 1.0f;
            data_.sourceSupportRadius[i] = data_.sinkThermodynamicStates[i].supportRadius = 0.2f;
            data_.sinkGravityAccelerationX[i] = data_.sinkGravityAccelerationY[i] =
                data_.sinkGravityAccelerationZ[i] = 0.0f;
        }
        for (int tree = 0; tree < treeCount; tree++) {
            data_.targetNodeIndices[tree] = data_.neighborNodeIndices[tree] = tree;
            auto& node = data_.treeNodes[tree];
            node.particleRange = {tree * count / treeCount, (tree + 1) * count / treeCount};
            node.sinkParticleRange = node.particleRange;
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
        for (void* allocation : allocations_) {
            cudaFree(allocation);
        }
    }

    void solveSmoothing(float minimum, float maximum, float initial) {
        params_.minimumSupportRadius = minimum;
        params_.maximumSupportRadius = maximum;
        for (int i = 0; i < count_; i++) {
            data_.sinkSmoothingConverged[i] = false;
            data_.sourceSupportRadius[i] = data_.sinkThermodynamicStates[i].supportRadius = initial;
        }
        resetReduction();
        sphCudaPrehydro1<<<1, PREHYDRO1_BLOCK_SIZE>>>(params_, data_, reduction_);
        checkCuda(cudaGetLastError(), "prehydro1 launch");
        checkCuda(cudaDeviceSynchronize(), "prehydro1 execution");
        require(reduction_->pendingChanges == 0,
            "complete neighbor fixture must converge without refresh");
        for (int i = 0; i < count_; i++) {
            require(data_.sinkSmoothingConverged[i], "smoothing solve did not mark convergence");
            data_.sourceSupportRadius[i] = data_.sinkThermodynamicStates[i].supportRadius;
            require(
                data_.sourceSupportRadius[i] >= minimum && data_.sourceSupportRadius[i] <= maximum,
                "smoothing solve exceeded configured bounds");
        }
    }

    void enableCompositionEos() {
        data_.sourcePressureEnergyRatio = allocate<float>(count_);
        data_.sourceAdiabaticIndex = allocate<float>(count_);
    }

    void density() {
        resetReduction();
        sphCudaPrehydro2<<<1, PREHYDRO2_BLOCK_SIZE>>>(params_, data_, reduction_);
        checkCuda(cudaGetLastError(), "prehydro2 launch");
        checkCuda(cudaDeviceSynchronize(), "prehydro2 execution");
        require(
            reduction_->pendingChanges == 0, "density fixture unexpectedly needs neighbor refresh");
        for (int i = 0; i < count_; i++) {
            require(data_.sourceDensity[i] > 0.0f && std::isfinite(data_.sourceDensity[i]),
                "positive finite density");
        }
    }

    void hydro(int phase = 0) {
        params_.phase = phase;
        resetReduction();
        sphCudaHydro<<<1, HYDRO_BLOCK_SIZE>>>(params_, data_, reduction_);
        checkCuda(cudaGetLastError(), "hydro launch");
        checkCuda(cudaDeviceSynchronize(), "hydro execution");
        for (int i = 0; i < count_; i++) {
            require(std::isfinite(data_.sinkInternalEnergyRate[i]),
                "finite active-kernel thermal rate");
            for (int dim = 0; dim < NDIM; dim++) {
                require(std::isfinite(data_.sinkAccelerations[i].acceleration[dim]),
                    "finite active-kernel acceleration");
            }
            if (phase == 1) {
                require(data_.sourceRungs[i] >= 1 && data_.sourceRungs[i] < MAX_RUNG,
                    "valid active-kernel timestep rung");
            }
        }
    }

    void checkConservation() const {
        std::array<double, NDIM> momentum{};
        double energy = 0.0;
        double absoluteEnergy = 0.0;
        double absoluteForce = 0.0;
        for (int i = 0; i < count_; i++) {
            auto const& acceleration = data_.sinkAccelerations[i].acceleration;
            double const kinetic = data_.gasMass *
                (data_.sourceVelocityX[i] * acceleration[XDIM] +
                    data_.sourceVelocityY[i] * acceleration[YDIM] +
                    data_.sourceVelocityZ[i] * acceleration[ZDIM]);
            double const thermal = data_.gasMass * data_.sinkInternalEnergyRate[i];
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

// Run both backends from the same immutable sources and initial sink state.
// CPU writes go to host copies; the following CUDA launch owns the managed sinks.
void compareCpuCudaPhase(Fixture& fixture, int runType, int phase = 0) {
    auto& data = fixture.data_;
    require(data.sourceDensity != data.sinkDensity && data.sourceRungs != data.sinkParticleRungs,
        "CPU/CUDA parity requires independent source snapshots");
    fixture.params_.runType = runType;
    fixture.params_.phase = phase;
    SphRunData cpuData = data;
    std::vector<float> density(data.sinkDensity, data.sinkDensity + fixture.count_);
    std::vector<float> energyRate(
        data.sinkInternalEnergyRate, data.sinkInternalEnergyRate + fixture.count_);
    std::vector<SphAcceleration> acceleration(
        data.sinkAccelerations, data.sinkAccelerations + fixture.count_);
    std::vector<SphThermodynamicState> thermodynamicStates(
        data.sinkThermodynamicStates, data.sinkThermodynamicStates + fixture.count_);
    std::vector<char> rungs(data.sinkParticleRungs, data.sinkParticleRungs + fixture.count_);
    std::vector<char> semiactive(data.sinkSemiactive, data.sinkSemiactive + fixture.count_);
    std::vector<char> converged(
        data.sinkSmoothingConverged, data.sinkSmoothingConverged + fixture.count_);
    std::vector<float> const sourceDensity(data.sourceDensity, data.sourceDensity + fixture.count_);
    std::vector<float> const sourceSupport(
        data.sourceSupportRadius, data.sourceSupportRadius + fixture.count_);
    std::vector<char> const sourceRungs(data.sourceRungs, data.sourceRungs + fixture.count_);
    cpuData.sinkDensity = density.data();
    cpuData.sinkInternalEnergyRate = energyRate.data();
    cpuData.sinkAccelerations = acceleration.data();
    cpuData.sinkThermodynamicStates = thermodynamicStates.data();
    cpuData.sinkParticleRungs = rungs.data();
    cpuData.sinkSemiactive = semiactive.data();
    cpuData.sinkSmoothingConverged = converged.data();
    SphRunReturn cpuResult;
    for (int node = 0; node < data.numberOfTargetNodes; node++) {
        cpuResult += sphRunCpuNode(fixture.params_, cpuData, node);
    }
    *fixture.reduction_ = {};
    fixture.reduction_->minimumSupportRadius = std::numeric_limits<float>::max();
    switch (runType) {
    case SPH_RUN_PREHYDRO1:
        sphCudaPrehydro1<<<1, PREHYDRO1_BLOCK_SIZE>>>(fixture.params_, data, fixture.reduction_);
        break;
    case SPH_RUN_PREHYDRO2:
        sphCudaPrehydro2<<<1, PREHYDRO2_BLOCK_SIZE>>>(fixture.params_, data, fixture.reduction_);
        break;
    case SPH_RUN_HYDRO:
        sphCudaHydro<<<1, HYDRO_BLOCK_SIZE>>>(fixture.params_, data, fixture.reduction_);
        break;
    case SPH_RUN_RUNGS:
        sphCudaRungs<<<1, RUNGS_BLOCK_SIZE>>>(fixture.params_, data, fixture.reduction_);
        break;
    default:
        require(false, "invalid CPU/CUDA parity phase");
    }
    checkCuda(cudaGetLastError(), "CPU/CUDA parity launch");
    checkCuda(cudaDeviceSynchronize(), "CPU/CUDA parity execution");
    auto const& cudaResult = *fixture.reduction_;
    require(cpuResult.needsAnotherPass == (cudaResult.pendingChanges != 0),
        "CPU/CUDA pending-change parity");
    require(cpuResult.maximumHydroRung == cudaResult.maximumHydroRung &&
            cpuResult.maximumGravityRung == cudaResult.maximumGravityRung &&
            cpuResult.maximumRung == cudaResult.maximumRung,
        "CPU/CUDA rung-reduction parity");
    requireClose("CPU/CUDA operation count", cpuResult.floatingPointOperations,
        cudaResult.floatingPointOperations, 0.0);
    requireClose("CPU/CUDA minimum support", cpuResult.minimumSupportRadius,
        cudaResult.minimumSupportRadius, 3.0e-5);
    requireClose("CPU/CUDA maximum support", cpuResult.maximumSupportRadius,
        cudaResult.maximumSupportRadius, 3.0e-5);
    requireClose("CPU/CUDA signal speed", cpuResult.maximumSignalSpeed,
        cudaResult.maximumSignalSpeed, 5.0e-5);
    requireClose("CPU/CUDA sound timestep", cpuResult.inverseSoundTimestep,
        cudaResult.inverseSoundTimestep, 5.0e-5);
    requireClose("CPU/CUDA viscosity timestep", cpuResult.inverseViscosityTimestep,
        cudaResult.inverseViscosityTimestep, 5.0e-5);
    requireClose("CPU/CUDA acceleration timestep", cpuResult.inverseAccelerationTimestep,
        cudaResult.inverseAccelerationTimestep, 5.0e-5);
    for (int i = 0; i < fixture.count_; i++) {
        requireClose("CPU/CUDA density", density[i], data.sinkDensity[i], 5.0e-5);
        requireClose("CPU/CUDA support", thermodynamicStates[i].supportRadius,
            data.sinkThermodynamicStates[i].supportRadius, 3.0e-5);
        require(
            thermodynamicStates[i].internalEnergy == data.sinkThermodynamicStates[i].internalEnergy,
            "CPU/CUDA kernels preserve stored internal energy");
        requireClose(
            "CPU/CUDA thermal rate", energyRate[i], data.sinkInternalEnergyRate[i], 5.0e-5);
        for (int dim = 0; dim < NDIM; dim++) {
            requireClose("CPU/CUDA acceleration", acceleration[i].acceleration[dim],
                data.sinkAccelerations[i].acceleration[dim], 5.0e-5);
        }
        require(rungs[i] == data.sinkParticleRungs[i], "CPU/CUDA particle-rung parity");
        require(semiactive[i] == data.sinkSemiactive[i], "CPU/CUDA semiactive parity");
        require(converged[i] == data.sinkSmoothingConverged[i], "CPU/CUDA convergence parity");
        require(data.sourceDensity[i] == sourceDensity[i] &&
                data.sourceSupportRadius[i] == sourceSupport[i] &&
                data.sourceRungs[i] == sourceRungs[i],
            "CPU/CUDA phases must preserve source snapshots");
    }
}

void testCpuCudaParity() {
    // Three unequal leaves and 19 neighbors exercise indirection and SIMD tails.
    Fixture fixture(19, 3, true);
    auto& data = fixture.data_;
    std::vector<int> sinkIndices(fixture.count_);
    int const sinkBegins[] = {7, 13, 0};
    for (int tree = 0; tree < 3; tree++) {
        auto& node = data.treeNodes[tree];
        int const count = node.particleRange.second - node.particleRange.first;
        node.sinkParticleRange = {sinkBegins[tree], sinkBegins[tree] + count};
        data.targetNodeIndices[tree] = 2 - tree;
        data.neighborNodeIndices[tree] = (tree + 1) % 3;
        for (int source = node.particleRange.first; source < node.particleRange.second; source++) {
            sinkIndices[source] = node.sinkParticleRange.first + source - node.particleRange.first;
        }
    }
    fixture.params_.minimumRung = 2;
    fixture.params_.minimumSupportRadius = 0.03f;
    fixture.params_.maximumSupportRadius = 0.24f;
    fixture.params_.scaleFactor = 0.7f;
    fixture.params_.expansionRate = 0.08f;
    fixture.params_.baseTimestep = 1.11f;
    fixture.params_.maximumTimestep = 0.7f;
    fixture.params_.uniformAccelerationY = 0.03f;
    fixture.params_.zPlaneAcceleration = 0.02f;
    fixture.params_.velocityDampingRate = 0.04f;
    data.adiabaticIndex = 1.4f;
    data.selfGravityEnabled = true;
    data.centralAcceleration = 0.002f;
    data.centralSupportRadius = 0.2f;
    for (int source = 0; source < fixture.count_; source++) {
        int const sink = sinkIndices[source];
        int const particle = (2 * source) % fixture.count_;
        data.sinkParticleIndices[sink] = particle;
        double const positionX = 0.96 + 0.009 * source;
        data.sourcePositionX[source] = positionX - std::floor(positionX);
        data.sourcePositionY[source] = 0.48 + 0.005 * (source % 3);
        data.sourcePositionZ[source] = 0.49 + 0.004 * (source % 5);
        data.sourceSupportRadius[source] = data.sinkThermodynamicStates[sink].supportRadius =
            0.08f + 0.005f * (source % 4);
        data.sourceInternalEnergy[source] = data.sinkThermodynamicStates[sink].internalEnergy =
            0.7f + 0.03f * source;
        data.sourceVelocityX[source] = 0.03f * (source % 3 - 1);
        data.sourceVelocityY[source] = -0.02f * (source % 5 - 2);
        data.sourceVelocityZ[source] = 0.01f * (source % 2 ? 1 : -1);
        data.sourceRungs[source] = data.sinkParticleRungs[particle] = source % 3 ? 1 : 3;
        data.sinkGravityAccelerationX[sink] = 0.01f * (source % 3 - 1);
        data.sinkGravityAccelerationY[sink] = -0.005f;
        data.sinkGravityAccelerationZ[sink] = 0.002f;
    }
    auto publishSupport = [&] {
        for (int source = 0; source < fixture.count_; source++) {
            data.sourceSupportRadius[source] =
                data.sinkThermodynamicStates[sinkIndices[source]].supportRadius;
        }
    };
    auto publishRungs = [&] {
        for (int source = 0; source < fixture.count_; source++) {
            data.sourceRungs[source] =
                data.sinkParticleRungs[data.sinkParticleIndices[sinkIndices[source]]];
        }
    };
    compareCpuCudaPhase(fixture, SPH_RUN_PREHYDRO1);
    publishSupport();
    compareCpuCudaPhase(fixture, SPH_RUN_PREHYDRO2);
    publishSupport();
    for (int source = 0; source < fixture.count_; source++) {
        data.sourceDensity[source] = data.sinkDensity[sinkIndices[source]];
    }
    compareCpuCudaPhase(fixture, SPH_RUN_HYDRO, 0);
    compareCpuCudaPhase(fixture, SPH_RUN_HYDRO, 1);
    fixture.params_.minimumRung = 1;
    for (int source = 0; source < fixture.count_; source++) {
        data.sinkParticleRungs[data.sinkParticleIndices[sinkIndices[source]]] = source == 0 ? 6 : 1;
    }
    publishRungs();
    compareCpuCudaPhase(fixture, SPH_RUN_RUNGS);
    publishRungs();
    compareCpuCudaPhase(fixture, SPH_RUN_RUNGS);

    // Keep an invalid thermodynamic state in a gathered but excluded SIMD lane.
    int const excluded = fixture.count_ - 1;
    int const excludedSink = sinkIndices[excluded];
    data.sourcePositionX[excluded] = data.sourcePositionY[excluded] =
        data.sourcePositionZ[excluded] = 0.5;
    data.sourceRungs[excluded] = data.sinkParticleRungs[data.sinkParticleIndices[excludedSink]] = 0;
    data.sourceSupportRadius[excluded] = data.sinkThermodynamicStates[excludedSink].supportRadius =
        0.025f;
    data.sourceDensity[excluded] = data.sourceInternalEnergy[excluded] = 0.0f;
    data.sinkDensity[excludedSink] = -9.0f;
    compareCpuCudaPhase(fixture, SPH_RUN_PREHYDRO2);
    require(!data.sinkSemiactive[excludedSink] && data.sinkDensity[excludedSink] == -9.0f,
        "distant inactive particle must leave density untouched");
    for (int source = 0; source < excluded; source++) {
        data.sourceDensity[source] = data.sinkDensity[sinkIndices[source]];
    }
    compareCpuCudaPhase(fixture, SPH_RUN_HYDRO, 0);
    compareCpuCudaPhase(fixture, SPH_RUN_HYDRO, 1);

    // The same periodic particles now exceed an intentionally unexpanded box.
    for (int tree = 0; tree < 3; tree++) {
        for (int dim = 0; dim < NDIM; dim++) {
            data.treeNodes[tree].outerBox.begin[dim] = 0.0;
            data.treeNodes[tree].outerBox.end[dim] = 1.0;
        }
    }
    for (int sink = 0; sink < fixture.count_; sink++) {
        data.sinkSmoothingConverged[sink] = false;
        data.sinkThermodynamicStates[sink].supportRadius = 0.1f;
    }
    compareCpuCudaPhase(fixture, SPH_RUN_PREHYDRO1);
    require(fixture.reduction_->pendingChanges > 0,
        "undersized neighbor boxes must request another pass on both backends");
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
// table as an independent CPU path. Core tests also check the table mathematics.
void testSelectedKernel() {
    Fixture fixture(2);
    auto& data = fixture.data_;
    data.sourcePositionX[0] = 0.40;
    data.sourcePositionX[1] = 0.58;
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
            "selected kernel CPU/CUDA density", data.sourceDensity[i], expectedDensity, 3.0e-6);
        gradient += 0.5 * dkernelWDq(radius / support) / std::pow(support, 4);
        pressurePair +=
            (data.adiabaticIndex - 1.0f) * data.sourceInternalEnergy[i] / data.sourceDensity[i];
        double const sound = std::sqrt(
            data.adiabaticIndex * (data.adiabaticIndex - 1.0f) * data.sourceInternalEnergy[i]);
        inverseCfl = std::max(
            inverseCfl, sound * selectedHydroKernel().supportRadiusPerEffectiveRadius / support);
    }
    fixture.hydro(1);
    requireClose("selected kernel CPU/CUDA pressure force",
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

void testZPlaneAcceleration() {
    Fixture fixture(1);
    int const source = 0;
    int const sink = 0;
    auto& data = fixture.data_;
    data.sourcePositionX[source] = 0.25;
    data.sourcePositionY[source] = 0.75;
    data.sourcePositionZ[source] = 0.5;
    data.sourceInternalEnergy[source] = 1.0e-6f;
    data.sinkSmoothingConverged[sink] = true;
    fixture.density();

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
        auto const& acceleration = data.sinkAccelerations[sink].acceleration;
        requireClose("midplane force has no x component", acceleration[XDIM], 0.0, 1.0e-7);
        requireClose("midplane force has no y component", acceleration[YDIM], 0.0, 1.0e-7);
        requireClose("midplane force points inward including adjacent fixed coordinates",
            acceleration[ZDIM], expectedAccelerations[i], 1.0e-7);
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
    requireClose("zero magnitude disables midplane force",
        data.sinkAccelerations[sink].acceleration[ZDIM], 0.0, 1.0e-7);

    fixture.params_.zPlaneAcceleration = 0.5f;
    fixture.params_.uniformAccelerationY = 0.25f;
    data.sinkGravityAccelerationX[sink] = 0.125f;
    data.sinkGravityAccelerationY[sink] = -0.0625f;
    data.sinkGravityAccelerationZ[sink] = 0.25f;
    for (bool const selfGravity : {false, true}) {
        data.selfGravityEnabled = selfGravity;
        fixture.hydro();
        auto const& acceleration = data.sinkAccelerations[sink].acceleration;
        requireClose("midplane force preserves enabled self gravity in x", acceleration[XDIM],
            selfGravity ? 0.125 : 0.0, 1.0e-7);
        requireClose("uniform y force remains independent of midplane force", acceleration[YDIM],
            selfGravity ? 0.1875 : 0.25, 1.0e-7);
        requireClose("midplane force adds to enabled self gravity in z", acceleration[ZDIM],
            selfGravity ? -0.25 : -0.5, 1.0e-7);
    }
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

}    // namespace detail

class MonaghanSphCuda : public detail::CudaTest {
protected:
    void SetUp() override {
        detail::CudaTest::SetUp();
        if (HasFatalFailure() || IsSkipped()) {
            return;
        }
        detail::checkCuda(cudaSetDevice(0), "select CUDA device");
        // The production allocator has process lifetime and no shutdown API.
        static bool allocatorInitialized = false;
        if (!allocatorInitialized) {
            cudaMemoryInitialize(4 * 1024 * 1024);
            detail::checkCuda(cudaGetLastError(), "production workspace allocator initialization");
            allocatorInitialized = true;
        }
        setKernelSelections(
            {radial_kernel::Family::Ferrers, 8}, {radial_kernel::Family::Spline, 3}, true);
    }
};

TEST_F(MonaghanSphCuda, PeriodicSupport) {
    detail::testPeriodicSupport();
}

TEST_F(MonaghanSphCuda, Lattice) {
    detail::testLattice();
}

TEST_F(MonaghanSphCuda, UnequalSupports) {
    detail::testUnequalSupports();
}

TEST_F(MonaghanSphCuda, ZPlaneAcceleration) {
    detail::testZPlaneAcceleration();
}

TEST_F(MonaghanSphCuda, DefaultTimestepCalibration) {
    detail::testDefaultTimestepCalibration();
}

TEST_F(MonaghanSphCuda, DefaultThermalTimestep) {
    detail::testDefaultThermalTimestep();
}

TEST_F(MonaghanSphCuda, CompositionEos) {
    detail::testCompositionEos();
}

TEST_F(MonaghanSphCuda, SelectedKernelsGravityTimestepsAndCpuParity) {
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
        setKernelSelections(gravity, hydro, true);
        detail::testSelectedKernel();
        detail::testSelectedCentralGravity();
        detail::testFixedGravityTimestep();
        detail::testCpuCudaParity();
    }
}
