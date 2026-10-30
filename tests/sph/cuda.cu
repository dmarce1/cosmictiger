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
#include <tests/cuda_test.hpp>
#include <tests/sph/global_timestep_cases.hpp>

// Only the options-constructor, CPU counter, and host assertion services are replaced here.
// Density, smoothing, forces, heating, device assertions, and allocation use the
// production translation units. The test does not start an HPX runtime.
void addCpuFlops(FlopCount) {}

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

// Count additions inside the production block reduction independently of its
// analytic cost. This catches the extra work in 96- and 160-thread SPH blocks.
struct CountedReductionValue {
    float value;
    FlopCount* operations;

    __device__ CountedReductionValue operator+(CountedReductionValue const& other) const {
        atomicAdd(operations, FlopCount(1));
        return {value + other.value, operations};
    }
};

template <int blockSize>
__global__ void countReductionOperations(FlopCount* executed, FlopCount* reported) {
    CountedReductionValue number{1.0f, executed};
    sharedReduceAdd<CountedReductionValue, blockSize>(number);
    atomicAdd(reported, sphReductionFlops<blockSize>());
}

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

    void checkFlops() const {
        require(reduction_->floatingPointOperations > 0, "SPH kernel did not count floating work");
        require(getGpuFlops() == reduction_->floatingPointOperations,
            "SPH launch count did not reach the global GPU counter exactly once");
    }

    void resetReduction() {
        resetGpuFlops();
        *reduction_ = {};
        reduction_->minimumSupportRadius = std::numeric_limits<float>::max();
        reduction_->minimumTimestep = std::numeric_limits<float>::max();
    }

public:
    int count_;
    SphRunParameters params_;
    SphRunCudaData data_{};
    SphReduction* reduction_;

    explicit Fixture(int count, int treeCount = 1, bool separateSources = false)
      : count_(count) {
        data_.sourcePositionX = allocate<Position>(count);
        data_.sourcePositionY = allocate<Position>(count);
        data_.sourcePositionZ = allocate<Position>(count);
        data_.sourceVelocityX = allocate<float>(count);
        data_.sourceVelocityY = allocate<float>(count);
        data_.sourceVelocityZ = allocate<float>(count);
        data_.sourceInternalEnergy = allocate<float>(count);
        data_.sourceSupportRadius = allocate<float>(count);
        data_.sourceDensity = allocate<float>(count);
        data_.sinkDensity = separateSources ? allocate<float>(count) : data_.sourceDensity;
        data_.sourceBalsaraFactor = allocate<float>(count);
        data_.sinkBalsaraFactor =
            separateSources ? allocate<float>(count) : data_.sourceBalsaraFactor;
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
        params_.balsaraSwitch = false;
        params_.variableViscosity = false;
        params_.cullenDehnen = false;
        params_.priceEtAl = false;
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
        sphCudaPrehydro1<<<1, kPrehydro1BlockSize>>>(params_, data_, reduction_);
        checkCuda(cudaGetLastError(), "prehydro1 launch");
        checkCuda(cudaDeviceSynchronize(), "prehydro1 execution");
        checkFlops();
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

    void enableVariableViscosity(bool cullenDehnen = false, bool priceEtAl = false) {
        params_.variableViscosity = true;
        params_.cullenDehnen = cullenDehnen;
        params_.priceEtAl = priceEtAl;
        params_.viscosityAlphaMin = 0.1f;
        params_.viscosityDecay = 0.2f;
        params_.viscosityAlphaMax = 2.0f;
        params_.viscosityDecayCD = 0.05f;
        params_.viscosityAlphaMaxPrice = 1.0f;
        params_.viscosityDecayPrice = 0.1f;
        data_.sourceViscosityAlpha = allocate<float>(count_);
        data_.sinkViscosityStates = allocate<SphViscosityState>(count_);
        if (cullenDehnen || priceEtAl) {
            data_.sourceAccelerationX = allocate<float>(count_);
            data_.sourceAccelerationY = allocate<float>(count_);
            data_.sourceAccelerationZ = allocate<float>(count_);
            if (cullenDehnen) {
                data_.sourceVelocityDivergence = allocate<float>(count_);
            }
            data_.sinkCullenDehnenStates = allocate<SphCullenDehnenState>(count_);
        }
        for (int i = 0; i < count_; ++i) {
            data_.sourceViscosityAlpha[sourceIndex(i)] = 0.7f;
            data_.sinkViscosityStates[sinkIndex(i)] = {0.7f, -9.0f, 0.6f};
        }
    }

    int sourceIndex(int index) const {
        return index;
    }

    int sinkIndex(int index) const {
        return index;
    }

    int particleIndex(int index) const {
        return index;
    }

    void densityPass() {
        resetReduction();
        sphCudaPrehydro2<<<1, kPrehydro2BlockSize>>>(params_, data_, reduction_);
        checkCuda(cudaGetLastError(), "viscosity density launch");
        checkCuda(cudaDeviceSynchronize(), "viscosity density execution");
        checkFlops();
        require(reduction_->pendingChanges == 0, "viscosity density needs no refresh");
    }

    void enableCompositionEos() {
        data_.sourcePressureEnergyRatio = allocate<float>(count_);
        data_.sourceAdiabaticIndex = allocate<float>(count_);
    }

    void density() {
        resetReduction();
        sphCudaPrehydro2<<<1, kPrehydro2BlockSize>>>(params_, data_, reduction_);
        checkCuda(cudaGetLastError(), "prehydro2 launch");
        checkCuda(cudaDeviceSynchronize(), "prehydro2 execution");
        checkFlops();
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
        sphCudaHydro<<<1, kHydroBlockSize>>>(params_, data_, reduction_);
        checkCuda(cudaGetLastError(), "hydro launch");
        checkCuda(cudaDeviceSynchronize(), "hydro execution");
        checkFlops();
        for (int i = 0; i < count_; i++) {
            require(std::isfinite(data_.sinkInternalEnergyRate[i]),
                "finite active-kernel thermal rate");
            for (int dim = 0; dim < kDimensions; dim++) {
                require(std::isfinite(data_.sinkAccelerations[i].acceleration[dim]),
                    "finite active-kernel acceleration");
            }
            if (phase == 1) {
                require(data_.sourceRungs[i] >= (params_.globalTimestep ? 0 : 1) &&
                        data_.sourceRungs[i] < kMaxRung,
                    "valid active-kernel timestep rung");
            }
        }
    }

    void checkConservation() const {
        std::array<double, kDimensions> momentum{};
        double energy = 0.0;
        double absoluteEnergy = 0.0;
        double absoluteForce = 0.0;
        for (int i = 0; i < count_; i++) {
            auto const& acceleration = data_.sinkAccelerations[i].acceleration;
            double const kinetic = data_.gasMass *
                (data_.sourceVelocityX[i] * acceleration[kXDimension] +
                    data_.sourceVelocityY[i] * acceleration[kYDimension] +
                    data_.sourceVelocityZ[i] * acceleration[kZDimension]);
            double const thermal = data_.gasMass * data_.sinkInternalEnergyRate[i];
            energy += kinetic + thermal;
            absoluteEnergy += std::abs(kinetic) + std::abs(thermal);
            for (int dim = 0; dim < kDimensions; dim++) {
                momentum[dim] += data_.gasMass * acceleration[dim];
                absoluteForce += std::abs(data_.gasMass * acceleration[dim]);
            }
        }
        for (int dim = 0; dim < kDimensions; dim++) {
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
    std::vector<float> balsaraFactor(
        data.sinkBalsaraFactor, data.sinkBalsaraFactor + fixture.count_);
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
    std::vector<float> const sourceBalsaraFactor(
        data.sourceBalsaraFactor, data.sourceBalsaraFactor + fixture.count_);
    std::vector<float> const sourceSupport(
        data.sourceSupportRadius, data.sourceSupportRadius + fixture.count_);
    std::vector<char> const sourceRungs(data.sourceRungs, data.sourceRungs + fixture.count_);
    std::vector<SphViscosityState> viscosityStates;
    std::vector<SphCullenDehnenState> cullenDehnenStates;
    std::vector<float> sourceAlpha;
    std::vector<float> sourceDivergence;
    std::array<std::vector<float>, 3> sourceAcceleration;
    if (fixture.params_.variableViscosity) {
        viscosityStates.assign(data.sinkViscosityStates, data.sinkViscosityStates + fixture.count_);
        cpuData.sinkViscosityStates = viscosityStates.data();
        sourceAlpha.assign(data.sourceViscosityAlpha, data.sourceViscosityAlpha + fixture.count_);
    }
    if (fixture.params_.gradientSwitch()) {
        cullenDehnenStates.assign(
            data.sinkCullenDehnenStates, data.sinkCullenDehnenStates + fixture.count_);
        cpuData.sinkCullenDehnenStates = cullenDehnenStates.data();
        if (data.sourceVelocityDivergence) {
            sourceDivergence.assign(
                data.sourceVelocityDivergence, data.sourceVelocityDivergence + fixture.count_);
        }
        float* const fields[] = {
            data.sourceAccelerationX, data.sourceAccelerationY, data.sourceAccelerationZ};
        for (int dim = 0; dim < kDimensions; ++dim) {
            sourceAcceleration[dim].assign(fields[dim], fields[dim] + fixture.count_);
        }
    }
    cpuData.sinkDensity = density.data();
    cpuData.sinkBalsaraFactor = balsaraFactor.data();
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
    resetGpuFlops();
    *fixture.reduction_ = {};
    fixture.reduction_->minimumSupportRadius = std::numeric_limits<float>::max();
    fixture.reduction_->minimumTimestep = std::numeric_limits<float>::max();
    switch (runType) {
    case kSphRunPrehydro1:
        sphCudaPrehydro1<<<1, kPrehydro1BlockSize>>>(fixture.params_, data, fixture.reduction_);
        break;
    case kSphRunPrehydro2:
        sphCudaPrehydro2<<<1, kPrehydro2BlockSize>>>(fixture.params_, data, fixture.reduction_);
        break;
    case kSphRunHydro:
        sphCudaHydro<<<1, kHydroBlockSize>>>(fixture.params_, data, fixture.reduction_);
        break;
    case kSphRunRungs:
        sphCudaRungs<<<1, kRungsBlockSize>>>(fixture.params_, data, fixture.reduction_);
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
    // SIMD executes padded lanes/all polynomial spans; CUDA takes scalar
    // branches and performs block reductions. Their work counts should differ.
    require(cpuResult.floatingPointOperations > 0.0 && cudaResult.floatingPointOperations > 0,
        "CPU/CUDA SPH must both count their floating work");
    require(getGpuFlops() == cudaResult.floatingPointOperations,
        "SPH phase did not reach the global GPU counter exactly once");
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
        if (fixture.params_.variableViscosity) {
            auto const& cpu = viscosityStates[i];
            auto const& gpu = data.sinkViscosityStates[i];
            requireClose("CPU/CUDA viscosity alpha", cpu.alpha, gpu.alpha, 0.0);
            requireClose("CPU/CUDA viscosity rate", cpu.rate, gpu.rate, 7.0e-5);
            requireClose(
                "CPU/CUDA predicted viscosity", cpu.predictedAlpha, gpu.predictedAlpha, 0.0);
            require(sourceAlpha[i] == data.sourceViscosityAlpha[i],
                "CPU/CUDA phases preserve viscosity source snapshots");
        }
        if (fixture.params_.gradientSwitch()) {
            auto const& cpu = cullenDehnenStates[i];
            auto const& gpu = data.sinkCullenDehnenStates[i];
            requireClose("CPU/CUDA previous divergence", cpu.divergence, gpu.divergence, 0.0);
            requireClose("CPU/CUDA reconstructed divergence", cpu.pendingDivergence,
                gpu.pendingDivergence, 3.0e-4);
            requireClose("CPU/CUDA CD viscosity target", cpu.targetAlpha, gpu.targetAlpha, 3.0e-4);
            requireClose(
                "CPU/CUDA CD decay rate", cpu.inverseDecayTime, gpu.inverseDecayTime, 7.0e-5);
            require((data.sourceVelocityDivergence == nullptr ||
                        sourceDivergence[i] == data.sourceVelocityDivergence[i]) &&
                    sourceAcceleration[0][i] == data.sourceAccelerationX[i] &&
                    sourceAcceleration[1][i] == data.sourceAccelerationY[i] &&
                    sourceAcceleration[2][i] == data.sourceAccelerationZ[i],
                "CPU/CUDA phases preserve CD history and acceleration snapshots");
        }
        requireClose("CPU/CUDA density", density[i], data.sinkDensity[i], 5.0e-5);
        requireClose(
            "CPU/CUDA Balsara factor", balsaraFactor[i], data.sinkBalsaraFactor[i], 5.0e-5);
        require(data.sourceBalsaraFactor[i] == sourceBalsaraFactor[i],
            "CPU/CUDA phases must preserve Balsara source snapshots");
        requireClose("CPU/CUDA support", thermodynamicStates[i].supportRadius,
            data.sinkThermodynamicStates[i].supportRadius, 3.0e-5);
        require(
            thermodynamicStates[i].internalEnergy == data.sinkThermodynamicStates[i].internalEnergy,
            "CPU/CUDA kernels preserve stored internal energy");
        requireClose(
            "CPU/CUDA thermal rate", energyRate[i], data.sinkInternalEnergyRate[i], 5.0e-5);
        for (int dim = 0; dim < kDimensions; dim++) {
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

void testCpuCudaParity(int viscosityMode = 0) {
    // Three unequal leaves and 19 neighbors exercise indirection and SIMD tails.
    Fixture fixture(19, 3, true);
    if (viscosityMode != 0) {
        fixture.enableVariableViscosity(viscosityMode == 2, viscosityMode == 3);
        fixture.params_.viscosityBeta = 3.5f;
    }
    fixture.params_.balsaraSwitch = true;
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
        if (viscosityMode != 0) {
            data.sourceViscosityAlpha[source] = 0.15f + 0.05f * source;
            data.sinkViscosityStates[sink] = {data.sourceViscosityAlpha[source], -9.0f, 0.1f};
        }
        if (viscosityMode >= 2) {
            data.sourceAccelerationX[source] = 0.02f * (source % 3 - 1);
            data.sourceAccelerationY[source] = -0.01f * (source % 5 - 2);
            data.sourceAccelerationZ[source] = 0.015f * (source % 2 ? 1 : -1);
            float const divergence = float(source % 3 - 1);
            if (viscosityMode == 2) {
                data.sourceVelocityDivergence[source] = divergence;
            }
            data.sinkCullenDehnenStates[sink] = {divergence, 0.0f, 0.0f, 0.0f};
        }
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
    compareCpuCudaPhase(fixture, kSphRunPrehydro1);
    publishSupport();
    compareCpuCudaPhase(fixture, kSphRunPrehydro2);
    publishSupport();
    for (int source = 0; source < fixture.count_; source++) {
        data.sourceDensity[source] = data.sinkDensity[sinkIndices[source]];
        data.sourceBalsaraFactor[source] = data.sinkBalsaraFactor[sinkIndices[source]];
    }
    compareCpuCudaPhase(fixture, kSphRunHydro, 0);
    compareCpuCudaPhase(fixture, kSphRunHydro, 1);
    fixture.params_.minimumRung = 1;
    for (int source = 0; source < fixture.count_; source++) {
        data.sinkParticleRungs[data.sinkParticleIndices[sinkIndices[source]]] = source == 0 ? 6 : 1;
    }
    publishRungs();
    compareCpuCudaPhase(fixture, kSphRunRungs);
    publishRungs();
    compareCpuCudaPhase(fixture, kSphRunRungs);

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
    compareCpuCudaPhase(fixture, kSphRunPrehydro2);
    require(!data.sinkSemiactive[excludedSink] && data.sinkDensity[excludedSink] == -9.0f,
        "distant inactive particle must leave density untouched");
    for (int source = 0; source < excluded; source++) {
        data.sourceDensity[source] = data.sinkDensity[sinkIndices[source]];
        data.sourceBalsaraFactor[source] = data.sinkBalsaraFactor[sinkIndices[source]];
    }
    compareCpuCudaPhase(fixture, kSphRunHydro, 0);
    compareCpuCudaPhase(fixture, kSphRunHydro, 1);

    // The same periodic particles now exceed an intentionally unexpanded box.
    for (int tree = 0; tree < 3; tree++) {
        for (int dim = 0; dim < kDimensions; dim++) {
            data.treeNodes[tree].outerBox.begin[dim] = 0.0;
            data.treeNodes[tree].outerBox.end[dim] = 1.0;
        }
    }
    for (int sink = 0; sink < fixture.count_; sink++) {
        data.sinkSmoothingConverged[sink] = false;
        data.sinkThermodynamicStates[sink].supportRadius = 0.1f;
    }
    compareCpuCudaPhase(fixture, kSphRunPrehydro1);
    require(fixture.reduction_->pendingChanges > 0,
        "undersized neighbor boxes must request another pass on both backends");
}

void initializeViscosityStencil(Fixture& fixture) {
    auto& data = fixture.data_;
    for (int i = 0; i < fixture.count_; ++i) {
        int const source = fixture.sourceIndex(i);
        int const sink = fixture.sinkIndex(i);
        data.sourcePositionX[source] = 0.5 + double(i / 9 - 1) * 0.0625;
        data.sourcePositionY[source] = 0.5 + double(i / 3 % 3 - 1) * 0.0625;
        data.sourcePositionZ[source] = 0.5 + double(i % 3 - 1) * 0.0625;
        data.sourceSupportRadius[source] = data.sinkThermodynamicStates[sink].supportRadius = 0.25f;
        data.sourceDensity[source] = 1.0f;
        data.sourceInternalEnergy[source] = 0.9f;
        data.sinkSmoothingConverged[sink] = true;
    }
}

void testMorrisMonaghanDensity() {
    Fixture fixture(27, 3, true);
    fixture.enableVariableViscosity();
    initializeViscosityStencil(fixture);
    auto& data = fixture.data_;
    int const center = fixture.sinkIndex(13);
    for (int i = 0; i < fixture.count_; ++i) {
        data.sourceRungs[fixture.sourceIndex(i)] =
            data.sinkParticleRungs[fixture.particleIndex(i)] = i == 13 ? 1 : 0;
    }
    // Continuum compression, pure shear, and background expansion have different
    // sources. The scalar sum retains the finite stencil's SPH consistency error.
    for (float const scaleFactor : {1.0f, 0.4f}) {
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
                            (expansion - compression) * radius * radius + 0.3 * x * y;
                        densityDivergence -= data.gasMass * velocityDotRadius *
                            dkernelWDq(radius / 0.25) / (std::pow(0.25, 4) * radius);
                    }
                }
                double const sound =
                    std::sqrt(double(data.adiabaticIndex) * (data.adiabaticIndex - 1.0) * 0.9);
                double const inverseDecayTime = 0.2 * sound / (scaleFactor * 0.125);
                double const expected = -0.6 * inverseDecayTime +
                    std::max(-densityDivergence / (density * scaleFactor), 0.0);
                fixture.densityPass();
                requireClose("M&M density source and conformal decay",
                    data.sinkViscosityStates[center].rate, expected, 2.0e-5);
                float const rate = data.sinkViscosityStates[center].rate;
                fixture.densityPass();
                requireClose("repeated density passes do not integrate alpha",
                    data.sinkViscosityStates[center].rate, rate, 0.0);
                for (int i = 0; i < fixture.count_; ++i) {
                    auto const& state = data.sinkViscosityStates[fixture.sinkIndex(i)];
                    require(state.alpha == 0.7f && state.predictedAlpha == 0.6f,
                        "density preserves viscosity integration state");
                    require(data.sourceViscosityAlpha[fixture.sourceIndex(i)] == 0.7f,
                        "density preserves the viscosity source snapshot");
                    if (i != 13) {
                        require(data.sinkSemiactive[fixture.sinkIndex(i)] && state.rate == -9.0f,
                            "semiactive M&M neighbors do not advance their viscosity rate");
                    }
                }
            }
        }
    }
    fixture.params_.variableViscosity = false;
    data.sourceViscosityAlpha = nullptr;
    data.sinkViscosityStates = nullptr;
    fixture.densityPass();
    require(data.sourceAccelerationX == nullptr && data.sourceVelocityDivergence == nullptr &&
            data.sinkCullenDehnenStates == nullptr,
        "M&M needs no persistent divergence or acceleration snapshot");
}

void testCullenDehnenDensity() {
    Fixture fixture(27, 3, true);
    fixture.enableVariableViscosity(true);
    initializeViscosityStencil(fixture);
    auto& data = fixture.data_;
    // Affine gradients are recovered exactly, including at the stencil boundary.
    // The third flow has strong shear and compressive acceleration; signs from
    // the preceding sweep must control equation (18)'s limiter.
    for (float const scaleFactor : {1.0f, 0.7f}) {
        for (int flow = 0; flow < 4; ++flow) {
            for (int history = -1; history <= 2; ++history) {
                double const a2 = double(scaleFactor) * scaleFactor;
                double const expansion = 0.12;
                double const expansionAcceleration = 0.03;
                fixture.params_.scaleFactor = scaleFactor;
                fixture.params_.expansionRate = float(expansion);
                fixture.params_.expansionAcceleration = float(expansionAcceleration);
                double gradient[3][3]{};
                double accelerationGradient[3][3]{};
                if (flow == 0 || flow == 1) {
                    for (int dim = 0; dim < kDimensions; ++dim) {
                        gradient[dim][dim] = -a2;
                        accelerationGradient[dim][dim] = flow == 1 ? a2 * a2 / scaleFactor : 0.0;
                    }
                } else if (flow == 2) {
                    gradient[0][0] = -a2;
                    gradient[0][1] = 10.0 * a2;
                } else {
                    gradient[0][1] = 4.0 * a2;
                }
                for (int dim = 0; dim < kDimensions; ++dim) {
                    accelerationGradient[dim][dim] -= expansionAcceleration / scaleFactor;
                }
                for (int i = 0; i < fixture.count_; ++i) {
                    int const source = fixture.sourceIndex(i);
                    int const sink = fixture.sinkIndex(i);
                    double const position[3] = {data.sourcePositionX[source].toDouble() - 0.5,
                        data.sourcePositionY[source].toDouble() - 0.5,
                        data.sourcePositionZ[source].toDouble() - 0.5};
                    float* velocity[] = {
                        data.sourceVelocityX, data.sourceVelocityY, data.sourceVelocityZ};
                    float* acceleration[] = {data.sourceAccelerationX, data.sourceAccelerationY,
                        data.sourceAccelerationZ};
                    for (int dim = 0; dim < kDimensions; ++dim) {
                        double value = 0.125 * (dim + 1) - expansion * position[dim];
                        double force = 0.01 * (dim + 1);
                        for (int axis = 0; axis < kDimensions; ++axis) {
                            value += gradient[dim][axis] * position[axis];
                            force += accelerationGradient[dim][axis] * position[axis];
                        }
                        velocity[dim][source] = float(value);
                        acceleration[dim][source] = float(force);
                    }
                    data.sourceVelocityDivergence[source] =
                        history == 2 ? (i == 13 ? -1.0f : 0.0f) : float(history);
                    data.sinkCullenDehnenStates[sink] = {9.0f, 8.0f, 7.0f, 6.0f};
                    data.sourceRungs[source] = data.sinkParticleRungs[fixture.particleIndex(i)] =
                        i == 13 ? 1 : 0;
                }
                fixture.densityPass();
                for (int i = 0; i < fixture.count_; ++i) {
                    int const source = fixture.sourceIndex(i);
                    int const sink = fixture.sinkIndex(i);
                    auto const& state = data.sinkCullenDehnenStates[sink];
                    double const expectedDivergence = flow < 2 ? -3.0 : (flow == 2 ? -1.0 : 0.0);
                    requireClose("CD affine physical divergence", state.pendingDivergence,
                        expectedDivergence, 3.0e-5);
                    require(state.divergence == 9.0f,
                        "CD density preserves previous divergence until all sinks finish");
                    require(data.sourceVelocityDivergence[source] ==
                            (history == 2 ? (i == 13 ? -1.0f : 0.0f) : float(history)),
                        "CD source divergence remains immutable");
                    require(data.sinkViscosityStates[sink].alpha == 0.7f &&
                            data.sinkViscosityStates[sink].rate == -9.0f,
                        "CD density does not apply the M&M update");
                    if (flow == 1 || flow == 3 || history == 1) {
                        requireClose("nonsteepening flow and expanding neighbor history",
                            state.targetAlpha, 0.0, 3.0e-5);
                    }
                }
                int const center = fixture.sinkIndex(13);
                double signAverage = history;
                if (history == 2) {
                    signAverage = -data.gasMass * kernelW(0.0) /
                        (std::pow(0.25, 3) * data.sinkDensity[center]);
                }
                double const divergence = flow < 2 ? -3.0 : (flow == 2 ? -1.0 : 0.0);
                double const shearSquared = flow == 2 ? 50.0 + 2.0 / 3.0 : (flow == 3 ? 8.0 : 0.0);
                double const numerator =
                    std::pow(2.0 * std::pow(1.0 - signAverage, 4) * divergence, 2);
                double const limiter =
                    numerator + shearSquared > 0 ? numerator / (numerator + shearSquared) : 0.0;
                double const detector = flow == 0 ? 3.0 : (flow == 2 ? 1.0 : 0.0);
                double signal =
                    std::sqrt(double(data.adiabaticIndex) * (data.adiabaticIndex - 1.0) * 0.9);
                for (int i = 0; i < fixture.count_; ++i) {
                    double const position[3] = {double(i / 9 - 1) * 0.0625,
                        double(i / 3 % 3 - 1) * 0.0625, double(i % 3 - 1) * 0.0625};
                    double radiusSquared = 0.0;
                    double projection = 0.0;
                    for (int dim = 0; dim < kDimensions; ++dim) {
                        radiusSquared += position[dim] * position[dim];
                        for (int axis = 0; axis < kDimensions; ++axis) {
                            projection += position[dim] * gradient[dim][axis] * position[axis];
                        }
                    }
                    if (radiusSquared > 0.0) {
                        double const sound = std::sqrt(
                            double(data.adiabaticIndex) * (data.adiabaticIndex - 1.0) * 0.9);
                        signal = std::max(
                            signal, sound - std::min(projection / std::sqrt(radiusSquared), 0.0));
                    }
                }
                double const supportSquaredDetector = 0.25 * 0.25 * a2 * a2 * limiter * detector;
                double const expectedTarget =
                    2.0 * supportSquaredDetector / (signal * signal + supportSquaredDetector);
                requireClose("CD shock target including self history and pair signal",
                    data.sinkCullenDehnenStates[center].targetAlpha, expectedTarget, 4.0e-5);
                requireClose("CD conformal decay uses full support",
                    data.sinkCullenDehnenStates[center].inverseDecayTime,
                    0.1 * signal / (scaleFactor * 0.25), 3.0e-5);
                float const oldTarget = data.sinkCullenDehnenStates[center].targetAlpha;
                fixture.densityPass();
                requireClose("CD repeated density preserves its source history",
                    data.sinkCullenDehnenStates[center].targetAlpha, oldTarget, 0.0);
            }
        }
    }
}

void testPriceEtAlDensity() {
    Fixture fixture(27, 3, true);
    fixture.enableVariableViscosity(false, true);
    initializeViscosityStencil(fixture);
    auto& data = fixture.data_;
    // Price et al. (2018), equations (46)-(48), on affine flows: steepening and
    // nonsteepening homologous compression, sheared compression, and pure shear.
    for (float const scaleFactor : {1.0f, 0.7f}) {
        for (int flow = 0; flow < 4; ++flow) {
            double const a2 = double(scaleFactor) * scaleFactor;
            double const expansion = 0.12;
            double const expansionAcceleration = 0.03;
            fixture.params_.scaleFactor = scaleFactor;
            fixture.params_.expansionRate = float(expansion);
            fixture.params_.expansionAcceleration = float(expansionAcceleration);
            double gradient[3][3]{};
            double accelerationGradient[3][3]{};
            if (flow == 0 || flow == 1) {
                for (int dim = 0; dim < kDimensions; ++dim) {
                    gradient[dim][dim] = -a2;
                    accelerationGradient[dim][dim] = flow == 1 ? a2 * a2 / scaleFactor : 0.0;
                }
            } else if (flow == 2) {
                gradient[0][0] = -a2;
                gradient[0][1] = 10.0 * a2;
            } else {
                gradient[0][1] = 4.0 * a2;
            }
            for (int dim = 0; dim < kDimensions; ++dim) {
                accelerationGradient[dim][dim] -= expansionAcceleration / scaleFactor;
            }
            for (int i = 0; i < fixture.count_; ++i) {
                int const source = fixture.sourceIndex(i);
                int const sink = fixture.sinkIndex(i);
                double const position[3] = {data.sourcePositionX[source].toDouble() - 0.5,
                    data.sourcePositionY[source].toDouble() - 0.5,
                    data.sourcePositionZ[source].toDouble() - 0.5};
                float* velocity[] = {
                    data.sourceVelocityX, data.sourceVelocityY, data.sourceVelocityZ};
                float* acceleration[] = {
                    data.sourceAccelerationX, data.sourceAccelerationY, data.sourceAccelerationZ};
                for (int dim = 0; dim < kDimensions; ++dim) {
                    double value = 0.125 * (dim + 1) - expansion * position[dim];
                    double force = 0.01 * (dim + 1);
                    for (int axis = 0; axis < kDimensions; ++axis) {
                        value += gradient[dim][axis] * position[axis];
                        force += accelerationGradient[dim][axis] * position[axis];
                    }
                    velocity[dim][source] = float(value);
                    acceleration[dim][source] = float(force);
                }
                data.sinkCullenDehnenStates[sink] = {9.0f, 8.0f, 7.0f, 6.0f};
                data.sourceRungs[source] = data.sinkParticleRungs[fixture.particleIndex(i)] =
                    i == 13 ? 1 : 0;
            }
            fixture.densityPass();
            for (int i = 0; i < fixture.count_; ++i) {
                int const sink = fixture.sinkIndex(i);
                auto const& state = data.sinkCullenDehnenStates[sink];
                double const expectedDivergence = flow < 2 ? -3.0 : (flow == 2 ? -1.0 : 0.0);
                requireClose("Price affine physical divergence", state.pendingDivergence,
                    expectedDivergence, 3.0e-5);
                require(state.divergence == 9.0f,
                    "Price density preserves previous divergence until all sinks finish");
                require(data.sinkViscosityStates[sink].alpha == 0.7f &&
                        data.sinkViscosityStates[sink].rate == -9.0f,
                    "Price density does not apply the M&M update");
                if (flow == 1 || flow == 3) {
                    requireClose(
                        "nonsteepening flow has no Price target", state.targetAlpha, 0.0, 3.0e-5);
                }
            }
            // Equation (47) from the reconstructed gradient: |curl v|² is 100 for the sheared
            // compression and 16 for pure shear, in the same a⁴ units as |div v|².
            int const center = fixture.sinkIndex(13);
            double const divergence = flow < 2 ? -3.0 : (flow == 2 ? -1.0 : 0.0);
            double const curlSquared = flow == 2 ? 100.0 : (flow == 3 ? 16.0 : 0.0);
            double const limiter = divergence != 0.0 ?
                divergence * divergence / (divergence * divergence + curlSquared) :
                0.0;
            double const detector = flow == 0 ? 3.0 : (flow == 2 ? 1.0 : 0.0);
            double const sound =
                std::sqrt(double(data.adiabaticIndex) * (data.adiabaticIndex - 1.0) * 0.9);
            double const viscosityLength =
                0.25 / selectedHydroKernel().supportRadiusPerEffectiveRadius * 0.9128709291752769;
            double const expectedTarget = std::min(1.0,
                10.0 * viscosityLength * viscosityLength * a2 * a2 * limiter * detector /
                    (sound * sound));
            requireClose("Price shock target uses the calibrated smoothing length and sound speed",
                data.sinkCullenDehnenStates[center].targetAlpha, expectedTarget, 4.0e-5);
            requireClose("Price conformal decay uses the sound speed",
                data.sinkCullenDehnenStates[center].inverseDecayTime,
                0.1 * sound / (scaleFactor * viscosityLength), 3.0e-5);
            require(data.sourceVelocityDivergence == nullptr,
                "Price et al. needs no neighbor divergence snapshot");
        }
    }
}

void testVariableViscosityPairs() {
    Fixture fixture(2, 2, true);
    fixture.enableVariableViscosity();
    auto& data = fixture.data_;
    for (int i = 0; i < 2; ++i) {
        int const source = fixture.sourceIndex(i);
        int const sink = fixture.sinkIndex(i);
        data.sourcePositionX[source] = i == 0 ? 0.35 : 0.65;
        data.sourcePositionY[source] = data.sourcePositionZ[source] = 0.5;
        data.sourceVelocityX[source] = i == 0 ? 0.1f : -0.1f;
        data.sourceDensity[source] = 1.0f + i;
        data.sourceSupportRadius[source] = i == 0 ? 0.2f : 0.45f;
        data.sourceViscosityAlpha[source] = i == 0 ? 0.2f : 1.4f;
        data.sinkViscosityStates[sink].alpha = data.sourceViscosityAlpha[source];
    }
    // Morris-Monaghan and Cullen-Dehnen pairs use beta_ab = viscosityBeta alpha_ab; Price
    // et al. pairs keep the constant beta of their equation (41).
    for (bool const priceEtAl : {false, true}) {
        for (float const beta : {0.0f, 2.0f, 3.5f}) {
            fixture.params_.variableViscosity = true;
            fixture.params_.priceEtAl = priceEtAl;
            fixture.params_.viscosityAlpha = 8.0f;    // Must not replace particle alpha.
            fixture.params_.viscosityBeta = beta;
            fixture.hydro();
            fixture.checkConservation();
            std::array<float, 2> heating;
            std::array<SphAcceleration, 2> force;
            for (int i = 0; i < 2; ++i) {
                heating[i] = data.sinkInternalEnergyRate[fixture.sinkIndex(i)];
                force[i] = data.sinkAccelerations[fixture.sinkIndex(i)];
            }
            fixture.params_.variableViscosity = false;
            fixture.params_.priceEtAl = false;
            fixture.params_.viscosityAlpha = 0.8f;
            fixture.params_.viscosityBeta = priceEtAl ? beta : 0.8f * beta;
            auto* sourceAlpha = data.sourceViscosityAlpha;
            auto* states = data.sinkViscosityStates;
            data.sourceViscosityAlpha = nullptr;
            data.sinkViscosityStates = nullptr;
            fixture.hydro();
            fixture.checkConservation();
            for (int i = 0; i < 2; ++i) {
                int const sink = fixture.sinkIndex(i);
                requireClose("variable pair alpha and beta heating", heating[i],
                    data.sinkInternalEnergyRate[sink], 1.0e-5);
                for (int dim = 0; dim < kDimensions; ++dim) {
                    requireClose("symmetric variable pair alpha and beta force",
                        force[i].acceleration[dim], data.sinkAccelerations[sink].acceleration[dim],
                        1.0e-5);
                }
            }
            data.sourceViscosityAlpha = sourceAlpha;
            data.sinkViscosityStates = states;
        }
    }
}

void testCullenDehnenExcludedThermodynamics() {
    Fixture fixture(2, 2, true);
    fixture.enableVariableViscosity(true);
    fixture.params_.viscosityBeta = 4.0f;
    auto& data = fixture.data_;
    for (int i = 0; i < 2; ++i) {
        int const source = fixture.sourceIndex(i);
        int const sink = fixture.sinkIndex(i);
        data.sourcePositionX[source] = i == 0 ? 0.1 : 0.45;
        data.sourcePositionY[source] = data.sourcePositionZ[source] = 0.5;
        data.sourceSupportRadius[source] = data.sinkThermodynamicStates[sink].supportRadius = 0.1f;
        data.sourceDensity[source] = 1.0f;
        data.sourceRungs[source] = data.sinkParticleRungs[fixture.particleIndex(i)] =
            i == 0 ? 1 : 0;
        data.sourceInternalEnergy[source] = i == 0 ? 0.9f : -1.0f;
        data.sourceViscosityAlpha[source] = i == 0 ? 0.7f : 1.0e38f;
        data.sinkSmoothingConverged[sink] = true;
        data.sinkDensity[sink] = -9.0f;
        data.sinkCullenDehnenStates[sink] = {3.0f, 4.0f, 5.0f, 6.0f};
    }
    // The invalid energy and enormous alpha belong to a gathered source outside
    // every support. Neither its sound speed nor its beta may be evaluated.
    fixture.densityPass();
    int const active = fixture.sinkIndex(0);
    int const inactive = fixture.sinkIndex(1);
    requireClose("isolated active CD density", data.sinkDensity[active],
        data.gasMass * kernelW(0.0) / std::pow(0.1, 3), 2.0e-5);
    requireClose("unresolved CD neighborhood divergence",
        data.sinkCullenDehnenStates[active].pendingDivergence, 0.0, 0.0);
    require(data.sinkDensity[inactive] == -9.0f && !data.sinkSemiactive[inactive],
        "excluded inactive thermodynamics remain untouched");
    auto const& state = data.sinkCullenDehnenStates[inactive];
    require(state.divergence == 3.0f && state.pendingDivergence == 4.0f &&
            state.targetAlpha == 5.0f && state.inverseDecayTime == 6.0f,
        "excluded inactive CD history remains untouched");
    fixture.hydro();
    requireClose("excluded alpha leaves isolated heating zero", data.sinkInternalEnergyRate[active],
        0.0, 0.0);
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
    Array<Position, kDimensions> source;
    for (int dim = 0; dim < kDimensions; dim++) {
        box.begin[dim] = 0.49;
        box.end[dim] = 0.51;
        source[dim] = 0.5;
    }
    box.begin[kXDimension] = 0.90;
    box.end[kXDimension] = 0.95;
    source[kXDimension] = 0.40;
    require(sphSupportIntersects(box, source, 0.49f),
        "periodic support reaches the far end of a sink box");
    require(!sphSupportIntersects(box, source, 0.40f),
        "periodic support excludes a genuinely separated box");
    source[kXDimension] = 0.02;
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
        for (int dim = 0; dim < kDimensions; dim++) {
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
        forceNorm += std::abs(fixture.data_.sinkAccelerations[i].acceleration[kXDimension]);
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
        for (int dim = 0; dim < kDimensions; dim++) {
            double const position = dim == kXDimension ? data.sourcePositionX[i].toDouble() : 0.5;
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
    require(data.sinkAccelerations[0].acceleration[kXDimension] < 0.0f &&
            data.sinkAccelerations[1].acceleration[kXDimension] > 0.0f,
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
        data.sinkAccelerations[0].acceleration[kXDimension], data.gasMass * pressurePair * gradient,
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
        data.sinkAccelerations[0].acceleration[kXDimension], data.gasMass * pressurePair * gradient,
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
    double const radius = distance(data.sourcePositionX[0], Position(0.5));
    double const expected = -data.centralAcceleration * radius /
        std::pow(double(data.centralSupportRadius), 3) *
        radial_kernel::forceFactor(selectedGravityKernel(), radius / data.centralSupportRadius);
    requireClose("selected gravity kernel central force",
        data.sinkAccelerations[0].acceleration[kXDimension], expected, 5.0e-6);
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
#if COSMICTIGER_WITH_GRAVITY
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
#else
    require(fixture.reduction_->maximumGravityRung == 0,
        "SPH-only builds must not report a self-gravity timestep");
    requireClose("SPH-only ignores stored self-gravity",
        data.sinkAccelerations[0].acceleration[kXDimension], 0.0, 0.0);
    int const hydroRung = fixture.reduction_->maximumHydroRung;
    data.sinkGravityAccelerationX[0] = 1.0e20f;
    data.gravityEffectiveRadius = 1.0e-20f;
    fixture.hydro(1);
    require(fixture.reduction_->maximumGravityRung == 0 &&
            fixture.reduction_->maximumHydroRung == hydroRung &&
            fixture.reduction_->maximumRung == hydroRung,
        "stored self-gravity must not constrain SPH-only timesteps");
    requireClose("SPH-only ignores changed stored self-gravity",
        data.sinkAccelerations[0].acceleration[kXDimension], 0.0, 0.0);
#endif
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
        requireClose("midplane force has no x component", acceleration[kXDimension], 0.0, 1.0e-7);
        requireClose("midplane force has no y component", acceleration[kYDimension], 0.0, 1.0e-7);
        requireClose("midplane force points inward including adjacent fixed coordinates",
            acceleration[kZDimension], expectedAccelerations[i], 1.0e-7);
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
        data.sinkAccelerations[sink].acceleration[kZDimension], 0.0, 1.0e-7);

    fixture.params_.zPlaneAcceleration = 0.5f;
    fixture.params_.uniformAccelerationY = 0.25f;
    data.sinkGravityAccelerationX[sink] = 0.125f;
    data.sinkGravityAccelerationY[sink] = -0.0625f;
    data.sinkGravityAccelerationZ[sink] = 0.25f;
    for (bool const selfGravity : {false, true}) {
        data.selfGravityEnabled = selfGravity;
        fixture.hydro();
        auto const& acceleration = data.sinkAccelerations[sink].acceleration;
        bool const expectSelfGravity = COSMICTIGER_WITH_GRAVITY && selfGravity;
        requireClose("midplane force preserves enabled self gravity in x",
            acceleration[kXDimension], expectSelfGravity ? 0.125 : 0.0, 1.0e-7);
        requireClose("uniform y force remains independent of midplane force",
            acceleration[kYDimension], expectSelfGravity ? 0.1875 : 0.25, 1.0e-7);
        requireClose("midplane force adds to enabled self gravity in z", acceleration[kZDimension],
            expectSelfGravity ? -0.25 : -0.5, 1.0e-7);
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
        double const acceleration = std::abs(data.sinkAccelerations[i].acceleration[kXDimension]);
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

TEST_F(MonaghanSphCuda, VariableViscosityCpuParity) {
    detail::testCpuCudaParity(1);
    detail::testCpuCudaParity(2);
    detail::testCpuCudaParity(3);
}

TEST_F(MonaghanSphCuda, CullenDehnenExcludedThermodynamics) {
    detail::testCullenDehnenExcludedThermodynamics();
}

TEST_F(MonaghanSphCuda, MorrisMonaghanDensity) {
    detail::testMorrisMonaghanDensity();
}

TEST_F(MonaghanSphCuda, CullenDehnenDensity) {
    detail::testCullenDehnenDensity();
}

TEST_F(MonaghanSphCuda, PriceEtAlDensity) {
    detail::testPriceEtAlDensity();
}

TEST_F(MonaghanSphCuda, VariableViscosityPairs) {
    detail::testVariableViscosityPairs();
}

TEST_F(MonaghanSphCuda, BalsaraVelocityGradients) {
    detail::Fixture fixture(27, 3);
    fixture.params_.balsaraSwitch = true;
    auto& data = fixture.data_;
    struct Flow {
        float compression;
        float rotation;
        float shear;
        float expansion;
        float expected;
    };
    Flow const flows[] = {{0, 0, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 1, 0, 0, 0}, {1, 0, 0, 0, 1},
        {1, 1, 0, 0, 0.6f}, {0, 0, 0, -1, 1}, {1, 1, 0, 0.5f, 3.0f / 7.0f}};
    for (auto const& flow : flows) {
        fixture.params_.expansionRate = flow.expansion;
        for (int i = 0; i < fixture.count_; ++i) {
            float const x = float(i / 9 - 1) * 0.0625f;
            float const y = float(i / 3 % 3 - 1) * 0.0625f;
            float const z = float(i % 3 - 1) * 0.0625f;
            data.sourcePositionX[i] = 0.5 + x;
            data.sourcePositionY[i] = 0.5 + y;
            data.sourcePositionZ[i] = 0.5 + z;
            data.sourceVelocityX[i] =
                0.25f - flow.compression * x - flow.rotation * y + flow.shear * y;
            data.sourceVelocityY[i] = -0.5f - flow.compression * y + flow.rotation * x;
            data.sourceVelocityZ[i] = 0.125f - flow.compression * z;
            data.sourceSupportRadius[i] = data.sinkThermodynamicStates[i].supportRadius = 0.25f;
            data.sinkSmoothingConverged[i] = true;
        }
        fixture.density();
        EXPECT_NEAR(data.sourceBalsaraFactor[13], flow.expected, 2.0e-6f);
        for (int i = 0; i < fixture.count_; ++i) {
            EXPECT_GE(data.sourceBalsaraFactor[i], 0.0f);
            EXPECT_LE(data.sourceBalsaraFactor[i], 1.0f);
        }
    }
    fixture.params_.balsaraSwitch = false;
    fixture.density();
    for (int i = 0; i < fixture.count_; ++i) {
        EXPECT_EQ(data.sourceBalsaraFactor[i], 1.0f);
    }
}

TEST_F(MonaghanSphCuda, BalsaraSymmetricForceAndHeating) {
    detail::Fixture fixture(2, 2);
    auto& data = fixture.data_;
    for (int i = 0; i < 2; ++i) {
        data.sourcePositionX[i] = i == 0 ? 0.35 : 0.65;
        data.sourcePositionY[i] = data.sourcePositionZ[i] = 0.5;
        data.sourceVelocityX[i] = i == 0 ? 0.1f : -0.1f;
        data.sourceDensity[i] = 1.0f + i;
        data.sourceSupportRadius[i] = i == 0 ? 0.2f : 0.45f;
        data.sourceBalsaraFactor[i] = i == 0 ? 0.2f : 0.8f;
    }
    fixture.hydro();
    std::array<float, 2> const fullHeating = {
        data.sinkInternalEnergyRate[0], data.sinkInternalEnergyRate[1]};
    std::array<SphAcceleration, 2> const fullForce = {
        data.sinkAccelerations[0], data.sinkAccelerations[1]};
    fixture.params_.viscosityAlpha = fixture.params_.viscosityBeta = 0.0f;
    fixture.hydro();
    std::array<float, 2> const pressureHeating = {
        data.sinkInternalEnergyRate[0], data.sinkInternalEnergyRate[1]};
    std::array<SphAcceleration, 2> const pressureForce = {
        data.sinkAccelerations[0], data.sinkAccelerations[1]};
    fixture.params_.viscosityAlpha = 1.0f;
    fixture.params_.viscosityBeta = 2.0f;
    fixture.params_.balsaraSwitch = true;
    fixture.hydro();
    fixture.checkConservation();
    for (int i = 0; i < 2; ++i) {
        EXPECT_GT(fullHeating[i], pressureHeating[i]);
        detail::requireClose("Balsara pair heating", data.sinkInternalEnergyRate[i],
            pressureHeating[i] + 0.5f * (fullHeating[i] - pressureHeating[i]), 5.0e-6);
        for (int dim = 0; dim < kDimensions; ++dim) {
            detail::requireClose("Balsara pair force", data.sinkAccelerations[i].acceleration[dim],
                pressureForce[i].acceleration[dim] +
                    0.5f * (fullForce[i].acceleration[dim] - pressureForce[i].acceleration[dim]),
                5.0e-6);
        }
    }
}

TEST_F(MonaghanSphCuda, BlockReductionFlops) {
    FlopCount* counts = nullptr;
    ASSERT_EQ(cudaMallocManaged(&counts, 2 * sizeof(FlopCount)), cudaSuccess);
    for (int blockSize : {64, 96, 160, 256}) {
        counts[0] = counts[1] = 0;
        switch (blockSize) {
        case 64:
            detail::countReductionOperations<64><<<1, 64>>>(counts, counts + 1);
            break;
        case 96:
            detail::countReductionOperations<96><<<1, 96>>>(counts, counts + 1);
            break;
        case 160:
            detail::countReductionOperations<160><<<1, 160>>>(counts, counts + 1);
            break;
        case 256:
            detail::countReductionOperations<256><<<1, 256>>>(counts, counts + 1);
            break;
        }
        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(counts[0], counts[1]) << "block size " << blockSize;
    }
    EXPECT_EQ(cudaFree(counts), cudaSuccess);
}

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

TEST_F(MonaghanSphCuda, GlobalRawTimestep) {
    detail::testGlobalRawTimestep<detail::Fixture>();
}
