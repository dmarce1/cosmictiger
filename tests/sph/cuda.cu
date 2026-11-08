/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/sph/cpu.hpp>
#include <cosmictiger/sph/cuda.hpp>
#include <cosmictiger/sph/viscosity/balsara.hpp>
#include <cosmictiger/sph/viscosity/monaghan.hpp>
#include <tests/cuda_test.hpp>
#include <tests/sph/conduction_cases.hpp>
#include <tests/sph/global_timestep_cases.hpp>
#include <tests/sph/midpoint_cases.hpp>
#include <tests/sph/pair_kick_cases.hpp>
#include <tests/sph/rung_limiter_cases.hpp>
#include <tests/sph/source_completion_cases.hpp>

// Only the options-constructor, CPU counter, and host assertion services are replaced here.
// Density, smoothing, forces, heating, device assertions, and allocation use the
// production translation units. The test does not start an HPX runtime.
void addCpuFlops(FlopCount, FlopPrecision) {}

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

__global__ void accumulateCompensatedFloat(SphCompensatedSum* sum, FlopCount* counts) {
    FlopCount operations = 0;
    for (int i = 0; i < 256; ++i) {
        operations += sum->add(0x1p-25f);
    }
    counts[blockIdx.x * blockDim.x + threadIdx.x] = operations;
}

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
        require(getGpuFlopCounts().unclassified == 0,
            "SPH launch retained unclassified floating-point work");
    }

    void resetReduction() {
        resetGpuFlops();
        *reduction_ = {};
        reduction_->minimumSmoothingLength = std::numeric_limits<float>::max();
        reduction_->minimumTimestep = std::numeric_limits<float>::max();
        reduction_->minimumGradHOmega = std::numeric_limits<float>::max();
    }

public:
    int count_;
    SphRunParameters params_;
    SphRunCudaData data_{};
    SphReduction* reduction_;
    SphPairSourceId* pairSourceIds_ = nullptr;

    explicit Fixture(int count, int treeCount = 1, bool separateSources = false)
      : count_(count) {
        data_.sourcePositionX = allocate<Position>(count);
        data_.sourcePositionY = allocate<Position>(count);
        data_.sourcePositionZ = allocate<Position>(count);
        data_.sourceVelocityX = allocate<float>(count);
        data_.sourceVelocityY = allocate<float>(count);
        data_.sourceVelocityZ = allocate<float>(count);
        data_.sourceInternalEnergy = allocate<float>(count);
        data_.sourceSmoothingLength = allocate<float>(count);
        data_.sourceDensity = allocate<float>(count);
        data_.sourceGradHCorrection = allocate<float>(count);
        data_.sinkGradHCorrection =
            separateSources ? allocate<float>(count) : data_.sourceGradHCorrection;
        data_.sinkDensity = separateSources ? allocate<float>(count) : data_.sourceDensity;
        data_.sourceBalsaraFactor = allocate<float>(count);
        data_.sinkBalsaraFactor = allocate<uint8_t>(count);
        data_.sourceBulkViscosityCoefficient = allocate<float>(count);
        data_.sinkBulkViscosityCoefficient = allocate<Float16Storage>(count);
        data_.sinkGravityAccelerationX = allocate<float>(count);
        data_.sinkGravityAccelerationY = allocate<float>(count);
        data_.sinkGravityAccelerationZ = allocate<float>(count);
        data_.sinkSmoothingLength = allocate<float>(count);
        data_.sourceRungs = allocate<char>(count);
        data_.sinkParticleRungs = separateSources ? allocate<char>(count) : data_.sourceRungs;
        data_.sinkFlags = allocate<SphParticleFlags>(count);
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
        params_.vonNeumannRichtmyerCoefficient = 2.0f;
        params_.viscosityEpsilon = 0.01f;
        params_.viscosityDecay = 0.2f;
        params_.minimumSmoothingLength = 0.1f;
        // The initial isolated support is held at the upper bound.
        params_.maximumSmoothingLength = 0.2f;
        for (int i = 0; i < count; i++) {
            data_.sinkParticleIndices[i] = i;
            data_.sourceBalsaraFactor[i] = 1.0f;
            data_.sinkBalsaraFactor[i] = 255;
            data_.sourceRungs[i] = data_.sinkParticleRungs[i] = 1;
            data_.sourceInternalEnergy[i] = 1.0f;
            data_.sourceSmoothingLength[i] = data_.sinkSmoothingLength[i] = 0.2f;
            data_.sinkGravityAccelerationX[i] = data_.sinkGravityAccelerationY[i] =
                data_.sinkGravityAccelerationZ[i] = 0.0f;
        }
        setBulkViscosityCoefficient(1.0f);
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
        params_.minimumSmoothingLength = minimum;
        params_.maximumSmoothingLength = maximum;
        for (int i = 0; i < count_; i++) {
            data_.sinkFlags[i].setConverged(false);
            data_.sourceSmoothingLength[i] = data_.sinkSmoothingLength[i] = initial;
        }
        resetReduction();
        sphCudaSmoothingLengths<<<1, kSmoothingLengthsBlockSize>>>(params_, data_, reduction_);
        checkCuda(cudaGetLastError(), "smoothing lengths launch");
        checkCuda(cudaDeviceSynchronize(), "smoothing lengths execution");
        checkFlops();
        require(reduction_->pendingChanges == 0,
            "complete neighbor fixture must converge without refresh");
        for (int i = 0; i < count_; i++) {
            require(data_.sinkFlags[i].converged(), "smoothing solve did not mark convergence");
            data_.sourceSmoothingLength[i] = data_.sinkSmoothingLength[i];
            require(data_.sourceSmoothingLength[i] >= minimum &&
                    data_.sourceSmoothingLength[i] <= maximum,
                "smoothing solve exceeded configured bounds");
        }
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
        sphCudaDensityAndGradients<<<1, kDensityAndGradientsBlockSize>>>(
            params_, data_, reduction_);
        checkCuda(cudaGetLastError(), "viscosity density launch");
        checkCuda(cudaDeviceSynchronize(), "viscosity density execution");
        checkFlops();
        require(reduction_->pendingChanges == 0, "viscosity density needs no refresh");
    }

    void smoothingLengthsPass() {
        resetReduction();
        sphCudaSmoothingLengths<<<1, kSmoothingLengthsBlockSize>>>(params_, data_, reduction_);
        checkCuda(cudaGetLastError(), "midpoint support launch");
        checkCuda(cudaDeviceSynchronize(), "midpoint support execution");
        checkFlops();
        require(reduction_->pendingChanges == 0, "midpoint support needs no refresh");
    }

    void enableEntropyRateOutput() {
        if (!data_.sinkEntropyRate) {
            data_.sinkEntropyRate = allocate<float>(count_);
        }
    }

    void enableMidpointOutputs() {
        enableEntropyRateOutput();
        data_.proposeRungs = true;
        for (int i = 0; i < count_; i++) {
            data_.sinkFlags[i].setProposedRung(kMaxRung - 1);
        }
    }

    SphPairSourceId* enablePairKickOutputs() {
        pairSourceIds_ = allocate<SphPairSourceId>(count_);
        data_.sourceIds = pairSourceIds_;
        data_.sourcePairUpdates = allocate<SphPairUpdate>(count_);
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
        data_.sinkRungCandidates = allocate<char>(count_);
    }

    void density() {
        resetReduction();
        sphCudaDensityAndGradients<<<1, kDensityAndGradientsBlockSize>>>(
            params_, data_, reduction_);
        checkCuda(cudaGetLastError(), "density and gradients launch");
        checkCuda(cudaDeviceSynchronize(), "density and gradients execution");
        checkFlops();
        require(
            reduction_->pendingChanges == 0, "density fixture unexpectedly needs neighbor refresh");
        for (int i = 0; i < count_; i++) {
            require(data_.sourceDensity[i] > 0.0f && std::isfinite(data_.sourceDensity[i]),
                "positive finite density");
            data_.sourceBalsaraFactor[i] = balsara::decodeFactor(data_.sinkBalsaraFactor[i]);
        }
    }

    void hydro(int phase = 0) {
        params_.phase = phase;
        if (data_.sourcePairUpdates) {
            clearPairUpdates();    // Each call reports the kicks of one pass.
        }
        resetReduction();
        sphCudaHydro<<<data_.sourcePairUpdates ? 2 : 1, kHydroBlockSize>>>(
            params_, data_, reduction_);
        checkCuda(cudaGetLastError(), "hydro launch");
        checkCuda(cudaDeviceSynchronize(), "hydro execution");
        reduction_->timestepError.throwIfFailed([](SphTimestepFailure const&) {
            throw std::runtime_error("SPH timestep exceeds rung range");
        });
        checkFlops();
        for (int i = 0; i < count_; i++) {
            if (data_.sinkEntropyRate) {
                require(
                    std::isfinite(data_.sinkEntropyRate[i]), "finite active-kernel entropy rate");
            }
            if (data_.sourcePairUpdates) {
                auto const& update = data_.sourcePairUpdates[i];
                require(std::isfinite(float(update.entropy)), "finite active-kernel entropy kick");
                for (int dim = 0; dim < kSpatialDimensions; dim++) {
                    require(std::isfinite(float(update.velocity[dim])),
                        "finite active-kernel pair kick");
                }
            }
            if (phase == 1) {
                // A midpoint pass proposes from minimumProposedRung; a closing pass keeps rung 1.
                int const lowestRung = params_.globalTimestep || params_.midpointHydro ? 0 : 1;
                require(data_.sourceRungs[i] >= lowestRung && data_.sourceRungs[i] < kMaxRung,
                    "valid active-kernel timestep rung");
            }
        }
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
    std::vector<uint8_t> balsaraFactor(
        data.sinkBalsaraFactor, data.sinkBalsaraFactor + fixture.count_);
    std::vector<float> gradHCorrection;
    std::vector<float> sourceGradHCorrection;
    gradHCorrection.assign(data.sinkGradHCorrection, data.sinkGradHCorrection + fixture.count_);
    sourceGradHCorrection.assign(
        data.sourceGradHCorrection, data.sourceGradHCorrection + fixture.count_);
    cpuData.sinkGradHCorrection = gradHCorrection.data();
    std::vector<float> entropyRate;
    if (data.sinkEntropyRate) {
        entropyRate.assign(data.sinkEntropyRate, data.sinkEntropyRate + fixture.count_);
        cpuData.sinkEntropyRate = entropyRate.data();
    }
    // Both backends start the pass from cleared pair updates; the CPU writes a host copy.
    bool const comparesKicks = data.sourcePairUpdates && fixture.params_.hydroKickTimestep > 0.0;
    std::vector<SphPairUpdate> pairUpdates;
    if (data.sourcePairUpdates) {
        fixture.clearPairUpdates();
        pairUpdates.assign(data.sourcePairUpdates, data.sourcePairUpdates + fixture.count_);
        cpuData.sourcePairUpdates = pairUpdates.data();
    }
    std::vector<SphParticleFlags> flags(data.sinkFlags, data.sinkFlags + fixture.count_);
    cpuData.sinkFlags = flags.data();
    std::vector<char> rungCandidates;
    if (data.sinkRungCandidates) {
        rungCandidates.assign(data.sinkRungCandidates, data.sinkRungCandidates + fixture.count_);
        cpuData.sinkRungCandidates = rungCandidates.data();
    }
    std::vector<float> smoothingLengths(
        data.sinkSmoothingLength, data.sinkSmoothingLength + fixture.count_);
    std::vector<char> rungs(data.sinkParticleRungs, data.sinkParticleRungs + fixture.count_);
    std::vector<float> const sourceDensity(data.sourceDensity, data.sourceDensity + fixture.count_);
    std::vector<float> const sourceBalsaraFactor(
        data.sourceBalsaraFactor, data.sourceBalsaraFactor + fixture.count_);
    std::vector<float> const sourceSmoothingLengths(
        data.sourceSmoothingLength, data.sourceSmoothingLength + fixture.count_);
    std::vector<char> const sourceRungs(data.sourceRungs, data.sourceRungs + fixture.count_);
    std::vector<Float16Storage> bulkViscosityCoefficients;
    std::vector<float> sourceAlpha;
    {
        bulkViscosityCoefficients.assign(
            data.sinkBulkViscosityCoefficient, data.sinkBulkViscosityCoefficient + fixture.count_);
        cpuData.sinkBulkViscosityCoefficient = bulkViscosityCoefficients.data();
        sourceAlpha.assign(data.sourceBulkViscosityCoefficient,
            data.sourceBulkViscosityCoefficient + fixture.count_);
    }
    cpuData.sinkDensity = density.data();
    cpuData.sinkBalsaraFactor = balsaraFactor.data();
    cpuData.sinkSmoothingLength = smoothingLengths.data();
    cpuData.sinkParticleRungs = rungs.data();
    SphRunReturn cpuResult;
    for (int node = 0; node < data.numberOfTargetNodes; node++) {
        cpuResult += sphRunCpuNode(fixture.params_, cpuData, node);
    }
    resetGpuFlops();
    *fixture.reduction_ = {};
    fixture.reduction_->minimumSmoothingLength = std::numeric_limits<float>::max();
    fixture.reduction_->minimumTimestep = std::numeric_limits<float>::max();
    fixture.reduction_->minimumGradHOmega = std::numeric_limits<float>::max();
    switch (runType) {
    case kSphRunSmoothingLengths:
        sphCudaSmoothingLengths<<<1, kSmoothingLengthsBlockSize>>>(
            fixture.params_, data, fixture.reduction_);
        break;
    case kSphRunDensityAndGradients:
        sphCudaDensityAndGradients<<<1, kDensityAndGradientsBlockSize>>>(
            fixture.params_, data, fixture.reduction_);
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
    requireClose("CPU/CUDA minimum support", cpuResult.minimumSmoothingLength,
        cudaResult.minimumSmoothingLength, 3.0e-5);
    requireClose("CPU/CUDA maximum support", cpuResult.maximumSmoothingLength,
        cudaResult.maximumSmoothingLength, 3.0e-5);
    requireClose("CPU/CUDA signal speed", cpuResult.maximumSignalSpeed,
        cudaResult.maximumSignalSpeed, 5.0e-5);
    requireClose("CPU/CUDA sound timestep", cpuResult.inverseSoundTimestep,
        cudaResult.inverseSoundTimestep, 5.0e-5);
    requireClose("CPU/CUDA viscosity timestep", cpuResult.inverseViscosityTimestep,
        cudaResult.inverseViscosityTimestep, 5.0e-5);
    requireClose("CPU/CUDA acceleration timestep", cpuResult.inverseAccelerationTimestep,
        cudaResult.inverseAccelerationTimestep, 5.0e-5);
    require(cpuResult.heldSupports == cudaResult.heldSupports &&
            cpuResult.illConditionedSupports == cudaResult.illConditionedSupports,
        "CPU/CUDA held and ill-conditioned support counts");
    requireClose("CPU/CUDA minimum grad-H Omega", cpuResult.minimumGradHOmega,
        cudaResult.minimumGradHOmega, 5.0e-5);
    requireClose("CPU/CUDA support residual", cpuResult.maximumSupportResidual,
        cudaResult.maximumSupportResidual, 1.0e-6);
    for (int i = 0; i < fixture.count_; i++) {
        requireClose("CPU/CUDA grad-H correction", gradHCorrection[i] / data.gasMass,
            data.sinkGradHCorrection[i] / data.gasMass, 5.0e-5);
        require(std::bit_cast<uint32_t>(data.sourceGradHCorrection[i]) ==
                std::bit_cast<uint32_t>(sourceGradHCorrection[i]),
            "CPU/CUDA phases preserve grad-H source snapshots");
        {
            auto const& cpu = bulkViscosityCoefficients[i];
            auto const& gpu = data.sinkBulkViscosityCoefficient[i];
            requireClose("CPU/CUDA bulk viscosity coefficient", cpu, gpu, 7.0e-5);
            require(sourceAlpha[i] == data.sourceBulkViscosityCoefficient[i],
                "CPU/CUDA phases preserve viscosity source snapshots");
        }
        requireClose("CPU/CUDA density", density[i], data.sinkDensity[i], 5.0e-5);
        require(balsaraFactor[i] == data.sinkBalsaraFactor[i], "CPU/CUDA Balsara byte parity");
        require(data.sourceBalsaraFactor[i] == sourceBalsaraFactor[i],
            "CPU/CUDA phases must preserve Balsara source snapshots");
        requireClose("CPU/CUDA support", smoothingLengths[i], data.sinkSmoothingLength[i], 3.0e-5);
        if (data.sinkEntropyRate) {
            requireClose("CPU/CUDA entropy rate", entropyRate[i], data.sinkEntropyRate[i], 5.0e-5);
        }
        require(flags[i].bits == data.sinkFlags[i].bits, "CPU/CUDA packed-flag byte parity");
        if (data.sinkRungCandidates) {
            require(
                rungCandidates[i] == data.sinkRungCandidates[i], "CPU/CUDA candidate-rung parity");
        }
        if (comparesKicks) {
            // Compensated sums deposited in another order agree to float rounding.
            double const kickTimestep = fixture.params_.hydroKickTimestep;
            auto const& cpu = pairUpdates[i];
            auto const& gpu = data.sourcePairUpdates[i];
            for (int dim = 0; dim < kSpatialDimensions; dim++) {
                requireClose("CPU/CUDA kick acceleration",
                    compensatedKickSum(cpu.velocity[dim]) / kickTimestep,
                    compensatedKickSum(gpu.velocity[dim]) / kickTimestep, 5.0e-5);
            }
            requireClose("CPU/CUDA kick entropy rate",
                compensatedKickSum(cpu.entropy) / kickTimestep,
                compensatedKickSum(gpu.entropy) / kickTimestep, 5.0e-5);
        }
        require(rungs[i] == data.sinkParticleRungs[i], "CPU/CUDA particle-rung parity");
        require(data.sourceDensity[i] == sourceDensity[i] &&
                data.sourceSmoothingLength[i] == sourceSmoothingLengths[i] &&
                data.sourceRungs[i] == sourceRungs[i],
            "CPU/CUDA phases must preserve source snapshots");
    }
}

void testCpuCudaParity(bool morrisAndMonaghan = false, float conductionCoefficient = 0.0f) {
    // Three unequal leaves and 19 neighbors exercise indirection and SIMD tails.
    Fixture fixture(19, 3, true);
    fixture.params_.conductionCoefficient = conductionCoefficient;
    if (morrisAndMonaghan) {
        fixture.enableVariableViscosity();
        fixture.params_.vonNeumannRichtmyerCoefficient = 3.5f;
    }

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
    fixture.params_.minimumSmoothingLength = 0.03f;
    fixture.params_.maximumSmoothingLength = 0.24f;
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
    data.centralSmoothingLength = 0.2f;
    for (int source = 0; source < fixture.count_; source++) {
        int const sink = sinkIndices[source];
        int const particle = (2 * source) % fixture.count_;
        data.sinkParticleIndices[sink] = particle;
        double const positionX = 0.96 + 0.009 * source;
        data.sourcePositionX[source] = positionX - std::floor(positionX);
        data.sourcePositionY[source] = 0.48 + 0.005 * (source % 3);
        data.sourcePositionZ[source] = 0.49 + 0.004 * (source % 5);
        data.sourceSmoothingLength[source] = data.sinkSmoothingLength[sink] =
            0.08f + 0.005f * (source % 4);
        data.sourceInternalEnergy[source] = 0.7f + 0.03f * source;
        data.sourceVelocityX[source] = 0.03f * (source % 3 - 1);
        data.sourceVelocityY[source] = -0.02f * (source % 5 - 2);
        data.sourceVelocityZ[source] = 0.01f * (source % 2 ? 1 : -1);
        data.sourceRungs[source] = data.sinkParticleRungs[particle] = source % 3 ? 1 : 3;
        if (morrisAndMonaghan) {
            data.sourceBulkViscosityCoefficient[source] = 0.15f + 0.05f * source;
            data.sinkBulkViscosityCoefficient[sink] = data.sourceBulkViscosityCoefficient[source];
        }
        data.sinkGravityAccelerationX[sink] = 0.01f * (source % 3 - 1);
        data.sinkGravityAccelerationY[sink] = -0.005f;
        data.sinkGravityAccelerationZ[sink] = 0.002f;
    }
    auto publishSupport = [&] {
        for (int source = 0; source < fixture.count_; source++) {
            data.sourceSmoothingLength[source] = data.sinkSmoothingLength[sinkIndices[source]];
        }
    };
    auto publishRungs = [&] {
        for (int source = 0; source < fixture.count_; source++) {
            data.sourceRungs[source] =
                data.sinkParticleRungs[data.sinkParticleIndices[sinkIndices[source]]];
        }
    };
    compareCpuCudaPhase(fixture, kSphRunSmoothingLengths);
    publishSupport();
    if (morrisAndMonaghan) {
        fixture.params_.viscosityTimestep = 0.08;
        for (int i = 0; i < fixture.count_; ++i) {
            data.sinkFlags[i].setConverged(false);
        }
    }
    compareCpuCudaPhase(fixture, kSphRunDensityAndGradients);
    fixture.params_.viscosityTimestep = 0.0;
    publishSupport();
    for (int source = 0; source < fixture.count_; source++) {
        data.sourceDensity[source] = data.sinkDensity[sinkIndices[source]];
        data.sourceBalsaraFactor[source] =
            balsara::decodeFactor(data.sinkBalsaraFactor[sinkIndices[source]]);
        data.sourceGradHCorrection[source] = data.sinkGradHCorrection[sinkIndices[source]];
    }
    compareCpuCudaPhase(fixture, kSphRunHydro, 0);
    compareCpuCudaPhase(fixture, kSphRunHydro, 1);
    // The same passes as a midpoint operator compare the pair and external kicks.
    fixture.enableKickObservation();
    compareCpuCudaPhase(fixture, kSphRunHydro, 0);
    compareCpuCudaPhase(fixture, kSphRunHydro, 1);
    fixture.disableKickObservation();
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
    data.sourceSmoothingLength[excluded] = data.sinkSmoothingLength[excludedSink] = 0.025f;
    data.sourceDensity[excluded] = data.sourceInternalEnergy[excluded] = 0.0f;
    data.sinkDensity[excludedSink] = -9.0f;
    // An unbuilt correction on the excluded source must stay masked on both backends.
    data.sinkGradHCorrection[excludedSink] = -9.0f;
    data.sourceGradHCorrection[excluded] = std::numeric_limits<float>::quiet_NaN();
    compareCpuCudaPhase(fixture, kSphRunDensityAndGradients);
    require(!data.sinkFlags[excludedSink].semiactive() && data.sinkDensity[excludedSink] == -9.0f,
        "distant inactive particle must leave density untouched");
    require(data.sinkGradHCorrection[excludedSink] == -9.0f,
        "distant inactive particle must leave its correction untouched");
    for (int source = 0; source < excluded; source++) {
        data.sourceDensity[source] = data.sinkDensity[sinkIndices[source]];
        data.sourceBalsaraFactor[source] =
            balsara::decodeFactor(data.sinkBalsaraFactor[sinkIndices[source]]);
        data.sourceGradHCorrection[source] = data.sinkGradHCorrection[sinkIndices[source]];
    }
    compareCpuCudaPhase(fixture, kSphRunHydro, 0);
    compareCpuCudaPhase(fixture, kSphRunHydro, 1);
    fixture.enableKickObservation();
    compareCpuCudaPhase(fixture, kSphRunHydro, 0);
    compareCpuCudaPhase(fixture, kSphRunHydro, 1);
    fixture.disableKickObservation();

    // The same periodic particles now exceed an intentionally unexpanded box.
    for (int tree = 0; tree < 3; tree++) {
        for (int dim = 0; dim < kDimensions; dim++) {
            data.treeNodes[tree].outerBox.begin[dim] = 0.0;
            data.treeNodes[tree].outerBox.end[dim] = 1.0;
        }
    }
    for (int sink = 0; sink < fixture.count_; sink++) {
        data.sinkFlags[sink].setConverged(false);
        data.sinkSmoothingLength[sink] = 0.1f;
    }
    compareCpuCudaPhase(fixture, kSphRunSmoothingLengths);
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
        data.sourceSmoothingLength[source] = data.sinkSmoothingLength[sink] = 0.25f;
        data.sourceDensity[source] = 1.0f;
        data.sourceInternalEnergy[source] = 0.9f;
        data.sinkFlags[sink].setConverged(true);
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
                    requireClose("M&M backward Euler source and conformal decay",
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
    Fixture fixture(2, 2, true);
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

// The stiffness psi of a stored support in the fixture's bounds, mirroring
// sphGradHEvaluation: zero for a support on or outside distinct bounds or when fixed.
double referenceStiffness(Fixture const& fixture, double support) {
    double const minimum = fixture.params_.minimumSmoothingLength;
    double const maximum = fixture.params_.maximumSmoothingLength;
    if (!(minimum < maximum && support > minimum && support < maximum)) {
        return 0.0;
    }
    return sph_volume::boundedTarget<double>(
        support, minimum, maximum, sph_volume::supportLayers(minimum, maximum))
        .stiffness;
}

// Every solved support lies strictly inside the bounds and in the expected bound layer of
// sph_volume::supportLayers; the CPU parity tests cover the residual itself.
void requireLayerSupports(Fixture const& fixture, bool lowerLayer) {
    auto const layers = sph_volume::supportLayers(
        fixture.params_.minimumSmoothingLength, fixture.params_.maximumSmoothingLength);
    for (int i = 0; i < fixture.count_; i++) {
        float const support = fixture.data_.sourceSmoothingLength[i];
        require(support > fixture.params_.minimumSmoothingLength &&
                support < fixture.params_.maximumSmoothingLength,
            "bounded support stays strictly inside the bounds");
        require(lowerLayer ? support < layers.lowerJunction : support > layers.upperJunction,
            "bounded support lies in the expected layer");
    }
}

// A 6³ lattice: the free support of 32 neighbors, 0.33, lies below the upper layer of the
// half-box bound 0.49 (which starts at 0.392) and above bounds of 0.24 and below.
void testLattice() {
    constexpr int kSide = 6;
    Fixture fixture(kSide * kSide * kSide);
    initializeLattice(fixture, kSide);
    fixture.solveSmoothing(0.1f, 0.24f, 0.2f);
    requireLayerSupports(fixture, false);
    fixture.solveSmoothing(0.43f, 0.49f, 0.48f);
    requireLayerSupports(fixture, true);
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
// table as an independent CPU path. Core tests also check the table mathematics.
void testSelectedKernel() {
    Fixture fixture(2);
    fixture.params_.maximumSmoothingLength = 0.49f;
    auto& data = fixture.data_;
    data.sourcePositionX[0] = 0.40;
    data.sourcePositionX[1] = 0.58;
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
            "selected kernel CPU/CUDA density", data.sourceDensity[i], expectedDensity, 3.0e-6);
        double const normalizedRadius = radius / support;
        double const derivative = dkernelWDq(normalizedRadius);
        // The second support, 0.42, lies in the upper layer of the 0.49 bound: Omega blends
        // the free value -q w'/(3 sum w) with one by the stiffness psi.
        auto const sums = sph_volume::equalWeightSupportSums<double>(
            kernelW(0.0) + kernelW(normalizedRadius), normalizedRadius * derivative, data.gasMass);
        double const omega =
            sph_volume::gradHCorrection(sums, referenceStiffness(fixture, support)).supportJacobian;
        double const pressureFactor =
            (data.adiabaticIndex - 1.0f) * data.sourceInternalEnergy[i] / data.sourceDensity[i];
        pressureForce += data.gasMass * pressureFactor / omega * derivative / std::pow(support, 4);
        double const soundSpeed = std::sqrt(
            data.adiabaticIndex * (data.adiabaticIndex - 1.0f) * data.sourceInternalEnergy[i]);
        inverseCfl = std::max(inverseCfl,
            soundSpeed * selectedHydroKernel().smoothingLengthPerEffectiveRadius / support);
    }
    fixture.enableKickObservation();
    fixture.hydro(1);
    requireClose("selected kernel CPU/CUDA pressure force", fixture.kickAcceleration(0)[kXi],
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
        double const zeta = sph_volume::gradHCorrection(sums, referenceStiffness(fixture, support))
                                .weightedSupportCorrection;
        require(std::abs(zeta) > 0.1 * data.gasMass,
            "unequal pair exercises a nonzero grad-H correction");
        requireClose("unequal-pair grad-H correction", data.sourceGradHCorrection[i], zeta, 8.0e-6);
        directedGradient[i] = pressureFactor * (data.gasMass - zeta) * radialGradient;
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
    Fixture fixture(1);
    int const source = 0;
    int const sink = 0;
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

}    // namespace detail

class SphCudaTest : public detail::CudaTest {
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
            cudaMemoryInitialize(256 * 1024 * 1024);
            detail::checkCuda(cudaGetLastError(), "production workspace allocator initialization");
            allocatorInitialized = true;
        }
        setKernelSelections(
            {radial_kernel::Family::Ferrers, 8}, {radial_kernel::Family::Spline, 3}, true);
    }
};

TEST_F(SphCudaTest, VariableViscosityCpuParity) {
    detail::testCpuCudaParity(true);
}

TEST_F(SphCudaTest, MorrisMonaghanDensity) {
    detail::testMorrisMonaghanDensity();
}

TEST_F(SphCudaTest, VariableViscosityPairs) {
    detail::testVariableViscosityPairs();
}

TEST_F(SphCudaTest, BalsaraVelocityGradients) {
    detail::Fixture fixture(27, 3);
    fixture.enableVariableViscosity();
    auto& data = fixture.data_;
    struct Flow {
        float compression;
        float rotation;
        float shear;
        float expansion;
        float expected;
    };
    Flow const flows[] = {{0, 0, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 1, 0, 0, 0}, {1, 0, 0, 0, 1},
        {1, 1, 0, 0, 0.6f}, {0, 0, 0, -1, 0}, {1, 1, 0, 0.5f, 0.6f}};
    for (double const viscosityTimestep : {0.0, 0.08}) {
        fixture.params_.viscosityTimestep = viscosityTimestep;
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
                data.sourceSmoothingLength[i] = data.sinkSmoothingLength[i] = 0.25f;
                data.sinkFlags[i].setConverged(true);
            }
            fixture.density();
            EXPECT_EQ(data.sinkBalsaraFactor[13], balsara::encodeFactor(flow.expected));
            EXPECT_EQ(data.sourceBalsaraFactor[13],
                balsara::decodeFactor(balsara::encodeFactor(flow.expected)));
            for (int i = 0; i < fixture.count_; ++i) {
                EXPECT_GE(data.sourceBalsaraFactor[i], 0.0f);
                EXPECT_LE(data.sourceBalsaraFactor[i], 1.0f);
            }
        }
    }
}

TEST_F(SphCudaTest, BalsaraSymmetricForceAndHeating) {
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
        detail::requireClose(
            "Balsara pair heating", fixture.dissipativeHeating(i), 0.5 * fullHeating[i], 5.0e-6);
        auto const acceleration = fixture.kickAcceleration(i);
        for (int dim = 0; dim < kSpatialDimensions; ++dim) {
            detail::requireClose("Balsara pair force", acceleration[dim],
                pressureForce[i][dim] + 0.5f * (fullForce[i][dim] - pressureForce[i][dim]), 5.0e-6);
        }
    }
}

TEST_F(SphCudaTest, BlockReductionFlops) {
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

TEST_F(SphCudaTest, PeriodicSupport) {
    detail::testPeriodicSupport();
}

TEST_F(SphCudaTest, InactiveRungCandidateClosure) {
    detail::Fixture fixture(7, 7, true);
    detail::testInactiveRungCandidateClosure(fixture, [&] {
        detail::compareCpuCudaPhase(fixture, kSphRunRungs);
        return fixture.reduction_->pendingChanges != 0;
    });
}

TEST_F(SphCudaTest, SupportBoundaryRungCandidates) {
    detail::Fixture fixture(2, 2, true);
    detail::testSupportBoundaryRungCandidates(fixture, [&] {
        detail::compareCpuCudaPhase(fixture, kSphRunRungs);
        return fixture.reduction_->pendingChanges != 0;
    });
}

TEST_F(SphCudaTest, Lattice) {
    detail::testLattice();
}

TEST_F(SphCudaTest, UnequalSupports) {
    detail::testUnequalSupports();
}

TEST_F(SphCudaTest, ZPlaneAcceleration) {
    detail::testZPlaneAcceleration();
}

TEST_F(SphCudaTest, TimestepFailureReturnsNormallyAndThrowsOnHost) {
    detail::Fixture fixture(125);
    fixture.params_.minimumSmoothingLength = fixture.params_.maximumSmoothingLength;
    for (int i = 0; i < fixture.count_; ++i) {
        fixture.data_.sourcePositionX[i] = (i % 5 + 0.5) / 5.0;
        fixture.data_.sourcePositionY[i] = ((i / 5) % 5 + 0.5) / 5.0;
        fixture.data_.sourcePositionZ[i] = (i / 25 + 0.5) / 5.0;
        fixture.data_.sinkFlags[i].setConverged(true);
    }
    fixture.density();
    auto const originalTimestep = fixture.params_.baseTimestep;
    fixture.params_.baseTimestep = 1.0e20;
    EXPECT_THROW(fixture.hydro(1), std::runtime_error);
    EXPECT_TRUE(fixture.reduction_->timestepError.pending());
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    for (int i = 0; i < fixture.count_; ++i) {
        EXPECT_LT(fixture.data_.sourceRungs[i], kMaxRung);
    }
    // A caught host exception must leave the CUDA context usable.
    fixture.params_.baseTimestep = originalTimestep;
    EXPECT_NO_THROW(fixture.hydro(1));
    EXPECT_FALSE(fixture.reduction_->timestepError.pending());
}

TEST_F(SphCudaTest, DefaultTimestepCalibration) {
    detail::testDefaultTimestepCalibration();
}

TEST_F(SphCudaTest, DefaultThermalTimestep) {
    detail::testDefaultThermalTimestep();
}

TEST_F(SphCudaTest, UnequalPairEos) {
    detail::testUnequalPairEos();
}

TEST_F(SphCudaTest, SelectedKernelsGravityTimestepsAndCpuParity) {
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

TEST_F(SphCudaTest, GlobalRawTimestep) {
    detail::testGlobalRawTimestep<detail::Fixture>();
}

TEST_F(SphCudaTest, DeferredSourceCompletion) {
    detail::testDeferredSourceCompletion<detail::Fixture>();
}

TEST_F(SphCudaTest, MidpointHydroUsesExactRungWithoutGravityStorage) {
    detail::Fixture fixture(4, 2, true);
    detail::testMidpointHydroTargets(fixture);
    detail::compareCpuCudaPhase(fixture, kSphRunHydro, 1);
}

TEST_F(SphCudaTest, EndpointMorrisMonaghanUsesCompletedIntervalsAndClosingRungs) {
    detail::Fixture fixture(3, 1, true);
    detail::testEndpointMorrisInterval(fixture);
    fixture.params_.viscosityTimestep = 0.08;
    for (int index = 0; index < fixture.count_; ++index) {
        fixture.data_.sinkFlags[fixture.sinkIndex(index)].setConverged(false);
    }
    detail::compareCpuCudaPhase(fixture, kSphRunDensityAndGradients);
}

TEST_F(SphCudaTest, ConservativeMixedRungPairKicks) {
    detail::Fixture fixture(2, 2, true);
    detail::testConservativePairKick(fixture, false);
}

TEST_F(SphCudaTest, CompensatedFloatAtomicContention) {
    SphCompensatedSum* sum = nullptr;
    FlopCount* counts = nullptr;
    ASSERT_EQ(cudaMallocManaged(&sum, sizeof(*sum)), cudaSuccess);
    ASSERT_EQ(cudaMallocManaged(&counts, 512 * sizeof(*counts)), cudaSuccess);
    new (sum) SphCompensatedSum();
    sum->add(1.0f);
    detail::accumulateCompensatedFloat<<<4, 128>>>(sum, counts);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(float(*sum), 1.0f + 0x1p-8f);
    for (int i = 0; i < 512; ++i) {
        EXPECT_GE(counts[i], 4 * 256);
        EXPECT_EQ(counts[i] % 4, 0);
    }
    EXPECT_EQ(cudaFree(counts), cudaSuccess);
    EXPECT_EQ(cudaFree(sum), cudaSuccess);
}

TEST_F(SphCudaTest, ConservativeMorrisMonaghanPairKicks) {
    detail::Fixture fixture(2, 2, true);
    detail::testConservativePairKick(fixture, true);
}

TEST_F(SphCudaTest, LargeNeighborListsCpuParity) {
    // More neighbors than any shared buffer, an uneven tail, then a smaller list
    // on the same block. Only the first few particles in each leaf are active.
    detail::Fixture fixture(1537, 3, true);
    auto& data = fixture.data_;
    fixture.params_.minimumRung = 2;
    for (int i = 0; i < fixture.count_; ++i) {
        data.sourcePositionX[i] = (i % 13 + 0.5) / 13.0;
        data.sourcePositionY[i] = (i / 13 % 13 + 0.5) / 13.0;
        data.sourcePositionZ[i] = (i / 169 + 0.5) / 13.0;
        data.sourceDensity[i] = data.sinkDensity[i] = 1.0f;
        data.sourceVelocityX[i] = 0.01f * (i % 7 - 3);
        data.sourceVelocityY[i] = 0.02f * (i % 5 - 2);
        data.sourceVelocityZ[i] = 0.03f * (i % 3 - 1);
        data.sourceRungs[i] = data.sinkParticleRungs[i] = i % 512 < 3 ? 3 : 1;
    }
    data.treeNodes[2].neighborRange = {2, 3};
    detail::compareCpuCudaPhase(fixture, kSphRunSmoothingLengths);
    for (int i = 0; i < fixture.count_; ++i) {
        data.sourceSmoothingLength[i] = data.sinkSmoothingLength[i];
    }
    detail::compareCpuCudaPhase(fixture, kSphRunDensityAndGradients);
    for (int i = 0; i < fixture.count_; ++i) {
        data.sourceDensity[i] = data.sinkDensity[i];
        data.sourceGradHCorrection[i] = data.sinkGradHCorrection[i];
    }
    detail::compareCpuCudaPhase(fixture, kSphRunHydro);
    detail::compareCpuCudaPhase(fixture, kSphRunRungs);
}

namespace detail {

__global__ void resetSphBenchmark(
    SphRunCudaData data, SphReduction* reduction, int count, bool solveSupport) {
    if (threadIdx.x == 0) {
        *reduction = {};
        reduction->minimumSmoothingLength = std::numeric_limits<float>::max();
        reduction->minimumTimestep = std::numeric_limits<float>::max();
        reduction->minimumGradHOmega = std::numeric_limits<float>::max();
    }
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        data.sinkFlags[i].setConverged(!solveSupport);
        data.sinkSmoothingLength[i] = 0.2f;
    }
}

}    // namespace detail

// Explicit opt-in microbenchmark: no timing assertions in the regression suite.
TEST_F(SphCudaTest, DISABLED_NeighborStagingBenchmark) {
    constexpr int kParticleCount = 8192;
    constexpr int kTreeCount = 256;
    detail::Fixture fixture(kParticleCount, kTreeCount, true);
    auto& data = fixture.data_;
    for (int i = 0; i < kParticleCount; ++i) {
        data.sourcePositionX[i] = (i % 32 + 0.5) / 32.0;
        data.sourcePositionY[i] = (i / 32 % 16 + 0.5) / 16.0;
        data.sourcePositionZ[i] = (i / 512 + 0.5) / 16.0;
        data.sourceDensity[i] = data.sinkDensity[i] = 1.0f;
        data.sourceVelocityX[i] = 0.01f * (i % 7 - 3);
        data.sourceVelocityY[i] = 0.02f * (i % 5 - 2);
        data.sourceVelocityZ[i] = 0.03f * (i % 3 - 1);
    }
    cudaDeviceProp device{};
    detail::checkCuda(cudaGetDeviceProperties(&device, 0), "benchmark device properties");
    void const* const kernels[] = {(void const*) sphCudaSmoothingLengths,
        (void const*) sphCudaDensityAndGradients, (void const*) sphCudaHydro,
        (void const*) sphCudaRungs};
    int const blockSizes[] = {kSmoothingLengthsBlockSize, kDensityAndGradientsBlockSize,
        kHydroBlockSize, kRungsBlockSize};
    char const* const names[] = {"smoothing", "density", "hydro", "rungs"};
    cudaEvent_t start;
    cudaEvent_t stop;
    detail::checkCuda(cudaEventCreate(&start), "benchmark start event");
    detail::checkCuda(cudaEventCreate(&stop), "benchmark stop event");
    for (int neighbors : {128, 512, 2048}) {
        int const leaves = neighbors / (kParticleCount / kTreeCount);
        for (int tree = 0; tree < kTreeCount; ++tree) {
            int const begin = tree / leaves * leaves;
            data.treeNodes[tree].neighborRange = {begin, begin + leaves};
        }
        for (int pass = 0; pass < 4; ++pass) {
            cudaFuncAttributes attributes{};
            int residentBlocks;
            detail::checkCuda(
                cudaFuncGetAttributes(&attributes, kernels[pass]), "kernel attributes");
            detail::checkCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                                  &residentBlocks, kernels[pass], blockSizes[pass], 0),
                "kernel occupancy");
            int const blocks = std::min(kTreeCount, residentBlocks * device.multiProcessorCount);
            std::array<float, 9> samples{};
            for (int iteration = -2; iteration < int(samples.size()); ++iteration) {
                detail::resetSphBenchmark<<<1, 256>>>(
                    data, fixture.reduction_, kParticleCount, pass == 0);
                detail::checkCuda(cudaEventRecord(start), "benchmark start");
                void* arguments[] = {&fixture.params_, &data, &fixture.reduction_};
                detail::checkCuda(cudaLaunchKernel(kernels[pass], dim3(blocks),
                                      dim3(blockSizes[pass]), arguments),
                    "benchmark launch");
                detail::checkCuda(cudaEventRecord(stop), "benchmark stop");
                detail::checkCuda(cudaEventSynchronize(stop), "benchmark completion");
                float milliseconds;
                detail::checkCuda(
                    cudaEventElapsedTime(&milliseconds, start, stop), "benchmark time");
                if (iteration >= 0) {
                    samples[iteration] = milliseconds;
                }
            }
            std::sort(samples.begin(), samples.end());
            std::printf("SPH_BENCH %s neighbors=%d shared=%zu registers=%d blocks_per_sm=%d "
                        "median_ms=%.6f\n",
                names[pass], neighbors, attributes.sharedSizeBytes, attributes.numRegs,
                residentBlocks, samples[samples.size() / 2]);
        }
    }
    detail::checkCuda(cudaEventDestroy(start), "destroy benchmark start");
    detail::checkCuda(cudaEventDestroy(stop), "destroy benchmark stop");
}

namespace detail {

__global__ void checkSharedNeighborBuffer(int* errors) {
    constexpr int kCapacity = 67;
    __shared__ SharedNeighborBuffer<int, kCapacity> neighbors;
    new (&neighbors) SharedNeighborBuffer<int, kCapacity>();
    int const sizes[] = {0, 66, 67, 68, 197, 11, 301, 0, 67};
    int previousSize = 0;
    bool usesShared = true;
    for (int stage = 0; stage < int(sizeof(sizes) / sizeof(sizes[0])); ++stage) {
        int const count = sizes[stage];
        neighbors.resize(count);
        usesShared = count == 0 || (usesShared && count <= kCapacity);
        if (neighbors.size() != count) {
            atomicAdd(errors, 1);
        }
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            if (i < previousSize && neighbors[i] != 1000 * (stage - 1) + i) {
                atomicAdd(errors, 1);
            }
            if ((usesShared && !__isShared(&neighbors[i])) ||
                (!usesShared && !__isGlobal(&neighbors[i]))) {
                atomicAdd(errors, 1);
            }
            neighbors[i] = 1000 * stage + i;
        }
        __syncthreads();
        // Read values written by other lanes after shared-to-global relocation.
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            int const reverse = count - 1 - i;
            if (neighbors[reverse] != 1000 * stage + reverse) {
                atomicAdd(errors, 1);
            }
        }
        previousSize = count;
    }
    neighbors.~SharedNeighborBuffer();
}

}    // namespace detail

TEST_F(SphCudaTest, SharedNeighborStorageAndResize) {
    int* errors;
    detail::checkCuda(cudaMallocManaged(&errors, sizeof(int)), "shared neighbor test allocation");
    *errors = 0;
    detail::checkSharedNeighborBuffer<<<4, 96>>>(errors);
    detail::checkCuda(cudaGetLastError(), "shared neighbor test launch");
    detail::checkCuda(cudaDeviceSynchronize(), "shared neighbor test completion");
    EXPECT_EQ(*errors, 0);
    detail::checkCuda(cudaFree(errors), "shared neighbor test release");
}

TEST_F(SphCudaTest, ArtificialConduction) {
    detail::Fixture fixture(2, 2, true);
    detail::testConductionPair(fixture);
}

TEST_F(SphCudaTest, ArtificialConductionCpuParity) {
    detail::testCpuCudaParity(false, 0.3f);
    detail::testCpuCudaParity(true, 0.3f);
}

TEST_F(SphCudaTest, DissipativeRateSuperposition) {
    for (bool const morrisAndMonaghan : {false, true}) {
        detail::Fixture fixture(2, 2, true);
        detail::testDissipativeRateSuperposition(fixture, morrisAndMonaghan);
    }
}
