/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#ifndef SPH_HPP_
#define SPH_HPP_

#include <atomic>

#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/stack_vector.hpp>
#include <cosmictiger/core/tree_id.hpp>
#include <cosmictiger/math/range.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/particle_types.hpp>

#ifndef __CUDACC__
// Reversed unit-box bounds seed min/max accumulation for particle positions
// in [0, 1) and their expanded supports. These are signed RangeFixed values,
// not unsigned Fixed32 positions; periodic image shifts remain representable.
inline Fixed32Range emptySphBounds() {
    Fixed32Range box;
    for (int dim = 0; dim < kDimensions; ++dim) {
        box.begin[dim] = 1.0;
        box.end[dim] = 0.0;
    }
    return box;
}

struct SphTreeNeighborReturn {
    Fixed32Range innerBox = emptySphBounds();
    Fixed32Range outerBox = emptySphBounds();
    CUDA_EXPORT
    SphTreeNeighborReturn& operator+=(SphTreeNeighborReturn const& other) {
        for (int dim = 0; dim < kDimensions; dim++) {
            innerBox.begin[dim] = std::min(innerBox.begin[dim], other.innerBox.begin[dim]);
            innerBox.end[dim] = std::max(innerBox.end[dim], other.innerBox.end[dim]);
            outerBox.begin[dim] = std::min(outerBox.begin[dim], other.outerBox.begin[dim]);
            outerBox.end[dim] = std::max(outerBox.end[dim], other.outerBox.end[dim]);
        }
        return *this;
    }
    template <typename A>
    void serialize(A&& arc, unsigned) {
        arc & innerBox;
        arc & outerBox;
    }
};
#endif

enum class SphNeighborOperation { RefreshBounds, FindNeighbors };

// Which particle's kernel support can admit a pair to the neighbor list.
enum class SphInteractionDirection { TargetSupport, SourceSupport, EitherSupport };

// Active means rung >= minimumRung; this selection applies only to bounds refresh.
enum class SphParticleSelection { Active, All };

struct SphTreeNeighborParameters {
    SphNeighborOperation operation = SphNeighborOperation::RefreshBounds;
    float smoothingLengthMultiplier = 1.0f;
    int minimumRung = 0;
    SphInteractionDirection interactionDirection = SphInteractionDirection::EitherSupport;
    SphParticleSelection innerParticleSelection = SphParticleSelection::All;
    SphParticleSelection outerParticleSelection = SphParticleSelection::Active;
    template <typename T>
    void serialize(T&& arc, unsigned) {
        arc & interactionDirection;
        arc & innerParticleSelection;
        arc & outerParticleSelection;
        arc & operation;
        arc & smoothingLengthMultiplier;
        arc & minimumRung;
    }
};

struct SphRunReturn {
    double minimumTimestep = std::numeric_limits<double>::max();
    float minimumSmoothingLength = std::numeric_limits<float>::max();
    float maximumSmoothingLength = 0.0f;
    double floatingPointOperations = 0.0;
    int maximumHydroRung = 0;
    int maximumGravityRung = 0;
    int maximumRung = 0;
    float maximumSignalSpeed = 0.0f;
    bool needsAnotherPass = false;
    float inverseSoundTimestep = 0.0f;
    float inverseViscosityTimestep = 0.0f;
    float inverseAccelerationTimestep = 0.0f;
    // Variational density-pass diagnostics, accumulated over every evaluated support.
    // A leaf repeated by a later pass contributes again to the counts.
    float minimumGradHOmega = std::numeric_limits<float>::max();    // Over free supports.
    float maximumSupportResidual = 0.0f;    // Largest |sum w - target|/target of a free support.
    int heldSupports = 0;                   // Supports at an active bound, with zeta = 0.
    int illConditionedSupports = 0;         // Free supports with a small Omega.
    template <typename Archive>
    void serialize(Archive&& arc, unsigned) {
        arc & minimumTimestep;
        arc & needsAnotherPass;
        arc & minimumGradHOmega;
        arc & maximumSupportResidual;
        arc & heldSupports;
        arc & illConditionedSupports;
        arc & floatingPointOperations;
        arc & inverseSoundTimestep;
        arc & inverseViscosityTimestep;
        arc & inverseAccelerationTimestep;
        arc & minimumSmoothingLength;
        arc & maximumSmoothingLength;
        arc & maximumHydroRung;
        arc & maximumGravityRung;
        arc & maximumRung;
        arc & maximumSignalSpeed;
    }
    SphRunReturn& operator+=(SphRunReturn const& other) {
        minimumTimestep = std::min(minimumTimestep, other.minimumTimestep);
        floatingPointOperations += other.floatingPointOperations;
        maximumSmoothingLength = std::max(maximumSmoothingLength, other.maximumSmoothingLength);
        minimumSmoothingLength = std::min(minimumSmoothingLength, other.minimumSmoothingLength);
        maximumHydroRung = std::max(maximumHydroRung, other.maximumHydroRung);
        maximumGravityRung = std::max(maximumGravityRung, other.maximumGravityRung);
        maximumRung = std::max(maximumRung, other.maximumRung);
        maximumSignalSpeed = std::max(maximumSignalSpeed, other.maximumSignalSpeed);
        inverseSoundTimestep = std::max(inverseSoundTimestep, other.inverseSoundTimestep);
        inverseViscosityTimestep =
            std::max(inverseViscosityTimestep, other.inverseViscosityTimestep);
        inverseAccelerationTimestep =
            std::max(inverseAccelerationTimestep, other.inverseAccelerationTimestep);
        needsAnotherPass = needsAnotherPass || other.needsAnotherPass;
        minimumGradHOmega = std::min(minimumGradHOmega, other.minimumGradHOmega);
        maximumSupportResidual = std::max(maximumSupportResidual, other.maximumSupportResidual);
        heldSupports += other.heldSupports;
        illConditionedSupports += other.illConditionedSupports;
        return *this;
    }
};

struct SphRunParameters {
    int runType;
    int phase;
    int minimumRung;
    int maximumRung = kMaxRung - 1;
    int minimumProposedRung = 0;
    float baseTimestep;
    float scaleFactor;      // Cosmological scale factor; one in isolated test problems.
    float expansionRate;    // Expansion rate; zero in isolated test problems.
    float courantFactor;
    float minimumSmoothingLength;
    float maximumSmoothingLength;
    float uniformAccelerationY;    // Optional uniform force along y.
    float zPlaneAcceleration;      // Magnitude of the force toward z = 0.5.
    float maximumTimestep = 1e30f;
    bool globalTimestep = false;
    // A midpoint hydro operator uses the current split-stage sources and owns
    // only nongravitational rates. Gravity supplies its own scalar timestep bound.
    bool midpointHydro = false;
    // Positive hydro interval; zero only estimates forces/timestep bounds.
    double hydroKickTimestep = 0.0;
    bool deferPairCommit = false;
    bool retainProposedRungs = false;
    // Phase zero may publish pair rates before the fresh gravity field arrives.
    // Source acceleration is then assembled before the closing corrector.
    bool deferAccelerationSources = false;
    bool refreshConvergedDensity =
        false;    // Recompute diagnostics with fixed, converged supports.
    // Initialization only: publish the converged kernel-sum density alongside H.
    // This pass uses the CPU solver before any thermal state is initialized.
    bool initializeDensity = false;
    float bulkViscosityCoefficient;
    float vonNeumannRichtmyerCoefficient;
    float viscosityEpsilon;
    float conductionCoefficient = 0.0f;
    bool balsaraSwitch;
    bool variableViscosity;
    float minimumBulkViscosityCoefficient;
    float viscosityDecay;
    // Completed interval for Morris-Monaghan: dyadic base duration (divided by
    // the live rung in the density pass), or the actual global duration. Zero
    // disables evolution at midpoints, initialization, and diagnostic refreshes.
    double viscosityTimestep = 0.0;
    float velocityDampingRate;    // Velocity damping used while relaxing the gas sphere.
    SphRunParameters() {
        auto const opts = getOptions();
        minimumProposedRung = opts.run.minRung;
        minimumSmoothingLength = opts.hydro.minSmoothingLength;
        maximumSmoothingLength = opts.hydro.maxSmoothingLength;
        velocityDampingRate = opts.hydro.damping;
        uniformAccelerationY = opts.hydro.yAcceleration;
        zPlaneAcceleration = opts.hydro.zPlaneAcceleration;
        bulkViscosityCoefficient = opts.hydro.bulkViscosityCoefficient;
        vonNeumannRichtmyerCoefficient = opts.hydro.vonNeumannRichtmyerCoefficient;
        viscosityEpsilon = opts.hydro.viscosityEpsilon;
        conductionCoefficient = opts.hydro.conductionCoefficient;
        balsaraSwitch = opts.hydro.balsaraSwitch;
        variableViscosity = opts.hydro.viscosityEvolution != "none";
        minimumBulkViscosityCoefficient = opts.hydro.minimumBulkViscosityCoefficient;
        viscosityDecay = opts.hydro.viscosityDecay;
    }
    CUDA_EXPORT bool isActiveRung(int rung) const {
        return rung >= minimumRung && rung <= maximumRung;
    }
    // Morris-Monaghan pairs use beta_ab = viscosityBeta alpha_ab; fixed Monaghan coefficients
    // use the constant beta.
    CUDA_EXPORT bool vonNeumannRichtmyerCoefficientFollowsBulk() const {
        return variableViscosity;
    }
    template <typename Archive>
    void serialize(Archive&& arc, unsigned) {
        arc & bulkViscosityCoefficient;
        arc & vonNeumannRichtmyerCoefficient;
        arc & viscosityEpsilon;
        arc & balsaraSwitch;
        arc & variableViscosity;
        arc & minimumBulkViscosityCoefficient;
        arc & viscosityDecay;
        arc & viscosityTimestep;
        arc & velocityDampingRate;
        arc & minimumSmoothingLength;
        arc & maximumSmoothingLength;
        arc & expansionRate;
        arc & maximumTimestep;
        arc & globalTimestep;
        arc & midpointHydro;
        arc & hydroKickTimestep;
        arc & deferPairCommit;
        arc & retainProposedRungs;
        arc & deferAccelerationSources;
        arc & refreshConvergedDensity;
        arc & initializeDensity;
        arc & uniformAccelerationY;
        arc & baseTimestep;
        arc & phase;
        arc & runType;
        arc & minimumRung;
        arc & maximumRung;
        arc & minimumProposedRung;
        arc & scaleFactor;
        arc & courantFactor;
        arc & zPlaneAcceleration;
        arc & conductionCoefficient;
    }
    CUDA_EXPORT
    float midplaneAccelerationZ(Position const z) const {
        Position const middle(0.5);
        return z < middle ? zPlaneAcceleration : (z > middle ? -zPlaneAcceleration : 0.0f);
    }
};

inline constexpr int kSphRunSmoothingLengths = 0;
inline constexpr int kSphRunDensityAndGradients = 1;
inline constexpr int kSphRunHydro = 2;
inline constexpr int kSphRunRungs = 3;

SphRunReturn sphRun(SphRunParameters params, bool cuda = false);
// Collective explicit-midpoint RK2 at fixed positions. Both force stages read
// immutable all-locality snapshots; the final increment is relative to the
// saved incoming state, including every inactive recipient of a pair impulse.
// A zero interval performs only a nonmutating force/timestep estimate.
SphRunReturn sphMidpointForces(SphRunParameters params, bool cuda = true);
// Event-local, gas-indexed irreversible entropy rate.
// Allocated after gas sorting and released before migration/checkpoint/output.
float* sphStageEntropyRates();
#ifndef __CUDACC__
hpx::future<SphTreeNeighborReturn> sphTreeNeighbor(
    SphTreeNeighborParameters params, TreeId self, Vector<TreeId> checklist, int level = 0);
#endif

#endif /* SPH_HPP_ */
