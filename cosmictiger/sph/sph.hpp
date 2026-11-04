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

inline constexpr int kSphTreeNeighborBoxes = 0;
inline constexpr int kSphTreeNeighborNeighbors = 1;
inline constexpr int kSphInteractionsI = 1;
inline constexpr int kSphInteractionsJ = 2;
inline constexpr int kSphInteractionsIJ = 3;
inline constexpr int kSphSetActive = 1;
inline constexpr int kSphSetAll = 4;

struct SphTreeNeighborParameters {
    int runType = 0;
    float supportRadiusMultiplier = 1.0f;
    int minimumRung = 0;
    int innerSelection = 0;
    int outerSelection = 0;
    template <typename T>
    void serialize(T&& arc, unsigned) {
        arc & innerSelection;
        arc & outerSelection;
        arc & runType;
        arc & supportRadiusMultiplier;
        arc & minimumRung;
    }
};

struct SphRunReturn {
    double minimumTimestep = std::numeric_limits<double>::max();
    float minimumSupportRadius = std::numeric_limits<float>::max();
    float maximumSupportRadius = 0.0f;
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
        arc & minimumSupportRadius;
        arc & maximumSupportRadius;
        arc & maximumHydroRung;
        arc & maximumGravityRung;
        arc & maximumRung;
        arc & maximumSignalSpeed;
    }
    SphRunReturn& operator+=(SphRunReturn const& other) {
        minimumTimestep = std::min(minimumTimestep, other.minimumTimestep);
        floatingPointOperations += other.floatingPointOperations;
        maximumSupportRadius = std::max(maximumSupportRadius, other.maximumSupportRadius);
        minimumSupportRadius = std::min(minimumSupportRadius, other.minimumSupportRadius);
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
    float minimumSupportRadius;
    float maximumSupportRadius;
    float uniformAccelerationY;    // Optional uniform force along y.
    float zPlaneAcceleration;      // Magnitude of the force toward z = 0.5.
    float maximumTimestep = 1e30f;
    bool globalTimestep = false;
    // A midpoint hydro operator uses the current split-stage sources and owns
    // only nongravitational rates. Gravity supplies its own scalar timestep bound.
    bool midpointHydro = false;
    // Phase zero may publish pair rates before the fresh gravity field arrives.
    // Source acceleration is then assembled before the closing corrector.
    bool deferAccelerationSources = false;
    bool refreshConvergedDensity =
        false;    // Recompute diagnostics with fixed, converged supports.
    float bulkViscosityCoefficient;
    float vonNeumannRichtmyerCoefficient;
    float viscosityEpsilon;
    bool balsaraSwitch;
    bool variableViscosity;
    bool cullenDehnen;
    bool priceEtAl;
    bool huEtAl;
    float maximumBulkViscosityCoefficientHu;
    float viscosityDecayHu;
    float maximumBulkViscosityCoefficient;
    float viscosityDecayCD;
    float maximumBulkViscosityCoefficientPrice;
    float viscosityDecayPrice;
    float expansionAcceleration = 0.0f;    // a³ d²a/dt² for the physical CD acceleration gradient.
    float minimumBulkViscosityCoefficient;
    float viscosityDecay;
    // Completed interval for Morris-Monaghan: a base interval for legacy individual
    // rungs, or the actual global/midpoint interval. Zero disables refresh evolution.
    double viscosityTimestep = 0.0;
    float velocityDampingRate;    // Velocity damping used while relaxing the gas sphere.
    SphRunParameters() {
        auto const opts = getOptions();
        minimumProposedRung = opts.run.minRung;
        minimumSupportRadius = opts.hydro.minSupportRadius;
        maximumSupportRadius = opts.hydro.maxSupportRadius;
        velocityDampingRate = opts.hydro.damping;
        uniformAccelerationY = opts.hydro.yAcceleration;
        zPlaneAcceleration = opts.hydro.zPlaneAcceleration;
        bulkViscosityCoefficient = opts.hydro.bulkViscosityCoefficient;
        vonNeumannRichtmyerCoefficient = opts.hydro.vonNeumannRichtmyerCoefficient;
        viscosityEpsilon = opts.hydro.viscosityEpsilon;
        balsaraSwitch = opts.hydro.balsaraSwitch;
        variableViscosity = opts.hydro.viscosityEvolution != "none";
        cullenDehnen = opts.hydro.viscosityEvolution == "cullenAndDehnen";
        priceEtAl = opts.hydro.viscosityEvolution == "priceEtAl";
        huEtAl = opts.hydro.viscosityEvolution == "hu_etal";
        maximumBulkViscosityCoefficientHu = opts.hydro.maximumBulkViscosityCoefficientHu;
        viscosityDecayHu = opts.hydro.viscosityDecayHu;
        maximumBulkViscosityCoefficient = opts.hydro.maximumBulkViscosityCoefficient;
        viscosityDecayCD = opts.hydro.viscosityDecayCD;
        maximumBulkViscosityCoefficientPrice = opts.hydro.maximumBulkViscosityCoefficientPrice;
        viscosityDecayPrice = opts.hydro.viscosityDecayPrice;
        minimumBulkViscosityCoefficient = opts.hydro.minimumBulkViscosityCoefficient;
        viscosityDecay = opts.hydro.viscosityDecay;
    }
    // Cullen-Dehnen, Hu, and Price et al. all need the gradient sums, previous accelerations,
    // and the shared target-alpha state of the density pass.
    CUDA_EXPORT bool gradientSwitch() const {
        return cullenDehnen || priceEtAl || huEtAl;
    }
    CUDA_EXPORT bool isActiveRung(int rung) const {
        return rung >= minimumRung && rung <= maximumRung;
    }
    // Morris-Monaghan, Cullen-Dehnen, and Hu et al. pairs use beta_ab = viscosityBeta alpha_ab;
    // fixed Monaghan coefficients and Price et al. (2018) equation (41) use the constant beta.
    CUDA_EXPORT bool vonNeumannRichtmyerCoefficientFollowsBulk() const {
        return variableViscosity && !priceEtAl;
    }
    template <typename Archive>
    void serialize(Archive&& arc, unsigned) {
        arc & bulkViscosityCoefficient;
        arc & vonNeumannRichtmyerCoefficient;
        arc & viscosityEpsilon;
        arc & balsaraSwitch;
        arc & variableViscosity;
        arc & cullenDehnen;
        arc & priceEtAl;
        arc & huEtAl;
        arc & maximumBulkViscosityCoefficientHu;
        arc & viscosityDecayHu;
        arc & maximumBulkViscosityCoefficient;
        arc & viscosityDecayCD;
        arc & maximumBulkViscosityCoefficientPrice;
        arc & viscosityDecayPrice;
        arc & expansionAcceleration;
        arc & minimumBulkViscosityCoefficient;
        arc & viscosityDecay;
        arc & viscosityTimestep;
        arc & velocityDampingRate;
        arc & minimumSupportRadius;
        arc & maximumSupportRadius;
        arc & expansionRate;
        arc & maximumTimestep;
        arc & globalTimestep;
        arc & midpointHydro;
        arc & deferAccelerationSources;
        arc & refreshConvergedDensity;
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
    }
    CUDA_EXPORT
    float midplaneAccelerationZ(Position const z) const {
        Position const middle(0.5);
        return z < middle ? zPlaneAcceleration : (z > middle ? -zPlaneAcceleration : 0.0f);
    }
};

inline constexpr int kSphRunPrehydro1 = 0;
inline constexpr int kSphRunPrehydro2 = 1;
inline constexpr int kSphRunHydro = 2;
inline constexpr int kSphRunRungs = 3;

SphRunReturn sphRun(SphRunParameters params, bool cuda = false);
// Collective, after density/neighbor preparation at the hydro midpoint. Freeze
// all source states once, then publish H, irreversible entropy rates and scalar
// timestep proposals. The caller applies the full positive kick duration.
SphRunReturn sphMidpointForces(SphRunParameters params, bool cuda = true);
void sphParticlesCompleteForces(SphRunParameters const& parameters);
#ifndef __CUDACC__
hpx::future<SphTreeNeighborReturn> sphTreeNeighbor(
    SphTreeNeighborParameters params, TreeId self, Vector<TreeId> checklist, int level = 0);
#endif

#endif /* SPH_HPP_ */
