/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>

#include <cosmictiger/chemistry/gadget4_cooling.hpp>
#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/float16_storage.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/particle_flags.hpp>
#include <cosmictiger/sph/thermodynamics.hpp>
#include <cosmictiger/sph/viscosity/scaled_divergence.hpp>

#ifdef SPH_PARTICLES_CPP
#define SPH_PARTICLES_EXTERN
#else
#define SPH_PARTICLES_EXTERN extern
#endif
#ifdef CHECK_BOUNDS
#define CHECK_SPH_PART_BOUNDS(i)                                                                \
    if (i < 0 || i >= sphParticlesSize()) {                                                     \
        std::cout << std::format(                                                               \
                         "particle bound check failure {:d} should be between {:d} and {:d}\n", \
                         (long long) i, (long long) 0, (long long) sphParticlesSize())          \
                  << std::flush;                                                                \
        ALWAYS_ASSERT(false);                                                                   \
    }
#else
#define CHECK_SPH_PART_BOUNDS(i)
#endif

using SphAcceleration = ParticleVector<float>;

// Last evaluated hydrodynamic rates in native conformal time tau. Output refreshes
// all gas at snapshot coordinates; migration invalidates them. These derived
// fields are neither checkpoint state nor migration payload.
struct SphHydroDiagnostics {
    SphAcceleration acceleration{};
    SphAcceleration pressureAcceleration{};
    SphAcceleration viscosityAcceleration{};
    float entropyRate{};
    float viscosityEntropyRate{};
    float conductionEntropyRate{};
    float densityRate{};
    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & acceleration & pressureAcceleration & viscosityAcceleration & entropyRate &
            viscosityEntropyRate & conductionEntropyRate & densityRate;
    }
};
SPH_PARTICLES_EXTERN SphHydroDiagnostics* sphParticlesHydroDiagnostics;
// Snapshot-scoped derived storage; never allocated by the integrator.
void sphParticlesAllocateHydroDiagnostics();
void sphParticlesFreeHydroDiagnostics();

// Cooling stores an equilibrium electron-fraction guess; hydro owns entropy.
SPH_PARTICLES_EXTERN float* sphParticlesElectronFractions;

// Canonical A=P_physical/rho_physical^gamma. Before the first density solve these
// slots temporarily hold the initial E=a^2 u; prepareThermodynamics converts once.
extern bool sphParticlesEntropyInitialized;
extern double sphParticlesThermalScaleFactor;
void sphParticlesPrepareThermodynamics(double scaleFactor, bool initializeEntropy);
SPH_PARTICLES_EXTERN float* sphParticlesEntropies;
// Stored smoothing length is the full compact-support radius H of the SPH kernel.
// For the cubic spline H = 2h; renaming this storage does not rescale its values.
SPH_PARTICLES_EXTERN float* sphParticlesSmoothingLengths;
SPH_PARTICLES_EXTERN ParticleInteger* sphParticlesParticleIndices;
SPH_PARTICLES_EXTERN float* sphParticlesDensities;
// Gradient models retain their endpoint factor; fixed-alpha and M&M density passes
// rebuild it directly from weighted divergence and curl.
SPH_PARTICLES_EXTERN uint8_t* sphParticlesBalsaraFactors;
// Variational operator only: zeta_i of cosmictiger/sph/volume.hpp, written by the density
// pass for every semiactive particle before each force pass. Never migrated or checkpointed.
SPH_PARTICLES_EXTERN float* sphParticlesGradHCorrections;
SPH_PARTICLES_EXTERN Float16Storage* sphParticlesBulkViscosityCoefficients;
SPH_PARTICLES_EXTERN Float16Storage* sphScaledDivergence;
// Event-local density/viscosity diagnostics, allocated only for one evaluation.
SPH_PARTICLES_EXTERN SphViscosityDiagnostics* sphParticlesViscosityDiagnostics;

// Convergence and semiactive working flags share one byte with the latest hydro
// timestep requirement. The proposed rung is separate from the live particle rung,
// retained until the next estimate, and applied only at common boundaries.
// This storage persists across SPH passes, migration, and checkpoints.
SPH_PARTICLES_EXTERN SphParticleFlags* sphParticlesFlags;

inline int sphParticlesProposedRung(ParticleInteger gas) {
    return sphParticlesFlags[gas].proposedRung();
}

inline void sphParticlesSetProposedRung(ParticleInteger gas, int rung) {
    sphParticlesFlags[gas].setProposedRung(rung);
}

inline int sphParticlesHydroRung(ParticleInteger gas) {
    return sphParticlesProposedRung(gas);
}

ParticleInteger sphParticlesSize();
void sphParticlesResize(ParticleInteger particleCount, bool resizeGravityParticles = true);
void sphParticlesFree();
void sphParticlesRemove(ParticleInteger index);
void sphParticlesClearHydroCaches();
void sphParticlesClearCaches();
void sphParticlesResolveWithParticles();
void sphParticlesSwap(ParticleInteger i, ParticleInteger j);
ParticleInteger sphParticlesSort(
    Pair<ParticleInteger> particleRange, Position splitPosition, int splitDimension);
void sphParticlesGlobalReadPositions(
    ParticleGlobalRange range, Position* x, Position* y, Position* z, ParticleInteger offset);
void sphParticlesGlobalReadInternalEnergyAndSmoothingLength(ParticleGlobalRange range,
    float* internalEnergy, float* smoothingLength, ParticleInteger offset);
void sphParticlesGlobalReadRungs(ParticleGlobalRange range, char*, ParticleInteger offset);
void sphParticlesGlobalReadVelocities(
    ParticleGlobalRange range, float*, float*, float*, ParticleInteger offset);
void sphParticlesGlobalReadDensity(ParticleGlobalRange range, float*, ParticleInteger offset,
    float* balsaraFactor = nullptr, float* bulkViscosityCoefficient = nullptr,
    float* gradHCorrection = nullptr);
void sphParticlesResetConverged();
void sphParticlesResetSemiactive();
void sphParticlesLoad(FILE* fp);
void sphParticlesSave(FILE* fp);
void sphCheckpointTest();
struct SphRunParameters;
void sphParticlesBeginViscosityDiagnostics();
void sphParticlesEndViscosityDiagnostics();
// Scope ownership remains active on a locality with no gas and no allocation.
bool sphParticlesViscosityDiagnosticsActive();
// Commit history and viscosity only at the particle's completed endpoint.
void sphParticlesUpdateTargetAlpha(SphRunParameters const& parameters);

inline float sphParticlesMass(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return float(getOptions().initial.gasMass);
}

inline bool sphParticlesConverged(ParticleInteger index) {
    return sphParticlesFlags[index].converged();
}

inline void sphParticlesSetConverged(ParticleInteger index, bool value) {
    sphParticlesFlags[index].setConverged(value);
}

inline bool sphParticlesSemiactive(ParticleInteger index) {
    return sphParticlesFlags[index].semiactive();
}

inline void sphParticlesSetSemiactive(ParticleInteger index, bool value) {
    sphParticlesFlags[index].setSemiactive(value);
}

inline ParticleInteger& sphParticlesParticleIndex(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesParticleIndices[index];
}

inline Position& sphParticlesPosition(int dim, int index) {
    return particlesPos(dim, sphParticlesParticleIndex(index));
}

inline float& sphParticlesVelocity(int dim, int index) {
    return particlesVel(dim, sphParticlesParticleIndex(index));
}

inline Position sphParticlesPositionValue(int dim, int index) {
    return particlesPositionValue(dim, sphParticlesParticleIndex(index));
}
inline float sphParticlesVelocityValue(int dim, int index) {
    return particlesVelocityValue(dim, sphParticlesParticleIndex(index));
}

inline char& sphParticlesRung(int index) {
    return particlesRung(sphParticlesParticleIndex(index));
}

inline float& sphParticlesEntropy(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    ALWAYS_ASSERT(sphParticlesEntropyInitialized);
    return sphParticlesEntropies[index];
}

inline float sphParticlesInternalEnergy(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    if (!sphParticlesEntropyInitialized) {
        return sphParticlesEntropies[index];
    }
    return sph_entropy::storedEnergy(sphParticlesEntropies[index], sphParticlesDensities[index],
        getOptions().hydro.gamma, sphParticlesThermalScaleFactor);
}

// Initial conditions supply E before density exists. Later energy writes must convert to A.
inline void sphParticlesSetInternalEnergy(ParticleInteger index, float energy) {
    CHECK_SPH_PART_BOUNDS(index);
    sphParticlesEntropies[index] = sphParticlesEntropyInitialized ?
        sph_entropy::fromStoredEnergy(energy, sphParticlesDensities[index],
            getOptions().hydro.gamma, sphParticlesThermalScaleFactor) :
        energy;
}

inline float& sphParticlesSmoothingLength(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesSmoothingLengths[index];
}

inline float& sphParticlesDensity(ParticleInteger index) {
    return sphParticlesDensities[index];
}

inline float sphParticlesPressure(ParticleInteger index) {
    // Options require hydro.gamma=5/3 whenever chemistry or star formation is enabled.
    float const ratio = getOptions().hydro.gamma - 1.0f;
    return ratio * sphParticlesDensity(index) * sphParticlesInternalEnergy(index);
}
