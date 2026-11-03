/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */
#pragma once

#include <cosmictiger/chemistry/gadget4_cooling.hpp>
#include <cosmictiger/hydro/sph/cullen_and_dehnen/state.hpp>
#include <cosmictiger/hydro/sph/entropy.hpp>
#include <cosmictiger/hydro/sph/viscosity_state.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/particles.hpp>

#ifdef SPH_PARTICLES_CPP
#define SPH_PARTICLES_EXTERN
#else
#define SPH_PARTICLES_EXTERN extern
#endif
#ifdef CHECK_BOUNDS
#define CHECK_SPH_PART_BOUNDS(i)                                                                 \
    if (i < 0 || i >= sphParticlesSize()) {                                                      \
        PRINT("particle bound check failure %li should be between %li and %li\n", (long long) i, \
            (long long) 0, (long long) sphParticlesSize());                                      \
        ALWAYS_ASSERT(false);                                                                    \
    }
#else
#define CHECK_SPH_PART_BOUNDS(i)
#endif

struct SphAcceleration {
    Vector3<float> acceleration;
};

// Deferred cooling stores an equilibrium electron-fraction guess; hydro owns entropy.
SPH_PARTICLES_EXTERN gadget4::State* sphParticlesChemistryStates;

// Canonical A=P_physical/rho_physical^gamma. Before the first density solve these
// slots temporarily hold the initial E=a^2 u; prepareThermodynamics converts once.
extern bool sphParticlesEntropyInitialized;
extern double sphParticlesThermalScaleFactor;
void sphParticlesPrepareThermodynamics(double scaleFactor, bool initializeEntropy);
SPH_PARTICLES_EXTERN float* sphParticlesEntropies;
// Compact support radius of the SPH smoothing kernel.
SPH_PARTICLES_EXTERN float* sphParticlesSupportRadii;
SPH_PARTICLES_EXTERN SphAcceleration* sphParticlesAccelerationStates;
SPH_PARTICLES_EXTERN ParticleInteger* sphParticlesParticleIndices;
// Gravity of the latest walk. The walk does not kick gas: hydro adds it to the
// gas acceleration and owns the gas momentum.
SPH_PARTICLES_EXTERN Array<float*, kDimensions> sphParticlesGravityAccelerations;
SPH_PARTICLES_EXTERN float* sphParticlesEntropyRates;
SPH_PARTICLES_EXTERN float* sphParticlesDensities;
// Density diagnostics, rebuilt by every density pass.
SPH_PARTICLES_EXTERN float* sphParticlesBalsaraFactors;
// Variational operator only: zeta_i of cosmictiger/hydro/sph/volume.hpp, written by the density
// pass for every semiactive particle before each force pass. Never migrated or checkpointed.
SPH_PARTICLES_EXTERN float* sphParticlesGradHCorrections;
SPH_PARTICLES_EXTERN SphViscosityState* sphParticlesViscosityStates;
SPH_PARTICLES_EXTERN SphCullenDehnenState* sphParticlesCullenDehnenStates;
SPH_PARTICLES_EXTERN char* sphParticlesConvergenceFlags;
SPH_PARTICLES_EXTERN char* sphParticlesSemiactiveFlags;
// Next rung proposed by the force pass that closes a step; -1 when none is staged. The
// corrector and the source closes still need the closing rung, so the proposal reaches
// particlesRung only through sphParticlesCommitProposedRungs. Never migrated or checkpointed.
SPH_PARTICLES_EXTERN char* sphParticlesProposedRungs;

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
void sphParticlesGlobalReadInternalEnergyAndSupportRadius(
    ParticleGlobalRange range, float* internalEnergy, float* supportRadius, ParticleInteger offset);
void sphParticlesGlobalReadRungs(ParticleGlobalRange range, char*, ParticleInteger offset);
void sphParticlesGlobalReadVelocities(
    ParticleGlobalRange range, float*, float*, float*, ParticleInteger offset);
void sphParticlesGlobalReadDensity(ParticleGlobalRange range, float*, ParticleInteger offset,
    float* balsaraFactor = nullptr, float* viscosityAlpha = nullptr,
    float* gradHCorrection = nullptr);
void sphParticlesResetConverged();
void sphParticlesResetSemiactive();
void sphParticlesLoad(FILE* fp);
void sphParticlesSave(FILE* fp);
void sphCheckpointTest();
// Phase 0 opens the stage-time prediction of the active gas, phase 1 closes the step with
// the fresh rates, phase 2 opens the next step, and phase 3 only sums. Phases 2 and 3
// return the gas energies and momenta at the synchronized state, before any opening kick.
EnergiesT sphParticlesApplyUpdates(
    int minimumRung, int phase, float timeStepScale, float time, double explicitTimestep = -1.0);
// Move every staged proposal into particlesRung on all localities and clear it. Each gas
// particle at or above minimumRung must have one.
void sphParticlesCommitProposedRungs(int minimumRung);
// Cullen-Dehnen and Price et al.: commit the pending divergences and move alpha toward the
// density pass's target before the force pass snapshots it.
void sphParticlesUpdateTargetAlpha(
    int minimumRung, float timeStepScale, float time, double explicitTimestep = -1.0);

inline float sphParticlesMass(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return float(getOptions().initial.gasMass);
}

inline char& sphParticlesConverged(ParticleInteger index) {
    return sphParticlesConvergenceFlags[index];
}

inline char& sphParticlesSemiactive(ParticleInteger index) {
    return sphParticlesSemiactiveFlags[index];
}

inline SphAcceleration& sphParticlesAccelerationState(ParticleInteger index) {
    return sphParticlesAccelerationStates[index];
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

inline float& sphParticlesEntropyRate(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesEntropyRates[index];
}

inline float& sphParticlesAcceleration(int dim, ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesAccelerationStates[index].acceleration[dim];
}

inline float& sphParticlesGravityAcceleration(int dim, ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesGravityAccelerations[dim][index];
}

inline float& sphParticlesSupportRadius(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesSupportRadii[index];
}

inline float& sphParticlesDensity(ParticleInteger index) {
    return sphParticlesDensities[index];
}

inline float sphParticlesPressure(ParticleInteger index) {
    // Options require hydro.gamma=5/3 whenever chemistry or star formation is enabled.
    float const ratio = getOptions().hydro.gamma - 1.0f;
    return ratio * sphParticlesDensity(index) * sphParticlesInternalEnergy(index);
}
