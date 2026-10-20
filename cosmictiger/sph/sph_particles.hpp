/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */
#pragma once

#include <cosmictiger/chemistry/primordial_types.hpp>
#include <cosmictiger/math/math.hpp>
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

struct SphThermodynamicState {
    float internalEnergy;    // Stored a^2 times the specific internal energy.
    float supportRadius;     // Compact support radius of the SPH smoothing kernel.
};

struct SphAcceleration {
    Array<float, NDIM> acceleration;
};

// Optional chemistry storage. Abundances are authoritative; the source bridge
// derives chemistry energy from the hydrodynamic stored a^2 u at each use.
SPH_PARTICLES_EXTERN primordial::State* sphParticlesChemistryStates;
SPH_PARTICLES_EXTERN Pair<float>* sphParticlesChemistryEos;

SPH_PARTICLES_EXTERN SphThermodynamicState* sphParticlesThermodynamicStates;
SPH_PARTICLES_EXTERN SphAcceleration* sphParticlesAccelerationStates;
SPH_PARTICLES_EXTERN ParticleInteger* sphParticlesParticleIndices;
// Rates used by the predictor, retained until the corrector replaces their kick.
SPH_PARTICLES_EXTERN Array<float*, NDIM> sphParticlesPredictedAccelerations;
SPH_PARTICLES_EXTERN Array<float*, NDIM> sphParticlesGravityAccelerations;
SPH_PARTICLES_EXTERN float* sphParticlesPredictedInternalEnergyRates;
SPH_PARTICLES_EXTERN float* sphParticlesInternalEnergyRates;
SPH_PARTICLES_EXTERN float* sphParticlesDensities;
SPH_PARTICLES_EXTERN char* sphParticlesConvergenceFlags;
SPH_PARTICLES_EXTERN char* sphParticlesSemiactiveFlags;

ParticleInteger sphParticlesSize();
void sphParticlesResize(ParticleInteger particleCount, bool resizeGravityParticles = true);
void sphParticlesFree();
void sphParticlesClearHydroCaches();
void sphParticlesClearCaches();
void sphParticlesResolveWithParticles();
void sphParticlesSwap(ParticleInteger i, ParticleInteger j);
ParticleInteger sphParticlesSort(
    Pair<ParticleInteger> particleRange, Fixed32 splitPosition, int splitDimension);
void sphParticlesGlobalReadPositions(
    ParticleGlobalRange range, Fixed32* x, Fixed32* y, Fixed32* z, ParticleInteger offset);
void sphParticlesGlobalReadInternalEnergyAndSupportRadius(
    ParticleGlobalRange range, float* internalEnergy, float* supportRadius, ParticleInteger offset);
void sphParticlesGlobalReadRungs(ParticleGlobalRange range, char*, ParticleInteger offset);
void sphParticlesGlobalReadVelocities(
    ParticleGlobalRange range, float*, float*, float*, ParticleInteger offset);
void sphParticlesGlobalReadDensity(ParticleGlobalRange range, float*, ParticleInteger offset);
void sphParticlesGlobalReadChemistryEos(ParticleGlobalRange range, float* pressureEnergyRatio,
    float* adiabaticIndex, ParticleInteger offset);
void sphParticlesRefreshChemistryEos(double scaleFactor);
void sphParticlesResetConverged();
void sphParticlesResetSemiactive();
void sphParticlesLoad(FILE* fp);
void sphParticlesSave(FILE* fp);
void sphCheckpointTest();
void sphParticlesApplyUpdates(int minimumRung, int phase, float timeStepScale, float time);

inline float& sphParticlesPredictedInternalEnergyRate(ParticleInteger index) {
    return sphParticlesPredictedInternalEnergyRates[index];
}

inline char& sphParticlesConverged(ParticleInteger index) {
    return sphParticlesConvergenceFlags[index];
}

inline char& sphParticlesSemiactive(ParticleInteger index) {
    return sphParticlesSemiactiveFlags[index];
}

inline SphThermodynamicState& sphParticlesThermodynamicState(ParticleInteger index) {
    return sphParticlesThermodynamicStates[index];
}

inline SphAcceleration& sphParticlesAccelerationState(ParticleInteger index) {
    return sphParticlesAccelerationStates[index];
}

inline ParticleInteger& sphParticlesParticleIndex(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesParticleIndices[index];
}

inline Fixed32& sphParticlesPosition(int dim, int index) {
    return particlesPos(dim, sphParticlesParticleIndex(index));
}

inline float& sphParticlesVelocity(int dim, int index) {
    return particlesVel(dim, sphParticlesParticleIndex(index));
}

inline char& sphParticlesRung(int index) {
    return particlesRung(sphParticlesParticleIndex(index));
}

inline float& sphParticlesInternalEnergy(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesThermodynamicStates[index].internalEnergy;
}

inline float& sphParticlesInternalEnergyRate(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesInternalEnergyRates[index];
}

inline float& sphParticlesPredictedAcceleration(int dim, ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesPredictedAccelerations[dim][index];
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
    return sphParticlesThermodynamicStates[index].supportRadius;
}

inline float& sphParticlesDensity(ParticleInteger index) {
    return sphParticlesDensities[index];
}

inline float sphParticlesPressure(ParticleInteger index) {
    float const ratio = getOptions().chemistry.enabled ? sphParticlesChemistryEos[index].first :
                                                         getOptions().hydro.gamma - 1.0f;
    return ratio * sphParticlesDensity(index) * sphParticlesInternalEnergy(index);
}
