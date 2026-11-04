/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <format>
#include <iostream>

#include <cosmictiger/chemistry/gadget4_cooling.hpp>
#include <cosmictiger/cuda/cuda_export.hpp>
#include <cosmictiger/math/float16_storage.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/vector3.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/thermodynamics.hpp>
#include <cosmictiger/sph/viscosity/cullen_and_dehnen_state.hpp>

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

using SphAcceleration = Vector3<float>;

// Cooling stores an equilibrium electron-fraction guess; hydro owns entropy.
SPH_PARTICLES_EXTERN float* sphParticlesElectronFractions;

// Canonical A=P_physical/rho_physical^gamma. Before the first density solve these
// slots temporarily hold the initial E=a^2 u; prepareThermodynamics converts once.
extern bool sphParticlesEntropyInitialized;
extern double sphParticlesThermalScaleFactor;
void sphParticlesPrepareThermodynamics(double scaleFactor, bool initializeEntropy);
SPH_PARTICLES_EXTERN float* sphParticlesEntropies;
// Compact support radius of the SPH smoothing kernel.
SPH_PARTICLES_EXTERN float* sphParticlesSupportRadii;
SPH_PARTICLES_EXTERN ParticleInteger* sphParticlesParticleIndices;
SPH_PARTICLES_EXTERN float* sphParticlesDensities;
// Density diagnostics, rebuilt by every density pass.
SPH_PARTICLES_EXTERN float* sphParticlesBalsaraFactors;
// Variational operator only: zeta_i of cosmictiger/sph/volume.hpp, written by the density
// pass for every semiactive particle before each force pass. Never migrated or checkpointed.
SPH_PARTICLES_EXTERN float* sphParticlesGradHCorrections;
SPH_PARTICLES_EXTERN Float16Storage* sphParticlesBulkViscosityCoefficients;
SPH_PARTICLES_EXTERN SphCullenDehnenState* sphParticlesCullenDehnenStates;
// Temporary density/viscosity diagnostics, allocated only for one evaluation.
SPH_PARTICLES_EXTERN SphViscosityDiagnostics* sphParticlesViscosityDiagnostics;
SPH_PARTICLES_EXTERN char* sphParticlesConvergenceFlags;
SPH_PARTICLES_EXTERN char* sphParticlesSemiactiveFlags;
// Latest hydro timestep requirement, separate from the live particle rung.
// Retained until the next force estimate and applied only at common boundaries.
// Migrated and checkpointed across intervening finer-rung events.
SPH_PARTICLES_EXTERN char* sphParticlesProposedRungs;

inline int sphParticlesHydroRung(ParticleInteger gas) {
    return std::max(0, int(sphParticlesProposedRungs[gas]));
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
void sphParticlesGlobalReadInternalEnergyAndSupportRadius(
    ParticleGlobalRange range, float* internalEnergy, float* supportRadius, ParticleInteger offset);
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
// Commit history and viscosity only at the particle's own midpoint.
void sphParticlesUpdateTargetAlpha(SphRunParameters const& parameters);

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
