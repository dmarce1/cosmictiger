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

// Cooling stores an equilibrium electron-fraction guess; hydro owns entropy.
SPH_PARTICLES_EXTERN float* sphParticlesElectronFractions;

// Canonical A=P_physical/rho_physical^gamma. Before the first density solve these
// slots temporarily hold the initial E=a^2 u; prepareThermodynamics converts once.
extern bool sphParticlesEntropyInitialized;
extern double sphParticlesThermalScaleFactor;
void sphParticlesPrepareThermodynamics(double scaleFactor, bool initializeEntropy);
SPH_PARTICLES_EXTERN float* sphParticlesEntropies;
// The stored smoothing length h is the full compact support of the SPH kernel.
// For the cubic spline h = 2h_M; renaming this storage does not rescale its values.
SPH_PARTICLES_EXTERN float* sphParticlesSmoothingLengths;
SPH_PARTICLES_EXTERN ParticleInteger* sphParticlesParticleIndices;
SPH_PARTICLES_EXTERN float* sphParticlesDensities;
// Every density pass rebuilds this factor from weighted divergence and curl, so it is
// neither migrated nor checkpointed.
SPH_PARTICLES_EXTERN uint8_t* sphParticlesBalsaraFactors;
// Variational operator only: zeta_i of cosmictiger/sph/volume.hpp, written by the density
// pass for every semiactive particle before each force pass. Never migrated or checkpointed.
SPH_PARTICLES_EXTERN float* sphParticlesGradHCorrections;
SPH_PARTICLES_EXTERN Float16Storage* sphParticlesBulkViscosityCoefficients;

// Convergence and semiactive working flags share one byte with the latest hydro
// timestep requirement. The proposed rung is separate from the live particle rung,
// retained until the next estimate, and applied only at common boundaries.
// This storage persists across SPH passes, migration, and checkpoints.
SPH_PARTICLES_EXTERN SphParticleFlags* sphParticlesFlags;

/// Read the gas particle's proposed timestep rung from packed flags.
inline int sphParticlesProposedRung(ParticleInteger gas) {
    return sphParticlesFlags[gas].proposedRung();
}

/// Validate and replace the gas particle's packed timestep proposal.
inline void sphParticlesSetProposedRung(ParticleInteger gas, int rung) {
    sphParticlesFlags[gas].setProposedRung(rung);
}

/// Return the hydrodynamic rung proposal currently stored for this gas particle.
inline int sphParticlesHydroRung(ParticleInteger gas) {
    return sphParticlesProposedRung(gas);
}

ParticleInteger sphParticlesSize();
void sphParticlesResize(ParticleInteger particleCount, bool resizeGravityParticles = true);
void sphParticlesFree();
void sphParticlesRemove(ParticleInteger index);
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
// Sets every local and child gas particle's alpha, for scenario stages with constant alpha.
void sphParticlesSetBulkViscosityCoefficients(float bulkViscosityCoefficient);
void sphParticlesResetSemiactive();
void sphParticlesLoad(FILE* fp);
void sphParticlesSave(FILE* fp);
void sphCheckpointTest();

/// Return the configured uniform gas-particle mass after checking the gas index.
inline float sphParticlesMass(ParticleInteger index) {
    auto const& opts = getOptions();
    CHECK_SPH_PART_BOUNDS(index);
    return float(opts.initial.gasMass);
}

/// Read the gas particle's smoothing-solve convergence flag.
inline bool sphParticlesConverged(ParticleInteger index) {
    return sphParticlesFlags[index].converged();
}

/// Set the gas particle's smoothing-solve convergence flag.
inline void sphParticlesSetConverged(ParticleInteger index, bool value) {
    sphParticlesFlags[index].setConverged(value);
}

/// Read whether neighbor activity has marked this gas particle semiactive.
inline bool sphParticlesSemiactive(ParticleInteger index) {
    return sphParticlesFlags[index].semiactive();
}

/// Update the gas particle's neighbor-induced activity flag.
inline void sphParticlesSetSemiactive(ParticleInteger index, bool value) {
    sphParticlesFlags[index].setSemiactive(value);
}

/// Access the mapping from a gas-array index to the main particle arrays.
inline ParticleInteger& sphParticlesParticleIndex(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesParticleIndices[index];
}

/// Access an active position component through the gas-to-particle mapping.
inline Position& sphParticlesPosition(int dim, int index) {
    return particlesPos(dim, sphParticlesParticleIndex(index));
}

/// Access an active stored velocity component through the gas-to-particle mapping.
inline float& sphParticlesVelocity(int dim, int index) {
    return particlesVel(dim, sphParticlesParticleIndex(index));
}

/// Read a mapped particle position component, using the inactive-dimension fallback where
/// applicable.
inline Position sphParticlesPositionValue(int dim, int index) {
    return particlesPositionValue(dim, sphParticlesParticleIndex(index));
}
/// Read a mapped particle velocity component, returning zero for inactive dimensions.
inline float sphParticlesVelocityValue(int dim, int index) {
    return particlesVelocityValue(dim, sphParticlesParticleIndex(index));
}

/// Access the main particle timestep rung associated with this gas index.
inline char& sphParticlesRung(int index) {
    return particlesRung(sphParticlesParticleIndex(index));
}

/// Access initialized gas entropy storage; thermodynamic preparation must have completed.
inline float& sphParticlesEntropy(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    ALWAYS_ASSERT(sphParticlesEntropyInitialized);
    return sphParticlesEntropies[index];
}

/// Return stored specific energy, converting from entropy after thermodynamic
/// initialization.
inline float sphParticlesInternalEnergy(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    if (!sphParticlesEntropyInitialized) {
        return sphParticlesEntropies[index];
    }
    auto const& opts = getOptions();
    return sph_entropy::storedEnergy(sphParticlesEntropies[index], sphParticlesDensities[index],
        opts.hydro.heatCapacityRatio, sphParticlesThermalScaleFactor);
}

// Initial conditions supply E before density exists. Later energy writes must convert to A.
inline void sphParticlesSetInternalEnergy(ParticleInteger index, float energy) {
    auto const& opts = getOptions();
    CHECK_SPH_PART_BOUNDS(index);
    sphParticlesEntropies[index] = sphParticlesEntropyInitialized ?
        sph_entropy::fromStoredEnergy(energy, sphParticlesDensities[index],
            opts.hydro.heatCapacityRatio, sphParticlesThermalScaleFactor) :
        energy;
}

/// Access the gas particle's smoothing length.
inline float& sphParticlesSmoothingLength(ParticleInteger index) {
    CHECK_SPH_PART_BOUNDS(index);
    return sphParticlesSmoothingLengths[index];
}

/// Access the stored gas density.
inline float& sphParticlesDensity(ParticleInteger index) {
    return sphParticlesDensities[index];
}

/// Compute stored gas pressure from density, internal energy, and the configured adiabatic
/// index.
inline float sphParticlesPressure(ParticleInteger index) {
    auto const& opts = getOptions();
    // Options require hydro.heatCapacityRatio=5/3 whenever chemistry or star formation is enabled.
    float const ratio = opts.hydro.heatCapacityRatio - 1.0f;
    return ratio * sphParticlesDensity(index) * sphParticlesInternalEnergy(index);
}
