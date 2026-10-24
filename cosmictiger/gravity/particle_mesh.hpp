#pragma once

#include <cosmictiger/particles/particles.hpp>

struct ParticleMeshDiagnostics {
    double sourceMass = 0;
    double depositedMass = 0;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & sourceMass & depositedMass;
    }
};

// Collective root entry points: call once on the controlling HPX locality.
// Sorting preserves particle ownership and permutes the local particle arrays.
void particleMeshSolve(bool doPotential);
void particleMeshDestroy();
ParticleMeshDiagnostics particleMeshDiagnostics();

// Local results are valid until a subsequent particle permutation or PM solve.
Array<float, NDIM> const& particleMeshAcceleration(ParticleInteger index);
float particleMeshPotential(ParticleInteger index);
double particleMeshEffectiveRadius();
