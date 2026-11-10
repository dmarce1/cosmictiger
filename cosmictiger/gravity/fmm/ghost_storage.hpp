/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <cosmictiger/gravity/sources.hpp>
#include <cosmictiger/math/vector3.hpp>

// Frozen source record sent once per occupied remote PMMM cell.
struct ParticleGhost {
    Vector3<Position> position;
    char type = kDarkMatterType;
    char rung = 0;
    float mass = 1.0f;

    /// Archive this record in the field order required by distributed transport.
    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & position & type & rung & mass;
    }
};

FmmSourceView fmmGhostView();
// Reserve or clear only after readers of the previous arena epoch finish.
// Clear retains the allocation; free also releases the backing storage.
void fmmGhostReserve(ParticleInteger count);
void fmmGhostClear();
void fmmGhostFree();
