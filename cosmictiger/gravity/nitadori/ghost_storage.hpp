/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <cosmictiger/gravity/sources.hpp>

// Frozen source record sent once per occupied PMMM cell fragment.
struct ParticleGhost {
    Array<Position, NDIM> x;
    char type = DARK_MATTER_TYPE;
    char rung = 0;

    template <class Archive>
    void serialize(Archive& archive, unsigned) {
        archive & x & type & rung;
    }
};

FmmSourceView fmmGhostView();
// Reserve or clear only after readers of the previous arena epoch finish.
// Clear retains the allocation; free also releases the backing storage.
void fmmGhostReserve(ParticleInteger count);
void fmmGhostClear();
void fmmGhostFree();
