/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 */

#pragma once

#include <cosmictiger/core/time.hpp>

// Derived dyadic schedule for G(h/2) D(h/2) H(h) D(h/2) G(h/2).
// At t = (2m+1) h_r/2, minRung(t) = r+1; only exact rung r has
// a hydro midpoint. A coarse boundary (integer clock zero) has none.
// h_r = baseTimestep * 2^(-r); boundaryTime is the interval-local clock.
inline int sphMidpointRung(TimeType boundaryTime) {
    return boundaryTime == 0 ? -1 : int(minRung(boundaryTime)) - 1;
}
