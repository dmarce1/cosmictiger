/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2026  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
*/

#ifndef DRIFT_HPP_
#define DRIFT_HPP_

#include <cosmictiger/core/defs.hpp>
#include <cosmictiger/particles/particles.hpp>

void drift(double scale, double dt, double tau0, double tau1, double tauMax, int rung = -1,
    double drag = 0.0);

void cudaDrift(int rung, float a, float dt, float tau0, float tau1, float tauMax, double drag);

void cpuDrift(int rung, float a, float dt, float tau0, float tau1, float tauMax, double drag);

#endif /* DRIFT_HPP_ */
