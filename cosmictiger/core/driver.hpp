/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cstdio>
#include <stddef.h>

#include <cosmictiger/core/time.hpp>
#include <cosmictiger/gravity/kick.hpp>
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/sph_tree.hpp>

struct DriverParameters {
    double a;
    double adot;
    double dummy;
    double tau;
    double tauMax;
    double energy0;
    EnergiesT energies;
    int maxRung;
    int iter;
    int minrung0;
    int step;
    int bucketSize;
    size_t totalProcessed;
    double flops;
    double runtime;
    double years;
    TimeType itime;
    template <typename A>
    void serialize(A&& arc, unsigned) {
        arc & bucketSize;
        arc & maxRung;
        arc & adot;
        arc & minrung0;
        arc & step;
        arc & a;
        arc & tau;
        arc & tauMax;
        arc & energies;
        arc & energy0;
        arc & iter;
        arc & totalProcessed;
        arc & flops;
        arc & runtime;
        arc & itime;
        arc & years;
    }
};

SphTreeCreateReturn sphStep1(int minrung, bool verbose = true);
SphRunReturn sphStep2(int minrung, double scale, double tau, double t0, int phase, double adot,
    EnergiesT* energies, bool verbose = true);
std::pair<KickReturn, TreeCreateReturn> kickStep(
    int minrung, double scale, double, double t0, double theta, bool firstCall, bool fullEval);

std::pair<KickReturn, TreeCreateReturn> advanceForces(int minRung, double scale,
    double scaleDerivative, double time, double baseTimestep, double theta, bool firstCall,
    bool fullEvaluation, EnergiesT& energies);

void runFixedScenario();
void driver();
