/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cstdio>
#include <stddef.h>

#include <cosmictiger/core/force_step.hpp>
#include <cosmictiger/core/time.hpp>
#include <cosmictiger/particles/particles.hpp>
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/sph_tree.hpp>
#endif

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

#if COSMICTIGER_WITH_SPH
SphTreeCreateReturn sphStep1(int minrung, bool verbose = true);
SphRunReturn sphStep2(int minrung, double scale, double tau, double t0, int phase, double adot,
    EnergiesT* energies, bool verbose = true, int uniformMinimumRung = -1,
    bool closingOnly = false);
#endif
#if COSMICTIGER_WITH_GRAVITY
ForceStepResult kickStep(
    int minrung, double scale, double, double t0, double theta, bool firstCall, bool fullEval);
ForceStepResult kickStepHierarchical(int& minimumRung, int maximumRung, double scale,
    double scaleDerivative, double time, double baseTimestep, double theta, EnergiesT* energies,
    int minimumRungFloor, bool doPotential, hpx::future<void>* lightcone, bool keepTop, double drag,
    int gravityErrorStep = -1, bool closingOnly = false);

#endif

ForceStepResult advanceForces(int minRung, double scale, double scaleDerivative, double time,
    double baseTimestep, double theta, bool firstCall, bool fullEvaluation, EnergiesT& energies,
    int gravityErrorStep = -1, bool closingOnly = false);

void runFixedScenario();
void driver();
