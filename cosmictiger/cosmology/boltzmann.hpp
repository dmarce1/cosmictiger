/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
*/

#pragma once

#include <cstddef>
#include <vector>

#include <cosmictiger/cosmology/zero_order.hpp>

struct BoltzmannSolverOptions {
    // Local error control for matter/metric transfers; tiny radiation tails
    // use an amplitude floor. CMB spectra require separate hierarchy studies.
    double relativeTolerance = 1.e-6;
    double absoluteTolerance = 1.e-10;
    double maxLogStep = 0.1;
    int maxMultipole = 64;
    size_t maxSteps = 2000000;
};

// Signed synchronous-gauge transfers for initial eta=+1 (MB C=1/2).
// k is in Mpc^-1; thetaBaryonOverK is theta_b/k; primes denote d/dln(a).
// CAMB's unit-curvature convention has initial eta=-1, hence opposite signs.
struct BoltzmannTransfer {
    double scaleFactor;
    double deltaCdm;
    double deltaBaryon;
    double deltaNeutrino;
    double thetaBaryonOverK;
    double hPrime;
    double eta;
    double etaPrime;
    double photonShear;
    double neutrinoShear;
    double conformalTime;
    // Hamiltonian constraint compared with an independently integrated
    // Einstein trace equation, divided by the sum of absolute terms.
    double constraintResidual;
    size_t acceptedSteps;
    size_t rejectedSteps;
};

// Flat CDM+baryon+photon+massless-neutrino cosmologies only. Scale factors
// must be nondecreasing and inside the supplied background's domain.
std::vector<BoltzmannTransfer> solveBoltzmannTransfers(ZeroOrderUniverse const& universe, double k,
    std::vector<double> const& scaleFactors, BoltzmannSolverOptions const& options = {});
