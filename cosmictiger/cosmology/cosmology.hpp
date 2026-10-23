/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
*/

#ifndef COSMOLOGY_HPP_
#define COSMOLOGY_HPP_

#include <cosmictiger/core/containers.hpp>

struct CosmosAdvance {
    double physicalTime;
    double driftIntegral;
};

// Advance signed physical adot and a by a conformal interval. Returned time
// and drift quadratures use the same adaptive stages as the background.
CosmosAdvance cosmosAdvance(double& adot, double& a, double dt);
void cosmosUpdate(double& adotdot, double& adot, double& a, double dt);
double cosmosGrowthFactor(double omegaM, float a);
// Diagnostic/output callers disable accounting; evolution callers keep it enabled.
double cosmosDadtau(double a, bool countFlops = true);
double cosmosDadt(double a);
double cosmosD2adt2(double a);
// Scale-to-time queries refer to the expanding branch; reversed limits are signed.
// Simple turnaround endpoints are allowed, inaccessible intervals throw.
double cosmosTime(double a0, double a1);
double cosmosConformalTime(double a0, double a1, bool countFlops = true);
double cosmosTauToScale(double a0, double t1);
void cosmosNfwFit(Vector<double> n, double& rs, double& rho0);
double cosmosKlypinFit(double vmax, double rvir, double mvir);
double cosmosAinv(double& adot, double& a, double dt0);

#endif /* COSMOLOGY_HPP_ */
