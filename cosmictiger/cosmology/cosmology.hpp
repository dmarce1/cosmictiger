/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

Distributed under the Boost Software License, Version 1.0.
See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
*/

#ifndef COSMOLOGY_HPP_
#define COSMOLOGY_HPP_

#include <cosmictiger/core/containers.hpp>

void cosmosUpdate(double& adotdot, double& adot, double& a, double dt);
double cosmosGrowthFactor(double omegaM, float a);
double cosmosDadtau(double a);
double cosmosDadt(double a);
double cosmosTime(double a0, double a1);
double cosmosConformalTime(double a0, double a1);
double cosmosTauToScale(double a0, double t1);
void cosmosNfwFit(Vector<double> n, double& rs, double& rho0);
double cosmosKlypinFit(double vmax, double rvir, double mvir);
double cosmosAinv(double& adot, double& a, double dt0);

#endif /* COSMOLOGY_HPP_ */
