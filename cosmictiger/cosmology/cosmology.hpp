/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
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
