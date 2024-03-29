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

#include <cosmictiger/containers.hpp>


void cosmos_update(double& adotdot, double& adot, double& a, double dt);
double cosmos_growth_factor(double omega_m, float a);
double cosmos_dadtau(double a);
double cosmos_dadt(double a);
double cosmos_time(double a0, double a1);
double cosmos_conformal_time(double a0, double a1);
double cosmos_tau_to_scale(double a0, double t1);
void cosmos_NFW_fit(vector<double> n, double& Rs, double& rho0);
double cosmos_Klypin_fit(double vmax, double rvir, double mvir);
double cosmos_ainv(double& adot, double& a, double dt0);



#endif /* COSMOLOGY_HPP_ */
