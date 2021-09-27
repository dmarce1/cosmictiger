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


#ifndef GPUTIGER_BOLTZMANN_HPP_
#define GPUTIGER_BOLTZMANN_HPP_
#include <cosmictiger/zero_order.hpp>
#include <cosmictiger/interp.hpp>

#define LMAX 1024

#define hdoti 0
#define etai 1
#define taui 2
#define deltaci 3
#define deltabi 4
#define thetabi 5
#define FLi 6
#define GLi (6+LMAX)
#define NLi (6+2*LMAX)

#define deltagami (FLi+0)
#define thetagami (FLi+1)
#define F2i (FLi+2)
#define deltanui (NLi+0)
#define thetanui (NLi+1)
#define N2i (NLi+2)
#define G0i (GLi+0)
#define G1i (GLi+1)
#define G2i (GLi+2)

#define NFIELD (6+(3*LMAX))

#include <array>

#include <cosmictiger/zero_order.hpp>

using cos_state = std::array<double,NFIELD>;

void set_zeroverse(zero_order_universe* z);
void free_zeroverse();

void einstein_boltzmann_init(cos_state* uptr, const zero_order_universe* uni_ptr, double k,
		double normalization, double a, double ns);
void einstein_boltzmann(cos_state* uptr, const zero_order_universe *uni_ptr, double k, double amin, double amax);

void einstein_boltzmann_init_set(cos_state* U, zero_order_universe* uni, double kmin, double kmax, int N,
		double amin, double normalization);

void einstein_boltzmann_interpolation_function(interp_functor<double>* cdm_k_func,  interp_functor<double>* vel_k_func,
		cos_state* U, zero_order_universe* uni, double kmin, double kmax, double norm, int N, double astart, double astop, bool cont, double ns);

#endif /* GPUTIGER_BOLTZMANN_HPP_ */
