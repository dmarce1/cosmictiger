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

#pragma once

#include <cosmictiger/time.hpp>
#include <cstdio>
#include <cosmictiger/sph.hpp>
#include <cosmictiger/sph_tree.hpp>

struct driver_params {
	double a;
	double dummy;
	double tau;
	double tau_max;
	double energy0;
	energies_t energies;
	int max_rung;
	int iter;
	int step;
	size_t total_processed;
	double flops;
	double runtime;
	double years;
	time_type itime;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & step;
		arc & a;
		arc & tau;
		arc & tau_max;
		arc & energies;
		arc & energy0;
		arc & iter;
		arc & total_processed;
		arc & flops;
		arc & runtime;
		arc & itime;
		arc & years;
	}
};

#include <cosmictiger/kick.hpp>

void write_checkpoint(driver_params params);
driver_params read_checkpoint();
sph_tree_create_return sph_step1(int minrung, double scale, double tau, double t0, int phase, double adot, int max_rung, int iter, double dt, double* eheat, bool verbose = true);
sph_run_return sph_step2(int minrung, double scale, double tau, double t0, int phase, double adot, int max_rung, int iter, double dt, energies_t*  eheat, bool verbose = true);
std::pair<kick_return, tree_create_return> kick_step(int minrung, double scale, double, double t0, double theta, bool first_call, bool full_eval);


void driver();
