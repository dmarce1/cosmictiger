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

struct driver_params {
	double a;
	double dummy;
	double tau;
	double tau_max;
	double cosmicK;
	double esum0;
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
		arc & cosmicK;
		arc & esum0;
		arc & iter;
		arc & total_processed;
		arc & flops;
		arc & runtime;
		arc & itime;
		arc & years;
	}
};

void write_checkpoint(driver_params params);
driver_params read_checkpoint();

void driver();
