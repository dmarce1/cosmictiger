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


#ifndef DOMAIN_HPP_
#define DOMAIN_HPP_

#include <cosmictiger/defs.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/range.hpp>

void domains_begin(int rung);
void domains_end();
void domains_rebound();
range<double> domains_find_my_box();
range<double> domains_range(size_t key);
void domains_save(FILE* fp);
void domains_load(FILE *fp);
double domains_get_load_imbalance();
pair<int, double> domains_find_ewald_level(double theta, size_t key = 1, range<double> box = unit_box<double>(), int depth = 0, double rb = -1.0);


struct domain_t {
	range<double> box;
	double midx;
	int rank;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & box;
		arc & midx;
		arc & rank;
	}
};

struct domain_global {
	range<double> box;
	pair<int> proc_range;
	size_t total_count;
	size_t lo_count;
	double midhi;
	double midlo;
	size_t key;
};

struct domain_local {
	range<double> box;
	pair<int> proc_range;
	pair<part_int> part_range;
	int depth;
};

#endif /* DOMAIN_HPP_ */
