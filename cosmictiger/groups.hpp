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


#ifndef GROUPS_HPP_
#define GROUPS_HPP_


#include <cosmictiger/defs.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/range.hpp>
#include <cosmictiger/safe_io.hpp>


struct particle_data {
	array<fixed32, NDIM> x;
	array<float, NDIM> v;
	group_int last_group;
	part_int index;
	int rank;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & x;
		arc & v;
		arc & last_group;
		arc & index;
		arc & rank;
	}
};

void groups_add_particles(int wave, double scale);
void groups_reduce(double);
std::pair<size_t,size_t> groups_save(int number, double time);
void groups_cull();

#endif /* GROUPS_HPP_ */
