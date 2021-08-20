/*
 * groups.hpp
 *
 *  Created on: Aug 10, 2021
 *      Author: dmarce1
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
std::pair<size_t,size_t> groups_save(int number);
void groups_cull();

#endif /* GROUPS_HPP_ */
