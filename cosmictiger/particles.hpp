/*
 * particles.hpp
 *
 *  Created on: Jul 17, 2021
 *      Author: dmarce1
 */

#ifndef PARTICLES_HPP_
#define PARTICLES_HPP_

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/fixed.hpp>

#include <fstream>

using part_int = int;

#ifdef PARTICLES_CPP
#define PARTICLES_EXTERN
#else
#define PARTICLES_EXTERN extern
#endif

struct particle {
	array<fixed32, NDIM> x;
	array<float, NDIM> v;
	char r;
	template<class A>
	void serialize(A && a, unsigned) {
		for (int dim = 0; dim < NDIM; dim++) {
			a & x[dim];
			a & v[dim];
		}
		a & r;
	}
};

struct particle_sample {
	array<fixed32, NDIM> x;
	array<float, NDIM> g;
	float p;
	template<class A>
	void serialize(A && a, unsigned) {
		for (int dim = 0; dim < NDIM; dim++) {
			a & x[dim];
			a & g[dim];
		}
		a & p;
	}
};

PARTICLES_EXTERN array<vector<fixed32>, NDIM> particles_x;
PARTICLES_EXTERN array<vector<float, pinned_allocator<float>>, NDIM> particles_v;
PARTICLES_EXTERN vector<char, pinned_allocator<char>> particles_r;
PARTICLES_EXTERN array<vector<float, pinned_allocator<float>>, NDIM> particles_g;
PARTICLES_EXTERN vector<float, pinned_allocator<float>> particles_p;

struct particle_global_range {
	int proc;
	pair<part_int> range;
};

inline float& particles_pot(part_int index) {
	return particles_p[index];
}

inline fixed32& particles_pos(int dim, part_int index) {
	return particles_x[dim][index];
}

inline float& particles_vel(int dim, part_int index) {
	return particles_v[dim][index];
}

inline char& particles_rung(part_int index) {
	return particles_r[index];
}

inline float& particles_gforce(int dim, part_int index) {
	return particles_g[dim][index];
}


inline particle particles_get_particle(part_int index) {
	particle p;
	for (int dim = 0; dim < NDIM; dim++) {
		p.x[dim] = particles_pos(dim, index);
		p.v[dim] = particles_vel(dim, index);
	}
	p.r = particles_rung(index);
	return p;
}

inline void particles_set_particle(particle p, part_int index) {
	for (int dim = 0; dim < NDIM; dim++) {
		particles_pos(dim, index) = p.x[dim];
		particles_vel(dim, index) = p.v[dim];
	}
	particles_rung(index) = p.r;
}



part_int particles_size();
void particles_resize(part_int);
void particles_random_init();
void particles_destroy();
void particles_global_read_pos(particle_global_range, fixed32* x, fixed32* y, fixed32* z, part_int offset);
void particles_global_touch_pos(particle_global_range range);
part_int particles_sort(pair<part_int> rng, double xm, int xdim);
void particles_cache_free();
vector<particle_sample> particles_sample(int cnt);
void particles_load(FILE* fp);
void particles_save(std::ofstream& fp);

#endif /* PARTICLES_HPP_ */
