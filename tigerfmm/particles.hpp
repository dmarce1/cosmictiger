/*
 * particles.hpp
 *
 *  Created on: Jul 17, 2021
 *      Author: dmarce1
 */

#ifndef PARTICLES_HPP_
#define PARTICLES_HPP_

#include <tigerfmm/containers.hpp>
#include <tigerfmm/defs.hpp>
#include <tigerfmm/fixed.hpp>

#include <thrust/system/cuda/experimental/pinned_allocator.h>

#ifdef PARTICLES_CPP
#define PARTICLES_EXTERN
#else
#define PARTICLES_EXTERN extern
#endif

template<class T>
using pinned_allocator = thrust::system::cuda::experimental::pinned_allocator< T >;

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
	pair<int> range;
};

inline fixed32& particles_pos(int dim, int index) {
	return particles_x[dim][index];
}

inline float& particles_vel(int dim, int index) {
	return particles_v[dim][index];
}

inline char& particles_rung(int index) {
	return particles_r[index];
}

inline float& particles_gforce(int dim, int index) {
	return particles_g[dim][index];
}

inline float& particles_pot(int index) {
	return particles_p[index];
}

inline particle particles_get_particle(int index) {
	particle p;
	for (int dim = 0; dim < NDIM; dim++) {
		p.x[dim] = particles_pos(dim, index);
		p.v[dim] = particles_vel(dim, index);
	}
	p.r = particles_rung(index);
	return p;
}

inline void particles_set_particle(particle p, int index) {
	for (int dim = 0; dim < NDIM; dim++) {
		particles_pos(dim, index) = p.x[dim];
		particles_vel(dim, index) = p.v[dim];
	}
	particles_rung(index) = p.r;
}

int particles_size();
void particles_resize(int);
void particles_random_init();
void particles_destroy();
void particles_global_read_pos(particle_global_range, vector<fixed32>& x, vector<fixed32>& y, vector<fixed32>& z, int offset);
int particles_sort(pair<int, int> rng, double xm, int xdim);
void particles_cache_free();
vector<particle_sample> particles_sample(int cnt);

#endif /* PARTICLES_HPP_ */
