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
	array<fixed32,NDIM> x;
	array<float,NDIM> v;
	char r;
	template<class A>
	void serialize(A && a, unsigned) {
		for( int dim = 0; dim < NDIM; dim++) {
			a & x[dim];
			a & v[dim];
		}
		a & r;
	}
};



PARTICLES_EXTERN array<vector<fixed32, pinned_allocator<fixed32>>, NDIM> particles_x;
PARTICLES_EXTERN array<vector<float, pinned_allocator<float>>, NDIM> particles_v;
PARTICLES_EXTERN vector<char, pinned_allocator<char>> particles_r;




inline fixed32& particles_pos(int dim, int index) {
	return particles_x[dim][index];
}

inline float& particles_vel(int dim, int index) {
	return particles_v[dim][index];
}

inline char& particles_rung(int index) {
	return particles_r[index];
}

inline particle particles_get_particle(int index) {
	particle p;
	for( int dim = 0; dim < NDIM; dim++) {
		p.x[dim] = particles_pos(dim,index);
		p.v[dim] = particles_vel(dim,index);
	}
	p.r = particles_rung(index);
	return p;
}

inline void particles_set_particle(particle p, int index) {
	for( int dim = 0; dim < NDIM; dim++) {
		particles_pos(dim,index) = p.x[dim];
		particles_vel(dim,index) = p.v[dim];
	}
	particles_rung(index) = p.r;
}

int particles_size();
void particles_resize(int);
int particles_size_pos();
void particles_resize_pos(int);
void particles_random_init();
void particles_destroy();
int particles_sort(pair<int,int> rng, double xm, int xdim);

#endif /* PARTICLES_HPP_ */
