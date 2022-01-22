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


#ifndef SPH_PARTICLES_HPP_
#define SPH_PARTICLES_HPP_

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/fixed.hpp>
#include <cosmictiger/group_entry.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/range.hpp>

#include <atomic>
#include <fstream>
#include <unordered_map>

#ifdef LONG_LONG_PART_INT
using part_int = long long;
#else
using part_int = int;
#endif

#define NO_GROUP group_int(0x7FFFFFFFFFFFFFFFLL)

#ifdef SPH_PARTICLES_CPP
#define SPH_PARTICLES_EXTERN
#else
#define SPH_PARTICLES_EXTERN extern
#endif

struct group_sph_particle {
	array<fixed32, NDIM> x;
	group_int g;
	template<class A>
	void serialize(A && arc, unsigned) {
		arc & x;
		arc & g;
	}
};

struct output_sph_particle {
	array<fixed32, NDIM> x;
	array<float, NDIM> v;
	char r;
	template<class A>
	void serialize(A && a, unsigned) {
		a & x;
		a & v;
		a & r;
	}
};

struct sph_particle {
	array<fixed32, NDIM> x;
	array<float, NDIM> v;
	group_int lg;
	char r;
	char t;
	template<class A>
	void serialize(A && a, unsigned) {
		static bool do_groups = get_options().do_groups;
		static bool do_tracers = get_options().do_tracers;
		for (int dim = 0; dim < NDIM; dim++) {
			a & x[dim];
			a & v[dim];
		}
		a & r;
		if (do_groups) {
			a & lg;
		}
		if (do_tracers) {
			a & t;
		}
	}
};

struct sph_particle_sample {
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

#ifdef CHECK_BOUNDS
#define CHECK_PART_BOUNDS(i)                                                                                                                            \
	if( i < 0 || i >= sph_particles_size()) {                                                                                                            \
		PRINT( "sph_particle bound check failure %li should be between %li and %li\n", (long long) i, (long long) 0, (long long) sph_particles_size());  \
		ALWAYS_ASSERT(false);                                                                                                                           \
	}
#else
#define CHECK_PART_BOUNDS(i)
#endif

SPH_PARTICLES_EXTERN array<fixed32*, NDIM> sph_particles_x;
SPH_PARTICLES_EXTERN array<float*, NDIM> sph_particles_v;
SPH_PARTICLES_EXTERN char* sph_particles_r;
SPH_PARTICLES_EXTERN array<float*, NDIM> sph_particles_g;
SPH_PARTICLES_EXTERN float* sph_particles_p;
SPH_PARTICLES_EXTERN std::atomic<group_int>* sph_particles_grp
#ifdef SPH_PARTICLES_CPP
= nullptr
#endif
;
SPH_PARTICLES_EXTERN group_int* sph_particles_lgrp;
SPH_PARTICLES_EXTERN char* sph_particles_tr;
SPH_PARTICLES_EXTERN size_t sph_particles_global_offset;

struct sph_particle_global_range {
	int proc;
	pair<part_int> range;
};

part_int sph_particles_size();
std::unordered_map<int, part_int> sph_particles_groups_init();
void sph_particles_groups_destroy();
void sph_particles_resize(part_int);
void sph_particles_random_init();
void sph_particles_destroy();
void sph_particles_global_read_pos(sph_particle_global_range, fixed32* x, fixed32* y, fixed32* z, part_int offset);
void sph_particles_global_read_pos_and_group(sph_particle_global_range range, fixed32* x, fixed32* y, fixed32* z, group_int* g, part_int offset);
part_int sph_particles_sort(pair<part_int> rng, double xm, int xdim);
void sph_particles_cache_free();
void sph_particles_group_cache_free();
vector<output_sph_particle> sph_particles_get_sample(const range<double>& box);
vector<sph_particle_sample> sph_particles_sample(int cnt);
void sph_particles_load(FILE* fp);
void sph_particles_save(FILE* fp);
void sph_particles_inc_group_cache_epoch();
int sph_particles_group_home(group_int);
void sph_particles_set_tracers(size_t count=0);
vector<output_sph_particle> sph_particles_get_tracers();
void sph_particles_memadvise_cpu();
void sph_particles_memadvise_gpu();
void sph_particles_free();



inline float& sph_particles_pot(part_int index) {
	CHECK_PART_BOUNDS(index);
	return sph_particles_p[index];
}

inline fixed32& sph_particles_pos(int dim, part_int index) {
	CHECK_PART_BOUNDS(index);
	return sph_particles_x[dim][index];
}

inline float& sph_particles_vel(int dim, part_int index) {
	CHECK_PART_BOUNDS(index);
	return sph_particles_v[dim][index];
}

inline char& sph_particles_rung(part_int index) {
	CHECK_PART_BOUNDS(index);
	return sph_particles_r[index];
}

inline float& sph_particles_gforce(int dim, part_int index) {
	CHECK_PART_BOUNDS(index);
	return sph_particles_g[dim][index];
}

inline group_int sph_particles_group_init(part_int index) {
	CHECK_PART_BOUNDS(index);
	return sph_particles_global_offset + index;
}

inline std::atomic<group_int>& sph_particles_group(part_int index) {
	CHECK_PART_BOUNDS(index);
	ASSERT(sph_particles_grp);
	return sph_particles_grp[index];
}

inline group_int& sph_particles_lastgroup(part_int index) {
	CHECK_PART_BOUNDS(index);
	ASSERT(sph_particles_lgrp);
	return sph_particles_lgrp[index];
}

inline char& sph_particles_tracer(part_int index) {
	CHECK_PART_BOUNDS(index);
	return sph_particles_tr[index];
}


inline sph_particle sph_particles_get_sph_particle(part_int index) {
	static bool do_groups = get_options().do_groups;
	static bool do_tracers = get_options().do_tracers;
	CHECK_PART_BOUNDS(index);
	sph_particle p;
	for (int dim = 0; dim < NDIM; dim++) {
		p.x[dim] = sph_particles_pos(dim, index);
		p.v[dim] = sph_particles_vel(dim, index);
	}
	p.r = sph_particles_rung(index);
	if (do_groups) {
		p.lg = sph_particles_lastgroup(index);
	}
	if (do_tracers) {
		p.t = sph_particles_tracer(index);
	}
	return p;
}

inline void sph_particles_set_sph_particle(sph_particle p, part_int index) {
	static bool do_groups = get_options().do_groups;
	static bool do_tracers = get_options().do_tracers;
	CHECK_PART_BOUNDS(index);
	for (int dim = 0; dim < NDIM; dim++) {
		sph_particles_pos(dim, index) = p.x[dim];
		sph_particles_vel(dim, index) = p.v[dim];
	}
	sph_particles_rung(index) = p.r;
	if (do_groups) {
		sph_particles_lastgroup(index) = p.lg;
	}
	if (do_tracers) {
		sph_particles_tracer(index) = p.t;
	}
}

#endif /* SPH_PARTICLES_HPP_ */
