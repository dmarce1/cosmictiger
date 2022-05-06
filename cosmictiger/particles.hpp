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

#ifndef PARTICLES_HPP_
#define PARTICLES_HPP_

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



static __constant__ float rung_dt[MAX_RUNG] = { 1.0 / (1 << 0), 1.0 / (1 << 1), 1.0 / (1 << 2), 1.0 / (1 << 3), 1.0 / (1 << 4), 1.0 / (1 << 5), 1.0 / (1 << 6),
		1.0 / (1 << 7), 1.0 / (1 << 8), 1.0 / (1 << 9), 1.0 / (1 << 10), 1.0 / (1 << 11), 1.0 / (1 << 12), 1.0 / (1 << 13), 1.0 / (1 << 14), 1.0 / (1 << 15), 1.0
				/ (1 << 16), 1.0 / (1 << 17), 1.0 / (1 << 18), 1.0 / (1 << 19), 1.0 / (1 << 20), 1.0 / (1 << 21), 1.0 / (1 << 22), 1.0 / (1 << 23), 1.0 / (1 << 24),
		1.0 / (1 << 25), 1.0 / (1 << 26), 1.0 / (1 << 27), 1.0 / (1 << 28), 1.0 / (1 << 29), 1.0 / (1 << 30), 1.0 / (1 << 31) };

#define NO_INDEX ((part_int) 0xFFFFFFFFU)

#define DARK_MATTER_TYPE 0
#define SPH_TYPE 1
#define STAR_TYPE 2

#ifdef LONG_LONG_PART_INT
using part_int = long long;
#else
using part_int = int;
#endif

#define NO_GROUP group_int(0x7FFFFFFFFFFFFFFFLL)

#ifdef PARTICLES_CPP
#define PARTICLES_EXTERN
#else
#define PARTICLES_EXTERN extern
#endif

struct group_particle {
	array<fixed32, NDIM> x;
	group_int g;
	template<class A>
	void serialize(A && arc, unsigned) {
		arc & x;
		arc & g;
	}
};

struct output_particle {
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

struct particle {
	array<fixed32, NDIM> x;
	array<float, NDIM> v;
	group_int lg;
	part_int cat_index;
	char r;
	char t;
	char type;
	template<class A>
	void serialize(A && a, unsigned) {
		static bool do_groups = get_options().do_groups;
		static bool do_tracers = get_options().do_tracers;
		static bool sph = get_options().sph;
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
		if (sph) {
			a & cat_index;
			a & type;
		}
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

#ifdef CHECK_BOUNDS
#define CHECK_PART_BOUNDS(i)                                                                                                                            \
	if( i < 0 || i >= particles_size()) {                                                                                                            \
		PRINT( "particle bound check failure %li should be between %li and %li\n", (long long) i, (long long) 0, (long long) particles_size());  \
		ALWAYS_ASSERT(false);                                                                                                                           \
	}
#else
#define CHECK_PART_BOUNDS(i)
#endif

PARTICLES_EXTERN array<fixed32*, NDIM> particles_x;
PARTICLES_EXTERN array<float*, NDIM> particles_v;
PARTICLES_EXTERN char* particles_r;
PARTICLES_EXTERN array<float*, NDIM> particles_g;
PARTICLES_EXTERN float* particles_s;
PARTICLES_EXTERN float* particles_z;
PARTICLES_EXTERN float* particles_rh;
PARTICLES_EXTERN char* particles_c;
PARTICLES_EXTERN char* particles_sa;
PARTICLES_EXTERN float* particles_p;
PARTICLES_EXTERN float* particles_dv;
PARTICLES_EXTERN std::atomic<group_int>* particles_grp
#ifdef PARTICLES_CPP
= nullptr
#endif
;
PARTICLES_EXTERN part_int* particles_sph;
PARTICLES_EXTERN group_int* particles_lgrp;
PARTICLES_EXTERN char* particles_tr;
PARTICLES_EXTERN char* particles_ty;
PARTICLES_EXTERN size_t particles_global_offset;

struct particle_global_range {
	int proc;
	pair<part_int> range;
};

part_int particles_size();
std::unordered_map<int, part_int> particles_groups_init();
void particles_groups_destroy();
void particles_resize(part_int);
void particles_reset_converged();
void particles_random_init();
void particles_resolve_with_sph_particles();
int particles_apply_updates(int minrung, float t0, float a );
void particles_destroy();
void particles_sort_by_sph(pair<part_int> rng);
void particles_global_read_pos(particle_global_range, fixed32* x, fixed32* y, fixed32* z, char* type, float* zeta, part_int offset);
void particles_global_read_softlens(particle_global_range range, float* h, part_int offset);
void particles_global_read_pos_and_group(particle_global_range range, fixed32* x, fixed32* y, fixed32* z, group_int* g, part_int offset);
part_int particles_sort(pair<part_int> rng, double xm, int xdim);
void particles_cache_free();
void particles_group_cache_free();
vector<output_particle> particles_get_sample(const range<double>& box);
vector<particle_sample> particles_sample(int cnt);
void particles_load(FILE* fp);
void particles_save(FILE* fp);
void particles_inc_group_cache_epoch();
void particles_global_read_rungs(particle_global_range range, char* r, part_int offset);
int particles_group_home(group_int);
void particles_set_tracers(size_t count = 0);
vector<output_particle> particles_get_tracers();
void particles_memadvise_cpu();
void particles_global_read_vels(particle_global_range range, float* vx, float* vy, float* z, part_int offset);
void particles_memadvise_gpu();
void particles_free();
void particles_save_glass(const char* filename);

inline char& particles_type(part_int index) {
	static const bool sph = get_options().sph;
	if (sph) {
		CHECK_PART_BOUNDS(index);
		return particles_ty[index];
	} else {
		static char dm = DARK_MATTER_TYPE;
		return dm;
	}
}


inline float particles_mass(part_int index) {
	const float sph_mass = get_options().sph_mass;
	const float dm_mass = get_options().dm_mass;
	if( particles_type(index) == SPH_TYPE) {
		return sph_mass;
	} else {
		return dm_mass;
	}
}

inline float& particles_divv(part_int index) {
	CHECK_PART_BOUNDS(index);
	return particles_dv[index];
}

inline float& particles_pot(part_int index) {
	CHECK_PART_BOUNDS(index);
	return particles_p[index];
}

inline float& particles_rho(part_int index) {
	CHECK_PART_BOUNDS(index);
	return particles_rh[index];
}

inline char& particles_converged(part_int index) {
	CHECK_PART_BOUNDS(index);
	return particles_c[index];
}

inline char& particles_semiactive(part_int index) {
	CHECK_PART_BOUNDS(index);
	return particles_sa[index];
}

inline fixed32& particles_pos(int dim, part_int index) {
	CHECK_PART_BOUNDS(index);
	return particles_x[dim][index];
}

inline float& particles_vel(int dim, part_int index) {
	CHECK_PART_BOUNDS(index);
	return particles_v[dim][index];
}

inline char& particles_rung(part_int index) {
	CHECK_PART_BOUNDS(index);
	return particles_r[index];
}

inline float& particles_softlen(part_int index) {
	CHECK_PART_BOUNDS(index);
	return particles_s[index];
}

inline float& particles_zeta(part_int index) {
	CHECK_PART_BOUNDS(index);
	return particles_z[index];
}

inline float& particles_gforce(int dim, part_int index) {
	CHECK_PART_BOUNDS(index);
	return particles_g[dim][index];
}

inline group_int particles_group_init(part_int index) {
	CHECK_PART_BOUNDS(index);
	return particles_global_offset + index;
}

inline std::atomic<group_int>& particles_group(part_int index) {
	CHECK_PART_BOUNDS(index);
	ASSERT(particles_grp);
	return particles_grp[index];
}

inline group_int& particles_lastgroup(part_int index) {
	CHECK_PART_BOUNDS(index);
	ASSERT(particles_lgrp);
	return particles_lgrp[index];
}

inline char& particles_tracer(part_int index) {
	CHECK_PART_BOUNDS(index);
	return particles_tr[index];
}

inline part_int& particles_cat_index(part_int index) {
	CHECK_PART_BOUNDS(index);
	return particles_sph[index];
}

inline particle particles_get_particle(part_int index) {
	static bool do_groups = get_options().do_groups;
	static bool do_tracers = get_options().do_tracers;
	static bool sph = get_options().sph;
	CHECK_PART_BOUNDS(index);
	particle p;
	for (int dim = 0; dim < NDIM; dim++) {
		p.x[dim] = particles_pos(dim, index);
		p.v[dim] = particles_vel(dim, index);
	}
	p.r = particles_rung(index);
	if (do_groups) {
		p.lg = particles_lastgroup(index);
	}
	if (do_tracers) {
		p.t = particles_tracer(index);
	}
	if (sph) {
		p.type = particles_type(index);
		p.cat_index = particles_cat_index(index);
	}
	return p;
}

inline void particles_set_particle(particle p, part_int index) {
	static bool do_groups = get_options().do_groups;
	static bool do_tracers = get_options().do_tracers;
	static bool sph = get_options().sph;
	CHECK_PART_BOUNDS(index);
	for (int dim = 0; dim < NDIM; dim++) {
		particles_pos(dim, index) = p.x[dim];
		particles_vel(dim, index) = p.v[dim];
	}
	particles_rung(index) = p.r;
	if (do_groups) {
		particles_lastgroup(index) = p.lg;
	}
	if (do_tracers) {
		particles_tracer(index) = p.t;
	}
	if (sph) {
		particles_type(index) = p.type;
		particles_cat_index(index) = p.cat_index;
	}
}

inline bool particles_is_sph(int index) {
	return particles_type(index) == SPH_TYPE;
}

struct energies_t {
	double pot;
	double kin;
	double therm;
	double heating;
	double cosmic;
	energies_t() {
		cosmic = heating = pot = kin = therm = 0.f;
	}
	energies_t& operator+=(const energies_t& other) {
		pot += other.pot;
		kin += other.kin;
		therm += other.therm;
		heating += other.heating;
		cosmic += other.cosmic;
		return *this;
	}
	template<class A>
	void serialize(A&& arc, unsigned ) {
		arc & pot;
		arc & kin;
		arc & therm;
		arc & heating;
		arc & cosmic;
	}
};

energies_t particles_sum_energies();

#endif /* PARTICLES_HPP_ */
