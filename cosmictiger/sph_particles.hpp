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

#include <cosmictiger/particles.hpp>
#include <cosmictiger/math.hpp>

#ifdef SPH_PARTICLES_CPP
#define SPH_PARTICLES_EXTERN
#else
#define SPH_PARTICLES_EXTERN extern
#endif
#ifdef CHECK_BOUNDS
#define CHECK_SPH_PART_BOUNDS(i)                                                                                                                            \
	if( i < 0 || i >= sph_particles_size()) {                                                                                                            \
		PRINT( "particle bound check failure %li should be between %li and %li\n", (long long) i, (long long) 0, (long long) sph_particles_size());  \
		ALWAYS_ASSERT(false);                                                                                                                           \
	}
#else
#define CHECK_SPH_PART_BOUNDS(i)
#endif

struct sph_particle {
	float ent;
	array<float, NDIM> v;
	float H2;
	template<class T>
	void serialize(T&&arc, unsigned) {
		arc & ent;
		arc & v;
	}
};

#define NCHEMFRACS 5
#define CHEM_HP 0
#define CHEM_HN 1
#define CHEM_H2 2
#define CHEM_HEP 3
#define CHEM_HEPP 4

SPH_PARTICLES_EXTERN part_int* sph_particles_dm;
SPH_PARTICLES_EXTERN float* sph_particles_h;
SPH_PARTICLES_EXTERN float* sph_particles_e;
SPH_PARTICLES_EXTERN char* sph_particles_sa;
SPH_PARTICLES_EXTERN array<float*, NDIM> sph_particles_dv;
SPH_PARTICLES_EXTERN array<float*, NCHEMFRACS> sph_particles_chem;
SPH_PARTICLES_EXTERN float* sph_particles_dvv;
SPH_PARTICLES_EXTERN float* sph_particles_de;
SPH_PARTICLES_EXTERN float* sph_particles_fv;
SPH_PARTICLES_EXTERN float* sph_particles_f0;
SPH_PARTICLES_EXTERN float* sph_particles_ts;
#ifdef CHECK_MUTUAL_SORT
SPH_PARTICLES_EXTERN part_int* sph_particles_tst;
inline part_int& sph_particles_test(int index) {
	return sph_particles_tst[index];
}
#endif

part_int sph_particles_size();
void sph_particles_resize(part_int sz);
void sph_particles_free();
void sph_particles_cache_free();
void sph_particles_resolve_with_particles();
void sph_particles_sort_by_particles(pair<part_int> rng);
void sph_particles_swap(part_int i, part_int j);
part_int sph_particles_sort(pair<part_int> rng, fixed32 xm, int xdim);
void sph_particles_global_read_gforce(particle_global_range range, float* x, float* y, float* z, part_int offset);
void sph_particles_global_read_pos(particle_global_range range, fixed32* x, fixed32* y, fixed32* z, part_int offset);
void sph_particles_global_read_sph(particle_global_range range, float* ent, float* vx, float* vy, float* vz, float* H2, part_int offset);
void sph_particles_global_read_rungs_and_smoothlens(particle_global_range range, char*, float*, part_int offset);
void sph_particles_global_read_fvels(particle_global_range range, float* fvels, float* fpre, part_int offset);
void sph_particles_load(FILE* fp);
void sph_particles_save(FILE* fp);
float sph_particles_max_smooth_len();

inline float& sph_particles_time_to_star(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_ts[index];
}

inline float& sph_particles_Hp(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_chem[CHEM_HP][index];
}

inline float& sph_particles_Hn(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_chem[CHEM_HN][index];
}

inline float& sph_particles_H2(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_chem[CHEM_H2][index];
}

inline float& sph_particles_Hep(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_chem[CHEM_HEP][index];
}

inline float& sph_particles_Hepp(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_chem[CHEM_HEPP][index];
}

inline char& sph_particles_semi_active(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_sa[index];
}

inline float& sph_particles_fvel(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_fv[index];
}

inline float& sph_particles_fpre(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_f0[index];
}

inline part_int& sph_particles_dm_index(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_dm[index];
}

inline fixed32& sph_particles_pos(int dim, int index) {
	return particles_pos(dim, sph_particles_dm_index(index));
}

inline float& sph_particles_vel(int dim, int index) {
	return particles_vel(dim, sph_particles_dm_index(index));
}

inline char& sph_particles_rung(int index) {
	return particles_rung(sph_particles_dm_index(index));
}

inline float& sph_particles_dent(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_de[index];
}

inline float& sph_particles_tcool(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_de[index];
}

inline float& sph_particles_ent(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_e[index];
}

inline float& sph_particles_dvel(int dim, part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_dv[dim][index];
}

inline float& sph_particles_divv(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_dvv[index];
}

inline float& sph_particles_gforce(int dim, part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_dv[dim][index];
}

inline float& sph_particles_smooth_len(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_h[index];
}

inline float sph_particles_rho(part_int index) {
	const float h = sph_particles_smooth_len(index);
	static const float mass = get_options().sph_mass;
	static const float N = get_options().neighbor_number;
	static const float c0 = N * mass * 3.0 / (4.0 * M_PI);
	return c0 / (h * sqr(h));
}

inline float sph_particles_energy(part_int index) {
	const float rho = sph_particles_rho(index);
	const float K = sph_particles_ent(index);
	const float h = sph_particles_smooth_len(index);
	const float H2 = sph_particles_H2(index);
	const float cv = 1.5f + H2;
	const float gamma = 1.f + 1.f / cv;
	float E = K * powf(rho, gamma) / (gamma - 1.f);
	E *= (4.0 * M_PI / 3.0) * h * sqr(h);
	return E;
}

inline sph_particle sph_particles_get_particle(part_int index) {
	sph_particle p;
	p.ent = sph_particles_ent(index);
	for (int dim = 0; dim < NDIM; dim++) {
		p.v[dim] = sph_particles_vel(dim, index);
	}
	static const bool chem = get_options().chem;
	if (chem) {
		p.H2 = sph_particles_H2(index);
	}
	return p;
}
