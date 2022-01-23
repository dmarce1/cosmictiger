#pragma once

#include <cosmictiger/particles.hpp>

#ifdef SPH_PARTICLES_CPP
#define SPH_PARTICLES_EXTERN
#else
#define SPH_PARTICLES_EXTERN extern
#endif

#define NOT_SPH ((part_int) 0xFFFFFFFFU)

#ifdef CHECK_BOUNDS
#define CHECK_SPH_PART_BOUNDS(i)                                                                                                                            \
	if( i < 0 || i >= sph_particles_size()) {                                                                                                            \
		PRINT( "particle bound check failure %li should be between %li and %li\n", (long long) i, (long long) 0, (long long) particles_size());  \
		ALWAYS_ASSERT(false);                                                                                                                           \
	}
#else
#define CHECK_SPH_PART_BOUNDS(i)
#endif

SPH_PARTICLES_EXTERN part_int* sph_particles_dm;
SPH_PARTICLES_EXTERN float* sph_particles_h;
SPH_PARTICLES_EXTERN float* sph_particles_r;
SPH_PARTICLES_EXTERN float* sph_particles_drdh;
SPH_PARTICLES_EXTERN float* sph_particles_e;
SPH_PARTICLES_EXTERN array<float*, NDIM> sph_particles_dv;
SPH_PARTICLES_EXTERN float* sph_particles_de;

part_int sph_particles_size();
void sph_particles_resize(part_int sz);
void sph_particles_free();


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

inline float& sph_particles_rho(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_r[index];
}

inline float& sph_particles_drhodh(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_drdh[index];
}

inline float& sph_particles_dent(part_int index) {
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



inline float& sph_particles_smooth_len(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_h[index];
}
