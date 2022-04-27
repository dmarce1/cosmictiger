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
#include <cosmictiger/chemistry.hpp>
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

#define NCHEMFRACS 8
#define CHEM_H 0
#define CHEM_HP 1
#define CHEM_HN 2
#define CHEM_H2 3
#define CHEM_HE 4
#define CHEM_HEP 5
#define CHEM_HEPP 6
#define CHEM_Z 7

class frac_real {
	unsigned short i;
	constexpr static float c0 = 10000.0f;
	constexpr static float c0inv = 1.0f / c0;
public:
	CUDA_EXPORT
	operator float() const {
		return expf(-sqr(i * c0inv));
	}
	CUDA_EXPORT
	frac_real& operator=(float y) {
		ALWAYS_ASSERT(y <= 1.0f);
		if (y > 0.0) {
			i = c0 * sqrtf(-logf(y));
		} else {
			i = (1 << 16) - 1;
		}
		return *this;
	}
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & i;
	}
};

struct sph_particle0 {
	float entr0;
	array<float, NCHEMFRACS> chem0;
	template<class A>
	void serialize(A && arc, unsigned) {
		arc & entr0;
		arc & chem0;
	}
};
struct sph_particle {
	array<float, NCHEMFRACS> frac;
	array<float, NDIM> dvel;
	float omega;
	float h;
	float shearv;
	float alpha;
	float A;
	float fcold;
	float divv;
	template<class Arc>
	void serialize(Arc&&arc, unsigned) {
		arc & frac;
		arc & dvel;
		arc & omega;
		arc & h;
		arc & shearv;
		arc & alpha;
		arc & A;
		arc & fcold;
		arc & divv;
	}
};

struct sph_record1 {
	array<float, NCHEMFRACS> frac;
	float alpha;
};

struct sph_record2 {
	float A;
	float h;
};

struct sph_record5 {
	array<float, NCHEMFRACS> dfrac;
	float dfcold;
};

struct sph_record6 {
	array<float, NDIM> dvel;
};

/*SPH_PARTICLES_EXTERN array<float*, NDIM> sph_particles_dv1; // dvel_pred
 SPH_PARTICLES_EXTERN float* sph_particles_de1; // dentr_pred
 SPH_PARTICLES_EXTERN array<float, NCHEMFRACS>* sph_particles_dchem1; // chemistry
 SPH_PARTICLES_EXTERN float* sph_particles_drc1;
 */

SPH_PARTICLES_EXTERN sph_record1* sph_particles_r1;
SPH_PARTICLES_EXTERN sph_record2* sph_particles_r2;
SPH_PARTICLES_EXTERN sph_record5* sph_particles_r5;
SPH_PARTICLES_EXTERN sph_record6* sph_particles_r6;

SPH_PARTICLES_EXTERN part_int* sph_particles_dm;   // dark matter index
SPH_PARTICLES_EXTERN array<float*, NDIM> sph_particles_dv2; // dvel_pred
SPH_PARTICLES_EXTERN array<float*, NDIM> sph_particles_g; // dvel_pred
SPH_PARTICLES_EXTERN float* sph_particles_dv;
SPH_PARTICLES_EXTERN float* sph_particles_kap;
SPH_PARTICLES_EXTERN float* sph_particles_e0;
SPH_PARTICLES_EXTERN float* sph_particles_de3;
SPH_PARTICLES_EXTERN float* sph_particles_de2;
SPH_PARTICLES_EXTERN float* sph_particles_fp1;
SPH_PARTICLES_EXTERN float* sph_particles_s2;
SPH_PARTICLES_EXTERN float* sph_particles_fc;
SPH_PARTICLES_EXTERN float* sph_particles_rh;
SPH_PARTICLES_EXTERN char* sph_particles_or;
SPH_PARTICLES_EXTERN char* sph_particles_sa;
SPH_PARTICLES_EXTERN char* sph_particles_st;

struct aux_quantities {
	float omega;
	float shearv;
	float alpha;
	float divv;
	array<float, NCHEMFRACS> fracs;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & omega;
		arc & shearv;
		arc & alpha;
		arc & fracs;
		arc & divv;
	}
};

part_int sph_particles_size();
void sph_particles_resize(part_int sz, bool parts2 = true);
void sph_particles_free();
void sph_particles_cache_free1();
void sph_particles_cache_free2();
void sph_particles_cache_free_entr();
void sph_particles_resolve_with_particles();
void sph_particles_sort_by_particles(pair<part_int> rng);
void sph_particles_swap(part_int i, part_int j);
void sph_particles_swap2(part_int i, part_int j);
part_int sph_particles_sort(pair<part_int> rng, fixed32 xm, int xdim);
void sph_particles_global_read_pos(particle_global_range range, fixed32* x, fixed32* y, fixed32* z, part_int offset);
void sph_particles_global_read_fcold(particle_global_range range, float*, char*, part_int offset);
void sph_particles_global_read_entr_and_smoothlen(particle_global_range range, float*, float*, part_int offset);
void sph_particles_global_read_rungs(particle_global_range range, char*, part_int offset);
void sph_particles_global_read_vels(particle_global_range range, float*, float*, float*, part_int offset);
void sph_particles_global_read_kappas(particle_global_range range, float*, part_int offset);
void sph_particles_global_read_rho(particle_global_range range, float*, part_int offset);
void sph_particles_global_read_aux(particle_global_range range, float* alpha, float* omega, float* shearv, array<float, NCHEMFRACS>* fracs,
		part_int offset);
void sph_particles_reset_converged();
void sph_particles_reset_semiactive();
void sph_particles_load(FILE* fp);
void sph_particles_save(FILE* fp);
float sph_particles_max_smooth_len();
float sph_particles_temperature(part_int, float);
float sph_particles_mmw(part_int);
void sph_particles_softlens2smoothlens(int minrung);

std::pair<double, double> sph_particles_apply_updates(int, int, float, float, float = 1.0);
/*
 inline float& sph_particles_SN(part_int index) {
 return sph_particles_sn[index];
 }
 */

inline float& sph_particles_entr0(part_int index) {
	return sph_particles_e0[index];
}

inline char& sph_particles_isstar(part_int index) {
	return sph_particles_st[index];
}

inline char& sph_particles_converged(part_int index) {
	return sph_particles_or[index];
}

inline char& sph_particles_semiactive(part_int index) {
	return sph_particles_sa[index];
}

inline float& sph_particles_kappa(part_int index) {
	return sph_particles_kap[index];
}

inline float& sph_particles_cold_mass(part_int index) {
	return sph_particles_fc[index];
}

inline float& sph_particles_dcold_mass(part_int index) {
	return sph_particles_r5[index].dfcold;
}

inline float& sph_particles_alpha(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_r1[index].alpha;
}

inline float& sph_particles_frac(int j, part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_r1[index].frac[j];
}

inline float& sph_particles_shear(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_s2[index];
}

inline float& sph_particles_Z(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_r1[index].frac[CHEM_Z];
}

inline void sph_particles_normalize_fracs(part_int index) {
	double sum = 0.0;
	for (double fi = 0; fi < NCHEMFRACS; fi++) {
		sum += sph_particles_frac(fi, index);
	}
	for (double fi = 0; fi < NCHEMFRACS; fi++) {
		sph_particles_frac(fi, index) /= sum;
	}
}

inline float& sph_particles_H(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_r1[index].frac[CHEM_H];
}

inline float& sph_particles_He0(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_r1[index].frac[CHEM_HE];
}

inline float& sph_particles_Hp(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_r1[index].frac[CHEM_HP];
}

inline float& sph_particles_Hn(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_r1[index].frac[CHEM_HN];
}

inline float& sph_particles_H2(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_r1[index].frac[CHEM_H2];
}

inline float& sph_particles_Hep(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_r1[index].frac[CHEM_HEP];
}

inline float& sph_particles_Hepp(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_r1[index].frac[CHEM_HEPP];
}

inline float sph_particles_Y(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_He0(index) + sph_particles_Hep(index) + sph_particles_Hepp(index);
}

inline float sph_particles_gamma(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	const float H = sph_particles_H(index);
	const float Hp = sph_particles_Hp(index);
	const float Hn = sph_particles_Hn(index);
	const float H2 = sph_particles_H2(index);
	const float He = sph_particles_He0(index);
	const float Hep = sph_particles_Hep(index);
	const float Hepp = sph_particles_Hepp(index);
	const float Z = sph_particles_Z(index);
	const float ne = Hp - Hn + Hep + 2.0f * Hepp;
	const float n = H + Hp + Hn + 0.5f * H2 + 0.25f * He + 0.25f * Hep + 0.25f * Hepp + 0.1f * Z + ne;
	const float cv = 1.5f + 0.5f * Hn / n;
	return 1.f + 1.f / cv;
}

inline sph_record1& sph_particles_rec1(part_int index) {
	return sph_particles_r1[index];
}

inline sph_record2& sph_particles_rec2(part_int index) {
	return sph_particles_r2[index];
}

inline sph_record5& sph_particles_rec5(part_int index) {
	return sph_particles_r5[index];
}

inline sph_record6& sph_particles_rec6(part_int index) {
	return sph_particles_r6[index];
}

inline float& sph_particles_omega(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_fp1[index];
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

inline float& sph_particles_entr(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_r2[index].A;
}

inline float& sph_particles_dentr2(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_de3[index];
}

inline float& sph_particles_dentr1(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_de2[index];
}

inline float& sph_particles_dvel0(int dim, part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_dv2[dim][index];
}

inline float& sph_particles_dvel(int dim, part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_r6[index].dvel[dim];
}

inline array<float, NCHEMFRACS>& sph_particles_chem(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_r1[index].frac;
}

inline array<float, NCHEMFRACS>& sph_particles_dchem(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_r5[index].dfrac;
}

inline float& sph_particles_divv(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_dv[index];
}

inline float& sph_particles_gforce(int dim, part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_g[dim][index];
}

inline float sph_particles_ekin(part_int index) {
	float ekin = 0.f;
	for (int dim = 0; dim < NDIM; dim++) {
		ekin += 0.5f * sqr(sph_particles_vel(dim, index));
	}
	return ekin;
}

inline float& sph_particles_smooth_len(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_r2[index].h;
}

inline float& sph_particles_rho(part_int index) {
	return sph_particles_rh[index];
}

inline float sph_particles_eint(part_int index) {
	static const float gamma0 = get_options().gamma;
	static const float stars = get_options().stars;
	float hfrac = 1.0f;
	if (stars) {
		hfrac = 1.0f - sph_particles_cold_mass(index);
	}
	const float rho = sph_particles_rho(index) * hfrac;
	const float K = sph_particles_entr(index);
	return K * pow(rho, gamma0 - 1.0) / (gamma0 - 1.0);
}

inline aux_quantities sph_particles_aux_quantities(part_int index) {
	aux_quantities aux;
	aux.alpha = sph_particles_alpha(index);
	aux.omega = sph_particles_omega(index);
	aux.divv = sph_particles_divv(index);
	if (get_options().diffusion) {
		aux.shearv = sph_particles_shear(index);
	}
	if (get_options().chem) {
		aux.fracs = sph_particles_chem(index);
	}
	return aux;
}

inline sph_particle get_sph_particle(part_int i) {
	sph_particle part;
	for (int fi = 0; fi < NCHEMFRACS; fi++) {
		part.frac[fi] = sph_particles_frac(fi, i);
	}
	for (int dim = 0; dim < NDIM; dim++) {
		part.dvel[dim] = sph_particles_dvel(dim, i);
	}
	part.omega = sph_particles_omega(i);
	part.h = sph_particles_smooth_len(i);
	part.shearv = sph_particles_shear(i);
	part.alpha = sph_particles_alpha(i);
	part.A = sph_particles_entr(i);
	part.fcold = sph_particles_cold_mass(i);
	part.divv = sph_particles_divv(i);
	return part;
}

