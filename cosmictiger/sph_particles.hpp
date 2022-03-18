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

#define NCHEMFRACS 7
#define CHEM_HP 0
#define CHEM_HN 1
#define CHEM_H2 2
#define CHEM_HE 3
#define CHEM_HEP 4
#define CHEM_HEPP 5
#define CHEM_Z 6

struct sph_particle {
	float eint;
	array<float, NDIM> v;
	float gamma;
	float alpha;
	array<float, NCHEMFRACS> chem;
	template<class A>
	void serialize(A&&arc, unsigned) {
		static const bool stars = get_options().stars;
		arc & eint;
		arc & v;
		arc & gamma;
		arc & alpha;
		arc & chem;
	}
};

SPH_PARTICLES_EXTERN float* sph_particles_a;			// alpha
SPH_PARTICLES_EXTERN part_int* sph_particles_dm;   // dark matter index
SPH_PARTICLES_EXTERN float* sph_particles_h; // smoothing length
SPH_PARTICLES_EXTERN float* sph_particles_e; // energy
SPH_PARTICLES_EXTERN float* sph_particles_da1; // dalpha_pred
SPH_PARTICLES_EXTERN float* sph_particles_fp; // potential correction
SPH_PARTICLES_EXTERN array<float*, NDIM> sph_particles_dvx; // dvel_pred
SPH_PARTICLES_EXTERN array<float*, NDIM> sph_particles_dv1; // dvel_pred
SPH_PARTICLES_EXTERN float* sph_particles_de1; // deint_pred
SPH_PARTICLES_EXTERN array<float*, NDIM> sph_particles_g; // gravity
SPH_PARTICLES_EXTERN array<float, NCHEMFRACS>* sph_particles_chem0; // chemistry
SPH_PARTICLES_EXTERN array<float, NCHEMFRACS>* sph_particles_dchem1; // chemistry
SPH_PARTICLES_EXTERN array<float, NCHEMFRACS>* sph_particles_dchem2; // chemistry
SPH_PARTICLES_EXTERN float* sph_particles_dvv; // divv
SPH_PARTICLES_EXTERN float* sph_particles_crsv; // divv
SPH_PARTICLES_EXTERN float* sph_particles_fv; // balsara
SPH_PARTICLES_EXTERN float* sph_particles_f0; // kernel correction
SPH_PARTICLES_EXTERN float* sph_particles_s2; //
SPH_PARTICLES_EXTERN float* sph_particles_dc; // diffusion constant
SPH_PARTICLES_EXTERN float* sph_particles_ta; // conduction constant
SPH_PARTICLES_EXTERN float* sph_particles_ta0; // conduction constant
SPH_PARTICLES_EXTERN float* sph_particles_dvv0; //
SPH_PARTICLES_EXTERN float* sph_particles_de2; // deint_con
SPH_PARTICLES_EXTERN float* sph_particles_gt; // deint_con
SPH_PARTICLES_EXTERN array<float*, NDIM> sph_particles_dv2; // dvel_con
SPH_PARTICLES_EXTERN char* sph_particles_sa;   // semi-activel_con
SPH_PARTICLES_EXTERN char* sph_particles_or;   // semi-active
SPH_PARTICLES_EXTERN float* sph_particles_da2; // dalpha_con

struct aux_quantities {
	float fpre;
	float divv;
	float crsv;
	float shearv;
	float gradT;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & fpre;
		arc & divv;
		arc & crsv;
		arc & shearv;
		arc & gradT;
	}
};

part_int sph_particles_size();
void sph_particles_resize(part_int sz, bool parts2 = true);
void sph_particles_free();
void sph_particles_cache_free();
void sph_particles_resolve_with_particles();
void sph_particles_sort_by_particles(pair<part_int> rng);
void sph_particles_swap(part_int i, part_int j);
part_int sph_particles_sort(pair<part_int> rng, fixed32 xm, int xdim);
void sph_particles_global_read_force(particle_global_range range, float* x, float* y, float* z, float* divv, part_int offset);
void sph_particles_global_read_gforce(particle_global_range range, float* x, float* y, float* z, part_int offset);
void sph_particles_global_read_pos(particle_global_range range, fixed32* x, fixed32* y, fixed32* z, part_int offset);
void sph_particles_global_read_sph(particle_global_range range, float a, float* eint, float* vx, float* vy, float* vz, float* gamma, float* alpha,
		array<float, NCHEMFRACS>* chems, part_int offset);
void sph_particles_global_read_rungs_and_smoothlens(particle_global_range range, char*, float*, part_int offset);
void sph_particles_global_read_aux(particle_global_range range, float* fpre, float* divv, float* crossv, float* shearv, float* gradT, part_int offset);

void sph_particles_load(FILE* fp);
void sph_particles_save(FILE* fp);
float sph_particles_max_smooth_len();
float sph_particles_temperature(part_int, float);
float sph_particles_mmw(part_int);
float sph_particles_lambda_e(part_int, float, float);

std::pair<double, double> sph_particles_apply_updates(int, int, float, float, float = 1.0);
/*
 inline float& sph_particles_SN(part_int index) {
 return sph_particles_sn[index];
 }
 */

inline float& sph_particles_difco(part_int index) {
	return sph_particles_dc[index];
}

inline float& sph_particles_divv0(part_int index) {
	return sph_particles_dvv0[index];
}

inline float& sph_particles_taux(part_int index) {
	return sph_particles_ta[index];
}

inline float& sph_particles_taux0(part_int index) {
	return sph_particles_ta0[index];
}

inline float& sph_particles_gradT(part_int index) {
	return sph_particles_gt[index];
}

inline float& sph_particles_alpha(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_a[index];
}
/*
 inline float& sph_particles_formY(part_int index) {
 CHECK_SPH_PART_BOUNDS(index);
 return sph_particles_fY[index];
 }
 inline float& sph_particles_formZ(part_int index) {
 CHECK_SPH_PART_BOUNDS(index);
 return sph_particles_fZ[index];
 }
 */
inline float sph_particles_H(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	float H = 1.f;
	for (int fi = 0; fi < NCHEMFRACS; fi++) {
		H -= sph_particles_chem0[index][fi];
	}
	return H;
}

inline float& sph_particles_frac(int j, part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_chem0[index][j];
}

inline float& sph_particles_shear(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_s2[index];
}

inline float& sph_particles_Z(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_chem0[index][CHEM_Z];
}

inline float& sph_particles_He0(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_chem0[index][CHEM_HE];
}

inline float& sph_particles_Hp(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_chem0[index][CHEM_HP];
}

inline float& sph_particles_Hn(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_chem0[index][CHEM_HN];
}

inline float& sph_particles_H2(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_chem0[index][CHEM_H2];
}

inline float& sph_particles_Hep(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_chem0[index][CHEM_HEP];
}

inline float& sph_particles_Hepp(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_chem0[index][CHEM_HEPP];
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

inline char& sph_particles_semi_active(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_sa[index];
}

inline float& sph_particles_fpre(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_f0[index];
}

inline float& sph_particles_fpot(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_fp[index];
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

inline char& sph_particles_oldrung(int index) {
	return sph_particles_or[index];
}

/*inline float& sph_particles_dchem(part_int index) {
 CHECK_SPH_PART_BOUNDS(index);
 return sph_particles_dz[index];
 }
 */
inline float& sph_particles_eint(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_e[index];
}

inline float& sph_particles_deint_pred(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_de1[index];
}

inline float& sph_particles_dalpha_pred(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_da1[index];
}

inline float& sph_particles_dvel_pred(int dim, part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_dv1[dim][index];
}

inline float& sph_particles_xvel(int dim, part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_dvx[dim][index];
}

inline float& sph_particles_deint_con(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_de2[index];
}

inline array<float,NCHEMFRACS>& sph_particles_chem(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_chem0[index];
}

inline array<float,NCHEMFRACS>& sph_particles_dchem_con(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_dchem2[index];
}

inline array<float,NCHEMFRACS>& sph_particles_dchem_pred(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_dchem1[index];
}

inline float& sph_particles_dalpha_con(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_da2[index];
}

inline float& sph_particles_dvel_con(int dim, part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_dv2[dim][index];
}

inline float& sph_particles_divv(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_dvv[index];
}

inline float& sph_particles_crossv(part_int index) {
	CHECK_SPH_PART_BOUNDS(index);
	return sph_particles_crsv[index];
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

inline float sph_particles_egas(part_int index) {
	return sph_particles_eint(index) + sph_particles_ekin(index);
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
	const float K = sph_particles_eint(index);
	const float h = sph_particles_smooth_len(index);
	const float H2 = sph_particles_H2(index);
	const float cv = 1.5f + H2;
	const float gamma = 1.f + 1.f / cv;
	float E = K * powf(rho, gamma) / (gamma - 1.f);
	E *= (4.0 * M_PI / 3.0) * h * sqr(h);
	return E;
}

float sph_particles_coloumb_log(part_int i, float a);

inline sph_particle sph_particles_get_particle(part_int index, float a) {
	sph_particle p;
	p.eint = sph_particles_eint(index);
	for (int dim = 0; dim < NDIM; dim++) {
		p.v[dim] = sph_particles_vel(dim, index);
	}
	static const bool chem = get_options().chem;
	static const bool stars = get_options().stars;
	if (chem) {
		float rhoH2 = sph_particles_H2(index);
		float nh2 = 0.5f * rhoH2 / (1.f - .75f * sph_particles_Y(index) - 0.5f * rhoH2);
		float cv = 1.5 + nh2;
		float gamma = 1.f + 1.f / cv;
		p.gamma = gamma;
		p.chem = sph_particles_chem(index);
	}
	p.alpha = sph_particles_alpha(index);
	return p;
}

inline aux_quantities sph_particles_aux_quantities(part_int index) {
	aux_quantities aux;
	aux.fpre = sph_particles_fpre(index);
	aux.divv = sph_particles_divv(index);
	aux.crsv = sph_particles_crossv(index);
	aux.shearv = sph_particles_shear(index);
	aux.gradT = sph_particles_gradT(index);
	return aux;
}
