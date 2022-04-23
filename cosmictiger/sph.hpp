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

#ifndef SPH_HPP_
#define SPH_HPP_

#define SPH_KERNEL_ORDER 5

#include <cosmictiger/defs.hpp>
#include <cosmictiger/options.hpp>

#include <cosmictiger/stack_vector.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/range.hpp>
#include <cosmictiger/constants.hpp>

#include <atomic>

#define SPH_AV_HU 0
#define SPH_AV_MM 1
#define SPH_AV_CONSTANT 2

struct sph_values {
	float vx;
	float vy;
	float vz;
	float rho;
	float p;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & vx;
		arc & vy;
		arc & vz;
		arc & rho;
		arc & p;
	}
};

#ifndef __CUDACC__
struct sph_tree_neighbor_return {
	fixed32_range inner_box;
	fixed32_range outer_box;
	sph_values value_at;
	bool has_value_at;CUDA_EXPORT
	sph_tree_neighbor_return() {
		has_value_at = false;
		for (int dim = 0; dim < NDIM; dim++) {
			inner_box.begin[dim] = 1.9;
			inner_box.end[dim] = -0.9;
			outer_box.begin[dim] = 1.9;
			outer_box.end[dim] = -0.9;
		}
	}
	CUDA_EXPORT
	sph_tree_neighbor_return& operator+=(const sph_tree_neighbor_return& other) {
		for (int dim = 0; dim < NDIM; dim++) {
			inner_box.begin[dim] = std::min(inner_box.begin[dim], other.inner_box.begin[dim]);
			inner_box.end[dim] = std::max(inner_box.end[dim], other.inner_box.end[dim]);
			outer_box.begin[dim] = std::min(outer_box.begin[dim], other.outer_box.begin[dim]);
			outer_box.end[dim] = std::max(outer_box.end[dim], other.outer_box.end[dim]);
		}
		if (!has_value_at && other.has_value_at) {
			value_at = other.value_at;
			has_value_at = true;
		}
		return *this;

	}
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & has_value_at;
		arc & inner_box;
		arc & outer_box;
		arc & value_at;
	}
};
#endif

#define SPH_TREE_NEIGHBOR_BOXES 0
#define SPH_TREE_NEIGHBOR_NEIGHBORS 1
#define SPH_TREE_NEIGHBOR_VALUE_AT 2
#define SPH_INTERACTIONS_I 1
#define SPH_INTERACTIONS_J 2
#define SPH_INTERACTIONS_IJ 3
#define SPH_SET_ACTIVE 1
#define SPH_SET_ALL 4

struct sph_tree_neighbor_params {
	int run_type;
	float h_wt;
	int min_rung;
	int seti;
	int seto;
	double x;
	double y;
	double z;
	template<class T>
	void serialize(T&& arc, unsigned) {
		arc & seti;
		arc & seto;
		arc & run_type;
		arc & h_wt;
		arc & min_rung;
		arc & x;
		arc & y;
		arc & z;
	}
};

struct sph_run_return {
	float hmin;
	float hmax;
	double flops;
	int max_rung_hydro;
	int max_rung_grav;
	int max_rung;
	float max_vsig;
	bool rc;
	float ekin;
	float ent;
	float etherm;
	float momx;
	float momy;
	float momz;
	float vol;
	float dtinv_cfl;
	float dtinv_visc;
	float dtinv_diff;
	float dtinv_cond;
	float dtinv_divv;
	float dtinv_acc;
	float dtinv_omega;
	sph_run_return() {
		ent = 0.0;
		hmax = 0.0;
		hmin = std::numeric_limits<float>::max();
		max_rung_hydro = 0;
		max_rung_grav = 0;
		max_rung = 0;
		max_vsig = 0.0;
		etherm = vol = ekin = momx = momy = momz = 0.0;
		dtinv_cfl = 0.0f;
		dtinv_visc = 0.0f;
		dtinv_diff = 0.0f;
		dtinv_cond = 0.0f;
		dtinv_divv = 0.0f;
		dtinv_acc = 0.0f;
		dtinv_omega = 0.0f;
		rc = false;
	}
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & dtinv_cfl;
		arc & dtinv_visc;
		arc & dtinv_diff;
		arc & dtinv_cond;
		arc & dtinv_divv;
		arc & dtinv_acc;
		arc & hmin;
		arc & hmax;
		arc & max_rung_hydro;
		arc & max_rung_grav;
		arc & max_rung;
		arc & max_vsig;
		arc & ekin;
		arc & momx;
		arc & momy;
		arc & momz;
		arc & vol;
		arc & ent;
		arc & etherm;
		arc & dtinv_omega;
	}
	sph_run_return& operator+=(const sph_run_return& other) {
		hmax = std::max(hmax, other.hmax);
		hmin = std::min(hmin, other.hmin);
		max_rung_hydro = std::max(max_rung_hydro, other.max_rung_hydro);
		max_rung_grav = std::max(max_rung_grav, other.max_rung_grav);
		max_rung = std::max(max_rung, other.max_rung);
		max_vsig = std::max(max_vsig, other.max_vsig);
		dtinv_cfl = std::max(dtinv_cfl, other.dtinv_cfl);
		dtinv_visc = std::max(dtinv_visc, other.dtinv_visc);
		dtinv_acc = std::max(dtinv_acc, other.dtinv_acc);
		dtinv_diff = std::max(dtinv_diff, other.dtinv_diff);
		dtinv_cond = std::max(dtinv_cond, other.dtinv_cond);
		dtinv_divv = std::max(dtinv_divv, other.dtinv_divv);
		dtinv_omega = std::max(dtinv_omega, other.dtinv_omega);
		ekin += other.ekin;
		momx += other.momx;
		momy += other.momy;
		momz += other.momz;
		vol += other.vol;
		etherm += other.etherm;
		ent += other.ent;
		rc = rc || other.rc;
		return *this;
	}
};

void sph_particles_energy_to_entropy(float a);

struct sph_run_params {
	int av_type;
	int run_type;
	int set;
	int phase;
	int min_rung;
	float t0;
	float a;
	float adot;
	bool tzero;
	float cfl;
	int max_rung;
	float gy;
	float tau;
	int iter;
	float max_dt;
	float alpha0;
	float alpha1;
	float beta;
	float alpha_decay;
	float omega_m;
	float H0;
	float damping;
	bool diffusion;
	bool conduction;
	double code_to_s;
	double code_to_g;
	double code_to_cm;
	bool stars;
	sph_run_params() {
		iter = 0;
		const auto opts = get_options();
		stars = opts.stars;
		av_type = opts.visc_type;
		code_to_s = opts.code_to_s;
		code_to_g = opts.code_to_g;
		code_to_cm = opts.code_to_cm;
		damping = opts.damping;
		diffusion = opts.diffusion;
		conduction = opts.conduction;
		gy = opts.gy;
		max_dt = 1e30;
		alpha0 = opts.alpha0;
		alpha1 = opts.alpha1;
		beta = opts.beta;
		alpha_decay = opts.alpha_decay;
		if (opts.test == "") {
			H0 = opts.hubble * constants::H0 * opts.code_to_s;
			omega_m = opts.omega_m;
		} else {
			H0 = omega_m = 0.0;
		}
	}
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & omega_m;
		arc & H0;
		arc & adot;
		arc & iter;
		arc & max_dt;
		arc & gy;
		arc & av_type;
		arc & t0;
		arc & phase;
		arc & run_type;
		arc & set;
		arc & min_rung;
		arc & t0;
		arc & a;
		arc & cfl;
		arc & max_rung;
	}
};

#define SPH_RUN_PREHYDRO 1
#define SPH_RUN_HYDRO 2
#define SPH_RUN_AUX 3
#define SPH_RUN_RUNGS 4
#define SPH_RUN_COND_INIT 5
#define SPH_RUN_CONDUCTION 6

float sph_apply_diffusion_update(int minrung, float toler);
void sph_init_diffusion();

struct cond_update_return {
	float err_max;
	float err_rms;
	size_t N;
	template<class A>
	void serialize(A& arc, unsigned) {
		arc & err_max;
		arc & err_rms;
		arc & N;
	}
};
cond_update_return sph_apply_conduction_update(int minrung);
sph_run_return sph_run(sph_run_params params, bool cuda = false);
#ifndef __CUDACC__
hpx::future<sph_tree_neighbor_return> sph_tree_neighbor(sph_tree_neighbor_params params, tree_id self, vector<tree_id> checklist, int level = 0);
#endif
vector<sph_values> sph_values_at(vector<double> x, vector<double> y, vector<double> z);
/*
 template<class T>
 CUDA_EXPORT
 inline T sph_divW(T r, Thinv, Th3inv) {
 const T c0 = T(21.f/M_PI/2.f);
 const T C = c0 * h3inv * sqr(hinv);
 const T q = r * hinv;
 T w = 120.0;
 w = fmaf(q, w, T(120));
 w = fmaf(q, w, -T(300));
 w = fmaf(q, w, T(240));
 w = fmaf(q, w, -T(60));
 w *= C;
 return w;
 }
 */
template<class T>
inline T sph_den(T hinv3) {
	static const T m = get_options().sph_mass;
	static const T N = get_options().neighbor_number;
	static const T c0 = T(3.0 / 4.0 / M_PI) * N;
//	PRINT("%e %e\n", m, N);
	return m * c0 * hinv3;
}
void sph_deposit_sn(float a);

void sph_particles_energy_to_entropy(float a);

#endif /* SPH_HPP_ */
