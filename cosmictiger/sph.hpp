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

#include <cosmictiger/cuda_vector.hpp>
#include <cosmictiger/stack_vector.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/range.hpp>

#include <atomic>

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
	}
	CUDA_EXPORT
	sph_tree_neighbor_return& operator+=(const sph_tree_neighbor_return& other) {
		inner_box.accumulate(other.inner_box);
		outer_box.accumulate(other.outer_box);
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
#define SPH_SET_ACTIVE 1
#define SPH_SET_SEMIACTIVE 2
#define SPH_SET_ALL 4

struct sph_tree_neighbor_params {
	int run_type;
	float h_wt;
	int min_rung;
	int set;
	double x;
	double y;
	double z;
	template<class T>
	void serialize(T&& arc, unsigned) {
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
	sph_run_return() {
		ent = 0.0;
		hmax = 0.0;
		hmin = std::numeric_limits<float>::max();
		max_rung_hydro = 0;
		max_rung_grav = 0;
		max_rung = 0;
		max_vsig = 0.0;
		etherm = vol = ekin = momx = momy = momz = 0.0;
		rc = false;
	}
	template<class A>
	void serialize(A&& arc, unsigned) {
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
	}
	sph_run_return& operator+=(const sph_run_return& other) {
		hmax = std::max(hmax, other.hmax);
		hmin = std::min(hmin, other.hmin);
		max_rung_hydro = std::max(max_rung_hydro, other.max_rung_hydro);
		max_rung_grav = std::max(max_rung_grav, other.max_rung_grav);
		max_rung = std::max(max_rung, other.max_rung);
		max_vsig = std::max(max_vsig, other.max_vsig);
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

struct sph_run_params {
	int run_type;
	int set;
	int phase;
	int min_rung;
	float t0;
	float a;
	float cfl;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & phase;
		arc & run_type;
		arc & set;
		arc & min_rung;
		arc & t0;
		arc & a;
		arc & cfl;
	}
};

#define SPH_RUN_SMOOTHLEN 0
#define SPH_RUN_MARK_SEMIACTIVE 1
#define SPH_RUN_COURANT 2
#define SPH_RUN_GRAVITY 3
#define SPH_RUN_HYDRO 5
#define SPH_RUN_UPDATE 6
#define SPH_RUN_RUNGS 7

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

#endif /* SPH_HPP_ */
