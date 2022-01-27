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

struct sph_tree_neighbor_return {
	fixed32_range inner_box;
	fixed32_range outer_box;
	CUDA_EXPORT
	sph_tree_neighbor_return() {
	}
	CUDA_EXPORT
	sph_tree_neighbor_return& operator+=(const sph_tree_neighbor_return& other) {
		inner_box.accumulate(other.inner_box);
		outer_box.accumulate(other.outer_box);
		return *this;

	}
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & inner_box;
		arc & outer_box;
	}
};

#define SPH_TREE_NEIGHBOR_BOXES 0
#define SPH_TREE_NEIGHBOR_NEIGHBORS 1
#define SPH_SET_ACTIVE 1
#define SPH_SET_SEMIACTIVE 2
#define SPH_SET_ALL 4

struct sph_tree_neighbor_params {
	int run_type;
	float h_wt;
	int min_rung;
	int set;
	template<class T>
	void serialize(T&& arc, unsigned) {
		arc & run_type;
		arc & h_wt;
		arc & min_rung;
	}
};


struct sph_run_return {
	float hmin;
	float hmax;
	int max_rung;
	float max_vsig;
	bool rc;
	sph_run_return() {
		hmax = 0.0;
		hmin = std::numeric_limits<float>::max();
		max_rung = 0;
		max_vsig = 0.0;
		rc = false;
	}
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & hmin;
		arc & hmax;
		arc & max_rung;
		arc & max_vsig;
	}
	sph_run_return& operator+=(const sph_run_return& other) {
		hmax = std::max(hmax, other.hmax);
		hmin = std::min(hmin, other.hmin);
		max_rung = std::max(max_rung, other.max_rung);
		max_vsig = std::max(max_vsig, other.max_vsig);
		return *this;
	}
};

struct sph_run_params {
	int run_type;
	int set;
	int min_rung;
	float t0;
	float a;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & run_type;
		arc & set;
		arc & min_rung;
		arc & t0;
		arc & a;
	}
};

#define SPH_RUN_SMOOTHLEN 0
#define SPH_RUN_MARK_SEMIACTIVE 1
#define SPH_RUN_COURANT 2
#define SPH_RUN_GRAVITY 3
#define SPH_RUN_FVELS 4
#define SPH_RUN_HYDRO 5
#define SPH_RUN_UPDATE 6

sph_run_return sph_run(sph_run_params params);
hpx::future<sph_tree_neighbor_return> sph_tree_neighbor(sph_tree_neighbor_params params, tree_id self, vector<tree_id> checklist, int level=0);


template<class T, int N>
T ipow(T x) {
	if (N == 0) {
		return T(1);
	} else if (N == 1) {
		return x;
	} else if (N == 2) {
		return sqr(x);
	} else {
		constexpr int M = N / 2;
		constexpr int L = N - M;
		return ipow<T, L>(x) * ipow<T, M>(x);
	}
}

template<class T>
inline T sph_W(T r, T hinv, T h3inv) {
	static const T c0 = T(21.0 / M_PI / 2.0);
	const T C = c0 * h3inv;
	const T q = r * hinv;
	const T tmp = T(1) - q;
	return C * sqr(sqr(tmp)) * (T(1) + T(4) * q);
}

template<class T>
inline T sph_dWdr_rinv(T r, T hinv, T h3inv) {
	static const T c0 = T(210.0 / M_PI);
	const T C = c0 * h3inv * hinv;
	const T q = r * hinv;
	const T tmp = T(1) - q;
	return -C * sqr(tmp) * tmp * hinv;
}

template<class T>
inline T sph_dWdh(T r, T hinv, T h3inv) {
	static const T c0 = T(21.0 / M_PI / 2.0);
	const T C = c0 * h3inv;
	const T q = r * hinv;
	const T tmp = T(1) - q;
	return -C * sqr(tmp) * tmp * (T(3) + T(9) * q - T(32) * q);
}


template<class T>
inline T sph_den(T hinv3) {
	static const T m = get_options().sph_mass;
	static const T c0 = T(3.0 / 4.0 / M_PI * SPH_NEIGHBOR_COUNT);
	return m * c0 * hinv3;
}
#endif /* SPH_HPP_ */
