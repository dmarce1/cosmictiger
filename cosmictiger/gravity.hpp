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

#ifndef GRAVITY_HPP_
#define GRAVITY_HPP_

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/cuda_mem.hpp>
#include <cosmictiger/fixed.hpp>
#include <cosmictiger/fixedcapvec.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/kick.hpp>

struct force_vectors {
	vector<float> phi;
	vector<float> gx;
	vector<float> gy;
	vector<float> gz;
	force_vectors() = default;
	force_vectors(int sz) {
		phi.resize(sz);
		gx.resize(sz);
		gy.resize(sz);
		gz.resize(sz);
	}
};

using gravity_cc_type = int;
constexpr gravity_cc_type GRAVITY_DIRECT = 0;
constexpr gravity_cc_type GRAVITY_EWALD = 1;

void cpu_gravity_cc(gravity_cc_type, expansion<float>&, const vector<tree_id>&, tree_id, bool do_phi);
void cpu_gravity_cp(expansion<float>&, const vector<tree_id>&, tree_id, bool do_phi);
void cpu_gravity_pc(force_vectors&, int, tree_id, const vector<tree_id>&);
void cpu_gravity_pp(force_vectors&, int, tree_id, const vector<tree_id>&, float h);

void show_timings();

#ifdef __CUDACC__
__device__
void cuda_gravity_cc_direct( const cuda_kick_data&, expansion<float>&, const tree_node&, const device_vector<int>&, bool do_phi);
__device__
void cuda_gravity_cp_direct( const cuda_kick_data&, expansion<float>&, const tree_node&, const device_vector<int>&, bool do_phi);
__device__
void cuda_gravity_pc_direct( const cuda_kick_data& data, const tree_node&, const device_vector<int>&,bool);
__device__
void cuda_gravity_cc_ewald( const cuda_kick_data&, expansion<float>&, const tree_node&, const device_vector<int>&, bool do_phi);
__device__
void cuda_gravity_pp_direct(const cuda_kick_data& data, const tree_node&, const device_vector<int>&, float h, bool);
#endif
#endif /* GRAVITY_HPP_ */

void reset_gravity_counters();
void get_gravity_counters(double& close, double& direct);
void set_gravity_counter_use(bool code);

#define FERRER_N 3
#define NFERRER 4

template<class T>
CUDA_EXPORT inline void gsoft(T& f, T& phi, T q2, T h2inv, T hinv, T h3inv, bool do_phi) {
	constexpr float Af[NFERRER][NFERRER] = { { 1.0 }, { 2.5, -1.5 }, { 35.0 / 8.0, -21.0 / 4.0, 15.0 / 8.0 }, { 105.0 / 16.0, -189.0 / 16.0, 135.0 / 16.0, -35.0
			/ 16.0 } };
	constexpr float Aphi[NFERRER][NFERRER + 1] = { { 1.5, -0.5 }, { 15.0 / 8.0, -5.0 / 4.0, 3.0 / 8.0 }, { 35.0 / 16.0, -35.0 / 16.0, 21.0 / 16.0, -5.0 / 16.0 },
			{ 315.0 / 128.0, -105.0 / 32.0, 189.0 / 64.0, -45.0 / 32.0, 35.0 / 128.0 } };
	q2 *= h2inv;
	f = Af[FERRER_N][FERRER_N];
	for (int n = FERRER_N - 1; n >= 0; n--) {
		f = fmaf(f, q2, Af[FERRER_N][n]);
	}
	f *= h3inv;
	if (do_phi) {
		phi = Aphi[FERRER_N][FERRER_N + 1];
		for (int n = FERRER_N; n >= 0; n--) {
			phi = fmaf(phi, q2, Aphi[FERRER_N][n]);
		}
		phi *= hinv; // 1
	}
}

CUDA_EXPORT inline float self_phi() {
	float f, phi;
	gsoft(f, phi, 0.0f, 1.f, 1.f, 1.f, true);
	return phi;
}
