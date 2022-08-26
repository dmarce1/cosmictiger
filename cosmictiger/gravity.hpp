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
/*
 #define SELF_PHI float(-105/32.0)

 template<class T>
 CUDA_EXPORT inline void gsoft(T& f, T& phi, T q2, T h2, T hinv, T h3inv, bool do_phi) {
 q2 *= hinv;
 q2 *= hinv;
 f = T(135.f / 16.0f);
 f = fmaf(f, q2, T(-147.0f / 8.f));
 f = fmaf(f, q2, T(135.0f / 16.0f));
 f *= h3inv;
 if (do_phi) {
 phi = float(-45.0f / 32.0f);
 phi = fmaf(q2, phi, T(147.f / 32.f)); // 2
 phi = fmaf(q2, phi, T(-175.0f / 32.0f)); // 2
 phi = fmaf(q2, phi, T(105.0f / 32.0f)); // 2
 phi *= hinv;                    // 1
 }
 }
 #define SELF_PHI float(-14.0/5.0)

 template<class T>
 CUDA_EXPORT inline void gsoft(T& f, T& phi, T q2, T h2, T hinv, T h3inv, bool do_phi) {
 q2 *= hinv;
 q2 *= hinv;
 T q = sqrt(q2);
 T f1, f2;
 T phi1, phi2;
 const T w1 = (q2 < T(0.25f) * h2);
 const T w2 = T(1) - w1;
 const T qinv = T(1.f) / (q + w1);
 const T q3inv = sqr(qinv) * qinv;
 f = T(-32.0f / 3.0f);
 f = fmaf(f, q, T(192.0f / 5.0f));
 f = fmaf(f, q, T(-48.0f));
 f = fmaf(f, q, T(64.0f / 3.0f));
 f -= T(1.0f / 15.0f) * q3inv;
 f *= h3inv;
 if (do_phi) {
 phi = T(32.0f / 15.0f);
 phi = fmaf(phi, q, T(-48.0f / 5.0f));
 phi = fmaf(phi, q, T(16.0f));
 phi = fmaf(phi, q, T(-32.0f / 3.0f));
 phi *= q;
 phi = fmaf(phi, q, T(16.0f / 5.0f));
 phi -= T(1.0f / 15.0f) * qinv;
 phi *= hinv;
 }
 phi2 = phi;
 f2 = f;
 f = T(32.0f);
 f = fmaf(f, q, T(-192.0f / 5.0f));
 f *= q;
 f = fmaf(f, q, T(32.0f / 3.0f));
 f *= h3inv;
 if (do_phi) {
 phi = T(-32.0f / 5.0f);
 phi = fmaf(phi, q, T(48.0f / 5.0f));
 phi *= q;
 phi = fmaf(phi, q, T(-16.0f / 3.0f));
 phi *= q;
 phi = fmaf(phi, q, T(14.0f / 5.0f));
 phi *= hinv;
 }
 phi1 = phi;
 f1 = f;
 f = w1 * f1 + w2 * f2;
 phi = w1 * phi1 + w2 * phi2;
 }
 */

#define SELF_PHI float(-35.0/16.0)

template<class T>
CUDA_EXPORT inline void gsoft(T& f, T& phi, T q2, T h2, T hinv, T h3inv, bool do_phi) {
	q2 *= sqr(hinv);
	f = T(15.f / 8.0f);
	f = fmaf(f, q2, T(-21.0f / 4.f));
	f = fmaf(f, q2, T(35.0f / 8.0f));
	f *= h3inv;
	if (do_phi) {
		phi = float(-5.0f / 16.0f);
		phi = fmaf(q2, phi, T(21.f / 16.f)); // 2
		phi = fmaf(q2, phi, T(-35.0f / 16.0f)); // 2
		phi = fmaf(q2, phi, T(35.0f / 16.0f)); // 2
		phi *= hinv; // 1
	}
}

