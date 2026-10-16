/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

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

#include <boost/align/aligned_allocator.hpp>

#include <cosmictiger/flops.hpp>
#include <cosmictiger/fmm_kernels.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/safe_io.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/tree.hpp>

void cpu_gravity_cc(gravity_cc_type type, expansion<float>& L, vector<tree_id> const& list, tree_id self, bool do_phi) {
	flop_counter<int> flops = 0;
	if (list.size()) {
		static simd_float const _2float(fixed2float);
		vector<tree_node const*> tree_ptrs(list.size());
		tree_node const* self_ptr = tree_get_node(self);
		int const nsink = self_ptr->nparts();
		int const nsource = round_up((int)list.size(), SIMD_FLOAT_SIZE) / SIMD_FLOAT_SIZE;
		for (int i = 0; i < list.size(); i++) {
			tree_ptrs[i] = tree_get_node(list[i]);
		}
		static thread_local vector<multipole<simd_float>, boost::alignment::aligned_allocator<multipole<simd_float>, SIMD_FLOAT_SIZE * sizeof(float)>> M;
		static thread_local vector<array<simd_int, NDIM>, boost::alignment::aligned_allocator<array<simd_int, NDIM>, SIMD_FLOAT_SIZE * sizeof(float)>> Y;
		M.resize(nsource);
		Y.resize(nsource);
		for (int i = 0; i < tree_ptrs.size(); i++) {
			int const k = i / SIMD_FLOAT_SIZE;
			int const l = i % SIMD_FLOAT_SIZE;
			auto const& m = tree_ptrs[i]->mpos->multi;
			auto const& y = tree_ptrs[i]->mpos->pos;
			for (int j = 0; j < MULTIPOLE_SIZE; j++) {
				M[k][j][l] = m[j];
			}
			for (int j = 0; j < NDIM; j++) {
				Y[k][j][l] = y[j].raw();
			}
		}
		int const lastk = (tree_ptrs.size() - 1) / SIMD_FLOAT_SIZE;
		int const lastl = (tree_ptrs.size() - 1) % SIMD_FLOAT_SIZE;
		for (int i = tree_ptrs.size(); i < nsource * SIMD_FLOAT_SIZE; i++) {
			int const k = i / SIMD_FLOAT_SIZE;
			int const l = i % SIMD_FLOAT_SIZE;
			for (int j = 0; j < MULTIPOLE_SIZE; j++) {
				M[k][j][l] = 0.0;
			}
			for (int j = 0; j < NDIM; j++) {
				Y[k][j][l] = Y[lastk][j][lastl];
			}
		}
		array<simd_int, NDIM> X;
		for (int dim = 0; dim < NDIM; dim++) {
			X[dim] = self_ptr->mpos->pos[dim].raw();
		}
		expansion<simd_float> L0;
		L0 = simd_float(0.0f);
		for (int j = 0; j < nsource; j++) {
			int const count = std::min(SIMD_FLOAT_SIZE, (int)(tree_ptrs.size() - j * SIMD_FLOAT_SIZE));
			array<simd_float, NDIM> dx;
			for (int dim = 0; dim < NDIM; dim++) {
				dx[dim] = simd_float(X[dim] - Y[j][dim]) * _2float;
			}
			flops += 3 * count;
			expansion<simd_float> D;
			if (type == GRAVITY_DIRECT) {
				flops += count * greens_function(D, dx);
			} else {
				flops += count * ewald_greens_function(D, dx);
				//				flops += count * apply_scale_factor(M[j]);
			}
			M2L(L0, M[j], D, do_phi);
		}
		if (type == GRAVITY_EWALD) {
			//			flops += SIMD_FLOAT_SIZE * apply_scale_factor_inv(L0);
		}
		for (int i = 0; i < EXPANSION_SIZE; i++) {
			L[i] += L0[i].sum();
		}
		flops += 7 * EXPANSION_SIZE;
	}
	add_cpu_flops(flops);
}

void cpu_gravity_cp(expansion<float>& L, vector<tree_id> const& list, tree_id self, bool do_phi) {
	constexpr int chunk_size = 32;
	flop_counter<int> flops = 0;
	if (list.size()) {
		static simd_float const _2float(fixed2float);
		simd_float const one(1.0);
		simd_float const tiny(1.0e-20);
		tree_node const* self_ptr = tree_get_node(self);
		int const nsink = self_ptr->nparts();
		for (int li = 0; li < list.size(); li += chunk_size) {
			array<tree_node const*, chunk_size> tree_ptrs;
			int nsource = 0;
			int const maxi = std::min((int)list.size(), li + chunk_size) - li;
			for (int i = 0; i < maxi; i++) {
				tree_ptrs[i] = tree_get_node(list[i + li]);
				nsource += tree_ptrs[i]->nparts();
			}
			nsource = round_up(nsource, SIMD_FLOAT_SIZE);
			vector<fixed32> srcx;
			vector<fixed32> srcy;
			vector<fixed32> srcz;
			vector<float> masses;
			vector<char> types;
			srcx.resize(nsource);
			srcy.resize(nsource);
			srcz.resize(nsource);
			masses.resize(nsource);
			types.resize(nsource);
			int count = 0;
			for (int i = 0; i < maxi; i++) {
				particles_global_read_pos(tree_ptrs[i]->global_part_range(), srcx.data(), srcy.data(), srcz.data(), types.data(), nullptr, count);
				count += tree_ptrs[i]->nparts();
			}
			for (int i = 0; i < count; i++) {
				masses[i] = get_options().sph ? (types[i] == DARK_MATTER_TYPE ? get_options().dm_mass : get_options().sph_mass) : 1.0f;
			}
			for (int i = count; i < nsource; i++) {
				srcx[i] = 0.0;
				srcy[i] = 0.0;
				srcz[i] = 0.0;
				masses[i] = 0.0;
			}
			auto const range = self_ptr->part_range;
			array<simd_int, NDIM> X;
			array<simd_int, NDIM> Y;
			for (int dim = 0; dim < NDIM; dim++) {
				X[dim] = self_ptr->mpos->pos[dim].raw();
			}
			expansion<simd_float> L0;
			L0 = simd_float(0.0f);
			for (int j = 0; j < nsource; j += SIMD_FLOAT_SIZE) {
				int const cnt = std::min(count - j, SIMD_FLOAT_SIZE);
				int const k = j / SIMD_FLOAT_SIZE;
				simd_float mass;
				for (int l = 0; l < SIMD_FLOAT_SIZE; l++) {
					Y[XDIM][l] = srcx[j + l].raw();
					Y[YDIM][l] = srcy[j + l].raw();
					Y[ZDIM][l] = srcz[j + l].raw();
					mass[l] = masses[j + l];
				}
				array<simd_float, NDIM> dx;
				for (int dim = 0; dim < NDIM; dim++) {
					dx[dim] = simd_float(X[dim] - Y[dim]) * _2float;
				}
				flops += cnt * 3;
				expansion<simd_float> D;
				flops += count * greens_function(D, dx);
				for (int l = 0; l < EXPANSION_SIZE; l++) {
					L0[l] += mass * D[l];
				}
				flops += cnt * EXPANSION_SIZE;
			}
			for (int i = 0; i < EXPANSION_SIZE; i++) {
				L[i] += L0[i].sum();
			}
			flops += 7 * EXPANSION_SIZE;
		}
	}
	add_cpu_flops(flops);
}

void cpu_gravity_pc(force_vectors& f, int do_phi, tree_id self, vector<tree_id> const& list) {
	flop_counter<int> flops = 0;
	if (list.size()) {
		static simd_float const _2float(fixed2float);
		vector<tree_node const*> tree_ptrs(list.size());
		tree_node const* self_ptr = tree_get_node(self);
		int const nsink = self_ptr->nparts();
		int const nsource = round_up((int)list.size(), SIMD_FLOAT_SIZE) / SIMD_FLOAT_SIZE;
		for (int i = 0; i < list.size(); i++) {
			tree_ptrs[i] = tree_get_node(list[i]);
		}
		static thread_local vector<multipole<simd_float>, boost::alignment::aligned_allocator<multipole<simd_float>, SIMD_FLOAT_SIZE * sizeof(float)>> M;
		static thread_local vector<array<simd_int, NDIM>, boost::alignment::aligned_allocator<array<simd_int, NDIM>, SIMD_FLOAT_SIZE * sizeof(float)>> Y;
		M.resize(nsource);
		Y.resize(nsource);
		int count = 0;
		for (int i = 0; i < tree_ptrs.size(); i++) {
			int const k = i / SIMD_FLOAT_SIZE;
			int const l = i % SIMD_FLOAT_SIZE;
			auto const& m = tree_ptrs[i]->mpos->multi;
			auto const& y = tree_ptrs[i]->mpos->pos;
			for (int j = 0; j < MULTIPOLE_SIZE; j++) {
				M[k][j][l] = m[j];
			}
			for (int j = 0; j < NDIM; j++) {
				Y[k][j][l] = y[j].raw();
			}
		}
		int const lastk = (tree_ptrs.size() - 1) / SIMD_FLOAT_SIZE;
		int const lastl = (tree_ptrs.size() - 1) % SIMD_FLOAT_SIZE;
		for (int i = tree_ptrs.size(); i < nsource * SIMD_FLOAT_SIZE; i++) {
			int const k = i / SIMD_FLOAT_SIZE;
			int const l = i % SIMD_FLOAT_SIZE;
			for (int j = 0; j < MULTIPOLE_SIZE; j++) {
				M[k][j][l] = 0.0;
			}
			for (int j = 0; j < NDIM; j++) {
				Y[k][j][l] = Y[lastk][j][lastl];
			}
		}
		auto const range = self_ptr->part_range;
		array<simd_int, NDIM> X;
		for (part_int i = range.first; i < range.second; i++) {
			expansion2<simd_float> L;
			L(0, 0, 0) = simd_float(0.0f);
			L(1, 0, 0) = simd_float(0.0f);
			L(0, 1, 0) = simd_float(0.0f);
			L(0, 0, 1) = simd_float(0.0f);
			for (int dim = 0; dim < NDIM; dim++) {
				X[dim] = particles_pos(dim, i).raw();
			}
			for (int j = 0; j < nsource; j++) {
				int const count = std::min(SIMD_FLOAT_SIZE, (int)(tree_ptrs.size() - j * SIMD_FLOAT_SIZE));
				array<simd_float, NDIM> dx;
				for (int dim = 0; dim < NDIM; dim++) {
					dx[dim] = simd_float(X[dim] - Y[j][dim]) * _2float;
				}
				flops += 3 * count;
				expansion<simd_float> D;
				flops += count * greens_function(D, dx);
				flops += count * M2L(L, M[j], D, do_phi);
			}
			int const j = i - range.first;
			f.gx[j] -= SCALE_FACTOR2 * L(1, 0, 0).sum();
			f.gy[j] -= SCALE_FACTOR2 * L(0, 1, 0).sum();
			f.gz[j] -= SCALE_FACTOR2 * L(0, 0, 1).sum();
			f.phi[j] += SCALE_FACTOR1 * L(0, 0, 0).sum();
			flops += 28;
		}
	}
	add_cpu_flops(flops);
}

void cpu_gravity_pp(force_vectors& f, int do_phi, tree_id self, vector<tree_id> const& list, float hfloat) {

	flop_counter<int> flops = 0;
	timer tm;
	tm.start();
	constexpr int chunk_size = 32;
	size_t near_count = 0;
	size_t far_count = 0;
	if (list.size()) {
		simd_float sink_hsoft;
		sink_hsoft = get_options().hsoft;
		simd_float const hinv_i = simd_float(1.f) / sink_hsoft; // 4
		simd_float const h2inv_i = sqr(hinv_i);					// 1
		simd_float const h3inv_i = (hinv_i)*h2inv_i;			// 1
		simd_float const h2 = sqr(sink_hsoft);					// 1;
		flops += 7;
		static simd_float const _2float(fixed2float);
		simd_float const one(1.0);
		simd_float const tiny(1.0e-20);
		tree_node const* self_ptr = tree_get_node(self);
		int const nsink = self_ptr->nparts();
		auto const range = self_ptr->part_range;
		for (int li = 0; li < list.size(); li += chunk_size) {
			array<simd_int, NDIM> X;
			array<simd_int, NDIM> Y;
			simd_int src_type;
			array<tree_node const*, chunk_size> tree_ptrs;
			int nsource = 0;
			int const maxi = std::min((int)list.size(), li + chunk_size) - li;
			for (int i = li; i < li + maxi; i++) {
				tree_ptrs[i - li] = tree_get_node(list[i]);
				nsource += tree_ptrs[i - li]->nparts();
			}
			nsource = round_up(nsource, SIMD_FLOAT_SIZE);
			vector<fixed32> srcx;
			vector<fixed32> srcy;
			vector<fixed32> srcz;
			vector<float> zetas;
			vector<float> hs;
			vector<char> type;
			vector<float> masses;
			vector<char> types;
			srcx.resize(nsource);
			srcy.resize(nsource);
			srcz.resize(nsource);
			masses.resize(nsource);
			types.resize(nsource);
			int count = 0;
			for (int i = 0; i < maxi; i++) {
				particles_global_read_pos(tree_ptrs[i]->global_part_range(), srcx.data(), srcy.data(), srcz.data(), types.data(), nullptr, count);
				count += tree_ptrs[i]->nparts();
			}
			for (int i = 0; i < count; i++) {
				masses[i] = get_options().sph ? (types[i] == DARK_MATTER_TYPE ? get_options().dm_mass : get_options().sph_mass) : 1.0f;
			}
			for (int i = count; i < nsource; i++) {
				srcx[i] = 0.f;
				srcy[i] = 0.f;
				srcz[i] = 0.f;
				masses[i] = 0.0f;
			}
			simd_float const tiny = 1.0e-15;
			//			feenableexcept (FE_DIVBYZERO);
			//			feenableexcept (FE_INVALID);
			//			feenableexcept (FE_OVERFLOW);
			for (part_int i = range.first; i < range.second; i++) {
				simd_float gx(0.0);
				simd_float gy(0.0);
				simd_float gz(0.0);
				simd_float phi(0.0);
				for (int dim = 0; dim < NDIM; dim++) {
					X[dim] = particles_pos(dim, i).raw();
				}
				//					simd_float self_flags(0);
				for (int j = 0; j < nsource; j += SIMD_FLOAT_SIZE) {
					int const& count = maxi;
					int const k = j / SIMD_FLOAT_SIZE;
					simd_float mass;
					for (int l = 0; l < SIMD_FLOAT_SIZE; l++) {
						Y[XDIM][l] = srcx[j + l].raw();
						Y[YDIM][l] = srcy[j + l].raw();
						Y[ZDIM][l] = srcz[j + l].raw();
						mass[l] = masses[j + l];
					}
					array<simd_float, NDIM> dx;
					for (int dim = 0; dim < NDIM; dim++) {
						dx[dim] = simd_float(X[dim] - Y[dim]) * _2float; // 3
					}
					simd_float const r2 = max(sqr(dx[XDIM], dx[YDIM], dx[ZDIM]), tiny); // 6
					simd_float const rinv1_far = simd_float(1) / sqrt(r2);				// 8
					simd_float const rinv3_far = rinv1_far * sqr(rinv1_far);			// 3
					simd_float rinv3_near;												// 3
					simd_float rinv1_near, rinv1(0.0f);
					gsoft(rinv3_near, rinv1_near, r2, hinv_i, h2inv_i, h3inv_i, do_phi);
					simd_float const sw_far = r2 > h2;							  // 1
					simd_float const sw_near = simd_float(1) - sw_far;			  // 1
					simd_float rinv3 = rinv3_near * sw_near + rinv3_far * sw_far; // 3
					if (do_phi) {
						rinv1 = rinv1_near * sw_near + rinv1_far * sw_far; // 3
						flops += 7;
					}
					rinv3 *= mass;					// 1
					rinv1 *= mass;					// 1
					gx = fmaf(rinv3, dx[XDIM], gx); // 2
					gy = fmaf(rinv3, dx[YDIM], gy); // 2
					gz = fmaf(rinv3, dx[ZDIM], gz); // 2
					phi -= rinv1;					// 1
					flops += 38;
				}
				int const j = i - range.first;
				f.gx[j] -= gx.sum();
				f.gy[j] -= gy.sum();
				f.gz[j] -= gz.sum();
				f.phi[j] += phi.sum();
				flops += 28;
			}
		}
	}
	add_cpu_flops(flops);
}
