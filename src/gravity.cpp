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

#include <cosmictiger/fmm_kernels.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/safe_io.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/kernel.hpp>

#include <boost/align/aligned_allocator.hpp>

size_t cpu_gravity_cc(expansion<float>& L, const vector<tree_id>& list, tree_id self, gravity_cc_type type, bool do_phi) {
	size_t flops = 0;
	if (list.size()) {
		static const simd_float _2float(fixed2float);
		vector<const tree_node*> tree_ptrs(list.size());
		const tree_node* self_ptr = tree_get_node(self);
		const int nsink = self_ptr->nparts();
		const int nsource = round_up((int) list.size(), SIMD_FLOAT_SIZE) / SIMD_FLOAT_SIZE;
		for (int i = 0; i < list.size(); i++) {
			tree_ptrs[i] = tree_get_node(list[i]);
		}
		static thread_local vector<multipole<simd_float>, boost::alignment::aligned_allocator<multipole<simd_float>, SIMD_FLOAT_SIZE * sizeof(float)>> M;
		static thread_local vector<array<simd_int, NDIM>, boost::alignment::aligned_allocator<array<simd_int, NDIM>, SIMD_FLOAT_SIZE * sizeof(float)>> Y;
		M.resize(nsource);
		Y.resize(nsource);
		for (int i = 0; i < tree_ptrs.size(); i++) {
			const int k = i / SIMD_FLOAT_SIZE;
			const int l = i % SIMD_FLOAT_SIZE;
			const auto& m = tree_ptrs[i]->multi;
			const auto& y = tree_ptrs[i]->pos;
			for (int j = 0; j < MULTIPOLE_SIZE; j++) {
				M[k][j][l] = m[j];
			}
			for (int j = 0; j < NDIM; j++) {
				Y[k][j][l] = y[j].raw();
			}
		}
		const int lastk = (tree_ptrs.size() - 1) / SIMD_FLOAT_SIZE;
		const int lastl = (tree_ptrs.size() - 1) % SIMD_FLOAT_SIZE;
		for (int i = tree_ptrs.size(); i < nsource * SIMD_FLOAT_SIZE; i++) {
			const int k = i / SIMD_FLOAT_SIZE;
			const int l = i % SIMD_FLOAT_SIZE;
			for (int j = 0; j < MULTIPOLE_SIZE; j++) {
				M[k][j][l] = 0.0;
			}
			for (int j = 0; j < NDIM; j++) {
				Y[k][j][l] = Y[lastk][j][lastl];
			}
		}
		array<simd_int, NDIM> X;
		for (int dim = 0; dim < NDIM; dim++) {
			X[dim] = self_ptr->pos[dim].raw();
		}
		expansion<simd_float> L0;
		L0 = simd_float(0.0f);
		for (int j = 0; j < nsource; j++) {
			const int count = std::min(SIMD_FLOAT_SIZE, (int) (tree_ptrs.size() - j * SIMD_FLOAT_SIZE));
			array<simd_float, NDIM> dx;
			for (int dim = 0; dim < NDIM; dim++) {
				dx[dim] = simd_float(X[dim] - Y[j][dim]) * _2float;
			}
			flops += 3 * count;
			expansion<simd_float> D;
			if (type == GRAVITY_CC_DIRECT) {
				flops += count * greens_function(D, dx);
			} else {
				flops += count * ewald_greens_function(D, dx);
			}
			M2L(L0, M[j], D, do_phi);
		}
		for (int i = 0; i < EXPANSION_SIZE; i++) {
			L[i] += L0[i].sum();
		}
	}
	return flops;
}

size_t cpu_gravity_cp(expansion<float>& L, const vector<tree_id>& list, tree_id self, bool do_phi) {
	constexpr int chunk_size = 32;
	const static bool do_sph = get_options().sph;
	size_t flops = 0;
	const static float dm_mass = get_options().dm_mass;
	const static float sph_mass = get_options().sph_mass;
	if (list.size()) {
		static const simd_float _2float(fixed2float);
		const simd_float one(1.0);
		const simd_float tiny(1.0e-20);
		const tree_node* self_ptr = tree_get_node(self);
		const int nsink = self_ptr->nparts();
		for (int li = 0; li < list.size(); li += chunk_size) {
			array<const tree_node*, chunk_size> tree_ptrs;
			int nsource = 0;
			const int maxi = std::min((int) list.size(), li + chunk_size) - li;
			for (int i = 0; i < maxi; i++) {
				tree_ptrs[i] = tree_get_node(list[i + li]);
				nsource += tree_ptrs[i]->nparts();
			}
			nsource = round_up(nsource, SIMD_FLOAT_SIZE);
			vector<fixed32> srcx;
			vector<fixed32> srcy;
			vector<fixed32> srcz;
			vector<float> fpot;
			vector<float> masks;
			srcx.resize(nsource);
			srcy.resize(nsource);
			srcz.resize(nsource);
			masks.resize(nsource);
			if (do_sph) {
				fpot.resize(nsource);
			}
			int count = 0;
			for (int i = 0; i < maxi; i++) {
				particles_global_read_pos(tree_ptrs[i]->global_part_range(), srcx.data(), srcy.data(), srcz.data(), nullptr, do_sph ? fpot.data() : nullptr, count);
				count += tree_ptrs[i]->nparts();
			}
			if (do_sph) {
				for (int i = 0; i < count; i++) {
					masks[i] = fpot[i] != 0.f ? sph_mass : dm_mass;
					;
				}
			} else {
				for (int i = 0; i < count; i++) {
					masks[i] = 1.0;
				}
			}
			for (int i = count; i < nsource; i++) {
				masks[i] = 0.0;
			}
			const auto range = self_ptr->part_range;
			array<simd_int, NDIM> X;
			array<simd_int, NDIM> Y;
			for (int dim = 0; dim < NDIM; dim++) {
				X[dim] = self_ptr->pos[dim].raw();
			}
			expansion<simd_float> L0;
			L0 = simd_float(0.0f);
			for (int j = 0; j < nsource; j += SIMD_FLOAT_SIZE) {
				const int cnt = std::min(count - j, SIMD_FLOAT_SIZE);
				const int k = j / SIMD_FLOAT_SIZE;
				simd_float mask;
				for (int l = 0; l < SIMD_FLOAT_SIZE; l++) {
					Y[XDIM][l] = srcx[j + l].raw();
					Y[YDIM][l] = srcy[j + l].raw();
					Y[ZDIM][l] = srcz[j + l].raw();
					mask[l] = masks[j + l];
				}
				array<simd_float, NDIM> dx;
				for (int dim = 0; dim < NDIM; dim++) {
					dx[dim] = simd_float(X[dim] - Y[dim]) * _2float;
				}
				flops += cnt * 3;
				expansion<simd_float> D;
				flops += cnt * greens_function(D, dx);
				for (int l = 0; l < EXPANSION_SIZE; l++) {
					L0[l] += mask * D[l];
				}
				flops += cnt * EXPANSION_SIZE;
			}
			for (int i = 0; i < EXPANSION_SIZE; i++) {
				L[i] += L0[i].sum();
			}
		}
	}
	return flops;
}

size_t cpu_gravity_pc(force_vectors& f, int min_rung, tree_id self, const vector<tree_id>& list) {
	size_t flops = 0;
	if (list.size()) {
		static const simd_float _2float(fixed2float);
		vector<const tree_node*> tree_ptrs(list.size());
		const tree_node* self_ptr = tree_get_node(self);
		const int nsink = self_ptr->nparts();
		const int nsource = round_up((int) list.size(), SIMD_FLOAT_SIZE) / SIMD_FLOAT_SIZE;
		for (int i = 0; i < list.size(); i++) {
			tree_ptrs[i] = tree_get_node(list[i]);
		}
		static thread_local vector<multipole<simd_float>, boost::alignment::aligned_allocator<multipole<simd_float>, SIMD_FLOAT_SIZE * sizeof(float)>> M;
		static thread_local vector<array<simd_int, NDIM>, boost::alignment::aligned_allocator<array<simd_int, NDIM>, SIMD_FLOAT_SIZE * sizeof(float)>> Y;
		M.resize(nsource);
		Y.resize(nsource);
		int count = 0;
		for (int i = 0; i < tree_ptrs.size(); i++) {
			const int k = i / SIMD_FLOAT_SIZE;
			const int l = i % SIMD_FLOAT_SIZE;
			const auto& m = tree_ptrs[i]->multi;
			const auto& y = tree_ptrs[i]->pos;
			for (int j = 0; j < MULTIPOLE_SIZE; j++) {
				M[k][j][l] = m[j];
			}
			for (int j = 0; j < NDIM; j++) {
				Y[k][j][l] = y[j].raw();
			}
		}
		const int lastk = (tree_ptrs.size() - 1) / SIMD_FLOAT_SIZE;
		const int lastl = (tree_ptrs.size() - 1) % SIMD_FLOAT_SIZE;
		for (int i = tree_ptrs.size(); i < nsource * SIMD_FLOAT_SIZE; i++) {
			const int k = i / SIMD_FLOAT_SIZE;
			const int l = i % SIMD_FLOAT_SIZE;
			for (int j = 0; j < MULTIPOLE_SIZE; j++) {
				M[k][j][l] = 0.0;
			}
			for (int j = 0; j < NDIM; j++) {
				Y[k][j][l] = Y[lastk][j][lastl];
			}
		}
		const auto range = self_ptr->part_range;
		array<simd_int, NDIM> X;
		for (part_int i = range.first; i < range.second; i++) {
			if (particles_rung(i) >= min_rung) {
				expansion2<simd_float> L;
				L(0, 0, 0) = simd_float(0.0f);
				L(1, 0, 0) = simd_float(0.0f);
				L(0, 1, 0) = simd_float(0.0f);
				L(0, 0, 1) = simd_float(0.0f);
				for (int dim = 0; dim < NDIM; dim++) {
					X[dim] = particles_pos(dim, i).raw();
				}
				for (int j = 0; j < nsource; j++) {
					const int count = std::min(SIMD_FLOAT_SIZE, (int) (tree_ptrs.size() - j * SIMD_FLOAT_SIZE));
					array<simd_float, NDIM> dx;
					for (int dim = 0; dim < NDIM; dim++) {
						dx[dim] = simd_float(X[dim] - Y[j][dim]) * _2float;
					}
					flops += 3 * count;
					expansion<simd_float> D;
					flops += count * greens_function(D, dx);
					flops += count * M2L(L, M[j], D, min_rung == 0);
				}
				const int j = i - range.first;
				f.gx[j] -= L(1, 0, 0).sum();
				f.gy[j] -= L(0, 1, 0).sum();
				f.gz[j] -= L(0, 0, 1).sum();
				f.phi[j] += L(0, 0, 0).sum();
			}
		}
	}
	return flops;
}

size_t cpu_gravity_pp(force_vectors& f, int min_rung, tree_id self, const vector<tree_id>& list, float hfloat) {
	size_t flops = 0;
	timer tm;
	tm.start();
	constexpr int chunk_size = 32;
	size_t near_count = 0;
	size_t far_count = 0;
	const static bool do_sph = get_options().sph;
	const static float dm_mass = get_options().dm_mass;
	const static float sph_mass = get_options().sph_mass;
	if (list.size()) {
		static const simd_float _2float(fixed2float);
		simd_float h = hfloat;
		simd_float h2 = h * h;
		const simd_float one(1.0);
		const simd_float tiny(1.0e-20);
		simd_float hinv = simd_float(1) / h;
		simd_float hinv3 = hinv * hinv * hinv;
		const tree_node* self_ptr = tree_get_node(self);
		const int nsink = self_ptr->nparts();
		for (int li = 0; li < list.size(); li += chunk_size) {
			array<const tree_node*, chunk_size> tree_ptrs;
			int nsource = 0;
			const int maxi = std::min((int) list.size(), li + chunk_size) - li;
			for (int i = li; i < li + maxi; i++) {
				tree_ptrs[i - li] = tree_get_node(list[i]);
				nsource += tree_ptrs[i - li]->nparts();
			}
			nsource = round_up(nsource, SIMD_FLOAT_SIZE);
			vector<fixed32> srcx;
			vector<fixed32> srcy;
			vector<fixed32> srcz;
			vector<float> hsoft;
			vector<float> fpot;
			vector<float> masks;
			srcx.resize(nsource);
			srcy.resize(nsource);
			srcz.resize(nsource);
			masks.resize(nsource);
			if (do_sph) {
				fpot.resize(nsource);
				hsoft.resize(nsource);
			}
			int count = 0;
			for (int i = 0; i < maxi; i++) {
				particles_global_read_pos(tree_ptrs[i]->global_part_range(), srcx.data(), srcy.data(), srcz.data(), hsoft.size() ? hsoft.data() : nullptr,
						do_sph ? fpot.data() : nullptr, count);
				count += tree_ptrs[i]->nparts();
			}
			if (do_sph) {
				for (int i = 0; i < count; i++) {
					masks[i] = fpot[i] != 0.f ? sph_mass : dm_mass;
					;
				}
			} else {
				for (int i = 0; i < count; i++) {
					masks[i] = 1.0;
				}
			}
			for (int i = count; i < nsource; i++) {
				srcx[i] = 0.f;
				srcy[i] = 0.f;
				srcz[i] = 0.f;
				masks[i] = 0.0f;
				hsoft[i] = 1.0f;
				fpot[i] = 0.f;
			}
			const auto range = self_ptr->part_range;
			array<simd_int, NDIM> X;
			array<simd_int, NDIM> Y;
			simd_float src_hsoft;
			simd_float src_fpot;
			simd_float sink_hsoft;
			simd_float sink_fpot;
			simd_float dm_soft = get_options().hsoft;
			const simd_float tiny = 1.0e-15;
			for (part_int i = range.first; i < range.second; i++) {
				if (particles_rung(i) >= min_rung) {
					const int type = particles_type(i);
					if (type == SPH_TYPE) {
						const auto kk = particles_cat_index(i);
						sink_hsoft = sph_particles_smooth_len(kk);
						sink_fpot = sph_particles_fpot(kk);
					} else {
						sink_hsoft = dm_soft;
						sink_fpot = simd_float(0);
					}
					simd_float gx(0.0);
					simd_float gy(0.0);
					simd_float gz(0.0);
					simd_float phi(0.0);
					for (int dim = 0; dim < NDIM; dim++) {
						X[dim] = particles_pos(dim, i).raw();
					}
					for (int j = 0; j < nsource; j += SIMD_FLOAT_SIZE) {
						const int& count = maxi;
						const int k = j / SIMD_FLOAT_SIZE;
						simd_float mask;
						for (int l = 0; l < SIMD_FLOAT_SIZE; l++) {
							Y[XDIM][l] = srcx[j + l].raw();
							Y[YDIM][l] = srcy[j + l].raw();
							Y[ZDIM][l] = srcz[j + l].raw();
							mask[l] = masks[j + l];
							if (do_sph) {
								src_hsoft[l] = hsoft[j + l];
								src_fpot[l] = fpot[j + l];
							}
						}
						array<simd_float, NDIM> dx;
						for (int dim = 0; dim < NDIM; dim++) {
							dx[dim] = simd_float(X[dim] - Y[dim]) * _2float;                                 // 3
						}

						simd_float rinv1 = 0.f, rinv3 = 0.f;
						if (do_sph) {
							const simd_float r2 = max(sqr(dx[XDIM], dx[YDIM], dx[ZDIM]), tiny);                 // 5
							const simd_float h_i = sink_hsoft;
							const simd_float h_j = src_hsoft;
							const auto h2_j = sqr(h_j);
							const auto h2_i = sqr(h_i);
							const simd_float far_flag = r2 > max(h2_i, h2_j);                                                // 1
							if (far_flag.sum() == SIMD_FLOAT_SIZE) {                                            // 7/8
								rinv1 = mask * rsqrt(r2);                                                        // 5
								rinv3 = -rinv1 * rinv1 * rinv1;                                                  // 2
								far_count += count;
							} else {
								const auto fpot_i = sink_fpot;
								const auto fpot_j = src_fpot;
								const auto hinv_i = simd_float(1.f) / h_i;
								const auto hinv_j = simd_float(1.f) / h_j;
								const auto h3inv_i = sqr(hinv_i) * hinv_i;
								const auto h3inv_j = sqr(hinv_j) * hinv_j;
								const simd_float r = sqrt(r2);                                                    // 4
								const simd_float rinv1_far = mask * simd_float(1) / (r + tiny);                            // 5
								const simd_float rinv3_far = rinv1_far * rinv1_far * rinv1_far;                   // 2
								const auto q_j = r * hinv_j;
								const auto q_i = r * hinv_i;
								simd_float rinv3_near = simd_float(0.5f) * (kernelFqinv(q_i) * h3inv_i + kernelFqinv(q_j) * h3inv_j);
								const auto dWdr_i_rinv = dkernelW_dq(q_i) * hinv_i * h3inv_i * rinv1_far;
								const auto dWdr_j_rinv = dkernelW_dq(q_j) * hinv_j * h3inv_j * rinv1_far;
								auto correction = simd_float(0.5f) * (fpot_i * dWdr_i_rinv + fpot_j * dWdr_j_rinv);
								rinv3_near += correction;
								simd_float rinv1_near = simd_float(0);
								if (min_rung == 0) {
									const auto W_i = kernelW(q_i) * h3inv_i;
									const auto W_j = kernelW(q_j) * h3inv_j;
									rinv1_near = simd_float(0.5f) * (kernelPot(q_i) * hinv_i + kernelPot(q_j) * hinv_j);
									correction = simd_float(0.5f) * (fpot_i * W_i + fpot_j * W_j);
									rinv1_near += correction;
								}
								const auto near_flag = (simd_float(1) - far_flag);                                // 1
								rinv1 = (far_flag * rinv1_far + near_flag * rinv1_near) * mask;                      // 4
								rinv3 = -(far_flag * rinv3_far + near_flag * rinv3_near) * mask;                     // 5
								near_count += count;
								flops += 52;
							}
						} else {

							const simd_float r2 = max(sqr(dx[XDIM], dx[YDIM], dx[ZDIM]), tiny);                 // 5
							const simd_float far_flag = (r2 > h2) * (r2 > simd_float(0));                                                // 1
							if (far_flag.sum() == SIMD_FLOAT_SIZE) {                                            // 7/8
								rinv1 = mask * rsqrt(r2);                                                        // 5
								rinv3 = -rinv1 * rinv1 * rinv1;                                                  // 2
								far_count += count;
							} else {
								const simd_float r = sqrt(r2);                                                    // 4
								const simd_float rinv1_far = mask * simd_float(1) / (r + tiny);                            // 5
								const simd_float rinv3_far = rinv1_far * rinv1_far * rinv1_far;                   // 2
								const simd_float r1overh1 = r * hinv;                                             // 1
								const simd_float r2oh2 = r1overh1 * r1overh1;                                     // 1
								simd_float rinv3_near = +15.0f / 8.0f;
								rinv3_near = fmaf(rinv3_near, r2oh2, simd_float(-21.0f / 4.0f));                    // 2
								rinv3_near = fmaf(rinv3_near, r2oh2, simd_float(+35.0f / 8.0f));                    // 2
								rinv3_near *= hinv3;                                                               // 1
								simd_float rinv1_near = -5.0f / 16.0f;
								rinv1_near = fmaf(rinv1_near, r2oh2, simd_float(21.0f / 16.0f));                    // 2
								rinv1_near = fmaf(rinv1_near, r2oh2, simd_float(-35.0f / 16.0f));                   // 2
								rinv1_near = fmaf(rinv1_near, r2oh2, simd_float(35.0f / 16.0f));                    // 2
								rinv1_near *= hinv;                                                                // 1
								const auto near_flag = (simd_float(1) - far_flag);                                // 1
								rinv1 = (far_flag * rinv1_far + near_flag * rinv1_near) * mask;                      // 4
								rinv3 = -(far_flag * rinv3_far + near_flag * rinv3_near) * mask;                     // 5
								near_count += count;
								flops += 52;
							}
						}
						rinv3 *= mask;
						rinv1 *= mask;
						gx = fmaf(rinv3, dx[XDIM], gx);																			// 2
						gy = fmaf(rinv3, dx[YDIM], gy);																			// 2
						gz = fmaf(rinv3, dx[ZDIM], gz);																			// 2
						phi -= rinv1;																									// 1
					}
					const int j = i - range.first;
					f.gx[j] += gx.sum();
					f.gy[j] += gy.sum();
					f.gz[j] += gz.sum();
					f.phi[j] += phi.sum();
				}
			}
		}
	}
	return 24 * far_count + 52 * near_count;
}
