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

size_t cpu_gravity_cc(gravity_cc_type type, expansion<float>& L, const vector<tree_id>& list, tree_id self, bool do_phi) {
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
			if (type == GRAVITY_DIRECT) {
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

size_t cpu_gravity_cp(gravity_cc_type gtype, expansion<float>& L, const vector<tree_id>& list, tree_id self, bool do_phi) {
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
			vector<char> type;
			vector<float> masses;
			srcx.resize(nsource);
			srcy.resize(nsource);
			srcz.resize(nsource);
			masses.resize(nsource);
			type.resize(nsource);
			int count = 0;
			for (int i = 0; i < maxi; i++) {
				particles_global_read_pos(tree_ptrs[i]->global_part_range(), srcx.data(), srcy.data(), srcz.data(), type.data(), nullptr, count);
				count += tree_ptrs[i]->nparts();
			}
			if (do_sph) {
				for (int i = 0; i < count; i++) {
					masses[i] = type[i] != DARK_MATTER_TYPE ? sph_mass : dm_mass;
					;
				}
			} else {
				for (int i = 0; i < count; i++) {
					masses[i] = 1.0;
				}
			}
			for (int i = count; i < nsource; i++) {
				masses[i] = 0.0;
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
				if (gtype == GRAVITY_DIRECT) {
					flops += count * greens_function(D, dx);
				} else {
					flops += count * ewald_greens_function(D, dx);
				}
				for (int l = 0; l < EXPANSION_SIZE; l++) {
					L0[l] += mass * D[l];
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

size_t cpu_gravity_pc(gravity_cc_type type, force_vectors& f, int min_rung, tree_id self, const vector<tree_id>& list) {
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
					if (type == GRAVITY_DIRECT) {
						flops += count * greens_function(D, dx);
					} else {
						flops += count * ewald_greens_function(D, dx);
					}
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

#include <fenv.h>

size_t cpu_gravity_pp(gravity_cc_type gtype, force_vectors& f, int min_rung, tree_id self, const vector<tree_id>& list, float hfloat) {
	size_t flops = 0;
	timer tm;
	tm.start();
	constexpr int chunk_size = 32;
	size_t near_count = 0;
	size_t far_count = 0;
	const static bool do_sph = get_options().sph;
	const static float dm_mass = get_options().dm_mass;
	const static float sph_mass = get_options().sph_mass;
	const static bool vsoft = get_options().vsoft;
	if (list.size()) {
		static const simd_float _2float(fixed2float);
		const simd_float one(1.0);
		const simd_float tiny(1.0e-20);
		const tree_node* self_ptr = tree_get_node(self);
		const int nsink = self_ptr->nparts();
		const auto range = self_ptr->part_range;
		for (int li = 0; li < list.size(); li += chunk_size) {
			array<simd_int, NDIM> X;
			array<simd_int, NDIM> Y;
			simd_int src_type;
			simd_int sink_type;
			simd_float sink_hsoft;
			simd_float sink_zeta;
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
			vector<float> zetas;
			vector<float> hs;
			vector<char> type;
			vector<float> masses;
			srcx.resize(nsource);
			srcy.resize(nsource);
			srcz.resize(nsource);
			masses.resize(nsource);
			zetas.resize(nsource);
			type.resize(nsource);
			hs.resize(nsource);
			int count = 0;
			for (int i = 0; i < maxi; i++) {
				particles_global_read_pos(tree_ptrs[i]->global_part_range(), srcx.data(), srcy.data(), srcz.data(), type.data(), vsoft ? zetas.data() : nullptr,
						count);
				if (vsoft) {
					particles_global_read_softlens(tree_ptrs[i]->global_part_range(), hs.data(), count);
				}
				count += tree_ptrs[i]->nparts();
			}
			if (do_sph) {
				for (int i = 0; i < count; i++) {
					masses[i] = type[i] != DARK_MATTER_TYPE ? sph_mass : dm_mass;
					;
				}
			} else {
				for (int i = 0; i < count; i++) {
					masses[i] = 1.0;
				}
			}
			for (int i = count; i < nsource; i++) {
				srcx[i] = 0.f;
				srcy[i] = 0.f;
				srcz[i] = 0.f;
				masses[i] = 0.0f;
				type[i] = 0;
				zetas[i] = 0.0;
				hs[i] = 1.0;
			}
			const simd_float tiny = 1.0e-15;
			feenableexcept (FE_DIVBYZERO);
			feenableexcept (FE_INVALID);
			feenableexcept (FE_OVERFLOW);
			if (gtype == GRAVITY_DIRECT) {
				for (part_int i = range.first; i < range.second; i++) {
					bool active = particles_rung(i) >= min_rung;
					const int this_type = particles_type(i);
					part_int kk;
					if (this_type == SPH_TYPE) {
						kk = particles_cat_index(i);
					}
					if (active) {
						if (this_type == SPH_TYPE) {
							sink_type = SPH_TYPE;
						} else {
							sink_type = DARK_MATTER_TYPE;
						}
						if (vsoft) {
							sink_hsoft = particles_softlen(i);
							sink_zeta = particles_zeta(i);
						} else {
							sink_hsoft = get_options().hsoft;
							sink_zeta = simd_float(0);
						}
						simd_float gx(0.0);
						simd_float gy(0.0);
						simd_float gz(0.0);
						simd_float phi(0.0);
						for (int dim = 0; dim < NDIM; dim++) {
							X[dim] = particles_pos(dim, i).raw();
						}
						//					simd_float self_flags(0);
						for (int j = 0; j < nsource; j += SIMD_FLOAT_SIZE) {
							const int& count = maxi;
							const int k = j / SIMD_FLOAT_SIZE;
							simd_float mass;
							simd_float hsoft;
							simd_float zeta;
							for (int l = 0; l < SIMD_FLOAT_SIZE; l++) {
								Y[XDIM][l] = srcx[j + l].raw();
								Y[YDIM][l] = srcy[j + l].raw();
								Y[ZDIM][l] = srcz[j + l].raw();
								mass[l] = masses[j + l];
								src_type[l] = type[j + l];
								if (vsoft) {
									hsoft[l] = hs[j + l];
									zeta[l] = zetas[j + l];
								} else {
									hsoft[l] = get_options().hsoft;
									zeta[l] = 0.0;
								}
							}
							array<simd_float, NDIM> dx;
							for (int dim = 0; dim < NDIM; dim++) {
								dx[dim] = simd_float(X[dim] - Y[dim]) * _2float;                                 // 3
							}
							simd_float rinv1 = 0.f, rinv3 = 0.f;
							const simd_float r2 = max(sqr(dx[XDIM], dx[YDIM], dx[ZDIM]), tiny);                 // 5
							const simd_float near_flags = r2 < max(sqr(sink_hsoft), sqr(hsoft));
							const simd_int type_i = sink_type;
//							self_flags += simd_float(1) - (r2 > simd_float(0));
							if (active) {
								if (near_flags.sum() == 0) {
									rinv1 = simd_float(1) / sqrt(r2);
									rinv3 = rinv1 * sqr(rinv1);
								} else {
									const auto& h_i = sink_hsoft;
									const auto& h_j = hsoft;
									const auto sw_ij = simd_float(src_type == sink_type);
									const auto zeta_i = sink_zeta * sw_ij;
									const auto zeta_j = zeta * sw_ij;
									const auto hinv_i = simd_float(1) / h_i;
									const auto hinv_j = simd_float(1) / h_j;
									const auto h2inv_i = sqr(hinv_i);
									const auto h2inv_j = sqr(hinv_j);
									const auto h3inv_i = h2inv_i * hinv_i;
									const auto h3inv_j = h2inv_j * hinv_j;
									const auto type_j = src_type;
									const simd_float r = sqrt(r2);                                                    // 4
									rinv1 = simd_float(1) / (r + tiny);                                                    // 5
									const auto q_i = r * hinv_i;
									const auto q_j = r * hinv_j;
									rinv3 = kernelFqinv(q_i) * h3inv_i;
									rinv3 += kernelFqinv(q_j) * h3inv_j;
									rinv3 += (q_i < simd_float(1)) * zeta_i * dkernelW_dq(q_i) * h2inv_i * rinv1;
									rinv3 += (q_j < simd_float(1)) * zeta_j * dkernelW_dq(q_j) * h2inv_j * rinv1;
									rinv3 *= simd_float(0.5);
									if (min_rung == 0) {
										rinv1 = kernelPot(q_i) * hinv_i;
										rinv1 += kernelPot(q_j) * hinv_j;
										rinv1 *= simd_float(0.5);
									}
								}
								rinv3 *= mass;
								rinv1 *= mass;
								gx = fmaf(rinv3, dx[XDIM], gx);																			// 2
								gy = fmaf(rinv3, dx[YDIM], gy);																			// 2
								gz = fmaf(rinv3, dx[ZDIM], gz);																			// 2
								phi -= rinv1;																			// 1
							}
						}
//						PRINT( "%e\n", self_flags.sum());
//						ALWAYS_ASSERT(self_flags.sum() > 0.0);
						const int j = i - range.first;
						if (active) {
							f.gx[j] -= gx.sum();
							f.gy[j] -= gy.sum();
							f.gz[j] -= gz.sum();
							f.phi[j] += phi.sum();
						}
					}
				}
			} else if (gtype == GRAVITY_EWALD) {

				for (part_int i = range.first; i < range.second; i++) {
					if (particles_rung(i) >= min_rung) {
						const int type = particles_type(i);
						if (type == SPH_TYPE) {
							const auto kk = particles_cat_index(i);
							sink_type = SPH_TYPE;
						} else {
							sink_type = DARK_MATTER_TYPE;
						}
						for (int dim = 0; dim < NDIM; dim++) {
							X[dim] = particles_pos(dim, i).raw();
						}
						simd_float fx(0.f);
						simd_float fy(0.f);
						simd_float fz(0.f);
						simd_float pot(0.f);
						for (int j = 0; j < nsource; j += SIMD_FLOAT_SIZE) {
							const int& count = maxi;
							const int k = j / SIMD_FLOAT_SIZE;
							simd_float mass;
							for (int l = 0; l < SIMD_FLOAT_SIZE; l++) {
								Y[XDIM][l] = srcx[j + l].raw();
								Y[YDIM][l] = srcy[j + l].raw();
								Y[ZDIM][l] = srcz[j + l].raw();
								mass[l] = masses[j + l];
							}
							array<simd_float, NDIM> dx0;
							for (int dim = 0; dim < NDIM; dim++) {
								dx0[dim] = simd_float(X[dim] - Y[dim]) * _2float;                                 // 3
							}
							const simd_float R2 = sqr(dx0[XDIM], dx0[YDIM], dx0[ZDIM]);
							const auto m_j = mass;
							const auto r0mask = simd_float(1) - (R2 > 0.0f);
							const auto r1mask = simd_float(1) - r0mask;
							for (int xi = -4; xi <= +4; xi++) {
								for (int yi = -4; yi <= +4; yi++) {
									for (int zi = -4; zi <= +4; zi++) {
										const auto dx = dx0[XDIM] - simd_float(xi);
										const auto dy = dx0[YDIM] - simd_float(yi);
										const auto dz = dx0[ZDIM] - simd_float(zi);
										const auto r2 = sqr(dx, dy, dz);
										const auto mask = r2 < simd_float(2.6f * 2.6f);
										if (mask.sum()) {
											const auto r = sqrt(r2);
											const auto rinv = simd_float(1.f) / (r + r0mask);
											const auto r2inv = rinv * rinv;
											const auto r3inv = r2inv * rinv;
											const auto exp0 = expf(-4.f * r2);
											const auto erfc0 = erfc(2.f * r);
											const auto expfactor = simd_float(4.0f / sqrt(M_PI)) * r * exp0;
											const auto d0 = -erfc0 * rinv;
											const auto d1 = (expfactor + erfc0) * r3inv;
											pot += r1mask * d0;
											fx -= r1mask * mask * dx * d1;
											fy -= r1mask * mask * dy * d1;
											fz -= r1mask * mask * dz * d1;
										}
									}
								}
							}
							pot += simd_float(M_PI / 4.f);
							for (int xi = -2; xi <= +2; xi++) {
								for (int yi = -2; yi <= +2; yi++) {
									for (int zi = -2; zi <= +2; zi++) {
										const auto hx = xi;
										const auto hy = yi;
										const auto hz = zi;
										const auto h2 = simd_float(sqr(hx, hy, hz));
										const auto hmask = (h2 > 0.f) * (h2 < simd_float(8.001f));
										const auto hdotx = dx0[XDIM] * hx + dx0[YDIM] * hy + dx0[ZDIM] * hz;
										const auto omega = simd_float(2.0 * M_PI) * hdotx;
										const auto c = cos(omega);
										const auto s = sin(omega);
										const auto c0 = -hmask / (h2 + (simd_float(1) - hmask)) * expf(simd_float(-M_PI * M_PI * 0.25f) * h2) * simd_float(1.f / M_PI);
										const auto c1 = -hmask * s * 2.0 * M_PI * c0;
										pot += r1mask * c0 * c;
										fx -= r1mask * c1 * hx;
										fy -= r1mask * c1 * hy;
										fz -= r1mask * c1 * hz;
									}
								}
							}
							pot = pot * r1mask + r0mask * simd_float(2.8372975);
							const simd_float r1inv = rsqrt(R2 + r0mask);
							const simd_float r3inv = sqr(r1inv) * r1inv;
							pot += r1inv * r1mask;
							fx += dx0[XDIM] * r3inv * r1mask;
							fy += dx0[YDIM] * r3inv * r1mask;
							fz += dx0[ZDIM] * r3inv * r1mask;
							pot *= m_j;
							fx *= m_j;
							fy *= m_j;
							fz *= m_j;
						}
						const int l = i - range.first;
						f.gx[l] += fx.sum();
						f.gy[l] += fy.sum();
						f.gz[l] += fz.sum();
						f.phi[l] += pot.sum();
					}
				}
			}
		}
	}
	return 24 * far_count + 52 * near_count;
}
