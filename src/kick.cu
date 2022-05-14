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

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/fixedcapvec.hpp>
#include <cosmictiger/fmm_kernels.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/kick.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/stars.hpp>
#include <cosmictiger/cuda_mem.hpp>

#include <atomic>

/*__managed__ int node_count;
 __managed__ double total_time;
 __managed__ double tree_time;
 __managed__ double gravity_time;
 static __managed__ double kick_time;*/

struct cuda_lists_type {
	stack_vector<int> echecks;
	stack_vector<int> dchecks;
	device_vector<int> leaflist;
	device_vector<int> cplist;
	device_vector<int> cclist;
	device_vector<int> pclist;
	device_vector<expansion<float>> L;
	device_vector<int> nextlist;
	device_vector<kick_return> returns;
	device_vector<array<fixed32, NDIM>> Lpos;
	device_vector<int> phase;
	device_vector<int> self;

};

struct cuda_kick_params {
	array<fixed32, NDIM> Lpos;
	expansion<float> L;
	int self;
	int* dchecks;
	int* echecks;
	int dcount;
	int ecount;
	kick_return* kreturn;
};

__device__ int __noinline__ do_kick(kick_return& return_, kick_params params, const cuda_kick_data& data, const expansion<float>& L, int nactive,
		const tree_node& self, float dm_mass, float sph_mass) {
//	auto tm = clock64();
	const int& tid = threadIdx.x;
	extern __shared__ int shmem_ptr[];
	cuda_kick_shmem& shmem = *(cuda_kick_shmem*) shmem_ptr;
	int flops = 0;
	const bool sph = data.sph;
	const bool vsoft = data.vsoft;
	auto* all_phi = data.pot;
	auto* all_gx = data.gx;
	auto* all_gy = data.gy;
	auto* all_gz = data.gz;
	auto* write_rungs = data.rungs;
	auto* vel_x = data.vx;
	auto* vel_y = data.vy;
	auto* vel_z = data.vz;
#ifndef DM_CON_H_ONLY
	auto* softlens = data.h_snk;
#endif
	auto* sph_index = data.cat_index;
#ifndef DM_CON_H_ONLY
	auto* type = data.type_snk;
#endif
	auto* sph_gx = data.sph_gx;
	auto* sph_gy = data.sph_gy;
	auto* sph_gz = data.sph_gz;
	auto& phi = shmem.phi;
	auto& gx = shmem.gx;
	auto& gy = shmem.gy;
	auto& gz = shmem.gz;
	auto& active_indexes = shmem.active;
	const auto& sink_x = shmem.sink_x;
	const auto& sink_y = shmem.sink_y;
	const auto& sink_z = shmem.sink_z;
	const auto& read_rungs = shmem.rungs;
	const float log2ft0 = log2f(params.t0);
	const float tfactor = params.eta * sqrtf(params.a);
	int max_rung = 0;
	expansion2<float> L2;
	float vx;
	float vy;
	float vz;
	float dt;
	float g2;
	float phi_tot = 0.0f;
	float kin_tot = 0.f;
	float xmom_tot = 0.0f;
	float ymom_tot = 0.0f;
	float zmom_tot = 0.0f;
	float nmom_tot = 0.0f;
	int rung;
	array<float, NDIM> dx;
	part_int snki;
	for (int i = tid; i < nactive; i += WARP_SIZE) {
		snki = active_indexes[i];
		ASSERT(snki >= 0);
		ASSERT(snki < data.sink_size);
		dx[XDIM] = distance(sink_x[i], self.pos[XDIM]); // 1
		dx[YDIM] = distance(sink_y[i], self.pos[YDIM]); // 1
		dx[ZDIM] = distance(sink_z[i], self.pos[ZDIM]); // 1
		flops += 537 + (true) * 178;
		L2 = L2P(L, dx, true);
		int j = NO_INDEX;
		char my_type = DARK_MATTER_TYPE;
#ifndef DM_CON_H_ONLY
		if (sph) {
			j = sph_index[snki];
			my_type = type[snki];
		}
		const float mass = sph ? (my_type == DARK_MATTER_TYPE ? dm_mass : sph_mass) : 1.0f;
#else
		constexpr float mass = 1.f;
#endif
		phi[i] += L2(0, 0, 0);
		gx[i] -= L2(1, 0, 0);
		gy[i] -= L2(0, 1, 0);
		gz[i] -= L2(0, 0, 1);
//		PRINT( "%e %e %e\n", gx[i], gy[i], gz[i]);
		phi[i] *= params.GM;
		gx[i] *= params.GM;
		gy[i] *= params.GM;
		gz[i] *= params.GM;
		if (params.glass) {
			gx[i] *= -1.f;
			gy[i] *= -1.f;
			gz[i] *= -1.f;
		}
		vx = vel_x[snki];
		vy = vel_y[snki];
		vz = vel_z[snki];
		float hsoft = params.h;
#ifndef DM_CON_H_ONLY
		if (vsoft) {
			hsoft = softlens[snki];
		}
#endif
		if (params.htime) {
			ALWAYS_ASSERT(!sph);
			float sgn = params.top ? 1.f : -1.f;
			if (params.ascending ) {
				dt = 0.5f * rung_dt[params.min_rung] * params.t0;
				if (!params.first_call) {
					vx = fmaf(sgn * gx[i], dt, vx);
					vy = fmaf(sgn * gy[i], dt, vy);
					vz = fmaf(sgn * gz[i], dt, vz);
				}
			}
			kin_tot += 0.5f * mass * sqr(vx, vy, vz);
			xmom_tot += mass * vx;
			ymom_tot += mass * vy;
			zmom_tot += mass * vz;
			nmom_tot += mass * sqrtf(sqr(vx, vy, vz));
			if (params.descending ) {
				g2 = sqr(gx[i], gy[i], gz[i]);
				dt = fminf(tfactor * sqrt(hsoft / sqrtf(g2)), params.t0);
				rung = params.min_rung + int((int) ceilf(log2ft0 - log2f(dt)) > params.min_rung);
				if( rung > 4 ) {
//					PRINT( "%i\n", rung);
				}
				max_rung = max(rung, max_rung);
				write_rungs[snki] = rung;
				ALWAYS_ASSERT(rung >= 0);
				ALWAYS_ASSERT(rung < MAX_RUNG);
				dt = 0.5f * rung_dt[params.min_rung] * params.t0;
				ALWAYS_ASSERT(!vsoft);
				vx = fmaf(sgn * gx[i], dt, vx);
				vy = fmaf(sgn * gy[i], dt, vy);
				vz = fmaf(sgn * gz[i], dt, vz);
			}
		} else {
			if (params.save_force || vsoft) {
				all_gx[snki] = gx[i];
				all_gy[snki] = gy[i];
				all_gz[snki] = gz[i];
				all_phi[snki] = phi[i];
			}
			rung = read_rungs[i];
			dt = 0.5f * rung_dt[rung] * params.t0;
			if (my_type == SPH_TYPE && !params.glass) {
				sph_gx[j] = gx[i];
				sph_gy[j] = gy[i];
				sph_gz[j] = gz[i];
			} else {
				if (!params.first_call) {
					vx = fmaf(gx[i], dt, vx);
					vy = fmaf(gy[i], dt, vy);
					vz = fmaf(gz[i], dt, vz);
				}
			}
			g2 = sqr(gx[i], gy[i], gz[i]);
			kin_tot += 0.5f * mass * sqr(vx, vy, vz);
			xmom_tot += mass * vx;
			ymom_tot += mass * vy;
			zmom_tot += mass * vz;
			nmom_tot += mass * sqrtf(sqr(vx, vy, vz));
			if (my_type != SPH_TYPE || params.glass) {
				dt = fminf(fminf(tfactor * sqrt(hsoft / sqrtf(g2+1e-35f)), params.t0), params.max_dt);
				rung = max(max((int) ceilf(log2ft0 - log2f(dt)), max(rung - 1, params.min_rung)), 1);
				max_rung = max(rung, max_rung);
				if (rung < 0 || rung >= MAX_RUNG) {
					PRINT("Rung out of range %i\n", rung);
				}
				ASSERT(rung >= 0);
				ASSERT(rung < MAX_RUNG);
				dt = 0.5f * rung_dt[rung] * params.t0;
				if (!vsoft) {
					vx = fmaf(gx[i], dt, vx);
					vy = fmaf(gy[i], dt, vy);
					vz = fmaf(gz[i], dt, vz);
				}
				write_rungs[snki] = rung;
			}
		}
		vel_x[snki] = vx;
		vel_y[snki] = vy;
		vel_z[snki] = vz;
		phi_tot += 0.5f * mass * phi[i];
		flops += 52;
	}
	shared_reduce_add(phi_tot);
	shared_reduce_add(xmom_tot);
	shared_reduce_add(ymom_tot);
	shared_reduce_add(zmom_tot);
	shared_reduce_add(nmom_tot);
	shared_reduce_add(kin_tot);
	shared_reduce_max(max_rung);
	shared_reduce_add(flops);
	if (tid == 0) {
		return_.max_rung = max(return_.max_rung, max_rung);
		return_.pot += phi_tot;
		return_.xmom += xmom_tot;
		return_.ymom += ymom_tot;
		return_.zmom += zmom_tot;
		return_.nmom += nmom_tot;
		return_.kin += kin_tot;
	}
//	atomicAdd(&kick_time, (double) clock64() - tm);
	return flops;
}

__global__ void cuda_kick_kernel(kick_params global_params, cuda_kick_data data, cuda_lists_type* lists, cuda_kick_params* params, int item_count,
		int* next_item, int ntrees) {
//	auto tm1 = clock64();
	const bool sph = data.sph;
	const bool vsoft = data.vsoft;
	const float dm_mass = !sph ? 1.0f : global_params.dm_mass;
	const float sph_mass = global_params.sph_mass;
	const int& tid = threadIdx.x;
	const int& bid = blockIdx.x;
	extern __shared__ int shmem_ptr[];
	cuda_kick_shmem& shmem = *(cuda_kick_shmem*) shmem_ptr;
	new (&lists[bid]) cuda_lists_type;
	auto& L = lists[bid].L;
	auto& phase = lists[bid].phase;
	auto& Lpos = lists[bid].Lpos;
	auto& self_index = lists[bid].self;
	auto& returns = lists[bid].returns;
	auto& dchecks = lists[bid].dchecks;
	auto& echecks = lists[bid].echecks;
	auto& nextlist = lists[bid].nextlist;
	auto& cclist = lists[bid].cclist;
	auto& cplist = lists[bid].cplist;
	auto& pclist = lists[bid].pclist;
	auto& leaflist = lists[bid].leaflist;
	auto& activei = shmem.active;
	auto& sink_x = shmem.sink_x;
	auto& sink_y = shmem.sink_y;
	auto& sink_z = shmem.sink_z;
#ifndef DM_CON_H_ONLY
	auto& sink_h = shmem.sink_h;
	auto& sink_zeta = shmem.sink_zeta;
	auto& sink_type = shmem.sink_type;
#endif
	auto& rungs = shmem.rungs;
	auto& phi = shmem.phi;
	auto& gx = shmem.gx;
	auto& gy = shmem.gy;
	auto& gz = shmem.gz;
	const float thetainv = 1.f / global_params.theta;
	const int min_rung = global_params.min_rung;
	const float sink_bias = 1.5;
	auto* tree_nodes = data.tree_nodes;
	auto* all_rungs = data.rungs;
	auto* src_x = data.x;
	auto* src_y = data.y;
	auto* src_z = data.z;
#ifndef DM_CON_H_ONLY
	auto* src_h = data.h;
	auto* src_zeta = data.zeta;
	auto* src_type = data.type;
#endif
	int index;
	if (tid == 0) {
		index = atomicAdd(next_item, 1);
	}
	index = __shfl_sync(0xFFFFFFFF, index, 0);
	while (index < item_count) {
		L.resize(0);
		dchecks.resize(0);
		echecks.resize(0);
		phase.resize(0);
		self_index.resize(0);
		Lpos.resize(0);
		returns.push_back(kick_return());
		L.push_back(params[index].L);
		dchecks.resize(params[index].dcount);
		echecks.resize(params[index].ecount);
		for (int i = tid; i < params[index].dcount; i += WARP_SIZE) {
			dchecks[i] = params[index].dchecks[i];
		}
		for (int i = tid; i < params[index].ecount; i += WARP_SIZE) {
			echecks[i] = params[index].echecks[i];
		}
		phase.push_back(0);
		self_index.push_back(params[index].self);
		Lpos.push_back(params[index].Lpos);
		__syncwarp();
		int depth = 0;
		int maxi;
		while (depth >= 0) {
//			auto tm2 = clock64();
//			node_count++;
			ASSERT(Lpos.size() == depth + 1);
			ASSERT(self_index.size() == depth + 1);
			ASSERT(phase.size() == depth + 1);
			const auto& self = tree_nodes[self_index.back()];
			switch (phase.back()) {

			case 0: {
				array<float, NDIM> dx;
				for (int dim = 0; dim < NDIM; dim++) {
					dx[dim] = distance(self.pos[dim], Lpos.back()[dim]);
				}
				auto this_L = L2L_cuda(L.back(), dx, true);
				if (tid == 0) {
					L.back() = this_L;
				}
				int nactive = 0;
				if (self.leaf) {
					const part_int begin = self.sink_part_range.first;
					const part_int end = self.sink_part_range.second;
					const part_int src_begin = self.part_range.first;
					maxi = round_up(end - begin, (part_int) WARP_SIZE);
					for (int i = tid; i < maxi; i += WARP_SIZE) {
						bool active = false;
						char rung;
						if (i < end - begin) {
							ASSERT(begin + i < data.sink_size);
							rung = all_rungs[begin + i];
							active = rung >= min_rung;
						}
						int l;
						int total;
						l = active;
						compute_indices(l, total);
						l += nactive;
						if (active) {
							const part_int srci = src_begin + i;
							ASSERT(begin + i < data.sink_size);
							activei[l] = begin + i;
							rungs[l] = rung;
							sink_x[l] = src_x[srci];
							sink_y[l] = src_y[srci];
							sink_z[l] = src_z[srci];
#ifndef DM_CON_H_ONLY
							if (vsoft) {
								sink_h[l] = src_h[srci];
								sink_zeta[l] = src_zeta[srci];
							} else {
								sink_h[l] = global_params.h;
								sink_zeta[l] = 0.f;
							}
							sink_type[l] = src_type[srci];
#endif
						}
						nactive += total;
						__syncwarp();
					}
					for (int i = tid; i < nactive; i += WARP_SIZE) {
						phi[i] = 0.0;
						gx[i] = gy[i] = gz[i] = 0.f;
					}
					__syncwarp();
				}
				__syncwarp();
				for (int gtype = GRAVITY_DIRECT; gtype <= GRAVITY_EWALD; gtype++) {
					nextlist.resize(0);
					leaflist.resize(0);
					cclist.resize(0);
					cplist.resize(0);
					pclist.resize(0);
					auto& checks = gtype == GRAVITY_DIRECT ? dchecks : echecks;
					float my_hsoft;
					if (vsoft) {
						my_hsoft = self.hsoft_max;
					} else {
						my_hsoft = global_params.h;
					}
					do {
						maxi = round_up(checks.size(), WARP_SIZE);
						for (int i = tid; i < maxi; i += WARP_SIZE) {
							bool cc = false;
							bool next = false;
							bool leaf = false;
							bool cp = false;
							bool pc = false;
							if (i < checks.size()) {
								const tree_node& other = tree_nodes[checks[i]];
								float other_hsoft;
								if (vsoft) {
									other_hsoft = other.hsoft_max;
								} else {
									other_hsoft = global_params.h;
								}
								const float hsoft = fmaxf(other_hsoft, my_hsoft);
								for (int dim = 0; dim < NDIM; dim++) {
									dx[dim] = distance(self.pos[dim], other.pos[dim]); // 3
								}
								float R2 = sqr(dx[XDIM], dx[YDIM], dx[ZDIM]);
								if (gtype == GRAVITY_EWALD) {
									R2 = fmaxf(R2, sqr(fmaxf(0.5f - (self.radius + other.radius),0.f)));
								}
								const float mind = self.radius + other.radius + hsoft;
								const float dcc = fmaxf((self.radius + other.radius) * thetainv, mind);
								const float dcp = fmaxf((self.radius * thetainv + other.radius), mind);
								const float dpc = fmaxf((self.radius + other.radius * thetainv), mind);
								const bool far = R2 > sqr(dcc);
								cc = far;
								if (!cc && other.leaf && self.leaf) {
									pc = R2 > sqr(dpc) && dpc > dcp;
									cp = R2 > sqr(dcp) && dcp > dpc;
								}
								if (!cc && !cp && !pc) {
									leaf = other.leaf;
									next = !leaf;
								}
							}
							int l;
							int total;
							int start;
							l = cc;
							compute_indices(l, total);
							start = cclist.size();
							cclist.resize(start + total);
							if (cc) {
								cclist[l + start] = checks[i];
							}
							l = cp;
							compute_indices(l, total);
							start = cplist.size();
							cplist.resize(start + total);
							if (cp) {
								cplist[l + start] = checks[i];
							}
							l = pc;
							compute_indices(l, total);
							start = pclist.size();
							pclist.resize(start + total);
							if (pc) {
								pclist[l + start] = checks[i];
							}
							l = next;
							compute_indices(l, total);
							start = nextlist.size();
							nextlist.resize(start + total);
							if (next) {
								nextlist[l + start] = checks[i];
							}
							l = leaf;
							compute_indices(l, total);
							start = leaflist.size();
							leaflist.resize(start + total);
							if (leaf) {
								leaflist[l + start] = checks[i];
							}
						}
						__syncwarp();
						checks.resize(NCHILD * nextlist.size());
						for (int i = tid; i < nextlist.size(); i += WARP_SIZE) {
							const auto& node = tree_nodes[nextlist[i]];
							const auto& children = node.children;
							checks[NCHILD * i + LEFT] = children[LEFT].index;
							checks[NCHILD * i + RIGHT] = children[RIGHT].index;
						}
						nextlist.resize(0);
						__syncwarp();

					} while (checks.size() && self.leaf);
					if (gtype == GRAVITY_DIRECT) {
						cuda_gravity_cc_direct(data, L.back(), self, cclist, true);
						cuda_gravity_cp_direct(data, L.back(), self, cplist, dm_mass, sph_mass, true);
					} else {
						cuda_gravity_cc_ewald(data, L.back(), self, cclist, true);
						cuda_gravity_cp_ewald(data, L.back(), self, cplist, dm_mass, sph_mass, true);
					}
					if (self.leaf) {
						__syncwarp();
						const float h = global_params.h;
						if (gtype == GRAVITY_DIRECT) {
							cuda_gravity_pc_direct(data, self, pclist, nactive, true);
							cuda_gravity_pp_direct(data, self, leaflist, nactive, h, dm_mass, sph_mass, true);
						} else {
							cuda_gravity_pc_ewald(data, self, pclist, nactive, true);
							cuda_gravity_pp_ewald(data, self, leaflist, nactive, h, dm_mass, sph_mass, true);
						}
					} else {
						const int start = checks.size();
						checks.resize(start + leaflist.size());
						for (int i = tid; i < leaflist.size(); i += WARP_SIZE) {
							checks[start + i] = leaflist[i];
						}
						__syncwarp();
					}
				}
				if (self.leaf) {
//					atomicAdd(&gravity_time, (double) clock64() - tm);
					__syncwarp();
					do_kick(returns.back(), global_params, data, L.back(), nactive, self, dm_mass, sph_mass);
					phase.pop_back();
					self_index.pop_back();
					Lpos.pop_back();
					depth--;
				} else {
					ALWAYS_ASSERT(self.children[LEFT].index!=-1);
					ALWAYS_ASSERT(self.children[RIGHT].index!=-1);
					const int active_left = tree_nodes[self.children[LEFT].index].nactive;
					const int active_right = tree_nodes[self.children[RIGHT].index].nactive;
					Lpos.push_back(self.pos);
					returns.push_back(kick_return());
					if (active_left && active_right) {
						const tree_id child = self.children[LEFT];
						const int i1 = L.size() - 1;
						const int i2 = L.size();
						L.resize(i2 + 1);
						for (int l = tid; l < EXPANSION_SIZE; l += WARP_SIZE) {
							L[i2][l] = L[i1][l];
						}
						__syncwarp();
						dchecks.push_top();
						echecks.push_top();
						phase.back() += 1;
						phase.push_back(0);
						self_index.push_back(child.index);
					} else {
						ALWAYS_ASSERT(active_left || active_right);
						const tree_id child = active_left ? self.children[LEFT] : self.children[RIGHT];
						phase.back() += 2;
						phase.push_back(0);
						self_index.push_back(child.index);
					}
					depth++;
				}

			}
				break;
			case 1: {
				L.pop_back();
				Lpos.push_back(self.pos);
				dchecks.pop_top();
				echecks.pop_top();
				phase.back() += 1;
				phase.push_back(0);
				const tree_id child = self.children[RIGHT];
				ASSERT(child.proc == data.rank);
				self_index.push_back(child.index);
				const auto this_return = returns.back();
				returns.pop_back();
				if (tid == 0) {
					returns.back() += this_return;
				}
				returns.push_back(kick_return());
				depth++;
			}
				break;
			case 2: {
				self_index.pop_back();
				phase.pop_back();
				Lpos.pop_back();
				const auto this_return = returns.back();
				returns.pop_back();
				if (tid == 0) {
					returns.back() += this_return;
				}
				depth--;
			}
				break;
			}
		}

		if (tid == 0) {
			*(params[index].kreturn) = returns.back();
		}
		if (tid == 0) {
			index = atomicAdd(next_item, 1);
		}
		index = __shfl_sync(0xFFFFFFFF, index, 0);
		returns.pop_back();
		ASSERT(returns.size() == 0);
		ASSERT(L.size() == 1);
		ASSERT(Lpos.size() == 0);
		ASSERT(phase.size() == 0);
		ASSERT(self_index.size() == 0);
	}
	(&lists[bid])->~cuda_lists_type();
//	atomicAdd(&total_time, ((double) (clock64() - tm1)));
}

vector<kick_return> cuda_execute_kicks(kick_params kparams, fixed32* dev_x, fixed32* dev_y, fixed32* dev_z, char* dev_type, float* dev_h, float* dev_zeta,
		tree_node* dev_tree_nodes, vector<kick_workitem> workitems, cudaStream_t stream, int part_count, int ntrees, std::function<void()> acquire_inner,
		std::function<void()> release_outer) {
	static const bool do_sph = get_options().sph;
	timer tm;
//	PRINT("shmem size = %i\n", sizeof(cuda_kick_shmem));
	tm.start();
	int* current_index;
	int zero = 0;
//	kick_time = total_time = tree_time = gravity_time = 0.0f;
//	node_count = 0;
	CUDA_MALLOC(&current_index, sizeof(int));
	CUDA_CHECK(cudaMemcpyAsync(current_index, &zero, sizeof(int), cudaMemcpyHostToDevice, stream));
	vector<kick_return> returns;
	static vector<cuda_kick_params, pinned_allocator<cuda_kick_params>> kick_params;
	static vector<int, pinned_allocator<int>> dchecks;
	static vector<int, pinned_allocator<int>> echecks;
	dchecks.resize(0);
	echecks.resize(0);
	returns.resize(workitems.size());
	kick_params.resize(workitems.size());
	int* dev_dchecks;
	int* dev_echecks;
	kick_return* dev_returns;
	cuda_kick_params* dev_kick_params;
	int nblocks = kick_block_count();
	nblocks = std::min(nblocks, (int) workitems.size());
	cuda_lists_type* dev_lists;
	(CUDA_MALLOC(&dev_lists, sizeof(cuda_lists_type) * nblocks));
	(CUDA_MALLOC(&dev_kick_params, sizeof(cuda_kick_params) * kick_params.size()));
	(CUDA_MALLOC(&dev_returns, sizeof(kick_return) * returns.size()));

	vector<int> dindices(workitems.size() + 1);
	vector<int> eindices(workitems.size() + 1);

	int dcount = 0;
	int ecount = 0;
	for (int i = 0; i < workitems.size(); i++) {
		//	PRINT( "%i\n", workitems[i].echecklifst.size());
		dcount += workitems[i].dchecklist.size();
		ecount += workitems[i].echecklist.size();
	}
	dchecks.reserve(dcount);
	echecks.reserve(ecount);
	dcount = 0;
	ecount = 0;
	for (int i = 0; i < workitems.size(); i++) {
		dindices[i] = dcount;
		eindices[i] = ecount;
		for (int j = 0; j < workitems[i].dchecklist.size(); j++) {
			dchecks.push_back(workitems[i].dchecklist[j].index);
			dcount++;
		}
		for (int j = 0; j < workitems[i].echecklist.size(); j++) {
			echecks.push_back(workitems[i].echecklist[j].index);
			ecount++;
		}
	}
	dindices[workitems.size()] = dcount;
	eindices[workitems.size()] = ecount;
	CUDA_MALLOC(&dev_dchecks, sizeof(int) * dchecks.size());
	CUDA_MALLOC(&dev_echecks, sizeof(int) * echecks.size());
	CUDA_CHECK(cudaMemcpyAsync(dev_dchecks, dchecks.data(), sizeof(int) * dchecks.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(dev_echecks, echecks.data(), sizeof(int) * echecks.size(), cudaMemcpyHostToDevice, stream));
	tm.stop();
	cuda_kick_data data;
	data.source_size = part_count;
	data.tree_size = ntrees;
#ifndef DM_CON_H_ONLY
	data.cat_index_snk = &particles_cat_index(0);
#endif
	data.sink_size = particles_size();
	data.x = dev_x;
	data.y = dev_y;
	data.z = dev_z;
#ifndef DM_CON_H_ONLY
	data.h = dev_h;
	data.divv_snk = &particles_divv(0);
	data.h_snk = &particles_softlen(0);
	data.zeta = dev_zeta;
	data.type = dev_type;
#endif
	data.sph = do_sph;
	data.vsoft = get_options().vsoft;
	data.x_snk = &particles_pos(XDIM, 0);
	data.y_snk = &particles_pos(YDIM, 0);
	data.z_snk = &particles_pos(ZDIM, 0);
	if (do_sph) {
		data.cat_index = &particles_cat_index(0);
#ifndef DM_CON_H_ONLY
		data.type_snk = &particles_type(0);
#endif
		data.sph_gx = &sph_particles_gforce(XDIM, 0);
		data.sph_gy = &sph_particles_gforce(YDIM, 0);
		data.sph_gz = &sph_particles_gforce(ZDIM, 0);
	}
	data.tree_nodes = dev_tree_nodes;
	data.vx = &particles_vel(XDIM, 0);
	data.vy = &particles_vel(YDIM, 0);
	data.vz = &particles_vel(ZDIM, 0);
	data.rungs = &particles_rung(0);
	data.rank = hpx_rank();
	if (kparams.save_force || get_options().vsoft) {
		data.gx = &particles_gforce(XDIM, 0);
		data.gy = &particles_gforce(YDIM, 0);
		data.gz = &particles_gforce(ZDIM, 0);
		data.pot = &particles_pot(0);
	} else {
		data.gx = data.gy = data.gz = data.pot = nullptr;
	}

	for (int i = 0; i < workitems.size(); i++) {
		cuda_kick_params params;
		params.Lpos = workitems[i].pos;
		params.L = workitems[i].L;
		params.self = workitems[i].self.index;
		params.dchecks = dev_dchecks + dindices[i];
		params.echecks = dev_echecks + eindices[i];
		params.dcount = dindices[i + 1] - dindices[i];
		params.ecount = eindices[i + 1] - eindices[i];
		params.kreturn = dev_returns + i;
		kick_params[i] = std::move(params);
	}
	CUDA_CHECK(cudaMemcpyAsync(dev_kick_params, kick_params.data(), sizeof(cuda_kick_params) * kick_params.size(), cudaMemcpyHostToDevice, stream));
	tm.reset();
	tm.start();
	acquire_inner();
	cuda_set_device();
	cuda_stream_synchronize(stream);
	release_outer();
	cuda_set_device();
//	PRINT( "Invoking kernel\n");
	tm.reset();
	tm.start();
	cuda_kick_kernel<<<nblocks, WARP_SIZE, sizeof(cuda_kick_shmem), stream>>>(kparams, data,dev_lists, dev_kick_params, kick_params.size(), current_index, ntrees);
//	PRINT("One done\n");
	CUDA_CHECK(cudaMemcpyAsync(returns.data(), dev_returns, sizeof(kick_return) * returns.size(), cudaMemcpyDeviceToHost, stream));
	cuda_stream_synchronize(stream);
	tm.stop();
	PRINT("GPU took %e with %i blocks\n", tm.read(), nblocks);
//	PRINT( "%i %e\n", nblocks, tm.read());
//	PRINT("%i nodes traversed\n", node_count);
	CUDA_CHECK(cudaFree(dev_dchecks));
	CUDA_CHECK(cudaFree(dev_echecks));
	CUDA_CHECK(cudaFree(dev_returns));
	CUDA_CHECK(cudaFree(dev_lists));
	CUDA_CHECK(cudaFree(dev_kick_params));
	CUDA_CHECK(cudaFree(current_index));
//	PRINT("%i %i %i %i %i %i %i\n", max_depth, max_dchecks, max_echecks, max_nextlist, max_leaflist, max_partlist, max_multlist);
//	PRINT("%i %e %e %e\n", nblocks, tree_time / total_time, gravity_time / total_time,  kick_time / total_time);
	return returns;
}

int kick_block_count() {
	int nblocks;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, (const void*) cuda_kick_kernel, WARP_SIZE, sizeof(cuda_kick_shmem)));
	static bool shown = false;
	if (!shown) {
		PRINT("Occupancy is %i shmem size = %li\n", nblocks, sizeof(cuda_kick_shmem));
		shown = true;
	}
	nblocks *= cuda_smp_count();
	return nblocks;

}

size_t kick_estimate_cuda_mem_usage(double theta, int nparts, int check_count) {
	size_t mem = 0;
	size_t innerblocks = nparts / CUDA_KICK_PARTS_MAX;
	size_t nblocks = std::pow(std::pow(innerblocks, 1.0 / 3.0) + 1 + 1.0 / theta, 3);
	size_t total_parts = CUDA_KICK_PARTS_MAX * nblocks;
	size_t ntrees = 3 * total_parts / BUCKET_SIZE;
	size_t nchecks = 2 * innerblocks * check_count;
	mem += total_parts * NDIM * sizeof(fixed32);
	mem += ntrees * sizeof(tree_node);
	mem += nchecks * sizeof(int);
	mem += sizeof(cuda_lists_type) * kick_block_count();
	mem += sizeof(kick_return) * 2 * innerblocks;
	mem += sizeof(cuda_kick_params) * 2 * innerblocks;
	return mem;
}
