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
#include <cosmictiger/timer.hpp>
#include <cosmictiger/cuda_mem.hpp>
#include <cosmictiger/flops.hpp>
#ifdef TREEPM
#include <cosmictiger/treepm.hpp>
#endif

#define MIN_PARTS_PCCP 38
#define MIN_PARTS2_CC 78

#include <atomic>

/*__managed__ int node_count;
 __managed__ double total_time;
 __managed__ double tree_time;
 __managed__ double gravity_time;
 static __managed__ double kick_time;*/

struct cuda_kick_params {
	array<fixed32, NDIM> Lpos;
	expansion<float> L;
	int self;
	int* dchecks;
	int* echecks;
	int dcount;
	int ecount;
};

__device__ void do_kick(kick_return& return_, kick_params params, const cuda_kick_data& data, const expansion<float>& L, const tree_node& self) {
//	auto tm = clock64();
	const int& tid = threadIdx.x;
	extern __shared__ int shmem_ptr[];
	cuda_kick_shmem& shmem = *(cuda_kick_shmem*) shmem_ptr;
	auto& all_phi = data.pot;
	auto& all_gx = data.gx;
	auto& all_gy = data.gy;
	auto& all_gz = data.gz;
	auto* __restrict__ rungs = data.rungs;
	auto* __restrict__ vels = data.vel;
	auto& force = shmem.f;
	const auto* sink_x = data.x + self.part_range.first;
	const auto* sink_y = data.y + self.part_range.first;
	const auto* sink_z = data.z + self.part_range.first;
	const int nsink = self.part_range.second - self.part_range.first;
	int max_rung = 0;
	float phi_tot = 0.0f;
	float kin_tot = 0.f;
	float xmom_tot = 0.0f;
	float ymom_tot = 0.0f;
	float zmom_tot = 0.0f;
	float nmom_tot = 0.0f;
	float dkin_tot = 0.f;
	const float& hsoft = params.h;
	flop_counter<int> flops = 0;
	for (int i = tid; i < nsink; i += WARP_SIZE) {
		expansion2<float> L2;
		array<float, NDIM> dx;
		part_int snki;
		float vx;
		float vy;
		float vz;
		float dt;
		float g2;
		int rung;
		snki = self.part_range.first + i;
		ASSERT(snki >= 0);
		dx[XDIM] = distance(sink_x[i], self.pos[XDIM]); // 1
		dx[YDIM] = distance(sink_y[i], self.pos[YDIM]); // 1
		dx[ZDIM] = distance(sink_z[i], self.pos[ZDIM]); // 1
		L2 = L2P(L, dx, params.do_phi);
		auto& F = force[i];
#ifndef TREEPM
		F.phi += SCALE_FACTOR1 * L2(0, 0, 0);
		F.gx -= SCALE_FACTOR2 * L2(1, 0, 0);
		F.gy -= SCALE_FACTOR2 * L2(0, 1, 0);
		F.gz -= SCALE_FACTOR2 * L2(0, 0, 1);
#else
		F.phi += L2(0, 0, 0);
		F.gx -= L2(1, 0, 0);
		F.gy -= L2(0, 1, 0);
		F.gz -= L2(0, 0, 1);
		const float x = sink_x[i].to_float();
		const float y = sink_y[i].to_float();
		const float z = sink_z[i].to_float();
		if( params.do_phi) {
			F.phi += treepm_get_field(NDIM, x, y, z);
		}
		F.gx -= treepm_get_field(XDIM, x, y, z);
		F.gy -= treepm_get_field(YDIM, x, y, z);
		F.gz -= treepm_get_field(ZDIM, x, y, z);
#endif
		F.gz *= params.GM;
		F.gy *= params.GM;
		F.gx *= params.GM;
		F.phi *= params.GM;
		vx = vels[snki][XDIM];
		vy = vels[snki][YDIM];
		vz = vels[snki][ZDIM];
		float kin0 = 0.5 * sqr(vx, vy, vz);
		const float sgn = params.sign * (params.top ? 1.f : -1.f);
		if (params.ascending) {
			dt = 0.5f * rung_dt[params.min_rung] * params.t0;
			flops += 2;
			if (!params.first_call) {
				vx = fmaf(sgn * F.gx, dt, vx);
				vy = fmaf(sgn * F.gy, dt, vy);
				vz = fmaf(sgn * F.gz, dt, vz);
				flops += 6;
			}
		}
		kin_tot += 0.5f * sqr(vx, vy, vz);
		xmom_tot += vx;
		ymom_tot += vy;
		zmom_tot += vz;
		nmom_tot += sqrtf(sqr(vx, vy, vz));
		if (params.save_force) {
			all_gx[snki] = F.gx;
			all_gy[snki] = F.gy;
			all_gz[snki] = F.gz;
			all_phi[snki] = F.phi;
		}

		if (params.descending) {
			g2 = sqr(F.gx, F.gy, F.gz);
			dt = fminf(params.eta * sqrt(params.a * hsoft * rsqrtf(g2)), params.t0); // 12
			dt = fminf(params.max_dt, dt);
			rung = rungs[snki];
			rung = max(params.min_rung + int((int) ceilf(log2f(params.t0 / dt)) > params.min_rung), rung - 1); // 13
			rungs[snki] = rung;
			max_rung = max(rung, max_rung);
			ALWAYS_ASSERT(rung >= 0);
			ALWAYS_ASSERT(rung < MAX_RUNG);
			dt = 0.5f * rung_dt[params.min_rung] * params.t0; // 2
			vx = fmaf(sgn * F.gx, dt, vx); // 3
			vy = fmaf(sgn * F.gy, dt, vy); // 3
			vz = fmaf(sgn * F.gz, dt, vz); // 3
			flops += 36;
		}
		float kin1 = 0.5 * sqr(vx, vy, vz);
		vels[snki][XDIM] = vx;
		vels[snki][YDIM] = vy;
		vels[snki][ZDIM] = vz;
		phi_tot += 0.5f * force[i].phi;
		dkin_tot += kin1 - kin0;
		flops += 557 + params.do_phi * 178;

	}
	shared_reduce_add(phi_tot);
	shared_reduce_add(xmom_tot);
	shared_reduce_add(ymom_tot);
	shared_reduce_add(zmom_tot);
	shared_reduce_add(nmom_tot);
	shared_reduce_add(kin_tot);
	shared_reduce_add(dkin_tot);
	flops += 30;
	shared_reduce_max(max_rung);
	if (tid == 0) {
		return_.max_rung = max(return_.max_rung, max_rung);
		return_.pot += phi_tot;
		return_.xmom += xmom_tot;
		return_.ymom += ymom_tot;
		return_.zmom += zmom_tot;
		return_.nmom += nmom_tot;
		return_.kin += kin_tot;
		return_.dkin += dkin_tot;
		flops += 6;
	}
	add_gpu_flops(flops);
}

__global__ void cuda_kick_kernel(kick_return* rc, kick_params global_params, cuda_kick_data data, cuda_kick_params* params, int item_count, int* next_item) {
//	auto tm1 = clock64();
	const int& tid = threadIdx.x;
	extern __shared__ int shmem_ptr[];
	cuda_kick_shmem& shmem = *(cuda_kick_shmem*) shmem_ptr;
	new (shmem_ptr) cuda_kick_shmem;
	auto& sparams = shmem.params;
	auto& L = shmem.L;
	auto& dchecks = shmem.dchecks;
	auto& echecks = shmem.echecks;
	auto& nextlist = shmem.nextlist;
	auto& cclist = shmem.cclist;
	auto& cplist = shmem.cplist;
	auto& pclist = shmem.pclist;
	auto& leaflist = shmem.leaflist;
	auto& barrier = shmem.barrier;
	auto& force = shmem.f;
	auto& tree_nodes = data.tree_nodes;
	const float& h = global_params.h;
	const float hinv = 1.f / h;
	int index;
	auto group = cooperative_groups::this_thread_block();
	if (group.thread_rank() == 0) {
		init(&barrier, group.size());
	}
	group.sync();
	if (tid == 0) {
		index = atomicAdd(next_item, 1);
	}
	index = __shfl_sync(0xFFFFFFFF, index, 0);
	__syncthreads();
	kick_return kr;
	while (index < item_count) {
		flop_counter<int> flops = 0;
		L.resize(0);
		dchecks.resize(0);
		echecks.resize(0);
		sparams.resize(0);
		{
			expansion_type this_L;
			this_L.pos = params[index].Lpos;
			this_L.expansion = params[index].L;
			L.push_back(this_L);
		}
		dchecks.resize(params[index].dcount);
		echecks.resize(params[index].ecount);
		for (int i = tid; i < params[index].dcount; i += WARP_SIZE) {
			dchecks[i] = params[index].dchecks[i];
		}
		for (int i = tid; i < params[index].ecount; i += WARP_SIZE) {
			echecks[i] = params[index].echecks[i];
		}
		{
			search_params sparam;
			sparam.phase = 0;
			sparam.self = params[index].self;
			sparams.push_back(sparam);
		}
		__syncwarp();
		int depth = 0;
		while (depth >= 0) {
//			auto tm2 = clock64();
//			node_count++;
			const auto& self = tree_nodes[sparams.back().self];
			switch (sparams.back().phase) {

			case 0: {
				array<float, NDIM> dx;
				for (int dim = 0; dim < NDIM; dim++) {
					dx[dim] = distance(self.pos[dim], L.back().pos[dim]);
				}
				flops += 3;
				{
					const auto this_L = L2L_cuda(L.back().expansion, dx, global_params.do_phi);
					if (tid == 0) {
						L.back().expansion = this_L;
						L.back().pos = self.pos;
					}
					flops += 2650 + global_params.do_phi * 332;
				}
				const part_int nsinks = self.part_range.second - self.part_range.first;
				if (self.leaf) {
					force.resize(nsinks);
					for (int l = tid; l < nsinks; l += WARP_SIZE) {
						auto& f = force[l];
						f.gx = f.gy = f.gz = 0.f;
						f.phi = self_phi() * hinv;
					}
				}
				__syncwarp();
				if (nsinks) {
#ifndef TREEPM
					constexpr int EWALD = 0;
					constexpr int DIRECT = 1;
					for (int type = 0; type < 2; type++) {
#endif
						nextlist.resize(0);
						leaflist.resize(0);
						cclist.resize(0);
						cplist.resize(0);
						pclist.resize(0);
						constexpr int CC = 0;
						constexpr int LEAF = 1;
						constexpr int CP = 2;
						constexpr int PC = 3;
						constexpr int NEXT = 4;
						constexpr int NLISTS = 5;
						array<bool, NLISTS> sw;
						device_vector<int>* lists[NLISTS] = { &cclist, &leaflist, &cplist, &pclist, &nextlist };
#ifdef TREEPM
						auto& checks = dchecks;
#else
						auto& checks = type == DIRECT ? dchecks : echecks;
#endif
						const float thetainv = 1.f / global_params.theta;
						do {
							const int maxi = round_up(checks.size(), WARP_SIZE);
							for (int i = tid; i < maxi; i += WARP_SIZE) {
								for (int n = 0; n < NLISTS; n++) {
									sw[n] = false;
								}
								if (i < checks.size()) {
									const tree_node& other = tree_nodes[checks[i]];
									if (other.nparts()) {
										for (int dim = 0; dim < NDIM; dim++) {
											dx[dim] = distance(self.pos[dim], other.pos[dim]); // 3
										}
										float R2 = sqr(dx[XDIM], dx[YDIM], dx[ZDIM]);
#ifndef TREEPM
										if (type == EWALD) {
											R2 = fmaxf(R2, sqr(fmaxf(0.5f - other.radius - self.radius, 0.f)));
										}
#endif
										const float mind = self.radius + other.radius + h;
										const float dcc = fmaxf((self.radius + other.radius) * thetainv, mind);
										const auto self_parts = self.nparts();
										const auto other_parts = other.nparts();
										sw[CC] = (R2 > sqr(dcc)); // && min(self_parts, (part_int) (2*MIN_PARTS2_CC)) * min(other_parts, (part_int) (2*MIN_PARTS2_CC)) >= MIN_PARTS2_CC;
										flops += 20;
										if (!sw[CC] && other.leaf && self.leaf) {
											const float dcp = fmaxf((self.radius * thetainv + other.radius), mind);
											const float dpc = fmaxf((self.radius + other.radius * thetainv), mind);
											sw[PC] = (R2 > sqr(dpc) || !box_intersects_sphere(self.box, other.pos, fmaxf(thetainv * other.radius, h)))
													&& self_parts >= MIN_PARTS_PCCP;
											sw[CP] = (R2 > sqr(dcp) || !box_intersects_sphere(other.box, self.pos, fmaxf(thetainv * self.radius, h)))
													&& other_parts >= MIN_PARTS_PCCP;
											if (sw[PC] && sw[CP]) {
												if (self_parts < other_parts) {
													sw[CP] = false;
												} else if (self_parts > other_parts) {
													sw[PC] = false;
												} else {
													sw[CP] = sw[PC] = false;
												}
											}
											flops += 33;
										}
										if (!sw[CC] && !sw[CP] && !sw[PC]) {
											sw[LEAF] = other.leaf;
											sw[NEXT] = !sw[LEAF];
										}
									}
								}
								{
									array<int, NLISTS> l, total;
									for (int n = 0; n < NLISTS; n++) {
										l[n] = sw[n];
									}
									compute_indices_array(l, total);
									for (int n = 0; n < NLISTS; n++) {
										const int start = lists[n]->size();
										__syncwarp();
										lists[n]->resize(start + total[n]);
										if (sw[n]) {
											(*lists[n])[l[n] + start] = checks[i];
										}
									}
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
#ifndef TREEPM
						if (type == EWALD) {
							if (self.leaf) {
								__syncwarp();
								const float h = global_params.h;
								cuda_gravity_cc_ewald(data, L.back().expansion, self, cclist, global_params.do_phi);
								cuda_gravity_cp_ewald(data, L.back().expansion, self, cplist, global_params.do_phi);
								cuda_gravity_pc_ewald(data, self, pclist, global_params.do_phi);
								cuda_gravity_pp_ewald(data, self, leaflist, h, global_params.do_phi);
							} else {
								const int start = checks.size();
								checks.resize(start + leaflist.size());
								cuda::memcpy_async(group, checks.data() + start, leaflist.data(), leaflist.size() * sizeof(int), barrier);
								cuda_gravity_cc_ewald(data, L.back().expansion, self, cclist, global_params.do_phi);
								barrier.arrive_and_wait();
							}
						} else {
#endif
							if (self.leaf) {
								__syncwarp();
								const float h = global_params.h;
								cuda_gravity_cc_direct(data, L.back().expansion, self, cclist, global_params.do_phi);
								cuda_gravity_cp_direct(data, L.back().expansion, self, cplist, global_params.do_phi);
								cuda_gravity_pc_direct(data, self, pclist, global_params.do_phi);
								cuda_gravity_pp_direct(data, self, leaflist, h, global_params.do_phi);
							} else {
								const int start = checks.size();
								checks.resize(start + leaflist.size());
								cuda::memcpy_async(group, checks.data() + start, leaflist.data(), leaflist.size() * sizeof(int), barrier);
								cuda_gravity_cc_direct(data, L.back().expansion, self, cclist, global_params.do_phi);
								barrier.arrive_and_wait();
							}
#ifndef TREEPM
						}
					}
#endif
				}
				if (self.leaf) {
					__syncwarp();
					if (nsinks) {
						do_kick(kr, global_params, data, L.back().expansion, self);
					}
					sparams.pop_back();
					depth--;
				} else {
					const tree_id child = self.children[LEFT];
					const int i1 = L.size() - 1;
					const int i2 = L.size();
					L.resize(i2 + 1);
					cuda::memcpy_async(group, &L[i2].expansion, &L[i1].expansion, sizeof(float) * EXPANSION_SIZE, barrier);
					if (tid < NDIM) {
						L[i2].pos[tid] = L[i1].pos[tid];
					}
					__syncwarp();
					dchecks.push_top();
					echecks.push_top();
					sparams.back().phase += 1;
					{
						search_params sparam;
						sparam.phase = 0;
						sparam.self = child.index;
						sparams.push_back(sparam);
					}
					depth++;
					barrier.arrive_and_wait();
				}

			}
				break;
			case 1: {
				L.pop_back();
				dchecks.pop_top();
				echecks.pop_top();
				sparams.back().phase += 1;
				{
					search_params sparam;
					sparam.phase = 0;
					sparam.self = self.children[RIGHT].index;
					sparams.push_back(sparam);
				}
				depth++;
			}
				break;
			case 2: {
				sparams.pop_back();
				depth--;
			}
				break;
			}
		}

		if (tid == 0) {
			index = atomicAdd(next_item, 1);
		}
		index = __shfl_sync(0xFFFFFFFF, index, 0);
		ASSERT(L.size() == 1);
		add_gpu_flops(flops);
	}
	if (tid == 0) {
		atomicAdd(&rc->dkin, kr.dkin);
		atomicAdd(&rc->kin, kr.kin);
		atomicAdd(&rc->pot, kr.pot);
		atomicAdd(&rc->xmom, kr.xmom);
		atomicAdd(&rc->ymom, kr.ymom);
		atomicAdd(&rc->zmom, kr.zmom);
		atomicAdd(&rc->nmom, kr.nmom);
		atomicMax(&rc->max_rung, kr.max_rung);
	}
	((cuda_kick_shmem*) shmem_ptr)->~cuda_kick_shmem();

//	atomicAdd(&total_time, ((double) (clock64() - tm1)));
}

kick_return cuda_execute_kicks(kick_params kparams, fixed32* dev_x, fixed32* dev_y, fixed32* dev_z, tree_node* dev_tree_nodes,
		vector<kick_workitem> workitems) {
	timer tm;
	tm.start();
	int nblocks = kick_block_count();
	ALWAYS_ASSERT(workitems.size());
	nblocks = std::min(nblocks, (int) workitems.size());
	static char* data_ptr = nullptr;
	static char* dev_data_ptr = nullptr;
	static int data_size = 0;
	int dcount = 0;
	int ecount = 0;
	for (int i = 0; i < workitems.size(); i++) {
		dcount += workitems[i].dchecklist.size();
		ecount += workitems[i].echecklist.size();
	}
	const int alloc_size = sizeof(int) + sizeof(cuda_kick_params) * workitems.size() + sizeof(kick_return) + sizeof(int) * dcount + sizeof(int) * ecount;
	if (data_size < alloc_size) {
		if (data_ptr) {
			CUDA_CHECK(cudaFree(dev_data_ptr));
			free(data_ptr);
		}
		data_size = alloc_size;
		data_ptr = (char*) malloc(alloc_size);
		CUDA_CHECK(cudaMalloc(&dev_data_ptr, alloc_size));
	}
	int offset = 0;
	kick_return* return_ = (kick_return*) (data_ptr + offset);
	offset += sizeof(kick_return);
	cuda_kick_params* ikick_params = (cuda_kick_params*) (data_ptr + offset);
	offset += sizeof(cuda_kick_params) * workitems.size();
	int* dchecks = (int*) (data_ptr + offset);
	offset += sizeof(int) * dcount;
	int* echecks = (int*) (data_ptr + offset);
	offset += sizeof(int) * ecount;
	int* current_index = (int*) (data_ptr + offset);
	offset += sizeof(int);
	offset = 0;
	kick_return* dev_return_ = (kick_return*) (dev_data_ptr + offset);
	offset += sizeof(kick_return);
	cuda_kick_params* dev_ikick_params = (cuda_kick_params*) (dev_data_ptr + offset);
	offset += sizeof(cuda_kick_params) * workitems.size();
	int* dev_dchecks = (int*) (dev_data_ptr + offset);
	offset += sizeof(int) * dcount;
	int* dev_echecks = (int*) (dev_data_ptr + offset);
	offset += sizeof(int) * ecount;
	int* dev_current_index = (int*) (dev_data_ptr + offset);
	offset += sizeof(int);
	*current_index = 0;
	*return_ = kick_return();
	vector<int> dindices(workitems.size() + 1);
	vector<int> eindices(workitems.size() + 1);
	dcount = 0;
	ecount = 0;
	for (int i = 0; i < workitems.size(); i++) {
		dindices[i] = dcount;
		eindices[i] = ecount;
		for (int j = 0; j < workitems[i].dchecklist.size(); j++) {
			dchecks[dcount] = workitems[i].dchecklist[j].index;
			dcount++;
		}
		for (int j = 0; j < workitems[i].echecklist.size(); j++) {
			echecks[ecount] = workitems[i].echecklist[j].index;
			ecount++;
		}
	}

	dindices[workitems.size()] = dcount;
	eindices[workitems.size()] = ecount;
	cuda_kick_data data;
	data.x = dev_x;
	data.y = dev_y;
	data.z = dev_z;
	data.tree_nodes = dev_tree_nodes;
	data.vel = particles_vel_data();
	data.rungs = &particles_rung(0);
	if (kparams.save_force) {
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
		ikick_params[i] = std::move(params);
	}
	cuda_set_device();
#ifdef TREEPM
	data.rs = kparams.rs;
#endif
	CUDA_CHECK(cudaMemcpyAsync(dev_data_ptr, data_ptr, sizeof(char) * alloc_size, cudaMemcpyHostToDevice, 0));
	cuda_kick_kernel<<<nblocks, WARP_SIZE, sizeof(cuda_kick_shmem)>>>(dev_return_, kparams, data, dev_ikick_params, workitems.size(), dev_current_index);
	CUDA_CHECK(cudaMemcpyAsync(return_, dev_return_, sizeof(kick_return), cudaMemcpyDeviceToHost, 0));
	CUDA_CHECK(cudaDeviceSynchronize());
	return *return_;
}

int kick_block_count() {
	static int nblocks;
	static bool shown = false;
	if (!shown) {
		cudaFuncAttributes attr;
		CUDA_CHECK(cudaFuncGetAttributes(&attr, (const void*) cuda_kick_kernel));
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, (const void*) cuda_kick_kernel, WARP_SIZE, sizeof(cuda_kick_shmem)));
		PRINT("Occupancy is %i shmem size = %li numregs = %i\n", nblocks, sizeof(cuda_kick_shmem), attr.numRegs);
		nblocks *= cuda_smp_count();
		shown = true;
	}
	return nblocks;

}
