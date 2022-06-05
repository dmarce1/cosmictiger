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

#define MIN_PARTS_PCCP 24
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
	const float& hsoft = params.h;
	int flops = 0;
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
		F.phi += L2(0, 0, 0);
		F.gx -= L2(1, 0, 0);
		F.gy -= L2(0, 1, 0);
		F.gz -= L2(0, 0, 1);
		F.gz *= params.GM;
		F.gy *= params.GM;
		F.gx *= params.GM;
		F.phi *= params.GM;
		vx = vels[snki][XDIM];
		vy = vels[snki][YDIM];
		vz = vels[snki][ZDIM];
		const float sgn = params.top ? 1.f : -1.f;
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

		vels[snki][XDIM] = vx;
		vels[snki][YDIM] = vy;
		vels[snki][ZDIM] = vz;
		phi_tot += 0.5f * force[i].phi;
		flops += 557 + params.do_phi * 178;

	}
	shared_reduce_add(phi_tot);
	shared_reduce_add(xmom_tot);
	shared_reduce_add(ymom_tot);
	shared_reduce_add(zmom_tot);
	shared_reduce_add(nmom_tot);
	shared_reduce_add(kin_tot);
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
		int flops = 0;
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
				if (self.leaf) {
					const int nsinks = self.part_range.second - self.part_range.first;
					force.resize(nsinks);
					for (int l = tid; l < nsinks; l += WARP_SIZE) {
						auto& f = force[l];
						f.gx = f.gy = f.gz = f.phi = 0.f;
					}
				}
				__syncwarp();
				{
					nextlist.resize(0);
					cclist.resize(0);
					leaflist.resize(0);
					auto& checks = echecks;
					do {
						const float thetainv = 1.f / global_params.theta;
						const int maxi = round_up(checks.size(), WARP_SIZE);
						for (int i = tid; i < maxi; i += WARP_SIZE) {
							bool cc = false;
							bool next = false;
							bool leaf = false;
							if (i < checks.size()) {
								const tree_node& other = tree_nodes[checks[i]];
								for (int dim = 0; dim < NDIM; dim++) {
									dx[dim] = distance(self.pos[dim], other.pos[dim]); // 3
								}
								float R2 = sqr(dx[XDIM], dx[YDIM], dx[ZDIM]);
								R2 = fmaxf(R2, sqr(fmaxf(0.5f - (self.radius + other.radius), 0.f)));
								const float dcc = (self.radius + other.radius) * thetainv;
								cc = R2 > sqr(dcc);
								flops += 24;
								if (!cc) {
									leaf = other.leaf;
									next = !leaf;
									ALWAYS_ASSERT(!(leaf && self.leaf));
								}
							}
							{
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
					cuda_gravity_cc_ewald(data, L.back().expansion, self, cclist, global_params.do_phi);
					if (!self.leaf) {
						const int start = checks.size();
						checks.resize(start + leaflist.size());
						cuda::memcpy_async(group, checks.data() + start, leaflist.data(), leaflist.size() * sizeof(int), barrier);
						barrier.arrive_and_wait();
					}
				}
				{
					nextlist.resize(0);
					leaflist.resize(0);
					cclist.resize(0);
					cplist.resize(0);
					pclist.resize(0);
					auto& checks = dchecks;
					const float thetainv = 1.f / global_params.theta;
					do {
						const int maxi = round_up(checks.size(), WARP_SIZE);
						for (int i = tid; i < maxi; i += WARP_SIZE) {
							bool cc = false;
							bool next = false;
							bool leaf = false;
							bool cp = false;
							bool pc = false;
							if (i < checks.size()) {
								const tree_node& other = tree_nodes[checks[i]];
								for (int dim = 0; dim < NDIM; dim++) {
									dx[dim] = distance(self.pos[dim], other.pos[dim]); // 3
								}
								const float R2 = sqr(dx[XDIM], dx[YDIM], dx[ZDIM]);
								const float mind = self.radius + other.radius + h;
								const float dcc = fmaxf((self.radius + other.radius) * thetainv, mind);
								const float dcp = fmaxf((self.radius * thetainv + other.radius), mind);
								const float dpc = fmaxf((self.radius + other.radius * thetainv), mind);
								const auto self_parts = self.nparts();
								const auto other_parts = other.nparts();
								cc = (R2 > sqr(dcc)); // && min(self_parts, (part_int) (2*MIN_PARTS2_CC)) * min(other_parts, (part_int) (2*MIN_PARTS2_CC)) >= MIN_PARTS2_CC;
								flops += 20;
								if (!cc && other.leaf && self.leaf) {
									pc = R2 > sqr(dpc) && self_parts >= MIN_PARTS_PCCP;
									cp = R2 > sqr(dcp) && other_parts >= MIN_PARTS_PCCP;
									if (pc && cp) {
										if (self_parts < other_parts) {
											cp = false;
										} else if (self_parts > other_parts) {
											pc = false;
										} else if (dcp > dpc) {
											pc = false;
										} else if (dcp < dpc) {
											cp = false;
										} else {
											cp = pc = false;
										}
									}
									flops += 4;
								}
								if (!cc && !cp && !pc) {
									leaf = other.leaf;
									next = !leaf;
								}
							}
							{
								int l;
								int total;
								int start;
								l = cc;
								compute_indices(l, total);
								start = cclist.size();
								__syncwarp();
								cclist.resize(start + total);
								if (cc) {
									cclist[l + start] = checks[i];
								}
								if (self.leaf) {
									l = cp;
									compute_indices(l, total);
									start = cplist.size();
									__syncwarp();
									cplist.resize(start + total);
									if (cp) {
										cplist[l + start] = checks[i];
									}
									l = pc;
									compute_indices(l, total);
									start = pclist.size();
									__syncwarp();
									pclist.resize(start + total);
									if (pc) {
										pclist[l + start] = checks[i];
									}
								}
								l = next;
								compute_indices(l, total);
								start = nextlist.size();
								__syncwarp();
								nextlist.resize(start + total);
								if (next) {
									nextlist[l + start] = checks[i];
								}
								l = leaf;
								compute_indices(l, total);
								start = leaflist.size();
								__syncwarp();
								leaflist.resize(start + total);
								if (leaf) {
									leaflist[l + start] = checks[i];
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
					cuda_gravity_cc_direct(data, L.back().expansion, self, cclist, global_params.do_phi);
					cuda_gravity_cp_direct(data, L.back().expansion, self, cplist, global_params.do_phi);
					if (self.leaf) {
						__syncwarp();
						const float h = global_params.h;
						cuda_gravity_pc_direct(data, self, pclist, global_params.do_phi);
						cuda_gravity_pp_direct(data, self, leaflist, h, global_params.do_phi);
					} else {
						const int start = checks.size();
						checks.resize(start + leaflist.size());
						cuda::memcpy_async(group, checks.data() + start, leaflist.data(), leaflist.size() * sizeof(int), barrier);
						barrier.arrive_and_wait();
					}
				}
				if (self.leaf) {
					__syncwarp();
					do_kick(kr, global_params, data, L.back().expansion, self);
					sparams.pop_back();
					depth--;
				} else {
					const tree_id child = self.children[LEFT];
					const int i1 = L.size() - 1;
					const int i2 = L.size();
					L.resize(i2 + 1);
					for (int l = tid; l < EXPANSION_SIZE; l += WARP_SIZE) {
						L[i2].expansion[l] = L[i1].expansion[l];
					}
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

kick_return cuda_execute_kicks(kick_params kparams, fixed32* dev_x, fixed32* dev_y, fixed32* dev_z, tree_node* dev_tree_nodes, vector<kick_workitem> workitems,
		cudaStream_t stream) {
	timer tm;
	tm.start();
	int nblocks = kick_block_count();
	ALWAYS_ASSERT(workitems.size());
	nblocks = std::min(nblocks, (int) workitems.size());
	static char* data_ptr = nullptr;
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
			CUDA_CHECK(cudaFree(data_ptr));
		}
		data_size = alloc_size;
		CUDA_CHECK(cudaMallocManaged(&data_ptr, alloc_size));
	}
	int offset = 0;
	cuda_kick_params* dev_kick_params = (cuda_kick_params*) (data_ptr + offset);
	offset += sizeof(cuda_kick_params) * workitems.size();
	kick_return* return_ = (kick_return*) (data_ptr + offset);
	offset += sizeof(kick_return);
	int* dchecks = (int*) (data_ptr + offset);
	offset += sizeof(int) * dcount;
	int* echecks = (int*) (data_ptr + offset);
	offset += sizeof(int) * ecount;
	int* current_index = (int*) (data_ptr + offset);
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
	data.x_snk = &particles_pos(XDIM, 0);
	data.y_snk = &particles_pos(YDIM, 0);
	data.z_snk = &particles_pos(ZDIM, 0);
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
		params.dchecks = dchecks + dindices[i];
		params.echecks = echecks + eindices[i];
		params.dcount = dindices[i + 1] - dindices[i];
		params.ecount = eindices[i + 1] - eindices[i];
		dev_kick_params[i] = std::move(params);
	}
	cuda_set_device();

	cuda_kick_kernel<<<nblocks, WARP_SIZE, sizeof(cuda_kick_shmem), stream>>>(return_, kparams,data, dev_kick_params, workitems.size(), current_index);
	cuda_stream_synchronize(stream);
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
