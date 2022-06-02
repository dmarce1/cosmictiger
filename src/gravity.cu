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

#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/flops.hpp>
#include <cosmictiger/gravity.hpp>
#include <cooperative_groups.h>
#include <cuda/barrier>

__device__
void cuda_gravity_cc_direct(const cuda_kick_data& data, expansion<float>& Lacc, const tree_node& self, const device_vector<int>& multlist, bool do_phi) {
	const int &tid = threadIdx.x;
	const auto& tree_nodes = data.tree_nodes;
	int flops = 0;
	if (multlist.size()) {
		expansion<float> L;
		expansion<float> D;
		for (int i = 0; i < EXPANSION_SIZE; i++) {
			L[i] = 0.0f;
		}
		for (int i = tid; i < multlist.size(); i += WARP_SIZE) {
			const tree_node& other = tree_nodes[multlist[i]];
			const multipole<float>& M = other.multi;
			array<float, NDIM> dx;
			for (int dim = 0; dim < NDIM; dim++) {
				dx[dim] = distance(self.pos[dim], other.pos[dim]);
			}
			flops += 3 + greens_function(D, dx);
			flops += M2L(L, M, D, do_phi);
		}
		for (int i = 0; i < EXPANSION_SIZE; i++) {
			shared_reduce_add(L[i]);
		}
		for (int i = tid; i < EXPANSION_SIZE; i += WARP_SIZE) {
			Lacc[i] += L[i];
		}
		flops += 6 * EXPANSION_SIZE;
		__syncwarp();
	}
	add_gpu_flops(flops);
}

__device__
void cuda_gravity_cp_direct(const cuda_kick_data& data, expansion<float>& Lacc, const tree_node& self, const device_vector<int>& partlist, bool do_phi) {
	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	const auto* main_src_x = data.x;
	const auto* main_src_y = data.y;
	const auto* main_src_z = data.z;
	auto& src_x = shmem.x;
	auto& src_y = shmem.y;
	auto& src_z = shmem.z;
	auto& barrier = shmem.barrier;
	const auto* tree_nodes = data.tree_nodes;
	const int &tid = threadIdx.x;
	int flops = 0;
	if (partlist.size()) {
		int part_index;
		expansion<float> L;
		for (int j = 0; j < EXPANSION_SIZE; j++) {
			L[j] = 0.0;
		}
		int i = 0;
		auto these_parts = tree_nodes[partlist[0]].part_range;
		const auto partsz = partlist.size();
		while (i < partsz) {
			auto group = cooperative_groups::this_thread_block();
			if (group.thread_rank() == 0) {
				init(&barrier, group.size());
			}
			group.sync();
			part_index = 0;
			while (part_index < KICK_PP_MAX && i < partsz) {
				while (i + 1 < partsz) {
					const auto other_tree_parts = tree_nodes[partlist[i + 1]].part_range;
					if (these_parts.second == other_tree_parts.first) {
						these_parts.second = other_tree_parts.second;
						i++;
					} else {
						break;
					}
				}
				const part_int imin = these_parts.first;
				const part_int imax = min(these_parts.first + (KICK_PP_MAX - part_index), these_parts.second);
				const int sz = imax - imin;
				cuda::memcpy_async(group, src_x.data() + part_index, main_src_x + imin, sizeof(fixed32) * sz, barrier);
				cuda::memcpy_async(group, src_y.data() + part_index, main_src_y + imin, sizeof(fixed32) * sz, barrier);
				cuda::memcpy_async(group, src_z.data() + part_index, main_src_z + imin, sizeof(fixed32) * sz, barrier);
				__syncwarp();
				these_parts.first += sz;
				part_index += sz;
				if (these_parts.first == these_parts.second) {
					i++;
					if (i < partsz) {
						these_parts = tree_nodes[partlist[i]].part_range;
					}
				}
			}
			barrier.arrive_and_wait();
			__syncwarp();
			for (int j = tid; j < part_index; j += warpSize) {
				array<float, NDIM> dx;
				dx[XDIM] = distance(self.pos[XDIM], src_x[j]);
				dx[YDIM] = distance(self.pos[YDIM], src_y[j]);
				dx[ZDIM] = distance(self.pos[ZDIM], src_z[j]);
				expansion<float> D;
				flops += EXPANSION_SIZE + 3 + greens_function(D, dx);
				for (int k = 0; k < EXPANSION_SIZE; k++) {
					L[k] += D[k];
				}
			}
		}
		for (int k = 0; k < EXPANSION_SIZE; k++) {
			shared_reduce_add(L[k]);
		}
		for (int i = tid; i < EXPANSION_SIZE; i += WARP_SIZE) {
			Lacc[i] += L[i];
		}
		flops += 6 * EXPANSION_SIZE;

		__syncwarp();
	}
	add_gpu_flops(flops);
}

__device__
void cuda_gravity_pc_direct(const cuda_kick_data& data, const tree_node& self, const device_vector<int>& multlist, bool do_phi) {
	const int &tid = threadIdx.x;
	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	auto &force = shmem.f;
	const int nsink = self.part_range.second - self.part_range.first;
	const auto& sink_x = data.x + self.part_range.first;
	const auto& sink_y = data.y + self.part_range.first;
	const auto& sink_z = data.z + self.part_range.first;
	const auto* tree_nodes = data.tree_nodes;
	int flops = 0;
	if (multlist.size()) {
		__syncwarp();
		for (int k = tid; k < nsink; k += WARP_SIZE) {
			auto& F = force[k];
			expansion2<float> L;
			L(0, 0, 0) = L(1, 0, 0) = L(0, 1, 0) = L(0, 0, 1) = 0.0f;
			for (int j = 0; j < multlist.size(); j++) {
				array<float, NDIM> dx;
				const auto& pos = tree_nodes[multlist[j]].pos;
				const auto& M = tree_nodes[multlist[j]].multi;
				dx[XDIM] = distance(sink_x[k], pos[XDIM]);
				dx[YDIM] = distance(sink_y[k], pos[YDIM]);
				dx[ZDIM] = distance(sink_z[k], pos[ZDIM]);
				expansion<float> D;
				flops += 3 + greens_function(D, dx);
				flops += M2L(L, M, D, do_phi);
			}
			F.gx -= L(1, 0, 0);
			F.gy -= L(0, 1, 0);
			F.gz -= L(0, 0, 1);
			F.phi += L(0, 0, 0);
			flops += 4;
		}
		__syncwarp();
	}
	__syncwarp();
	add_gpu_flops(flops);
}

__device__
void cuda_gravity_cc_ewald(const cuda_kick_data& data, expansion<float>& Lacc, const tree_node& self, const device_vector<int>& multlist, bool do_phi) {
	const int &tid = threadIdx.x;
	const auto& tree_nodes = data.tree_nodes;
	int flops = 0;
	if (multlist.size()) {
		expansion<float> L;
		expansion<float> D;
		for (int i = 0; i < EXPANSION_SIZE; i++) {
			L[i] = 0.0f;
		}
		for (int i = tid; i < multlist.size(); i += WARP_SIZE) {
			const tree_node& other = tree_nodes[multlist[i]];
			const multipole<float>& M = other.multi;
			array<float, NDIM> dx;
			for (int dim = 0; dim < NDIM; dim++) {
				dx[dim] = distance(self.pos[dim], other.pos[dim]);
			}
			flops += 3 + ewald_greens_function(D, dx);
			flops += M2L(L, M, D, do_phi);
		}
		for (int i = 0; i < EXPANSION_SIZE; i++) {
			shared_reduce_add(L[i]);
		}
		for (int i = tid; i < EXPANSION_SIZE; i += WARP_SIZE) {
			Lacc[i] += L[i];
		}
		flops += 6 * EXPANSION_SIZE;
		__syncwarp();
	}
	add_gpu_flops(flops);
}

__device__
void cuda_gravity_pp_direct(const cuda_kick_data& data, const tree_node& self, const device_vector<int>& partlist, float h, bool do_phi) {
	const int &tid = threadIdx.x;
	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	auto &force = shmem.f;
	const int nsink = self.part_range.second - self.part_range.first;
	const auto& sink_x = data.x + self.part_range.first;
	const auto& sink_y = data.y + self.part_range.first;
	const auto& sink_z = data.z + self.part_range.first;
	auto& barrier = shmem.barrier;
	const auto* main_src_x = data.x;
	const auto* main_src_y = data.y;
	const auto* main_src_z = data.z;
	auto& src_x = shmem.x;
	auto& src_y = shmem.y;
	auto& src_z = shmem.z;
	const auto* tree_nodes = data.tree_nodes;
	int part_index;
	const float h2 = sqr(h);
	const float hinv = 1.f / h;
	const float h2inv = sqr(hinv);
	const float h3inv = h2inv * hinv;
	int flops = 7;
	if (partlist.size()) {
		int i = 0;
		auto these_parts = tree_nodes[partlist[0]].part_range;
		const auto partsz = partlist.size();
		while (i < partsz) {
			auto group = cooperative_groups::this_thread_block();
			part_index = 0;
			while (part_index < KICK_PP_MAX && i < partsz) {
				while (i + 1 < partsz) {
					const auto other_tree_parts = tree_nodes[partlist[i + 1]].part_range;
					if (these_parts.second == other_tree_parts.first) {
						these_parts.second = other_tree_parts.second;
						i++;
					} else {
						break;
					}
				}
				const part_int imin = these_parts.first;
				const part_int imax = min(these_parts.first + (KICK_PP_MAX - part_index), these_parts.second);
				const int sz = imax - imin;
				cuda::memcpy_async(group, src_x.data() + part_index, main_src_x + imin, sizeof(fixed32) * sz, barrier);
				cuda::memcpy_async(group, src_y.data() + part_index, main_src_y + imin, sizeof(fixed32) * sz, barrier);
				cuda::memcpy_async(group, src_z.data() + part_index, main_src_z + imin, sizeof(fixed32) * sz, barrier);
				__syncwarp();
				these_parts.first += sz;
				part_index += sz;
				if (these_parts.first == these_parts.second) {
					i++;
					if (i < partsz) {
						these_parts = tree_nodes[partlist[i]].part_range;
					}
				}
			}
			barrier.arrive_and_wait();
			float fx;
			float fy;
			float fz;
			float pot;
			float dx0;
			float dx1;
			float dx2;
			float r3inv;
			float r1inv;
			__syncwarp();
			int kmid = (nsink / WARP_SIZE) * WARP_SIZE;
			if (nsink - kmid >= WARP_SIZE / 2) {
				kmid = nsink;
			}
			for (int k = tid; k < kmid; k += WARP_SIZE) {
				fx = 0.f;
				fy = 0.f;
				fz = 0.f;
				pot = 0.f;
				for (int j = 0; j < part_index; j++) {
					dx0 = distance(sink_x[k], src_x[j]); // 1
					dx1 = distance(sink_y[k], src_y[j]); // 1
					dx2 = distance(sink_z[k], src_z[j]); // 1
					const auto r2 = sqr(dx0, dx1, dx2);  // 5
					if (r2 > h2) {
						r1inv = rsqrt(r2);					// 4
						r3inv = sqr(r1inv) * r1inv;		// 2
					} else {
						const float q2 = r2 * h2inv;		// 1
						r3inv = fmaf(q2, -1.5f, 2.5f) * h3inv;	// 3
						flops -= 2;
						if (do_phi) {
							r1inv = float(3.0f / 8.0f);
							r1inv = fmaf(q2, r1inv, -float(5.f / 4.f)); // 2
							r1inv = fmaf(q2, r1inv, float(15.0f / 8.0f)); // 2
							r1inv *= hinv;                    // 1
							flops += 5;
						}
					}
					fx = fmaf(dx0, r3inv, fx);                     // 2
					fy = fmaf(dx1, r3inv, fy);                     // 2
					fz = fmaf(dx2, r3inv, fz);                     // 2
					pot -= r1inv;                                  // 1
					flops += 21;
				}
				auto& F = force[k];
				F.gx -= fx;
				F.gy -= fy;
				F.gz -= fz;
				flops += 4;
				F.phi += pot;
			}
			for (int k = kmid; k < nsink; k++) {
				fx = 0.f;
				fy = 0.f;
				fz = 0.f;
				pot = 0.f;
				for (int j = tid; j < part_index; j += WARP_SIZE) {
					dx0 = distance(sink_x[k], src_x[j]); // 1
					dx1 = distance(sink_y[k], src_y[j]); // 1
					dx2 = distance(sink_z[k], src_z[j]); // 1
					const auto r2 = sqr(dx0, dx1, dx2);  // 5
					if (r2 > h2) {
						r1inv = rsqrt(r2);					// 4
						r3inv = sqr(r1inv) * r1inv;		// 2
					} else {
						const float q2 = r2 * h2inv;		// 1
						r3inv = fmaf(q2, -1.5f, 2.5f) * h3inv;	// 3
						flops -= 2;
						if (do_phi) {
							r1inv = float(3.0f / 8.0f);
							r1inv = fmaf(q2, r1inv, -float(5.f / 4.f)); // 2
							r1inv = fmaf(q2, r1inv, float(15.0f / 8.0f)); // 2
							r1inv *= hinv;                    // 1
							flops += 5;
						}
					}
					fx = fmaf(dx0, r3inv, fx);                     // 2
					fy = fmaf(dx1, r3inv, fy);                     // 2
					fz = fmaf(dx2, r3inv, fz);                     // 2
					pot -= r1inv;                                  // 1
					flops += 21;
				}
				shared_reduce_add(fx);
				shared_reduce_add(fy);
				shared_reduce_add(fz);
				if (do_phi) {
					shared_reduce_add(pot);
				}
				if (tid == 0) {
					auto& F = force[k];
					F.gx -= fx;
					F.gy -= fy;
					F.gz -= fz;
					flops += 4;
					F.phi += pot;
				}
			}
		}
		__syncwarp();
	}
	__syncwarp();
	add_gpu_flops(flops);
}
