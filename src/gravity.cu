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

__managed__ double iclose;
__managed__ double idirect;
__managed__ double inparts;
__managed__ bool use_gravity_counters = false;

void reset_gravity_counters() {
	iclose = idirect = inparts = 0;
}

void set_gravity_counter_use(bool code) {
	use_gravity_counters = code;
}

void get_gravity_counters(double& close, double& direct) {
	close = iclose / inparts;
	direct = idirect / inparts;
}

__device__
void cuda_gravity_cc_direct(const cuda_kick_data& data, expansion<float>& Lacc, const tree_node& self, const device_vector<int>& multlist, bool do_phi) {
	const int &tid = threadIdx.x;
	const auto& tree_nodes = data.tree_nodes;
	flop_counter<int> flops = 0;
	if (multlist.size()) {
#ifdef TREEPM
		const float rsinv = 1.f / data.rs;
		const float rsinv2 = sqr(rsinv);
#endif
		expansion<float> L;
		expansion<float> D;
		for (int i = 0; i < EXPANSION_SIZE; i++) {
			L[i] = 0.0f;
		}
		for (int i = tid; i < multlist.size(); i += WARP_SIZE) {
			const tree_node& other = tree_nodes[multlist[i]];
			const multipole<float>& M = other.mpos->multi;
			array<float, NDIM> dx;
			for (int dim = 0; dim < NDIM; dim++) {
				dx[dim] = distance(self.mpos->pos[dim], other.mpos->pos[dim]);
			}
#ifdef TREEPM
			flops += 6 + greens_function(D, dx, rsinv, rsinv2);
#else
			flops += 6 + greens_function(D, dx);
#endif
			flops += M2L(L, M, D, do_phi);
		}
		shared_reduce_add_array(L);
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
	auto& src_x = shmem.X.x;
	auto& src_y = shmem.X.y;
	auto& src_z = shmem.X.z;
	auto& barrier = shmem.barrier;
	const auto* tree_nodes = data.tree_nodes;
	const int &tid = threadIdx.x;
	flop_counter<int> flops = 0;
	if (partlist.size()) {
#ifdef TREEPM
		const float rsinv = 1.f / data.rs;
		const float rsinv2 = sqr(rsinv);
#endif
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
				dx[XDIM] = distance(self.mpos->pos[XDIM], src_x[j]);
				dx[YDIM] = distance(self.mpos->pos[YDIM], src_y[j]);
				dx[ZDIM] = distance(self.mpos->pos[ZDIM], src_z[j]);
				expansion<float> D;
#ifdef TREEPM
				flops += 6 + greens_function(D, dx, rsinv, rsinv2);
#else
				flops += 6 + greens_function(D, dx);
#endif
				for (int k = 0; k < EXPANSION_SIZE; k++) {
					L[k] += D[k];
				}
			}
		}
		shared_reduce_add_array(L);
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
	auto& multis = shmem.mpos;
	const int nsink = self.part_range.second - self.part_range.first;
	const auto* sink_x = data.x + self.part_range.first;
	const auto* sink_y = data.y + self.part_range.first;
	const auto* sink_z = data.z + self.part_range.first;
	auto& barrier = shmem.barrier;
	const auto* tree_nodes = data.tree_nodes;
	flop_counter<int> flops = 0;
	auto group = cooperative_groups::this_thread_block();
	if (multlist.size()) {
#ifdef TREEPM
		const float rsinv = 1.f / data.rs;
		const float rsinv2 = sqr(rsinv);
#endif
		__syncwarp();
		int mi = 0;
		const int cnt = multlist.size();
		while (mi < cnt) {
			int mend = min(cnt, mi + KICK_C_MAX);
			for (int this_mi = mi; this_mi < mend; this_mi++) {
				cuda::memcpy_async(group, &multis[this_mi - mi], tree_nodes[multlist[this_mi]].mpos, sizeof(multi_pos), barrier);
			}
			mend -= mi;
			mi += mend;
			barrier.arrive_and_wait();
			expansion2<float> L;
			array<float, NDIM> dx;
			expansion<float> D;
			for (int j = 0; j < mend; j++) {
				const auto& pos = multis[j].pos;
				const auto& M = multis[j].multi;
				for (int k = tid; k < nsink; k += WARP_SIZE) {
					auto& F = force[k];
					L(0, 0, 0) = L(1, 0, 0) = L(0, 1, 0) = L(0, 0, 1) = 0.0f;
					dx[XDIM] = distance(sink_x[k], pos[XDIM]);
					dx[YDIM] = distance(sink_y[k], pos[YDIM]);
					dx[ZDIM] = distance(sink_z[k], pos[ZDIM]);
#ifdef TREEPM
					flops += 6 + greens_function(D, dx, rsinv, rsinv2);
#else
					flops += 6 + greens_function(D, dx);
#endif
					flops += M2L(L, M, D, do_phi);
#ifndef TREEPM
					F.gx -= SCALE_FACTOR2 * L(1, 0, 0);
					F.gy -= SCALE_FACTOR2 * L(0, 1, 0);
					F.gz -= SCALE_FACTOR2 * L(0, 0, 1);
					F.phi += SCALE_FACTOR1 * L(0, 0, 0);
#endif
					flops += 4;
				}
			}
			__syncwarp();
		}
	}
	__syncwarp();
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
	const auto* sink_x = data.x + self.part_range.first;
	const auto* sink_y = data.y + self.part_range.first;
	const auto* sink_z = data.z + self.part_range.first;
	auto& barrier = shmem.barrier;
	const auto* main_src_x = data.x;
	const auto* main_src_y = data.y;
	const auto* main_src_z = data.z;
	auto& src_x = shmem.X.x;
	auto& src_y = shmem.X.y;
	auto& src_z = shmem.X.z;
	const auto* tree_nodes = data.tree_nodes;
	int part_index;
	const float h2 = sqr(h);
	const float hinv = 1.f / h;
	const float h2inv = sqr(hinv);
	const float h3inv = h2inv * hinv;
	flop_counter<int> flops = 7;
	int close = 0;
	int direct = 0;
	if (partlist.size()) {
#ifdef TREEPM
		const float rsinv = 1.f / data.rs;
		const float rsinv2 = sqr(rsinv);
		const float cons = 2.0f / sqrtf(M_PI);
#endif
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
			float f;
			float phi;
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
#ifdef TREEPM
					const float r = sqrt(r2);
					const float erf0 = erff(r * rsinv);
					const float exp0 = expf(-r2 * rsinv2);
					float rinv;
					if (r2 > h2) {
						phi = rsqrt(r2);					// 4
						f = sqr(phi) * phi;		// 2
						rinv = phi;
						direct++;
					} else {
						close++;
						gsoft(f, phi, r2, hinv, h2inv, h3inv, do_phi);
						rinv = 1.f / r;
					}
					phi += erf0 * rinv;
					f += (-erf0 + cons * r * rsinv * exp0) * sqr(rinv) * rinv;
#else
					if (r2 > h2) {
						phi = rsqrt(r2);					// 4
						f = sqr(phi) * phi;// 2
						direct++;
					} else {
						close++;
						gsoft(f, phi, r2, hinv, h2inv, h3inv, do_phi);
					}
#endif
					fx = fmaf(dx0, f, fx);                     // 2
					fy = fmaf(dx1, f, fy);                     // 2
					fz = fmaf(dx2, f, fz);                     // 2
					pot -= phi;                                  // 1
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
					dx0 = distance(sink_x[k], src_x[j]); // 2
					dx1 = distance(sink_y[k], src_y[j]); // 2
					dx2 = distance(sink_z[k], src_z[j]); // 2
					const auto r2 = sqr(dx0, dx1, dx2);  // 5
#ifdef TREEPM
					const float r = sqrt(r2);
					const float erf0 = erff(r * rsinv);
					const float exp0 = expf(-r2 * rsinv2);
					float rinv;
					if (r2 > h2) {
						phi = rsqrt(r2);					// 4
						f = sqr(phi) * phi;		// 2
						rinv = phi;
						direct++;
					} else {
						close++;
						gsoft(f, phi, r2, hinv, h2inv, h3inv, do_phi);
						rinv = 1.f / r;
					}
					phi += erf0 * rinv;
					f += (-erf0 + cons * r * rsinv * exp0) * sqr(rinv) * rinv;
#else
					if (r2 > h2) {
						phi = rsqrt(r2);					// 4
						f = sqr(phi) * phi;// 2
						direct++;
					} else {
						close++;
						gsoft(f, phi, r2, hinv, h2inv, h3inv, do_phi);
					}
#endif
					fx = fmaf(dx0, f, fx);                     // 2
					fy = fmaf(dx1, f, fy);                     // 2
					fz = fmaf(dx2, f, fz);                     // 2
					pot -= phi;                                  // 1
					flops += 23;
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
	shared_reduce_add(close);
	shared_reduce_add(direct);
	if (tid == 0 && use_gravity_counters) {
		atomicAdd(&iclose, close);
		atomicAdd(&idirect, direct);
		atomicAdd(&inparts, nsink);
	}

	__syncwarp();
	add_gpu_flops(flops);

}

#ifndef TREEPM

__device__
void cuda_gravity_cc_ewald(const cuda_kick_data& data, expansion<float>& Lacc, const tree_node& self, const device_vector<int>& multlist, bool do_phi) {
	const int &tid = threadIdx.x;
	const auto& tree_nodes = data.tree_nodes;
	flop_counter<int> flops = 0;
	if (multlist.size()) {
		expansion<float> L;
		expansion<float> D;
		for (int i = 0; i < EXPANSION_SIZE; i++) {
			L[i] = 0.0f;
		}
		for (int i = tid; i < multlist.size(); i += WARP_SIZE) {
			const tree_node& other = tree_nodes[multlist[i]];
			multipole<float> M = other.mpos->multi;
			//	flops += apply_scale_factor(M);
			array<float, NDIM> dx;
			for (int dim = 0; dim < NDIM; dim++) {
				dx[dim] = distance(self.mpos->pos[dim], other.mpos->pos[dim]);
			}
			flops += 3 + ewald_greens_function(D, dx);
			flops += M2L(L, M, D, do_phi);
		}
		//flops += apply_scale_factor_inv(L);
		shared_reduce_add_array(L);
		for (int i = tid; i < EXPANSION_SIZE; i += WARP_SIZE) {
			Lacc[i] += L[i];
		}
		flops += 6 * EXPANSION_SIZE;
		__syncwarp();
	}
	add_gpu_flops(flops);
}

__device__
void cuda_gravity_cp_ewald(const cuda_kick_data& data, expansion<float>& Lacc, const tree_node& self, const device_vector<int>& partlist, bool do_phi) {
	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	const auto* main_src_x = data.x;
	const auto* main_src_y = data.y;
	const auto* main_src_z = data.z;
	auto& src_x = shmem.X.x;
	auto& src_y = shmem.X.y;
	auto& src_z = shmem.X.z;
	auto& barrier = shmem.barrier;
	const auto* tree_nodes = data.tree_nodes;
	const int &tid = threadIdx.x;
	flop_counter<int> flops = 0;
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
				dx[XDIM] = distance(self.mpos->pos[XDIM], src_x[j]);
				dx[YDIM] = distance(self.mpos->pos[YDIM], src_y[j]);
				dx[ZDIM] = distance(self.mpos->pos[ZDIM], src_z[j]);
				expansion<float> D;
				flops += EXPANSION_SIZE + 6 + ewald_greens_function(D, dx);
				for (int k = 0; k < EXPANSION_SIZE; k++) {
					L[k] += D[k];
				}
			}
		}
		shared_reduce_add_array(L);
		for (int i = tid; i < EXPANSION_SIZE; i += WARP_SIZE) {
			Lacc[i] += L[i];
		}
		flops += 6 * EXPANSION_SIZE;

		__syncwarp();
	}
	add_gpu_flops(flops);
}

__device__
void cuda_gravity_pc_ewald(const cuda_kick_data& data, const tree_node& self, const device_vector<int>& multlist, bool do_phi) {
	const int &tid = threadIdx.x;
	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	auto &force = shmem.f;
	auto& multis = shmem.mpos;
	const int nsink = self.part_range.second - self.part_range.first;
	const auto* sink_x = data.x + self.part_range.first;
	const auto* sink_y = data.y + self.part_range.first;
	const auto* sink_z = data.z + self.part_range.first;
	auto& barrier = shmem.barrier;
	const auto* tree_nodes = data.tree_nodes;
	flop_counter<int> flops = 0;
	auto group = cooperative_groups::this_thread_block();
	if (multlist.size()) {
		__syncwarp();
		int mi = 0;
		const int cnt = multlist.size();
		while (mi < cnt) {
			int mend = min(cnt, mi + KICK_C_MAX);
			for (int this_mi = mi; this_mi < mend; this_mi++) {
				cuda::memcpy_async(group, &multis[this_mi - mi], tree_nodes[multlist[this_mi]].mpos, sizeof(multi_pos), barrier);
			}
			mend -= mi;
			mi += mend;
			barrier.arrive_and_wait();
			expansion2<float> L;
			array<float, NDIM> dx;
			expansion<float> D;
			for (int j = 0; j < mend; j++) {
				const auto& pos = multis[j].pos;
				const auto& M = multis[j].multi;
				for (int k = tid; k < nsink; k += WARP_SIZE) {
					auto& F = force[k];
					L(0, 0, 0) = L(1, 0, 0) = L(0, 1, 0) = L(0, 0, 1) = 0.0f;
					dx[XDIM] = distance(sink_x[k], pos[XDIM]);
					dx[YDIM] = distance(sink_y[k], pos[YDIM]);
					dx[ZDIM] = distance(sink_z[k], pos[ZDIM]);
					flops += 6 + ewald_greens_function(D, dx);
					flops += M2L(L, M, D, do_phi);
					F.gx -= SCALE_FACTOR2 * L(1, 0, 0);
					F.gy -= SCALE_FACTOR2 * L(0, 1, 0);
					F.gz -= SCALE_FACTOR2 * L(0, 0, 1);
					F.phi += SCALE_FACTOR1 * L(0, 0, 0);
					flops += 4;
				}
			}
			__syncwarp();
		}
	}
	__syncwarp();
	add_gpu_flops(flops);
}

__device__
void cuda_gravity_pp_ewald(const cuda_kick_data& data, const tree_node& self, const device_vector<int>& partlist, float h, bool do_phi) {
	const int &tid = threadIdx.x;
	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	auto &force = shmem.f;
	const int nsink = self.part_range.second - self.part_range.first;
	const auto* sink_x = data.x + self.part_range.first;
	const auto* sink_y = data.y + self.part_range.first;
	const auto* sink_z = data.z + self.part_range.first;
	auto& barrier = shmem.barrier;
	const auto* main_src_x = data.x;
	const auto* main_src_y = data.y;
	const auto* main_src_z = data.z;
	auto& src_x = shmem.X.x;
	auto& src_y = shmem.X.y;
	auto& src_z = shmem.X.z;
	const auto* tree_nodes = data.tree_nodes;
	int part_index;
	const float h2 = sqr(h);
	const float hinv = 1.f / h;
	const float h2inv = sqr(hinv);
	const float h3inv = h2inv * hinv;
	flop_counter<int> flops = 7;
	int close = 0;
	int direct = 0;
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
			__syncwarp();
			ewald_const econst;
			const float dxinv = (EWALD_TABLE_SIZE - 3.0f);
			const float dx = 1.0f / dxinv;
			for (int k = tid; k < nsink; k += WARP_SIZE) {
				fx = 0.f;
				fy = 0.f;
				fz = 0.f;
				pot = 0.f;
				for (int j = 0; j < part_index; j++) {
					dx0 = distance(sink_x[k], src_x[j]); // 1
					dx1 = distance(sink_y[k], src_y[j]);// 1
					dx2 = distance(sink_z[k], src_z[j]);// 1
					const float sgnx = copysign(1.f, dx0);
					const float sgny = copysign(1.f, dx1);
					const float sgnz = copysign(1.f, dx2);
					dx0 = fabs(dx0);
					dx1 = fabs(dx1);
					dx2 = fabs(dx2);
					float this_fx;
					float this_fy;
					float this_fz;
					float this_pot;
					econst.table_interp(this_pot, this_fx, this_fy, this_fz, dx0, dx1, dx2, do_phi);
					fx += sgnx * this_fx;
					fy += sgny * this_fy;
					fz += sgnz * this_fz;
					pot += this_pot;

				}
				auto& F = force[k];
				F.gx += fx;
				F.gy += fy;
				F.gz += fz;
				flops += 4;
				F.phi += pot;
			}
		}
		__syncwarp();
	}
	__syncwarp();
	add_gpu_flops(flops);
}
#endif
