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
#include <cosmictiger/kernels.hpp>
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
		//	PRINT("CC\n");

#ifdef TREEPM
		const float rsinv = 1.f / data.rs;
		const float rsinv2 = sqr(rsinv);
#endif
		expansion<float> L;

		for (int i = 0; i < EXPANSION_SIZE; i++) {
			L[i] = 0.0f;
		}
		for (int i = tid; i < multlist.size(); i += WARP_SIZE) {
			const tree_node& other = tree_nodes[multlist[i]];
			const multipole<float>& M = other.mpos->multi;
			sfmm::vec3<float> dx;
			for (int dim = 0; dim < NDIM; dim++) {
				dx[dim] = distance(self.mpos->pos[dim], other.mpos->pos[dim]);
			}
			flops += M2L(L, M, dx, do_phi);
		}
		shared_reduce_add_array<float, EXPANSION_SIZE>(L.data());
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
		//	PRINT("CP\n");

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
				sfmm::vec3<float> dx;
				dx[XDIM] = distance(self.mpos->pos[XDIM], src_x[j]);
				dx[YDIM] = distance(self.mpos->pos[YDIM], src_y[j]);
				dx[ZDIM] = distance(self.mpos->pos[ZDIM], src_z[j]);
				P2L(L, 1.0f, dx, do_phi);
			}
		}
		shared_reduce_add_array<float, EXPANSION_SIZE>(L.data());
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
		__syncwarp();
		int mi = 0;
		const int cnt = multlist.size();
		while (mi < cnt) {
			int mend = min(cnt, mi + KICK_C_MAX);
			for (int this_mi = mi; this_mi < mend; this_mi++) {
				cuda::memcpy_async(group, (void*) &multis[this_mi - mi], tree_nodes[multlist[this_mi]].mpos, sizeof(multi_pos), barrier);
			}
			mend -= mi;
			mi += mend;
			barrier.arrive_and_wait();
			force_type<float> f;
			sfmm::vec3<float> dx;
			for (int j = 0; j < mend; j++) {
				const auto& pos = multis[j].pos;
				const auto& M = multis[j].multi;
				for (int k = tid; k < nsink; k += WARP_SIZE) {
					auto& F = force[k];
					f.init();
					dx[XDIM] = distance(sink_x[k], pos[XDIM]);
					dx[YDIM] = distance(sink_y[k], pos[YDIM]);
					dx[ZDIM] = distance(sink_z[k], pos[ZDIM]);
					flops += M2P(f, M, dx, do_phi);
					F += f;
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
					if (r2 > h2) {
						phi = rsqrt(r2);					// 4
						f = sqr(phi) * phi;					// 2
						direct++;
					} else {
						close++;
						gsoft(f, phi, r2, hinv, h2inv, h3inv, do_phi);
					}
					fx = fmaf(dx0, f, fx);                     // 2
					fy = fmaf(dx1, f, fy);                     // 2
					fz = fmaf(dx2, f, fz);                     // 2
					pot -= phi;                                  // 1
					flops += 21;
				}
				auto& F = force[k];
				F.force[0] -= fx;
				F.force[1] -= fy;
				F.force[2] -= fz;
				flops += 4;
				F.potential += pot;
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
					if (r2 > h2) {
						phi = rsqrt(r2);					// 4
						f = sqr(phi) * phi;					// 2
						direct++;
					} else {
						close++;
						gsoft(f, phi, r2, hinv, h2inv, h3inv, do_phi);
					}
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
					F.force[0] -= fx;
					F.force[1] -= fy;
					F.force[2] -= fz;
					flops += 4;
					F.potential += pot;
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

#ifdef FMM

__device__
void cuda_gravity_cc_ewald(const cuda_kick_data& data, expansion<float>& Lacc, const tree_node& self, const device_vector<int>& multlist, bool do_phi) {
	const int &tid = threadIdx.x;
	const auto& tree_nodes = data.tree_nodes;
	flop_counter<int> flops = 0;
	if (multlist.size()) {
		expansion<float> L;

		for (int i = 0; i < EXPANSION_SIZE; i++) {
			L[i] = 0.0f;
		}
		for (int i = tid; i < multlist.size(); i += WARP_SIZE) {
			const tree_node& other = tree_nodes[multlist[i]];
			multipole<float> M = other.mpos->multi;
			//	flops += apply_scale_factor(M);
			sfmm::vec3<float> dx;
			for (int dim = 0; dim < NDIM; dim++) {
				dx[dim] = distance(self.mpos->pos[dim], other.mpos->pos[dim]);
			}
			flops += M2L_ewald(L, M, dx, do_phi);
		}
		//flops += apply_scale_factor_inv(L);
		shared_reduce_add_array<float, EXPANSION_SIZE>(L.data());
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
				sfmm::vec3<float> dx;
				dx[XDIM] = distance(self.mpos->pos[XDIM], src_x[j]);
				dx[YDIM] = distance(self.mpos->pos[YDIM], src_y[j]);
				dx[ZDIM] = distance(self.mpos->pos[ZDIM], src_z[j]);
				flops += P2L_ewald(L, 1.0, dx, do_phi);
			}
		}
		shared_reduce_add_array<float, EXPANSION_SIZE>(L.data());
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
				cuda::memcpy_async(group, (void*) &multis[this_mi - mi], tree_nodes[multlist[this_mi]].mpos, sizeof(multi_pos), barrier);
			}
			mend -= mi;
			mi += mend;
			barrier.arrive_and_wait();
			force_type<float> f;
			sfmm::vec3<float> dx;

			for (int j = 0; j < mend; j++) {
				const auto& pos = multis[j].pos;
				const auto& M = multis[j].multi;
				for (int k = tid; k < nsink; k += WARP_SIZE) {
					auto& F = force[k];
					f.init();
					dx[XDIM] = distance(sink_x[k], pos[XDIM]);
					dx[YDIM] = distance(sink_y[k], pos[YDIM]);
					dx[ZDIM] = distance(sink_z[k], pos[ZDIM]);
					flops += M2P(f, M, dx, do_phi);
					F += f;
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
			__syncwarp();
			force_type<float> f;
			sfmm::vec3<float> dx;
			for (int k = tid; k < nsink; k += WARP_SIZE) {
				f.init();
				for (int j = 0; j < part_index; j++) {
					dx[0] = distance(sink_x[k], src_x[j]); // 1
					dx[1] = distance(sink_y[k], src_y[j]); // 1
					dx[2] = distance(sink_z[k], src_z[j]); // 1
					flops += P2P_ewald(f, 1.f, dx);
				}
				auto& F = force[k];
				F += f;
			}
		}
		__syncwarp();
	}
	__syncwarp();
	add_gpu_flops(flops);
}
#endif
