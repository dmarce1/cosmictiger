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
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/kernel.hpp>

__device__
int cuda_gravity_cc(gravity_cc_type type, const cuda_kick_data& data, expansion<float>& Lacc, const tree_node& self,
		const fixedcapvec<int, MULTLIST_SIZE>& multlist, bool do_phi) {
	int flops = 0;
	const int &tid = threadIdx.x;
	const auto& tree_nodes = data.tree_nodes;
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
			flops += 3;
			if (type == GRAVITY_DIRECT) {
				flops += greens_function(D, dx);
			} else {
				flops += ewald_greens_function(D, dx);
			}
			flops += M2L(L, M, D, do_phi);
		}
		for (int i = 0; i < EXPANSION_SIZE; i++) {
			shared_reduce_add(L[i]);
		}
		for (int i = tid; i < EXPANSION_SIZE; i += WARP_SIZE) {
			Lacc[i] += L[i];
		}
		__syncwarp();
	}
	shared_reduce_add(flops);
	return flops;
}

__device__
int cuda_gravity_cp(gravity_cc_type type, const cuda_kick_data& data, expansion<float>& Lacc, const tree_node& self,
		const fixedcapvec<int, PARTLIST_SIZE>& partlist, float dm_mass, float sph_mass, bool do_phi) {
	int flops = 0;
	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	const bool sph = data.sph;
	const auto* main_src_x = data.x;
	const auto* main_src_y = data.y;
	const auto* main_src_z = data.z;
	const auto* main_src_fpot = data.fpot;
	auto& src_x = shmem.src.x;
	auto& src_y = shmem.src.y;
	auto& src_z = shmem.src.z;
	auto& src_fpot = shmem.src.fpot;
	const auto* tree_nodes = data.tree_nodes;
	const int &tid = threadIdx.x;
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
				for (int j = tid; j < sz; j += WARP_SIZE) {
					const int i1 = part_index + j;
					const part_int i2 = j + imin;
					ASSERT(i2 >= 0);
					ASSERT(i2 < data.source_size);
					src_x[i1] = main_src_x[i2];
					src_y[i1] = main_src_y[i2];
					src_z[i1] = main_src_z[i2];
					src_fpot[i1] = main_src_fpot[i2];
				}
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
			__syncwarp();
			for (int j = tid; j < part_index; j += warpSize) {
				array<float, NDIM> dx;
				dx[XDIM] = distance(self.pos[XDIM], src_x[j]);
				dx[YDIM] = distance(self.pos[YDIM], src_y[j]);
				dx[ZDIM] = distance(self.pos[ZDIM], src_z[j]);
				const float mass = !sph ? 1.f : (sph && (src_fpot[j] != 0.f) ? sph_mass : dm_mass);
				flops += 3;
				expansion<float> D;
				if (type == GRAVITY_DIRECT) {
					flops += greens_function(D, dx);
				} else {
					flops += ewald_greens_function(D, dx);
				}
				for (int k = 0; k < EXPANSION_SIZE; k++) {
					L[k] += mass * D[k];
				}
				flops += 2 * EXPANSION_SIZE;
			}
		}
		for (int k = 0; k < EXPANSION_SIZE; k++) {
			shared_reduce_add(L[k]);
		}
		for (int i = tid; i < EXPANSION_SIZE; i += WARP_SIZE) {
			Lacc[i] += L[i];
		}

		__syncwarp();
	}
	shared_reduce_add(flops);
	return flops;

}

__device__
int cuda_gravity_pc(gravity_cc_type type, const cuda_kick_data& data, const tree_node&, const fixedcapvec<int, MULTLIST_SIZE>& multlist, int nactive,
		bool do_phi) {
	int flops = 0;
	const int &tid = threadIdx.x;
	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	auto &gx = shmem.gx;
	auto &gy = shmem.gy;
	auto &gz = shmem.gz;
	auto &phi = shmem.phi;
	const auto& sink_x = shmem.sink_x;
	const auto& sink_y = shmem.sink_y;
	const auto& sink_z = shmem.sink_z;
	const auto* tree_nodes = data.tree_nodes;
	if (multlist.size()) {
		__syncwarp();
		for (int k = tid; k < nactive; k += WARP_SIZE) {
			expansion2<float> L;
			L(0, 0, 0) = L(1, 0, 0) = L(0, 1, 0) = L(0, 0, 1) = 0.0f;
			for (int j = 0; j < multlist.size(); j++) {
				array<float, NDIM> dx;
				const auto& pos = tree_nodes[multlist[j]].pos;
				const auto& M = tree_nodes[multlist[j]].multi;
				dx[XDIM] = distance(sink_x[k], pos[XDIM]);
				dx[YDIM] = distance(sink_y[k], pos[YDIM]);
				dx[ZDIM] = distance(sink_z[k], pos[ZDIM]);
				flops += 3;
				expansion<float> D;
				if (type == GRAVITY_DIRECT) {
					flops += greens_function(D, dx);
				} else {
					flops += ewald_greens_function(D, dx);
				}
				flops += M2L(L, M, D, do_phi);
			}
			gx[k] -= L(1, 0, 0);
			gy[k] -= L(0, 1, 0);
			gz[k] -= L(0, 0, 1);
			phi[k] += L(0, 0, 0);
		}
		__syncwarp();
	}
	__syncwarp();
	shared_reduce_add(flops);
	__syncwarp();
	return flops;

}

__device__
int cuda_gravity_pp(gravity_cc_type type, const cuda_kick_data& data, const tree_node& self, const fixedcapvec<int, PARTLIST_SIZE>& partlist, int nactive,
		float h, float dm_mass, float sph_mass, bool do_phi) {
	const int &tid = threadIdx.x;
	__shared__
	extern int shmem_ptr[];
	cuda_kick_shmem &shmem = *(cuda_kick_shmem*) shmem_ptr;
	auto &gx = shmem.gx;
	auto &gy = shmem.gy;
	auto &gz = shmem.gz;
	auto &phi = shmem.phi;
	const bool sph = data.sph;
	auto& src_hsoft = shmem.src.hsoft;
	const auto& sink_hsoft = shmem.sink_hsoft;
	const auto& sink_fpot = shmem.sink_fpot;
	const auto& sink_x = shmem.sink_x;
	const auto& sink_y = shmem.sink_y;
	const auto& sink_z = shmem.sink_z;
	const auto* main_src_x = data.x;
	const auto* main_src_y = data.y;
	const auto* main_src_z = data.z;
	const auto* main_src_fpot = data.fpot;
	const auto* main_src_hsoft = data.hsoft;
	auto& src_x = shmem.src.x;
	auto& src_y = shmem.src.y;
	auto& src_z = shmem.src.z;
	auto& src_fpot = shmem.src.fpot;
	const auto* tree_nodes = data.tree_nodes;
	float h2 = sqr(h);
	float hinv = 1.f / (h);
	float h3inv = hinv * hinv * hinv;
	int part_index;
	int nnear = 0;
	int nfar = 0;
	int flops = 0;
	if (partlist.size()) {
		int i = 0;
		auto these_parts = tree_nodes[partlist[0]].part_range;
		const auto partsz = partlist.size();
		while (i < partsz) {
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
				for (int j = tid; j < sz; j += WARP_SIZE) {
					const int i1 = part_index + j;
					const part_int i2 = j + imin;
					ASSERT(i2 >= 0);
					ASSERT(i2 < data.source_size);
					src_x[i1] = main_src_x[i2];
					src_y[i1] = main_src_y[i2];
					src_z[i1] = main_src_z[i2];
					src_fpot[i1] = main_src_fpot[i2];
					src_hsoft[i1] = main_src_hsoft[i2];
				}
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
			for (int k = tid; k < nactive; k += WARP_SIZE) {
				fx = 0.f;
				fy = 0.f;
				fz = 0.f;
				pot = 0.f;
				const float fpot_i = sink_fpot[k];
				const float h_i = sink_hsoft[k];
				const float h2_i = sqr(h_i);
				const float hinv_i = 1.0f / h_i;
				const float h3inv_i = hinv_i * sqr(hinv_i);
				for (int j = 0; j < part_index; j++) {
					dx0 = distance(sink_x[k], src_x[j]); // 1
					dx1 = distance(sink_y[k], src_y[j]); // 1
					dx2 = distance(sink_z[k], src_z[j]); // 1
					const float fpot_j = src_fpot[j];
					const float m_j = sph ? (fpot_j != 0.f ? sph_mass : dm_mass) : 1.f;
					const auto r2 = sqr(dx0, dx1, dx2);  // 5
					const float h_j = src_hsoft[j];
					const float h2_j = sqr(h_j);
					if (r2 > fmaxf(h2_i, h2_j)) {
						r1inv = rsqrt(r2);
						r3inv = sqr(r1inv) * r1inv;
					} else {
						const float hinv_j = 1.0f / h_j;
						const float h3inv_j = hinv_j * sqr(hinv_j);
						const float r = sqrtf(r2);
						r1inv = 1.f / (r + 1e-30f);
						const float q_i = r * hinv_i;
						const float q_j = r * hinv_j;
						const float F0 = 0.5f * (kernelFqinv(q_i) * h3inv_i + kernelFqinv(q_j) * h3inv_j);
						float Fc = 0.5f * (fpot_i * dkernelW_dq(q_i) * hinv_i * h3inv_i + fpot_j * dkernelW_dq(q_j) * hinv_j * h3inv_j) * r1inv;
						r3inv = F0 + Fc;
						//		PRINT( "%e\n", Fc/F0);
						if (do_phi) {
							const float pot0 = 0.5f * (kernelPot(q_i) * hinv_i + kernelPot(q_j) * hinv_j);
							const float potc = q_i > 0.0f ? 0.5f * (fpot_i * kernelW(q_i) * h3inv_i + fpot_j * kernelW(q_j) * h3inv_j) : 0.f;
							r1inv = pot0 + potc;
						}
					}
					r3inv *= m_j;
					r1inv *= m_j;
					flops += 2;
					fx = fmaf(dx0, r3inv, fx);                     // 2
					fy = fmaf(dx1, r3inv, fy);                     // 2
					fz = fmaf(dx2, r3inv, fz);                     // 2
					pot -= r1inv;                                  // 1
				}
				gx[k] -= fx;
				gy[k] -= fy;
				gz[k] -= fz;
				phi[k] += pot;
			}
		}

		__syncwarp();
	}
	shared_reduce_add(nnear);
	shared_reduce_add(nfar);
	shared_reduce_add(flops);
	__syncwarp();
	return nfar * 22 + 37 * nnear + flops;

}
