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
#define SOFTLENS_BLOCK_SIZE 256
#define DERIVATIVES_BLOCK_SIZE 256

#include <cosmictiger/all_tree.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/kernel.hpp>
#include <cosmictiger/cuda_mem.hpp>
#include <cosmictiger/cuda_reduce.hpp>

struct softlens_record {
	fixed32 x;
	fixed32 y;
	fixed32 z;
	char type;
};

struct softlens_workspace {
	device_vector<softlens_record> rec;
};

struct all_tree_reduction {
	int counter;
	int flag;
	float hmin;
	float hmax;
	float flops;
};

__global__ void cuda_softlens(all_tree_data params, all_tree_reduction* reduce) {
	const int tid = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	__shared__ softlens_workspace ws;
	__syncthreads();
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	new (&ws) softlens_workspace();
	array<fixed32, NDIM> x;
	while (index < params.nselfs) {

		int flops = 0;
		__syncthreads();
		ws.rec.resize(0);
		const tree_node& self = params.trees[params.selfs[index]];
		for (int ni = self.neighbor_range.first; ni < self.neighbor_range.second; ni++) {
			const tree_node& other = params.trees[params.neighbors[ni]];
			const int maxpi = round_up(other.part_range.second - other.part_range.first, block_size) + other.part_range.first;
			for (int pi = other.part_range.first + tid; pi < maxpi; pi += block_size) {
				bool contains = false;
				int j;
				int total;
				if (pi < other.part_range.second) {
					x[XDIM] = params.x[pi];
					x[YDIM] = params.y[pi];
					x[ZDIM] = params.z[pi];
					contains = (self.obox.contains(x));
				}
				j = contains;
				compute_indices < SOFTLENS_BLOCK_SIZE > (j, total);
				const int offset = ws.rec.size();
				__syncthreads();
				const int next_size = offset + total;
				ws.rec.resize(next_size);
				if (contains) {
					const int k = offset + j;
					ws.rec[k].x = x[XDIM];
					ws.rec[k].y = x[YDIM];
					ws.rec[k].z = x[ZDIM];
					ws.rec[k].type = params.types[pi];
				}
			}
		}
		float hmin_all = 1e+20f;
		float hmax_all = 0.0;
		const float w0 = kernelW(0.f);
		__syncthreads();
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			__syncthreads();
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			const auto rung = params.rung_snk[snki];
			const bool active = rung >= params.minrung;
			auto& converged = params.converged_snk[snki];
			const bool use = active && !converged;
			if (use) {
				x[XDIM] = params.x[i];
				x[YDIM] = params.y[i];
				x[ZDIM] = params.z[i];
				const auto type_i = params.types[i];
				float& h = params.softlen_snk[snki];
				float h0 = params.hmin;
				const auto test = [h0,x,block_size,tid,&params,type_i](float h) {
					float w = 0.f;
					const float hinv = 1.f / h;
					for (int j = tid; j < ws.rec.size(); j += block_size) {
						const auto type_j = ws.rec[j].type;
						if( type_i == type_j ) {
							const float dx = distance(x[XDIM], ws.rec[j].x);
							const float dy = distance(x[YDIM], ws.rec[j].y);
							const float dz = distance(x[ZDIM], ws.rec[j].z);
							const float r2 = sqr(dx, dy, dz);
							const float r = sqrt(r2);
							const float q = r * hinv;
							if (q < 1.f) {
								w += kernelW(q);
							}
						}
					}
					shared_reduce_add<float, SOFTLENS_BLOCK_SIZE>(w);
					return (4.0f * float(M_PI) / 3.0f) * smoothX(h,h0) * w - params.N;
				};
				float hmax = 1.0f;
				for (int dim = 0; dim < NDIM; dim++) {
					hmax = fminf(hmax, fabs(distance(self.obox.begin[dim], x[dim])));
					hmax = fminf(hmax, fabs(distance(self.obox.end[dim], x[dim])));
				}
				float hmin = h0;
				const float hmax0 = hmax;
				const float hmin0 = hmin;
				float test_max = test(hmax);
				float hmid;
				int iters = 0;
				if (hmax > hmin) {
					do {
						hmid = 0.5f * hmax + 0.5f * hmin;
						const float test_mid = test(hmid);
						if (test_mid * test_max < 0.f) {
							hmin = hmid;
						} else {
							hmax = hmid;
							test_max = test_mid;
						}
						iters++;
					} while (hmax > 1.001f * hmin);
				}
				if (hmax <= hmin || hmin == hmin0 || hmax == hmax0) {
					if (tid == 0) {
						atomicAdd(&reduce->flag, 1);
						h = hmax0;
					}
				} else {
					if (tid == 0) {
						h = hmid;
						converged = true;
						hmin_all = fminf(hmin_all, h);
						hmax_all = fmaxf(hmax_all, h);
					}
				}
			}
		}
		shared_reduce_add<int, SOFTLENS_BLOCK_SIZE>(flops);
		if (tid == 0) {
			atomicMax(&reduce->hmax, hmax_all);
			atomicMin(&reduce->hmin, hmin_all);
			atomicAdd(&reduce->flops, (double) flops);
			index = atomicAdd(&reduce->counter, 1);
		}
		flops = 0;
		__syncthreads();
	}
	(&ws)->~softlens_workspace();
}

struct derivatives_record1 {
	fixed32 x;
	fixed32 y;
	fixed32 z;
	float h;
	char type;
};

struct derivatives_workspace {
	device_vector<derivatives_record1> rec1;
};

__global__ void cuda_derivatives(all_tree_data params, all_tree_reduction* reduce) {
	const int tid = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	__shared__ derivatives_workspace ws;
	__syncthreads();
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	new (&ws) derivatives_workspace();
	array<fixed32, NDIM> x;
	while (index < params.nselfs) {
		int flops = 0;
		__syncthreads();
		ws.rec1.resize(0);
		const tree_node& self = params.trees[params.selfs[index]];
		for (int ni = self.neighbor_range.first; ni < self.neighbor_range.second; ni++) {
			const tree_node& other = params.trees[params.neighbors[ni]];
			const int maxpi = round_up(other.part_range.second - other.part_range.first, block_size) + other.part_range.first;
			for (int pi = other.part_range.first + tid; pi < maxpi; pi += block_size) {
				bool contains = false;
				int j;
				int total;
				if (pi < other.part_range.second) {
					x[XDIM] = params.x[pi];
					x[YDIM] = params.y[pi];
					x[ZDIM] = params.z[pi];
					contains = (self.obox.contains(x));
					if (!contains) {
						contains = true;
						const float& h = params.h[pi];
						for (int dim = 0; dim < NDIM; dim++) {
							if (distance(x[dim], self.ibox.begin[dim]) + h < 0.f) {
								flops += 3;
								contains = false;
								break;
							}
							if (distance(self.ibox.end[dim], x[dim]) + h < 0.f) {
								flops += 3;
								contains = false;
								break;
							}
						}
					}
				}
				j = contains;
				compute_indices < DERIVATIVES_BLOCK_SIZE > (j, total);
				const int offset = ws.rec1.size();
				__syncthreads();
				const int next_size = offset + total;
				ws.rec1.resize(next_size);
				if (contains) {
					const int k = offset + j;
					ws.rec1[k].x = x[XDIM];
					ws.rec1[k].y = x[YDIM];
					ws.rec1[k].z = x[ZDIM];
					ws.rec1[k].h = params.h[pi];
					ws.rec1[k].type = params.types[pi];
				}
			}
		}
		__syncthreads();
		float hmin_all = 1e+20;
		float hmax_all = 0.0;
		const float w0 = kernelW(0.f);
		__syncthreads();
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			__syncthreads();
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			const auto rung = params.rung_snk[snki];
			const bool active = rung >= params.minrung;
			int semiactive = 0;
			x[XDIM] = params.x[i];
			x[YDIM] = params.y[i];
			x[ZDIM] = params.z[i];
			const auto& type_i = params.types[i];
			const fixed32& x_i = x[XDIM];
			const fixed32& y_i = x[YDIM];
			const fixed32& z_i = x[ZDIM];
			const float& h_i = params.softlen_snk[snki];
			const float hinv_i = 1.f / h_i; 										// 4
			const float h2_i = sqr(h_i);    										// 1
			auto& sa_snk = params.sa_snk[snki];
			if (active) {
				sa_snk = true;
			} else {
				sa_snk = false;
				const int jmax = round_up(ws.rec1.size(), block_size);
				for (int j = tid; j < jmax; j += block_size) {
					if (j < ws.rec1.size()) {
						const auto& rec1 = ws.rec1[j];
						const auto& x_j = rec1.x;
						const auto& y_j = rec1.y;
						const auto& z_j = rec1.z;
						const auto& h_j = rec1.h;
						const auto h2_j = sqr(h_j);									// 1
						const float x_ij = distance(x_i, x_j);						// 1
						const float y_ij = distance(y_i, y_j);						// 1
						const float z_ij = distance(z_i, z_j);						// 1
						const float r2 = sqr(x_ij, y_ij, z_ij);					// 5
						if (r2 < fmaxf(h2_i, h2_j)) {									// 2
							semiactive++;
						}
						flops += 11;
					}
					shared_reduce_add<int, DERIVATIVES_BLOCK_SIZE>(semiactive);
					if (semiactive) {
						sa_snk = true;
						break;
					}
				}
			}
			int box_xceeded = false;
			if (semiactive) {
				auto& converged = params.converged_snk[snki];
				const auto type_i = params.types[i];
				float& h = params.softlen_snk[snki];
				float h0 = params.hmin;
				const auto test = [h0,x,block_size,tid,&params,type_i](float h) {
					float w = 0.f;
					const float hinv = 1.f / h;
					for (int j = tid; j < ws.rec1.size(); j += block_size) {
						const auto type_j = ws.rec1[j].type;
						if( type_i == type_j ) {
							const float dx = distance(x[XDIM], ws.rec1[j].x);
							const float dy = distance(x[YDIM], ws.rec1[j].y);
							const float dz = distance(x[ZDIM], ws.rec1[j].z);
							const float r2 = sqr(dx, dy, dz);
							const float r = sqrt(r2);
							const float q = r * hinv;
							if (q < 1.f) {
								w += kernelW(q);
							}
						}
					}
					shared_reduce_add<float, SOFTLENS_BLOCK_SIZE>(w);
					return (4.0f * float(M_PI) / 3.0f) * smoothX(h,h0) * w - params.N;
				};
				float hmax = 1.0f;
				for (int dim = 0; dim < NDIM; dim++) {
					hmax = fminf(hmax, fabs(distance(self.obox.begin[dim], x[dim])));
					hmax = fminf(hmax, fabs(distance(self.obox.end[dim], x[dim])));
				}
				float hmin = h0;
				const float hmax0 = hmax;
				const float hmin0 = hmin;
				float test_max = test(hmax);
				float hmid;
				int iters = 0;
				if (hmax > hmin) {
					do {
						hmid = 0.5f * hmax + 0.5f * hmin;
						const float test_mid = test(hmid);
						if (test_mid * test_max < 0.f) {
							hmin = hmid;
						} else {
							hmax = hmid;
							test_max = test_mid;
						}
						iters++;
					} while (hmax > 1.001f * hmin);
				}
				if (hmax <= hmin || hmin == hmin0 || hmax == hmax0) {
					if (tid == 0) {
						atomicAdd(&reduce->flag, 1);
						h = hmax0;
					}
				} else {
					if (tid == 0) {
						h = hmid;
						converged = true;
						hmin_all = fminf(hmin_all, h);
						hmax_all = fmaxf(hmax_all, h);
					}
				}
			}
			if (active || (semiactive && !box_xceeded)) {
				__syncthreads();
				const int snki = self.sink_part_range.first - self.part_range.first + i;
				const float m_i = params.sph ? (type_i == DARK_MATTER_TYPE ? params.dm_mass : params.sph_mass) : 1.f;
				const float& h_i = params.softlen_snk[snki];
				const fixed32& x_i = x[XDIM];
				const fixed32& y_i = x[YDIM];
				const fixed32& z_i = x[ZDIM];
				const float hinv_i = 1.f / h_i; 										// 4
				const float h3inv_i = sqr(hinv_i) * hinv_i;
				const float h2_i = sqr(h_i);    										// 1
				float w_sum = 0.f;
				float dw_sum = 0.f;
				float dpot_dh = 0.f;
				__syncthreads();
				flops += 10;
				const int jmax = round_up(ws.rec1.size(), DERIVATIVES_BLOCK_SIZE);
				for (int j = tid; j < jmax; j += block_size) {
					if (j < ws.rec1.size()) {
						const auto& rec1 = ws.rec1[j];
						const auto& type_j = rec1.type;
						const fixed32& x_j = rec1.x;
						const fixed32& y_j = rec1.y;
						const fixed32& z_j = rec1.z;
						const float x_ij = distance(x_i, x_j); // 1
						const float y_ij = distance(y_i, y_j); // 1
						const float z_ij = distance(z_i, z_j); // 1
						const float r2 = sqr(x_ij, y_ij, z_ij);
						const float r = sqrtf(r2);                    // 4
						const float rinv = 1.0f / (r + 1e-37f);
						const float q = r * hinv_i;                    // 1
						if (q < 1.f) {                               // 1
							if (type_i == type_j) {
								const float w = kernelW(q);
								const float dwdq = dkernelW_dq(q);
								dw_sum -= q * dwdq;                      // 2
								w_sum += w;
							}
							const float m_j = params.sph ? (type_j == DARK_MATTER_TYPE ? params.dm_mass : params.sph_mass) : 1.f;
							const float pot = -kernelPot(q);
							const float force = kernelFqinv(q) * q;
							dpot_dh += m_j * (pot + q * force) / m_i;
							flops += 2;
						}
						flops += 9;
					}
				}
				shared_reduce_add<float, DERIVATIVES_BLOCK_SIZE>(w_sum);
				shared_reduce_add<float, DERIVATIVES_BLOCK_SIZE>(dw_sum);
				shared_reduce_add<float, DERIVATIVES_BLOCK_SIZE>(dpot_dh);
				const float A = 0.33333333333f * dw_sum / w_sum;
				float f, dfdh;
				dsmoothX_dh(h_i, params.hmin, f, dfdh);
				const float B = 0.33333333333f * h_i / f * dfdh;
				const float omega = (A + B) / (1.0f + B);
				const float zeta_i = dpot_dh / (omega * (3.f + dfdh * h_i / f) * w_sum);
				__syncthreads();
				if (tid == 0) {
					params.zeta_snk[snki] = zeta_i;
				}
			}
		}
		shared_reduce_add<int, DERIVATIVES_BLOCK_SIZE>(flops);
		if (tid == 0) {
			atomicMax(&reduce->hmax, hmax_all);
			atomicMin(&reduce->hmin, hmin_all);
			atomicAdd(&reduce->flops, (double) flops);
			index = atomicAdd(&reduce->counter, 1);
		}
		flops = 0;
		__syncthreads();
	}
	(&ws)->~derivatives_workspace();
}

struct divv_record1 {
	fixed32 x;
	fixed32 y;
	fixed32 z;
	char type;
};

struct divv_record2 {
	float vx;
	float vy;
	float vz;
};

struct divv_workspace {
	device_vector<divv_record1> rec1;
	device_vector<divv_record2> rec2;
};

__global__ void cuda_divv(all_tree_data params, all_tree_reduction* reduce) {
	const int tid = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	__shared__ divv_workspace ws;
	__syncthreads();
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	new (&ws) divv_workspace();
	array<fixed32, NDIM> x;
	while (index < params.nselfs) {
		int flops = 0;
		__syncthreads();
		ws.rec1.resize(0);
		ws.rec2.resize(0);
		const tree_node& self = params.trees[params.selfs[index]];
		for (int ni = self.neighbor_range.first; ni < self.neighbor_range.second; ni++) {
			const tree_node& other = params.trees[params.neighbors[ni]];
			const int maxpi = round_up(other.part_range.second - other.part_range.first, block_size) + other.part_range.first;
			for (int pi = other.part_range.first + tid; pi < maxpi; pi += block_size) {
				bool contains = false;
				int j;
				int total;
				if (pi < other.part_range.second) {
					x[XDIM] = params.x[pi];
					x[YDIM] = params.y[pi];
					x[ZDIM] = params.z[pi];
					contains = (self.obox.contains(x));
				}
				j = contains;
				compute_indices < DERIVATIVES_BLOCK_SIZE > (j, total);
				const int offset = ws.rec1.size();
				__syncthreads();
				const int next_size = offset + total;
				ws.rec1.resize(next_size);
				ws.rec2.resize(next_size);
				if (contains) {
					const int k = offset + j;
					ws.rec1[k].x = x[XDIM];
					ws.rec1[k].y = x[YDIM];
					ws.rec1[k].z = x[ZDIM];
					ws.rec1[k].type = params.types[pi];
					ws.rec2[k].vx = params.vx[pi];
					ws.rec2[k].vy = params.vy[pi];
					ws.rec2[k].vz = params.vz[pi];
				}
			}
		}
		__syncthreads();
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			__syncthreads();
			const int snki = self.sink_part_range.first - self.part_range.first + i;
			const auto rung = params.rung_snk[snki];
			const bool active = rung >= params.minrung;
			x[XDIM] = params.x[i];
			x[YDIM] = params.y[i];
			x[ZDIM] = params.z[i];
			const fixed32& x_i = x[XDIM];
			const fixed32& y_i = x[YDIM];
			const fixed32& z_i = x[ZDIM];
			if (active) {
				const float& h_i = params.softlen_snk[snki];
				const fixed32& x_i = x[XDIM];
				const fixed32& y_i = x[YDIM];
				const fixed32& z_i = x[ZDIM];
				const auto& type_i = params.types[i];
				const float& vx_i = params.vx[i];
				const float& vy_i = params.vy[i];
				const float& vz_i = params.vz[i];
				const float hinv_i = 1.f / h_i;
				float dw_sum = 0.f;
				float w_sum = 0.f;
				float rho = 0.f;
				const int jmax = round_up(ws.rec1.size(), DERIVATIVES_BLOCK_SIZE);
				float divv = 0.f;
				for (int j = tid; j < jmax; j += block_size) {
					if (j < ws.rec1.size()) {
						const auto& rec1 = ws.rec1[j];
						const auto& rec2 = ws.rec2[j];
						const fixed32& x_j = rec1.x;
						const fixed32& y_j = rec1.y;
						const fixed32& z_j = rec1.z;
						const auto& type_j = rec1.type;
						const float& vx_j = rec2.vx;
						const float& vy_j = rec2.vy;
						const float& vz_j = rec2.vz;
						const float vx_ij = vx_i - vx_j;
						const float vy_ij = vy_i - vy_j;
						const float vz_ij = vz_i - vz_j;
						const float x_ij = distance(x_i, x_j); // 1
						const float y_ij = distance(y_i, y_j); // 1
						const float z_ij = distance(z_i, z_j); // 1
						const float r2 = sqr(x_ij, y_ij, z_ij);
						const float r = sqrtf(r2);                    // 4
						const float rinv = 1.0f / (r + 1e-37f);
						const float q = r * hinv_i;                    // 1
						if (q < 1.f && (type_i == type_j)) {                               // 1
							const float w = kernelW(q);
							const float dwdq = dkernelW_dq(q);
							dw_sum -= q * dwdq;                      // 2
							w_sum += w;
							divv += (vx_ij * x_ij + vy_ij * y_ij + vz_ij * z_ij) * rinv * dwdq;
						}
					}
				}
				shared_reduce_add<float, DERIVATIVES_BLOCK_SIZE>(w_sum);
				shared_reduce_add<float, DERIVATIVES_BLOCK_SIZE>(dw_sum);
				shared_reduce_add<float, DERIVATIVES_BLOCK_SIZE>(rho);
				shared_reduce_add<float, DERIVATIVES_BLOCK_SIZE>(divv);
				const float A = 0.33333333333f * dw_sum / w_sum;
				float f, dfdh;
				dsmoothX_dh(h_i, params.hmin, f, dfdh);
				const float B = 0.33333333333f * h_i / f * dfdh;
				const float omega = (A + B) / (1.0f + B);
				divv /= omega * params.a;				  // 4
				__syncthreads();
				if (tid == 0) {
					params.divv_snk[snki] = f * divv;
				}
			}
		}
		shared_reduce_add<int, DERIVATIVES_BLOCK_SIZE>(flops);
		if (tid == 0) {
			atomicAdd(&reduce->flops, (double) flops);
			index = atomicAdd(&reduce->counter, 1);
		}
		flops = 0;
		__syncthreads();
	}
	(&ws)->~divv_workspace();
}

softlens_return all_tree_softlens_cuda(all_tree_data params, cudaStream_t stream) {
	softlens_return rc;
	all_tree_reduction* reduce;
	CUDA_CHECK(cudaMallocManaged(&reduce, sizeof(all_tree_reduction)));
	reduce->counter = reduce->flag = 0;
	reduce->hmin = std::numeric_limits<float>::max();
	reduce->hmax = 0.0f;
	reduce->flops = 0.0;
	static int softlens_nblocks;
	static bool first = true;
	timer tm;
	if (first) {
		first = false;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&softlens_nblocks, (const void*) cuda_softlens, SOFTLENS_BLOCK_SIZE, 0));
		softlens_nblocks *= cuda_smp_count();
	}
	tm.start();
	cuda_softlens<<<softlens_nblocks, SOFTLENS_BLOCK_SIZE,0,stream>>>(params,reduce);
	cuda_stream_synchronize(stream);
	rc.fail = reduce->flag;
	rc.hmin = reduce->hmin;
	rc.hmax = reduce->hmax;
	tm.stop();
	rc.flops = reduce->flops;
	CUDA_CHECK(cudaFree(reduce));
	return rc;

}

softlens_return all_tree_divv_cuda(all_tree_data params, cudaStream_t stream) {
	softlens_return rc;
	all_tree_reduction* reduce;
	CUDA_CHECK(cudaMallocManaged(&reduce, sizeof(all_tree_reduction)));
	reduce->counter = reduce->flag = 0;
	reduce->hmin = std::numeric_limits<float>::max();
	reduce->hmax = 0.0f;
	reduce->flops = 0.0;
	static int divv_nblocks;
	static bool first = true;
	timer tm;
	if (first) {
		first = false;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&divv_nblocks, (const void*) cuda_divv, DERIVATIVES_BLOCK_SIZE, 0));
		divv_nblocks *= cuda_smp_count();
	}
	tm.start();
	cuda_divv<<<divv_nblocks, DERIVATIVES_BLOCK_SIZE,0,stream>>>(params,reduce);
	cuda_stream_synchronize(stream);
	rc.fail = reduce->flag;
	rc.hmin = reduce->hmin;
	rc.hmax = reduce->hmax;
	tm.stop();
	rc.flops = reduce->flops;
	CUDA_CHECK(cudaFree(reduce));
	return rc;

}

softlens_return all_tree_derivatives_cuda(all_tree_data params, cudaStream_t stream) {
	softlens_return rc;
	all_tree_reduction* reduce;
	CUDA_CHECK(cudaMallocManaged(&reduce, sizeof(all_tree_reduction)));
	reduce->counter = reduce->flag = 0;
	reduce->hmin = std::numeric_limits<float>::max();
	reduce->hmax = 0.0f;
	reduce->flops = 0.0;
	params.dm_mass = get_options().dm_mass;
	params.sph_mass = get_options().sph_mass;
	params.sph = get_options().sph;
	static int derivatives_nblocks;
	static bool first = true;
	timer tm;
	if (first) {
		first = false;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&derivatives_nblocks, (const void*) cuda_derivatives, DERIVATIVES_BLOCK_SIZE, 0));
		derivatives_nblocks *= cuda_smp_count();
	}
	tm.start();
	cuda_derivatives<<<derivatives_nblocks, DERIVATIVES_BLOCK_SIZE,0,stream>>>(params,reduce);
	cuda_stream_synchronize(stream);
	rc.fail = reduce->flag;
	rc.hmin = reduce->hmin;
	rc.hmax = reduce->hmax;
	tm.stop();
	rc.flops = reduce->flops;
	CUDA_CHECK(cudaFree(reduce));
	return rc;

}
