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

struct smoothlen_shmem {
	int index;
};

#include <cosmictiger/sph_cuda.hpp>
#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/timer.hpp>

#define WORKSPACE_SIZE (256*SPH_NEIGHBOR_COUNT)

struct smoothlen_workspace {
	fixedcapvec<fixed32, WORKSPACE_SIZE> x;
	fixedcapvec<fixed32, WORKSPACE_SIZE> y;
	fixedcapvec<fixed32, WORKSPACE_SIZE> z;
};

#define SMOOTHLEN_BLOCK_SIZE 512

struct sph_reduction {
	int counter;
	int flag;
	float hmin;
	float hmax;
	double flops;
};

__global__ void sph_cuda_smoothlen(sph_run_params params, sph_run_cuda_data data, smoothlen_workspace* workspaces, sph_reduction* reduce) {
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	const int block_size = blockDim.x;
	__shared__
	int index;
	__shared__
	float error;
	smoothlen_workspace& ws = workspaces[bid];
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	array<fixed32, NDIM> x;
	while (index < data.nselfs) {
		int flops = 0;
		ws.x.resize(0);
		ws.y.resize(0);
		ws.z.resize(0);
		const sph_tree_node& self = data.trees[data.selfs[index]];
		//	PRINT( "%i\n", self.neighbor_range.second - self.neighbor_range.first);
		for (int ni = self.neighbor_range.first; ni < self.neighbor_range.second; ni++) {
			//PRINT( "%i\n", -self.neighbor_range.first+self.neighbor_range.second);
			const sph_tree_node& other = data.trees[data.neighbors[ni]];
			const int maxpi = round_up(other.part_range.second - other.part_range.first, block_size) + other.part_range.first;
			for (int pi = other.part_range.first + tid; pi < maxpi; pi += block_size) {
				bool contains = false;
				int j;
				int total;
				if (pi < other.part_range.second) {
					x[XDIM] = data.x[pi];
					x[YDIM] = data.y[pi];
					x[ZDIM] = data.z[pi];
					if (self.outer_box.contains(x)) {
						contains = true;
					}
				}
				j = contains;
				compute_indices<SMOOTHLEN_BLOCK_SIZE>(j, total);
				const int offset = ws.x.size();
				const int next_size = offset + total;
				ws.x.resize(next_size);
				ws.y.resize(next_size);
				ws.z.resize(next_size);
				if (contains) {
					const int k = offset + j;
					ws.x[k] = x[XDIM];
					ws.y[k] = x[YDIM];
					ws.z[k] = x[ZDIM];
				}
			}
		}

		constexpr float A = float(float(21.0 * 2.0 / 3.0));
		constexpr float B = float(float(840.0 / 3.0));
		if (ws.x.size() == 0) {
			PRINT("ZERO\n");
		}
		float hmin = 1e+20;
		float hmax = 0.0;
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			if (data.rungs[i] >= params.min_rung) {
				const int snki = self.sink_part_range.first - self.part_range.first + i;
				x[XDIM] = data.x[i];
				x[YDIM] = data.y[i];
				x[ZDIM] = data.z[i];
				int count;
				float f;
				float dfdh;
				int box_xceeded = false;
				int iter = 0;
				float dh;
				float& h = data.h_snk[snki];
				do {
					const float hinv = 1.f / h; // 4
					const float h2 = sqr(h);    // 1
					count = 0;
					f = 0.f;
					dfdh = 0.f;
					for (int j = tid; j < ws.x.size(); j += block_size) {
						const float dx = distance(x[XDIM], ws.x[j]); // 2
						const float dy = distance(x[YDIM], ws.y[j]); // 2
						const float dz = distance(x[ZDIM], ws.z[j]); // 2
						const float r2 = sqr(dx, dy, dz);            // 2
						count += int(r2 < h2);                       // 1
						const float r = sqrt(r2);                    // 4
						const float q = r * hinv;                    // 1
						flops += 15;
						if (q < 1.f) {                               // 1
							const float q2 = sqr(q);                  // 1
							const float _1mq = 1.f - q;               // 1
							const float _1mq2 = sqr(_1mq);            // 1
							const float _1mq3 = _1mq * _1mq2;         // 1
							const float _1mq4 = _1mq * _1mq3;         // 1
							const float w = A * _1mq4 * fmaf(4.f, q, 1.f); // 4
							const float dwdh = B * _1mq3 * q2 * hinv; // 3
							f += w;                                   // 1
							dfdh += dwdh;                             // 1
							flops += 15;
						}
					}
					shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(f);
					shared_reduce_add<int, SMOOTHLEN_BLOCK_SIZE>(count);
					shared_reduce_add<float, SMOOTHLEN_BLOCK_SIZE>(dfdh);
					if (tid == 0) {
						dh = 0.1f * h;
						if (count > 1) {
							f -= SPH_NEIGHBOR_COUNT;
							dh = -f / dfdh;
							if (dh > 0.5f * h) {
								dh = 0.5f * h;
							} else if (dh < -0.5f * h) {
								dh = -0.5f * h;
							}
							error = fabsf(logf(h + dh) - logf(h));
							if (tid == 0) {
								h += dh;
							}
						} else {
							if (count == 0) {
								PRINT("Can't find self\n");
							}
							if (tid == 0) {
								h *= 1.1;
							}
							error = 1.0;
						}
					}
					__syncthreads();
					for (int dim = 0; dim < NDIM; dim++) {
						if (distance(self.outer_box.end[dim], x[dim]) + h < 0.0f) {
							box_xceeded = true;
							break;
						}
						if (distance(x[dim], self.outer_box.begin[dim]) + h < 0.0f) {
							box_xceeded = true;
							break;
						}
					}
					iter++;
					//			if( tid == 0 )
					//			PRINT("%i %i %e %e\n", iter, count, h, dh);
					if (iter > 20) {
						PRINT("density solver failed to converge\n");
						__trap();
					}
				} while (error > SPH_SMOOTHLEN_TOLER && !box_xceeded);
				hmin = fminf(hmin, h);
				hmax = fmaxf(hmax, h);
				if (tid == 0) {
					if (box_xceeded) {
						atomicAdd(&reduce->flag, 1);
					}
				}
			}
		}
		shared_reduce_add<int, SMOOTHLEN_BLOCK_SIZE>(flops);
		if (tid == 0) {
			atomicAdd(&reduce->flops, (double) flops);
			atomicMax(&reduce->hmax, hmax);
			atomicMin(&reduce->hmin, hmin);
			index = atomicAdd(&reduce->counter, 1);
		}
		__syncthreads();
	}

}

sph_run_return sph_run_cuda(sph_run_params params, sph_run_cuda_data data, cudaStream_t stream) {
	timer tm;
	sph_run_return rc;
	sph_reduction* reduce;
	int nblocks;
	CUDA_CHECK(cudaMallocManaged(&reduce, sizeof(sph_reduction)));
	reduce->counter = reduce->flag = 0;
	reduce->hmin = std::numeric_limits<float>::max();
	reduce->hmax = 0.0f;
	reduce->flops = 0.0;
	switch (params.run_type) {
	case SPH_RUN_SMOOTHLEN: {
		smoothlen_workspace* workspaces;
		CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, (const void*) sph_cuda_smoothlen, SMOOTHLEN_BLOCK_SIZE, sizeof(smoothlen_shmem)));
		nblocks *= cuda_smp_count();
		CUDA_CHECK(cudaMalloc(&workspaces, sizeof(smoothlen_workspace) * nblocks));
		tm.start();
		sph_cuda_smoothlen<<<nblocks, SMOOTHLEN_BLOCK_SIZE>>>(params,data,workspaces,reduce);
		cuda_stream_synchronize(stream);
		tm.stop();
		CUDA_CHECK(cudaFree(workspaces));
		rc.rc = reduce->flag;
		rc.hmin = reduce->hmin;
		rc.hmax = reduce->hmax;
		tm.stop();
		PRINT("Kernel ran at %e GFLOPS\n", reduce->flops / (1024 * 1024 * 1024) / tm.read());
	}
		break;
	}
	CUDA_CHECK (cudaFree(reduce));

	return rc;
}
