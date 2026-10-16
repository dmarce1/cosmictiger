/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

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

#include <cosmictiger/sph_cuda.hpp>

struct rungs_workspace {
	device_vector<fixed32> x;
	device_vector<fixed32> y;
	device_vector<fixed32> z;
	device_vector<char> rungs;
};

__global__ void sph_cuda_rungs(sph_run_params params, sph_run_cuda_data data, sph_reduction* reduce) {
	int const tid = threadIdx.x;
	int const block_size = blockDim.x;
	__shared__ int index;
	__shared__ rungs_workspace ws;
	if (tid == 0) {
		index = atomicAdd(&reduce->counter, 1);
	}
	__syncthreads();
	new (&ws) rungs_workspace();
	int flops = 0;
	array<fixed32, NDIM> x;
	int changes = 0;
	while (index < data.nselfs) {
		__syncthreads();
		ws.x.resize(0);
		ws.y.resize(0);
		ws.z.resize(0);
		ws.rungs.resize(0);
		sph_tree_node const& self = data.trees[data.selfs[index]];
		for (int ni = self.neighbor_range.first; ni < self.neighbor_range.second; ni++) {
			sph_tree_node const& other = data.trees[data.neighbors[ni]];
			int const maxpi = round_up(other.part_range.second - other.part_range.first, block_size) + other.part_range.first;
			for (int pi = other.part_range.first + tid; pi < maxpi; pi += block_size) {
				bool contains = false;
				int j;
				int total;
				int rung;
				if (pi < other.part_range.second) {
					rung = data.rungs[pi];
					if (rung >= params.min_rung) {
						x[XDIM] = data.x[pi];
						x[YDIM] = data.y[pi];
						x[ZDIM] = data.z[pi];
						if (self.outer_box.contains(x)) {
							contains = true;
						}
					}
				}
				j = contains;
				compute_indices<RUNGS_BLOCK_SIZE>(j, total);
				int const offset = ws.x.size();
				__syncthreads();
				int const next_size = offset + total;
				ws.x.resize(next_size);
				ws.y.resize(next_size);
				ws.z.resize(next_size);
				ws.rungs.resize(next_size);
				if (contains) {
					ASSERT(j < total);
					int const k = offset + j;
					ws.x[k] = x[XDIM];
					ws.y[k] = x[YDIM];
					ws.z[k] = x[ZDIM];
					ws.rungs[k] = rung;
				}
			}
		}
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			__syncthreads();
			if (data.rungs[i] >= params.min_rung) {
				int const snki = self.sink_part_range.first - self.part_range.first + i;
				auto const x_i = data.x[i];
				auto const y_i = data.y[i];
				auto const z_i = data.z[i];
				auto const h_i = data.rec2_snk[snki].h;
				char& rung_i = data.rungs[i];
				auto const h2_i = sqr(h_i);
				int const jmax = round_up(ws.x.size(), block_size);
				int max_rung_j = 0;
				for (int j = tid; j < jmax; j += block_size) {
					if (j < ws.x.size()) {
						auto const x_j = ws.x[j];
						auto const y_j = ws.y[j];
						auto const z_j = ws.z[j];
						int const rung_j = ws.rungs[j];
						float const x_ij = distance(x_i, x_j);	// 1
						float const y_ij = distance(y_i, y_j);	// 1
						float const z_ij = distance(z_i, z_j);	// 1
						float const r2 = sqr(x_ij, y_ij, z_ij); // 5
						if (r2 < h2_i) {						// 1
							max_rung_j = max(max_rung_j, rung_j);
						}
						flops += 9;
					}
				}
				shared_reduce_max<RUNGS_BLOCK_SIZE>(max_rung_j);
				if (tid == 0) {
					if (rung_i < max_rung_j - MAX_RUNG_DIF) {
						changes++;
						rung_i = max_rung_j - MAX_RUNG_DIF;
					}
				}
			}
		}
		shared_reduce_add<int, RUNGS_BLOCK_SIZE>(flops);
		if (tid == 0) {
			index = atomicAdd(&reduce->counter, 1);
			atomicAdd(&reduce->flops, flops);
		}
		flops = 0;
		__syncthreads();
	}
	if (tid == 0) {
		atomicAdd(&reduce->flag, changes);
	}
	(&ws)->~rungs_workspace();
}
