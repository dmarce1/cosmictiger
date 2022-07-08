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

#include <cosmictiger/lightcone.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/hpx.hpp>

#define BLOCK_SIZE 32

__global__ void cuda_lightcone_kernel(lc_part_map_type* part_map_ptr, lc_tree_map_type* tree_map_ptr, int* rc_ptr, const lc_tree_id* leaves, int N,
		int* index_ptr, lc_group* next_id_ptr, double link_len, int hpx_size, int hpx_rank) {
	const int& tid = threadIdx.x;
	__shared__ int index;
	const float link_len2 = sqr(link_len);
	auto& part_map = *part_map_ptr;
	auto& tree_map = *tree_map_ptr;
	if (tid == 0) {
		index = atomicAdd(index_ptr, 1);
	}
	const auto generate_id = [&next_id_ptr, hpx_rank, hpx_size]() {
		lc_group id = atomicAdd(next_id_ptr, 1);
		id *= hpx_size;
		id += hpx_rank + 1;
		return id;
	};
	__syncwarp();
	while (index < N) {
		const auto& this_leaf = leaves[index];
		const auto& self = tree_map[this_leaf.pix][this_leaf.index];
		if (self.last_active) {
			range<double> mybox = self.box.pad(1.01 * link_len);
			int found_any_link = false;
			const auto& neighbors = self.neighbors;
			auto& self_parts = part_map[self.pix];
			for (int li = 0; li < neighbors.size(); li++) {
				if (this_leaf != neighbors[li]) {
					const auto ni = neighbors[li];
					const auto& other = tree_map[ni.pix][ni.index];
					const auto& other_parts = part_map[other.pix];
					for (int pi = other.part_range.first; pi < other.part_range.second; pi++) {
						const auto B = other_parts.group[pi];
						array<double, NDIM> b;
						b[XDIM] = other_parts.pos[XDIM][pi].to_double();
						b[YDIM] = other_parts.pos[YDIM][pi].to_double();
						b[ZDIM] = other_parts.pos[ZDIM][pi].to_double();
						if (mybox.contains(b)) {
							for (int pj = self.part_range.first + tid; pj < self.part_range.second; pj += BLOCK_SIZE) {
								auto& A = self_parts.group[pj];
								const float dx = distance(self_parts.pos[XDIM][pj], other_parts.pos[XDIM][pi]);
								const float dy = distance(self_parts.pos[YDIM][pj], other_parts.pos[YDIM][pi]);
								const float dz = distance(self_parts.pos[ZDIM][pj], other_parts.pos[ZDIM][pi]);
								const float R2 = sqr(dx, dy, dz);
								if (R2 < link_len2) {
									if (A == LC_NO_GROUP) {
										if (B != LC_NO_GROUP) {
											A = B;
										} else {
											A = generate_id();
										}
										__threadfence();
										found_any_link = true;
									}
									if (A > B) {
										A = B;
										__threadfence();
										found_any_link = true;
									}
								}
							}
						}
					}
				}
			}

			__syncwarp();
			bool found_link;
			do {
				found_link = false;
				for (int pj = self.part_range.first; pj < self.part_range.second; pj++) {
					auto& A = self_parts.group[pj];
					lc_group min_group = LC_NO_GROUP;
					int this_found_link = 0;
					const auto maxpi = round_up(self.part_range.second - self.part_range.first, BLOCK_SIZE) + self.part_range.first;
					for (int pi = self.part_range.first + tid; pi < maxpi; pi += BLOCK_SIZE) {
						lc_group this_min_group = LC_NO_GROUP;
						if (pi != pj && pi < self.part_range.second) {
							const auto B = self_parts.group[pi];
							const float dx = distance(self_parts.pos[XDIM][pi], self_parts.pos[XDIM][pj]);
							const float dy = distance(self_parts.pos[YDIM][pi], self_parts.pos[YDIM][pj]);
							const float dz = distance(self_parts.pos[ZDIM][pi], self_parts.pos[ZDIM][pj]);
							const float R2 = sqr(dx, dy, dz);
							if (R2 < link_len2) {
								this_min_group = B;
								this_found_link++;
							}
						}
						shared_reduce_min(this_min_group);
						min_group = min(min_group, this_min_group);
					}
					shared_reduce_add(this_found_link);
					if (this_found_link && (A > min_group || A == LC_NO_GROUP)) {
						found_link = true;
						found_any_link = true;
						if (tid == 0) {
							A = min_group;
							if (A == LC_NO_GROUP) {
								A = generate_id();
							}
							__threadfence();
						}
					}
				}
			} while (found_link);
			shared_reduce_add(found_any_link);
			if (found_any_link) {
				for (int li = tid; li < neighbors.size(); li += BLOCK_SIZE) {
					const auto ni = neighbors[li];
					auto& other = tree_map[ni.pix][ni.index];
					other.active = 1;
				}
				if (tid == 0) {
					atomicAdd(rc_ptr, 1);
				}
			}
		}
		if (tid == 0) {
			index = atomicAdd(index_ptr, 1);
			//	PRINT( "%i %i\n", index, N);
		}
		__syncwarp();
	}
//	PRINT( "! %lli\n", *rc_ptr);
}

size_t cuda_lightcone(const device_vector<lc_tree_id>& leaves, lc_part_map_type* part_map_ptr, lc_tree_map_type* tree_map_ptr, lc_group* next_id_value) {
	int nblocks;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, (const void*) cuda_lightcone_kernel, BLOCK_SIZE, 0));
	nblocks *= cuda_smp_count();
	int* index_ptr;
	lc_group* next_id_ptr;
	int* rc_ptr;
	next_id_ptr = (lc_group*) cuda_malloc(sizeof(lc_group));
	index_ptr = (int*) cuda_malloc(sizeof(int));
	rc_ptr = (int*) cuda_malloc(sizeof(int));
	*rc_ptr = 0;
	*index_ptr = 0;
	*next_id_ptr = *next_id_value;
	const auto b = get_options().lc_b;
	const auto N = get_options().parts_dim;
	const double link_len = b / N;
	cuda_lightcone_kernel<<<nblocks,BLOCK_SIZE>>>(part_map_ptr, tree_map_ptr, rc_ptr, leaves.data(), leaves.size(), index_ptr, next_id_ptr, link_len, hpx_size(),hpx_rank());
	CUDA_CHECK(cudaDeviceSynchronize());
	PRINT("lc blocks = %i next id = %lli\n", nblocks, *next_id_ptr);
	size_t rc = *rc_ptr;
	*next_id_value = *next_id_ptr;
	cuda_free(rc_ptr);
	cuda_free(index_ptr);
	cuda_free(next_id_ptr);
	return rc;

}
