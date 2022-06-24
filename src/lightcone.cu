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

static cuda_unordered_map<host_vector<lc_particle>>*& part_map_ptr() {
	static cuda_unordered_map<host_vector<lc_particle>>* a = nullptr;
	return a;
}

static cuda_unordered_map<host_vector<lc_tree_node>>*& tree_map_ptr() {
	static cuda_unordered_map<host_vector<lc_tree_node>>* a = nullptr;
	return a;
}

static __managed__ cuda_unordered_map<host_vector<lc_particle>>* dev_part_map_ptr;
static __managed__ cuda_unordered_map<host_vector<lc_tree_node>>* dev_tree_map_ptr;

cuda_unordered_map<host_vector<lc_particle>>& get_part_map() {
	if (part_map_ptr() == nullptr) {
		CUDA_CHECK(cudaMallocManaged(&part_map_ptr(), sizeof(cuda_unordered_map<host_vector<lc_particle>> )));
		new (part_map_ptr()) cuda_unordered_map<host_vector<lc_particle>>;
	}
	return *part_map_ptr();
}

cuda_unordered_map<host_vector<lc_tree_node>>& get_tree_map() {
	if (tree_map_ptr == nullptr) {
		CUDA_CHECK(cudaMallocManaged(&tree_map_ptr(), sizeof(cuda_unordered_map<host_vector<lc_tree_node>> )));
		new (part_map_ptr()) cuda_unordered_map<host_vector<lc_tree_node>>;
	}
	return *tree_map_ptr();
}

__global__ void cuda_lightcone_kernel(unsigned long long* rc_ptr, const lc_tree_id* leaves, int N, int* index_ptr, unsigned long long* next_id_ptr,
		double link_len, int hpx_size, int hpx_rank) {
	const int& tid = threadIdx.x;
	__shared__ int index;
	const float link_len2 = sqr(link_len);
	auto& part_map = *dev_part_map_ptr;
	auto& tree_map = *dev_tree_map_ptr;
	if (tid == 0) {
		index = atomicAdd(index_ptr, 1);
	}
	const auto generate_id = [&next_id_ptr, hpx_rank, hpx_size]() {
		long long id = atomicAdd(next_id_ptr, 1);
		id *= hpx_size;
		id += hpx_rank + 1;
		return id;
	};
	__syncthreads();
	while (index < N) {
		const auto& this_leaf = leaves[index];
		const auto& self = tree_map[this_leaf.pix][this_leaf.index];
		if (self.last_active) {
			const auto& mybox = self.box;
			int found_any_link = false;
			const auto& neighbors = self.neighbors;
			for (int li = 0; li < neighbors.size(); li++) {
				if (this_leaf != neighbors[li]) {
					const auto ni = neighbors[li];
					const auto& other = tree_map[ni.pix][ni.index];
					for (int pi = other.part_range.first; pi < other.part_range.second; pi++) {
						const auto& B = part_map[other.pix][pi];
						if (mybox.contains(B.pos)) {
							for (int pj = self.part_range.first + tid; pj < self.part_range.second; pj += BLOCK_SIZE) {
								auto& A = part_map[self.pix][pj];
								const float dx = distance(A.pos[XDIM], B.pos[XDIM]);
								const float dy = distance(A.pos[YDIM], B.pos[YDIM]);
								const float dz = distance(A.pos[ZDIM], B.pos[ZDIM]);
								const float R2 = sqr(dx, dy, dz);
								if (R2 < link_len2) {
									if (A.group == LC_NO_GROUP) {
										A.group = generate_id();
										__threadfence();
										found_any_link = true;
									}
									if (A.group > B.group) {
										A.group = B.group;
										__threadfence();
										found_any_link = true;
									}
								}
							}
						}
					}
				}
			}
			__syncthreads();
			int found_link;
			do {
				found_link = false;
				for (int pj = self.part_range.first; pj < self.part_range.second; pj++) {
					auto& A = part_map[self.pix][pj];
					unsigned long long min_group = LC_NO_GROUP;
					int this_found_link = false;
					const auto maxpi = round_up(self.part_range.second, BLOCK_SIZE);
					for (int pi = self.part_range.first + tid; pi < self.part_range.second; pi += maxpi) {
						unsigned long long this_min_group = LC_NO_GROUP;
						if (pi != pj && pi < self.part_range.second) {
							auto& B = part_map[self.pix][pi];
							const float dx = distance(A.pos[XDIM], B.pos[XDIM]);
							const float dy = distance(A.pos[YDIM], B.pos[YDIM]);
							const float dz = distance(A.pos[ZDIM], B.pos[ZDIM]);
							const float R2 = sqr(dx, dy, dz);
							if (R2 < link_len2) {
								min_group = B.group;
								this_found_link = true;
							}
						}
						shared_reduce_min < BLOCK_SIZE > (this_min_group);
						min_group = min(min_group, this_min_group);
					}
					shared_reduce_add<int, BLOCK_SIZE>(this_found_link);
					if (this_found_link) {
						found_link = true;
						shared_reduce_min < BLOCK_SIZE > (min_group);
						if (tid == 0) {
							A.group = min_group;
							if (A.group == LC_NO_GROUP && this_found_link) {
								A.group = generate_id();
							}
						}
						__syncthreads();
					}
				}
				shared_reduce_add<int, BLOCK_SIZE>(found_link);
			} while (found_link);
			shared_reduce_add<int, BLOCK_SIZE>(found_any_link);
			if (found_any_link) {
				for (int li = tid; li < neighbors.size(); li += BLOCK_SIZE) {
					const auto ni = neighbors[li];
					auto& other = tree_map[ni.pix][ni.index];
					atomicAdd((int*) &other.active, 1);
				}
				if (tid == 0) {
					atomicAdd(rc_ptr, 1);
				}
			}
		}
		if (tid == 0) {
			index = atomicAdd(index_ptr, 1);
		}
		__syncthreads();
	}

}

size_t cuda_lightcone(const host_vector<lc_tree_id>& leaves) {
	int nblocks;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, (const void*) cuda_lightcone_kernel, BLOCK_SIZE, 0));
	nblocks *= cuda_smp_count();
	int* index_ptr;
	unsigned long long* next_id_ptr;
	unsigned long long* rc_ptr;
	index_ptr = (int*) cuda_malloc(sizeof(int));
	next_id_ptr = (unsigned long long*) cuda_malloc(sizeof(unsigned long long));
	rc_ptr = (unsigned long long*) cuda_malloc(sizeof(unsigned long long));
	*index_ptr = 0;
	*next_id_ptr = 0;
	const auto b = get_options().lc_b;
	const auto N = get_options().parts_dim;
	const double link_len = b / N;
	dev_part_map_ptr = part_map_ptr();
	dev_tree_map_ptr = tree_map_ptr();
	cuda_lightcone_kernel<<<nblocks,BLOCK_SIZE>>>(rc_ptr, leaves.data(), leaves.size(), index_ptr, next_id_ptr, link_len, hpx_size(),hpx_rank());
	CUDA_CHECK(cudaDeviceSynchronize());
	size_t rc = *rc_ptr;
	cuda_free(rc_ptr);
	cuda_free(index_ptr);
	cuda_free(next_id_ptr);
	return rc;

}
