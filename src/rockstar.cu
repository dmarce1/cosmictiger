#include <cosmictiger/rockstar.hpp>
#include <cosmictiger/fixedcapvec.hpp>
#include <cosmictiger/stack_vector.hpp>
#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/math.hpp>

#define BLOCK_SIZE 32

__global__ void rockstar_find_subgroups_kernel(rockstar_tree* nodes, const int* leaves, rockstar_particle* parts, float link_len, int* next_index,
		int* active_cnt);
__global__ void rockstar_assign_linklen_kernel(const rockstar_tree* nodes, const int* leaves, rockstar_particle* parts, float link_len, bool phase);

static std::atomic<int> blocks_used(0);

bool rockstar_cuda_free() {
	return blocks_used < cuda_smp_count();
}

void rockstar_assign_linklen_cuda(const device_vector<rockstar_tree>& nodes, const device_vector<int>& leaves, device_vector<rockstar_particle>& parts,
		float link_len, bool phase) {
	auto stream = cuda_get_stream();
	blocks_used += leaves.size();
	rockstar_assign_linklen_kernel<<<leaves.size(), BLOCK_SIZE,0,stream>>>(nodes.data(), leaves.data(), parts.data(), link_len, phase);
	cuda_end_stream(stream);
	blocks_used -= leaves.size();
}

void rockstar_find_subgroups_cuda(device_vector<rockstar_tree>& nodes, const device_vector<int>& inleaves, device_vector<rockstar_particle>& parts,
		float link_len, int& next_id) {
	int* next_index;
	int* active_cnt;
	next_index = (int*) cuda_malloc(sizeof(int));
	active_cnt = (int*) cuda_malloc(sizeof(int));
	*next_index = next_id;
	int cnt = 0;
	do {
		device_vector<int> leaves;
		*active_cnt = 0;
		for (int i = 0; i < inleaves.size(); i++) {
			const int li = inleaves[i];
			nodes[li].last_active = nodes[li].active;
			if (nodes[li].active) {
				leaves.push_back(li);
			}
		}
		auto stream = cuda_get_stream();
		blocks_used += leaves.size();
		rockstar_find_subgroups_kernel<<<leaves.size(),BLOCK_SIZE,0,stream>>>( nodes.data(), leaves.data(), parts.data(),link_len, next_index, active_cnt);
		cuda_end_stream(stream);
		blocks_used -= leaves.size();
		cnt++;
	} while (*active_cnt);
	next_id = *next_index;
}

__global__ void rockstar_find_subgroups_kernel(rockstar_tree* nodes, const int* leaves, rockstar_particle* parts, float link_len, int* next_index,
		int* active_cnt) {
	const int& tid = threadIdx.x;
	const int& index = blockIdx.x;
	__syncthreads();
	const int selfi = leaves[index];
	const rockstar_tree& self = nodes[selfi];
	const auto selfbox = self.box.pad(1.0001 * link_len);
	if (self.last_active) {
		const auto& neighbors = self.neighbors;
		const float link_len2 = sqr(link_len);
		int found_any_link = false;
		const auto generate_id = [next_index]() {
			return atomicAdd(next_index,1);
		};
		for (int ni = 0; ni < neighbors.size(); ni++) {
			const int otheri = neighbors[ni];
			if (otheri != selfi) {
				const auto& other = nodes[otheri];
				for (int i = other.part_begin; i < other.part_end; i++) {
					const auto& Xb = parts[i].X;
					if (selfbox.contains(Xb)) {
						for (int j = self.part_begin + tid; j < self.part_end; j += BLOCK_SIZE) {
							const auto& Xa = parts[j].X;
							float dx2 = 0.0f;
							for (int dim = 0; dim < 2 * NDIM; dim++) {
								dx2 += sqr(Xa[dim] - Xb[dim]);
							}
							if (dx2 < link_len2) {
								auto& A = parts[j].subgroup;
								const auto& B = parts[i].subgroup;
								if (A == ROCKSTAR_NO_GROUP) {
									A = generate_id();
									found_any_link = true;
								}
								if (A > B) {
									A = B;
									found_any_link = true;
								}
								__threadfence();
							}
						}
					}
				}
			}
		}
		int found_link;
		int this_found_link;
		do {
			found_link = false;
			for (int i = self.part_begin; i < self.part_end; i++) {
				const auto& Xa = parts[i].X;
				const int jend = round_up(self.part_end - self.part_begin, BLOCK_SIZE) + self.part_begin;
				int Bmin = ROCKSTAR_NO_GROUP;
				for (int j = self.part_begin + tid; j < jend; j += BLOCK_SIZE) {
					this_found_link = false;
					const auto& Xb = parts[j].X;
					const auto& B = parts[j].subgroup;
					int this_min = ROCKSTAR_NO_GROUP;
					if (j != i && j < self.part_end) {
						float dx2 = 0.0f;
						for (int dim = 0; dim < 2 * NDIM; dim++) {
							dx2 += sqr(Xa[dim] - Xb[dim]);
						}
						if (dx2 < link_len2) {
							this_min = min(this_min, B);
							this_found_link = true;
						}
						Bmin = min(Bmin, this_min);
					}
					shared_reduce_min < BLOCK_SIZE > (this_min);
					shared_reduce_add<int, BLOCK_SIZE>(this_found_link);
				}
				auto& A = parts[i].subgroup;
				if (tid == 0) {
					if (this_found_link && A == ROCKSTAR_NO_GROUP && A == Bmin) {
						A = generate_id();
						found_link = true;
						found_any_link = true;
					}
					if (A > Bmin) {
						found_link = true;
						A = Bmin;
					}
					__threadfence();
				}
			}
			shared_reduce_add(found_link);
		} while (found_link);
		shared_reduce_add(found_any_link);
		if (found_any_link) {
			for (int i = tid; i < neighbors.size(); i += BLOCK_SIZE) {
				auto& other = nodes[neighbors[i]];
				other.active = true;
			}
			atomicAdd(active_cnt, 1);
		}
	}
}

__global__ void rockstar_assign_linklen_kernel(const rockstar_tree* nodes, const int* leaves, rockstar_particle* parts, float link_len, bool phase) {
	const int& tid = threadIdx.x;
	const int& index = blockIdx.x;
	__syncthreads();
	const int selfi = leaves[index];
	const rockstar_tree& self = nodes[selfi];
	const auto selfbox = self.box.pad(1.0001 * link_len);
	const auto& neighbors = self.neighbors;
	const float link_len2 = sqr(link_len);
	const int dimmax = phase ? 2 * NDIM : NDIM;
	for (int ni = 0; ni < neighbors.size(); ni++) {
		const int otheri = neighbors[ni];
		const auto& other = nodes[otheri];
		for (int i = other.part_begin; i < other.part_end; i++) {
			const auto& Xb = parts[i].X;
			if (selfbox.contains(Xb)) {
				for (int j = self.part_begin + tid; j < self.part_end; j += BLOCK_SIZE) {
					const auto& Xa = parts[j].X;
					float dx2 = 0.0f;
					for (int dim = 0; dim < dimmax; dim++) {
						dx2 += sqr(Xa[dim] - Xb[dim]);
					}
					if (dx2 < link_len2 && dx2 > 0.0f) {
						parts[j].min_dist2 = min(dx2, parts[j].min_dist2);
					}
				}
			}
		}
	}
}
