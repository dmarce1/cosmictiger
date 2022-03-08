#include <cosmictiger/rockstar.hpp>
#include <cosmictiger/device_vector.hpp>
#include <cosmictiger/stack_vector.hpp>
#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/math.hpp>

#define ROCKSTAR_MAX_LIST 1024
#define ROCKSTAR_MAX_DEPTH 64
#define ROCKSTAR_MAX_STACK 16384

struct rockstar_workspace {
	device_vector<int> nextlist;
	device_vector<int> leaflist;
	stack_vector<int> checklist;
	device_vector<int> phase;
	device_vector<int> self;
	device_vector<int> returns;
};

struct rockstar_shmem {
	array<float, ROCKSTAR_BUCKET_SIZE> src_x;
	array<float, ROCKSTAR_BUCKET_SIZE> src_y;
	array<float, ROCKSTAR_BUCKET_SIZE> src_z;
	array<float, ROCKSTAR_BUCKET_SIZE> src_vx;
	array<float, ROCKSTAR_BUCKET_SIZE> src_vy;
	array<float, ROCKSTAR_BUCKET_SIZE> src_vz;
	array<int, ROCKSTAR_BUCKET_SIZE> src_sg;
	array<float, ROCKSTAR_BUCKET_SIZE> snk_x;
	array<float, ROCKSTAR_BUCKET_SIZE> snk_y;
	array<float, ROCKSTAR_BUCKET_SIZE> snk_z;
	array<float, ROCKSTAR_BUCKET_SIZE> snk_vx;
	array<float, ROCKSTAR_BUCKET_SIZE> snk_vy;
	array<float, ROCKSTAR_BUCKET_SIZE> snk_vz;
	array<int, ROCKSTAR_BUCKET_SIZE> snk_sg;
};

__global__ void rockstar_find_subgroups_gpu(rockstar_tree* trees, int ntrees, array<int, ROCKSTAR_MAX_LIST>* checklists, int* checklistsz,
		rockstar_particles parts, rockstar_workspace* lists, int* self_ids, float link_len, int* next_id, int* active_cnt) {
	__shared__ rockstar_shmem shmem;
	const int& tid = threadIdx.x;
	const int& bid = blockIdx.x;
	new( lists + bid ) rockstar_workspace();
	auto& nextlist = lists[bid].nextlist;
	auto& leaflist = lists[bid].leaflist;
	auto& phase = lists[bid].phase;
	auto& self_index = lists[bid].self;
	auto& checklist = lists[bid].checklist;
	auto& returns = lists[bid].returns;
	const float link_len2 = sqr(link_len);

	phase.resize(0);
	self_index.resize(0);
	returns.push_back(0);
	checklist.resize(checklistsz[bid]);
	for (int ci = tid; ci < checklistsz[bid]; ci += WARP_SIZE) {
		checklist[ci] = checklists[bid][ci];
	}
	phase.push_back(0);
	self_index.push_back(self_ids[bid]);
	int depth = 0;
	while (depth >= 0) {
		auto& self = trees[self_index.back()];
		switch (phase.back()) {

		case 0: {
			nextlist.resize(0);
			leaflist.resize(0);
			range<float, 2 * NDIM> mybox = self.box.pad(link_len * 1.001);
			bool iamleaf = self.children[LEFT] == -1;
			do {
				for (int ci = tid; ci < round_up(checklist.size(), WARP_SIZE); ci += WARP_SIZE) {
//					PRINT( "%i\n", depth);
					bool use_next_list = false;
					bool use_leaf_list = false;
					if (ci < checklist.size()) {
						const auto& other = trees[checklist[ci]];
						if (other.last_active) {
							if (mybox.intersection(other.box).volume() > 0) {
								use_next_list = other.children[LEFT] != -1;
								use_leaf_list = !use_next_list;
							}
						}
					}
					int index;
					int total;

					index = use_next_list;
					compute_indices(index, total);
					index = 2 * index + nextlist.size();
					nextlist.resize(nextlist.size() + NCHILD * total);
					if (use_next_list) {
						const auto& other = trees[checklist[ci]];
						nextlist[index + LEFT] = other.children[LEFT];
						nextlist[index + RIGHT] = other.children[RIGHT];
					}

					index = use_leaf_list;
					compute_indices(index, total);
					index = index + leaflist.size();
					leaflist.resize(leaflist.size() + total);
					if (use_leaf_list) {
						leaflist[index] = checklist[ci];
					}
				}
				checklist.resize(nextlist.size());
				for (int ci = tid; ci < nextlist.size(); ci += WARP_SIZE) {
					checklist[ci] = nextlist[ci];
				}
				nextlist.resize(0);
			} while (checklist.size() && iamleaf);
			if (self.children[LEFT] == -1) {
				int nparts;
				for (int snk_i = self.part_begin + tid; snk_i < self.part_end; snk_i += WARP_SIZE) {
					const int j = snk_i - self.part_begin;
					shmem.snk_x[j] = parts.x[snk_i];
					shmem.snk_y[j] = parts.y[snk_i];
					shmem.snk_z[j] = parts.z[snk_i];
					shmem.snk_vx[j] = parts.vx[snk_i];
					shmem.snk_vy[j] = parts.vy[snk_i];
					shmem.snk_vz[j] = parts.vz[snk_i];
					shmem.snk_sg[j] = parts.subgroup[snk_i];
				}
				int found_any_link = 0;
				for (int ci = 0; ci < leaflist.size(); ci++) {
					if (leaflist[ci] != self_index.back()) {
						nparts = 0;
						const auto& other = trees[leaflist[ci]];
						const int maxpi = round_up(other.part_end - other.part_begin, WARP_SIZE) + other.part_begin;
						for (int pi = other.part_begin + tid; pi < maxpi; pi += WARP_SIZE) {
							array<float, 2 * NDIM> x;
							bool contains = false;
							if (pi < other.part_end) {
								x[XDIM] = parts.x[pi];
								x[YDIM] = parts.y[pi];
								x[ZDIM] = parts.z[pi];
								x[NDIM + XDIM] = parts.vx[pi];
								x[NDIM + YDIM] = parts.vy[pi];
								x[NDIM + ZDIM] = parts.vz[pi];
								contains = mybox.contains(x);
							}
							int index;
							int total;
							index = contains;
							compute_indices(index, total);
							const int offset = nparts;
							index += offset;
							if (contains) {
								shmem.src_x[index] = parts.x[pi];
								shmem.src_y[index] = parts.y[pi];
								shmem.src_z[index] = parts.z[pi];
								shmem.src_vx[index] = parts.vx[pi];
								shmem.src_vy[index] = parts.vy[pi];
								shmem.src_vz[index] = parts.vz[pi];
								shmem.src_sg[index] = parts.subgroup[pi];
							}
							nparts += total;
						}
						__syncwarp();
						for (int snk_i = ci; snk_i < self.part_end - self.part_begin; snk_i += WARP_SIZE) {
							for (int src_i = 0; src_i < nparts; src_i++) {
								const float dx = shmem.snk_x[snk_i] - shmem.src_x[src_i];
								const float dy = shmem.snk_y[snk_i] - shmem.src_y[src_i];
								const float dz = shmem.snk_z[snk_i] - shmem.src_z[src_i];
								const float dvx = shmem.snk_vx[snk_i] - shmem.src_vx[src_i];
								const float dvy = shmem.snk_vy[snk_i] - shmem.src_vy[src_i];
								const float dvz = shmem.snk_vz[snk_i] - shmem.src_vz[src_i];
								const float R2 = sqr(dx, dy, dz) + sqr(dvx, dvy, dvz);
								if (R2 <= link_len2) {
									if (shmem.snk_sg[snk_i] == ROCKSTAR_NO_GROUP) {
										shmem.snk_sg[snk_i] = atomicAdd(next_id, 1);
										found_any_link++;
									}
									if (shmem.snk_sg[snk_i] > shmem.src_sg[src_i] && shmem.src_sg[src_i] != ROCKSTAR_NO_GROUP) {
										shmem.snk_sg[snk_i] = shmem.src_sg[src_i];
										found_any_link++;
									}
								}
							}
						}
					}
				}
				int found_link;
				do {
					found_link = 0;
					nparts = self.part_end - self.part_begin;
					__syncwarp();
					int maxi = round_up(nparts, WARP_SIZE);
					for (int snk_i = tid; snk_i < maxi; snk_i += WARP_SIZE) {
						for (int src_i = snk_i - tid; src_i < nparts; src_i++) {
							int src_link = 0;
							float R2;
							auto& snk_grp = shmem.snk_sg[min(snk_i, nparts - 1)];
							auto& src_grp = shmem.snk_sg[src_i];
							if (src_i > snk_i && snk_i < nparts) {
								const float dx = shmem.snk_x[snk_i] - shmem.snk_x[src_i];
								const float dy = shmem.snk_y[snk_i] - shmem.snk_y[src_i];
								const float dz = shmem.snk_z[snk_i] - shmem.snk_z[src_i];
								const float dvx = shmem.snk_vx[snk_i] - shmem.snk_vx[src_i];
								const float dvy = shmem.snk_vy[snk_i] - shmem.snk_vy[src_i];
								const float dvz = shmem.snk_vz[snk_i] - shmem.snk_vz[src_i];
								R2 = sqr(dx, dy, dz) + sqr(dvx, dvy, dvz);
								if (R2 <= link_len2) {
									if (snk_grp == ROCKSTAR_NO_GROUP) {
										snk_grp = atomicAdd(next_id, 1);
										found_link++;
										src_link++;
									}
								}
							}
							shared_reduce_add(src_link);
							if (tid == 0) {
								if (src_link && src_grp == ROCKSTAR_NO_GROUP) {
									src_grp = atomicAdd(next_id, 1);
								}
							}
							__syncwarp();
							if (src_i > snk_i && snk_i < nparts) {
								if (R2 <= link_len2) {
									if (snk_grp > src_grp) {
										snk_grp = src_grp;
										found_link++;
									}
								}
							}
							__syncwarp();
							int sg = ROCKSTAR_NO_GROUP;
							if (src_i > snk_i && snk_i < nparts) {
								if (R2 <= link_len2) {
									if (src_grp > snk_grp) {
										sg = snk_grp;
										found_link++;
									}
								}
							}
							shared_reduce_min(sg);
							if (tid == 0) {
								src_grp = min(src_grp, sg);
							}
						}
					}
					__syncwarp();
					found_any_link += found_link;
					shared_reduce_add(found_link);
				} while (found_link);
				for (int snk_i = self.part_begin + tid; snk_i < self.part_end; snk_i += WARP_SIZE) {
					const int j = snk_i - self.part_begin;
					parts.subgroup[snk_i] = shmem.snk_sg[j];
				}

				shared_reduce_add(found_any_link);
				if (tid == 0) {
					self.active = found_any_link != 0;
				}
				phase.pop_back();
				self_index.pop_back();
				depth--;
				if (tid == 0) {
					returns.back() = int(self.active);
					self.active_count = returns.back();
				}
			} else {
				const int offset = checklist.size();
				checklist.resize(checklist.size() + leaflist.size());
				if (checklist.size()) {
					for (int ci = tid; ci < leaflist.size(); ci += WARP_SIZE) {
						checklist[ci + offset] = leaflist[ci];
					}
					returns.push_back(0);
					const auto child = self.children[LEFT];
					__syncwarp();
					checklist.push_top();
					phase.back() += 1;
					phase.push_back(0);
					self_index.push_back(child);
					depth++;
				} else {
					self_index.pop_back();
					phase.pop_back();
					if (tid == 0) {
						self.active = false;
						self.active_count = 0;
					}
					depth--;
				}
			}

		}
			break;
		case 1: {
			checklist.pop_top();
			phase.back() += 1;
			phase.push_back(0);
			const auto child = self.children[RIGHT];
			self_index.push_back(child);
			const auto this_return = returns.back();
			returns.pop_back();
			if (tid == 0) {
				returns.back() += this_return;
			}
			returns.push_back(0);
			depth++;
		}
			break;
		case 2: {
			self_index.pop_back();
			phase.pop_back();
			const auto this_return = returns.back();
			returns.pop_back();
			if (tid == 0) {
				returns.back() += this_return;
				self.active = returns.back() != 0;
				self.active_count = returns.back();
			}
			depth--;
		}
			break;
		}
	}
	if (tid == 0) {
		active_cnt[bid] = returns[0];
	}
	returns.pop_back();
	ASSERT(returns.size() == 0);
	ASSERT(phase.size() == 0);
	ASSERT(self_index.size() == 0);
	(lists + bid)->~rockstar_workspace();

}

__global__ void rockstar_find_link_len_gpu(rockstar_tree* trees, int ntrees, array<int, ROCKSTAR_MAX_LIST>* checklists, int* checklistsz,
		rockstar_particles parts, rockstar_workspace* lists, int* self_ids, float link_len, int* next_id, int* active_cnt) {
	__shared__ rockstar_shmem shmem;
	const int& tid = threadIdx.x;
	const int& bid = blockIdx.x;
	new (lists + bid) rockstar_workspace();
	auto& nextlist = lists[bid].nextlist;
	auto& leaflist = lists[bid].leaflist;
	auto& phase = lists[bid].phase;
	auto& self_index = lists[bid].self;
	auto& checklist = lists[bid].checklist;
	auto& returns = lists[bid].returns;
	const float link_len2 = sqr(link_len);

	phase.resize(0);
	self_index.resize(0);
	returns.push_back(0);
	checklist.resize(checklistsz[bid]);
	for (int ci = tid; ci < checklistsz[bid]; ci += WARP_SIZE) {
		checklist[ci] = checklists[bid][ci];
	}
	phase.push_back(0);
	self_index.push_back(self_ids[bid]);
	int depth = 0;
	while (depth >= 0) {
		auto& self = trees[self_index.back()];
		switch (phase.back()) {

		case 0: {
			nextlist.resize(0);
			leaflist.resize(0);
			range<float, 2 * NDIM> mybox = self.box.pad(link_len * 1.001);
			bool iamleaf = self.children[LEFT] == -1;
			do {
				for (int ci = tid; ci < round_up(checklist.size(), WARP_SIZE); ci += WARP_SIZE) {
//					PRINT( "%i\n", depth);
					bool use_next_list = false;
					bool use_leaf_list = false;
					if (ci < checklist.size()) {
						const auto& other = trees[checklist[ci]];
						if (other.last_active) {
							if (mybox.intersection(other.box).volume() > 0) {
								use_next_list = other.children[LEFT] != -1;
								use_leaf_list = !use_next_list;
							}
						}
					}
					int index;
					int total;

					index = use_next_list;
					compute_indices(index, total);
					index = 2 * index + nextlist.size();
					nextlist.resize(nextlist.size() + NCHILD * total);
					if (use_next_list) {
						const auto& other = trees[checklist[ci]];
						nextlist[index + LEFT] = other.children[LEFT];
						nextlist[index + RIGHT] = other.children[RIGHT];
					}

					index = use_leaf_list;
					compute_indices(index, total);
					index = index + leaflist.size();
					leaflist.resize(leaflist.size() + total);
					if (use_leaf_list) {
						leaflist[index] = checklist[ci];
					}
				}
				checklist.resize(nextlist.size());
				for (int ci = tid; ci < nextlist.size(); ci += WARP_SIZE) {
					checklist[ci] = nextlist[ci];
				}
				nextlist.resize(0);
			} while (checklist.size() && iamleaf);
			if (self.children[LEFT] == -1) {
				int nparts;
				for (int snk_i = self.part_begin + tid; snk_i < self.part_end; snk_i += WARP_SIZE) {
					const int j = snk_i - self.part_begin;
					shmem.snk_x[j] = parts.x[snk_i];
					shmem.snk_y[j] = parts.y[snk_i];
					shmem.snk_z[j] = parts.z[snk_i];
					shmem.snk_vx[j] = parts.vx[snk_i];
					shmem.snk_vy[j] = parts.vy[snk_i];
					shmem.snk_vz[j] = parts.vz[snk_i];
					shmem.snk_sg[j] = parts.subgroup[snk_i];
				}
				int found_any_link = 0;
				for (int ci = 0; ci < leaflist.size(); ci++) {
					if (leaflist[ci] != self_index.back()) {
						nparts = 0;
						const auto& other = trees[leaflist[ci]];
						const int maxpi = round_up(other.part_end - other.part_begin, WARP_SIZE) + other.part_begin;
						for (int pi = other.part_begin + tid; pi < maxpi; pi += WARP_SIZE) {
							array<float, 2 * NDIM> x;
							bool contains = false;
							if (pi < other.part_end) {
								x[XDIM] = parts.x[pi];
								x[YDIM] = parts.y[pi];
								x[ZDIM] = parts.z[pi];
								x[NDIM + XDIM] = parts.vx[pi];
								x[NDIM + YDIM] = parts.vy[pi];
								x[NDIM + ZDIM] = parts.vz[pi];
								contains = mybox.contains(x);
							}
							int index;
							int total;
							index = contains;
							compute_indices(index, total);
							const int offset = nparts;
							index += offset;
							if (contains) {
								shmem.src_x[index] = parts.x[pi];
								shmem.src_y[index] = parts.y[pi];
								shmem.src_z[index] = parts.z[pi];
								shmem.src_vx[index] = parts.vx[pi];
								shmem.src_vy[index] = parts.vy[pi];
								shmem.src_vz[index] = parts.vz[pi];
								shmem.src_sg[index] = parts.subgroup[pi];
							}
							nparts += total;
						}
						__syncwarp();
						for (int snk_i = ci; snk_i < self.part_end - self.part_begin; snk_i += WARP_SIZE) {
							for (int src_i = 0; src_i < nparts; src_i++) {
								const float dx = shmem.snk_x[snk_i] - shmem.src_x[src_i];
								const float dy = shmem.snk_y[snk_i] - shmem.src_y[src_i];
								const float dz = shmem.snk_z[snk_i] - shmem.src_z[src_i];
								const float dvx = shmem.snk_vx[snk_i] - shmem.src_vx[src_i];
								const float dvy = shmem.snk_vy[snk_i] - shmem.src_vy[src_i];
								const float dvz = shmem.snk_vz[snk_i] - shmem.src_vz[src_i];
								const float R2 = sqr(dx, dy, dz) + sqr(dvx, dvy, dvz);
								if (R2 <= link_len2) {
									if (shmem.snk_sg[snk_i] == ROCKSTAR_NO_GROUP) {
										shmem.snk_sg[snk_i] = atomicAdd(next_id, 1);
										found_any_link++;
									}
									if (shmem.snk_sg[snk_i] > shmem.src_sg[src_i] && shmem.src_sg[src_i] != ROCKSTAR_NO_GROUP) {
										shmem.snk_sg[snk_i] = shmem.src_sg[src_i];
										found_any_link++;
									}
								}
							}
						}
					}
				}
				int found_link;
				do {
					found_link = 0;
					nparts = self.part_end - self.part_begin;
					__syncwarp();
					int maxi = round_up(nparts, WARP_SIZE);
					for (int snk_i = tid; snk_i < maxi; snk_i += WARP_SIZE) {
						for (int src_i = snk_i - tid; src_i < nparts; src_i++) {
							int src_link = 0;
							float R2;
							auto& snk_grp = shmem.snk_sg[min(snk_i, nparts - 1)];
							auto& src_grp = shmem.snk_sg[src_i];
							if (src_i > snk_i && snk_i < nparts) {
								const float dx = shmem.snk_x[snk_i] - shmem.snk_x[src_i];
								const float dy = shmem.snk_y[snk_i] - shmem.snk_y[src_i];
								const float dz = shmem.snk_z[snk_i] - shmem.snk_z[src_i];
								const float dvx = shmem.snk_vx[snk_i] - shmem.snk_vx[src_i];
								const float dvy = shmem.snk_vy[snk_i] - shmem.snk_vy[src_i];
								const float dvz = shmem.snk_vz[snk_i] - shmem.snk_vz[src_i];
								R2 = sqr(dx, dy, dz) + sqr(dvx, dvy, dvz);
								if (R2 <= link_len2) {
									if (snk_grp == ROCKSTAR_NO_GROUP) {
										snk_grp = atomicAdd(next_id, 1);
										found_link++;
										src_link++;
									}
								}
							}
							shared_reduce_add(src_link);
							if (tid == 0) {
								if (src_link && src_grp == ROCKSTAR_NO_GROUP) {
									src_grp = atomicAdd(next_id, 1);
								}
							}
							__syncwarp();
							if (src_i > snk_i && snk_i < nparts) {
								if (R2 <= link_len2) {
									if (snk_grp > src_grp) {
										snk_grp = src_grp;
										found_link++;
									}
								}
							}
							__syncwarp();
							int sg = ROCKSTAR_NO_GROUP;
							if (src_i > snk_i && snk_i < nparts) {
								if (R2 <= link_len2) {
									if (src_grp > snk_grp) {
										sg = snk_grp;
										found_link++;
									}
								}
							}
							shared_reduce_min(sg);
							if (tid == 0) {
								src_grp = min(src_grp, sg);
							}
						}
					}
					__syncwarp();
					found_any_link += found_link;
					shared_reduce_add(found_link);
				} while (found_link);
				for (int snk_i = self.part_begin + tid; snk_i < self.part_end; snk_i += WARP_SIZE) {
					const int j = snk_i - self.part_begin;
					parts.subgroup[snk_i] = shmem.snk_sg[j];
				}

				shared_reduce_add(found_any_link);
				if (tid == 0) {
					self.active = found_any_link != 0;
				}
				phase.pop_back();
				self_index.pop_back();
				depth--;
				if (tid == 0) {
					returns.back() = int(self.active);
					self.active_count = returns.back();
				}
			} else {
				const int offset = checklist.size();
				checklist.resize(checklist.size() + leaflist.size());
				if (checklist.size()) {
					for (int ci = tid; ci < leaflist.size(); ci += WARP_SIZE) {
						checklist[ci + offset] = leaflist[ci];
					}
					returns.push_back(0);
					const auto child = self.children[LEFT];
					__syncwarp();
					checklist.push_top();
					phase.back() += 1;
					phase.push_back(0);
					self_index.push_back(child);
					depth++;
				} else {
					self_index.pop_back();
					phase.pop_back();
					if (tid == 0) {
						self.active = false;
						self.active_count = 0;
					}
					depth--;
				}
			}

		}
			break;
		case 1: {
			checklist.pop_top();
			phase.back() += 1;
			phase.push_back(0);
			const auto child = self.children[RIGHT];
			self_index.push_back(child);
			const auto this_return = returns.back();
			returns.pop_back();
			if (tid == 0) {
				returns.back() += this_return;
			}
			returns.push_back(0);
			depth++;
		}
			break;
		case 2: {
			self_index.pop_back();
			phase.pop_back();
			const auto this_return = returns.back();
			returns.pop_back();
			if (tid == 0) {
				returns.back() += this_return;
				self.active = returns.back() != 0;
				self.active_count = returns.back();
			}
			depth--;
		}
			break;
		}
	}
	if (tid == 0) {
		active_cnt[bid] = returns[0];
	}
	returns.pop_back();
	ASSERT(returns.size() == 0);
	ASSERT(phase.size() == 0);
	ASSERT(self_index.size() == 0);
	(lists + bid)->~rockstar_workspace();

}

vector<size_t> rockstar_find_subgroups_gpu(vector<rockstar_tree, pinned_allocator<rockstar_tree>>& trees, rockstar_particles part_ptrs,
		const vector<int>& selves, const vector<vector<int>>& checklists, float link_len, int& next_index) {
	PRINT("%i blocks\n", selves.size());
	vector<int> active_cnts(selves.size());
	vector<array<int, ROCKSTAR_MAX_LIST>, pinned_allocator<array<int, ROCKSTAR_MAX_LIST>>> dev_checklists(selves.size());
	vector<int, pinned_allocator<int>> checklists_szs(selves.size());
	vector<int, pinned_allocator<int>> dev_selves(selves.begin(), selves.end());
	for (int i = 0; i < selves.size(); i++) {
		checklists_szs[i] = checklists[i].size();
		for (int j = 0; j < checklists[i].size(); j++) {
			dev_checklists[i][j] = checklists[i][j];
		}
	}
	int* dev_next_index;
	int* dev_active_cnt;
	rockstar_workspace* lists;
	auto stream = cuda_get_stream();
	CUDA_CHECK(cudaMalloc(&lists, selves.size() * sizeof(rockstar_workspace)));
	CUDA_CHECK(cudaMalloc(&dev_next_index, sizeof(int)));
	CUDA_CHECK(cudaMalloc(&dev_active_cnt, selves.size() * sizeof(int)));
	CUDA_CHECK(cudaMemcpyAsync(dev_next_index, &next_index, sizeof(int), cudaMemcpyHostToDevice, stream));

	rockstar_find_subgroups_gpu<<<selves.size(), WARP_SIZE, 0, stream>>>(trees.data(), trees.size(), dev_checklists.data(), checklists_szs.data(),
			part_ptrs, lists, dev_selves.data(), link_len, dev_next_index, dev_active_cnt);
	CUDA_CHECK(cudaMemcpyAsync(&next_index, dev_next_index, sizeof(int), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(active_cnts.data(), dev_active_cnt, selves.size() * sizeof(int), cudaMemcpyDeviceToHost, stream));
	cuda_stream_synchronize(stream);
//	PRINT("%i\n", active_cnt);
	CUDA_CHECK(cudaFree(dev_next_index));
	CUDA_CHECK(cudaFree(lists));
	cuda_end_stream(stream);
	return vector<size_t>(active_cnts.begin(), active_cnts.end());
}
