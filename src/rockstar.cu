#include <cosmictiger/rockstar.hpp>
#include <cosmictiger/fixedcapvec.hpp>
#include <cosmictiger/stack_vector.hpp>
#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/math.hpp>

#define ROCKSTAR_MAX_LIST 1024
#define ROCKSTAR_MAX_DEPTH 32
#define ROCKSTAR_MAX_STACK 16384

struct rockstar_workspace {
	fixedcapvec<int, ROCKSTAR_MAX_LIST> nextlist;
	fixedcapvec<int, ROCKSTAR_MAX_LIST> leaflist;
	stack_vector<int, ROCKSTAR_MAX_STACK, ROCKSTAR_MAX_DEPTH> checklist;
	fixedcapvec<int, ROCKSTAR_MAX_DEPTH> phase;
	fixedcapvec<int, ROCKSTAR_MAX_DEPTH> self;
	fixedcapvec<int, ROCKSTAR_MAX_DEPTH> returns;
};

struct rockstar_particles {
	float* x;
	float* y;
	float* z;
	float* vx;
	float* vy;
	float* vz;
	int* subgroup;
};

struct rockstar_shmem {
	array<float, ROCKSTAR_BUCKET_SIZE> x;
	array<float, ROCKSTAR_BUCKET_SIZE> y;
	array<float, ROCKSTAR_BUCKET_SIZE> z;
	array<float, ROCKSTAR_BUCKET_SIZE> vx;
	array<float, ROCKSTAR_BUCKET_SIZE> vy;
	array<float, ROCKSTAR_BUCKET_SIZE> vz;
	array<int, ROCKSTAR_BUCKET_SIZE> sg;
};

__global__ void rockstar_find_subgroups_gpu(rockstar_tree* trees, int ntrees, rockstar_particles parts, rockstar_workspace* lists, float link_len, int* next_id,
		int* active_cnt) {
	__shared__ rockstar_shmem shmem;
	auto& nextlist = lists->nextlist;
	auto& leaflist = lists->leaflist;
	auto& phase = lists->phase;
	auto& self_index = lists->self;
	auto& checklist = lists->checklist;
	auto& returns = lists->returns;
	const float link_len2 = sqr(link_len);
	const int& tid = threadIdx.x;
	nextlist.initialize();
	leaflist.initialize();
	phase.initialize();
	self_index.initialize();
	returns.initialize();
	checklist.initialize();

	phase.resize(0);
	self_index.resize(0);
	returns.push_back(0);
	checklist.resize(1);
	if (tid == 0) {
		checklist[0] = ntrees - 1;
	}
	phase.push_back(0);
	self_index.push_back(ntrees - 1);
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
				int found_any_link = 0;
				for (int ci = 0; ci < leaflist.size(); ci++) {
					if (leaflist[ci] != self_index.back()) {
						int nparts = 0;
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
								shmem.x[index] = parts.x[pi];
								shmem.y[index] = parts.y[pi];
								shmem.z[index] = parts.z[pi];
								shmem.vx[index] = parts.vx[pi];
								shmem.vy[index] = parts.vy[pi];
								shmem.vz[index] = parts.vz[pi];
								shmem.sg[index] = parts.subgroup[pi];
							}
							nparts += total;
						}
						__syncwarp();
						for (int snk_i = self.part_begin + ci; snk_i < self.part_end; snk_i += WARP_SIZE) {
							for (int src_i = 0; src_i < nparts; src_i++) {
								const float dx = parts.x[snk_i] - shmem.x[src_i];
								const float dy = parts.y[snk_i] - shmem.y[src_i];
								const float dz = parts.z[snk_i] - shmem.z[src_i];
								const float dvx = parts.vx[snk_i] - shmem.vx[src_i];
								const float dvy = parts.vy[snk_i] - shmem.vy[src_i];
								const float dvz = parts.vz[snk_i] - shmem.vz[src_i];
								const float R2 = sqr(dx, dy, dz) + sqr(dvx, dvy, dvz);
								if (R2 <= link_len2) {
									if (parts.subgroup[snk_i] == ROCKSTAR_NO_GROUP) {
										parts.subgroup[snk_i] = atomicAdd(next_id, 1);
										found_any_link++;
									}
									if (parts.subgroup[snk_i] > shmem.sg[src_i] && shmem.sg[src_i] != ROCKSTAR_NO_GROUP) {
										parts.subgroup[snk_i] = shmem.sg[src_i];
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
					for (int snk_i = self.part_begin + tid; snk_i < self.part_end; snk_i += WARP_SIZE) {
						for (int src_i = snk_i - tid; src_i < self.part_end; src_i++) {
							if (src_i > snk_i) {
								const float dx = parts.x[snk_i] - parts.x[src_i];
								const float dy = parts.y[snk_i] - parts.y[src_i];
								const float dz = parts.z[snk_i] - parts.z[src_i];
								const float dvx = parts.vx[snk_i] - parts.vx[src_i];
								const float dvy = parts.vy[snk_i] - parts.vy[src_i];
								const float dvz = parts.vz[snk_i] - parts.vz[src_i];
								const float R2 = sqr(dx, dy, dz) + sqr(dvx, dvy, dvz);
								if (R2 <= link_len2) {
									auto& snk_grp = parts.subgroup[snk_i];
									auto& src_grp = parts.subgroup[src_i];
									if (snk_grp == ROCKSTAR_NO_GROUP) {
										snk_grp = atomicAdd(next_id, 1);
										found_link++;
									}
									if (src_grp == ROCKSTAR_NO_GROUP) {
										src_grp = atomicAdd(next_id, 1);
										found_link++;
									}
									if (atomicMin(&snk_grp, src_grp) != snk_grp) {
										found_link++;
									} else if (atomicMin(&src_grp, snk_grp) != src_grp) {
										found_link++;
									}
								}
							}
						}
					}
					found_any_link += found_link;
					shared_reduce_add(found_link);
				} while (found_link);
				shared_reduce_add(found_any_link);
				if (tid == 0) {
					self.active = found_any_link != 0;
				}
				phase.pop_back();
				self_index.pop_back();
				depth--;
				if (tid == 0) {
					returns.back() = int(self.active);
				}
			} else {
				const int offset = checklist.size();
				checklist.resize(checklist.size() + leaflist.size());
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
			}
			depth--;
		}
			break;
		}
	}
	if (tid == 0) {
		*active_cnt = returns[0];
	}
	returns.pop_back();
	ASSERT(returns.size() == 0);
	ASSERT(phase.size() == 0);
	ASSERT(self_index.size() == 0);

}

int rockstar_find_subgroups_gpu(vector<rockstar_tree>& trees, vector<rockstar_particle>& parts, float link_len, int& next_index) {
	vector<float, pinned_allocator<float>> x(parts.size());
	vector<float, pinned_allocator<float>> y(parts.size());
	vector<float, pinned_allocator<float>> z(parts.size());
	vector<float, pinned_allocator<float>> vx(parts.size());
	vector<float, pinned_allocator<float>> vy(parts.size());
	vector<float, pinned_allocator<float>> vz(parts.size());
	vector<rockstar_tree, pinned_allocator<rockstar_tree>> dev_trees(trees.begin(), trees.end());
	vector<int, pinned_allocator<int>> sg(parts.size());
	int* dev_next_index;
	int* dev_active_cnt;
	int active_cnt;
	rockstar_workspace* lists;
	auto stream = cuda_get_stream();
	CUDA_CHECK(cudaMalloc(&lists, sizeof(rockstar_workspace)));
	CUDA_CHECK(cudaMalloc(&dev_next_index, sizeof(int)));
	CUDA_CHECK(cudaMalloc(&dev_active_cnt, sizeof(int)));
	CUDA_CHECK(cudaMemcpyAsync(dev_next_index, &next_index, sizeof(int), cudaMemcpyHostToDevice, stream));
	for (int i = 0; i < parts.size(); i++) {
		x[i] = parts[i].x;
		y[i] = parts[i].y;
		z[i] = parts[i].z;
		vx[i] = parts[i].vx;
		vy[i] = parts[i].vy;
		vz[i] = parts[i].vz;
		sg[i] = parts[i].subgroup;
	}
	rockstar_particles part_ptrs;
	part_ptrs.x = x.data();
	part_ptrs.y = y.data();
	part_ptrs.z = z.data();
	part_ptrs.vx = vx.data();
	part_ptrs.vy = vy.data();
	part_ptrs.vz = vz.data();
	part_ptrs.subgroup = sg.data();
	rockstar_find_subgroups_gpu<<<1,WARP_SIZE, 0, stream>>>(dev_trees.data(), dev_trees.size(), part_ptrs, lists, link_len, dev_next_index, dev_active_cnt);
	CUDA_CHECK(cudaMemcpyAsync(&next_index, dev_next_index, sizeof(int), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(&active_cnt, dev_active_cnt, sizeof(int), cudaMemcpyDeviceToHost, stream));
	cuda_stream_synchronize(stream);
	PRINT("%i\n", active_cnt);
	for (int i = 0; i < parts.size(); i++) {
		parts[i].subgroup = sg[i];
	}
	for (int i = 0; i < trees.size(); i++) {
		trees[i].active = dev_trees[i].active;
	}
	CUDA_CHECK(cudaFree(dev_next_index));
	CUDA_CHECK(cudaFree(lists));
	cuda_end_stream(stream);
	return active_cnt;
}
