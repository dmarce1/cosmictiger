#include <cosmictiger/bh.hpp>
#include <cosmictiger/cuda.hpp>
#include <cosmictiger/fixedcapvec.hpp>
#include <cosmictiger/stack_vector.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/timer.hpp>

#define BH_LIST_SIZE 1024
#define BH_SOURCE_LIST_SIZE (8*1024)
#define BH_STACK_SIZE 16384
#define BH_MAX_DEPTH 32

struct bh_workspace {
	fixedcapvec<int, BH_LIST_SIZE> nextlist;
	fixedcapvec<int, BH_LIST_SIZE> leaflist;
	fixedcapvec<float, BH_SOURCE_LIST_SIZE> src_x;
	fixedcapvec<float, BH_SOURCE_LIST_SIZE> src_y;
	fixedcapvec<float, BH_SOURCE_LIST_SIZE> src_z;
	fixedcapvec<float, BH_SOURCE_LIST_SIZE> src_m;
	stack_vector<int> checklist;
};

struct bh_shmem {
	int si;
	array<float, BH_BUCKET_SIZE> sink_x;
	array<float, BH_BUCKET_SIZE> sink_y;
	array<float, BH_BUCKET_SIZE> sink_z;
};

__global__ void bh_tree_evaluate_kernel(bh_workspace* workspaces, bh_tree_node* tree_nodes, int* sink_buckets, float* phi, array<float, NDIM>* parts,
		float theta, float hsoft, float GM, int* next_sink_bucket, int nsinks) {
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	__shared__ bh_shmem shmem;
	int& si = shmem.si;
	auto& sink_x = shmem.sink_x;
	auto& sink_y = shmem.sink_y;
	auto& sink_z = shmem.sink_z;
	if (tid == 0) {
		si = atomicAdd(next_sink_bucket, 1);
	}
	__syncwarp();
	bh_workspace& ws = workspaces[bid];
	new(&ws) bh_workspace();
	while (si < nsinks) {
		int sink_index = sink_buckets[si];
		auto& checklist = ws.checklist;
		auto& nextlist = ws.nextlist;
		auto& src_x = ws.src_x;
		auto& src_y = ws.src_y;
		auto& src_z = ws.src_z;
		auto& src_m = ws.src_m;
		auto& leaflist = ws.leaflist;
		checklist.resize(1);
		checklist[0] = 0;
		const auto& self = tree_nodes[sink_index];
		const float my_radius = self.radius;
		const float thetainv = 1.0 / theta;
		int sink_size = self.parts.second - self.parts.first;
		for (int i = tid; i < sink_size; i += WARP_SIZE) {
			sink_x[i] = parts[self.parts.first + i][XDIM];
			sink_y[i] = parts[self.parts.first + i][YDIM];
			sink_z[i] = parts[self.parts.first + i][ZDIM];
		}
		while (checklist.size()) {
			int maxci = round_up(checklist.size(), WARP_SIZE);
			for (int ci = tid; ci < maxci; ci += WARP_SIZE) {
				bool source = false;
				bool leaf = false;
				bool next = false;
				if (ci < checklist.size()) {
					const auto& other = tree_nodes[checklist[ci]];
					const float other_radius = other.radius;
					const float dx = self.pos[XDIM] - other.pos[XDIM];
					const float dy = self.pos[YDIM] - other.pos[YDIM];
					const float dz = self.pos[ZDIM] - other.pos[ZDIM];
					const float r2 = sqr(dx, dy, dz);
					if (r2 > sqr(thetainv * (other_radius + my_radius))) {
						source = true;
					} else if (other.children[LEFT] == -1) {
						leaf = true;
					} else {
						next = true;
					}
				}
				int total;
				int index;
				int offset;

				index = source;
				compute_indices(index, total);
				offset = src_x.size();
				src_x.resize(offset + total);
				src_y.resize(offset + total);
				src_z.resize(offset + total);
				src_m.resize(offset + total);
				index += offset;
				if (source) {
					const auto& other = tree_nodes[checklist[ci]];
					src_x[index] = other.pos[XDIM];
					src_y[index] = other.pos[YDIM];
					src_z[index] = other.pos[ZDIM];
					src_m[index] = other.mass;
				}

				index = leaf;
				compute_indices(index, total);
				offset = leaflist.size();
				leaflist.resize(offset + total);
				index += offset;
				if (leaf) {
					leaflist[index] = checklist[ci];
				}

				index = next;
				compute_indices(index, total);
				offset = nextlist.size();
				nextlist.resize(offset + 2 * total);
				index = 2 * index + offset;
				if (next) {
					nextlist[index + LEFT] = tree_nodes[checklist[ci]].children[LEFT];
					nextlist[index + RIGHT] = tree_nodes[checklist[ci]].children[RIGHT];
				}
			}
			checklist.resize(nextlist.size());
			for (int ci = tid; ci < nextlist.size(); ci += WARP_SIZE) {
				checklist[ci] = nextlist[ci];
			}

			nextlist.resize(0);
			for (int i = 0; i < leaflist.size(); i++) {
				const auto& other = tree_nodes[leaflist[i]];
				int offset = src_x.size();
				int new_size = offset + other.parts.second - other.parts.first;
				src_x.resize(new_size);
				src_y.resize(new_size);
				src_z.resize(new_size);
				src_m.resize(new_size);
				for (int j = tid + other.parts.first; j < other.parts.second; j += WARP_SIZE) {
					int k = j + offset - other.parts.first;
					src_x[k] = parts[j][XDIM];
					src_y[k] = parts[j][YDIM];
					src_z[k] = parts[j][ZDIM];
					src_m[k] = 1.0f;
				}
				__syncwarp();
			}
			leaflist.resize(0);
		}

		const float h = hsoft;
		const float hinv = 1.0 / (2.f * h);
		const float h2inv = 1.0 / (4.f * h * h);
		const float h2 = 4.f * h * h;
		for (int i = tid; i < sink_size; i += WARP_SIZE) {
			phi[i + self.parts.first] = -SELF_PHI * hinv;
			for (int j = 0; j < src_x.size(); j++) {
				float rinv1, m;
				const float dx = src_x[j] - sink_x[i];
				const float dy = src_y[j] - sink_y[i];
				const float dz = src_z[j] - sink_z[i];
				m = src_m[j];
				const float r2 = sqr(dx, dy, dz);
				const float r = sqrt(r2);                                                    // 4
				if (r2 > h2) {
					rinv1 = float(1) / r;                                                    // 5
				} else {
					const float r1overh1 = r * hinv;                                                    // 1
					const float r2oh2 = r1overh1 * r1overh1;                                                    // 1
					rinv1 = -5.0f / 16.0f;
					rinv1 = fmaf(rinv1, r2oh2, float(21.0f / 16.0f));                                                    // 2
					rinv1 = fmaf(rinv1, r2oh2, float(-35.0f / 16.0f));                                                    // 2
					rinv1 = fmaf(rinv1, r2oh2, float(35.0f / 16.0f));                                                    // 2
					rinv1 *= hinv;                                                    // 1
				}
				float this_phi = -m * rinv1;
				phi[i + self.parts.first] += this_phi;
			}
			phi[i + self.parts.first] *= GM;
		}
		if (tid == 0) {
			si = atomicAdd(next_sink_bucket, 1);
		}
		__syncwarp();
	}
	(&ws)->~bh_workspace();
}

__global__ void bh_tree_evaluate_points_kernel(bh_workspace* workspaces, bh_tree_node* tree_nodes, float* phi, array<float, NDIM>* parts,
		array<float, NDIM>* sinks, float theta, float hsoft, float GM, int* next_sink, int nsinks) {
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	__shared__ int si;
	if (tid == 0) {
		si = atomicAdd(next_sink, 1);
	}
	__syncwarp();
	bh_workspace& ws = workspaces[bid];
	new(&ws) bh_workspace();
	while (si < nsinks) {
		const array<float, NDIM>& sink = sinks[si];
		auto& checklist = ws.checklist;
		auto& nextlist = ws.nextlist;
		auto& src_x = ws.src_x;
		auto& src_y = ws.src_y;
		auto& src_z = ws.src_z;
		auto& src_m = ws.src_m;
		auto& leaflist = ws.leaflist;
		checklist.resize(1);
		checklist[0] = 0;
		const float thetainv = 1.0 / theta;
		while (checklist.size()) {
			int maxci = round_up(checklist.size(), WARP_SIZE);
			for (int ci = tid; ci < maxci; ci += WARP_SIZE) {
				bool source = false;
				bool leaf = false;
				bool next = false;
				if (ci < checklist.size()) {
					const auto& other = tree_nodes[checklist[ci]];
					const float other_radius = other.radius;
					const float dx = sink[XDIM] - other.pos[XDIM];
					const float dy = sink[YDIM] - other.pos[YDIM];
					const float dz = sink[ZDIM] - other.pos[ZDIM];
					const float r2 = sqr(dx, dy, dz);
					if (r2 > sqr(thetainv * (other_radius))) {
						source = true;
					} else if (other.children[LEFT] == -1) {
						leaf = true;
					} else {
						next = true;
					}
				}
				int total;
				int index;
				int offset;

				index = source;
				compute_indices(index, total);
				offset = src_x.size();
				src_x.resize(offset + total);
				src_y.resize(offset + total);
				src_z.resize(offset + total);
				src_m.resize(offset + total);
				index += offset;
				if (source) {
					const auto& other = tree_nodes[checklist[ci]];
					src_x[index] = other.pos[XDIM];
					src_y[index] = other.pos[YDIM];
					src_z[index] = other.pos[ZDIM];
					src_m[index] = other.mass;
				}

				index = leaf;
				compute_indices(index, total);
				offset = leaflist.size();
				leaflist.resize(offset + total);
				index += offset;
				if (leaf) {
					leaflist[index] = checklist[ci];
				}

				index = next;
				compute_indices(index, total);
				offset = nextlist.size();
				nextlist.resize(offset + 2 * total);
				index = 2 * index + offset;
				if (next) {
					nextlist[index + LEFT] = tree_nodes[checklist[ci]].children[LEFT];
					nextlist[index + RIGHT] = tree_nodes[checklist[ci]].children[RIGHT];
				}
			}
			checklist.resize(nextlist.size());
			for (int ci = tid; ci < nextlist.size(); ci += WARP_SIZE) {
				checklist[ci] = nextlist[ci];
			}

			nextlist.resize(0);
			for (int i = 0; i < leaflist.size(); i++) {
				const auto& other = tree_nodes[leaflist[i]];
				int offset = src_x.size();
				int new_size = offset + other.parts.second - other.parts.first;
				src_x.resize(new_size);
				src_y.resize(new_size);
				src_z.resize(new_size);
				src_m.resize(new_size);
				for (int j = tid + other.parts.first; j < other.parts.second; j += WARP_SIZE) {
					int k = j + offset - other.parts.first;
					src_x[k] = parts[j][XDIM];
					src_y[k] = parts[j][YDIM];
					src_z[k] = parts[j][ZDIM];
					src_m[k] = 1.0f;
				}
				__syncwarp();
			}
			leaflist.resize(0);
		}

		const float h = hsoft;
		const float hinv = 1.0 / (2.f * h);
		const float h2inv = 1.0 / (4.f * h * h);
		const float h2 = 4.f * h * h;
		if (tid == 0) {
			phi[si] = 0.f;
		}
		float this_phi = 0.0f;
		for (int j = tid; j < src_x.size(); j += WARP_SIZE) {
			float rinv1, m;
			const float dx = src_x[j] - sink[XDIM];
			const float dy = src_y[j] - sink[YDIM];
			const float dz = src_z[j] - sink[ZDIM];
			m = src_m[j];
			const float r2 = sqr(dx, dy, dz);
			const float r = sqrt(r2);                                                    // 4
			if (r2 > h2) {
				rinv1 = float(1) / r;                                                    // 5
			} else {
				const float r1overh1 = r * hinv;                                                    // 1
				const float r2oh2 = r1overh1 * r1overh1;                                                    // 1
				rinv1 = -5.0f / 16.0f;
				rinv1 = fmaf(rinv1, r2oh2, float(21.0f / 16.0f));                                                    // 2
				rinv1 = fmaf(rinv1, r2oh2, float(-35.0f / 16.0f));                                                    // 2
				rinv1 = fmaf(rinv1, r2oh2, float(35.0f / 16.0f));                                                    // 2
				rinv1 *= hinv;                                                    // 1
			}
			this_phi -= m * rinv1;
		}
		shared_reduce_add(this_phi);
		if (tid == 0) {
			phi[si] += this_phi;
			phi[si] *= GM;
			si = atomicAdd(next_sink, 1);
		}
		__syncwarp();
	}
	(&ws)->~bh_workspace();
}

vector<float> bh_evaluate_potential_gpu(const vector<bh_tree_node>& tree_nodes, const vector<array<float, NDIM>>& x, const vector<int> sink_buckets,
		float theta, float hsoft, float GM) {
	auto stream = cuda_get_stream();
	bh_tree_node* dev_tree_nodes;
	bh_workspace* workspaces;
	array<float, NDIM>* dev_x;
	int* dev_sink_buckets;
	float* dev_phi;
	int* next_sink_bucket;
	int zero = 0;
	int nblocks;
	timer tm1;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, (const void*) bh_tree_evaluate_kernel, WARP_SIZE, sizeof(bh_shmem)));
	nblocks *= cuda_smp_count();
	PRINT("%i blocks\n", nblocks);
	CUDA_CHECK(cudaMalloc(&workspaces, sizeof(bh_workspace) * nblocks));
	CUDA_CHECK(cudaMalloc(&dev_tree_nodes, sizeof(bh_tree_node) * tree_nodes.size()));
	CUDA_CHECK(cudaMalloc(&dev_x, sizeof(array<float, NDIM> ) * x.size()));
	CUDA_CHECK(cudaMalloc(&dev_sink_buckets, sizeof(int) * sink_buckets.size()));
	CUDA_CHECK(cudaMalloc(&dev_phi, sizeof(float) * x.size()));
	CUDA_CHECK(cudaMalloc(&next_sink_bucket, sizeof(int)));
	CUDA_CHECK(cudaMemcpyAsync(next_sink_bucket, &zero, sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpyAsync(dev_tree_nodes, tree_nodes.data(), sizeof(bh_tree_node) * tree_nodes.size(), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpyAsync(dev_x, x.data(), sizeof(array<float, NDIM> ) * x.size(), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpyAsync(dev_sink_buckets, sink_buckets.data(), sizeof(int) * sink_buckets.size(), cudaMemcpyHostToDevice));
	tm1.start();
	bh_tree_evaluate_kernel<<<nblocks, WARP_SIZE, 0, stream>>>(workspaces, dev_tree_nodes, dev_sink_buckets, dev_phi, dev_x,
			theta, hsoft, GM, next_sink_bucket, sink_buckets.size());
	vector<float> phi(x.size());
	cuda_stream_synchronize(stream);
	tm1.stop();
	PRINT("2 %e %i\n", tm1.read(), sink_buckets.size());
	CUDA_CHECK(cudaMemcpyAsync(phi.data(), dev_phi, sizeof(float) * x.size(), cudaMemcpyDeviceToHost));
	cuda_end_stream(stream);
	CUDA_CHECK(cudaFree(dev_tree_nodes));
	CUDA_CHECK(cudaFree(workspaces));
	CUDA_CHECK(cudaFree(dev_x));
	CUDA_CHECK(cudaFree(dev_sink_buckets));
	CUDA_CHECK(cudaFree(dev_phi));
	CUDA_CHECK(cudaFree(next_sink_bucket));

	return phi;
}

vector<float> bh_evaluate_potential_points_gpu(const vector<bh_tree_node>& tree_nodes, const vector<array<float, NDIM>>& x, const vector<array<float, NDIM>>& y,
		float theta, float hsoft, float GM) {
	auto stream = cuda_get_stream();
	bh_tree_node* dev_tree_nodes;
	bh_workspace* workspaces;
	array<float, NDIM>* dev_x;
	array<float, NDIM>* dev_y;
	float* dev_phi;
	int* next_sink;
	int zero = 0;
	int nblocks;
	timer tm1;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, (const void*) bh_tree_evaluate_kernel, WARP_SIZE, sizeof(bh_shmem)));
	nblocks *= cuda_smp_count();
	PRINT("%i blocks\n", nblocks);
	CUDA_CHECK(cudaMalloc(&workspaces, sizeof(bh_workspace) * nblocks));
	CUDA_CHECK(cudaMalloc(&dev_tree_nodes, sizeof(bh_tree_node) * tree_nodes.size()));
	CUDA_CHECK(cudaMalloc(&dev_x, sizeof(array<float, NDIM> ) * x.size()));
	CUDA_CHECK(cudaMalloc(&dev_y, sizeof(array<float, NDIM> ) * y.size()));
	CUDA_CHECK(cudaMalloc(&dev_phi, sizeof(float) * x.size()));
	CUDA_CHECK(cudaMalloc(&next_sink, sizeof(int)));
	CUDA_CHECK(cudaMemcpyAsync(next_sink, &zero, sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpyAsync(dev_tree_nodes, tree_nodes.data(), sizeof(bh_tree_node) * tree_nodes.size(), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpyAsync(dev_x, x.data(), sizeof(array<float, NDIM> ) * x.size(), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpyAsync(dev_y, y.data(), sizeof(array<float, NDIM> ) * y.size(), cudaMemcpyHostToDevice));
	tm1.start();
	bh_tree_evaluate_points_kernel<<<nblocks, WARP_SIZE, 0, stream>>>(workspaces, dev_tree_nodes, dev_phi, dev_x, dev_y,
			theta, hsoft, GM, next_sink, y.size());
	vector<float> phi(x.size());
	cuda_stream_synchronize(stream);
	tm1.stop();
	PRINT("2 %e %i\n", tm1.read(), y.size());
	CUDA_CHECK(cudaMemcpyAsync(phi.data(), dev_phi, sizeof(float) * y.size(), cudaMemcpyDeviceToHost));
	cuda_end_stream(stream);
	CUDA_CHECK(cudaFree(dev_tree_nodes));
	CUDA_CHECK(cudaFree(workspaces));
	CUDA_CHECK(cudaFree(dev_x));
	CUDA_CHECK(cudaFree(dev_y));
	CUDA_CHECK(cudaFree(dev_phi));
	CUDA_CHECK(cudaFree(next_sink));

	return phi;
}
