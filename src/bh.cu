#include <cosmictiger/bh.hpp>
#include <cosmictiger/cuda.hpp>
#include <cosmictiger/fixedcapvec.hpp>
#include <cosmictiger/stack_vector.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/device_vector.hpp>
#include <cosmictiger/gravity.hpp>

#define BH_LIST_SIZE 1024
#define BH_SOURCE_LIST_SIZE (16*1024)
#define BH_STACK_SIZE 16384
#define BH_MAX_DEPTH 32
#define BH_WORKSIZE 1024

struct bh_workspace {
	device_vector<int> nextlist;
	device_vector<int> leaflist;
	device_vector<float> src_x;
	device_vector<float> src_y;
	device_vector<float> src_z;
	device_vector<float> src_m;
	stack_vector<int> checklist;
};

struct bh_shmem {
	int si;
	array<float, BH_BUCKET_SIZE> sink_x;
	array<float, BH_BUCKET_SIZE> sink_y;
	array<float, BH_BUCKET_SIZE> sink_z;
};

__global__ void bh_tree_evaluate_kernel(const bh_tree_node* tree_nodes, const int* sink_buckets, float* phi, const array<float, NDIM>* parts, float theta,
		float hsoft, float GM, int* next_sink_bucket, int nsinks) {
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
	__shared__ bh_workspace ws;
	new (&ws) bh_workspace();
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
				__syncwarp();
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
				__syncwarp();
				leaflist.resize(offset + total);
				index += offset;
				if (leaf) {
					leaflist[index] = checklist[ci];
				}

				index = next;
				compute_indices(index, total);
				offset = nextlist.size();
				__syncwarp();
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
		for (int i = 0; i < sink_size; i++) {
			float this_phi = 0.0;
			for (int j = tid; j < src_x.size(); j += WARP_SIZE) {
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
					rinv1 = 3.0f / 8.0f;
					rinv1 = fmaf(rinv1, r2oh2, float(-5.0f / 4.0f));                                                    // 2
					rinv1 = fmaf(rinv1, r2oh2, float(15.0f / 8.0f));                                                    // 2
					rinv1 *= hinv;                                                    // 1
				}
				this_phi += -m * rinv1;
			}
			shared_reduce_add(this_phi);
			if( tid == 0 ) {
				this_phi -= SELF_PHI * hinv;
				phi[i + self.parts.first] = GM * this_phi;
			}
		}
		if (tid == 0) {
			si = atomicAdd(next_sink_bucket, 1);
		}
		__syncwarp();
	}
	(&ws)->~bh_workspace();
}

__global__ void bh_tree_evaluate_points_kernel(const bh_tree_node* tree_nodes, float* phi, const array<float, NDIM>* parts, const array<float, NDIM>* sinks,
		float theta, float hsoft, float GM, int* next_sink, int nsinks) {
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	__shared__ int si;
	if (tid == 0) {
		si = atomicAdd(next_sink, 1);
	}
	__syncwarp();
	__shared__ bh_workspace ws;
	new (&ws) bh_workspace();
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
				__syncwarp();
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
				__syncwarp();
				leaflist.resize(offset + total);
				index += offset;
				if (leaf) {
					leaflist[index] = checklist[ci];
				}

				index = next;
				compute_indices(index, total);
				offset = nextlist.size();
				__syncwarp();
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
			} else if (r2 > 0.f) {
				const float r1overh1 = r * hinv;                                                    // 1
				const float r2oh2 = r1overh1 * r1overh1;                                                    // 1
				rinv1 = 3.0f / 8.0f;
				rinv1 = fmaf(rinv1, r2oh2, float(-5.0f / 4.0f));                                                    // 2
				rinv1 = fmaf(rinv1, r2oh2, float(15.0f / 8.0f));                                                    // 2
				rinv1 *= hinv;                                                    // 1
			} else {
				rinv1 = 0.0f;
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

device_vector<float> bh_evaluate_potential_gpu(const device_vector<bh_tree_node>& dev_tree_nodes, const device_vector<array<float, NDIM>>& dev_x,
		const device_vector<int> dev_sink_buckets, float theta, float hsoft, float GM) {
	auto stream = cuda_get_stream();
	bh_workspace* workspaces;
	int* next_sink_bucket;
	int zero = 0;
	int nblocks;
	timer tm1;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, (const void*) bh_tree_evaluate_kernel, WARP_SIZE, sizeof(bh_shmem)));
	nblocks *= cuda_smp_count();
	nblocks = std::min(nblocks, dev_sink_buckets.size());
//	nblocks *= cuda_smp_count();
	next_sink_bucket = (int*) cuda_malloc(sizeof(int));
	*next_sink_bucket = 0;
	tm1.start();
	device_vector<float> dev_phi(dev_x.size());
	bh_tree_evaluate_kernel<<<nblocks, WARP_SIZE, 0, stream>>>(dev_tree_nodes.data(), dev_sink_buckets.data(), dev_phi.data(), dev_x.data(),
			theta, hsoft, GM, next_sink_bucket, dev_sink_buckets.size());
	cuda_stream_synchronize(stream);
	tm1.stop();
	cuda_end_stream(stream);
	cuda_free(next_sink_bucket);
	return dev_phi;
}

device_vector<float> bh_evaluate_potential_points_gpu(const device_vector<bh_tree_node>& dev_tree_nodes, const device_vector<array<float, NDIM>>& dev_x,
		const device_vector<array<float, NDIM>>& dev_y, float theta, float hsoft, float GM) {
	auto stream = cuda_get_stream();
	int* next_sink;
	int zero = 0;
	int nblocks;
	timer tm1;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, (const void*) bh_tree_evaluate_kernel, WARP_SIZE, sizeof(bh_shmem)));
	nblocks *= cuda_smp_count();
	nblocks = std::max(1,std::min(nblocks, dev_y.size() / BH_BUCKET_SIZE));
	next_sink = (int*) cuda_malloc(sizeof(int));
	*next_sink = 0;
	tm1.start();
	device_vector<float> dev_phi(dev_y.size());
	bh_tree_evaluate_points_kernel<<<nblocks, WARP_SIZE, 0, stream>>>(dev_tree_nodes.data(), dev_phi.data(), dev_x.data(), dev_y.data(),
			theta, hsoft, GM, next_sink, dev_y.size());
	cuda_stream_synchronize(stream);
	tm1.stop();
	ALWAYS_ASSERT(dev_y.size());
	cuda_end_stream(stream);
	cuda_free(next_sink);
	return dev_phi;
}
