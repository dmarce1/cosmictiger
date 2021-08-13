#include <cosmictiger/bh.hpp>
#include <cosmictiger/cuda.hpp>
#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/math.hpp>

#define BH_LIST_SIZE 32767

struct bh_lists {
	fixedcapvec<int, 4095> checklist;
	fixedcapvec<int, 2048> nextlist;
	fixedcapvec<bh_source, 8192> sourcelist;
	fixedcapvec<int, 4096> leaflist;
};

__global__ void bh_kernel(bh_lists* lists, bh_tree_node* nodes, int* sink_buckets, array<float, NDIM>* parts, float* phi, int nsink_buckets, int* current,
		float theta, float h, float GM) {

	const int& tid = threadIdx.x;
	const int& bid = blockIdx.x;
	__shared__ int index;
	const float thetainv = 1.f / theta;
	const float hinv = 1.0f / h;
	const float h2 = h * h;
	const float h2inv = hinv * hinv;

	if (tid == 0) {
		index = atomicAdd(current, 1);
	}
	__syncwarp();
	while (index < nsink_buckets) {
		auto& checklist = lists[bid].checklist;
		auto& nextlist = lists[bid].nextlist;
		auto& sourcelist = lists[bid].sourcelist;
		auto& leaflist = lists[bid].leaflist;
		nextlist.resize(0);
		sourcelist.resize(0);
		leaflist.resize(0);
		checklist.resize(1);
		checklist[0] = 0;
		const auto& mynode = nodes[sink_buckets[index]];
		const float& myradius = mynode.radius;
		const auto& mypos = mynode.pos;
		const auto& myparts = mynode.parts;
		for (int i = myparts.first + tid; i < myparts.second; i += WARP_SIZE) {
			phi[i] = -SELF_PHI * hinv;
		}
		while (checklist.size()) {
			const int maxi = round_up(checklist.size(), WARP_SIZE);
			for (int ci = tid; ci < maxi; ci += WARP_SIZE) {
				bool next = false;
				bool pc = false;
				bool pp = false;
				float r2;
				const bh_tree_node* node_ptr;
				if (ci < checklist.size()) {
					node_ptr = &nodes[checklist[ci]];
					const float dx = mypos[XDIM] - node_ptr->pos[XDIM];
					const float dy = mypos[YDIM] - node_ptr->pos[YDIM];
					const float dz = mypos[ZDIM] - node_ptr->pos[ZDIM];
					r2 = sqr(dx, dy, dz);
					if (r2 > sqr(thetainv * (node_ptr->radius + myradius))) {
						pc = true;
					} else if (node_ptr->children[LEFT] == -1) {
						pp = true;
					} else {
						next = true;
					}
				}
				int total;
				int index;

				index = next;
				compute_indices(index, total);
				index += nextlist.size();
				nextlist.resize(nextlist.size() + total);
				if (next) {
					nextlist[index] = checklist[ci];
				}

				index = pc;
				compute_indices(index, total);
				index += sourcelist.size();
				sourcelist.resize(sourcelist.size() + total);
				if (pc) {
					bh_source src;
					src.x = node_ptr->pos;
					src.m = node_ptr->mass;
					sourcelist[index] = src;
				}

				index = pp;
				compute_indices(index, total);
				index += leaflist.size();
				leaflist.resize(leaflist.size() + total);
				if (pp) {
					leaflist[index] = checklist[ci];
				}

			}
			checklist.resize(NCHILD * nextlist.size());
			for (int i = tid; i < nextlist.size(); i += WARP_SIZE) {
				const auto& this_node = nodes[nextlist[i]];
				checklist[NCHILD * i + LEFT] = this_node.children[LEFT];
				checklist[NCHILD * i + RIGHT] = this_node.children[RIGHT];
			}
			nextlist.resize(0);
			for (int i = 0; i < leaflist.size(); i++) {
				const auto& other_node = nodes[leaflist[i]];
				const auto& other_parts = other_node.parts;
				const auto other_size = other_parts.second - other_parts.first;
				const int offset = sourcelist.size();
				sourcelist.resize(other_size + offset);
				for (int j = other_parts.first + tid; i < other_parts.second; i += WARP_SIZE) {
					bh_source src;
					src.x = parts[j];
					src.m = 1.f;
					sourcelist[offset + j - other_parts.first] = src;
				}
			}
			leaflist.resize(0);
			__syncwarp();

		}
		for (int i = 0; i < sourcelist.size(); i++) {
			for (int j = myparts.first + tid; j < myparts.second; j += WARP_SIZE) {
				const auto& sink = parts[j];
				const auto& source = sourcelist[i];
				const auto& m = source.m;
				const float dx = sink[XDIM] - source.x[XDIM];
				const float dy = sink[YDIM] - source.x[YDIM];
				const float dz = sink[ZDIM] - source.x[ZDIM];
				const float r2 = sqr(dx, dy, dz);
				if (r2 > h2) {
					phi[j] -= m * rsqrt(r2);
				} else {
					const float q2 = r2 * h2inv;
					float rinv = -5.0f / 16.0f;
					rinv = fmaf(rinv, q2, 21.0f / 16.0f);
					rinv = fmaf(rinv, q2, -35.0f / 16.0f);
					rinv = fmaf(rinv, q2, 35.0f / 16.0f);
					rinv *= hinv;
					phi[j] -= m * rinv;
				}

			}
		}
		for (int i = myparts.first + tid; i < myparts.second; i += WARP_SIZE) {
			phi[i] *= GM;
		}

		if (tid == 0) {
			index = atomicAdd(current, 1);
		}
		__syncwarp();
	}

}

vector<float> bh_cuda_tree_evaluate(const vector<bh_tree_node>& nodes, vector<int>& sink_buckets, vector<array<float, NDIM>>& parts, float theta) {
	vector<float> phi;
//	PRINT( "%i %i\n", sinks.size(), nodes.size());
	bh_tree_node* dev_nodes;
	array<float, NDIM>* dev_parts;
	float* dev_phi;
	int* dev_current;
	int* dev_sink_buckets;
	int zero = 0;
	bh_lists* dev_lists;
	int blocks_per;
	int max_blocks;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per, (const void*) bh_kernel, WARP_SIZE, sizeof(int)));
	max_blocks = cuda_smp_count() * blocks_per;
//	blocks *= std::min(std::max((cuda_smp_count() - 1) / hpx_hardware_concurrency() + 1, 1), max_blocks);
	int blocks = std::min((int) sink_buckets.size(), max_blocks);
	CUDA_CHECK(cudaMalloc(&dev_lists, sizeof(bh_lists) * blocks));
	CUDA_CHECK(cudaMalloc(&dev_nodes, sizeof(bh_tree_node) * nodes.size()));
	CUDA_CHECK(cudaMalloc(&dev_phi, sizeof(float) * parts.size()));
	CUDA_CHECK(cudaMalloc(&dev_current, sizeof(int)));
	CUDA_CHECK(cudaMalloc(&dev_sink_buckets, sizeof(int) * sink_buckets.size()));
	CUDA_CHECK(cudaMalloc(&dev_parts, sizeof(array<float, NDIM> ) * parts.size()));
	auto stream = cuda_get_stream();
	CUDA_CHECK(cudaMemcpyAsync(dev_nodes, nodes.data(), sizeof(bh_tree_node) * nodes.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(dev_parts, parts.data(), sizeof(array<float, NDIM> ) * parts.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(dev_sink_buckets, sink_buckets.data(), sizeof(int) * sink_buckets.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(dev_current, &zero, sizeof(int), cudaMemcpyHostToDevice, stream));
	bh_kernel<<<blocks,WARP_SIZE,0,stream>>>(dev_lists, dev_nodes, dev_sink_buckets, dev_parts, dev_phi, sink_buckets.size(), dev_current, 0.5, get_options().hsoft, get_options().GM);
	phi.resize(parts.size());
	CUDA_CHECK(cudaMemcpyAsync(phi.data(), dev_phi, sizeof(float) * phi.size(), cudaMemcpyDeviceToHost, stream));
	while (cudaStreamQuery(stream) != cudaSuccess) {
		hpx_yield();
	}
	cuda_end_stream(stream);
	CUDA_CHECK(cudaFree(dev_nodes));
	CUDA_CHECK(cudaFree(dev_parts));
	CUDA_CHECK(cudaFree(dev_current));
	CUDA_CHECK(cudaFree(dev_lists));

	return phi;
}

