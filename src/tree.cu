/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2022  Dominic C. Marcello

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

#include <cosmictiger/tree.hpp>
#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/cuda_mem.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/fmm_kernels.hpp>

#define BLOCK_SIZE 32

__global__ void cuda_tree_sort_kernel(tree_sort_local_params* local_params, tree_sort_return* returns, tree_sort_global_params global_params);

cudaStream_t cuda_tree_sort(tree_sort_local_params* local_params, tree_sort_return* returns, tree_sort_global_params global_params) {
	auto stream = cuda_get_stream();
	int nblocks;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, (const void*) cuda_tree_sort_kernel, BLOCK_SIZE, 0));
	nblocks *= cuda_smp_count();
	nblocks = std::min(nblocks, global_params.N);
	cuda_tree_sort_kernel<<<nblocks,BLOCK_SIZE,0,stream>>>(local_params, returns, global_params);
	return stream;
}

struct tree_sort_shmem {
	int next_alloc;
	int last_alloc;
	device_vector<int> losnhi;
	device_vector<int> hisnlo;
	__device__ tree_sort_shmem() {
		if (threadIdx.x == 0) {
			next_alloc = last_alloc = 0;
		}
	}
	__device__ int allocate(tree_sort_global_params global_params) {
		int alloc;
		const int& tid = threadIdx.x;
		if (next_alloc == last_alloc) {
			if (tid == 0) {
				last_alloc = atomicAdd(global_params.next_alloc, -global_params.alloc_line_size);
				next_alloc = last_alloc - global_params.alloc_line_size;
			}
		}
		__syncthreads();
		alloc = next_alloc;
		__syncthreads();
		if (tid == 0) {
			next_alloc++;
		}
		__syncthreads();
		return alloc;
	}
};

#define PARTS_PER_BUCKET (8*BLOCK_SIZE)

template<class T>
__device__ inline void cswap(T& a, T& b) {
	const T c = a;
	a = b;
	b = c;
}

__device__ tree_sort_return cuda_tree_sort_node(const tree_sort_local_params& params, const tree_sort_global_params& global_params, tree_sort_shmem& shmem) {
	tree_sort_return rc;
	const int& tid = threadIdx.x;
	const auto& X = global_params.X;
	const auto& V = global_params.V;
	const auto& rungs = global_params.rungs;
	const auto xdim = params.box.longest_dim();
	const auto& xmin = params.box.begin[xdim];
	const auto& xmax = params.box.end[xdim];
	auto& losnhi = shmem.losnhi;
	auto& hisnlo = shmem.hisnlo;
	const auto& begin = params.part_range.first;
	const auto& end = params.part_range.second;
	const part_int nparts = end - begin;
	if (tid == 0) {
		//	PRINT("Entering %i %i parts\n", params.part_range.first, params.part_range.second);
	}
	range<fixed32> xbox;
	array<float, NDIM> dx;
	for (int dim = 0; dim < NDIM; dim++) {
		xbox.end[dim] = xbox.begin[dim] = X[dim][begin].to_double();
	}
	for (part_int i = tid + begin; i < end; i += BLOCK_SIZE) {
		for (int dim = 0; dim < NDIM; dim++) {
			const fixed32 x = X[dim][i];
			xbox.end[dim] = max(xbox.end[dim], x);
			xbox.begin[dim] = min(xbox.begin[dim], x);
		}
	}
	for (int dim = 0; dim < NDIM; dim++) {
		shared_reduce_min < BLOCK_SIZE > (xbox.begin[dim]);
		shared_reduce_max < BLOCK_SIZE > (xbox.end[dim]);
	}
	array<fixed32, NDIM> x_center;
	for (int dim = 0; dim < NDIM; dim++) {
		x_center[dim] = 0.5 * (xbox.begin[dim].to_double() + xbox.end[dim].to_double());
	}
	float rmax = 0.f;
	for (part_int i = tid + begin; i < end; i += BLOCK_SIZE) {
		for (int dim = 0; dim < NDIM; dim++) {
			dx[dim] = distance(X[dim][i], x_center[dim]);
		}
		rmax = fmaxf(rmax, sqr(dx[XDIM], dx[YDIM], dx[ZDIM]));
	}
	rmax = sqrtf(rmax);
	shared_reduce_max < BLOCK_SIZE > (rmax);
	rc.r = rmax;
	for (int dim = 0; dim < NDIM; dim++) {
		rc.pos[dim] = x_center[dim];
	}
	bool leaf;
	int node_count;
	array<tree_id, NCHILD> children;
	children[LEFT].proc = children[RIGHT].proc = global_params.rank;
	float box_r = 0.0f;
	for (int dim = 0; dim < NDIM; dim++) {
		box_r += sqr(0.5 * (params.box.end[dim] - params.box.begin[dim]));
	}
	box_r = sqrt(box_r);
	const bool ewald_satisfied = (box_r < 0.25 * (global_params.theta / (1.0 + global_params.theta)) && box_r < 0.125 - 0.25 * global_params.h);
	if (nparts > global_params.bucket_size || (!ewald_satisfied && nparts > 0)) {
		leaf = false;
		part_int mid;
		int count_lo = 0;
		const fixed32 xmid = 0.5 * (xmin + xmax);
		for (int i = begin + tid; i < end; i += BLOCK_SIZE) {
			if (X[xdim][i] < xmid) {
				count_lo++;
			}
		}
		shared_reduce_add<int, BLOCK_SIZE>(count_lo);
		mid = begin + count_lo;
		int lo = begin;
		int hi = end;
		losnhi.resize(0);
		while (lo < mid) {
			hisnlo.resize(0);
			while (lo < mid && hisnlo.size() < PARTS_PER_BUCKET) {
				__syncthreads();
				int index;
				bool flag = false;
				int total;
				const int j = lo + tid;
				if (j < mid) {
					flag = (X[xdim][j] >= xmid);
				}
				index = flag;
				compute_indices(index, total);
				const int start = hisnlo.size();
				hisnlo.resize(start + total);
				if (flag) {
					index += start;
					hisnlo[index] = j;
				}
				lo += BLOCK_SIZE;
			}
			__syncthreads();
			while (losnhi.size() < hisnlo.size()) {
				__syncthreads();
				int index;
				bool flag = false;
				int total;
				const int j = hi - tid - 1;
				if (j >= mid) {
					flag = (X[xdim][j] < xmid);
				}
				index = flag;
				compute_indices(index, total);
				const int start = losnhi.size();
				losnhi.resize(start + total);
				if (flag) {
					index += start;
					losnhi[index] = j;
				}
				hi -= BLOCK_SIZE;
			}
			__syncthreads();
			for (int j = tid; j < hisnlo.size(); j += BLOCK_SIZE) {
				const int k = hisnlo[j];
				const int l = losnhi[j];
				ALWAYS_ASSERT(k < l);
				for (int dim = 0; dim < NDIM; dim++) {
					cswap(X[dim][k], X[dim][l]);
					cswap(V[dim][k], V[dim][l]);
				}
				cswap(rungs[k], rungs[l]);
			}
			__syncthreads();
			const int nread = hisnlo.size();
			hisnlo.resize(losnhi.size() - hisnlo.size());
			for (int j = tid + nread; j < losnhi.size(); j += BLOCK_SIZE) {
				hisnlo[j - nread] = losnhi[j];
			}
			hisnlo.swap(losnhi);
		}
		ALWAYS_ASSERT(losnhi.size() == 0);
		tree_sort_local_params left_params, right_params;
		left_params.box = right_params.box = params.box;
		left_params.part_range = right_params.part_range = params.part_range;
		left_params.box.end[xdim] = right_params.box.begin[xdim] = xmid.to_double();
		left_params.part_range.second = right_params.part_range.first = mid;
		left_params.depth = right_params.depth = params.depth + 1;
		tree_sort_return rc_left;
		tree_sort_return rc_right;
		rc_left = cuda_tree_sort_node(left_params, global_params, shmem);
		rc_right = cuda_tree_sort_node(right_params, global_params, shmem);
		for (int dim = 0; dim < NDIM; dim++) {
			dx[dim] = distance(rc_left.pos[dim], x_center[dim]);
		}
		auto ML = M2M(rc_left.M, dx);
		for (int dim = 0; dim < NDIM; dim++) {
			dx[dim] = distance(rc_right.pos[dim], x_center[dim]);
		}
		auto MR = M2M(rc_right.M, dx);
		for (int i = 0; i < MULTIPOLE_SIZE; i++) {
			ALWAYS_ASSERT(isfinite(ML[i]));
			ALWAYS_ASSERT(isfinite(MR[i]));
			rc.M[i] = ML[i] + MR[i];
		}
		__syncthreads();
		children[LEFT].index = rc_left.index;
		children[RIGHT].index = rc_right.index;
		node_count = rc_left.node_count + rc_right.node_count;
	} else {
		leaf = true;
		multipole<float> myM;
		for (int i = 0; i < MULTIPOLE_SIZE; i++) {
			myM[i] = 0.f;
		}
		for (int i = tid + begin; i < end; i += BLOCK_SIZE) {
			for (int dim = 0; dim < NDIM; dim++) {
				dx[dim] = distance(X[dim][i], x_center[dim]);
			}
			auto this_M = P2M(dx);
			for (int i = 0; i < MULTIPOLE_SIZE; i++) {
				myM[i] += this_M[i];
			}
		}
		for (int i = 0; i < MULTIPOLE_SIZE; i++) {
			shared_reduce_add<float, BLOCK_SIZE>(myM[i]);
		}
		rc.M = myM;
		__syncthreads();
		node_count = 1;
		children[LEFT].index = children[RIGHT].index = -1;
	}
	const int ti = shmem.allocate(global_params);
	if (tid == 0) {
		auto& node = global_params.tree_nodes[ti];
		node.part_range.first = begin;
		node.part_range.second = end;
		node.depth = params.depth;
		node.leaf = leaf;
		node.index = ti;
		node.local_root = false;
		node.multi = rc.M;
		node.pos = rc.pos;
		node.node_count = node_count;
		node.proc_range.first = global_params.rank;
		node.proc_range.second = global_params.rank + 1;
		node.radius = rc.r;
		ALWAYS_ASSERT(!node.valid);
		node.valid = true;
		node.sink_part_range = node.part_range;
		node.children = children;
	}
	__syncthreads();
	rc.node_count = node_count;
	rc.index = ti;
	if (tid == 0) {
		//	PRINT("Leaving %i\n", params.depth);
	}

	return rc;
}

__global__ void cuda_tree_sort_kernel(tree_sort_local_params* local_params, tree_sort_return* returns, tree_sort_global_params global_params) {
	const int& tid = threadIdx.x;
	__shared__ int i;
	__shared__ tree_sort_shmem allocator;
	new (&allocator) tree_sort_shmem();
	if (tid == 0) {
		i = atomicAdd(global_params.index, 1);
//		PRINT("... %i\n", i);
	}
	__syncthreads();
	while (i < global_params.N) {
		__syncthreads();
		const auto return_ = cuda_tree_sort_node(local_params[i], global_params, allocator);
		if (tid == 0) {
			returns[i] = return_;
			i = atomicAdd(global_params.index, 1);
//			PRINT("??? %i\n", i);
		}
		__syncthreads();
	}
	(&allocator)->~tree_sort_shmem();

}
