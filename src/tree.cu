#include <cosmictiger/cuda.hpp>
#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/device_vector.hpp>
#include <cosmictiger/fmm_kernels.hpp>
#include <cosmictiger/particles.hpp>

#define MAX_LO2HI 1024

#define MAX_KERNEL_COUNT 128
#define MAX_KERNEL_DEPTH 9

struct tree_create_data {
	tree_node* nodes;
	multi_pos* multis;
	fixed32* x;
	fixed32* y;
	fixed32* z;
	array<float, NDIM>* V;
	char* rungs;
	int num_kernels;
	int next_node;
	int alloc_size;
	int bucket_size;
	int proc;
	float soft_len;

};

template<class T>
__device__
void cswap(T& a, T& b) {
	const T c = a;
	a = b;
	b = c;
}

struct tree_allocator {
	int alloc_begin;
	int alloc_next;
	int alloc_end;
	tree_create_data* data;
	__device__
	void new_alloc() {
		alloc_next = alloc_begin = atomicAdd(&data->next_node, data->alloc_size);
		alloc_end = alloc_begin + data->alloc_size;
	}
	__device__ tree_allocator() {
	}
	__device__ tree_allocator(tree_create_data* data_) {
		data = data_;
		new_alloc();
	}
	__device__ int allocate() {
		if (alloc_next >= alloc_end) {
			new_alloc();
		}
		return alloc_next++;
	}
};

#define BLOCK_SIZE 64

struct cuda_shmem {
	tree_allocator allocator;
	device_vector<part_int> lo2hi;
	device_vector<part_int> hi2lo;
	tree_create_return tmp_rc;
	int kernel_num;
};

__global__ void cuda_tree_sort_kernel(tree_create_data* data, const tree_create_params params, pair<part_int> part_range, range<double> box, int depth,
		bool local_root, int kernel_depth, tree_create_return*);

__device__ tree_create_return cuda_tree_sort_local(cuda_shmem& shmem, tree_create_data* data, const tree_create_params& params, pair<part_int> part_range,
		range<double> box, int depth, bool local_root = false, int kernel_depth = 0) {
	const int& tid = threadIdx.x;
	auto& allocator = shmem.allocator;
	auto& lo2hi = shmem.lo2hi;
	auto& hi2lo = shmem.hi2lo;
	auto& tmp_rc = shmem.tmp_rc;
	auto& kernel_num = shmem.kernel_num;
	fixed32* X[] = { data->x, data->y, data->z };
	array<float, NDIM>* V = data->V;
	char* rungs = data->rungs;
	tree_create_return rc;
	const float& h = data->soft_len;
	float max_box_len = 0.0f;
	for (int dim = 0; dim < NDIM; dim++) {
		max_box_len += sqr(box.end[dim] - box.begin[dim]);
	}
	max_box_len = sqrtf(max_box_len);
	const part_int nparts = part_range.second - part_range.first;
	const bool ewald_satisfied = (max_box_len < 0.25f * (params.theta / (1.0f + params.theta)) && max_box_len < 0.125f - 0.25f * h); // 10
	const bool isleaf = !((nparts > data->bucket_size) || (!ewald_satisfied && nparts > 0));
	const int xdim = box.longest_dim();
	range<double> rbox;
	float radius;
	array<double, NDIM> Xc;
	multipole<float> M;
	array<fixed32, NDIM> Y;
	array<tree_id, NCHILD> children;
	const auto compute_radius = [&radius,&part_range,&X,&Xc]() {
		float max_radius2 = 0.f;
		for (part_int i = part_range.first; i < part_range.second; i += BLOCK_SIZE) {
			float this_radius2 = 0.f;
			for (int dim = 0; dim < NDIM; dim++) {
				const auto x = X[dim][i].to_double();
				this_radius2 += sqr(x - Xc[dim]);
			}
			max_radius2 = fmaxf(max_radius2, this_radius2);
		}
		shared_reduce_max < BLOCK_SIZE > (max_radius2);
		radius = sqrtf(max_radius2);
	};
	if (isleaf) {
		children[LEFT].index = children[RIGHT].index = -1;
		for (int dim = 0; dim < NDIM; dim++) {
			rbox.begin[dim] = box.end[dim];
			rbox.end[dim] = box.begin[dim];
		}
		for (part_int i = part_range.first; i < part_range.second; i += BLOCK_SIZE) {
			for (int dim = 0; dim < NDIM; dim++) {
				const auto x = X[dim][i].to_double();
				rbox.begin[dim] = min(rbox.begin[dim], x);
				rbox.end[dim] = min(rbox.end[dim], x);
			}
		}
		for (int dim = 0; dim < NDIM; dim++) {
			shared_reduce_min < BLOCK_SIZE > (rbox.begin[dim]);
			shared_reduce_max < BLOCK_SIZE > (rbox.end[dim]);
			Xc[dim] = (rbox.begin[dim] + rbox.end[dim]) * 0.5;
		}
		compute_radius();
		for (int dim = 0; dim < NDIM; dim++) {
			Y[dim] = Xc[dim];
		}
		for (int i = 0; i < MULTIPOLE_SIZE; i++) {
			M[i] = 0.f;
		}
		for (part_int i = part_range.first + tid; i < part_range.second; i += BLOCK_SIZE) {
			array<float, NDIM> dx;
			for (int dim = 0; dim < NDIM; dim++) {
				dx[dim] = distance(X[dim][i], Y[dim]);
			}
			auto m = P2M(dx);
			for (int i = 0; i < MULTIPOLE_SIZE; i++) {
				M[i] += m[i];
			}
		}
		for (int i = 0; i < MULTIPOLE_SIZE; i++) {
			shared_reduce_add<float, BLOCK_SIZE>(M[i]);
		}
	} else {
		const double xm = (box.begin[xdim] + box.end[xdim]) * 0.5;
		const fixed32 xmid = xm;
		part_int lo = part_range.first;
		part_int hi = part_range.second;
		part_int cntlo = 0;
		for (part_int i = lo + tid; i < hi; i += BLOCK_SIZE) {
			if (X[xdim][i] < xmid) {
				cntlo++;
			}
		}
		shared_reduce_add<part_int, BLOCK_SIZE>(cntlo);
		part_int mid = lo + cntlo;
		const part_int omid = mid;
		lo2hi.resize(0);
		hi2lo.resize(0);
		while (lo < mid) {
			auto maxi = round_up(mid, BLOCK_SIZE);
			for (part_int i = lo + tid; i < maxi && lo2hi.size() < MAX_LO2HI; i += BLOCK_SIZE) {
				bool ishi = false;
				if (i < mid) {
					if (X[xdim][i] >= xmid) {
						ishi = true;
					}
				}
				int index = ishi;
				int total;
				compute_indices < BLOCK_SIZE > (index, total);
				const auto start = lo2hi.size();
				lo2hi.resize(start + total);
				if (ishi) {
					lo2hi[start + index] = i;
				}
				__syncthreads();
				lo += BLOCK_SIZE;
			}
			maxi = round_up(hi, BLOCK_SIZE);
			for (part_int i = mid + tid; i < maxi && lo2hi.size() < MAX_LO2HI; i += BLOCK_SIZE) {
				bool islo = false;
				if (i < hi) {
					if (X[xdim][i] < xmid) {
						islo = true;
					}
				}
				int index = islo;
				int total;
				compute_indices < BLOCK_SIZE > (index, total);
				const auto start = hi2lo.size();
				hi2lo.resize(start + total);
				if (islo) {
					hi2lo[start + index] = i;
				}
				__syncthreads();
				mid += BLOCK_SIZE;
			}
			for (int i = tid; i < lo2hi.size(); i += BLOCK_SIZE) {
				const int& j = lo2hi[i];
				const int& k = hi2lo[i];
				for (int dim = 0; dim < NDIM; dim++) {
					cswap(X[dim][j], X[dim][k]);
				}
				cswap(V[j], V[k]);
				cswap(rungs[j], rungs[k]);
			}
			const int nremainder = hi2lo.size() - lo2hi.size();
			const int offset = lo2hi.size();
			lo2hi.resize(nremainder);
			for (int i = tid; i < nremainder; i += BLOCK_SIZE) {
				lo2hi[i] = hi2lo[offset + i];
			}
			hi2lo.resize(0);
			hi2lo.swap(lo2hi);
		}
		mid = omid;
		auto left_parts = part_range;
		auto right_parts = part_range;
		auto left_box = box;
		auto right_box = box;
		left_parts.second = right_parts.first = mid;
		left_box.end[xdim] = right_box.begin[xdim] = xmid.to_double();
		if (tid == 0) {
			kernel_num = atomicAdd(&data->num_kernels, 1);
		}
		__syncthreads();
		tree_create_return rcl, rcr;
		if (kernel_num < MAX_KERNEL_COUNT && kernel_depth < MAX_KERNEL_DEPTH) {
			if (tid == 0) {
				cuda_tree_sort_kernel<<<1,BLOCK_SIZE,sizeof(cuda_shmem)>>>(data, params, left_parts, left_box, depth + 1, false, kernel_depth + 1, &rcr);
			}
			rcl = cuda_tree_sort_local(shmem, data, params, left_parts, left_box, depth+1);
			if( tid == 0 ) {
				CUDA_CHECK(cudaDeviceSynchronize());
				atomicAdd(&data->num_kernels, -1);
				tmp_rc = rcr;
			}
			__syncthreads();
			rcr = tmp_rc;
			__syncthreads();
		} else {
			if( tid == 0 ) {
				atomicAdd(&data->num_kernels, -1);
			}
			rcr = cuda_tree_sort_local(shmem, data, params, right_parts, right_box, depth+1);
			rcl = cuda_tree_sort_local(shmem, data, params, left_parts, left_box, depth+1);
		}
		children[LEFT] = rcl.id;
		children[RIGHT] = rcr.id;
		for (int dim = 0; dim < NDIM; dim++) {
			rbox.begin[dim] = fmin(rcr.box.begin[dim], rcl.box.begin[dim]);
			rbox.end[dim] = fmax(rcr.box.end[dim], rcl.box.end[dim]);
			Xc[dim] = (rbox.begin[dim] + rbox.end[dim]) * 0.5;
		}
		if (nparts < 64 * data->bucket_size && max_box_len < 0.5) {
			compute_radius();
		} else {
			radius = 10.0f;
		}

		double r = 0.0;
		array<double, NDIM> N;
		double norminv = 0.0;
		array<double, NDIM> Xl;
		array<double, NDIM> Xc2;
		array<double, NDIM> Xr;
		const auto& mr = rcr.multi;
		const auto& ml = rcl.multi;
		const auto& Rr = rcr.radius;
		const auto& Rl = rcl.radius;
		for (int dim = 0; dim < NDIM; dim++) {
			Xl[dim] = rcl.pos[dim].to_double();                      // 3
			Xr[dim] = rcr.pos[dim].to_double();                      // 3
		}
		for (int dim = 0; dim < NDIM; dim++) {
			N[dim] = Xl[dim] - Xr[dim];                         // 3
			norminv += sqr(N[dim]);                         // 6
		}
		norminv = 1.0 / sqrt(norminv);                    // 8
		for (int dim = 0; dim < NDIM; dim++) {
			N[dim] *= norminv;                                  // 3
			N[dim] = fabs(N[dim]);                                  // 3
		}
		if (mr[0] != 0.0) {
			if (ml[0] != 0.0) {
				for (int dim = 0; dim < NDIM; dim++) {
					const double xmax = std::max(Xl[dim] + N[dim] * Rl, Xr[dim] + N[dim] * Rr);
					const double xmin = std::min(Xl[dim] - N[dim] * Rl, Xr[dim] - N[dim] * Rr);
					Xc2[dim] = (xmax + xmin) * 0.5;
					r += sqr((xmax - xmin) * 0.5);
				}
			} else {
				Xc2 = Xr;
				r = Rr * Rr;
			}
		} else {
			if (ml[0] != 0.0) {
				Xc2 = Xl;
				r = Rl * Rl;
			} else {
				ALWAYS_ASSERT(false);
			}
		}
		float radius2 = sqrtf(r);                                         // 4
		if (radius > radius2) {
			Xc = Xc2;
			radius = radius2;
		}
		r = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			const double span = (rbox.end[dim] - rbox.begin[dim]);
			r += sqr(span * 0.5);            // 12
		}
		r = sqrtf(r);                                              // 4
		if (r < radius) {                                              // 1
			radius = r;
			for (int dim = 0; dim < NDIM; dim++) {
				Xc[dim] = (rbox.begin[dim] + rbox.end[dim]) * 0.5;         // 6
			}
		}
		array<float, NDIM> dx;
		for (int dim = 0; dim < NDIM; dim++) {
			Y[dim] = Xc[dim];
		}
		for (int dim = 0; dim < NDIM; dim++) {
			const auto& x = tid == 0 ? rcr.pos[dim] : rcl.pos[dim];
			dx[dim] = distance(x, Y[dim]);
		}
		const auto& this_m = tid == 0 ? rcr.multi : rcl.multi;
		M = M2M(this_m, dx);
		for (int i = 0; i < MULTIPOLE_SIZE; i++) {
			M[i] += __shfl_down_sync(0xFFFFFFFF, M[i], tid == 0);
		}
	}
	if (tid == 0) {
		const int my_index = allocator.allocate();
		rc.box = rbox;
		rc.id.index = my_index;
		rc.id.proc = data->proc;
		rc.radius = radius;
		rc.multi = M;
		for (int dim = 0; dim < NDIM; dim++) {
			rc.pos[dim] = Xc[dim];
		}
		tmp_rc = rc;
		data->multis[my_index].multi = M;
		data->multis[my_index].pos = rc.pos;
		auto& node = data->nodes[my_index];
		for (int dim = 0; dim < NDIM; dim++) {
			node.box.begin[dim] = rbox.begin[dim];
			node.box.end[dim] = rbox.end[dim];
		}
		node.leaf = isleaf;
		node.children = children;
		node.depth = depth;
		node.local_root = local_root;
		node.mpos = data->multis + my_index;
		node.part_range = part_range;
		node.pos = rc.pos;
		node.radius = rc.radius;
	}
	__syncthreads();
	rc = tmp_rc;
	__syncthreads();

}
__global__ void cuda_tree_sort_kernel(tree_create_data* data, const tree_create_params params, pair<part_int> part_range, range<double> box, int depth,
		bool local_root, int kernel_depth, tree_create_return* rc_ptr) {
	const int& tid = threadIdx.x;

	extern __shared__ int shmem_ptr[];
	auto& shmem = *((cuda_shmem*) (shmem_ptr));
	auto& allocator = shmem.allocator;
	auto& lo2hi = shmem.lo2hi;
	auto& hi2lo = shmem.hi2lo;
	new (&allocator) tree_allocator(data);
	new (&lo2hi) device_vector<part_int>;
	new (&hi2lo) device_vector<part_int>;
	auto rc = cuda_tree_sort_local(shmem, data, params, part_range, box, depth, local_root, kernel_depth);
	if (tid == 0) {
		*rc_ptr = rc;
	}
	__syncthreads();
	(&lo2hi)->~device_vector<part_int>();
	(&hi2lo)->~device_vector<part_int>();
	(&allocator)->~tree_allocator();
}

tree_create_return cuda_tree_sort(tree_node* nodes, multi_pos* multis, int next_node, const tree_create_params params, range<double> box, int depth) {
	auto* rc_ptr = (tree_create_return*) cuda_malloc(sizeof(tree_create_return));
	auto* data_ptr = (tree_create_data*) cuda_malloc(sizeof(tree_create_data));
	tree_create_return rc;
	data_ptr->V = particles_vel_data();
	data_ptr->x = &particles_pos(XDIM, 0);
	data_ptr->y = &particles_pos(YDIM, 0);
	data_ptr->z = &particles_pos(ZDIM, 0);
	data_ptr->alloc_size = get_options().tree_alloc_line_size;
	data_ptr->bucket_size = get_options().bucket_size;
	data_ptr->multis = multis;
	data_ptr->next_node = next_node;
	data_ptr->nodes = nodes;
	data_ptr->num_kernels = 1;
	data_ptr->proc = hpx_rank();
	data_ptr->rungs = &particles_rung(0);
	data_ptr->soft_len = get_options().hsoft;
	const auto part_range = particles_current_range();
	cuda_tree_sort_kernel<<<1,BLOCK_SIZE>>>(data_ptr, params, part_range, box, depth, true, 0, rc_ptr);
	CUDA_CHECK(cudaDeviceSynchronize());
	rc = *rc_ptr;
	cuda_free(rc_ptr);
	cuda_free(data_ptr);

}

