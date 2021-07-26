#include <tigerfmm/cuda.hpp>
#include <tigerfmm/cuda_reduce.hpp>
#include <tigerfmm/cuda_vector.hpp>
#include <tigerfmm/defs.hpp>
#include <tigerfmm/fixedcapvec.hpp>
#include <tigerfmm/fmm_kernels.hpp>
#include <tigerfmm/gravity.hpp>
#include <tigerfmm/kick.hpp>
#include <tigerfmm/particles.hpp>
#include <tigerfmm/stack_vector.hpp>
#include <tigerfmm/timer.hpp>

struct cuda_kick_shmem {
	cuda_vector<int> nextlist;
	cuda_vector<int> multlist;
	cuda_vector<int> partlist;
	cuda_vector<int> leaflist;
	cuda_vector<expansion<float>> L;
	stack_vector<int> dchecks;
	stack_vector<int> echecks;
};

struct cuda_kick_params {
	tree_node* tree_nodes;
	fixed32* x;
	fixed32* y;
	fixed32* z;
	float* vx;
	float* vy;
	float* vz;
	char* rungs;
	float* gx;
	float* gy;
	float* gz;
	float* pot;
	array<fixed32, NDIM> Lpos;
	expansion<float> L;
	int self;
	int* dchecks;
	int* echecks;
	int dcount;
	int ecount;
	kick_return* kreturn;
	kick_params kparams;
};

__device__ kick_return do_kick(cuda_kick_params* params, array<fixed32, NDIM> Lpos, const tree_node& self) {
	const int& tid = threadIdx.x;
	if (tid == 0) {
		//	PRINT("%i\n", self.nactive);
	}
	extern __shared__ int shmem_ptr[];
	cuda_kick_shmem& shmem = *(cuda_kick_shmem*) shmem_ptr;
	auto& nextlist = shmem.nextlist;
	auto& multlist = shmem.multlist;
	auto& partlist = shmem.partlist;
	auto& leaflist = shmem.leaflist;
	auto& dchecks = shmem.dchecks;
	auto& echecks = shmem.echecks;
	const float& h = params->kparams.h;
	const float thetainv = 1.f / params->kparams.theta;
	auto& L = shmem.L;
	kick_return kr;
	array<float, NDIM> dx;
	const float sink_bias = 1.5;
	for (int dim = 0; dim < NDIM; dim++) {
		dx[dim] = distance(self.pos[dim], Lpos[dim]);
	}
	L2L_cuda(L.back(), dx, params->kparams.min_rung == 0);
	nextlist.resize(0);
	multlist.resize(0);
	const int maxi = round_up(echecks.size(), WARP_SIZE);
	for (int i = tid; i < maxi; i += WARP_SIZE) {
		bool mult = false;
		bool next = false;
		if (i < echecks.size()) {
			const tree_node& other = params->tree_nodes[echecks[i]];
			for (int dim = 0; dim < NDIM; dim++) {
				dx[dim] = distance(self.pos[dim], other.pos[dim]);
			}
			float R2 = sqr(dx[XDIM], dx[YDIM], dx[ZDIM]);
			R2 = fmaxf(R2, EWALD_DIST2);
			const float r2 = sqr((sink_bias * self.radius + other.radius) * thetainv + h); // 5
			mult = R2 > r2;
			next = !mult;
		}
		int index;
		int total;
		int start;
		index = mult;
		compute_indices(index, total);
		start = multlist.size();
		multlist.resize(start + total);
		if (mult) {
			multlist[index + start] = echecks[i];
		}
		index = next;
		compute_indices(index, total);
		start = nextlist.size();
		nextlist.resize(start + total);
		if (next) {
			nextlist[index + start] = echecks[i];
		}
	}
	echecks.resize(NCHILD * nextlist.size());
	for (int i = tid; i < nextlist.size(); i += WARP_SIZE) {
		const auto children = params->tree_nodes[nextlist[i]].children;
		echecks[NCHILD * i + LEFT] = children[LEFT].index;
		echecks[NCHILD * i + RIGHT] = children[RIGHT].index;
	}
	nextlist.resize(0);
	partlist.resize(0);
	leaflist.resize(0);
	multlist.resize(0);
	do {
		const int maxi = round_up(dchecks.size(), WARP_SIZE);
		for (int i = tid; i < maxi; i += WARP_SIZE) {

			bool mult = false;
			bool next = false;
			bool leaf = false;
			bool part = false;
			if (i < dchecks.size()) {
				const tree_node& other = params->tree_nodes[dchecks[i]];
				for (int dim = 0; dim < NDIM; dim++) {
					dx[dim] = distance(self.pos[dim], other.pos[dim]);
				}
				const float R2 = sqr(dx[XDIM], dx[YDIM], dx[ZDIM]);
				const bool far1 = R2 > sqr((sink_bias * self.radius + other.radius) * thetainv + h);     // 5
				const bool far2 = R2 > sqr(sink_bias * self.radius * thetainv + other.radius + h);       // 5
//				PRINT("%e %e\n", R2, sqr((sink_bias * self.radius + other.radius) * thetainv + h));
				mult = far1;                                                                  // 4
				part = !mult && (far2 && other.source_leaf && (self.part_range.second - self.part_range.first) > MIN_CP_PARTS);
				leaf = !mult && !part && other.source_leaf;
				next = !mult && !part && !leaf;
			}
			int index;
			int total;
			int start;
			index = mult;
			compute_indices(index, total);
			start = multlist.size();
			multlist.resize(start + total);
			if (mult) {
				multlist[index + start] = dchecks[i];
			}
			index = next;
			compute_indices(index, total);
			start = nextlist.size();
			nextlist.resize(start + total);
			if (next) {
				nextlist[index + start] = dchecks[i];
			}
			index = part;
			compute_indices(index, total);
			start = partlist.size();
			partlist.resize(start + total);
			if (part) {
				partlist[index + start] = dchecks[i];
			}
			index = leaf;
			compute_indices(index, total);
			start = leaflist.size();
			leaflist.resize(start + total);
			if (leaf) {
				leaflist[index + start] = dchecks[i];
			}
		}
		dchecks.resize(NCHILD * nextlist.size());
		for (int i = tid; i < nextlist.size(); i += WARP_SIZE) {
			const int index = nextlist[i];
			const auto children = params->tree_nodes[index].children;
			dchecks[NCHILD * i + LEFT] = children[LEFT].index;
			dchecks[NCHILD * i + RIGHT] = children[RIGHT].index;
		}
		nextlist.resize(0);

	} while (dchecks.size() && self.sink_leaf);

	if (self.sink_leaf) {

	} else {
		const auto& children = self.children;
		const tree_node& cl = params->tree_nodes[children[LEFT].index];
		const tree_node& cr = params->tree_nodes[children[RIGHT].index];
		if (cl.nactive && cr.nactive) {
			L.push_back(L.back());
			dchecks.push_top();
			echecks.push_top();
			do_kick(params, self.pos, cl);
			L.pop_back();
			dchecks.pop_top();
			echecks.pop_top();
			do_kick(params, self.pos, cr);
		} else if (cl.nactive) {
			do_kick(params, self.pos, cl);
		} else if (cr.nactive) {
			do_kick(params, self.pos, cr);
		}
	}
//	PRINT("Returning \n");

	return kr;
}

__global__ void cuda_kick_kernel(cuda_kick_params* params, int item_count, int* next_item) {

	const int& tid = threadIdx.x;
	extern __shared__ int shmem_ptr[];
	cuda_kick_shmem& shmem = *(cuda_kick_shmem*) shmem_ptr;

	new (&shmem) cuda_kick_shmem();

	int index;
	if (tid == 0) {
		index = atomicAdd(next_item, 1);
	}
	index = __shfl_sync(0xFFFFFFFF, index, 0);
	while (index < item_count) {
		shmem.L.resize(0);
		shmem.dchecks.resize(0);
		shmem.echecks.resize(0);
		shmem.L.push_back(params[index].L);
		shmem.dchecks.resize(params[index].dcount);
		shmem.echecks.resize(params[index].ecount);
		for (int i = tid; i < params[index].dcount; i += WARP_SIZE) {
			shmem.dchecks[i] = params[index].dchecks[i];
		}
		for (int i = tid; i < params[index].ecount; i += WARP_SIZE) {
			shmem.echecks[i] = params[index].echecks[i];
		}
		kick_return kr = do_kick(params + index, params[index].Lpos, params[index].tree_nodes[params[index].self]);
		return;
		if (tid == 0) {
			*(params[index].kreturn) = kr;
		}
		if (tid == 0) {
			index = atomicAdd(next_item, 1);
		}
		index = __shfl_sync(0xFFFFFFFF, index, 0);
	}
	shmem.cuda_kick_shmem::~cuda_kick_shmem();

}

vector<kick_return, pinned_allocator<kick_return>> cuda_execute_kicks(kick_params kparams, fixed32* dev_x, fixed32* dev_y, fixed32* dev_z,
		tree_node* dev_tree_nodes, vector<kick_workitem> workitems, cudaStream_t stream) {
	timer tm;
	PRINT("shmem size = %i\n", sizeof(cuda_kick_shmem));
	tm.start();
	int* current_index;
	int zero = 0;
	CUDA_CHECK(cudaMallocAsync(&current_index, sizeof(int), stream));
	CUDA_CHECK(cudaMemcpyAsync(current_index, &zero, sizeof(int), cudaMemcpyHostToDevice, stream));
	vector<kick_return, pinned_allocator<kick_return>> returns(workitems.size());
	vector<cuda_kick_params, pinned_allocator<cuda_kick_params>> kick_params(workitems.size());
	vector<int, pinned_allocator<int>> dchecks;
	vector<int, pinned_allocator<int>> echecks;
	vector<int> dindices(workitems.size() + 1);
	vector<int> eindices(workitems.size() + 1);
	int dcount = 0;
	int ecount = 0;
	for (int i = 0; i < workitems.size(); i++) {
		//	PRINT( "%i\n", workitems[i].echecklist.size());
		dcount += workitems[i].dchecklist.size();
		ecount += workitems[i].echecklist.size();
	}
	dchecks.reserve(dcount);
	echecks.reserve(ecount);
	dcount = 0;
	ecount = 0;
	for (int i = 0; i < workitems.size(); i++) {
		dindices[i] = dcount;
		eindices[i] = ecount;
		for (int j = 0; j < workitems[i].dchecklist.size(); j++) {
			dchecks.push_back(workitems[i].dchecklist[j].index);
			dcount++;
		}
		for (int j = 0; j < workitems[i].echecklist.size(); j++) {
			echecks.push_back(workitems[i].echecklist[j].index);
			ecount++;
		}
	}
	dindices[workitems.size()] = dcount;
	eindices[workitems.size()] = ecount;
	tm.stop();

	for (int i = 0; i < workitems.size(); i++) {
		cuda_kick_params params;
		params.x = dev_x;
		params.y = dev_y;
		params.z = dev_z;
		params.tree_nodes = dev_tree_nodes;
		params.vx = &particles_vel(XDIM, 0);
		params.vy = &particles_vel(YDIM, 0);
		params.vz = &particles_vel(ZDIM, 0);
		params.rungs = &particles_rung(0);
		params.Lpos = workitems[i].pos;
		params.L = workitems[i].L;
		params.self = workitems[i].self.index;
		params.dchecks = dchecks.data() + dindices[i];
		params.echecks = echecks.data() + eindices[i];
		params.dcount = dindices[i + 1] - dindices[i];
		params.ecount = eindices[i + 1] - eindices[i];
		params.kparams = kparams;
		if (get_options().save_force) {
			params.gx = &particles_gforce(XDIM, 0);
			params.gy = &particles_gforce(YDIM, 0);
			params.gz = &particles_gforce(ZDIM, 0);
			params.pot = &particles_pot(0);
		} else {
			params.gx = params.gy = params.gz = params.pot = nullptr;
		}
		params.kreturn = &returns[i];
		kick_params[i] = std::move(params);
	}
	int nblocks;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, (const void*) cuda_kick_kernel, WARP_SIZE, sizeof(cuda_kick_shmem)));
	nblocks *= cuda_smp_count();
	nblocks = std::min(nblocks, (int) workitems.size());
	cuda_kick_kernel<<<nblocks, WARP_SIZE, sizeof(cuda_kick_shmem), stream>>>(kick_params.data(), kick_params.size(), current_index);
	CUDA_CHECK(cudaFreeAsync(current_index, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));
	return returns;
}
