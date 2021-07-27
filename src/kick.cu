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
	cuda_vector<int> phase;
	cuda_vector<int> self;
	cuda_vector<array<fixed32, NDIM>> Lpos;
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

__global__ void cuda_kick_kernel(cuda_kick_params* params, int item_count, int* next_item) {

	const int& tid = threadIdx.x;
	extern __shared__ int shmem_ptr[];
	cuda_kick_shmem& shmem = *(cuda_kick_shmem*) shmem_ptr;
	auto& L = shmem.L;
	auto& dchecks = shmem.dchecks;
	auto& echecks = shmem.echecks;
	auto& nextlist = shmem.nextlist;
	auto& multlist = shmem.multlist;
	auto& partlist = shmem.partlist;
	auto& leaflist = shmem.leaflist;
	auto& phase = shmem.phase;
	auto& Lpos = shmem.Lpos;
	auto& self_index = shmem.self;
	L.initialize();
	dchecks.initialize();
	echecks.initialize();
	nextlist.initialize();
	multlist.initialize();
	leaflist.initialize();
	nextlist.initialize();
	phase.initialize();
	Lpos.initialize();
	self_index.initialize();
	int index;
	if (tid == 0) {
		index = atomicAdd(next_item, 1);
	}
	index = __shfl_sync(0xFFFFFFFF, index, 0);
	while (index < item_count) {
		L.resize(0);
		dchecks.resize(0);
		echecks.resize(0);
		L.push_back(params[index].L);
		dchecks.resize(params[index].dcount);
		echecks.resize(params[index].ecount);
		for (int i = tid; i < params[index].dcount; i += WARP_SIZE) {
			dchecks[i] = params[index].dchecks[i];
		}
		for (int i = tid; i < params[index].ecount; i += WARP_SIZE) {
			echecks[i] = params[index].echecks[i];
		}
		__syncwarp();
		kick_return kr;
		int depth = 0;
		phase.push_back(0);
		self_index.push_back(params[index].self);
		Lpos.push_back(params[index].Lpos);
		while (depth >= 0) {
			const auto& self = params[index].tree_nodes[self_index.back()];
			switch (phase.back()) {
			case 0: {
				if (self.sink_leaf) {
					phase.pop_back();
					self_index.pop_back();
					Lpos.pop_back();
					depth--;
				} else {
					const auto l = L.back();
					L.push_back(l);
					Lpos.push_back(self.pos);
					dchecks.push_top();
					echecks.push_top();
					phase.back()++;phase
					.push_back(0);
					self_index.push_back(self.children[LEFT].index);
					depth++;
				}

			}
				break;
			case 1: {
				L.pop_back();
				Lpos.push_back(self.pos);
				dchecks.pop_top();
				echecks.pop_top();
				phase.back()++;
				phase.push_back(0);
				self_index.push_back(self.children[RIGHT].index);
				depth++;
			}
				break;
			case 2: {
				self_index.pop_back();
				phase.pop_back();
				Lpos.pop_back();
				depth--;
			}
				break;
			}
		}

		if (tid == 0) {
			*(params[index].kreturn) = kr;
		}
		if (tid == 0) {
			index = atomicAdd(next_item, 1);
		}
		index = __shfl_sync(0xFFFFFFFF, index, 0);
	}
	assert(L.size()==1);
	assert(Lpos.size()==0);
	assert(phase.size()==0);
	assert(self_index.size()==0);
	L.initialize();
	dchecks.destroy();
	echecks.destroy();
	nextlist.destroy();
	multlist.destroy();
	leaflist.destroy();
	nextlist.destroy();
	phase.destroy();
	Lpos.destroy();
	self_index.destroy();
}

#define HEAP_SIZE (1024*1024*1024)

vector<kick_return, pinned_allocator<kick_return>> cuda_execute_kicks(kick_params kparams, fixed32* dev_x, fixed32* dev_y, fixed32* dev_z,
		tree_node* dev_tree_nodes, vector<kick_workitem> workitems, cudaStream_t stream) {
	timer tm;
	size_t value = HEAP_SIZE;
	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, value));
	CUDA_CHECK(cudaDeviceGetLimit(&value, cudaLimitMallocHeapSize));
	if (value != HEAP_SIZE) {
		THROW_ERROR("Unable to set heap to %li\n", HEAP_SIZE);
	}
	PRINT("shmem size = %i\n", sizeof(cuda_kick_shmem));
	tm.start();
	int* current_index;
	int zero = 0;
	CUDA_CHECK(cudaMalloc(&current_index, sizeof(int)));
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
	cuda_kick_params* dev_kick_params;
	CUDA_CHECK(cudaMalloc(&dev_kick_params, sizeof(cuda_kick_params) * kick_params.size()));
	CUDA_CHECK(cudaMemcpyAsync(dev_kick_params, kick_params.data(), sizeof(cuda_kick_params) * kick_params.size(), cudaMemcpyHostToDevice, stream));
	int nblocks;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, (const void*) cuda_kick_kernel, WARP_SIZE, sizeof(cuda_kick_shmem)));
	nblocks *= cuda_smp_count();
	nblocks = std::min(nblocks, (int) workitems.size());
	cuda_kick_kernel<<<nblocks, WARP_SIZE, sizeof(cuda_kick_shmem), stream>>>(dev_kick_params, kick_params.size(), current_index);
	CUDA_CHECK(cudaFreeAsync(current_index, stream));
	CUDA_CHECK(cudaFreeAsync(dev_kick_params, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));
	return returns;
}
