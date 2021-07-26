#include <tigerfmm/kick.hpp>

#include <tigerfmm/cuda.hpp>
#include <tigerfmm/cuda_vector.hpp>
#include <tigerfmm/defs.hpp>
#include <tigerfmm/fixedcapvec.hpp>
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

__device__ kick_return do_kick(cuda_kick_params* params) {
	extern __shared__ int shmem_ptr[];
	cuda_kick_shmem& shmem = *(cuda_kick_shmem*) shmem_ptr;
	kick_return kr;

	return kr;
}

__global__ void cuda_kick_kernel(cuda_kick_params* params, int item_count, int* next_item) {
	extern __shared__ int shmem_ptr[];
	cuda_kick_shmem& shmem = *(cuda_kick_shmem*) shmem_ptr;
	const int tid = threadIdx.x;
	new (&shmem) cuda_kick_shmem();
	shmem.L.resize(MAX_DEPTH);
	int index = atomicAdd(next_item, 1);
	do {
		kick_return kr = do_kick(params + index);
		if( tid == 0 ) {
			*(params[index].kreturn) = kr;
		}
		index = atomicAdd(next_item, 1);
	} while (index < item_count);
	shmem.cuda_kick_shmem::~cuda_kick_shmem();

}

vector<kick_return, pinned_allocator<kick_return>> cuda_execute_kicks(kick_params params, fixed32* dev_x, fixed32* dev_y, fixed32* dev_z,
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
		dindices[i] = dcount;
		eindices[i] = ecount;
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
		if (get_options().save_force) {
			params.gx = &particles_gforce(XDIM, 0);
			params.gy = &particles_gforce(YDIM, 0);
			params.gz = &particles_gforce(ZDIM, 0);
			params.pot = &particles_pot(0);
		} else {
			params.gx = params.gy = params.gz = params.pot = nullptr;
		}
		params.kreturn = &returns[i];
		kick_params.push_back(std::move(params));
	}
	int nblocks;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, (const void*) cuda_kick_kernel, WARP_SIZE, sizeof(cuda_kick_shmem)));
	nblocks *= cuda_smp_count();
	nblocks = std::min(nblocks, (int) workitems.size());
	cuda_kick_kernel<<<nblocks, WARP_SIZE, 0, stream>>>(kick_params.data(), kick_params.size(), current_index);
	CUDA_CHECK(cudaFreeAsync(current_index, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));
	return returns;
}
