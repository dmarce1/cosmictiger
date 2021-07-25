#include <tigerfmm/kick.hpp>

#include <tigerfmm/cuda.hpp>
#include <tigerfmm/defs.hpp>
#include <tigerfmm/fixedcapvec.hpp>
#include <tigerfmm/kick.hpp>
#include <tigerfmm/particles.hpp>
#include <tigerfmm/stack_vector.hpp>

struct cuda_list_set {
	fixedcapvec<int, CUDA_CHECKLIST_SIZE> nextlist;
	fixedcapvec<int, CUDA_CHECKLIST_SIZE> partlist;
	fixedcapvec<int, CUDA_CHECKLIST_SIZE> multlist;
	fixedcapvec<int, CUDA_CHECKLIST_SIZE> leaflist;
	stack_vector<int, CUDA_STACK_SIZE, MAX_DEPTH> dchecklist;
	stack_vector<int, CUDA_STACK_SIZE, MAX_DEPTH> echecklist;
	array<expansion<float>, MAX_DEPTH> L;
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
	int self;
	cuda_list_set* lists;
	kick_return* kreturn;
	kick_params kparams;
};

__global__ void cuda_kick_kernel(cuda_kick_params* params) {

}

vector<kick_return, pinned_allocator<kick_return>> cuda_execute_kicks(kick_params params, fixed32* dev_x, fixed32* dev_y, fixed32* dev_z,
		tree_node* dev_tree_nodes, vector<kick_workitem> workitems, cudaStream_t stream) {
	vector<kick_return, pinned_allocator<kick_return>> returns(workitems.size());
	vector<cuda_kick_params, pinned_allocator<cuda_kick_params>> kick_params(workitems.size());
	vector<cuda_list_set, pinned_allocator<cuda_list_set>> lists;
	cuda_list_set* dev_lists;
	for (int i = 0; i < workitems.size(); i++) {
		for (int j = 0; j < workitems[i].dchecklist.size(); i++) {
			lists[i].dchecklist.push(workitems[i].dchecklist[i].index);
		}
		for (int j = 0; j < workitems[i].echecklist.size(); i++) {
			lists[i].echecklist.push(workitems[i].echecklist[i].index);
		}
		lists[i].L[0] = workitems[i].L;
	}
	CUDA_CHECK(cudaMalloc(&dev_lists, sizeof(cuda_list_set) * workitems.size()));
	CUDA_CHECK(cudaMemcpyAsync(dev_lists, lists.data(), sizeof(cuda_list_set) * workitems.size(), cudaMemcpyHostToDevice, stream));
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
		params.self = workitems[i].self.index;
		if (get_options().save_force) {
			params.gx = &particles_gforce(XDIM, 0);
			params.gy = &particles_gforce(YDIM, 0);
			params.gz = &particles_gforce(ZDIM, 0);
			params.pot = &particles_pot(0);
		} else {
			params.gx = params.gy = params.gz = params.pot = nullptr;
		}
		params.lists = dev_lists + i;
		params.kreturn = &returns[i];
		kick_params.push_back(std::move(params));
	}
	cuda_kick_kernel<<<workitems.size(), WARP_SIZE, 0, stream>>>(kick_params.data());
	cudaStreamSynchronize(stream);
	return returns;
}
