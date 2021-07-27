#include <tigerfmm/gravity.hpp>

__device__
int cuda_gravity_cc(expansion<float>&, const tree_node&, gravity_cc_type, bool do_phi) {
	int flops = 0;
	return flops;
}

__device__
int cuda_gravity_cp(expansion<float>&, const tree_node&, bool do_phi) {
	int flops = 0;
	return flops;

}

__device__
int cuda_gravity_pc(const tree_node&, int, bool) {
	int flops = 0;
	return flops;

}

__device__
int cuda_gravity_pp(const tree_node&, int, float h, bool) {
	int flops = 0;
	return flops;

}
