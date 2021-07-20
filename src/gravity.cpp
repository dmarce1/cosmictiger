constexpr bool verbose = true;
#include <tigerfmm/gravity.hpp>
#include <tigerfmm/safe_io.hpp>
#include <tigerfmm/tree.hpp>

void gravity_cc_ewald(const vector<tree_id>&) {

}

void gravity_cc(const vector<tree_id>&) {

}

void gravity_cp(const vector<tree_id>&) {

}

void gravity_pc(force_vectors& f, int min_rung, tree_id self, const vector<tree_id>&) {

}

void gravity_pp(force_vectors& f, int min_rung, tree_id self, const vector<tree_id>& list) {
	vector<const tree_node*> tree_ptrs(list.size());
	const tree_node* self_ptr = tree_get_node(self);
	const int nsink = self_ptr->nparts();
	int nsource = 0;
	for (int i = 0; i < list.size(); i++) {
		tree_ptrs[i] = tree_get_node(list[i]);
		nsource += tree_ptrs[i]->nparts();
	}
	vector<fixed32> srcx(nsource);
	vector<fixed32> srcy(nsource);
	vector<fixed32> srcz(nsource);
	int count = 0;
	for (int i = 0; i < tree_ptrs.size(); i++) {
		particles_global_read_pos(tree_ptrs[i]->global_part_range(), srcx, srcy, srcz, count);
		count += tree_ptrs[i]->nparts();
	}
//	PRINT( "%e %e %e\n", srcx[0].to_float(), srcy[0].to_float(), srcz[0].to_float());




}
