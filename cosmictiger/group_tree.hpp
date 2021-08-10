#pragma once

#include <cosmictiger/defs.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/range.hpp>
#include <cosmictiger/tree.hpp>

using group_range = range<double>;

struct group_tree_node {
	group_range box;
	array<tree_id, NCHILD> children;
	pair<part_int> part_range;
	pair<int> proc_range;
	bool local_root;
	bool active;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & active;
		arc & box;
		arc & children;
		arc & part_range;
		arc & proc_range;
		arc & local_root;
	}
	inline particle_global_range global_part_range() const {
		particle_global_range r;
		r.proc = proc_range.first;
		r.range = part_range;
		return r;
	}
};

struct group_tree_return {
	tree_id id;
	group_range box;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & id;
		arc & box;
	}
};

group_tree_return group_tree_create(pair<int, int> proc_range = { 0, hpx_size() }, pair<part_int> part_range = { -1, -1 }, group_range box = unit_box<double>(),
		int depth = 0, bool local_root = hpx_size() == 1);
void group_tree_destroy();
const group_tree_node* group_tree_get_node(tree_id id);
void group_tree_set_active(tree_id, bool);
void group_tree_inc_cache_epoch();
