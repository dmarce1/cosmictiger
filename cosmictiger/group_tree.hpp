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
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & box;
		arc & children;
		arc & part_range;
		arc & proc_range;
		arc & local_root;
	}
};

tree_id group_tree_create(pair<int, int> proc_range, pair<part_int> part_range, group_range box, int depth, bool local_root);
