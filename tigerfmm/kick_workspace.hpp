/*
 * kick_workspace.hpp
 *
 *  Created on: Jul 24, 2021
 *      Author: dmarce1
 */

#ifndef KICK_WORKSPACE_HPP_
#define KICK_WORKSPACE_HPP_

#include <tigerfmm/cuda.hpp>
#include <tigerfmm/defs.hpp>
#include <tigerfmm/kick.hpp>
#include <tigerfmm/tree.hpp>

#include <unordered_map>

struct kick_workspace_tree_id_hash {
	inline size_t operator()(tree_id id) const {
		const int line_size = get_options().tree_cache_line_size;
		const int i = id.index / line_size;
		return i * hpx_size() + id.proc;
	}
};

struct kick_workspace {
	vector<tree_node, pinned_allocator<tree_node>> tree_space;
	fixed32* dev_x;
	fixed32* dev_y;
	fixed32* dev_z;
	tree_node* dev_tree_space;
	std::atomic<int> lock;
	int current_tree;
	int current_part;
	int nparts;
	int ntrees;
	cudaStream_t stream;
	std::unordered_map<tree_id, int, kick_workspace_tree_id_hash> tree_map;

	kick_workspace();
	kick_workspace(const kick_workspace&) = delete;
	kick_workspace(kick_workspace&&) = delete;
	kick_workspace& operator=(const kick_workspace&) = delete;
	kick_workspace& operator=(kick_workspace&&) = delete;
	~kick_workspace();

	bool add_tree_list(vector<tree_id>& nodes);
	void add_tree_node(tree_id, int part_base, int tree_base);
	void add_tree_node_descendants(tree_id, int part_offset, int& index);
	void to_gpu();
};
#endif /* KICK_WORKSPACE_HPP_ */
