/*
 * kick_workspace.hpp
 *
 *  Created on: Jul 24, 2021
 *      Author: dmarce1
 */

#ifndef KICK_WORKSPACE_HPP_
#define KICK_WORKSPACE_HPP_

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/kick.hpp>
#include <cosmictiger/range_set.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/unordered_set_ts.hpp>

#include <unordered_map>
#include <set>

struct kick_workspace_tree_id_hash {
	inline size_t operator()(tree_id id) const {
		return id.index * hpx_size() + id.proc;
	}
};

#ifndef __CUDACC__

class kick_workspace {
	mutex_type mutex;
	vector<kick_workitem> workitems;
	vector<hpx::lcos::local::promise<kick_return>> promises;
	part_int total_parts;
	part_int nparts;
	kick_params params;
	std::unordered_set<tree_id, kick_workspace_tree_id_hash> tree_ids;
	static vector<fixed32, pinned_allocator<fixed32>> host_x;
	static vector<fixed32, pinned_allocator<fixed32>> host_y;
	static vector<fixed32, pinned_allocator<fixed32>> host_z;
	static vector<tree_node, pinned_allocator<tree_node>> tree_nodes;
	static hpx::lcos::local::counting_semaphore lock1;
	static hpx::lcos::local::counting_semaphore lock2;
public:
	static void clear_buffers();
	kick_workspace() = default;
	kick_workspace(kick_params, part_int);
	kick_workspace(const kick_workspace&) = delete;
	kick_workspace(kick_workspace&&) = delete;
	kick_workspace& operator=(const kick_workspace&) = delete;
	kick_workspace& operator=(kick_workspace&&) = delete;
	~kick_workspace();
	template<class A>
	void serialize(A&&, unsigned) {
	}
	hpx::future<kick_return> add_work(std::shared_ptr<kick_workspace> ptr, expansion<float> L, array<fixed32, NDIM> pos, tree_id self,
			vector<tree_id> && dchecklist, vector<tree_id> && echecklist);
	void add_parts(std::shared_ptr<kick_workspace> ptr, part_int n);
	void to_gpu();
};

#ifdef HPX_LITE
namespace hpx {
	namespace serialization {
		template<class A>
		void serialize(A&&, std::shared_ptr<kick_workspace>&, unsigned) {
		}
	}
}
#endif

#endif
#endif /* KICK_WORKSPACE_HPP_ */
