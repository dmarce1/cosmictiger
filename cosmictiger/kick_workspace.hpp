/*
CosmicTiger - A cosmological N-Body code
Copyright (C) 2021  Dominic C. Marcello

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/


#ifndef KICK_WORKSPACE_HPP_
#define KICK_WORKSPACE_HPP_

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/kick.hpp>
#include <cosmictiger/tree.hpp>
#include <cosmictiger/unordered_set_ts.hpp>
#include <cosmictiger/semaphore.hpp>

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
	part_int total_parts;
	part_int nparts;
	kick_params params;
	std::unordered_set<tree_id, kick_workspace_tree_id_hash> tree_ids;
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
	void add_work(std::shared_ptr<kick_workspace> ptr, expansion<float> L, array<fixed32, NDIM> pos, tree_id self,
			vector<tree_id> && dchecklist, vector<tree_id> && echecklist);
	void add_parts(std::shared_ptr<kick_workspace> ptr, part_int n);
	void to_gpu();
};

#endif
#endif /* KICK_WORKSPACE_HPP_ */
