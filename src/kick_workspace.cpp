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

#include <cosmictiger/kick_workspace.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/timer.hpp>
#include <set>

HPX_PLAIN_ACTION(kick_workspace::clear_buffers, clear_buffers_action);

kick_workspace::kick_workspace(kick_params p, part_int total_parts_) {
	total_parts = total_parts_;
	params = p;
	nparts = 0;
}

kick_workspace::~kick_workspace() {
}

static void add_tree_node(std::unordered_map<tree_id, int, kick_workspace_tree_id_hash>& tree_map, tree_id id, int& index) {
	tree_map.insert(std::make_pair(id, index));
	const tree_node* node = tree_get_node(id);
	index++;
	if (node->children[LEFT].index != -1) {
		add_tree_node(tree_map, node->children[LEFT], index);
		add_tree_node(tree_map, node->children[RIGHT], index);
	}
}

void kick_workspace::to_gpu() {

	timer tm;
	tm.start();
	if (workitems.size() == 0) {
		return;
	}

	struct global_part_range {
		int rank;
		pair<part_int> range;
		bool operator==(global_part_range other) const {
			return rank == other.rank && range == other.range;
		}
	};

	struct global_part_range_hash {
		size_t operator()(global_part_range g) const {
			return (size_t) hpx_size() * g.range.first + g.rank;
		}
	};

	vector<tree_id> tree_ids_vector(tree_ids.begin(), tree_ids.end());
	vector<vector<tree_id>> ids_by_depth(MAX_DEPTH);
	for (int i = 0; i < tree_ids_vector.size(); i++) {
		const tree_node* ptr = tree_get_node(tree_ids_vector[i]);
		ids_by_depth[ptr->depth].push_back(tree_ids_vector[i]);
	}
	std::unordered_map<tree_id, int, kick_workspace_tree_id_hash> tree_map;

	std::unordered_map<int, vector<pair<part_int>>> part_requests;
	std::unordered_map<global_part_range, pair<part_int>, global_part_range_hash> part_map;

	std::set<tree_id> remote_roots;
	part_int next_index = 0;
	part_int part_index = particles_size();
	const size_t opartsize = particles_size();
	for (int depth = 0; depth < MAX_DEPTH; depth++) {
		const auto& ids = ids_by_depth[depth];
		for (int i = 0; i < ids.size(); i++) {
			if (tree_map.find(ids[i]) == tree_map.end()) {
				const tree_node* node = tree_get_node(ids[i]);
				add_tree_node(tree_map, ids[i], next_index);
				next_index += node->node_count;
				const int rank = node->proc_range.first;
				if (rank != hpx_rank()) {
					const auto range = node->part_range;
					part_requests[rank].push_back(range);
					global_part_range gpr;
					gpr.rank = rank;
					gpr.range = range;
					const size_t nparts = node->nparts();
					part_map[gpr].first = part_index;
					part_map[gpr].second = part_index + nparts;
					part_index += nparts;
					particles_resize(part_index);
					remote_roots.insert(ids[i]);
				}
			}
		}
	}

	vector<hpx::future<void>> futs1;
	vector<hpx::future<void>> futs2;
	vector<hpx::future<void>> futs3;
	for (auto i = part_requests.begin(); i != part_requests.end(); i++) {
		futs1.push_back(particles_get(i->first, i->second).then([i,&part_map](hpx::future<array<vector<fixed32>, NDIM>> fut) {
			auto data = fut.get();
			part_int index = 0;
			for( int k = 0; k < i->second.size(); k++) {
				global_part_range gpr;
				gpr.rank = i->first;
				gpr.range = i->second[k];
				auto local_range = part_map[gpr];
				const auto nparts = local_range.second - local_range.first;
				for( int dim = 0; dim < NDIM; dim++) {
					std::memcpy(&particles_pos(dim,local_range.first), &data[dim][index], sizeof(fixed32)*nparts);
				}
				index += nparts;
			}
		}));
	}
	tree_node* tree_nodes;
	size_t tree_nodes_size = next_index;
	CUDA_CHECK(cudaMallocManaged(&tree_nodes, sizeof(tree_node) * tree_nodes_size));
	for (auto i = tree_map.begin(); i != tree_map.end(); i++) {
		auto& node = tree_nodes[i->second];
		node = *tree_get_node(i->first);
	}
	const int nthreads = hpx::thread::hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs2.push_back(hpx::async(HPX_PRIORITY_HI, [proc,nthreads,&tree_map,&tree_nodes, tree_nodes_size]() {
			for (int i = proc; i < tree_nodes_size; i+=nthreads) {
				if (tree_nodes[i].children[LEFT].index != -1) {
					tree_nodes[i].children[LEFT].index = tree_map[tree_nodes[i].children[LEFT]];
					tree_nodes[i].children[RIGHT].index = tree_map[tree_nodes[i].children[RIGHT]];
				}
			}
		}));
	}
	hpx::wait_all(futs2.begin(), futs2.end());
	futs2.resize(0);
	const std::function<void(int, part_int)> adjust_part_refs = [&adjust_part_refs,&tree_nodes](int index, part_int offset) -> void {
		tree_nodes[index].part_range.first += offset;
		tree_nodes[index].part_range.second += offset;
		ALWAYS_ASSERT(tree_nodes[index].part_range.first >= 0);
		ALWAYS_ASSERT(tree_nodes[index].part_range.second <= particles_size());
		if( tree_nodes[index].children[LEFT].index != -1) {
			adjust_part_refs(tree_nodes[index].children[LEFT].index, offset);
			adjust_part_refs(tree_nodes[index].children[RIGHT].index, offset);
		}
	};
	for (auto i = remote_roots.begin(); i != remote_roots.end(); i++) {
		futs3.push_back(hpx::async(HPX_PRIORITY_HI,[i,&tree_map,&part_map,adjust_part_refs,&tree_nodes]() {
			global_part_range gpr;
			int index = tree_map[*i];
			auto& node = tree_nodes[index];
			gpr.rank = node.proc_range.first;
			gpr.range = node.part_range;
			ALWAYS_ASSERT(part_map.find(gpr) != part_map.end());
			auto local_range = part_map[gpr];
			part_int offset = local_range.first - gpr.range.first;
			ALWAYS_ASSERT(tree_nodes[index].part_range.first == gpr.range.first);
			adjust_part_refs(index, offset);
		}));
	}
	for (int proc = 0; proc < nthreads; proc++) {
		futs2.push_back(hpx::async(HPX_PRIORITY_HI, [this,proc,nthreads,&tree_map]() {
			for (int i = proc; i < workitems.size(); i+=nthreads) {
				for (int j = 0; j < workitems[i].dchecklist.size(); j++) {
					auto iter = tree_map.find(workitems[i].dchecklist[j]);
					if( iter == tree_map.end()) {
						THROW_ERROR( "Tree map error %i\n", tree_map.size());
					}
					workitems[i].dchecklist[j].index = iter->second;
				}
				for (int j = 0; j < workitems[i].echecklist.size(); j++) {
					auto iter = tree_map.find(workitems[i].echecklist[j]);
					if( iter == tree_map.end()) {
						THROW_ERROR( "Tree map error\n");
					}
					workitems[i].echecklist[j].index = iter->second;
				}
				workitems[i].self.index = tree_map[workitems[i].self];
				ASSERT(workitems[i].self.proc == hpx_rank());
			}
		}));
	}
	hpx::wait_all(futs2.begin(), futs2.end());
	futs2.resize(0);
	auto sfut = hpx::sort(PAR_EXECUTION_POLICY, workitems.begin(), workitems.end(), [](kick_workitem a, kick_workitem b) {
		array<unsigned, NDIM> abits;
		array<unsigned, NDIM> bbits;
		for(int dim = 0; dim < NDIM; dim++) {
			abits[dim] = a.pos[dim].raw();
			bbits[dim] = b.pos[dim].raw();
		}
		for( int i = 0; i < 64; i++) {
			for( int dim = 0; dim < NDIM; dim++) {
				const int abit = abits[dim] & (unsigned) 0x80000000;
				const int bbit = bbits[dim] & (unsigned) 0x80000000;
				if( bbit && !abit) {
					return true;
				} else if( !bbit && abit ) {
					return false;
				}
				abits[dim] <<=1;
				bbits[dim] <<=1;
			}
		}
		return false;
	});

	next_index = 0;
	hpx::wait_all(futs3.begin(), futs3.end());
	hpx::wait_all(futs1.begin(), futs1.end());
	sfut.get();
#ifdef MULTI_GPU
	const int device_count = cuda_device_count();
	vector<hpx::future<vector<kick_return>>>futs;
	for (int gpu = 0; gpu < device_count; gpu++) {
		futs.push_back(
				hpx::async(
						[tree_nodes, gpu, device_count,this]() {
							cuda_set_device(gpu);
							const int b = gpu * workitems.size() / device_count;
							const int e = (gpu + 1) * workitems.size() / device_count;
							vector<kick_workitem> myworkitems(workitems.begin() + b, workitems.begin() + e);
							auto stream = cuda_get_stream();
							const auto rc = cuda_execute_kicks(params, &particles_pos(XDIM, 0),&particles_pos(YDIM, 0), &particles_pos(ZDIM, 0), tree_nodes, std::move(myworkitems), stream);
							cuda_end_stream(stream);
							return rc;
						}));
	}
	for (auto& f : futs) {
		auto kick_returns = f.get();
		for (int i = 0; i < kick_returns.size(); i++) {
			promises[i].set_value(std::move(kick_returns[i]));
		}
	}
#else
	auto stream = cuda_get_stream();
	const auto kick_returns = cuda_execute_kicks(params, &particles_pos(XDIM, 0),&particles_pos(YDIM, 0), &particles_pos(ZDIM, 0), tree_nodes, std::move(workitems), stream);
	cuda_end_stream(stream);
	for (int i = 0; i < kick_returns.size(); i++) {
		promises[i].set_value(std::move(kick_returns[i]));
	}
#endif
	CUDA_CHECK(cudaFree(tree_nodes));
	particles_resize(opartsize);
}

void kick_workspace::add_parts(std::shared_ptr<kick_workspace> ptr, part_int n) {
	bool do_work = false;
	std::unique_lock<mutex_type> lock(mutex);
	nparts += n;
	if (nparts == total_parts) {
		do_work = true;
	}
	lock.unlock();
	if (do_work) {
		hpx::apply([ptr]() {
			ptr->to_gpu();
		});
	}
}

void kick_workspace::clear_buffers() {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<clear_buffers_action>(c));
	}
	hpx::wait_all(futs.begin(), futs.end());
}

hpx::future<kick_return> kick_workspace::add_work(std::shared_ptr<kick_workspace> ptr, expansion<float> L, array<fixed32, NDIM> pos, tree_id self,
		vector<tree_id> && dchecks, vector<tree_id> && echecks) {
	kick_workitem item;
	item.L = L;
	item.pos = pos;
	item.self = self;
	bool do_work = false;
	{
		const part_int these_nparts = tree_get_node(self)->nparts();
		std::lock_guard<mutex_type> lock(mutex);
		nparts += these_nparts;
		if (nparts == total_parts) {
			do_work = true;
		}
		for (int i = 0; i < dchecks.size(); i++) {
			tree_ids.insert(dchecks[i]);
		}
		for (int i = 0; i < echecks.size(); i++) {
			tree_ids.insert(echecks[i]);
		}
	}
	item.dchecklist = std::move(dchecks);
	item.echecklist = std::move(echecks);
	std::unique_lock<mutex_type> lock(mutex);
	promises.resize(promises.size() + 1);
	auto fut = promises.back().get_future();
	workitems.push_back(std::move(item));
	lock.unlock();
	if (do_work) {
		hpx::apply([ptr]() {
			ptr->to_gpu();
		});
	}
	return fut;
}
