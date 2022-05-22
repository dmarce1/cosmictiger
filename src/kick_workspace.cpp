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

static void add_tree_node(std::vector<tree_id>& trees, tree_id id) {
	const tree_node* node = tree_get_node(id);
	trees.push_back(id);
	if (node->children[LEFT].index != -1) {
		add_tree_node(trees, node->children[LEFT]);
		add_tree_node(trees, node->children[RIGHT]);
	}
}

bool morton_compare(array<fixed32, NDIM> a, array<fixed32, NDIM> b) {
	array<unsigned, NDIM> abits;
	array<unsigned, NDIM> bbits;
	for (int dim = 0; dim < NDIM; dim++) {
		abits[dim] = a[dim].raw();
		bbits[dim] = b[dim].raw();
	}
	for (int i = 0; i < 64; i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			const int abit = abits[dim] & (unsigned) 0x80000000;
			const int bbit = bbits[dim] & (unsigned) 0x80000000;
			if (bbit && !abit) {
				return true;
			} else if (!bbit && abit) {
				return false;
			}
			abits[dim] <<= 1;
			bbits[dim] <<= 1;
		}
	}
	return false;
}

void kick_workspace::to_gpu() {
	cuda_set_device();
	PRINT("Preparing gpu send on %i\n", hpx_rank());

	timer tm;
	tm.start();
	const auto tree_nodes_remote_begin = tree_nodes_next_index();
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

	mutex_type mutex;
	vector<hpx::future<void>> futs1;
	vector<tree_id> tree_ids_vector(tree_ids.begin(), tree_ids.end());
	vector<vector<tree_id>> ids_by_depth(MAX_DEPTH);
	int nthreads = hpx_size() == 1 ? 1 : SORT_OVERSUBSCRIPTION * hpx::thread::hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs1.push_back(hpx::async([proc,nthreads,&tree_ids_vector,&ids_by_depth,&mutex]() {
			std::unique_lock<mutex_type> lock(mutex);
			const int b = (size_t) proc * tree_ids_vector.size() / nthreads;
			const int e = (size_t) (proc+1) * tree_ids_vector.size() / nthreads;
			for (int i = b; i < e; i++) {
				lock.unlock();
				const tree_node* ptr = tree_get_node(tree_ids_vector[i]);
				lock.lock();
				ids_by_depth[ptr->depth].push_back(tree_ids_vector[i]);
			}
		}));
	}
	hpx::wait_all(futs1.begin(), futs1.end());
	struct part_request {
		vector<vector<pair<part_int>>> data;
		size_t count;
		part_request() {
			count = 0;
			data.resize(1);
		}
	};
	std::unordered_map<tree_id, int, kick_workspace_tree_id_hash> tree_map;
	std::unordered_map<int, part_request> part_requests;
	std::unordered_map<global_part_range, pair<part_int>, global_part_range_hash> part_map;
	std::set<tree_id> remote_roots;
	part_int part_index = particles_size();
	vector<hpx::future<void>> futs2;
	vector<hpx::future<void>> futs3;
	const size_t opartsize = particles_size();
	const size_t max_parts = 64 * 1024 * 1024;
	timer tm2;
	tm2.start();
	vector<hpx::future<vector<tree_id>> > futs0;
	vector<tree_id> trees;
	for (int depth = 0; depth < MAX_DEPTH; depth++) {
		const auto& ids = ids_by_depth[depth];
		for (int proc = 0; proc < nthreads; proc++) {
			futs0.push_back(hpx::async([proc,nthreads,&ids,&tree_map, &mutex,&part_requests,&part_map,&part_index,&remote_roots]() {
				vector<tree_id> trees;
				const int b = (size_t) proc * ids.size() / nthreads;
				const int e = (size_t)(proc + 1) * ids.size() / nthreads;
				for (int i = b; i < e; i++) {
					if (tree_map.find(ids[i]) == tree_map.end()) {
						const tree_node* node = tree_get_node(ids[i]);
						add_tree_node(trees, ids[i]);
						const int rank = node->proc_range.first;
						if (rank != hpx_rank()) {
							const auto range = node->part_range;
							const auto this_count = range.second - range.first;
							global_part_range gpr;
							gpr.rank = rank;
							gpr.range = range;
							const size_t nparts = node->nparts();
							std::unique_lock<shared_mutex_type> lock(particles_shared_mutex());
							auto& entry = part_requests[rank];
							entry.data.back().push_back(range);
							entry.count += this_count;
							part_map[gpr].first = part_index;
							part_map[gpr].second = part_index + nparts;
							part_index += nparts;
							particles_resize(part_index,false);
							remote_roots.insert(ids[i]);
							if (entry.count > max_parts) {
								size_t count = entry.count;
								part_int index = 0;
								entry.count = 0;
								entry.data.resize(entry.data.size() + 1);
							}
						}
					}
				}
				return trees;
			}));
		}
		hpx::wait_all(futs0.begin(), futs0.end());
		const auto start = trees.size();
		for (auto& f : futs0) {
			auto tmp = f.get();
			trees.reserve(trees.size() + tmp.size());
			for (const auto & entry : tmp) {
				trees.push_back(entry);
			}
		}
		/*	auto sfut = hpx::sort(PAR_EXECUTION_POLICY, trees.begin() + start,trees.end(), [](const tree_id& a,const tree_id& b) {
		 return morton_compare(tree_get_node(a)->pos, tree_get_node(b)->pos);
		 });
		 sfut.get();*/
		timer tm3;
		tm3.start();
		for (int i = start; i < trees.size(); i++) {
			if (trees[i].proc == hpx_rank()) {
				tree_map[trees[i]] = trees[i].index;
			} else {
				tree_map[trees[i]] = tree_add_remote(*tree_get_node(trees[i]));

			}
		}
		tm3.stop();
		PRINT( "!!!!!!!!!!! %e\n", tm3.read());
		futs0.resize(0);
	}

	tm2.stop();
	PRINT("%e to load tree nodes\n", tm2.read());
	tm2.reset();
	futs1.resize(0);
	nthreads = hpx::thread::hardware_concurrency();
	for (auto i = part_requests.begin(); i != part_requests.end(); i++) {
		const int rank = i->first;
		hpx::future<void> fut = hpx::make_ready_future();
		for (int j = 0; j < i->second.data.size(); j++) {
			const size_t count = i->second.data[j].size();
			if (count) {
				fut = fut.then([rank,count,&part_map,i,j](hpx::future<void> fut) {
					fut.get();
					const auto& ranges = i->second.data[j];
					const auto data = particles_get(rank,ranges).get();
					part_int index = 0;
					for( int k = 0; k < ranges.size(); k++) {
						index += ranges[k].second - ranges[k].first;
					}
					size_t sz = index;
					index = 0;
					for( int k = 0; k < ranges.size(); k++) {
						global_part_range gpr;
						gpr.rank = rank;
						gpr.range = ranges[k];
						auto local_range = part_map[gpr];
						const auto nparts = local_range.second - local_range.first;
						for( int dim = 0; dim < NDIM; dim++) {
							for( int l = 0; l < nparts; l++) {
								const int m = local_range.first + l;
								const int n = sz * dim + index + l;
								ALWAYS_ASSERT(m>=0);
								ALWAYS_ASSERT(n>=0);
								ALWAYS_ASSERT(m<particles_size());
								ALWAYS_ASSERT(n<data.size());
								particles_pos(dim,m).set_integer(data[n]);
							}
						}
						index += nparts;
					}
				});
			}
		}
		futs1.push_back(std::move(fut));
	}
	auto* tree_nodes = tree_data();
	for (int proc = 0; proc < nthreads; proc++) {
		futs2.push_back(hpx::async(HPX_PRIORITY_HI, [proc,nthreads,&tree_map,&tree_nodes, tree_nodes_remote_begin]() {
			for (int i = tree_nodes_remote_begin + proc; i < tree_nodes_next_index(); i+=nthreads) {
				if (tree_nodes[i].children[LEFT].index != -1) {
					ALWAYS_ASSERT(tree_map.find(tree_nodes[i].children[LEFT]) != tree_map.end());
					ALWAYS_ASSERT(tree_map.find(tree_nodes[i].children[RIGHT]) != tree_map.end());
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
		futs3.push_back(hpx::async(HPX_PRIORITY_HI,[i,&tree_map,&part_map,adjust_part_refs,&tree_nodes,tree_nodes_remote_begin]() {
			global_part_range gpr;
			ALWAYS_ASSERT(tree_map.find(*i) != tree_map.end());
			int index = tree_map[*i];
			ALWAYS_ASSERT( index >= tree_nodes_remote_begin);
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
					if( workitems[i].dchecklist[j].proc != hpx_rank()) {
						auto iter = tree_map.find(workitems[i].dchecklist[j]);
						if( iter == tree_map.end()) {
							THROW_ERROR( "Tree map error %i\n", tree_map.size());
						}
						workitems[i].dchecklist[j].index = iter->second;
					}
				}
				for (int j = 0; j < workitems[i].echecklist.size(); j++) {
					if( workitems[i].echecklist[j].proc != hpx_rank()) {
						auto iter = tree_map.find(workitems[i].echecklist[j]);
						if( iter == tree_map.end()) {
							THROW_ERROR( "Tree map error\n");
						}
						workitems[i].echecklist[j].index = iter->second;
					}
				}
			}
		}));
	}
	hpx::wait_all(futs2.begin(), futs2.end());
	futs2.resize(0);
	auto sfut = hpx::sort(PAR_EXECUTION_POLICY, workitems.begin(), workitems.end(), [](kick_workitem a, kick_workitem b) {
		return morton_compare(a.pos, b.pos);
	});
	hpx::wait_all(futs3.begin(), futs3.end());
	hpx::wait_all(futs1.begin(), futs1.end());
	sfut.get();
	auto stream = cuda_get_stream();
	tm.stop();
	PRINT("Took %e seconds to prepare gpu send\n", tm.read());
	tm.reset();
	tm.start();
	const auto kick_returns = cuda_execute_kicks(params, &particles_pos(XDIM, 0), &particles_pos(YDIM, 0), &particles_pos(ZDIM, 0), tree_nodes,
			std::move(workitems), stream);
	cuda_end_stream(stream);
	tm.stop();
	PRINT("GPU took %e seconds\n", tm.read());

	for (int i = 0; i < kick_returns.size(); i++) {
		promises[i].set_value(std::move(kick_returns[i]));
	}
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
