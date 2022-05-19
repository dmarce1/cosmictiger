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

vector<fixed32, pinned_allocator<fixed32>> kick_workspace::host_x;
vector<fixed32, pinned_allocator<fixed32>> kick_workspace::host_y;
vector<fixed32, pinned_allocator<fixed32>> kick_workspace::host_z;
#ifndef DM_CON_H_ONLY
vector<float, pinned_allocator<float>> kick_workspace::host_h;
vector<float, pinned_allocator<float>> kick_workspace::host_zeta;
vector<char> kick_workspace::host_type;
#endif
semaphore kick_workspace::lock1(1);
semaphore kick_workspace::lock2(1);
vector<tree_node, pinned_allocator<tree_node>> kick_workspace::tree_nodes;

HPX_PLAIN_ACTION(kick_workspace::clear_buffers, clear_buffers_action);

kick_workspace::kick_workspace(kick_params p, part_int total_parts_) {
	total_parts = total_parts_;
	params = p;
	nparts = 0;
}

kick_workspace::~kick_workspace() {
}

static void add_tree_node(std::unordered_map<tree_id, int, kick_workspace_tree_id_hash>& tree_map, tree_id id, int& index, int rank) {
	tree_map.insert(std::make_pair(id, index));
	const tree_node* node = tree_get_node(id);
	ASSERT(id.proc == rank);
	index++;
	if (node->children[LEFT].index != -1) {
		add_tree_node(tree_map, node->children[LEFT], index, rank);
		add_tree_node(tree_map, node->children[RIGHT], index, rank);
	}
}

static void adjust_part_references(vector<tree_node, pinned_allocator<tree_node>>& tree_nodes, int index, part_int offset) {
	tree_nodes[index].part_range.first += offset;
	ASSERT(tree_nodes[index].part_range.first >= 0);
	tree_nodes[index].part_range.second += offset;
	if (tree_nodes[index].children[LEFT].index != -1) {
		adjust_part_references(tree_nodes, tree_nodes[index].children[RIGHT].index, offset);
		adjust_part_references(tree_nodes, tree_nodes[index].children[LEFT].index, offset);
	}
}

void kick_workspace::to_gpu() {
	timer tm;
	if( workitems.size() == 0 ) {
		return;
	}

	lock1.wait();
	cuda_set_device();
//	PRINT("Preparing %i items on %i\n", workitems.size(), hpx_rank());
//	PRINT("To vector\n");
//	PRINT("%i tree ids\n", tree_ids.size());
	vector<tree_id> tree_ids_vector(tree_ids.begin(), tree_ids.end());
	vector<vector<tree_id>> ids_by_depth(MAX_DEPTH);
	part_int part_count = 0;
	for (int i = 0; i < tree_ids_vector.size(); i++) {
		const tree_node* ptr = tree_get_node(tree_ids_vector[i]);
		ids_by_depth[ptr->depth].push_back(tree_ids_vector[i]);
	}
	fixed32* dev_x;
	fixed32* dev_y;
	fixed32* dev_z;
	std::unordered_map<tree_id, int, kick_workspace_tree_id_hash> tree_map;
	std::atomic<part_int> next_index(0);
	std::unordered_set<tree_id, kick_workspace_tree_id_hash> tree_bases;
	for (int depth = 0; depth < MAX_DEPTH; depth++) {
		const auto& ids = ids_by_depth[depth];
		if (ids.size()) {
			for (int i = 0; i < ids.size(); i++) {
				if (tree_map.find(ids[i]) == tree_map.end()) {
					const tree_node* node = tree_get_node(ids[i]);
					int index = (next_index += node->node_count);
					index -= node->node_count;
					part_count += node->nparts();
					add_tree_node(tree_map, ids[i], index, ids[i].proc);
					tree_bases.insert(ids[i]);
				}
			}
		}
	}
	cuda_set_device();
	tree_node* dev_trees;
	tree_nodes.resize(next_index);
	for (auto i = tree_map.begin(); i != tree_map.end(); i++) {
		tree_nodes[i->second] = *tree_get_node(i->first);
	}
	vector<hpx::future<void>> futs;
	const int nthreads = hpx::thread::hardware_concurrency();
	futs.reserve(nthreads);
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async(HPX_PRIORITY_HI, [proc,nthreads,&tree_map]() {
							for (int i = proc; i < tree_nodes.size(); i+=nthreads) {
								if (tree_nodes[i].children[LEFT].index != -1) {
									tree_nodes[i].children[LEFT].index = tree_map[tree_nodes[i].children[LEFT]];
									tree_nodes[i].children[RIGHT].index = tree_map[tree_nodes[i].children[RIGHT]];
								}
							}
							return 'a';
						}));
	}

	hpx::wait_all(futs.begin(), futs.end());
	futs.resize(0);

	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async(HPX_PRIORITY_HI, [this,proc,nthreads,&tree_map]() {
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
							return 'a';
						}));
	}
	hpx::wait_all(futs.begin(), futs.end());
	tm.start();

	next_index = 0;
	host_x.resize(part_count);
	host_y.resize(part_count);
	host_z.resize(part_count);
	futs.resize(0);

	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async(HPX_PRIORITY_HI, [&next_index,&tree_ids_vector,&tree_map,proc,nthreads,&tree_bases]() {
							for (int i = proc; i < tree_ids_vector.size(); i+=nthreads) {
								if( tree_bases.find(tree_ids_vector[i]) != tree_bases.end()) {
									const tree_node* ptr = tree_get_node(tree_ids_vector[i]);
									const int local_index = tree_map[tree_ids_vector[i]];
									part_int part_index = (next_index += ptr->nparts()) - ptr->nparts();
									particles_global_read_pos(ptr->global_part_range(), host_x.data(), host_y.data(), host_z.data(),  part_index);
									adjust_part_references(tree_nodes, local_index, part_index - ptr->part_range.first);
								}
							}
							return 'a';
						}
				)
		);
	}
	hpx::wait_all(futs.begin(), futs.end());
	tm.stop();
	auto stream = cuda_get_stream();
	CUDA_CHECK(cudaMalloc(&dev_x, sizeof(fixed32) * part_count));
	CUDA_CHECK(cudaMalloc(&dev_y, sizeof(fixed32) * part_count));
	CUDA_CHECK(cudaMalloc(&dev_z, sizeof(fixed32) * part_count));
	CUDA_CHECK(cudaMalloc(&dev_trees, tree_nodes.size() * sizeof(tree_node)));
	CUDA_CHECK(cudaMemcpyAsync(dev_trees, tree_nodes.data(), tree_nodes.size() * sizeof(tree_node), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(dev_x, host_x.data(), sizeof(fixed32) * part_count, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(dev_y, host_y.data(), sizeof(fixed32) * part_count, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(dev_z, host_z.data(), sizeof(fixed32) * part_count, cudaMemcpyHostToDevice, stream));
	hpx::wait_all(futs.begin(), futs.end());
	tm.reset();
	tm.start();
	int nitems = workitems.size();
//	PRINT("Sending %i items on %i\n", workitems.size(), hpx_rank());
	const auto kick_returns = cuda_execute_kicks(params, dev_x, dev_y, dev_z, nullptr, nullptr, nullptr, dev_trees, std::move(workitems), stream, part_count, tree_nodes.size(), [&]() {lock2.wait();}, [&]() {lock1.signal();});
	tm.stop();
	cuda_end_stream(stream);
//PRINT("Done %i items on %i\n",nitems, hpx_rank());

	CUDA_CHECK(cudaFree(dev_x));
	CUDA_CHECK(cudaFree(dev_y));
	CUDA_CHECK(cudaFree(dev_z));
	CUDA_CHECK(cudaFree(dev_trees));
	for (int i = 0; i < kick_returns.size(); i++) {
		promises[i].set_value(std::move(kick_returns[i]));
	}
	lock2.signal();
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
		futs.push_back(hpx::async<clear_buffers_action>( c));
	}
	host_x = decltype(host_x)();
	host_y = decltype(host_y)();
	host_z = decltype(host_z)();
	tree_nodes = decltype(tree_nodes)();
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
