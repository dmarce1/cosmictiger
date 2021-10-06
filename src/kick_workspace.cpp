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

vector<vector<fixed32, pinned_allocator<fixed32>>> kick_workspace::host_xs;
vector<vector<fixed32, pinned_allocator<fixed32>>> kick_workspace::host_ys;
vector<vector<fixed32, pinned_allocator<fixed32>>> kick_workspace::host_zs;
vector<std::shared_ptr<hpx::lcos::local::counting_semaphore>> kick_workspace::lock1s;
vector<std::shared_ptr<hpx::lcos::local::counting_semaphore>> kick_workspace::lock2s;
vector<vector<tree_node, pinned_allocator<tree_node>>> kick_workspace::tree_nodess;
std::atomic<int> kick_workspace::next_device(0);

HPX_PLAIN_ACTION(kick_workspace::clear_buffers, clear_buffers_action);

void kick_workspace::initialize() {
#ifdef USE_CUDA
	const int dvccnt = cuda_device_count();
	host_xs.resize(dvccnt);
	host_ys.resize(dvccnt);
	host_zs.resize(dvccnt);
	lock1s.resize(dvccnt);
	lock2s.resize(dvccnt);
	tree_nodess.resize(dvccnt);
	for( int i = 0; i < dvccnt; i++) {
		lock1s[i] = std::make_shared<hpx::lcos::local::counting_semaphore>(1);
		lock2s[i] = std::make_shared<hpx::lcos::local::counting_semaphore>(1);
	}
#endif
}

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
#ifdef USE_CUDA
	int device = next_device++ % cuda_device_count();
	while( !lock1s[device]->try_wait()) {
		hpx_yield();
		device = next_device++ % cuda_device_count();
	}
	cuda_set_device(device);
	auto& lock1 = *(lock1s[device]);
	auto& lock2 = *(lock2s[device]);
	auto& tree_nodes = tree_nodess[device];
	auto& host_x = host_xs[device];
	auto& host_y = host_ys[device];
	auto& host_z = host_zs[device];
	timer tm;
	PRINT("Preparing GPU %i on %i\n", device, hpx_rank());

	auto sort_fut = hpx::parallel::sort(PAR_EXECUTION_POLICY, workitems.begin(), workitems.end(), [](const kick_workitem& a, const kick_workitem& b) {
				const auto* aptr = tree_get_node(a.self);
				const auto* bptr = tree_get_node(b.self);
				return 10 * aptr->active_nodes + aptr->nactive > 10 * aptr->active_nodes + bptr->nactive;
			}
	);

//	PRINT("To vector\n");
	vector<tree_id> tree_ids_vector(tree_ids.begin(), tree_ids.end());
//	PRINT("%i tree ids\n", tree_ids_vector.size());
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
	cuda_set_device(device);
	tree_node* dev_trees;
	tree_nodes.resize(next_index);
	CUDA_CHECK(cudaMalloc(&dev_x, sizeof(fixed32) * part_count));
	CUDA_CHECK(cudaMalloc(&dev_y, sizeof(fixed32) * part_count));
	CUDA_CHECK(cudaMalloc(&dev_z, sizeof(fixed32) * part_count));
	CUDA_CHECK(cudaMalloc(&dev_trees, tree_nodes.size() * sizeof(tree_node)));
	for (auto i = tree_map.begin(); i != tree_map.end(); i++) {
		tree_nodes[i->second] = *tree_get_node(i->first);
	}
	vector<hpx::future<void>> futs;
	const int nthreads = hpx::thread::hardware_concurrency();
	futs.reserve(nthreads);
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([proc,nthreads,&tree_map,&tree_nodes]() {
							for (int i = proc; i < tree_nodes.size(); i+=nthreads) {
								if (tree_nodes[i].children[LEFT].index != -1) {
									tree_nodes[i].children[LEFT].index = tree_map[tree_nodes[i].children[LEFT]];
									tree_nodes[i].children[RIGHT].index = tree_map[tree_nodes[i].children[RIGHT]];
								}
							}
						}));
	}

	sort_fut.get();
	hpx::wait_all(futs.begin(), futs.end());
	futs.resize(0);

	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([this,proc,nthreads,&tree_map]() {
							for (int i = proc; i < workitems.size(); i+=nthreads) {
								for (int j = 0; j < workitems[i].dchecklist.size(); j++) {
									workitems[i].dchecklist[j].index = tree_map[workitems[i].dchecklist[j]];
								}
								for (int j = 0; j < workitems[i].echecklist.size(); j++) {
									workitems[i].echecklist[j].index = tree_map[workitems[i].echecklist[j]];
								}
								workitems[i].self.index = tree_map[workitems[i].self];
								const tree_node& node = *tree_get_node(workitems[i].self);
								const part_int begin = node.sink_part_range.first;
								const part_int end = node.sink_part_range.second;
								const size_t size = end - begin;
								ASSERT(workitems[i].self.proc == hpx_rank());
							}
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
		futs.push_back(hpx::async([&next_index,&tree_ids_vector,&tree_map,proc,nthreads,&tree_bases,&host_x,&host_y,&host_z,&tree_nodes]() {
							for (int i = proc; i < tree_ids_vector.size(); i+=nthreads) {
								if( tree_bases.find(tree_ids_vector[i]) != tree_bases.end()) {
									const tree_node* ptr = tree_get_node(tree_ids_vector[i]);
									const int local_index = tree_map[tree_ids_vector[i]];
									part_int part_index = (next_index += ptr->nparts()) - ptr->nparts();
									particles_global_read_pos(ptr->global_part_range(), host_x.data(), host_y.data(), host_z.data(), part_index);
									adjust_part_references(tree_nodes, local_index, part_index - ptr->part_range.first);
								}
							}
						}
				)
		);
	}
	hpx::wait_all(futs.begin(), futs.end());
	tm.stop();
	auto stream = cuda_get_stream(device);
	CUDA_CHECK(cudaMemcpyAsync(dev_trees, tree_nodes.data(), tree_nodes.size() * sizeof(tree_node), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(dev_x, host_x.data(), sizeof(fixed32) * part_count, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(dev_y, host_y.data(), sizeof(fixed32) * part_count, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(dev_z, host_z.data(), sizeof(fixed32) * part_count, cudaMemcpyHostToDevice, stream));
	//cuda_stream_synchronize(stream);
//	PRINT("parts size = %li\n", sizeof(fixed32) * part_count * NDIM);
	const auto kick_returns = cuda_execute_kicks(device, params, dev_x, dev_y, dev_z, dev_trees, std::move(workitems), stream, part_count, tree_nodes.size(), [&]() {lock2.wait();}, [&]() {lock1.signal();});
	cuda_stream_synchronize(stream);
	cuda_end_stream(stream);
	PRINT("GPU %i done on %i\n", device, hpx_rank());
	CUDA_CHECK(cudaFree(dev_x));
	CUDA_CHECK(cudaFree(dev_y));
	CUDA_CHECK(cudaFree(dev_z));
	CUDA_CHECK(cudaFree(dev_trees));
	for (int i = 0; i < kick_returns.size(); i++) {
		promises[i].set_value(std::move(kick_returns[i]));
	}
	lock2.signal();
#endif
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
		hpx::async(HPX_PRIORITY_HI, [ptr]() {
			ptr->to_gpu();
		});
	}
}

void kick_workspace::clear_buffers() {
	vector<hpx::future<void>> futs;
	for (const auto& c : hpx_children()) {
		futs.push_back(hpx::async<clear_buffers_action>(c));
	}
	for (int dvc = 0; dvc < cuda_device_count(); dvc++) {
		host_xs[dvc] = vector<fixed32, pinned_allocator<fixed32>>();
		host_ys[dvc] = vector<fixed32, pinned_allocator<fixed32>>();
		host_zs[dvc] = vector<fixed32, pinned_allocator<fixed32>>();
		tree_nodess[dvc] = vector<tree_node, pinned_allocator<tree_node>>();
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
	}
	item.dchecklist = std::move(dchecks);
	item.echecklist = std::move(echecks);
	std::unique_lock<mutex_type> lock(mutex);
	promises.resize(promises.size() + 1);
	auto fut = promises.back().get_future();
	workitems.push_back(std::move(item));
	lock.unlock();
	if (do_work) {
		hpx::async(HPX_PRIORITY_HI, [ptr]() {
			ptr->to_gpu();
		});
	}
	return fut;
}
