#include <tigerfmm/kick_workspace.hpp>
#include <tigerfmm/particles.hpp>
#include <tigerfmm/timer.hpp>

kick_workspace::kick_workspace(kick_params p, int total_parts_) {
	total_parts = total_parts_;
	params = p;
	nparts = 0;
}

kick_workspace::~kick_workspace() {
}

static void add_tree_node(std::unordered_map<tree_id, int, kick_workspace_tree_id_hash>& tree_map, tree_id id, int& index, int rank) {
	tree_map.insert(std::make_pair(id, index));
	const tree_node* node = tree_get_node(id);
	assert(id.proc == rank);
	index++;
	if (node->children[LEFT].index != -1) {
		add_tree_node(tree_map, node->children[LEFT], index, rank);
		add_tree_node(tree_map, node->children[RIGHT], index, rank);
	}
}

static void create_sink_map(vector<pair<pair<int>>>& map, vector<tree_node>& tree_nodes, int self, int& index) {
	tree_node& node = tree_nodes[self];
	if (!node.leaf) {
		create_sink_map(map, tree_nodes, node.children[LEFT].index, index);
		create_sink_map(map, tree_nodes, node.children[RIGHT].index, index);
	} else {
		const pair<int> host = node.sink_part_range;
		pair<int> dev;
		dev.first = index;
		index += node.sink_part_range.second - node.sink_part_range.first;
		dev.second = index;
		node.sink_part_range = dev;
		pair<pair<int>> item;
		item.first = dev;
		item.second = host;
		map.push_back(item);
	}
}

static void adjust_part_references(vector<tree_node>& tree_nodes, int index, int offset) {
	tree_nodes[index].part_range.first += offset;
	assert(tree_nodes[index].part_range.first >= 0);
	tree_nodes[index].part_range.second += offset;
	if (tree_nodes[index].children[LEFT].index != -1) {
		adjust_part_references(tree_nodes, tree_nodes[index].children[RIGHT].index, offset);
		adjust_part_references(tree_nodes, tree_nodes[index].children[LEFT].index, offset);
	}
}

void kick_workspace::to_gpu(std::atomic<int>& outer_lock) {
	timer tm;
	tm.start();
	cuda_set_device();
	//PRINT("To GPU %i items\n", workitems.size());
	auto sort_fut = hpx::async([this]() {
		std::sort(workitems.begin(), workitems.end(), [](const kick_workitem& a, const kick_workitem& b) {
					const auto* aptr = tree_get_node(a.self);
					const auto* bptr = tree_get_node(b.self);
					return aptr->nactive > bptr->nactive;
				}
		);
	});
//	PRINT("To vector\n");
	vector<tree_id> tree_ids_vector(tree_ids.begin(), tree_ids.end());
//	PRINT("%i tree ids\n", tree_ids_vector.size());
	vector<vector<tree_id>> ids_by_depth(MAX_DEPTH);
	int node_count = 0;
	int part_count = 0;
	for (int i = 0; i < tree_ids_vector.size(); i++) {
		const tree_node* ptr = tree_get_node(tree_ids_vector[i]);
		node_count += ptr->node_count;
		part_count += ptr->nparts();
		ids_by_depth[ptr->depth].push_back(tree_ids_vector[i]);
	}
	fixed32* dev_x;
	fixed32* dev_y;
	fixed32* dev_z;
	CUDA_CHECK(cudaMalloc(&dev_x, sizeof(fixed32) * part_count));
	CUDA_CHECK(cudaMalloc(&dev_y, sizeof(fixed32) * part_count));
	CUDA_CHECK(cudaMalloc(&dev_z, sizeof(fixed32) * part_count));
	std::unordered_map<tree_id, int, kick_workspace_tree_id_hash> tree_map;
	std::atomic<int> next_index(0);
	std::unordered_set<tree_id, kick_workspace_tree_id_hash> tree_bases;
	for (int depth = 0; depth < MAX_DEPTH; depth++) {
		const auto& ids = ids_by_depth[depth];
		if (ids.size()) {
			for (int i = 0; i < ids.size(); i++) {
				if (tree_map.find(ids[i]) == tree_map.end()) {
					const tree_node* node = tree_get_node(ids[i]);
					int index = next_index;
					next_index += node->node_count;
					add_tree_node(tree_map, ids[i], index, ids[i].proc);
					tree_bases.insert(ids[i]);
				}
			}
		}
	}
	vector<tree_node> tree_nodes(next_index);
	tree_node* dev_trees;
	CUDA_CHECK(cudaMalloc(&dev_trees, tree_nodes.size() * sizeof(tree_node)));
	for (auto i = tree_map.begin(); i != tree_map.end(); i++) {
		tree_nodes[i->second] = *tree_get_node(i->first);
	}
	vector<hpx::future<void>> futs;
	const int nthreads = hpx::thread::hardware_concurrency();
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([proc,nthreads,&tree_nodes,&tree_map]() {
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
		futs.push_back(hpx::async([this,proc,nthreads,&tree_nodes,&tree_map]() {
			for (int i = proc; i < workitems.size(); i+=nthreads) {
				for (int j = 0; j < workitems[i].dchecklist.size(); j++) {
					workitems[i].dchecklist[j].index = tree_map[workitems[i].dchecklist[j]];
				}
				for (int j = 0; j < workitems[i].echecklist.size(); j++) {
					workitems[i].echecklist[j].index = tree_map[workitems[i].echecklist[j]];
				}
				workitems[i].self.index = tree_map[workitems[i].self];
				assert(workitems[i].self.proc == hpx_rank());
			}
		}));
	}
	hpx::wait_all(futs.begin(), futs.end());
	next_index = 0;
	vector<fixed32> host_x(part_count);
	vector<fixed32> host_y(part_count);
	vector<fixed32> host_z(part_count);
	futs.resize(0);
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([&next_index,&tree_ids_vector,&tree_map,proc,nthreads,&host_x,&host_y,&host_z,&tree_nodes,&tree_bases]() {
			for (int i = proc; i < tree_ids_vector.size(); i+=nthreads) {
				if( tree_bases.find(tree_ids_vector[i]) != tree_bases.end()) {
					const tree_node* ptr = tree_get_node(tree_ids_vector[i]);
					const int local_index = tree_map[tree_ids_vector[i]];
					int part_index = (next_index += ptr->nparts()) - ptr->nparts();
					particles_global_read_pos(ptr->global_part_range(), host_x.data(), host_y.data(), host_z.data(), part_index);
					adjust_part_references(tree_nodes, local_index, part_index - ptr->part_range.first);
				}
			}
		}));
	}
	vector<pair<pair<int>>> sink_map;
	int nsinks = 0;
	for (int i = 0; i < tree_ids_vector.size(); i++) {
		if (tree_bases.find(tree_ids_vector[i]) != tree_bases.end()) {
			const int local_index = tree_map[tree_ids_vector[i]];
			create_sink_map(sink_map, tree_nodes, local_index, nsinks);
		}
	}

	hpx::wait_all(futs.begin(), futs.end());
	auto stream = cuda_get_stream();
	CUDA_CHECK(cudaMemcpyAsync(dev_trees, tree_nodes.data(), tree_nodes.size() * sizeof(tree_node), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(dev_x, host_x.data(), sizeof(fixed32) * part_count, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(dev_y, host_y.data(), sizeof(fixed32) * part_count, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(dev_z, host_z.data(), sizeof(fixed32) * part_count, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));
	host_x = decltype(host_x)();
	host_y = decltype(host_y)();
	host_z = decltype(host_z)();
	vector<float> phi(nsinks);
	vector<float> gx(nsinks);
	vector<float> gy(nsinks);
	vector<float> gz(nsinks);
//	PRINT("parts size = %li\n", sizeof(fixed32) * part_count * NDIM);
	cuda_execute_kicks(params, dev_x, dev_y, dev_z, dev_trees, std::move(workitems), stream, part_count, tree_nodes.size(), outer_lock, phi, gx, gy, gz);
	cuda_end_stream(stream);
	CUDA_CHECK(cudaFree(dev_x));
	CUDA_CHECK(cudaFree(dev_y));
	CUDA_CHECK(cudaFree(dev_z));
	CUDA_CHECK(cudaFree(dev_trees));

	static float rung_dt[MAX_RUNG] = { 1.0 / (1 << 0), 1.0 / (1 << 1), 1.0 / (1 << 2), 1.0 / (1 << 3), 1.0 / (1 << 4), 1.0 / (1 << 5), 1.0 / (1 << 6), 1.0
			/ (1 << 7), 1.0 / (1 << 8), 1.0 / (1 << 9), 1.0 / (1 << 10), 1.0 / (1 << 11), 1.0 / (1 << 12), 1.0 / (1 << 13), 1.0 / (1 << 14), 1.0 / (1 << 15), 1.0
			/ (1 << 16), 1.0 / (1 << 17), 1.0 / (1 << 18), 1.0 / (1 << 19), 1.0 / (1 << 20), 1.0 / (1 << 21), 1.0 / (1 << 22), 1.0 / (1 << 23), 1.0 / (1 << 24),
			1.0 / (1 << 25), 1.0 / (1 << 26), 1.0 / (1 << 27), 1.0 / (1 << 28), 1.0 / (1 << 29), 1.0 / (1 << 30), 1.0 / (1 << 31) };

	timer tm1;
	tm1.start();
	vector<hpx::future<kick_return>> return_futs;
	for (int proc = 0; proc < nthreads; proc++) {
		return_futs.push_back(hpx::async([nthreads,proc,&sink_map,&gx,&gy,&gz,&phi, this] {
			kick_return kr;
			const float log2ft0 = log2f(params.t0);
			const float tfactor = params.eta * sqrtf(params.a * params.h);
			for (int i = proc; i < sink_map.size(); i+=nthreads) {
				const auto dev_range = sink_map[i].first;
				const auto host_range = sink_map[i].second;
				for (int j = dev_range.first; j < dev_range.second; j++) {
					const int k = host_range.first - dev_range.first + j;
					float& vx = particles_vel(XDIM, k);
					float& vy = particles_vel(YDIM, k);
					float& vz = particles_vel(ZDIM, k);
					char& rung = particles_rung(k);
					float dt = 0.5f * rung_dt[rung] * params.t0;
					if (!params.first_call) {
						vx = fmaf(gx[j], dt, vx);
						vy = fmaf(gy[j], dt, vy);
						vz = fmaf(gz[j], dt, vz);
					}
					float g2 = sqr(gx[j], gy[j], gz[j]);
					dt = std::min(tfactor * sqrt(1.0f / sqrtf(g2)), params.t0);
					rung = std::max((int) ceil(log2ft0 - log2f(dt)), std::max(rung - 1, params.min_rung));
					if (rung < 0 || rung >= MAX_RUNG) {
						PRINT("Rung out of range %i\n", rung);
					}
					assert(rung >= 0);
					assert(rung < MAX_RUNG);
					dt = 0.5f * rung_dt[rung] * params.t0;
					vx = fmaf(gx[j], dt, vx);
					vy = fmaf(gy[j], dt, vy);
					vz = fmaf(gz[j], dt, vz);
					kr.max_rung = std::max((int) kr.max_rung, (int) rung);
					kr.pot += phi[j];
					kr.fx += gx[j];
					kr.fy += gy[j];
					kr.fz += gz[j];
					kr.fnorm += g2;
					if (params.save_force) {
						particles_gforce(XDIM, k) = gx[j];
						particles_gforce(YDIM, k) = gy[j];
						particles_gforce(ZDIM, k) = gz[j];
						particles_pot(k) = phi[j];
					}
				}

			}
			return kr;
		}));
	}
	kick_return kr;
	for (auto& f : return_futs) {
		kr += f.get();
	}
	tm1.stop();
	//PRINT( "Kick took %e\n", tm1.read());
	promises[0].set_value(kr);

	for (int i = 1; i < promises.size(); i++) {
		promises[i].set_value(kick_return());
	}
	tm.stop();
	//PRINT("To GPU Done %i\n", nsinks);

}

void kick_workspace::add_parts(std::shared_ptr<kick_workspace> ptr, int n) {
	bool do_work = false;
	std::unique_lock<mutex_type> lock(mutex);
	nparts += n;
	if (nparts == total_parts) {
		do_work = true;
	}
	lock.unlock();
	if (do_work) {
		hpx::apply([ptr]() {
			static std::atomic<int> cnt(0);
			while( cnt++ != 0 ) {
				cnt--;
				hpx::this_thread::yield();
			}
//			PRINT( "sending to gpu\n");
				ptr->to_gpu(cnt);
			});
	}
}

hpx::future<kick_return> kick_workspace::add_work(std::shared_ptr<kick_workspace> ptr, expansion<float> L, array<fixed32, NDIM> pos, tree_id self,
		vector<tree_id> && dchecks, vector<tree_id> && echecks) {
	kick_workitem item;
	item.L = L;
	item.pos = pos;
	item.self = self;
	bool do_work = false;
	{
		const int these_nparts = tree_get_node(self)->nparts();
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
		hpx::apply([ptr]() {
			static std::atomic<int> cnt(0);
			while( cnt++ != 0 ) {
				cnt--;
				hpx::this_thread::yield();
			}
//			PRINT( "sending to gpu\n");
				ptr->to_gpu(cnt);
			});
	}
	return fut;
}
