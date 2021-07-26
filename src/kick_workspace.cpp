#include <tigerfmm/kick_workspace.hpp>
#include <tigerfmm/particles.hpp>
#include <tigerfmm/timer.hpp>
#include <tigerfmm/unordered_map_ts.hpp>

kick_workspace::kick_workspace(kick_params p) {
	params = p;
	nparts = size_t(KICK_WORKSPACE_PART_SIZE) * cuda_free_mem() / (NDIM * sizeof(fixed32)) / 100;
	CUDA_CHECK(cudaStreamCreate(&stream));
}

kick_workspace::~kick_workspace() {
	CUDA_CHECK(cudaStreamSynchronize(stream));
	CUDA_CHECK(cudaStreamDestroy(stream));
}

static void add_tree_node(unordered_map_ts<tree_id, int, kick_workspace_tree_id_hash>& tree_map, tree_id id, int& index) {
	tree_map.insert(std::make_pair(id, index));
	const tree_node* node = tree_get_node(id);
	index++;
	if (node->children[LEFT].index != -1) {
		add_tree_node(tree_map, node->children[LEFT], index);
		add_tree_node(tree_map, node->children[RIGHT], index);
	}
}

static void adjust_part_references(vector<tree_node, pinned_allocator<tree_node>>& tree_nodes, int index, int offset) {
	tree_nodes[index].part_range.first += offset;
	tree_nodes[index].part_range.second += offset;
	if (tree_nodes[index].children[LEFT].index != -1) {
		adjust_part_references(tree_nodes, tree_nodes[index].children[RIGHT].index, offset);
		adjust_part_references(tree_nodes, tree_nodes[index].children[LEFT].index, offset);
	}
}

void kick_workspace::to_gpu(std::shared_ptr<kick_workspace> ptr) {
	timer tm;
	tm.start();
	PRINT("To GPU\n");
	auto tree_ids_vector = tree_ids.to_vector();
	PRINT("%i tree ids\n", tree_ids_vector.size());
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
	CUDA_CHECK(cudaMallocAsync(&dev_x, sizeof(fixed32) * part_count, stream));
	CUDA_CHECK(cudaMallocAsync(&dev_y, sizeof(fixed32) * part_count, stream));
	CUDA_CHECK(cudaMallocAsync(&dev_z, sizeof(fixed32) * part_count, stream));
	unordered_map_ts<tree_id, int, kick_workspace_tree_id_hash> tree_map;
	int next_index = 0;
	for (int depth = 0; depth < MAX_DEPTH; depth++) {
		const auto& ids = ids_by_depth[depth];
		if (ids.size()) {
			for (int i = 0; i < ids.size(); i++) {
				if (!tree_map.exists(ids[i])) {
					const tree_node* node = tree_get_node(ids[i]);
					int index = next_index;
					next_index += node->node_count;
					add_tree_node(tree_map, ids[i], index);
				}
			}
		}
	}
	vector<tree_node, pinned_allocator<tree_node>> tree_nodes(next_index);
	auto tree_map_vector = tree_map.to_vector();
	tree_node* dev_trees;
	CUDA_CHECK(cudaMallocAsync(&dev_trees, tree_nodes.size() * sizeof(tree_node), stream));
	PRINT("%i %i\n", next_index, tree_map_vector.size());
	for (int i = 0; i < tree_map_vector.size(); i++) {
		tree_nodes[tree_map_vector[i].second] = *tree_get_node(tree_map_vector[i].first);
	}
	for (int i = 0; i < tree_nodes.size(); i++) {
		if (tree_nodes[i].children[LEFT].index != -1) {
			tree_nodes[i].children[LEFT].index = tree_map[tree_nodes[i].children[LEFT]];
			tree_nodes[i].children[RIGHT].index = tree_map[tree_nodes[i].children[RIGHT]];
		}
	}
	for (int i = 0; i < workitems.size(); i++) {
		for (int j = 0; j < workitems[i].dchecklist.size(); j++) {
			workitems[i].dchecklist[j].index = tree_map[workitems[i].dchecklist[j]];
		}
		for (int j = 0; j < workitems[i].echecklist.size(); j++) {
			workitems[i].echecklist[j].index = tree_map[workitems[i].echecklist[j]];
		}
		workitems[i].self.index = tree_map[workitems[i].self];
	}
	next_index = 0;
	vector<fixed32, pinned_allocator<fixed32>> host_x(part_count);
	vector<fixed32, pinned_allocator<fixed32>> host_y(part_count);
	vector<fixed32, pinned_allocator<fixed32>> host_z(part_count);
	for (int i = 0; i < tree_ids_vector.size(); i++) {
		const tree_node* ptr = tree_get_node(tree_ids_vector[i]);
		const int local_index = tree_map[tree_ids_vector[i]];
		int part_index = next_index;
		particles_global_read_pos(ptr->global_part_range(), host_x.data(), host_y.data(), host_z.data(), part_index);
		adjust_part_references(tree_nodes, local_index, part_index - ptr->part_range.first);
		next_index += ptr->nparts();
	}
	CUDA_CHECK(cudaMemcpyAsync(dev_trees, tree_nodes.data(), tree_nodes.size() * sizeof(tree_node), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(dev_x, host_x.data(), sizeof(fixed32) * part_count, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(dev_y, host_y.data(), sizeof(fixed32) * part_count, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(dev_z, host_z.data(), sizeof(fixed32) * part_count, cudaMemcpyHostToDevice, stream));
	const auto kick_returns = cuda_execute_kicks(params, dev_x, dev_y, dev_z, dev_trees, std::move(workitems), stream);
	CUDA_CHECK(cudaFreeAsync(dev_x, stream));
	CUDA_CHECK(cudaFreeAsync(dev_y, stream));
	CUDA_CHECK(cudaFreeAsync(dev_z, stream));
	CUDA_CHECK(cudaFreeAsync(dev_trees, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));
	for( int i = 0; i < kick_returns.size(); i++) {
		promises[i].set_value(std::move(kick_returns[i]));
	}
	tm.stop();
	PRINT("To GPU Done %e\n", tm.read());

}

std::pair<bool, hpx::future<kick_return>> kick_workspace::add_work(expansion<float> L, array<fixed32, NDIM> pos, tree_id self, vector<tree_id> dchecks,
		vector<tree_id> echecks) {
//PRINT("Adding work\n");
	kick_workitem item;
	item.L = L;
	item.pos = pos;
	item.self = self;
	item.self.proc = 0;
	for (int i = 0; i < dchecks.size(); i++) {
		tree_ids.insert(dchecks[i]);
	}
	for (int i = 0; i < echecks.size(); i++) {
		tree_ids.insert(echecks[i]);
	}
	tree_ids.insert(self);
	item.dchecklist = std::move(dchecks);
	item.echecklist = std::move(echecks);
	std::unique_lock<mutex_type> lock(mutex);
	promises.resize(promises.size() + 1);
	auto fut = promises.back().get_future();
	workitems.push_back(std::move(item));
	lock.unlock();
	return std::make_pair(true, std::move(fut));
}
