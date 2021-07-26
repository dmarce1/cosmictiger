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

void kick_workspace::to_gpu(std::shared_ptr<kick_workspace> ptr) {
	timer tm;
	tm.start();
	PRINT("To GPU\n");
	auto tree_ids_vector = tree_ids.to_vector();
	PRINT("%i tree ids\n", tree_ids_vector.size());
	vector<vector<tree_id>> ids_by_depth(MAX_DEPTH);
	int node_count = 0;
	for (int i = 0; i < tree_ids_vector.size(); i++) {
		const tree_node* ptr = tree_get_node(tree_ids_vector[i]);
		node_count += ptr->node_count;
		ids_by_depth[ptr->depth].push_back(tree_ids_vector[i]);
	}
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
	PRINT( "%i %i\n", next_index, tree_map_vector.size());
	for (int i = 0; i < tree_map_vector.size(); i++) {
		tree_nodes[tree_map_vector[i].second] = *tree_get_node(tree_map_vector[i].first);
	}
	for (int i = 0; i < tree_nodes.size(); i++) {
		if (tree_nodes[i].children[LEFT].index != -1) {
			tree_nodes[i].children[LEFT].index = tree_map[tree_nodes[i].children[LEFT]];
			tree_nodes[i].children[RIGHT].index = tree_map[tree_nodes[i].children[RIGHT]];
		}
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
	promises.resize(promises.size() + 1);
	auto fut = promises.back().get_future();
	workitems.push_back(std::move(item));
	//PRINT("Done adding work\n");
	return std::make_pair(true, std::move(fut));
}
