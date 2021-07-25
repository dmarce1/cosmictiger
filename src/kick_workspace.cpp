#include <tigerfmm/kick_workspace.hpp>
#include <tigerfmm/particles.hpp>

kick_workspace::kick_workspace(kick_params p) {
	params = p;
	current_part = 0;
	nparts = size_t(KICK_WORKSPACE_PART_SIZE) * cuda_free_mem() / (NDIM * sizeof(fixed32)) / 100;
	CUDA_CHECK(cudaStreamCreate(&stream));
	CUDA_CHECK(cudaMallocAsync(&dev_x, nparts * sizeof(fixed32), stream));
	CUDA_CHECK(cudaMallocAsync(&dev_y, nparts * sizeof(fixed32), stream));
	CUDA_CHECK(cudaMallocAsync(&dev_z, nparts * sizeof(fixed32), stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));
}

kick_workspace::~kick_workspace() {
	CUDA_CHECK(cudaFreeAsync(dev_x, stream));
	CUDA_CHECK(cudaFreeAsync(dev_y, stream));
	CUDA_CHECK(cudaFreeAsync(dev_z, stream));
	tree_space = decltype(tree_space)();
	CUDA_CHECK(cudaStreamSynchronize(stream));
	CUDA_CHECK(cudaStreamDestroy(stream));
}

void kick_workspace::add_tree_node_descendants(tree_id id, int part_offset) {
	tree_node node = *tree_get_node(id);
	node.part_range.first += part_offset;
	node.part_range.second += part_offset;
	const auto children = node.children;
	tree_map[id] = tree_space.size();
	tree_space.push_back(node);
	if (children[LEFT].index != -1) {
		add_tree_node_descendants(children[LEFT], part_offset);
		add_tree_node_descendants(children[RIGHT], part_offset);
	}
}

void kick_workspace::add_tree_node(tree_id id, int part_base) {
	const tree_node* node_ptr = tree_get_node(id);
	const int nparts = node_ptr->nparts();
	vector<fixed32, pinned_allocator<fixed32>> x(nparts);
	vector<fixed32, pinned_allocator<fixed32>> y(nparts);
	vector<fixed32, pinned_allocator<fixed32>> z(nparts);
	particles_global_read_pos(node_ptr->global_part_range(), x.data(), y.data(), z.data(), 0);
	const int part_offset = part_base - node_ptr->part_range.first;
	CUDA_CHECK(cudaMemcpyAsync(dev_x + part_base, x.data(), node_ptr->nparts(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(dev_y + part_base, y.data(), node_ptr->nparts(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(dev_z + part_base, z.data(), node_ptr->nparts(), cudaMemcpyHostToDevice, stream));
	add_tree_node_descendants(id, part_offset);
}

bool kick_workspace::add_tree_list(vector<tree_id>& nodes) {
	int this_ntrees = 0;
	int this_nparts = 0;
	for (int i = 0; i < nodes.size(); i++) {
		if (tree_map.find(nodes[i]) == tree_map.end()) {
			const tree_node* node = tree_get_node(nodes[i]);
			this_ntrees += node->node_count;
			this_nparts += node->nparts();
		}
	}
	const int part_base = (current_part += this_nparts);
	if ((int) current_part >= nparts) {
		return false;
	}
	int tree_base = tree_space.size();
	int part_index = part_base;
	for (int i = 0; i < nodes.size(); i++) {
		if (tree_map.find(nodes[i]) == tree_map.end()) {
			const tree_node* node = tree_get_node(nodes[i]);
			add_tree_node(nodes[i], part_index);
			part_index += node->nparts();
		}
	}
	for (int i = tree_base; i < tree_base + this_ntrees; i++) {
		if (tree_space[i].children[0].index != -1) {
			for (int i = 0; i < NCHILD; i++) {
				auto& child = tree_space[i].children[i];
				child.index = tree_map[child];
			}
		}
	}
	for (int i = 0; i < nodes.size(); i++) {
		nodes[i].index = tree_map[nodes[i]];
	}
	return true;
}

void kick_workspace::to_gpu(std::shared_ptr<kick_workspace> ptr) {
	PRINT( "To GPU\n");
	CUDA_CHECK(cudaMallocAsync(&dev_tree_space, (int ) tree_space.size() * sizeof(tree_node), stream));
	CUDA_CHECK(cudaMemcpyAsync(dev_tree_space, tree_space.data(), (int ) tree_space.size() * sizeof(tree_node), cudaMemcpyHostToDevice, stream));
	hpx::apply([ptr]() {
		auto returns = cuda_execute_kicks(ptr->params, ptr->dev_x, ptr->dev_y, ptr->dev_z, ptr->dev_tree_space, ptr->workitems, ptr->stream);
		PRINT( "GPU done\n");
		for( int i = 0; i < returns.size(); i++) {
			ptr->promises[i].set_value(std::move(returns[i]));
		}
	});

}

std::pair<bool, hpx::future<kick_return>> kick_workspace::add_work(expansion<float> L, array<fixed32, NDIM> pos, tree_id self, vector<tree_id> dchecks,
		vector<tree_id> echecks) {
	PRINT( "Adding work\n");
	bool rc = add_tree_list(dchecks);
	rc = rc && add_tree_list(echecks);
	if (!rc) {
		return std::make_pair(false, hpx::make_ready_future(kick_return()));
	}
	kick_workitem item;
	item.L = L;
	item.pos = pos;
	item.self.index = tree_map[self];
	item.self.proc = 0;
	item.dchecklist = std::move(dchecks);
	item.echecklist = std::move(echecks);
	promises.resize(promises.size() + 1);
	auto fut = promises.back().get_future();
	workitems.push_back(std::move(item));
	PRINT( "Done adding work\n");
	return std::make_pair(true, std::move(fut));
}
