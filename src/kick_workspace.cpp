#include <tigerfmm/kick_workspace.hpp>
#include <tigerfmm/particles.hpp>

kick_workspace::kick_workspace() :
		lock(0) {
	current_tree = current_part = 0;
	nparts = size_t(KICK_WORKSPACE_PART_SIZE) * cuda_free_mem() / (NDIM * sizeof(fixed32));
	const int ntrees1 = size_t(TREE_NODE_ALLOCATION_SIZE) * nparts / std::min(SOURCE_BUCKET_SIZE, SINK_BUCKET_SIZE);
	const int ntrees2 = NTREES_MIN * nparts / particles_size();
	ntrees = std::max(ntrees1, ntrees2);
	CUDA_CHECK(cudaStreamCreate(&stream));
	CUDA_CHECK(cudaMallocAsync(&dev_x, nparts * sizeof(fixed32), stream));
	CUDA_CHECK(cudaMallocAsync(&dev_y, nparts * sizeof(fixed32), stream));
	CUDA_CHECK(cudaMallocAsync(&dev_z, nparts * sizeof(fixed32), stream));
	tree_space.resize(ntrees);
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

void kick_workspace::add_tree_node_descendants(tree_id id, int part_offset, int& index) {
	tree_node node = *tree_get_node(id);
	node.part_range.first += part_offset;
	node.part_range.second += part_offset;
	const auto children = node.children;
	tree_space[index] = node;
	tree_map[id] = index;
	index++;
	add_tree_node_descendants(children[LEFT], part_offset, index);
	add_tree_node_descendants(children[RIGHT], part_offset, index);
}

void kick_workspace::add_tree_node(tree_id id, int part_base, int tree_base) {
	const tree_node* node_ptr = tree_get_node(id);
	vector<fixed32, pinned_allocator<fixed32>> x;
	vector<fixed32, pinned_allocator<fixed32>> y;
	vector<fixed32, pinned_allocator<fixed32>> z;
	particles_global_read_pos(node_ptr->global_part_range(), x.data(), y.data(), z.data(), 0);
	const int part_offset = part_base - node_ptr->part_range.first;
	CUDA_CHECK(cudaMemcpyAsync(dev_x + part_base, x.data(), node_ptr->nparts(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(dev_y + part_base, y.data(), node_ptr->nparts(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(dev_z + part_base, z.data(), node_ptr->nparts(), cudaMemcpyHostToDevice, stream));
	add_tree_node_descendants(id, part_offset, tree_base);
}

bool kick_workspace::add_tree_list(vector<tree_id>& nodes) {
	while (lock++ != 0) {
		lock--;
		hpx::this_thread::yield();
	}
	int this_ntrees = 0;
	int this_nparts = 0;
	for (int i = 0; i < nodes.size(); i++) {
		if (tree_map.find(nodes[i]) == tree_map.end()) {
			const tree_node* node = tree_get_node(nodes[i]);
			this_ntrees += node->node_count;
			this_nparts += node->nparts();
		}
	}
	const int tree_base = (current_tree += this_ntrees);
	const int part_base = (current_part += this_nparts);
	if ((int) current_tree >= ntrees) {
		lock--;
		return false;
	}
	if ((int) current_part >= nparts) {
		lock--;
		return false;
	}
	int tree_index = tree_base;
	int part_index = part_base;
	for (int i = 0; i < nodes.size(); i++) {
		if (tree_map.find(nodes[i]) == tree_map.end()) {
			const tree_node* node = tree_get_node(nodes[i]);
			add_tree_node(nodes[i], part_index, tree_index);
			tree_index += node->node_count;
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
	for( int i = 0; i < nodes.size(); i++) {
		nodes[i].index = tree_map[nodes[i]];
	}
	lock--;
	return true;
}

void kick_workspace::to_gpu() {
	CUDA_CHECK(cudaMallocAsync(&dev_tree_space, (int ) current_tree * sizeof(tree_node), stream));
	CUDA_CHECK(cudaMemcpyAsync(dev_tree_space, tree_space.data(), (int ) current_tree * sizeof(tree_node), cudaMemcpyHostToDevice, stream));

}
