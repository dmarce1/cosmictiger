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
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/all_tree.hpp>
#include <cosmictiger/sph_particles.hpp>


softlens_return all_tree_softlens_execute(int minrung);
softlens_return all_tree_derivatives_execute(int minrung);

HPX_PLAIN_ACTION (all_tree_find_ranges);
HPX_PLAIN_ACTION (all_tree_find_neighbors);
HPX_PLAIN_ACTION (all_tree_softlens_execute);
HPX_PLAIN_ACTION (all_tree_derivatives_execute);

static bool has_active_neighbors(const tree_node* self) {
	bool rc = false;
	for (int i = self->neighbor_range.first; i < self->neighbor_range.second; i++) {
		const auto id = tree_get_neighbor(i);
		if (tree_get_node(id)->nactive > 0) {
			rc = true;
			break;
		}
	}
	return rc;
}



softlens_return all_tree_softlens(int minrung) {
	tree_id root_id;
	vector<tree_id> checklist;
	root_id.proc = root_id.index = 0;
	checklist.push_back(root_id);
	softlens_return rc;
	do {
		all_tree_find_ranges(root_id, minrung, 1.2);
		all_tree_find_neighbors(root_id, std::move(checklist));
		rc = all_tree_softlens_execute(minrung);
	} while (rc.fail);
	return rc;
}

softlens_return all_tree_softlens_execute(int minrung) {
	vector<hpx::future<softlens_return>> rfuts;
	for (auto& c : hpx_children()) {
		rfuts.push_back(hpx::async<all_tree_softlens_execute_action>(c, minrung));
	}
	vector<tree_node, pinned_allocator<tree_node>> host_trees;
	vector<fixed32, pinned_allocator<fixed32>> host_x;
	vector<fixed32, pinned_allocator<fixed32>> host_y;
	vector<fixed32, pinned_allocator<fixed32>> host_z;
	vector<int, pinned_allocator<int>> host_neighbors;
	vector<int> host_selflist;
	std::unordered_map<tree_id, int, tree_id_hash> tree_map;
	std::unordered_map<int, pair<int>> neighbor_ranges;

	for (int i = 0; i < tree_leaflist_size(); i++) {
		const auto selfid = tree_get_leaf(i);
		const auto* self = tree_get_node(selfid);
		if (self->nactive) {
			std::unordered_map<tree_id, int, tree_id_hash>::iterator iter;
			iter = tree_map.find(selfid);
			if (iter == tree_map.end()) {
				int index = host_trees.size();
				host_trees.resize(index + 1);
				tree_map[selfid] = index;
				host_trees[index] = *self;
			}
			for (int i = self->neighbor_range.first; i < self->neighbor_range.second; i++) {
				const auto nid = tree_get_neighbor(i);
				iter = tree_map.find(nid);
				if (iter == tree_map.end()) {
					int index = host_trees.size();
					host_trees.resize(index + 1);
					tree_map[nid] = index;
					const auto* node = tree_get_node(nid);
					host_trees[index] = *node;
				}
			}
			int neighbor_begin = host_neighbors.size();
			for (int i = self->neighbor_range.first; i < self->neighbor_range.second; i++) {
				const auto nid = tree_get_neighbor(i);
				host_neighbors.push_back(tree_map[nid]);
			}
			int neighbor_end = host_neighbors.size();
			const int myindex = tree_map[selfid];
			auto& r = neighbor_ranges[host_selflist.size()];
			host_selflist.push_back(myindex);
			r.first = neighbor_begin;
			r.second = neighbor_end;
		}
	}
	size_t parts_size = 0;
	for (auto& node : host_trees) {
		parts_size += node.part_range.second - node.part_range.first;
	}
	host_x.resize(parts_size);
	host_y.resize(parts_size);
	host_z.resize(parts_size);
	vector<hpx::future<void>> futs;
	const int nthreads = 8 * hpx_hardware_concurrency();
	std::atomic<int> index(0);
	std::atomic<part_int> part_index(0);
	for (int i = 0; i < host_selflist.size(); i++) {
		host_trees[host_selflist[i]].neighbor_range = neighbor_ranges[i];
	}
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([&index,proc,nthreads,&part_index, &host_x, &host_y, &host_z, &host_trees]() {
			int this_index = index++;
			while( this_index < host_trees.size()) {
				auto& node = host_trees[this_index];
				const part_int size = node.part_range.second - node.part_range.first;
				const part_int offset = (part_index += size) - size;
				particles_global_read_pos(node.global_part_range(), host_x.data(), host_y.data(), host_z.data(), nullptr, offset);
				node.part_range.first = offset;
				node.part_range.second = offset + size;
				this_index = index++;
			}
		}));
	}
	hpx::wait_all(futs.begin(), futs.end());
	all_tree_data params;
	params.N = get_options().neighbor_number;
	params.softlen_snk = &particles_softlen(0);
	params.rung_snk = &particles_rung(0);
	params.converged_snk = &particles_converged(0);
	params.minrung = minrung;
	CUDA_CHECK(cudaMalloc(&params.selfs, sizeof(int) * host_selflist.size()));
	CUDA_CHECK(cudaMalloc(&params.x, sizeof(fixed32) * host_x.size()));
	CUDA_CHECK(cudaMalloc(&params.y, sizeof(fixed32) * host_y.size()));
	CUDA_CHECK(cudaMalloc(&params.z, sizeof(fixed32) * host_z.size()));
	CUDA_CHECK(cudaMalloc(&params.trees, sizeof(tree_node) * host_trees.size()));
	CUDA_CHECK(cudaMalloc(&params.neighbors, sizeof(int) * host_neighbors.size()));
	auto stream = cuda_get_stream();

	CUDA_CHECK(cudaMemcpyAsync(params.x, host_x.data(), sizeof(fixed32) * host_x.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.y, host_y.data(), sizeof(fixed32) * host_y.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.z, host_z.data(), sizeof(fixed32) * host_z.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.trees, host_trees.data(), sizeof(tree_node) * host_trees.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.selfs, host_selflist.data(), sizeof(int) * host_selflist.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.neighbors, host_neighbors.data(), sizeof(int) * host_neighbors.size(), cudaMemcpyHostToDevice, stream));
	auto rc = all_tree_softlens_cuda(params, stream);

	cuda_end_stream(stream);
	CUDA_CHECK(cudaFree(params.x));
	CUDA_CHECK(cudaFree(params.y));
	CUDA_CHECK(cudaFree(params.z));
	CUDA_CHECK(cudaFree(params.trees));
	CUDA_CHECK(cudaFree(params.selfs));
	CUDA_CHECK(cudaFree(params.neighbors));
	hpx::wait_all(rfuts.begin(), rfuts.end());
	return rc;
}


softlens_return all_tree_derivatives_execute(int minrung) {
	vector<hpx::future<softlens_return>> rfuts;
	for (auto& c : hpx_children()) {
		rfuts.push_back(hpx::async<all_tree_derivatives_execute_action>(c, minrung));
	}
	vector<tree_node, pinned_allocator<tree_node>> host_trees;
	vector<fixed32, pinned_allocator<fixed32>> host_x;
	vector<fixed32, pinned_allocator<fixed32>> host_y;
	vector<fixed32, pinned_allocator<fixed32>> host_z;
	vector<float, pinned_allocator<float>> host_h;
	vector<int, pinned_allocator<int>> host_neighbors;
	vector<int> host_selflist;
	std::unordered_map<tree_id, int, tree_id_hash> tree_map;
	std::unordered_map<int, pair<int>> neighbor_ranges;

	for (int i = 0; i < tree_leaflist_size(); i++) {
		const auto selfid = tree_get_leaf(i);
		const auto* self = tree_get_node(selfid);
		if (has_active_neighbors(self)) {
			std::unordered_map<tree_id, int, tree_id_hash>::iterator iter;
			iter = tree_map.find(selfid);
			if (iter == tree_map.end()) {
				int index = host_trees.size();
				host_trees.resize(index + 1);
				tree_map[selfid] = index;
				host_trees[index] = *self;
			}
			for (int i = self->neighbor_range.first; i < self->neighbor_range.second; i++) {
				const auto nid = tree_get_neighbor(i);
				iter = tree_map.find(nid);
				if (iter == tree_map.end()) {
					int index = host_trees.size();
					host_trees.resize(index + 1);
					tree_map[nid] = index;
					const auto* node = tree_get_node(nid);
					host_trees[index] = *node;
				}
			}
			int neighbor_begin = host_neighbors.size();
			for (int i = self->neighbor_range.first; i < self->neighbor_range.second; i++) {
				const auto nid = tree_get_neighbor(i);
				host_neighbors.push_back(tree_map[nid]);
			}
			int neighbor_end = host_neighbors.size();
			const int myindex = tree_map[selfid];
			auto& r = neighbor_ranges[host_selflist.size()];
			host_selflist.push_back(myindex);
			r.first = neighbor_begin;
			r.second = neighbor_end;
		}
	}
	size_t parts_size = 0;
	for (auto& node : host_trees) {
		parts_size += node.part_range.second - node.part_range.first;
	}
	host_x.resize(parts_size);
	host_y.resize(parts_size);
	host_z.resize(parts_size);
	host_h.resize(parts_size);
	vector<hpx::future<void>> futs;
	const int nthreads = 8 * hpx_hardware_concurrency();
	std::atomic<int> index(0);
	std::atomic<part_int> part_index(0);
	for (int i = 0; i < host_selflist.size(); i++) {
		host_trees[host_selflist[i]].neighbor_range = neighbor_ranges[i];
	}
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([&index,proc,nthreads,&part_index, &host_x, &host_y, &host_z, &host_trees, &host_h]() {
			int this_index = index++;
			while( this_index < host_trees.size()) {
				auto& node = host_trees[this_index];
				const part_int size = node.part_range.second - node.part_range.first;
				const part_int offset = (part_index += size) - size;
				particles_global_read_pos(node.global_part_range(), host_x.data(), host_y.data(), host_z.data(), nullptr, offset);
				particles_global_read_softlens(node.global_part_range(), host_h.data(),  offset);
				node.part_range.first = offset;
				node.part_range.second = offset + size;
				this_index = index++;
			}
		}));
	}
	hpx::wait_all(futs.begin(), futs.end());
	all_tree_data params;
	params.N = get_options().neighbor_number;
	params.softlen_snk = &particles_softlen(0);
	params.rung_snk = &particles_rung(0);
	params.converged_snk = &particles_converged(0);
	params.minrung = minrung;
	params.zeta_snk = &particles_zeta(0);
	params.cat_snk = &particles_cat_index(0);
	params.type_snk = &particles_type(0);
	params.sph_omega_snk = &sph_particles_omega(0);

	CUDA_CHECK(cudaMalloc(&params.selfs, sizeof(int) * host_selflist.size()));
	CUDA_CHECK(cudaMalloc(&params.x, sizeof(fixed32) * host_x.size()));
	CUDA_CHECK(cudaMalloc(&params.y, sizeof(fixed32) * host_y.size()));
	CUDA_CHECK(cudaMalloc(&params.z, sizeof(fixed32) * host_z.size()));
	CUDA_CHECK(cudaMalloc(&params.h, sizeof(float) * host_h.size()));
	CUDA_CHECK(cudaMalloc(&params.trees, sizeof(tree_node) * host_trees.size()));
	CUDA_CHECK(cudaMalloc(&params.neighbors, sizeof(int) * host_neighbors.size()));
	auto stream = cuda_get_stream();

	CUDA_CHECK(cudaMemcpyAsync(params.x, host_x.data(), sizeof(fixed32) * host_x.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.y, host_y.data(), sizeof(fixed32) * host_y.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.z, host_z.data(), sizeof(fixed32) * host_z.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.h, host_h.data(), sizeof(float) * host_h.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.trees, host_trees.data(), sizeof(tree_node) * host_trees.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.selfs, host_selflist.data(), sizeof(int) * host_selflist.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.neighbors, host_neighbors.data(), sizeof(int) * host_neighbors.size(), cudaMemcpyHostToDevice, stream));
	auto rc = all_tree_derivatives_cuda(params, stream);

	cuda_end_stream(stream);
	CUDA_CHECK(cudaFree(params.h));
	CUDA_CHECK(cudaFree(params.x));
	CUDA_CHECK(cudaFree(params.y));
	CUDA_CHECK(cudaFree(params.z));
	CUDA_CHECK(cudaFree(params.trees));
	CUDA_CHECK(cudaFree(params.selfs));
	CUDA_CHECK(cudaFree(params.neighbors));
	hpx::wait_all(rfuts.begin(), rfuts.end());
	return rc;
}

void all_tree_find_neighbors(tree_id self_id, vector<tree_id> checklist) {
	const auto& self = *tree_get_node(self_id);
	vector<tree_id> leaflist;
	vector<tree_id> nextlist;
	if (self.local_root) {
		tree_free_neighbor_list();
		tree_clear_neighbor_ranges();
	}
	do {
		for (int ci = 0; ci < checklist.size(); ci++) {
			const auto& other = *tree_get_node(checklist[ci]);
			if (self.ibox.periodic_intersects(other.obox) || self.obox.periodic_intersects(other.ibox)) {
				if (other.leaf) {
					leaflist.push_back(checklist[ci]);
				} else {
					nextlist.push_back(other.children[LEFT]);
					nextlist.push_back(other.children[RIGHT]);
				}
			}
		}
		checklist = std::move(nextlist);
		nextlist.resize(0);
	} while (checklist.size() && self.leaf);
	if (self.leaf) {
		pair<int> rng;
		rng.first = tree_allocate_neighbor_list(leaflist);
		rng.second = leaflist.size() + rng.first;
		tree_set_neighbor_range(self_id, rng);
	} else {
		checklist.insert(checklist.begin(), leaflist.begin(), leaflist.end());
		static const auto locs = hpx_localities();
		const auto children = self.children;
		all_tree_find_neighbors_action func;
		auto fut = hpx::async<all_tree_find_neighbors_action>(locs[children[LEFT].proc], children[LEFT], checklist);
		func(locs[children[RIGHT].proc], children[RIGHT], std::move(checklist));
		fut.get();
	}
}

pair<fixed32_range> all_tree_find_ranges(tree_id self_id, int minrung, double h_wt) {
	const auto& self = *tree_get_node(self_id);
	pair<fixed32_range> myrange;
	if (self.leaf) {
		for (int dim = 0; dim < NDIM; dim++) {
			myrange.first.begin[dim] = myrange.second.begin[dim] = 1.0;
			myrange.first.end[dim] = myrange.second.end[dim] = 0.0;
		}
		const auto tiny = 10.0 * range_fixed::min().to_double();
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			const double h = h_wt * particles_softlen(i) + tiny;
			for (int dim = 0; dim < NDIM; dim++) {
				const double x = particles_pos(dim, i).to_double();
				myrange.first.begin[dim] = std::min(myrange.first.begin[dim].to_double(), x - tiny);
				myrange.second.begin[dim] = std::min(myrange.second.begin[dim].to_double(), x - h);
				myrange.first.end[dim] = std::max(myrange.first.end[dim].to_double(), x + tiny);
				myrange.second.end[dim] = std::max(myrange.second.end[dim].to_double(), x + h);
			}
		}
	} else {
		static const auto locs = hpx_localities();
		const auto children = self.children;
		all_tree_find_ranges_action func;
		auto fut = hpx::async<all_tree_find_ranges_action>(locs[children[LEFT].proc], children[LEFT], minrung, h_wt);
		const auto rcr = func(locs[children[RIGHT].proc], children[RIGHT], minrung, h_wt);
		const auto rcl = fut.get();
		for (int dim = 0; dim < NDIM; dim++) {
			myrange.first.begin[dim] = min(rcr.first.begin[dim], rcl.first.begin[dim]);
			myrange.second.begin[dim] = min(rcr.second.begin[dim], rcl.second.begin[dim]);
			myrange.first.end[dim] = max(rcr.first.end[dim], rcl.first.end[dim]);
			myrange.second.end[dim] = max(rcr.second.end[dim], rcl.second.end[dim]);
		}
	}
	return myrange;
}
