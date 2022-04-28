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
#include <cosmictiger/timer.hpp>

softlens_return all_tree_softlens_execute(int minrung);
softlens_return all_tree_derivatives_execute(int minrung, float a);
softlens_return all_tree_divv(int minrung, float a);

HPX_PLAIN_ACTION (all_tree_find_ranges);
HPX_PLAIN_ACTION (all_tree_find_neighbors);
HPX_PLAIN_ACTION (all_tree_softlens_execute);
HPX_PLAIN_ACTION (all_tree_divv);
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

bool is_converged(const tree_node* self, int minrung) {
	bool converged = true;
	for (int i = self->part_range.first; i < self->part_range.second; i++) {
		if (!particles_converged(i)) {
			converged = false;
			break;
		}
	}
	return converged;
}

softlens_return all_tree_softlens(int minrung, float a) {
	tree_id root_id;
	vector<tree_id> checklist;
	root_id.proc = root_id.index = 0;
	checklist.push_back(root_id);
	softlens_return rc;
	timer tm;
	particles_reset_converged();
	double softlen_buffer = 1.201;
	do {
		tm.reset();
		tm.start();
		PRINT("Find ranges\n");
		all_tree_find_ranges(root_id, minrung, softlen_buffer).get();
		tm.stop();
		tm.reset();
		tm.start();
		PRINT("Find neighbors\n");
		all_tree_find_neighbors(root_id, checklist).get();
		tm.stop();
		tm.reset();
		tm.start();
		PRINT("Softlens\n");
		rc = all_tree_softlens_execute(minrung);
		tm.stop();
		PRINT("softlens %e %e %e\n", rc.hmin, rc.hmax, tm.read());
	} while (rc.fail);
	softlen_buffer = 1.201;
	particles_reset_converged();
	do {
		tm.reset();
		tm.start();
		all_tree_find_ranges(root_id, minrung, softlen_buffer).get();
		tm.stop();
		tm.reset();
		tm.start();
		all_tree_find_neighbors(root_id, checklist).get();
		tm.stop();
		tm.reset();
		tm.start();
		rc = all_tree_derivatives_execute(minrung, a);
		tm.stop();
		PRINT("derivs %e %e %e\n", rc.hmin, rc.hmax, tm.read());
	} while (rc.fail);
	return rc;
}

struct tree_id_hash2 {
	inline size_t operator()(tree_id id) const {
		const int i = id.index;
		return i * hpx_size() + id.proc;
	}
};

softlens_return all_tree_softlens_execute(int minrung) {
	vector<hpx::future<softlens_return>> rfuts;
	for (auto& c : hpx_children()) {
		rfuts.push_back(hpx::async<all_tree_softlens_execute_action>(c, minrung));
	}
	vector<tree_node, pinned_allocator<tree_node>> host_trees;
	vector<fixed32, pinned_allocator<fixed32>> host_x;
	vector<fixed32, pinned_allocator<fixed32>> host_y;
	vector<fixed32, pinned_allocator<fixed32>> host_z;
	vector<char, pinned_allocator<char>> host_types;
	vector<int, pinned_allocator<int>> host_neighbors;
	vector<int> host_selflist;
	std::unordered_map<tree_id, int, tree_id_hash2> tree_map;
	std::unordered_map<int, pair<int>> neighbor_ranges;
	mutex_type mutex;
	static std::atomic<int> next;
	next = 0;
	int nthreads = KICK_OVERSUBSCRIPTION * hpx_hardware_concurrency();
	vector<hpx::future<void>> futs2;
	for (int proc = 0; proc < nthreads; proc++) {
		futs2.push_back(hpx::async([proc,nthreads,&mutex,minrung,&tree_map,&host_trees,&host_neighbors,&host_selflist,&neighbor_ranges]() {
			int i = next++;
			while( i < tree_leaflist_size()) {
				const auto selfid = tree_get_leaf(i);
				const auto* self = tree_get_node(selfid);
				if (self->nactive && !is_converged(self,minrung)) {
					std::unordered_map<tree_id, int, tree_id_hash2>::iterator iter;
					std::unique_lock<mutex_type> lock(mutex);
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
							lock.unlock();
							const auto* node = tree_get_node(nid);
							lock.lock();
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
				i = next++;
			}
		}));
	}
	hpx::wait_all(futs2.begin(), futs2.end());
	size_t parts_size = 0;
	for (auto& node : host_trees) {
		parts_size += node.part_range.second - node.part_range.first;
	}
	host_x.resize(parts_size);
	host_y.resize(parts_size);
	host_z.resize(parts_size);
	host_types.resize(parts_size);
	vector<hpx::future<void>> futs;
	std::atomic<int> index(0);
	std::atomic<part_int> part_index(0);
	for (int i = 0; i < host_selflist.size(); i++) {
		host_trees[host_selflist[i]].neighbor_range = neighbor_ranges[i];
	}
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([&index,proc,nthreads,&part_index, &host_x, &host_y, &host_z, &host_types, &host_trees]() {
			int this_index = index++;
			while( this_index < host_trees.size()) {
				auto& node = host_trees[this_index];
				const part_int size = node.part_range.second - node.part_range.first;
				const part_int offset = (part_index += size) - size;
				particles_global_read_pos(node.global_part_range(), host_x.data(), host_y.data(), host_z.data(), host_types.data(), nullptr, offset);
				node.part_range.first = offset;
				node.part_range.second = offset + size;
				this_index = index++;
			}
		}));
	}
	hpx::wait_all(futs.begin(), futs.end());
	all_tree_data params;
	params.hmin = get_options().hmin;
	params.N = get_options().gneighbor_number;
	params.softlen_snk = &particles_softlen(0);
	params.rung_snk = &particles_rung(0);
	params.type_snk = &particles_type(0);
	params.cat_snk = &particles_cat_index(0);
	params.sph_h_snk = &sph_particles_smooth_len(0);
	params.converged_snk = &particles_converged(0);
	params.minrung = minrung;
	params.nselfs = host_selflist.size();
	CUDA_CHECK(cudaMalloc(&params.selfs, sizeof(int) * host_selflist.size()));
	CUDA_CHECK(cudaMalloc(&params.x, sizeof(fixed32) * host_x.size()));
	CUDA_CHECK(cudaMalloc(&params.y, sizeof(fixed32) * host_y.size()));
	CUDA_CHECK(cudaMalloc(&params.z, sizeof(fixed32) * host_z.size()));
	CUDA_CHECK(cudaMalloc(&params.types, sizeof(char) * host_types.size()));
	CUDA_CHECK(cudaMalloc(&params.trees, sizeof(tree_node) * host_trees.size()));
	CUDA_CHECK(cudaMalloc(&params.neighbors, sizeof(int) * host_neighbors.size()));
	auto stream = cuda_get_stream();

	CUDA_CHECK(cudaMemcpyAsync(params.x, host_x.data(), sizeof(fixed32) * host_x.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.y, host_y.data(), sizeof(fixed32) * host_y.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.z, host_z.data(), sizeof(fixed32) * host_z.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.types, host_types.data(), sizeof(char) * host_types.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.trees, host_trees.data(), sizeof(tree_node) * host_trees.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.selfs, host_selflist.data(), sizeof(int) * host_selflist.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.neighbors, host_neighbors.data(), sizeof(int) * host_neighbors.size(), cudaMemcpyHostToDevice, stream));
	auto rc = all_tree_softlens_cuda(params, stream);

	cuda_end_stream(stream);
	CUDA_CHECK(cudaFree(params.x));
	CUDA_CHECK(cudaFree(params.y));
	CUDA_CHECK(cudaFree(params.z));
	CUDA_CHECK(cudaFree(params.types));
	CUDA_CHECK(cudaFree(params.trees));
	CUDA_CHECK(cudaFree(params.selfs));
	CUDA_CHECK(cudaFree(params.neighbors));
	hpx::wait_all(rfuts.begin(), rfuts.end());
	return rc;
}

softlens_return all_tree_derivatives_execute(int minrung, float a) {
	vector<hpx::future<softlens_return>> rfuts;
	for (auto& c : hpx_children()) {
		rfuts.push_back(hpx::async<all_tree_derivatives_execute_action>(c, minrung, a));
	}
	vector<tree_node, pinned_allocator<tree_node>> host_trees;
	vector<fixed32, pinned_allocator<fixed32>> host_x;
	vector<fixed32, pinned_allocator<fixed32>> host_y;
	vector<fixed32, pinned_allocator<fixed32>> host_z;
	vector<float, pinned_allocator<float>> host_h;
	vector<char, pinned_allocator<char>> host_types;
	vector<int, pinned_allocator<int>> host_neighbors;
	vector<int> host_selflist;
	std::unordered_map<tree_id, int, tree_id_hash2> tree_map;
	std::unordered_map<int, pair<int>> neighbor_ranges;

	mutex_type mutex;
	static std::atomic<int> next;
	next = 0;
	int nthreads = KICK_OVERSUBSCRIPTION * hpx_hardware_concurrency();
	vector<hpx::future<void>> futs2;
	for (int proc = 0; proc < nthreads; proc++) {
		futs2.push_back(hpx::async([proc,nthreads,&mutex,&tree_map,minrung,&host_trees,&host_types, &host_neighbors,&host_selflist,&neighbor_ranges]() {
			int i = next++;
			while( i < tree_leaflist_size()) {
				const auto selfid = tree_get_leaf(i);
				const auto* self = tree_get_node(selfid);
				if (has_active_neighbors(self) && !is_converged(self,minrung)) {
					std::unordered_map<tree_id, int, tree_id_hash2>::iterator iter;
					std::unique_lock<mutex_type> lock(mutex);
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
							lock.unlock();
							const auto* node = tree_get_node(nid);
							lock.lock();
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
				i = next++;
			}
		}));
	}
	hpx::wait_all(futs2.begin(), futs2.end());
	size_t parts_size = 0;
	for (auto& node : host_trees) {
		parts_size += node.part_range.second - node.part_range.first;
	}
	host_x.resize(parts_size);
	host_y.resize(parts_size);
	host_z.resize(parts_size);
	host_types.resize(parts_size);
	host_h.resize(parts_size);
	vector<hpx::future<void>> futs;
	std::atomic<int> index(0);
	std::atomic<part_int> part_index(0);
	for (int i = 0; i < host_selflist.size(); i++) {
		host_trees[host_selflist[i]].neighbor_range = neighbor_ranges[i];
	}
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([&index,proc,nthreads,&part_index, &host_x, &host_y, &host_z,&host_types, &host_trees, &host_h]() {
			int this_index = index++;
			while( this_index < host_trees.size()) {
				auto& node = host_trees[this_index];
				const part_int size = node.part_range.second - node.part_range.first;
				const part_int offset = (part_index += size) - size;
				particles_global_read_pos(node.global_part_range(), host_x.data(), host_y.data(), host_z.data(), host_types.data(), nullptr, offset);
				particles_global_read_softlens(node.global_part_range(), host_h.data(), offset);
				node.part_range.first = offset;
				node.part_range.second = offset + size;
				this_index = index++;
			}
		}));
	}
	hpx::wait_all(futs.begin(), futs.end());
	all_tree_data params;
	params.hmin = get_options().hmin;
	params.N = get_options().gneighbor_number;
	params.softlen_snk = &particles_softlen(0);
	params.rung_snk = &particles_rung(0);
	params.converged_snk = &particles_converged(0);
	params.minrung = minrung;
	params.zeta_snk = &particles_zeta(0);
	params.cat_snk = &particles_cat_index(0);
	params.type_snk = &particles_type(0);
	params.nselfs = host_selflist.size();
	params.sa_snk = &particles_semiactive(0);
	params.sph_h_snk = &sph_particles_smooth_len(0);
	params.a = a;

	CUDA_CHECK(cudaMalloc(&params.selfs, sizeof(int) * host_selflist.size()));
	CUDA_CHECK(cudaMalloc(&params.x, sizeof(fixed32) * host_x.size()));
	CUDA_CHECK(cudaMalloc(&params.y, sizeof(fixed32) * host_y.size()));
	CUDA_CHECK(cudaMalloc(&params.z, sizeof(fixed32) * host_z.size()));
	CUDA_CHECK(cudaMalloc(&params.h, sizeof(float) * host_h.size()));
	CUDA_CHECK(cudaMalloc(&params.types, sizeof(char) * host_types.size()));
	CUDA_CHECK(cudaMalloc(&params.trees, sizeof(tree_node) * host_trees.size()));
	CUDA_CHECK(cudaMalloc(&params.neighbors, sizeof(int) * host_neighbors.size()));
	auto stream = cuda_get_stream();

	CUDA_CHECK(cudaMemcpyAsync(params.x, host_x.data(), sizeof(fixed32) * host_x.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.y, host_y.data(), sizeof(fixed32) * host_y.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.z, host_z.data(), sizeof(fixed32) * host_z.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.h, host_h.data(), sizeof(float) * host_h.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.types, host_types.data(), sizeof(char) * host_types.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.trees, host_trees.data(), sizeof(tree_node) * host_trees.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.selfs, host_selflist.data(), sizeof(int) * host_selflist.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.neighbors, host_neighbors.data(), sizeof(int) * host_neighbors.size(), cudaMemcpyHostToDevice, stream));
	auto rc = all_tree_derivatives_cuda(params, stream);

	cuda_end_stream(stream);
	CUDA_CHECK(cudaFree(params.h));
	CUDA_CHECK(cudaFree(params.x));
	CUDA_CHECK(cudaFree(params.y));
	CUDA_CHECK(cudaFree(params.z));
	CUDA_CHECK(cudaFree(params.types));
	CUDA_CHECK(cudaFree(params.trees));
	CUDA_CHECK(cudaFree(params.selfs));
	CUDA_CHECK(cudaFree(params.neighbors));
	hpx::wait_all(rfuts.begin(), rfuts.end());
	return rc;
}

softlens_return all_tree_divv(int minrung, float a) {
	vector<hpx::future<softlens_return>> rfuts;
	for (auto& c : hpx_children()) {
		rfuts.push_back(hpx::async<all_tree_divv_action>(c, minrung, a));
	}
	vector<tree_node, pinned_allocator<tree_node>> host_trees;
	vector<fixed32, pinned_allocator<fixed32>> host_x;
	vector<fixed32, pinned_allocator<fixed32>> host_y;
	vector<fixed32, pinned_allocator<fixed32>> host_z;
	vector<float, pinned_allocator<float>> host_vx;
	vector<float, pinned_allocator<float>> host_vy;
	vector<float, pinned_allocator<float>> host_vz;
	vector<char, pinned_allocator<char>> host_types;
	vector<int, pinned_allocator<int>> host_neighbors;
	vector<int> host_selflist;
	std::unordered_map<tree_id, int, tree_id_hash2> tree_map;
	std::unordered_map<int, pair<int>> neighbor_ranges;

	mutex_type mutex;
	static std::atomic<int> next;
	next = 0;
	int nthreads = KICK_OVERSUBSCRIPTION * hpx_hardware_concurrency();
	vector<hpx::future<void>> futs2;
	for (int proc = 0; proc < nthreads; proc++) {
		futs2.push_back(hpx::async([proc,nthreads,&mutex,&tree_map,minrung,&host_trees,&host_neighbors,&host_selflist,&neighbor_ranges]() {
			int i = next++;
			while( i < tree_leaflist_size()) {
				const auto selfid = tree_get_leaf(i);
				const auto* self = tree_get_node(selfid);
				if (has_active_neighbors(self) && !is_converged(self,minrung)) {
					std::unordered_map<tree_id, int, tree_id_hash2>::iterator iter;
					std::unique_lock<mutex_type> lock(mutex);
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
							lock.unlock();
							const auto* node = tree_get_node(nid);
							lock.lock();
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
				i = next++;
			}
		}));
	}
	hpx::wait_all(futs2.begin(), futs2.end());
	size_t parts_size = 0;
	for (auto& node : host_trees) {
		parts_size += node.part_range.second - node.part_range.first;
	}
	host_x.resize(parts_size);
	host_y.resize(parts_size);
	host_z.resize(parts_size);
	host_types.resize(parts_size);
	host_vx.resize(parts_size);
	host_vy.resize(parts_size);
	host_vz.resize(parts_size);
	vector<hpx::future<void>> futs;
	std::atomic<int> index(0);
	std::atomic<part_int> part_index(0);
	for (int i = 0; i < host_selflist.size(); i++) {
		host_trees[host_selflist[i]].neighbor_range = neighbor_ranges[i];
	}
	for (int proc = 0; proc < nthreads; proc++) {
		futs.push_back(hpx::async([&index,proc,nthreads,&part_index, &host_x, &host_y, &host_z, &host_vx, &host_vy, &host_types, &host_vz, &host_trees]() {
			int this_index = index++;
			while( this_index < host_trees.size()) {
				auto& node = host_trees[this_index];
				const part_int size = node.part_range.second - node.part_range.first;
				const part_int offset = (part_index += size) - size;
				particles_global_read_pos(node.global_part_range(), host_x.data(), host_y.data(), host_z.data(), host_types.data(), nullptr, offset);
				particles_global_read_vels(node.global_part_range(), host_vx.data(), host_vy.data(), host_vz.data(), offset);
				node.part_range.first = offset;
				node.part_range.second = offset + size;
				this_index = index++;
			}
		}));
	}
	hpx::wait_all(futs.begin(), futs.end());
	all_tree_data params;
	params.hmin = get_options().hmin;
	params.N = get_options().gneighbor_number;
	params.softlen_snk = &particles_softlen(0);
	params.rung_snk = &particles_rung(0);
	params.minrung = minrung;
	params.divv_snk = &particles_divv(0);
	params.nselfs = host_selflist.size();
	params.a = a;
	params.gx_snk = &particles_gforce(XDIM, 0);
	params.gy_snk = &particles_gforce(YDIM, 0);
	params.gz_snk = &particles_gforce(ZDIM, 0);

	CUDA_CHECK(cudaMalloc(&params.selfs, sizeof(int) * host_selflist.size()));
	CUDA_CHECK(cudaMalloc(&params.x, sizeof(fixed32) * host_x.size()));
	CUDA_CHECK(cudaMalloc(&params.y, sizeof(fixed32) * host_y.size()));
	CUDA_CHECK(cudaMalloc(&params.z, sizeof(fixed32) * host_z.size()));
	CUDA_CHECK(cudaMalloc(&params.trees, sizeof(tree_node) * host_trees.size()));
	CUDA_CHECK(cudaMalloc(&params.neighbors, sizeof(int) * host_neighbors.size()));
	CUDA_CHECK(cudaMalloc(&params.vx, sizeof(float) * host_vx.size()));
	CUDA_CHECK(cudaMalloc(&params.vy, sizeof(float) * host_vy.size()));
	CUDA_CHECK(cudaMalloc(&params.vz, sizeof(float) * host_vz.size()));
	CUDA_CHECK(cudaMalloc(&params.types, sizeof(char) * host_types.size()));
	auto stream = cuda_get_stream();

	CUDA_CHECK(cudaMemcpyAsync(params.vx, host_vx.data(), sizeof(float) * host_vx.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.vy, host_vy.data(), sizeof(float) * host_vy.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.vz, host_vz.data(), sizeof(float) * host_vz.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.x, host_x.data(), sizeof(fixed32) * host_x.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.y, host_y.data(), sizeof(fixed32) * host_y.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.z, host_z.data(), sizeof(fixed32) * host_z.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.types, host_types.data(), sizeof(char) * host_types.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.trees, host_trees.data(), sizeof(tree_node) * host_trees.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.selfs, host_selflist.data(), sizeof(int) * host_selflist.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(params.neighbors, host_neighbors.data(), sizeof(int) * host_neighbors.size(), cudaMemcpyHostToDevice, stream));
	auto rc = all_tree_divv_cuda(params, stream);

	cuda_end_stream(stream);
	CUDA_CHECK(cudaFree(params.x));
	CUDA_CHECK(cudaFree(params.y));
	CUDA_CHECK(cudaFree(params.z));
	CUDA_CHECK(cudaFree(params.types));
	CUDA_CHECK(cudaFree(params.trees));
	CUDA_CHECK(cudaFree(params.selfs));
	CUDA_CHECK(cudaFree(params.neighbors));
	CUDA_CHECK(cudaFree(params.vx));
	CUDA_CHECK(cudaFree(params.vy));
	CUDA_CHECK(cudaFree(params.vz));
	hpx::wait_all(rfuts.begin(), rfuts.end());
	return rc;
}

hpx::future<void> all_tree_find_neighbors_fork(tree_id self_id, vector<tree_id> checklist, bool threadme) {
	static std::atomic<int> nthreads(0);
	hpx::future<void> rc;
	const tree_node* self_ptr = tree_get_node(self_id);
	bool remote = false;
	bool all_local = true;
	for (const auto& i : checklist) {
		if (i.proc != hpx_rank()) {
			all_local = false;
			break;
		}
	}
	if (self_id.proc != hpx_rank()) {
		threadme = true;
		remote = true;
	} else if (threadme) {
		threadme = self_ptr->part_range.second - self_ptr->part_range.first > MIN_KICK_THREAD_PARTS;
		if (threadme) {
			if (nthreads++ < KICK_OVERSUBSCRIPTION * hpx::thread::hardware_concurrency() || !self_ptr->is_local()) {
				threadme = true;
			} else {
				threadme = false;
				nthreads--;
			}
		}
	}
	if (!threadme) {
		if (all_local) {
			hpx_yield();
		}
		rc = all_tree_find_neighbors(self_id, std::move(checklist));
	} else if (remote) {
		rc = hpx::async<all_tree_find_neighbors_action>(HPX_PRIORITY_HI, hpx_localities()[self_ptr->proc_range.first], self_id, std::move(checklist));
	} else {
		const auto thread_priority = all_local ? HPX_PRIORITY_LO : HPX_PRIORITY_NORMAL;
		rc = hpx::async(thread_priority, [self_id] (vector<tree_id> checklist) {
			auto rc = all_tree_find_neighbors(self_id,std::move(checklist));
			nthreads--;
			return rc;
		}, std::move(checklist));
	}
	return rc;

}

hpx::future<void> all_tree_find_neighbors(tree_id self_id, vector<tree_id> checklist) {
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
		return hpx::make_ready_future();
	} else {
		checklist.insert(checklist.begin(), leaflist.begin(), leaflist.end());
		static const auto locs = hpx_localities();
		const auto children = self.children;
		array<hpx::future<void>, NCHILD> futs;
		futs[LEFT] = all_tree_find_neighbors_fork(children[LEFT], checklist, true);
		futs[RIGHT] = all_tree_find_neighbors_fork(children[RIGHT], std::move(checklist), false);
		return hpx::when_all(futs.begin(), futs.end());
	}
}

hpx::future<all_tree_range_return> all_tree_find_ranges_fork(tree_id self_id, int minrung, double h_wt, bool threadme) {
	static std::atomic<int> nthreads(0);
	hpx::future<all_tree_range_return> rc;
	const tree_node* self_ptr = tree_get_node(self_id);
	bool remote = false;
	if (self_id.proc != hpx_rank()) {
		threadme = true;
		remote = true;
	} else if (threadme) {
		threadme = self_ptr->part_range.second - self_ptr->part_range.first > MIN_KICK_THREAD_PARTS;
		if (threadme) {
			if (nthreads++ < KICK_OVERSUBSCRIPTION * hpx::thread::hardware_concurrency() || !self_ptr->is_local()) {
				threadme = true;
			} else {
				threadme = false;
				nthreads--;
			}
		}
	}
	if (!threadme) {
		rc = all_tree_find_ranges(self_id, minrung, h_wt);
	} else if (remote) {
		rc = hpx::async<all_tree_find_ranges_action>(HPX_PRIORITY_HI, hpx_localities()[self_ptr->proc_range.first], self_id,minrung, h_wt);
	} else {
		rc = hpx::async([self_id,minrung,h_wt] () {
			auto rc = all_tree_find_ranges(self_id,minrung,h_wt);
			nthreads--;
			return rc;
		});
	}
	return rc;
}

hpx::future<all_tree_range_return> all_tree_find_ranges(tree_id self_id, int minrung, double h_wt) {
	auto& self = *tree_get_node(self_id);
	if (self.leaf) {
		pair<fixed32_range> myrange;
		float hmax;
		for (int dim = 0; dim < NDIM; dim++) {
			myrange.first.begin[dim] = myrange.second.begin[dim] = 1.9;
			myrange.first.end[dim] = myrange.second.end[dim] = -0.9;
		}
		const auto tiny = 10.0 * range_fixed::min().to_double();
		double rph = 0.0;
		for (int i = self.part_range.first; i < self.part_range.second; i++) {
			const double h = particles_softlen(i) + tiny;
			double r2 = 0.0;
			for (int dim = 0; dim < NDIM; dim++) {
				const double x = particles_pos(dim, i).to_double();
				r2 += sqr(x - self.pos[dim].to_double());
				myrange.first.begin[dim] = std::min(myrange.first.begin[dim].to_double(), x - tiny);
				myrange.second.begin[dim] = std::min(myrange.second.begin[dim].to_double(), x - h_wt * h);
				myrange.first.end[dim] = std::max(myrange.first.end[dim].to_double(), x + tiny);
				myrange.second.end[dim] = std::max(myrange.second.end[dim].to_double(), x + h_wt * h);
			}
			double r = sqrt(r2) + h;
			rph = std::max(r, rph);
		}
		hmax = rph - self.radius;
		tree_set_boxes(self_id, myrange.first, myrange.second, hmax);
		all_tree_range_return rc;
		rc.ibox = myrange.first;
		rc.obox = myrange.second;
		rc.hmax = hmax;
		return hpx::make_ready_future(rc);
	} else {
		static const auto locs = hpx_localities();
		const auto children = self.children;
		array<hpx::future<all_tree_range_return>, NCHILD> futs;
		futs[RIGHT] = all_tree_find_ranges_fork(children[RIGHT], minrung, h_wt, true);
		futs[LEFT] = all_tree_find_ranges_fork(children[LEFT], minrung, h_wt, false);
		return when_all(futs.begin(), futs.end()).then([self_id](hpx::future<std::vector<hpx::future<all_tree_range_return>>> fut) {
			auto futs = fut.get();
			auto rcr = futs[RIGHT].get();
			auto rcl = futs[LEFT].get();
			pair<fixed32_range> myrange;
			float hmax;
			for (int dim = 0; dim < NDIM; dim++) {
				myrange.first.begin[dim] = min(rcr.ibox.begin[dim], rcl.ibox.begin[dim]);
				myrange.second.begin[dim] = min(rcr.obox.begin[dim], rcl.obox.begin[dim]);
				myrange.first.end[dim] = max(rcr.ibox.end[dim], rcl.ibox.end[dim]);
				myrange.second.end[dim] = max(rcr.obox.end[dim], rcl.obox.end[dim]);
			}
			hmax = std::max(rcr.hmax, rcl.hmax);
			all_tree_range_return rc;
			rc.ibox = myrange.first;
			rc.obox = myrange.second;
			rc.hmax = hmax;
			tree_set_boxes(self_id, myrange.first, myrange.second, hmax);
			return rc;
		});
	}
}
