#include <cosmictiger/bh.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/rockstar.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/hpx.hpp>

void rockstar_assign_link_len(const vector<rockstar_tree>& trees, vector<rockstar_particle>& parts, int self_id, vector<int> checklist, float link_len) {
	static thread_local vector<int> nextlist;
	static thread_local vector<int> leaflist;
	nextlist.resize(0);
	leaflist.resize(0);
	auto& self = trees[self_id];
	const bool iamleaf = self.children[LEFT] == -1;
	auto mybox = self.box.pad(link_len * 1.0001);
	do {
		for (int ci = 0; ci < checklist.size(); ci++) {
			const auto check = checklist[ci];
			const auto& other = trees[check];
			if (mybox.intersection(other.box).volume() > 0) {
				if (other.children[LEFT] == -1) {
					leaflist.push_back(check);
				} else {
					nextlist.push_back(other.children[LEFT]);
					nextlist.push_back(other.children[RIGHT]);
				}
			}
		}
		std::swap(nextlist, checklist);
		nextlist.resize(0);
	} while (iamleaf && checklist.size());
	if (iamleaf) {
		const float link_len2 = sqr(link_len);
		for (int li = 0; li < leaflist.size(); li++) {
			const auto leafi = leaflist[li];
			const auto other = trees[leafi];
			for (int pi = other.part_begin; pi < other.part_end; pi++) {
				const auto& part = parts[pi];
				if (mybox.contains(part.X)) {
					for (int pj = self.part_begin; pj < self.part_end; pj++) {
						auto& mypart = parts[pj];
						const float dx = mypart.x - part.x;
						const float dy = mypart.y - part.y;
						const float dz = mypart.z - part.z;
						const float dvx = mypart.vx - part.vx;
						const float dvy = mypart.vy - part.vy;
						const float dvz = mypart.vz - part.vz;
						const float R2 = sqr(dx, dy, dz) + sqr(dvx, dvy, dvz);
						if (R2 <= link_len2 && R2 > 0.0) {
							mypart.min_dist2 = std::min(mypart.min_dist2, R2);
						}
					}
				}
			}
		}
	} else {
		checklist.insert(checklist.end(), leaflist.begin(), leaflist.end());
		if (checklist.size()) {
			rockstar_assign_link_len(trees, parts, self.children[LEFT], checklist, link_len);
			rockstar_assign_link_len(trees, parts, self.children[RIGHT], std::move(checklist), link_len);
		}
	}
}

int rockstar_particles_sort(vector<rockstar_particle>& parts, int begin, int end, float xmid, int xdim) {
	int lo = begin;
	int hi = end;
	while (lo < hi) {
		if (parts[lo].X[xdim] >= xmid) {
			while (lo != hi) {
				hi--;
				if (parts[hi].X[xdim] < xmid) {
					std::swap(parts[hi], parts[lo]);
					break;
				}
			}
		}
		lo++;
	}
	return hi;

}

int rockstar_form_tree(vector<rockstar_particle>& parts, vector<rockstar_tree>& trees, range<float, 2 * NDIM>& rng, int part_begin, int part_end) {
	array<int, NCHILD> children;
	rockstar_tree node;
	int active_count = 0;
	if (part_end - part_begin <= ROCKSTAR_BUCKET_SIZE) {
		children[LEFT] = children[RIGHT] = -1;
		for (int dim = 0; dim < 2 * NDIM; dim++) {
			rng.begin[dim] = std::numeric_limits<float>::max() / 10.0;
			rng.end[dim] = -std::numeric_limits<float>::max() / 10.0;
		}
		for (int i = part_begin; i != part_end; i++) {
			for (int dim = 0; dim < 2 * NDIM; dim++) {
				const float x = parts[i].X[dim];
				rng.begin[dim] = std::min(rng.begin[dim], x);
				rng.end[dim] = std::max(rng.end[dim], x);
			}
		}
		active_count++;
	} else {
		float midx;
		int max_dim;
		float total_max = 0.0;
		for (int dim = 0; dim < 2 * NDIM; dim++) {
			const float x_max = rng.end[dim];
			const float x_min = rng.begin[dim];
			if (x_max - x_min > total_max) {
				total_max = x_max - x_min;
				max_dim = dim;
				midx = (x_max + x_min) * 0.5f;
			}
		}
		const int part_mid = rockstar_particles_sort(parts, part_begin, part_end, midx, max_dim);
		range<float, 2 * NDIM> rng_left = rng;
		range<float, 2 * NDIM> rng_right = rng;
		rng_right.begin[max_dim] = midx;
		rng_left.end[max_dim] = midx;
		children[LEFT] = rockstar_form_tree(parts, trees, rng_left, part_begin, part_mid);
		children[RIGHT] = rockstar_form_tree(parts, trees, rng_right, part_mid, part_end);
		for (int dim = 0; dim < 2 * NDIM; dim++) {
			rng.begin[dim] = std::min(rng_left.begin[dim], rng_right.begin[dim]);
			rng.end[dim] = std::max(rng_left.end[dim], rng_right.end[dim]);
		}
		active_count += trees[children[LEFT]].active_count;
		active_count += trees[children[RIGHT]].active_count;
	}
	node.part_begin = part_begin;
	node.part_end = part_end;
	node.children = children;
	node.box = rng;
	node.active_count = active_count;
	trees.push_back(node);
	return trees.size() - 1;
}

int rockstar_form_tree(vector<rockstar_tree>& trees, vector<rockstar_particle>& parts) {
	range<float, 2 * NDIM> rng;
	for (int dim = 0; dim < 2 * NDIM; dim++) {
		rng.begin[dim] = std::numeric_limits<float>::max() / 10.0;
		rng.end[dim] = -std::numeric_limits<float>::max() / 10.0;
		for (int i = 0; i < parts.size(); i++) {
			const float x = parts[i].X[dim];
			rng.begin[dim] = std::min(rng.begin[dim], x);
			rng.end[dim] = std::max(rng.end[dim], x);
		}
	}
	return rockstar_form_tree(parts, trees, rng, 0, parts.size());
}

float rockstar_find_link_len(const vector<rockstar_tree>& trees, vector<rockstar_particle>& parts, int tree_root, float ff) {
	range<float, 2 * NDIM> rng = trees[tree_root].box;
//	PRINT("V %e\n", rng.volume());
	vector<pair<float, int>> seps;
	for (int i = 0; i < trees.size(); i++) {
		if (trees[i].children[LEFT] == -1) {
			int count = trees[i].part_end - trees[i].part_begin;
			if (count) {
				float mean_sep = pow(trees[i].box.volume() / count, 1.0 / 6.0);
				seps.push_back(pair<float, int>(mean_sep, count));
			}
		}
	}
	std::sort(seps.begin(), seps.end(), [](pair<float,int> a, pair<float,int> b) {
		return a.first < b.first;
	});
	int ff_cnt = parts.size() * ff;
	int cnt = 0;
	float max_link_len;
	for (int i = 0; i < seps.size(); i++) {
		cnt += seps[i].second;
		if (cnt >= ff_cnt) {
			max_link_len = seps[i].first;
			break;
		}
	}
//	PRINT("%e ", max_link_len);
	bool done = false;
	do {
		for (auto& p : parts) {
			p.min_dist2 = std::numeric_limits<float>::max();
		}
		rockstar_assign_link_len(trees, parts, tree_root, vector<int>(1, tree_root), max_link_len);
		vector<float> dist2s(parts.size());
		for (int i = 0; i < parts.size(); i++) {
			dist2s[i] = parts[i].min_dist2;
		}
		std::sort(dist2s.begin(), dist2s.end());
		const int i0 = std::min((int) (ff * parts.size()), (int) (parts.size() - 2));
		const int i1 = i0 + 1;
		if (dist2s[i1] == std::numeric_limits<float>::max()) {
			//		PRINT( "doubling\n");
			max_link_len *= 2.0;
		} else {
			max_link_len = (sqrt(dist2s[i0]) + sqrt(dist2s[i1])) * 0.5f;
			done = true;
		}
	} while (!done);
//	PRINT("%e\n", max_link_len);
	return max_link_len;
}

size_t rockstar_find_subgroups(vector<rockstar_tree>& trees, vector<rockstar_particle>& parts, int self_id, vector<int> checklist, float link_len,
		int& next_id) {
	static thread_local vector<int> nextlist;
	static thread_local vector<int> leaflist;
	nextlist.resize(0);
	leaflist.resize(0);
	auto& self = trees[self_id];
	const bool iamleaf = self.children[LEFT] == -1;
	auto mybox = self.box.pad(link_len * 1.0001);
	do {
		for (int ci = 0; ci < checklist.size(); ci++) {
			const auto check = checklist[ci];
			const auto& other = trees[check];
			if (other.last_active) {
				if (mybox.intersection(other.box).volume() > 0) {
					if (other.children[LEFT] == -1) {
						leaflist.push_back(check);
					} else {
						nextlist.push_back(other.children[LEFT]);
						nextlist.push_back(other.children[RIGHT]);
					}
				}
			}
		}
		std::swap(nextlist, checklist);
		nextlist.resize(0);
	} while (iamleaf && checklist.size());
	if (iamleaf) {
		bool found_any_link = false;
		const float link_len2 = sqr(link_len);
		for (int li = 0; li < leaflist.size(); li++) {
			if (self_id != leaflist[li]) {
				const auto leafi = leaflist[li];
				const auto other = trees[leafi];
				for (int pi = other.part_begin; pi < other.part_end; pi++) {
					auto& part = parts[pi];
					if (mybox.contains(part.X)) {
						for (int pj = self.part_begin; pj < self.part_end; pj++) {
							auto& mypart = parts[pj];
							const float dx = mypart.x - part.x;
							const float dy = mypart.y - part.y;
							const float dz = mypart.z - part.z;
							const float dvx = mypart.vx - part.vx;
							const float dvy = mypart.vy - part.vy;
							const float dvz = mypart.vz - part.vz;
							const float R2 = sqr(dx, dy, dz) + sqr(dvx, dvy, dvz);
							if (R2 <= link_len2) {
								if (mypart.subgroup == ROCKSTAR_NO_GROUP) {
									mypart.subgroup = next_id++;
									found_any_link = true;
								}
								if (part.subgroup == ROCKSTAR_NO_GROUP) {
									part.subgroup = next_id++;
									found_any_link = true;
								}
								if (mypart.subgroup > part.subgroup) {
									mypart.subgroup = part.subgroup;
									found_any_link = true;
								} else if (mypart.subgroup < part.subgroup) {
									part.subgroup = mypart.subgroup;
									found_any_link = true;
								}
							}
						}
					}
				}
			}
		}
		bool found_link;
		do {
			found_link = false;
			for (int pi = self.part_begin; pi < self.part_end; pi++) {
				auto& A = parts[pi];
				for (int pj = pi + 1; pj < self.part_end; pj++) {
					auto& B = parts[pj];
					const float dx = A.x - B.x;
					const float dy = A.y - B.y;
					const float dz = A.z - B.z;
					const float dvx = A.vx - B.vx;
					const float dvy = A.vy - B.vy;
					const float dvz = A.vz - B.vz;
					const float R2 = sqr(dx, dy, dz) + sqr(dvx, dvy, dvz);
					if (R2 <= link_len2) {
						if (A.subgroup == ROCKSTAR_NO_GROUP) {
							A.subgroup = next_id++;
							found_any_link = true;
							found_link = true;
						}
						if (B.subgroup == ROCKSTAR_NO_GROUP) {
							B.subgroup = next_id++;
							found_any_link = true;
							found_link = true;
						}
						if (A.subgroup != B.subgroup) {
							if (A.subgroup < B.subgroup) {
								B.subgroup = A.subgroup;
							} else {
								A.subgroup = B.subgroup;
							}
							found_any_link = true;
							found_link = true;
						}
					}
				}
			}
		} while (found_link);
		self.active = found_any_link;
		return int(found_any_link);
	} else {
		checklist.insert(checklist.end(), leaflist.begin(), leaflist.end());
		int nactive = 0;
		if (checklist.size()) {
			nactive += rockstar_find_subgroups(trees, parts, self.children[LEFT], checklist, link_len, next_id);
			nactive += rockstar_find_subgroups(trees, parts, self.children[RIGHT], std::move(checklist), link_len, next_id);
		}
		self.active = nactive;
		return nactive;
	}

}

struct rockstar_gpu_workspace {
	vector<int> selves;
	vector<vector<int>> lists;
	vector<hpx::promise<size_t>> promises;
};

hpx::future<size_t> rockstar_find_subgroups_gpu_prep(rockstar_gpu_workspace& ws, vector<rockstar_tree, pinned_allocator<rockstar_tree>>& trees, int self_id, vector<int> checklist,
		float link_len) {
	static thread_local vector<int> nextlist;
	static thread_local vector<int> leaflist;
	int active_cnt_threshold = std::max(1, (int) (trees.back().active_count / ROCKSTAR_TARGET_BLOCKS));
	if (self_id == trees.size() - 1) {
		PRINT("active_cnt_threshold = %i\n", active_cnt_threshold);
	}
	nextlist.resize(0);
	leaflist.resize(0);
	auto& self = trees[self_id];
	const bool iamleaf = self.children[LEFT] == -1;
	auto mybox = self.box.pad(link_len * 1.0001);
	do {
		for (int ci = 0; ci < checklist.size(); ci++) {
			const auto check = checklist[ci];
			const auto& other = trees[check];
			if (other.last_active) {
				if (mybox.intersection(other.box).volume() > 0) {
					if (other.children[LEFT] == -1) {
						leaflist.push_back(check);
					} else {
						nextlist.push_back(other.children[LEFT]);
						nextlist.push_back(other.children[RIGHT]);
					}
				}
			}
		}
		std::swap(nextlist, checklist);
		nextlist.resize(0);
	} while (iamleaf && checklist.size());
	auto* self_ptr = &self;
	checklist.insert(checklist.end(), leaflist.begin(), leaflist.end());
	if (iamleaf || self.active_count <= active_cnt_threshold) {
		hpx::future<size_t> fut;
		ws.selves.push_back(self_id);
		ws.lists.push_back(std::move(checklist));
		ws.promises.resize(ws.promises.size() + 1);
		fut = ws.promises.back().get_future();
		auto next_func = [self_ptr](hpx::future<size_t>&& fut) {
			self_ptr->active_count = fut.get();
			self_ptr->active = self_ptr->active_count;
			return (size_t) self_ptr->active_count;
		};
		auto next_fut = fut.then(next_func);
		return next_fut;
	} else {
		int nactive = 0;
		if (checklist.size()) {
			array<hpx::future<size_t>, NCHILD> futs;
			futs[LEFT] = rockstar_find_subgroups_gpu_prep(ws, trees, self.children[LEFT], checklist, link_len);
			futs[RIGHT] = rockstar_find_subgroups_gpu_prep(ws, trees, self.children[RIGHT], std::move(checklist), link_len);
			return hpx::when_all(futs.begin(), futs.end()).then([self_ptr](hpx::future<std::vector<hpx::future<size_t>>>&& fut) {
				auto futs = fut.get();
				const auto cnt = futs[LEFT].get() + futs[RIGHT].get();
				self_ptr->active_count = cnt;
				self_ptr->active = cnt;
				return cnt;
			});
		} else {
			self_ptr->active_count = 0;
			self_ptr->active = false;
			return hpx::make_ready_future((size_t) 0);
		}
	}
}

void rockstar_find_subgroups_gpu_run(rockstar_gpu_workspace& ws, vector<rockstar_tree, pinned_allocator<rockstar_tree>>& trees, rockstar_particles parts, float link_len, int& next_id) {
	auto results = rockstar_find_subgroups_gpu(trees, parts, ws.selves, ws.lists, link_len, next_id);
	for (int i = 0; i < results.size(); i++) {
		ws.promises[i].set_value(results[i]);
	}
}

void rockstar_find_subgroups(vector<rockstar_tree>& trees, vector<rockstar_particle>& parts, float link_len, int& next_id) {
	timer tm;
	tm.start();
	for (int i = 0; i < parts.size(); i++) {
		parts[i].subgroup = ROCKSTAR_NO_GROUP;
	}
	for (int i = 0; i < trees.size(); i++) {
		trees[i].active = true;
	}
	int cnt;
	bool gpu = false;
	rockstar_particles part_ptrs;
	vector<float, pinned_allocator<float>> x;
	vector<float, pinned_allocator<float>> y;
	vector<float, pinned_allocator<float>> z;
	vector<float, pinned_allocator<float>> vx;
	vector<float, pinned_allocator<float>> vy;
	vector<float, pinned_allocator<float>> vz;
	vector<int, pinned_allocator<int>> sg;
	vector<rockstar_tree, pinned_allocator<rockstar_tree>> dev_trees(trees.begin(), trees.end());
	if (gpu) {
		x.resize(parts.size());
		y.resize(parts.size());
		z.resize(parts.size());
		vx.resize(parts.size());
		vy.resize(parts.size());
		vz.resize(parts.size());
		sg.resize(parts.size());
		for (int i = 0; i < parts.size(); i++) {
			x[i] = parts[i].x;
			y[i] = parts[i].y;
			z[i] = parts[i].z;
			vx[i] = parts[i].vx;
			vy[i] = parts[i].vy;
			vz[i] = parts[i].vz;
			sg[i] = parts[i].subgroup;
		}
		CUDA_CHECK(cudaMalloc(&part_ptrs.x, sizeof(float) * parts.size()));
		CUDA_CHECK(cudaMalloc(&part_ptrs.y, sizeof(float) * parts.size()));
		CUDA_CHECK(cudaMalloc(&part_ptrs.z, sizeof(float) * parts.size()));
		CUDA_CHECK(cudaMalloc(&part_ptrs.vx, sizeof(float) * parts.size()));
		CUDA_CHECK(cudaMalloc(&part_ptrs.vy, sizeof(float) * parts.size()));
		CUDA_CHECK(cudaMalloc(&part_ptrs.vz, sizeof(float) * parts.size()));
		CUDA_CHECK(cudaMalloc(&part_ptrs.subgroup, sizeof(int) * parts.size()));
		CUDA_CHECK(cudaMemcpy(part_ptrs.x, x.data(), parts.size() * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(part_ptrs.y, y.data(), parts.size() * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(part_ptrs.z, z.data(), parts.size() * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(part_ptrs.vx, vx.data(), parts.size() * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(part_ptrs.vy, vy.data(), parts.size() * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(part_ptrs.vz, vz.data(), parts.size() * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(part_ptrs.subgroup, sg.data(), parts.size() * sizeof(int), cudaMemcpyHostToDevice));
	}
	do {
		double pct_active = 0.0;
		if( gpu ) {
			for (int i = 0; i < trees.size(); i++) {
				dev_trees[i].last_active = dev_trees[i].active;
				pct_active += int(dev_trees[i].active) / (double) dev_trees.size();
			}
		} else {
			for (int i = 0; i < trees.size(); i++) {
				trees[i].last_active = trees[i].active;
				pct_active += int(trees[i].active) / (double) trees.size();
			}
		}
		int root_id = trees.size() - 1;
		if (!gpu) {
			cnt = rockstar_find_subgroups(trees, parts, root_id, vector<int>(1, root_id), link_len, next_id);
		} else {
			rockstar_gpu_workspace workspace;
			auto fut = rockstar_find_subgroups_gpu_prep(workspace, dev_trees, root_id, vector<int>(1, root_id), link_len);
			rockstar_find_subgroups_gpu_run(workspace, dev_trees, part_ptrs, link_len, next_id);
			cnt = fut.get();
		}
		PRINT("%i %i %e\n", root_id + 1, parts.size(), pct_active);
	} while (cnt != 0);
	if (gpu) {
		CUDA_CHECK(cudaMemcpy(sg.data(), part_ptrs.subgroup, parts.size() * sizeof(int), cudaMemcpyDeviceToHost));
		for (int i = 0; i < sg.size(); i++) {
			parts[i].subgroup = sg[i];
		}
		CUDA_CHECK(cudaFree(part_ptrs.x));
		CUDA_CHECK(cudaFree(part_ptrs.y));
		CUDA_CHECK(cudaFree(part_ptrs.z));
		CUDA_CHECK(cudaFree(part_ptrs.vx));
		CUDA_CHECK(cudaFree(part_ptrs.vy));
		CUDA_CHECK(cudaFree(part_ptrs.vz));
		CUDA_CHECK(cudaFree(part_ptrs.subgroup));
	}
	tm.stop();
	PRINT( "find_subgroups = %e\n", tm.read());
}

struct number {
	int n;
	number() {
		n = 0;
	}
};

std::unordered_map<int, number> rockstar_subgroup_cnts(vector<rockstar_particle>& parts) {
	std::unordered_map<int, number> table;
	for (int i = 0; i < parts.size(); i++) {
		if (parts[i].subgroup != ROCKSTAR_NO_GROUP) {
			table[parts[i].subgroup].n++;
		}
	}
	for (int i = 0; i < parts.size(); i++) {
		if (parts[i].subgroup != ROCKSTAR_NO_GROUP) {
			if (table[parts[i].subgroup].n < ROCKSTAR_MIN_GROUP) {
				parts[i].subgroup = ROCKSTAR_NO_GROUP;
			}
		}
	}
	for (auto i = table.begin(); i != table.end();) {
		if (i->second.n < ROCKSTAR_MIN_GROUP) {
			i = table.erase(i);
		} else {
			i++;
		}
	}
	return table;
}

struct rockstar_seed {
	int id;
	array<float, 2 * NDIM> x;
};

bool rockstar_halo_bound(const subgroup& a, const subgroup& b, float scale) {
	vector<array<float, NDIM>> x;
	vector<array<float, NDIM>> y;
	for (int i = 0; i < a.parts.size(); i++) {
		array<float, NDIM> X;
		for (int dim = 0; dim < NDIM; dim++) {
			X[dim] = a.parts[i].X[dim];
		}
		x.push_back(X);
	}
	for (int i = 0; i < b.parts.size(); i++) {
		array<float, NDIM> X;
		for (int dim = 0; dim < NDIM; dim++) {
			X[dim] = b.parts[i].X[dim];
		}
		y.push_back(X);
	}
	auto phi = bh_evaluate_points(y, x);
	auto phi_tot = 0.0;
	auto kin_tot = 0.0;
	float vxa = 0.0;
	float vya = 0.0;
	float vza = 0.0;
	for (int i = 0; i < a.parts.size(); i++) {
		vxa += a.parts[i].vx;
		vya += a.parts[i].vy;
		vza += a.parts[i].vz;
	}
	for (int i = 0; i < b.parts.size(); i++) {
		vxa += b.parts[i].vx;
		vya += b.parts[i].vy;
		vza += b.parts[i].vz;
	}
	vxa /= a.parts.size() + b.parts.size();
	vya /= a.parts.size() + b.parts.size();
	vza /= a.parts.size() + b.parts.size();
	for (int i = 0; i < b.parts.size(); i++) {
		const float dvx = b.parts[i].vx - vxa;
		const float dvy = b.parts[i].vy - vya;
		const float dvz = b.parts[i].vz - vza;
		const float ekin = 0.5 * sqr(dvx, dvy, dvz);
		kin_tot += ekin;
		phi_tot += phi[i] / scale;
	}
//	PRINT( "%e %e\n", kin_tot, phi_tot);
	if (kin_tot + phi_tot > 0.0) {
		return false;
	} else {
		//	PRINT("BOUND\n");
		return true;
	}
}

vector<int> rockstar_all_subgroups(const subgroup& sg, vector<subgroup>& subgroups) {
	vector<int> sgs;
	if (sg.parts.size() >= ROCKSTAR_MIN_GROUP) {
		sgs.push_back(sg.id);
	}
	if (sg.children.size()) {
		for (int i = 0; i < sg.children.size(); i++) {
			int l;
			for (l = 0; l < subgroups.size(); l++) {
				if (subgroups[l].id == sg.children[i]) {
					break;
				}
			}
			auto these_sgs = rockstar_all_subgroups(subgroups[l], subgroups);
			sgs.insert(sgs.end(), these_sgs.begin(), these_sgs.end());
		}
	}
	return sgs;
}

vector<rockstar_particle> rockstar_unbind(vector<subgroup>& subgroups, vector<int> checklist, float scale, int depth) {
	vector<rockstar_particle> my_parts;
	for (int ci = 0; ci < checklist.size(); ci++) {
//		PRINT("checklist = %i\n", ci);
		auto& sg = subgroups[checklist[ci]];
		auto& parts = sg.parts;
		if (sg.children.size()) {
			vector<int> next_children;
			for (int k = 0; k < sg.children.size(); k++) {
				for (int j = 0; j < subgroups.size(); j++) {
					if (subgroups[j].id == sg.children[k]) {
						next_children.push_back(j);
						break;
					}
				}
			}
			auto my_parts = rockstar_unbind(subgroups, next_children, scale, depth + 1);
			parts.insert(parts.end(), my_parts.begin(), my_parts.end());
		}
		float vx0, vy0, vz0;
		float vx1, vy1, vz1;
		int p1sz;
		REDO: vx0 = vy0 = vz0 = 0.0;
		vx1 = vy1 = vz1 = 0.0;
		p1sz = 0;
		for (int j = 0; j < parts.size(); j++) {
			vx0 += parts[j].vx;
			vy0 += parts[j].vy;
			vz0 += parts[j].vz;
			if (parts[j].subgroup == sg.id) {
				vx1 += parts[j].vx;
				vy1 += parts[j].vy;
				vz1 += parts[j].vz;
				p1sz++;
			}
		}
		vx0 /= parts.size();
		vy0 /= parts.size();
		vz0 /= parts.size();
		if (p1sz == 0) {
			THROW_ERROR("err");
		}
		vx1 /= p1sz;
		vy1 /= p1sz;
		vz1 /= p1sz;
		vector<array<float, NDIM>> x0(parts.size());
		vector<array<float, NDIM>> x1;
		vector<int> map(parts.size());
		for (int i = 0; i < parts.size(); i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				x0[i][dim] = parts[i].X[dim];
			}
			map[i] = x1.size();
			if (parts[i].subgroup == sg.id) {
				x1.push_back(x0[i]);
			}
		}
		timer tm;
		tm.start();
//		PRINT("x0size = %i x1size = %i\n", x0.size(), x1.size());
		auto phi0 = bh_evaluate_potential(x0);
		auto phi1 = bh_evaluate_potential(x1);
		float ainv = 1.0 / scale;
		for (auto& p : phi0) {
			p *= ainv;
		}
		for (auto& p : phi1) {
			p *= ainv;
		}
		int cnt;
		int cnt0;
		cnt = cnt0 = 0;
		for (int i = 0; i < parts.size(); i++) {
			float v20 = sqr(parts[i].vx - vx0);
			v20 += sqr(parts[i].vy - vy0);
			v20 += sqr(parts[i].vz - vz0);
			float kin0 = 0.5 * v20;
			bool host_bound = false;
			if (parts[i].subgroup == sg.id) {
				float v21 = sqr(parts[i].vx - vx1);
				v21 += sqr(parts[i].vy - vy1);
				v21 += sqr(parts[i].vz - vz1);
				float kin1 = 0.5 * v21;
				host_bound = kin1 + phi1[map[i]] < 0.0;
			}
			if (parts[i].subgroup == sg.id) {
				if (!(kin0 + phi0[i] < 0.0 || host_bound)) {
					parts[i].subgroup = sg.parent;
					my_parts.push_back(parts[i]);
					parts[i] = parts.back();
					phi0[i] = phi0.back();
					map[i] = map.back();
					map.pop_back();
					parts.pop_back();
					phi0.pop_back();
					i--;
					cnt++;
				}
				cnt0++;
			}
		}

		if (parts.size() < ROCKSTAR_MIN_GROUP || cnt * 2 > cnt0) {
			for (int i = 0; i < parts.size(); i++) {
				if (parts[i].subgroup == sg.id) {
					parts[i].subgroup = sg.parent;
				}
			}
			for (int k = 0; k < subgroups.size(); k++) {
				if (subgroups[k].parent == sg.id) {
					subgroups[k].parent = sg.parent;
				}
			}
			int l = 0;
			while (l < parts.size()) {
				if (parts[l].subgroup == sg.parent) {
					my_parts.push_back(parts[l]);
					parts[l] = parts.back();
					parts.pop_back();
				} else {
					l++;
				}
			}
		} else {
			for (int i = 0; i < parts.size(); i++) {
				my_parts.push_back(parts[i]);
			}
		}
		assert(parts.size() >= ROCKSTAR_MIN_GROUP || parts.size() == 0);
//		PRINT("%e unbound of %i\n", (float ) cnt / cnt0, cnt0);
		tm.stop();
		sg.depth = depth;

	}
	return my_parts;
}

void rockstar_subgroup_statistics(subgroup& sg) {
	vector<rockstar_tree> trees;
	vector<rockstar_particle> parts;
	for (int i = 0; i < sg.parts.size(); i++) {
		if (sg.parts[i].subgroup == sg.id) {
			parts.push_back(sg.parts[i]);
		}
	}
	float sigma2_x = 0.0;
	float sigma2_v = 0.0;
	float xcom = 0.0;
	float ycom = 0.0;
	float zcom = 0.0;
	float vxcom = 0.0;
	float vycom = 0.0;
	float vzcom = 0.0;
	int N = 0;
	for (int i = 0; i < parts.size(); i++) {
		xcom += parts[i].x;
		ycom += parts[i].y;
		zcom += parts[i].z;
		vxcom += parts[i].vx;
		vycom += parts[i].vy;
		vzcom += parts[i].vz;
		N++;
	}
	xcom /= parts.size();
	ycom /= parts.size();
	zcom /= parts.size();
	vxcom /= parts.size();
	vycom /= parts.size();
	vzcom /= parts.size();
	for (int i = 0; i < parts.size(); i++) {
		sigma2_x += sqr(parts[i].x - xcom);
		sigma2_x += sqr(parts[i].y - ycom);
		sigma2_x += sqr(parts[i].z - zcom);
		sigma2_v += sqr(parts[i].vx - vxcom);
		sigma2_v += sqr(parts[i].vy - vycom);
		sigma2_v += sqr(parts[i].vz - vzcom);
		N++;
	}
	if (parts.size() >= ROCKSTAR_MIN_GROUP) {
		float xfac = 1.0 / sqrt(sigma2_x);
		float vfac = 1.0 / sqrt(sigma2_v);
		for (int i = 0; i < parts.size(); i++) {
			parts[i].x *= xfac;
			parts[i].y *= xfac;
			parts[i].z *= xfac;
			parts[i].vx *= vfac;
			parts[i].vy *= vfac;
			parts[i].vz *= vfac;
		}
		if (parts.size() < 2) {
//			PRINT("PARTS.SIZE = %i %i \n", parts.size(), sg.parts.size());
		}
		const int root_id = rockstar_form_tree(trees, parts);

		const float link_len = rockstar_find_link_len(trees, parts, root_id, ROCKSTAR_FF);
		std::sort(parts.begin(), parts.end(), [](rockstar_particle a, rockstar_particle b) {
			return a.min_dist2 < b.min_dist2;
		});
		sigma2_v = sigma2_x = 0.0;
		xcom = ycom = zcom = 0.0;
		N = 1;
		xcom = parts[0].x;
		ycom = parts[0].y;
		zcom = parts[0].z;
		float min_xfactor = std::numeric_limits<float>::max();
		float xcom_min, ycom_min, zcom_min;
		for (int i = 1; i < parts.size(); i++) {
			sigma2_x *= N;
			const float factor = (N * N + 1) / (N + 1) / (N + 1);
			sigma2_x += factor * sqr((parts[i].x - xcom));
			sigma2_x += factor * sqr((parts[i].y - ycom));
			sigma2_x += factor * sqr((parts[i].z - zcom));
			xcom = (N * xcom + parts[i].x) / (N + 1);
			ycom = (N * ycom + parts[i].y) / (N + 1);
			zcom = (N * zcom + parts[i].z) / (N + 1);
			sigma2_x /= N + 1;
			N++;
			if (N >= ROCKSTAR_MIN_GROUP) {
				float xfactor = sqrt(sigma2_x / N);
//				PRINT("---%i %e\n", N, xfactor);
				if (xfactor < min_xfactor) {
					min_xfactor = xfactor;
					xcom_min = xcom;
					ycom_min = ycom;
					zcom_min = zcom;
				}
			}
		}
	}
	sg.host_part_cnt = parts.size();
}

vector<subgroup> rockstar_seeds(vector<rockstar_particle> parts, int& next_id, float rfac, float vfac, float scale, int depth = 0) {

	float avg_x = 0.0;
	float avg_y = 0.0;
	float avg_z = 0.0;
	float avg_vx = 0.0;
	float avg_vy = 0.0;
	float avg_vz = 0.0;

	for (int i = 0; i < parts.size(); i++) {
		avg_x += parts[i].x;
		avg_y += parts[i].y;
		avg_z += parts[i].z;
		avg_vx += parts[i].vx;
		avg_vy += parts[i].vy;
		avg_vz += parts[i].vz;
	}
	avg_x /= parts.size();
	avg_y /= parts.size();
	avg_z /= parts.size();
	avg_vx /= parts.size();
	avg_vy /= parts.size();
	avg_vz /= parts.size();
	float sigma2_x = 0.0;
	float sigma2_v = 0.0;
	for (int i = 0; i < parts.size(); i++) {
		parts[i].x -= avg_x;
		parts[i].y -= avg_y;
		parts[i].z -= avg_z;
		parts[i].vx -= avg_vx;
		parts[i].vy -= avg_vy;
		parts[i].vz -= avg_vz;
		sigma2_x += sqr(parts[i].x);
		sigma2_x += sqr(parts[i].y);
		sigma2_x += sqr(parts[i].z);
		sigma2_v += sqr(parts[i].vx);
		sigma2_v += sqr(parts[i].vy);
		sigma2_v += sqr(parts[i].vz);
	}
	sigma2_x /= parts.size();
	sigma2_v /= parts.size();
	float sigma_x = sqrt(sigma2_x);
	float sigma_v = sqrt(sigma2_v);
	float sigmainv_x = 1.0 / sigma_x;
	float sigmainv_v = 1.0 / sigma_v;
	for (int i = 0; i < parts.size(); i++) {
		parts[i].x *= sigmainv_x;
		parts[i].y *= sigmainv_x;
		parts[i].z *= sigmainv_x;
		parts[i].vx *= sigmainv_v;
		parts[i].vy *= sigmainv_v;
		parts[i].vz *= sigmainv_v;
	}
	rfac *= sigmainv_x;
	vfac *= sigmainv_v;
	vector<rockstar_tree> trees;
//	PRINT("Finding link_len\n");
	timer tm;
	tm.start();
//	PRINT("forming trees = %e\n");
	const int root_id = rockstar_form_tree(trees, parts);
	tm.stop();
//	PRINT("form_trees = %e\n", tm.read());
	tm.reset();
	tm.start();
	const float link_len = rockstar_find_link_len(trees, parts, root_id, ROCKSTAR_FF);
	tm.stop();
	//PRINT("find_link_len = %e\n", tm.read());
	tm.reset();
//	PRINT("link_len = %e\n", link_len / rfac);
//	PRINT("Finding subgroups\n");
	tm.start();
	rockstar_find_subgroups(trees, parts, link_len, next_id);
	tm.stop();
//	PRINT("find_subgroups = %e\n", tm.read());
	tm.reset();
//	PRINT("Finding group_cnts\n");
	tm.start();
	auto group_cnts = rockstar_subgroup_cnts(parts);
	tm.stop();
	//PRINT("subgroups_cnts = %e\n", tm.read());
	tm.reset();
//	PRINT("Found %i groups\n", group_cnts.size());

	vector<subgroup> subgroups;

//	PRINT("Doing children\n");
	for (int i = 0; i < depth; i++) {
//	/	PRINT("\t");
	}
//	PRINT("%i I parts_size = %i group_cnt = %i\n", depth, parts.size(), group_cnts.size());
	for (auto i = group_cnts.begin(); i != group_cnts.end(); i++) {
		vector<rockstar_particle> these_parts;
		int j = 0;
		while (j < parts.size()) {
			if (parts[j].subgroup == i->first) {
				these_parts.push_back(parts[j]);
				parts[j] = parts.back();
				parts.pop_back();
			} else {
				j++;
			}
		}
		auto these_groups = rockstar_seeds(these_parts, next_id, rfac, vfac, scale, depth + 1);
		subgroups.insert(subgroups.end(), these_groups.begin(), these_groups.end());
	}

	const auto find_sigmas = [rfac,vfac,scale](subgroup& sg) {
		float sigma2_v = 0.0;
		float sigma2_x = 0.0;
		array<float,NDIM> X;
		vector<rockstar_particle>& these_parts = sg.parts;
		for (int dim = 0; dim < NDIM; dim++) {
			X[dim] = 0.0;
		}
		for( int j = 0; j < these_parts.size(); j++) {
			for (int dim = 0; dim < NDIM; dim++) {
				X[dim] += these_parts[j].X[dim];
			}
		}
		for (int dim = 0; dim < NDIM; dim++) {
			X[dim] /= these_parts.size();
		}
		for (int j = 0; j < these_parts.size(); j++) {
			const auto& p = these_parts[j];
			const float dx = p.x - X[XDIM];
			const float dy = p.y - X[YDIM];
			const float dz = p.z - X[ZDIM];
			const float dx2 = sqr(dx, dy, dz);
			sigma2_x += dx2;
		}
		sigma2_x /= these_parts.size();
		float this_xfactor = sqrt(sigma2_x / these_parts.size());
		if( this_xfactor < sg.min_xfactor) {
			sg.min_xfactor = this_xfactor;
			for( int dim = 0; dim < NDIM; dim++) {
				sg.X[dim] = X[dim];
			}
		}
		sigma2_x = 0.0;
		vector<double> radii;
		for (int j = 0; j < these_parts.size(); j++) {
			const auto& p = these_parts[j];
			const double dx = p.x - sg.x;
			const double dy = p.y - sg.y;
			const double dz = p.z - sg.z;
			const double dx2 = sqr(dx, dy, dz);
			sigma2_x += dx2;
			radii.push_back(sqrt(dx2));
		}
		std::sort(radii.begin(), radii.end());
		double mass = 0.0;
		double volume = 0.0;
		double r_vir;
		const double a = scale;
		const double Om = get_options().omega_m;
		const double Or = get_options().omega_r;
		const double Ol = 1.0 - Om - Or;
		double H = get_options().hubble * constants::H0 * sqrt((Or/(a*a*a*a) +Om/(a*a*a)+Ol));
		const double density_crit = 200.0 * 3.0 * H * H / constants::G / (8.0*M_PI) * a*a*a;
		double rmax = radii.back();
		const auto part_mass = get_options().code_to_g;
		for( int i = 0; i < radii.size(); i++) {
			mass += part_mass;
			const double r = radii[i] * get_options().code_to_cm / rfac;
			volume = std::pow(r,3) * (4.0/3.0*M_PI);
			double density = mass / volume;
			if( density > density_crit || i == 0) {
				r_vir = radii[i];
			}
		}
		if( r_vir == radii.back()) {
			r_vir = std::pow(mass / (4.0*M_PI/3.0*density_crit),1.0/3.0) / get_options().code_to_cm * rfac;
		}
		sigma2_x /= these_parts.size();
		sg.sigma2_x = sigma2_x;
		sg.r_vir = r_vir;
//		PRINT( "%e %e\n", r_vir, rmax);
			int n;
			float rfrac = 0.1;
			do {
				n = 0;
				sg.vx = 0.0;
				sg.vy = 0.0;
				sg.vz = 0.0;
				for (int j = 0; j < these_parts.size(); j++) {
					const auto& p = these_parts[j];
					const float dx = p.x - sg.x;
					const float dy = p.y - sg.y;
					const float dz = p.z - sg.z;
					const float dx2 = sqr(dx, dy, dz);
					if( dx2 < sqr(rfrac * sg.r_vir) ) {
						sg.vx += these_parts[j].vx;
						sg.vy += these_parts[j].vy;
						sg.vz += these_parts[j].vz;
						n++;
					}
				}
				if( n ) {
					sg.vx /= n;
					sg.vy /= n;
					sg.vz /= n;
				}
				rfrac *= 1.1;
			}while( n < ROCKSTAR_MIN_GROUP );
//		PRINT( "n = %i\n", n);
			for (int j = 0; j < these_parts.size(); j++) {
				const auto& p = these_parts[j];
				const float dvx = p.vx - sg.vx;
				const float dvy = p.vy - sg.vy;
				const float dvz = p.vz - sg.vz;
				const float dv2 = sqr(dvx, dvy, dvz);
				sigma2_v += dv2;
			}
			sigma2_v /= these_parts.size();
			sg.sigma2_v = sigma2_v;
			float vcirc_max = 0.0;
			H = get_options().hubble * constants::H0 * get_options().code_to_s;
			const float nparts = pow(get_options().parts_dim,3);
			for( int n = 0; n < radii.size(); n++) {
				float vcirc = sqrt(3.0 * get_options().omega_m * sqr(H) * n / nparts / radii[n]);
				vcirc_max = std::max(vcirc_max,vcirc);
			}
			sg.vcirc_max = vcirc_max * vfac;
			sg.r_dyn = vcirc_max / H / sqrt(180) * rfac;
		};
	if (subgroups.size() == 0) {
		const int subgrp = next_id++;
		for (int i = 0; i < parts.size(); i++) {
			parts[i].subgroup = subgrp;
		}
		subgroup sg;
		sg.parts = std::move(parts);
		sg.id = subgrp;
		find_sigmas(sg);
		subgroups.push_back(sg);
	} else if (subgroups.size() == 1) {
		const int subgrp = subgroups[0].parts[0].subgroup;
		for (int i = 0; i < parts.size(); i++) {
			parts[i].subgroup = subgrp;
			subgroups[0].parts.push_back(parts[i]);
		}
		find_sigmas(subgroups[0]);
	} else {
		bool found_merge;
		PRINT("Finding merges for %i subgroups\n", subgroups.size());
		for (int k = 0; k < subgroups.size();) {
			found_merge = false;
			std::sort(subgroups.begin(), subgroups.end(), [](const subgroup& a, const subgroup& b) {
				return a.parts.size() < b.parts.size();
			});
			for (int l = k + 1; l < subgroups.size(); l++) {
				const float sigma_x_inv = sqrt(subgroups[k].parts.size() / subgroups[k].sigma2_x);
				const float sigma_v_inv = sqrt(subgroups[k].parts.size() / subgroups[k].sigma2_v);
				const float dx = (subgroups[k].x - subgroups[l].x) * sigma_x_inv;
				const float dy = (subgroups[k].y - subgroups[l].y) * sigma_x_inv;
				const float dz = (subgroups[k].z - subgroups[l].z) * sigma_x_inv;
				const float dvx = (subgroups[k].vx - subgroups[l].vx) * sigma_v_inv;
				const float dvy = (subgroups[k].vy - subgroups[l].vy) * sigma_v_inv;
				const float dvz = (subgroups[k].vz - subgroups[l].vz) * sigma_v_inv;
//						PRINT("%e %i\n", sqr(dx, dy, dz) + sqr(dvx, dvy, dvz), subgroups[k].parts.size());
				if (sqr(dx, dy, dz) + sqr(dvx, dvy, dvz) < 200.0) {
					for (int i = 0; i < subgroups[k].parts.size(); i++) {
						subgroups[l].parts.push_back(subgroups[k].parts[i]);
						subgroups[l].parts.back().subgroup = subgroups[l].parts[0].subgroup;
					}
					subgroups[l].min_xfactor = std::numeric_limits<float>::max();
					find_sigmas(subgroups[l]);
					subgroups[k] = subgroups.back();
					subgroups.pop_back();
					found_merge = true;
//							PRINT("Merging\n");
					break;
				}
			}
			if (!found_merge) {
				k++;
			}
		}
		for (int i = 0; i < subgroups.size(); i++) {
			PRINT("rdyn %e sigma_x %e\n", subgroups[i].r_dyn, sqrt(subgroups[i].sigma2_x));
		}
		for (int j = 0; j < parts.size(); j++) {

			float min_dist = std::numeric_limits<float>::max();
			int min_index = -1;
			for (int i = 0; i < subgroups.size(); i++) {
				assert(subgroups[i].parts.size());
				const float rdyn_inv = 1.0 / subgroups[i].r_dyn;
				const float sigma_v_inv = 1.0 / sqrt(subgroups[i].sigma2_v);
				const float dx = (parts[j].x - subgroups[i].x) * rdyn_inv;
				const float dy = (parts[j].y - subgroups[i].y) * rdyn_inv;
				const float dz = (parts[j].z - subgroups[i].z) * rdyn_inv;
				const float dvx = (parts[j].vx - subgroups[i].vx) * sigma_v_inv;
				const float dvy = (parts[j].vy - subgroups[i].vy) * sigma_v_inv;
				const float dvz = (parts[j].vz - subgroups[i].vz) * sigma_v_inv;
				const float dist = sqrt(sqr(dx, dy, dz) + sqr(dvx, dvy, dvz));
				//PRINT("dist = %e mindist = %e %e %e %e %e %e %e \n", dist, min_dist, dx, dy, dz, dvz, dvy, subgroups[i].sigma2_v);
				if (dist < min_dist) {
					min_dist = dist;
					min_index = i;
				}
			}
			parts[j].subgroup = subgroups[min_index].parts[0].subgroup;
			subgroups[min_index].parts.push_back(parts[j]);
		}
		for (int k = 0; k < subgroups.size(); k++) {
			find_sigmas(subgroups[k]);
		}
	}
	for (int i = 0; i < depth; i++) {
//		PRINT("\t");
	}
	for (int k = 0; k < subgroups.size(); k++) {
		auto& parts = subgroups[k].parts;
		auto& sg = subgroups[k];
		sg.x *= sigma_x;
		sg.y *= sigma_x;
		sg.z *= sigma_x;
		sg.vx *= sigma_v;
		sg.vy *= sigma_v;
		sg.vz *= sigma_v;
		sg.x += avg_x;
		sg.y += avg_y;
		sg.z += avg_z;
		sg.vx += avg_vx;
		sg.vy += avg_vy;
		sg.vz += avg_vz;
		sg.r_dyn *= sigma_x;
		sg.sigma2_v *= sigma_v * sigma_v;
		sg.sigma2_x *= sigma_x * sigma_x;
		sg.vcirc_max *= sigma_v;
		sg.min_xfactor *= sigma_x;
		for (int i = 0; i < parts.size(); i++) {
			parts[i].x *= sigma_x;
			parts[i].y *= sigma_x;
			parts[i].z *= sigma_x;
			parts[i].vx *= sigma_v;
			parts[i].vy *= sigma_v;
			parts[i].vz *= sigma_v;
		}
		for (int i = 0; i < parts.size(); i++) {
			parts[i].x += avg_x;
			parts[i].y += avg_y;
			parts[i].z += avg_z;
			parts[i].vx += avg_vx;
			parts[i].vy += avg_vy;
			parts[i].vz += avg_vz;
		}
	}
	if (depth == 0) {
		for (int k = 0; k < subgroups.size(); k++) {
			auto& sgA = subgroups[k];
			float min_dist = std::numeric_limits<float>::max();
			int min_index = -1;
			for (int l = 0; l < subgroups.size(); l++) {
				auto& sgB = subgroups[l];
				if (sgA.parts.size() < sgB.parts.size()) {
					const float dx = sgA.x - sgB.x;
					const float dy = sgA.y - sgB.y;
					const float dz = sgA.z - sgB.z;
					const float r = sqrt(sqr(dx, dy, dz));
					if (r < sgB.r_vir) {
			//			if (rockstar_halo_bound(sgB, sgA, scale)) {
							PRINT("Testing subhalo\n");
							const float rdyn_inv = 1.0 / sgB.r_dyn;
							const float dvx = sgA.vx - sgB.vx;
							const float dvy = sgA.vy - sgB.vy;
							const float dvz = sgA.vz - sgB.vz;
							const float sigma2_v_inv = 1.0 / sqrt(sgB.sigma2_v);
							const float dist = sqrt(sqr(dvx, dvy, dvz) * sigma2_v_inv + sqr(r * rdyn_inv));
							if (dist < min_dist) {
								min_dist = dist;
								min_index = l;
							}
				//		}
					}
				}
			}
			if (min_index != -1) {
				auto& sgB = subgroups[min_index];
				sgA.parent = sgB.id;
				sgB.children.push_back(sgA.id);
			}
		}
		vector<int> checklist;
		for (int k = 0; k < subgroups.size(); k++) {
			if (subgroups[k].parent == ROCKSTAR_NO_GROUP) {
				checklist.push_back(k);
			}
		}
		PRINT("CHECLIST.SIZE = %i\n", checklist.size());
		auto all_parts = rockstar_unbind(subgroups, checklist, scale, 0);
		PRINT("%i\n", all_parts.size());
		int k = 0;
		while (k < subgroups.size()) {
			if (subgroups[k].parts.size() == 0) {
				for (int l = 0; l < subgroups.size(); l++) {
					for (int m = 0; m < subgroups[l].children.size(); m++) {
						if (subgroups[k].id == subgroups[l].children[m]) {
							subgroups[l].children[m] = subgroups[l].children.back();
							subgroups[l].children.pop_back();
						}
					}
				}
				subgroups[k] = std::move(subgroups.back());
				subgroups.pop_back();
			} else {
				k++;
			}
		}
		bool change;
		do {
			change = false;
			for (int k = 0; k < subgroups.size(); k++) {
				if (subgroups[k].parent != ROCKSTAR_NO_GROUP && subgroups[k].children.size()) {
					int l;
					for (l = 0; l < subgroups.size(); l++) {
						if (subgroups[k].parent == subgroups[l].id) {
							break;
						}
					}
					change = true;
					subgroups[l].children.insert(subgroups[l].children.end(), subgroups[k].children.begin(), subgroups[k].children.end());
					subgroups[k].children.clear();
				}
			}
		} while (change);
		for (int k = 0; k < subgroups.size(); k++) {
			rockstar_subgroup_statistics(subgroups[k]);
		}
		subgroup unbound;
		unbound.id = ROCKSTAR_NO_GROUP;
		auto& parts = all_parts;
		int l = 0;
		while (l < parts.size()) {
			if (parts[l].subgroup == ROCKSTAR_NO_GROUP) {
				unbound.parts.push_back(parts[l]);
				parts[l] = parts.back();
				parts.pop_back();
			} else {
				l++;
			}
		}
		subgroups.push_back(std::move(unbound));
		//	PRINT("ALLPARTSSIZE = %i\n", all_parts.size());
	}
//	PRINT("%i II parts_size = %i group_cnt = %i\n", depth, parts.size(), subgroups.size());
	for (int k = 0; k < subgroups.size(); k++) {
//		PRINT("%i %i %i %i %i\n", subgroups[k].id, subgroups[k].parts.size(), subgroups[k].depth, subgroups[k].parent != ROCKSTAR_NO_GROUP,
//				subgroups[k].children.size());
	}
	return subgroups;
}

void rockstar_find_subgroups(vector<rockstar_particle>& parts, float scale) {
	int next_id = 1;
	rockstar_seeds(parts, next_id, 1.0, 1.0, scale);
}

vector<subgroup> rockstar_find_subgroups(const vector<particle_data>& parts, float scale) {
	vector<rockstar_particle> rock_parts;
	array<fixed32, NDIM> x0;
	for (int dim = 0; dim < NDIM; dim++) {
		x0[dim] = parts[0].x[dim];
	}
	for (int i = 0; i < parts.size(); i++) {
		array<float, NDIM> x;
		for (int dim = 0; dim < NDIM; dim++) {
			x[dim] = distance(parts[i].x[dim], x0[dim]);
		}
		rockstar_particle part;
		part.x = x[XDIM];
		part.y = x[YDIM];
		part.z = x[ZDIM];
		part.vx = parts[i].v[XDIM];
		part.vy = parts[i].v[YDIM];
		part.vz = parts[i].v[ZDIM];
		part.index = parts[i].index;
		rock_parts.push_back(part);
	}
	int next_id = 1;
	return rockstar_seeds(rock_parts, next_id, 1.0, 1.0, scale);
}
