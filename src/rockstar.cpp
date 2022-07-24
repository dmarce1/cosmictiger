#include <cosmictiger/bh.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/rockstar.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/timer.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/lightcone.hpp>
#include <cosmictiger/device_vector.hpp>
#include <cosmictiger/cosmology.hpp>

void rockstar_find_subgroups(device_vector<rockstar_particle>& parts, float scale = 1.0);

void rockstar_assign_link_len(const device_vector<rockstar_tree>& trees, device_vector<rockstar_particle>& parts, int self_id, vector<int> checklist,
		float link_len, bool phase) {
	static thread_local vector<int> nextlist;
	static thread_local vector<int> leaflist;
	const int dimmax = phase ? 2 * NDIM : NDIM;
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
						float R2 = 0.f;
						for (int dim = 0; dim < dimmax; dim++) {
							R2 += sqr(mypart.X[dim] - part.X[dim]);
						}
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
			rockstar_assign_link_len(trees, parts, self.children[LEFT], checklist, link_len, phase);
			rockstar_assign_link_len(trees, parts, self.children[RIGHT], std::move(checklist), link_len, phase);
		}
	}
}

void rockstar_find_neighbors(device_vector<int>& leaves, device_vector<rockstar_tree>& trees, int self_id, vector<int> checklist, float link_len) {
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
		self.neighbors.resize(leaflist.size());
		for (int i = 0; i < leaflist.size(); i++) {
			self.neighbors[i] = leaflist[i];
		}
		leaves.push_back(self_id);
	} else {
		checklist.insert(checklist.end(), leaflist.begin(), leaflist.end());
		if (checklist.size()) {
			rockstar_find_neighbors(leaves, trees, self.children[LEFT], checklist, link_len);
			rockstar_find_neighbors(leaves, trees, self.children[RIGHT], std::move(checklist), link_len);
		}
	}
}

int rockstar_particles_sort(device_vector<rockstar_particle>& parts, int begin, int end, float xmid, int xdim) {
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

int rockstar_form_tree(device_vector<rockstar_particle>& parts, device_vector<rockstar_tree>& trees, range<float, 2 * NDIM>& rng, int part_begin,
		int part_end) {
	array<int, NCHILD> children;
	rockstar_tree node;
	int active_count = 0;
	const int bucket_size = ROCKSTAR_BUCKET_SIZE;
	if (part_end - part_begin <= bucket_size) {
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

int rockstar_form_tree(device_vector<rockstar_tree>& trees, device_vector<rockstar_particle>& parts) {
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

float rockstar_find_link_len(device_vector<rockstar_tree>& trees, device_vector<rockstar_particle>& parts, int tree_root, float ff, bool gpu) {
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
		gpu = parts.size() > 4 * 1024;
		if (gpu) {
			device_vector<int> leaves;
			rockstar_find_neighbors(leaves, trees, trees.size() - 1, vector<int>(1, trees.size() - 1), max_link_len);
			rockstar_assign_linklen_cuda(trees, leaves, parts, max_link_len, true);
		} else {
			rockstar_assign_link_len(trees, parts, tree_root, vector<int>(1, tree_root), max_link_len, true);
		}
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

void rockstar_find_all_link_lens(device_vector<rockstar_tree>& trees, device_vector<rockstar_particle>& parts, int tree_root) {
	for (auto& p : parts) {
		p.min_dist2 = std::numeric_limits<float>::max();
	}
	double max_link_len = get_options().lc_b / get_options().parts_dim;
	bool gpu = parts.size() > 4 * 1024;
	if (gpu) {
		device_vector<int> leaves;
		rockstar_find_neighbors(leaves, trees, trees.size() - 1, vector<int>(1, trees.size() - 1), max_link_len);
		rockstar_assign_linklen_cuda(trees, leaves, parts, max_link_len, false);
	} else {
		rockstar_assign_link_len(trees, parts, tree_root, vector<int>(1, tree_root), max_link_len, false);
	}
}

size_t rockstar_find_subgroups(device_vector<rockstar_tree>& trees, device_vector<rockstar_particle>& parts, int self_id, vector<int> checklist, float link_len,
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

void rockstar_find_subgroups(device_vector<rockstar_tree>& trees, device_vector<rockstar_particle>& parts, float link_len, int& next_id, bool gpu) {
	timer tm;
	tm.start();
	for (int i = 0; i < parts.size(); i++) {
		parts[i].subgroup = ROCKSTAR_NO_GROUP;
	}
	for (int i = 0; i < trees.size(); i++) {
		trees[i].active = true;
	}
	int cnt;
	gpu = parts.size() > 4 * 1024;
	if (gpu) {
		device_vector<int> leaves;
		rockstar_find_neighbors(leaves, trees, trees.size() - 1, vector<int>(1, trees.size() - 1), link_len);
		rockstar_find_subgroups_cuda(trees, leaves, parts, link_len, next_id);
	} else {
		do {
			double pct_active = 0.0;
			for (int i = 0; i < trees.size(); i++) {
				trees[i].last_active = trees[i].active;
				pct_active += int(trees[i].active) / (double) trees.size();
			}
			int root_id = trees.size() - 1;
			cnt = rockstar_find_subgroups(trees, parts, root_id, vector<int>(1, root_id), link_len, next_id);
//		PRINT("%i %i %e\n", root_id + 1, parts.size(), pct_active);
		} while (cnt != 0);
		tm.stop();
	}
//	PRINT( "find_subgroups = %e\n", tm.read());
}

struct number {
	int n;
	number() {
		n = 0;
	}
};

std::unordered_map<int, number> rockstar_subgroup_cnts(device_vector<rockstar_particle>& parts) {
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

bool rockstar_halo_bound_to(const subgroup& subhalo, const subgroup& halo, double scale) {
	device_vector<array<float, NDIM>> X;
	device_vector<array<float, NDIM>> Y;
	for (int i = 0; i < subhalo.parts.size(); i++) {
		array<float, NDIM> x;
		for (int dim = 0; dim < NDIM; dim++) {
			x[dim] = subhalo.parts[i].X[dim];
		}
		Y.push_back(x);
	}
	for (int i = 0; i < halo.parts.size(); i++) {
		array<float, NDIM> x;
		for (int dim = 0; dim < NDIM; dim++) {
			x[dim] = halo.parts[i].X[dim];
		}
		X.push_back(x);
	}
	const auto phi = bh_evaluate_points(Y, X, X.size() > 4 * 1024 || Y.size() > 4 * 1024);
	double etot = 0.0;
	for (int i = 0; i < subhalo.parts.size(); i++) {
		const float dvx = subhalo.parts[i].vx - halo.vxb;
		const float dvy = subhalo.parts[i].vy - halo.vyb;
		const float dvz = subhalo.parts[i].vz - halo.vzb;
		const float ekin = 0.5 * sqr(dvx, dvy, dvz);
		etot += ekin + phi[i] / scale;
	}
	return etot < 0.0;
}

bool rockstar_halo_bound(subgroup& halo, vector<rockstar_particle>& unbound_parts, float scale) {
	device_vector<array<float, NDIM>> x;
	for (int i = 0; i < halo.parts.size(); i++) {
		array<float, NDIM> X;
		for (int dim = 0; dim < NDIM; dim++) {
			X[dim] = halo.parts[i].X[dim];
		}
		x.push_back(X);
	}
	auto y = x;
	auto phi = bh_evaluate_potential(x, x.size() > 4 * 1024);
	auto phi_tot = 0.0;
	auto kin_tot = 0.0;
	int nunbound = 0;
	int onparts = halo.parts.size();
	for (int i = 0; i < halo.parts.size(); i++) {
		const float dvx = halo.parts[i].vx - halo.vxb;
		const float dvy = halo.parts[i].vy - halo.vyb;
		const float dvz = halo.parts[i].vz - halo.vzb;
		const float ekin = 0.5 * sqr(dvx, dvy, dvz);
		kin_tot += ekin;
		phi_tot += phi[i] / scale;
		if (ekin + phi[i] / scale > 0.0) {
			nunbound++;
			unbound_parts.push_back(halo.parts[i]);
			halo.parts[i] = halo.parts.back();
			halo.parts.pop_back();
			i--;
		}
	}
	halo.T = kin_tot;
	halo.W = 0.5 * phi_tot;
	//PRINT( "%e %e\n", halo.T, halo.W );
	const bool rc = (nunbound * 2 < onparts) && (onparts - nunbound) >= ROCKSTAR_MIN_GROUP;
	if (!rc) {
		for (int i = 0; i < halo.parts.size(); i++) {
			unbound_parts.push_back(halo.parts[i]);
		}
		halo.parts = decltype(halo.parts)();
	}
	//if( !rc ) {
//		PRINT( "%e\n", halo.T / (halo.W));
//	}
	return rc;
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

void rockstar_subgroup_statistics(subgroup& sg, bool gpu) {
	device_vector<rockstar_tree> trees;
	device_vector<rockstar_particle> parts;
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
		const int root_id = rockstar_form_tree(trees, parts);

		const float link_len = rockstar_find_link_len(trees, parts, root_id, ROCKSTAR_FF, gpu);

	}
	sg.host_part_cnt = parts.size();
}

vector<rockstar_particle> rockstar_gather_halo_parts(const vector<subgroup>& subgroups, const subgroup& sg) {
	vector<rockstar_particle> X;
	for (int i = 0; i < sg.parts.size(); i++) {
		X.push_back(sg.parts[i]);
	}
	for (int i = 0; i < sg.children.size(); i++) {
		const auto tmp = rockstar_gather_halo_parts(subgroups, subgroups[sg.children[i]]);
		X.insert(X.end(), tmp.begin(), tmp.end());
	}
	return X;
}

vector<subgroup> rockstar_seeds(device_vector<rockstar_particle> parts, int& next_id, float rfac, float vfac, float scale, bool gpu, int depth = 0) {
	vector<subgroup> subgroups;

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
	device_vector<rockstar_tree> trees;
	const int root_id = rockstar_form_tree(trees, parts);
	const float link_len = rockstar_find_link_len(trees, parts, root_id, ROCKSTAR_FF, gpu);
	rockstar_find_subgroups(trees, parts, link_len, next_id, gpu);
	auto group_cnts = rockstar_subgroup_cnts(parts);

	for (auto i = group_cnts.begin(); i != group_cnts.end(); i++) {
		device_vector<rockstar_particle> these_parts;
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
		auto these_groups = rockstar_seeds(these_parts, next_id, rfac, vfac, scale, gpu, depth + 1);
		subgroups.insert(subgroups.end(), these_groups.begin(), these_groups.end());
	}

	const auto find_sigmas = [rfac,vfac,scale](subgroup& sg) {
		float sigma2_v = 0.0;
		float sigma2_x = 0.0;
		array<float,2 * NDIM> X;
		const auto Ndim = get_options().parts_dim;
		device_vector<rockstar_particle>& these_parts = sg.parts;
		for (int dim = 0; dim < 2 * NDIM; dim++) {
			X[dim] = 0.0;
		}
		for( int j = 0; j < these_parts.size(); j++) {
			for (int dim = 0; dim < 2 * NDIM; dim++) {
				X[dim] += these_parts[j].X[dim];
			}
		}
		for (int dim = 0; dim < 2 * NDIM; dim++) {
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
		double sigx_norm = sqrt(sigma2_x / these_parts.size());
		if( sigx_norm < sg.sigma_x_min) {
			sg.sigma_x_min = sigx_norm;
			sg.x = X[XDIM];
			sg.y = X[YDIM];
			sg.z = X[ZDIM];
		}
		sg.vxb = X[NDIM + XDIM];
		sg.vyb = X[NDIM + YDIM];
		sg.vzb = X[NDIM + ZDIM];
		sigma2_x = 0.0;
		struct radial_part {
			double r;
			float vx;
			float vy;
			float vz;
		};
		vector<radial_part> radii;
		for (int j = 0; j < these_parts.size(); j++) {
			const auto& p = these_parts[j];
			const double dx = p.x - sg.x;
			const double dy = p.y - sg.y;
			const double dz = p.z - sg.z;
			const double dx2 = sqr(dx, dy, dz);
			sigma2_x += dx2;
			radial_part p0;
			p0.r = sqrt(dx2);
			p0.vx = p.vx;
			p0.vy = p.vy;
			p0.vz = p.vz;
			radii.push_back(p0);
		}
		std::sort(radii.begin(), radii.end(), [](radial_part a, radial_part b) {
					return a.r < b.r;
				});
		double mass = 0.0;
		double volume = 0.0;
		const double omega_m = get_options().omega_m;
		const double x = ((omega_m / (sqr(scale)*scale)) / (omega_m / (sqr(scale)*scale)) + (1.0 - omega_m)) - 1.0;
		double y = (18.0 * M_PI * M_PI + 82.0 * x - 39.0 * sqr(x))/(1.0+x);
		double r_vir = pow(these_parts.size() / (4.0/3.0*M_PI*y), 1.0/3.0) / Ndim;

		sigma2_x /= these_parts.size();
		sg.sigma2_x = sigma2_x;
		sg.r_vir = r_vir;
		int n;
		float rfrac = 0.1;
		n = 0;
		sg.vxc = 0.0;
		sg.vyc = 0.0;
		sg.vzc = 0.0;
		int rmax = std::max(ROCKSTAR_MIN_GROUP, (int) ((radii.size() + 0.5) / 10));
		for (int j = 0; j < rmax; j++) {
			sg.vxc += radii[j].vx;
			sg.vyc += radii[j].vy;
			sg.vzc += radii[j].vz;
		}
		sg.vxc /= rmax;
		sg.vyc /= rmax;
		sg.vzc /= rmax;
		for (int j = 0; j < these_parts.size(); j++) {
			const auto& p = these_parts[j];
			const float dvx = p.vx - sg.vxc;
			const float dvy = p.vy - sg.vyc;
			const float dvz = p.vz - sg.vzc;
			const float dv2 = sqr(dvx, dvy, dvz);
			sigma2_v += dv2;
		}
		sigma2_v /= these_parts.size();
		sg.sigma2_v = sigma2_v;
		float vcirc_max = 0.0;
		double H = get_options().hubble * constants::H0 * get_options().code_to_s;
		const float nparts = pow(get_options().parts_dim,3);
		for( int n = 1; n < radii.size(); n++) {
			float vcirc = sqrt(3.0 * get_options().omega_m * sqr(H) * n / (8.0 * M_PI * nparts * radii[n].r / rfac));
			vcirc_max = std::max(vcirc_max,vcirc);
		}
		sg.vcirc_max = vcirc_max * vfac;
		y *= (1.0+x);
		sg.r_dyn = sqrt(2.0)* vcirc_max / H / sqrt(y) * rfac;
	};
	if (subgroups.size() == 0) {
		const int subgrp = next_id++;
		for (int i = 0; i < parts.size(); i++) {
			parts[i].subgroup = subgrp;
		}
		subgroup sg;
		sg.parts = std::move(parts);
		sg.id = subgrp;
		sg.sigma_x_min = std::numeric_limits<float>::max();
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
		//PRINT("Finding merges for %i subgroups\n", subgroups.size());
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
				const float dvx = (subgroups[k].vxc - subgroups[l].vxc) * sigma_v_inv;
				const float dvy = (subgroups[k].vyc - subgroups[l].vyc) * sigma_v_inv;
				const float dvz = (subgroups[k].vzc - subgroups[l].vzc) * sigma_v_inv;
				if (sqr(dx, dy, dz) + sqr(dvx, dvy, dvz) < 200.0) {
					for (int i = 0; i < subgroups[k].parts.size(); i++) {
						subgroups[l].parts.push_back(subgroups[k].parts[i]);
						subgroups[l].parts.back().subgroup = subgroups[l].parts[0].subgroup;
					}
					find_sigmas(subgroups[l]);
					subgroups[k] = subgroups.back();
					subgroups.pop_back();
					found_merge = true;
					break;
				}
			}
			if (!found_merge) {
				k++;
			}
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
				const float dvx = (parts[j].vx - subgroups[i].vxc) * sigma_v_inv;
				const float dvy = (parts[j].vy - subgroups[i].vyc) * sigma_v_inv;
				const float dvz = (parts[j].vz - subgroups[i].vzc) * sigma_v_inv;
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
		sg.vxc *= sigma_v;
		sg.vyc *= sigma_v;
		sg.vzc *= sigma_v;
		sg.vxb *= sigma_v;
		sg.vyb *= sigma_v;
		sg.vzb *= sigma_v;
		sg.x += avg_x;
		sg.y += avg_y;
		sg.z += avg_z;
		sg.vxb += avg_vx;
		sg.vyb += avg_vy;
		sg.vzb += avg_vz;
		sg.vxc += avg_vx;
		sg.vyc += avg_vy;
		sg.vzc += avg_vz;
		sg.r_dyn *= sigma_x;
		sg.sigma2_v *= sigma_v * sigma_v;
		sg.sigma2_x *= sigma_x * sigma_x;
		sg.vcirc_max *= sigma_v;
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

		const std::function<vector<rockstar_particle>(subgroup&)> unbind = [&unbind, &subgroups, scale](subgroup& grp) {
			vector<rockstar_particle> unbounds;
			for( int ci = 0; ci < grp.children.size(); ci++) {
				auto tmp = unbind(subgroups[grp.children[ci]]);
				for( const auto& p : tmp ) {
					grp.parts.push_back(p);
				}
			}
			rockstar_halo_bound(grp, unbounds, scale);
			return unbounds;
		};

		int bound = 0;
		int unbound = 0;
		int rebound = 0;
		vector<rockstar_particle> unbounds;
		for (int k = 0; k < subgroups.size(); k++) {
			const int psize = subgroups[k].parts.size();
			if (subgroups[k].parent == -1) {
				auto tmp = unbind(subgroups[k]);
				unbounds.insert(unbounds.end(), tmp.begin(), tmp.end());
			}
			bound += psize;
		}
		int k = 0;
		while (k < subgroups.size()) {
			if (subgroups[k].parts.size() == 0) {
				subgroups[k] = subgroups.back();
				subgroups.pop_back();
			} else {
				k++;
			}
		}

		unbound += unbounds.size();
		bound -= unbound;
		for (int k = 0; k < subgroups.size(); k++) {
			auto& sgA = subgroups[k];
			float max_mass = 0.0;
			int min_index = -1;
			for (int l = 0; l < subgroups.size(); l++) {
				auto& sgB = subgroups[l];
				if (sgA.parts.size() < sgB.parts.size()) {
					if (sgB.parts.size() > max_mass) {
						const float dx = sgB.x - sgA.x;
						const float dy = sgB.y - sgA.y;
						const float dz = sgB.z - sgA.z;
						const float r = sqrt(sqr(dx, dy, dz));
						if (r < sgB.r_vir) {
							max_mass = sgB.parts.size();
							min_index = l;
						}
					}
				}
			}
			if (min_index != -1) {
				auto& sgB = subgroups[min_index];
				sgA.parent = min_index;
				sgB.children.push_back(k);
			} else {
				sgA.parent = -1;
			}
		}
		int halo_cnt = 0;
		int subhalo_cnt = 0;
		for (int k = 0; k < subgroups.size(); k++) {
			if (subgroups[k].parent == -1) {
				halo_cnt++;
			} else {
				subhalo_cnt++;
			}
		}
		PRINT("%i %i\n", halo_cnt, subhalo_cnt);
	}
	return subgroups;
}

void rockstar_find_subgroups(device_vector<rockstar_particle>& parts, float scale) {
	int next_id = 1;
	rockstar_seeds(parts, next_id, 1.0, 1.0, scale, false);
}

vector<rockstar_record> rockstar_find_subgroups(const vector<lc_entry>& parts, double tau_max) {
	device_vector<rockstar_particle> rock_parts(parts.size());
	vector<rockstar_record> recs;
	double xc = 0.0;
	double yc = 0.0;
	double zc = 0.0;
	for (int i = 0; i < parts.size(); i++) {
		auto& pi = parts[i];
		xc += pi.x.to_double();
		yc += pi.y.to_double();
		zc += pi.z.to_double();
	}

	xc /= parts.size();
	yc /= parts.size();
	zc /= parts.size();
	const auto this_tau = sqrt(sqr(xc, yc, zc));
	const auto z0 = get_options().z0;
	const auto a0 = 1.0 / (z0 + 1.0);
//	double scale = cosmos_tau_to_scale(a0, tau_max - this_tau);
	double scale = 1.0;
//	PRINT( "SCALE = %e %e %e\n", a0, this_tau, scale);
	for (int i = 0; i < parts.size(); i++) {
		auto& pi = parts[i];
		auto& pj = rock_parts[i];
		pj.x = pi.x.to_double() - xc;
		pj.y = pi.y.to_double() - yc;
		pj.z = pi.z.to_double() - zc;
		pj.vx = pi.vx;
		pj.vy = pi.vy;
		pj.vz = pi.vz;
	}
	int next_id = 1;
	auto subgroups = rockstar_seeds(rock_parts, next_id, 1.0, 1.0, scale, false);
	if (subgroups.size()) {
		const std::function<device_vector<rockstar_particle>(const subgroup&)> gather_subparticles = [&gather_subparticles, &subgroups](const subgroup& grp) {
			device_vector<rockstar_particle> parts;
			for( int i = 0; i < grp.parts.size(); i++) {
				parts.push_back(grp.parts[i]);
			}
			for( int ci = 0; ci < grp.children.size(); ci++) {
				auto tmp = gather_subparticles(subgroups[grp.children[ci]]);
				for( int i = 0; i < tmp.size(); i++) {
					parts.push_back(tmp[i]);
				}
			}
			return parts;
		};
		for (int k = 0; k < subgroups.size(); k++) {
			for (int I = 0; I < 2; I++) {
				auto& grp = subgroups[k];
				device_vector<rockstar_particle>* parts_ptr;
				device_vector<rockstar_particle> halo_parts;
				if (I == 0) {
					parts_ptr = &grp.parts;
				} else {
					if (grp.children.size() == 0) {
						continue;
					}
					halo_parts = gather_subparticles(grp);
					parts_ptr = &halo_parts;
				}
				auto& parts = *parts_ptr;
				device_vector<rockstar_tree> trees;
				const int root_id = rockstar_form_tree(trees, parts);
				vector<rockstar_particle> parts500;
				rockstar_find_all_link_lens(trees, parts, root_id);
				const double ll200 = pow(200, -1.0 / 3.0) / get_options().parts_dim;
				const double ll500 = pow(500, -1.0 / 3.0) / get_options().parts_dim;
				const double ll2500 = pow(2500, -1.0 / 3.0) / get_options().parts_dim;
				double m200 = 0.0;
				double m500 = 0.0;
				double m2500 = 0.0;
				for (int i = 0; i < parts.size(); i++) {
					const double dist = sqrt(parts[i].min_dist2);
					if (dist < ll200) {
						m200 += 1.0;
						if (dist < ll500) {
							parts500.push_back(parts[i]);
							m500 += 1.0;
							if (dist < ll2500) {
								m2500 += 1.0;
							}
						}
					}
				}
				vector<double> radii;
				double Rs, rho0, mvir, rvir, RKlypin, Jx, Jy, Jz, J, Etot, lambda, lambda_B, delta_X, delta_V, ToW, rcom_max, rvmax;
				double x = grp.x;
				double y = grp.y;
				double z = grp.z;
				double vx = grp.vxc;
				double vy = grp.vyc;
				double vz = grp.vzc;
				double xcom = 0.0;
				double ycom = 0.0;
				double zcom = 0.0;
				double sigma_v = 0.0;
				double sigma_vb = 0.0;
				double sigma_x = 0.0;
				double vxb = 0.0;
				double vyb = 0.0;
				double vzb = 0.0;
				Jx = 0.0;
				Jy = 0.0;
				Jz = 0.0;
				Etot = grp.T + grp.W;
				for (int i = 0; i < parts.size(); i++) {
					const double dx = parts[i].x - x;
					const double dy = parts[i].y - y;
					const double dz = parts[i].z - z;
					sigma_x += sqr(dx, dy, dz);
					xcom += parts[i].x;
					ycom += parts[i].y;
					zcom += parts[i].z;
					const double dvx = parts[i].vx - vx;
					const double dvy = parts[i].vy - vy;
					const double dvz = parts[i].vz - vz;
					vxb += parts[i].vx;
					vyb += parts[i].vy;
					vzb += parts[i].vz;
					sigma_v += sqr(dvx, dvy, dvz);
					const double r = sqrt(sqr(dx, dy, dz));
					radii.push_back(r);
					Jx += dy * dvz - dz * dvy;
					Jy -= dx * dvz - dz * dvx;
					Jz += dz * dvy - dy * dvz;
				}
				vxb /= parts.size();
				vyb /= parts.size();
				vzb /= parts.size();
				for (int i = 0; i < parts.size(); i++) {
					const double dvx = parts[i].vx - vxb;
					const double dvy = parts[i].vy - vyb;
					const double dvz = parts[i].vz - vzb;
					sigma_vb += sqr(dvx, dvy, dvz);
				}
				sigma_x /= parts.size();
				sigma_v /= parts.size();
				sigma_vb /= parts.size();
				sigma_v = sqrt(sigma_v);
				sigma_x = sqrt(sigma_x);
				sigma_vb = sqrt(sigma_vb);
				xcom /= parts.size();
				ycom /= parts.size();
				zcom /= parts.size();
				array<array<double, NDIM>, NDIM> Ixx, Ixx500;
				for (int i = 0; i < NDIM; i++) {
					for (int j = 0; j < NDIM; j++) {
						Ixx[i][j] = 0.0;
						Ixx500[i][j] = 0.0;
					}
				}
				rcom_max = 0.0;
				for (int i = 0; i < parts.size(); i++) {
					const double dx = parts[i].x - xcom;
					const double dy = parts[i].y - ycom;
					const double dz = parts[i].z - zcom;
					rcom_max = std::max(rcom_max, sqrt(sqr(dx, dy, dz)));
					Ixx[XDIM][XDIM] += dx * dx;
					Ixx[XDIM][YDIM] += dx * dy;
					Ixx[XDIM][ZDIM] += dx * dz;
					Ixx[YDIM][XDIM] += dy * dx;
					Ixx[YDIM][YDIM] += dy * dy;
					Ixx[YDIM][ZDIM] += dy * dz;
					Ixx[ZDIM][XDIM] += dz * dx;
					Ixx[ZDIM][YDIM] += dz * dy;
					Ixx[ZDIM][ZDIM] += dz * dz;
				}
				double xcom500 = 0.0;
				double ycom500 = 0.0;
				double zcom500 = 0.0;
				for (int i = 0; i < parts500.size(); i++) {
					xcom500 += parts500[i].x;
					ycom500 += parts500[i].y;
					zcom500 += parts500[i].z;
				}
				xcom500 /= parts500.size();
				ycom500 /= parts500.size();
				zcom500 /= parts500.size();
				double rcommax500 = 0.0;
				for (int i = 0; i < parts500.size(); i++) {
					const double dx = parts500[i].x - xcom500;
					const double dy = parts500[i].y - ycom500;
					const double dz = parts500[i].z - zcom500;
					rcommax500 = std::max(rcommax500, sqrt(sqr(dx,dy,dz)));
					Ixx500[XDIM][XDIM] += dx * dx;
					Ixx500[XDIM][YDIM] += dx * dy;
					Ixx500[XDIM][ZDIM] += dx * dz;
					Ixx500[YDIM][XDIM] += dy * dx;
					Ixx500[YDIM][YDIM] += dy * dy;
					Ixx500[YDIM][ZDIM] += dy * dz;
					Ixx500[ZDIM][XDIM] += dz * dx;
					Ixx500[ZDIM][YDIM] += dz * dy;
					Ixx500[ZDIM][ZDIM] += dz * dz;
				}
				for (int i = 0; i < NDIM; i++) {
					for (int j = 0; j < NDIM; j++) {
						Ixx[i][j] /= parts.size();
						Ixx500[i][j] /= parts500.size();
					}
				}
				const auto lambda2 = find_eigenpair(Ixx, rcom_max);
				const auto lambda0 = find_eigenpair(Ixx, 0.0);
				const auto lambda1 = find_eigenpair(Ixx, 0.5 * (lambda2.first + lambda0.first));
				double b_o_a = sqrt(lambda1.first / lambda2.first);
				double c_o_a = sqrt(lambda0.first / lambda2.first);
				array<double, NDIM> A, A500;
				for (int dim = 0; dim < NDIM; dim++) {
					A[dim] = lambda2.second[dim] * sqrt(lambda2.first);
				}
				double b_o_a500, c_o_a500;
				if( parts500.size() >= ROCKSTAR_MIN_GROUP ) {
					const auto lambda2 = find_eigenpair(Ixx500, rcommax500);
					const auto lambda0 = find_eigenpair(Ixx500, 0.0);
					const auto lambda1 = find_eigenpair(Ixx500, 0.5 * (lambda2.first + lambda0.first));
					b_o_a500 = sqrt(lambda1.first / lambda2.first);
					c_o_a500 = sqrt(lambda0.first / lambda2.first);
					for (int dim = 0; dim < NDIM; dim++) {
						A500[dim] = lambda2.second[dim] * sqrt(lambda2.first);
					}
				} else {
					b_o_a500 = 0.0;
					c_o_a500 = 0.0;
					for (int dim = 0; dim < NDIM; dim++) {
						A500[dim] = 0.0;
					}
				}
				delta_X = sqrt(sqr(xcom - grp.x, ycom - grp.y, zcom - grp.z));
				delta_V = sqrt(sqr(grp.vxb - grp.vxc, grp.vyb - grp.vyc, grp.vzb - grp.vzc));
				J = sqrt(sqr(Jx, Jy, Jz));
				std::sort(radii.begin(), radii.end());
				cosmos_NFW_fit(radii, Rs, rho0);
				double vcirc_max = 0.0;
				for (int i = 0; i < radii.size(); i++) {
					double menc = i + 1.0;
					double vcirc = sqrt(get_options().GM / radii[i] * menc);
					if (vcirc > vcirc_max) {
						rvmax = radii[i];
						vcirc_max = vcirc;
					}
				}
				mvir = parts.size();
			//	PRINT("%e %e %e %e\n", mvir, m200, m500, m2500);
				double rmax = radii.back();
				const double omega_m = get_options().omega_m;
				x = ((omega_m / (sqr(scale) * scale)) / (omega_m / (sqr(scale) * scale)) + (1.0 - omega_m)) - 1.0;
				y = (18.0 * M_PI * M_PI + 82.0 * x - 39.0 * sqr(x)) / (1.0 + x);
				rvir = pow(mvir / (4.0 / 3.0 * M_PI * y), 1.0 / 3.0) / get_options().parts_dim;
				RKlypin = cosmos_Klypin_fit(vcirc_max, radii.back(), mvir);
				const double G = get_options().GM;
				lambda = J * sqrt(fabs(Etot)) / G / pow(mvir, 2.5);
				lambda_B = J / sqrt(2.0 * G * sqr(mvir) * mvir * rvir);
				ToW = grp.T / grp.W;
				const double conv_mass = get_options().code_to_g * get_options().hubble / constants::M0;
				const double conv_len = get_options().code_to_cm * get_options().hubble / constants::mpc_to_cm * 1000.0;
				const double conv_pos = get_options().code_to_cm * get_options().hubble / constants::mpc_to_cm;
				const double conv_vel = get_options().code_to_cm / get_options().code_to_s / 100000.0;
				const double conv_ang = conv_mass * conv_pos * conv_vel;
				const double conv_ene = conv_mass * sqr(conv_vel);
				x += xc;
				y += yc;
				z += zc;
				const auto this_tau = sqrt(sqr(x, y, z));
				const auto z0 = get_options().z0;
				const auto a0 = 1.0 / (z0 + 1.0);
				double a = cosmos_tau_to_scale(a0, tau_max - this_tau);
				double Z = 1.0 / a - 1.0;
				mvir *= conv_mass;
				m200 *= conv_mass;
				m500 *= conv_mass;
				m2500 *= conv_mass;
				rmax *= conv_len;
				rvir *= conv_len;
				rvmax *= conv_len;
				sigma_x *= conv_len;
				delta_X *= conv_len;
				Rs *= conv_len;
				RKlypin *= conv_len;
				sigma_v *= conv_vel;
				sigma_vb *= conv_vel;
				vx *= conv_vel;
				vy *= conv_vel;
				vz *= conv_vel;
				vxb *= conv_vel;
				vyb *= conv_vel;
				vzb *= conv_vel;
				delta_V *= conv_vel;
				vcirc_max *= conv_vel;
				Etot *= conv_ene;
				x = grp.x * conv_pos;
				y = grp.y * conv_pos;
				z = grp.z * conv_pos;
				Jx *= conv_ang;
				Jy *= conv_ang;
				Jz *= conv_ang;
				for (int dim = 0; dim < NDIM; dim++) {
					A[dim] *= conv_len;
				}
				rockstar_record rec;

				rec.id.halo_id = k;
				rec.id.sys = I;
				rec.x = x;
				rec.y = y;
				rec.z = z;
				rec.mvir = mvir;
				rec.E = Etot;
				rec.Jx = Jx;
				rec.Jy = Jy;
				rec.Jz = Jz;
				rec.pid = grp.parent;
				rec.m200 = m200;
				rec.m500 = m500;
				rec.m2500 = m2500;
				const double f16max = std::numeric_limits < float16 > ::max();
				if (Rs > f16max) {
					PRINT("mvir %i rmax = %e Rs = %e RKly = %e f16max = %e\n", parts.size(), rmax, Rs, RKlypin, f16max);
				}
				rec.vxc = double2float16(vx);
				rec.vyc = double2float16(vy);
				rec.vzc = double2float16(vz);
				rec.vxb = double2float16(vxb);
				rec.vyb = double2float16(vyb);
				rec.vzb = double2float16(vzb);
				rec.sig_vb = double2float16(sigma_vb);
				rec.rmax = double2float16(rmax);
				rec.rvir = double2float16(rvir);
				rec.rvmax = double2float16(rvmax);
				rec.rs = double2float16(Rs);
				rec.rkly = double2float16(RKlypin);
				rec.sig_x = double2float16(sigma_x);
				rec.delta_x = double2float16(delta_X);
				rec.ToW = double2float16(ToW);
				rec.sig_v = double2float16(sigma_v);
				rec.vcirc_max = double2float16(vcirc_max);
				rec.delta_v = double2float16(delta_V);
				rec.boa = double2float16(b_o_a);
				rec.coa = double2float16(c_o_a);
				rec.Ax = double2float16(A[XDIM]);
				rec.Ay = double2float16(A[YDIM]);
				rec.Az = double2float16(A[ZDIM]);
				rec.boa500 = double2float16(b_o_a500);
				rec.coa500 = double2float16(c_o_a500);
				rec.Ax500 = double2float16(A500[XDIM]);
				rec.Ay500 = double2float16(A500[YDIM]);
				rec.Az500 = double2float16(A500[ZDIM]);
				rec.lambda = double2float16(lambda);
				rec.lambda_B = double2float16(lambda_B);
				rec.Z = double2float16(Z);
				recs.push_back(rec);
			}
		}
	}
	return recs;
}

vector<subgroup> rockstar_find_subgroups(const vector<particle_data>& parts, float scale) {
	device_vector<rockstar_particle> rock_parts;
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
	return rockstar_seeds(rock_parts, next_id, 1.0, 1.0, scale, false);
}
