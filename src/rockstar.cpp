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

void rockstar_find_subgroups(device_vector<rockstar_particle>& parts, float scale = 1.0);

void rockstar_assign_link_len(const device_vector<rockstar_tree>& trees, device_vector<rockstar_particle>& parts, int self_id, vector<int> checklist,
		float link_len) {
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

int rockstar_form_tree(device_vector<rockstar_particle>& parts, device_vector<rockstar_tree>& trees, range<float, 2 * NDIM>& rng, int part_begin, int part_end,
		bool gpu) {
	array<int, NCHILD> children;
	rockstar_tree node;
	int active_count = 0;
	const int bucket_size = gpu ? ROCKSTAR_GPU_BUCKET_SIZE : ROCKSTAR_CPU_BUCKET_SIZE;
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
		children[LEFT] = rockstar_form_tree(parts, trees, rng_left, part_begin, part_mid, gpu);
		children[RIGHT] = rockstar_form_tree(parts, trees, rng_right, part_mid, part_end, gpu);
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

int rockstar_form_tree(device_vector<rockstar_tree>& trees, device_vector<rockstar_particle>& parts, bool gpu) {
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
	return rockstar_form_tree(parts, trees, rng, 0, parts.size(), gpu);
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
		if (gpu) {
			device_vector<int> leaves;
			rockstar_find_neighbors(leaves, trees, trees.size() - 1, vector<int>(1, trees.size() - 1), max_link_len);
			rockstar_assign_linklen_cuda(trees, leaves, parts, max_link_len);
		} else {
			rockstar_assign_link_len(trees, parts, tree_root, vector<int>(1, tree_root), max_link_len);
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

bool rockstar_halo_bound(subgroup& halo, vector<rockstar_particle>& unbound_parts, float scale) {
	vector<array<float, NDIM>> x;
	for (int i = 0; i < halo.parts.size(); i++) {
		array<float, NDIM> X;
		for (int dim = 0; dim < NDIM; dim++) {
			X[dim] = halo.parts[i].X[dim];
		}
		x.push_back(X);
	}
	auto y = x;
	auto phi = bh_evaluate_potential(x, x.size() > ROCKSTAR_MIN_GPU);
	auto phi_tot = 0.0;
	auto kin_tot = 0.0;
	float vxa = 0.0;
	float vya = 0.0;
	float vza = 0.0;
	for (int i = 0; i < halo.parts.size(); i++) {
		vxa += halo.parts[i].vx;
		vya += halo.parts[i].vy;
		vza += halo.parts[i].vz;
	}
	vxa /= halo.parts.size();
	vya /= halo.parts.size();
	vza /= halo.parts.size();
	int nunbound = 0;
	int onparts = halo.parts.size();
	for (int i = 0; i < halo.parts.size(); i++) {
		const float dvx = halo.parts[i].vx - vxa;
		const float dvy = halo.parts[i].vy - vya;
		const float dvz = halo.parts[i].vz - vza;
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
		if (parts.size() < 2) {
//			PRINT("PARTS.SIZE = %i %i \n", parts.size(), sg.parts.size());
		}
		const int root_id = rockstar_form_tree(trees, parts, gpu);

		const float link_len = rockstar_find_link_len(trees, parts, root_id, ROCKSTAR_FF, gpu);
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
//	PRINT("Finding link_len\n");
	timer tm;
	tm.start();
//	PRINT("forming trees = %e\n");
	const int root_id = rockstar_form_tree(trees, parts, gpu);
	tm.stop();
//	PRINT("form_trees = %e\n", tm.read());
	tm.reset();
	tm.start();
	const float link_len = rockstar_find_link_len(trees, parts, root_id, ROCKSTAR_FF, gpu);
	tm.stop();
	//PRINT("find_link_len = %e\n", tm.read());
	tm.reset();
//	PRINT("link_len = %e\n", link_len / rfac);
//	PRINT("Finding subgroups\n");
	tm.start();
	rockstar_find_subgroups(trees, parts, link_len, next_id, gpu);
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
		array<float,NDIM> X;
		const auto Ndim = get_options().parts_dim;
		device_vector<rockstar_particle>& these_parts = sg.parts;
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
		const double omega_m = get_options().omega_m;
		const double x = ((omega_m / (sqr(scale)*scale)) / (omega_m / (sqr(scale)*scale)) + (1.0 - omega_m)) - 1.0;
		double y = (18.0 * M_PI * M_PI + 82.0 * x - 39.0 * sqr(x))/(1.0+x);
		double r_vir = pow(these_parts.size() / (4.0/3.0*M_PI*y), 1.0/3.0) / Ndim;

		sigma2_x /= these_parts.size();
		sg.sigma2_x = sigma2_x;
		sg.r_vir = r_vir;
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
			double H = get_options().hubble * constants::H0 * get_options().code_to_s;
			const float nparts = pow(get_options().parts_dim,3);
			for( int n = 1; n < radii.size(); n++) {
				float vcirc = sqrt(3.0 * get_options().omega_m * sqr(H) * n / (8.0 * M_PI * nparts * radii[n] / rfac));
				vcirc_max = std::max(vcirc_max,vcirc);
			}
			sg.vcirc_max = vcirc_max * vfac;
			y *= (1.0+x);
			//	PRINT( "%e\n", vcirc_max);
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
				const float dvx = (subgroups[k].vx - subgroups[l].vx) * sigma_v_inv;
				const float dvy = (subgroups[k].vy - subgroups[l].vy) * sigma_v_inv;
				const float dvz = (subgroups[k].vz - subgroups[l].vz) * sigma_v_inv;
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
					break;
				}
			}
			if (!found_merge) {
				k++;
			}
		}
		vector<vector<float>> ene(subgroups.size());
		vector<array<float, NDIM>> Y;
		for (int k = 0; k < parts.size(); k++) {
			array<float, NDIM> x;
			for (int dim = 0; dim < NDIM; dim++) {
				x[dim] = parts[k].X[dim];
			}
			Y.push_back(x);
		}
		for (int i = 0; i < subgroups.size(); i++) {
			vector<array<float, NDIM>> X;
			array<float, NDIM> vel;
			for (int dim = 0; dim < NDIM; dim++) {
				vel[dim] = 0.0;
			}
			auto& these_parts = subgroups[i].parts;
			for (int k = 0; k < these_parts.size(); k++) {
				array<float, NDIM> x;
				for (int dim = 0; dim < NDIM; dim++) {
					x[dim] = these_parts[k].X[dim];
					vel[dim] += these_parts[k].X[dim + NDIM];
				}
				X.push_back(x);
			}
			for (int dim = 0; dim < NDIM; dim++) {
				vel[dim] /= these_parts.size();
			}
			auto phi = bh_evaluate_points(Y, X);
			ene[i].resize(phi.size());
			for (int k = 0; k < parts.size(); k++) {
				const double dvx = parts[k].vx - vel[XDIM];
				const double dvy = parts[k].vy - vel[YDIM];
				const double dvz = parts[k].vz - vel[ZDIM];
				ene[i][k] = phi[k] / scale + 0.5 * sqr(dvx, dvy, dvz);
			}
		}
		for (int j = 0; j < parts.size(); j++) {
			float min_ene = std::numeric_limits<float>::max();
			int min_index = -1;
			for (int i = 0; i < subgroups.size(); i++) {
				if (ene[i][j] < min_ene) {
					min_ene = ene[i][j];
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
		int bound = 0;
		int unbound = 0;
		int rebound = 0;
		vector<rockstar_particle> unbounds;
		for (int k = 0; k < subgroups.size(); k++) {
			const int psize = subgroups[k].parts.size();
			bound += psize;
			if (!rockstar_halo_bound(subgroups[k], unbounds, scale)) {
				subgroups[k] = subgroups.back();
				subgroups.pop_back();
				k--;
			}
		}
		unbound += unbounds.size();
		bound -= unbound;
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
						if (r < min_dist) {
							min_dist = r;
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

		if (unbounds.size()) {
			vector<float> min_ene(unbounds.size(), std::numeric_limits<float>::max());
			vector<int> min_index(unbounds.size(), -1);
			vector<array<float, NDIM>> Y;
			for (int k = 0; k < unbounds.size(); k++) {
				array<float, NDIM> x;
				for (int dim = 0; dim < NDIM; dim++) {
					x[dim] = unbounds[k].X[dim];
				}
				Y.push_back(x);
			}
			for (int k = 0; k < subgroups.size(); k++) {
				if (subgroups[k].parent == -1) {
					auto tmp = rockstar_gather_halo_parts(subgroups, subgroups[k]);
					vector<array<float, NDIM>> X;
					array<float, NDIM> vel;
					for (int dim = 0; dim < NDIM; dim++) {
						vel[dim] = 0.0;
					}
					for (int l = 0; l < tmp.size(); l++) {
						array<float, NDIM> x;
						for (int dim = 0; dim < NDIM; dim++) {
							x[dim] = tmp[l].X[dim];
							vel[dim] += tmp[l].X[dim + NDIM];
						}
						X.push_back(x);
					}
					for (int dim = 0; dim < NDIM; dim++) {
						vel[dim] /= X.size();
					}
					auto phi = bh_evaluate_points(Y, X, X.size() * Y.size() >=  sqr(ROCKSTAR_MIN_GPU));
					for (int l = 0; l < unbounds.size(); l++) {
						const float dvx = unbounds[l].vx - vel[XDIM];
						const float dvy = unbounds[l].vy - vel[YDIM];
						const float dvz = unbounds[l].vz - vel[ZDIM];
						const float ekin = 0.5 * sqr(dvx, dvy, dvz);
						const float ene = ekin + phi[l] / scale;
						if (ene < 0.0f) {
							if (ene < min_ene[l]) {
								min_ene[l] = ene;
								min_index[l] = k;
							}
						}
					}
				}
			}
			for (int k = 0; k < unbounds.size(); k++) {
				if (min_index[k] != -1) {
					subgroups[min_index[k]].parts.push_back(unbounds[k]);
					unbound--;
					bound++;
					rebound++;
				}
			}
		}
		//PRINT("%e %e %i\n", (double ) unbound / (bound + unbound), (double ) rebound / (bound + unbound), (bound + unbound));

		vector<int> checklist;
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
		/*		for (int k = 0; k < subgroups.size(); k++) {
		 rockstar_subgroup_statistics(subgroups[k], gpu);
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
		 //				subgroups[k].children.size());*/
	}
	return subgroups;
}

void rockstar_find_subgroups(device_vector<rockstar_particle>& parts, float scale) {
	int next_id = 1;
	rockstar_seeds(parts, next_id, 1.0, 1.0, scale, false);
}

vector<subgroup> rockstar_find_subgroups(const vector<lc_entry>& parts, bool gpu) {
	device_vector<rockstar_particle> rock_parts(parts.size());
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
	return rockstar_seeds(rock_parts, next_id, 1.0, 1.0, 1.0, gpu);
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
