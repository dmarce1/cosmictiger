#include <cosmictiger/options.hpp>
#include <cosmictiger/rockstar.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/constants.hpp>

#define ROCKSTAR_BUCKET_SIZE 90
#define ROCKSTAR_NO_GROUP 0
#define ROCKSTAR_HAS_GROUP -1
#define ROCKSTAR_FF 0.7
#define ROCKSTAR_MIN_GROUP 10

struct rockstar_tree {
	int part_begin;
	int part_end;
	array<int, NCHILD> children;
	range<float, 2 * NDIM> box;
	bool active;
	bool last_active;
};

void rockstar_assign_link_len(const vector<rockstar_tree>& trees, vector<rockstar_particle>& parts, int self_id, vector<int> checklist, float link_len) {
	vector<int> nextlist;
	vector<int> leaflist;
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
						if (R2 <= link_len2) {
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
			rockstar_assign_link_len(trees, parts, self.children[RIGHT], checklist, link_len);
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

int rockstar_form_tree(vector<rockstar_particle>& parts, vector<rockstar_tree>& trees, range<float, 2 * NDIM> rng, int part_begin, int part_end) {
	array<int, NCHILD> children;
	rockstar_tree node;
	if (part_begin - part_end <= ROCKSTAR_BUCKET_SIZE) {
		children[LEFT] = children[RIGHT] = -1;
	} else {
		float midx;
		int max_dim;
		float total_max = 0.0;
		for (int dim = 0; dim < 2 * NDIM; dim++) {
			float x_max = -std::numeric_limits<float>::max();
			float x_min = std::numeric_limits<float>::max();
			for (int i = part_begin; i < part_end; i++) {
				const float x = parts[i].X[dim];
				x_max = std::max(x_max, x);
				x_min = std::min(x_min, x);
			}
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
	}
	node.part_begin = part_begin;
	node.part_end = part_end;
	node.children = children;
	node.box = rng;
	trees.push_back(node);
	return trees.size() - 1;
}

int rockstar_form_tree(vector<rockstar_tree>& trees, vector<rockstar_particle>& parts) {
	range<float, 2 * NDIM> rng;
	for (int dim = 0; dim < 2 * NDIM; dim++) {
		for (int i = 0; i < parts.size(); i++) {
			rng.begin[dim] = std::numeric_limits<float>::max();
			rng.end[dim] = -std::numeric_limits<float>::max();
			const float x = parts[i].X[dim];
			rng.begin[dim] = std::min(rng.begin[dim], x);
			rng.end[dim] = std::max(rng.end[dim], x);
		}
	}
	return rockstar_form_tree(parts, trees, rng, 0, parts.size());
}

float rockstar_find_link_len(const vector<rockstar_tree>& trees, vector<rockstar_particle>& parts, int tree_root) {
	range<float, 2 * NDIM> rng = trees[tree_root].box;
	float mean_sep = pow(rng.volume() / parts.size(), 1.0f / 6.0f);
	float max_link_len = 2.0 * mean_sep;
	bool done = false;
	do {
		for (auto& p : parts) {
			p.min_dist2 = std::numeric_limits<float>::max();
		}
		rockstar_assign_link_len(trees, parts, tree_root, vector<int>(1, tree_root), max_link_len);
		std::sort(parts.begin(), parts.end(), [](const rockstar_particle& a, const rockstar_particle& b) {
			return a.min_dist2 < b.min_dist2;
		});
		const int i0 = ROCKSTAR_FF * parts.size();
		const int i1 = ROCKSTAR_FF * parts.size() + 1;
		if (parts[i1].min_dist2 == std::numeric_limits<float>::max()) {
			max_link_len *= 2.0;
		} else {
			max_link_len = (sqrt(parts[i0].min_dist2) + sqrt(parts[i1].min_dist2)) * 0.5f;
			done = true;
		}
	} while (!done);
	return max_link_len;
}

size_t rockstar_find_subgroups(vector<rockstar_tree>& trees, vector<rockstar_particle>& parts, int self_id, vector<int> checklist, float link_len,
		int& next_id) {
	vector<int> nextlist;
	vector<int> leaflist;
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
							if (R2 <= link_len2) {
								if (mypart.subgroup == ROCKSTAR_NO_GROUP) {
									mypart.subgroup = next_id++;
									found_any_link = true;
								}
								if (mypart.subgroup > part.subgroup) {
									mypart.subgroup = part.subgroup;
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
				for (int pj = self.part_begin; pj < self.part_end; pj++) {
					if (pi != pj) {
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

void rockstar_find_subgroups(vector<rockstar_tree>& trees, vector<rockstar_particle>& parts, float link_len, int& next_id) {
	for (int i = 0; i < parts.size(); i++) {
		parts[i].subgroup = ROCKSTAR_NO_GROUP;
	}
	for (int i = 0; i < trees.size(); i++) {
		trees[i].active = true;
	}
	int cnt;
	do {
		for (int i = 0; i < trees.size(); i++) {
			trees[i].last_active = trees[i].active;
		}
		int root_id = trees.size() - 1;
		int cnt = rockstar_find_subgroups(trees, parts, root_id, vector<int>(1, root_id), link_len, next_id);
	} while (cnt != 0);
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
		if (table[parts[i].subgroup].n < ROCKSTAR_MIN_GROUP) {
			parts[i].subgroup = ROCKSTAR_NO_GROUP;
		}
	}
	for (auto i = table.begin(); i != table.end(); i++) {
		if (i->second.n < ROCKSTAR_MIN_GROUP) {
			i = table.erase(i);
		}
	}
	return table;
}

struct rockstar_seed {
	int id;
	array<float, 2 * NDIM> x;
};

struct subgroup {
	vector<rockstar_particle> parts;
	union {
		array<float, NDIM * 2> X;
		struct {
			float x;
			float y;
			float z;
			float vx;
			float vy;
			float vz;
		};
	};
	float r_dyn;
	float sigma2_v;
};

void rockstar_seeds(vector<rockstar_particle>& parts, int& next_id, float rfac, float vfac) {

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
	const float link_len = rockstar_find_link_len(trees, parts, rockstar_form_tree(trees, parts));
	rockstar_find_subgroups(trees, parts, link_len, next_id);

	auto group_cnts = rockstar_subgroup_cnts(parts);

	vector<subgroup> subgroups;

	for (auto i = group_cnts.begin(); i != group_cnts.end(); i++) {
		subgroups.resize(subgroups.size() + 1);
		auto& sg = subgroups.back();
		vector<rockstar_particle>& these_parts = sg.parts;
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
		rockstar_seeds(these_parts, next_id, rfac, vfac);
	}
	for (int i = 0; i < subgroups.size(); i++) {
		for (int j = 0; j < subgroups[i].parts.size(); j++) {
			parts.push_back(subgroups[i].parts[j]);
		}
	}

	group_cnts = rockstar_subgroup_cnts(parts);
	subgroups.resize(0);
	for (auto i = group_cnts.begin(); i != group_cnts.end(); i++) {
		subgroups.resize(subgroups.size() + 1);
		auto& sg = subgroups.back();
		vector<rockstar_particle>& these_parts = sg.parts;
		for (int dim = 0; dim < 2 * NDIM; dim++) {
			sg.X[dim] = 0.0;
		}
		int j = 0;
		while (j < parts.size()) {
			if (parts[j].subgroup == i->first) {
				these_parts.push_back(parts[j]);
				parts[j] = parts.back();
				parts.pop_back();
				for (int dim = 0; dim < 2 * NDIM; dim++) {
					sg.X[dim] += parts[j].X[dim];
				}
			} else {
				j++;
			}
		}
		for (int dim = 0; dim < 2 * NDIM; dim++) {
			sg.X[dim] /= these_parts.size();
		}
		float vcirc_max = 0.0;
		float sigma2_v = 0.0;
		for (int j = 0; j < these_parts.size(); j++) {
			const auto& p = these_parts[j];
			const float dx = p.x - sg.x;
			const float dy = p.y - sg.y;
			const float dz = p.z - sg.z;
			const float dvx = p.vx - sg.vx;
			const float dvy = p.vy - sg.vy;
			const float dvz = p.vz - sg.vz;
			sigma2_v += sqr(dvx, dvy, dvz);
			const float cx = dy * dvz - dz * dvy;
			const float cy = dx * dvz - dz * dvx;
			const float cz = dx * dvy - dy * dvx;
			const float vcirc = sqrt(sqr(cx, cy, cz)) / sqrt(sqr(dx, dy, dz));
			vcirc_max = std::max(vcirc_max, vcirc);
		}
		sigma2_v /= these_parts.size();
		sg.sigma2_v = sigma2_v;
		sg.r_dyn = vcirc_max * rfac / vfac / (get_options().hubble * constants::H0) / sqrt(180);
	}
	if (group_cnts.size() == 0) {
		const int subgrp = next_id++;
		for (int i = 0; i < parts.size(); i++) {
			parts[i].subgroup = subgrp;
		}
	} else if (group_cnts.size() == 1) {
		const int subgrp = subgroups[0].parts[0].subgroup;
		for (int i = 0; i < parts.size(); i++) {
			parts[i].subgroup = subgrp;
		}
		for (int i = 0; i < subgroups[0].parts.size(); i++) {
			parts.push_back(subgroups[0].parts[i]);
		}
	} else {
		for (int j = 0; j < parts.size(); j++) {
			float min_dist = std::numeric_limits<float>::max();
			int min_index;
			for (int i = 0; i < subgroups.size(); i++) {
				const float rdyn_inv = 1.0f / subgroups[i].r_dyn;
				const float sigma_v_inv = 1.0f / sqrt(subgroups[i].sigma2_v);
				const float dx = (parts[j].x - subgroups[i].x) * rdyn_inv;
				const float dy = (parts[j].y - subgroups[i].y) * rdyn_inv;
				const float dz = (parts[j].z - subgroups[i].z) * rdyn_inv;
				const float dvx = (parts[j].vx - subgroups[i].vx) * sigma_v_inv;
				const float dvy = (parts[j].vy - subgroups[i].vy) * sigma_v_inv;
				const float dvz = (parts[j].vz - subgroups[i].vz) * sigma_v_inv;
				const float dist = sqrt(sqr(dx, dy, dz) + sqr(dvx, dvy, dvz));
				if (dist < min_dist) {
					min_dist = dist;
					min_index = i;
				}
			}
			parts[j].subgroup = subgroups[min_index].parts[0].subgroup;
		}
		for (int k = 0; k < subgroups.size(); k++) {
			for (int i = 0; i < subgroups[k].parts.size(); i++) {
				parts.push_back(subgroups[k].parts[i]);
			}
		}
	}
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
