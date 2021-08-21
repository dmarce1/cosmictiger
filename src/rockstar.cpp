#include <cosmictiger/gravity.hpp>
#include <cosmictiger/groups.hpp>
#include <cosmictiger/rockstar.hpp>

#include <unordered_set>

#define ROCKSTAR_BUCKET_SIZE 8
#define ROCKSTAR_THRESHOLD 0.7
#define ROCKSTAR_MIN_PARTS 10
#define ROCKSTAR_NO_GROUP 0x7FFFFFFF

using phase_range = range<float, 2 * NDIM>;

struct rockstar_tree_node {
	phase_range box;
	array<int, NCHILD> children;
	pair<int> parts;
	bool active;
};

int rockstar_particles_sort(vector<phase_t>& parts, pair<int> rng, double xmid, int xdim) {
	int lo = rng.first;
	int hi = rng.second;
	while (lo < hi) {
		if (parts[lo][xdim] >= xmid) {
			while (lo != hi) {
				hi--;
				if (parts[hi][xdim] < xmid) {
					std::swap(parts[hi], parts[lo]);
					break;
				}
			}
		}
		lo++;
	}
	return hi;

}

int rockstar_tree_create(vector<rockstar_tree_node>& nodes, vector<phase_t>& parts, phase_range box, pair<int> part_range) {
	rockstar_tree_node node;
	const int nparts = part_range.second - part_range.first;
	if (nparts > ROCKSTAR_BUCKET_SIZE) {
		const int xdim = box.longest_dim();
		const float xmid = (box.begin[xdim] + box.end[xdim]) * 0.5;
		const int imid = rockstar_particles_sort(parts, part_range, xmid, xdim);
		phase_range left_box = box;
		phase_range right_box = box;
		pair<int> left_parts = part_range;
		pair<int> right_parts = part_range;
		left_box.end[xdim] = right_box.begin[xdim] = xmid;
		left_parts.second = right_parts.first = imid;
		node.children[LEFT] = rockstar_tree_create(nodes, parts, left_box, left_parts);
		node.children[RIGHT] = rockstar_tree_create(nodes, parts, right_box, right_parts);
		left_box = nodes[node.children[LEFT]].box;
		right_box = nodes[node.children[RIGHT]].box;
		for (int dim = 0; dim < 2 * NDIM; dim++) {
			box.begin[dim] = std::min(left_box.begin[dim], right_box.begin[dim]);
			box.end[dim] = std::max(left_box.end[dim], right_box.end[dim]);
		}
	} else {
		for (int dim = 0; dim < 2 * NDIM; dim++) {
			box.begin[dim] = std::numeric_limits<float>::max();
			box.end[dim] = -std::numeric_limits<float>::max();
		}
		for (int i = 0; i < parts.size(); i++) {
			for (int dim = 0; dim < 2 * NDIM; dim++) {
				box.begin[dim] = std::min(box.begin[dim], parts[i][dim]);
				box.end[dim] = std::max(box.end[dim], parts[i][dim]);
			}
		}
		for (int ci = 0; ci < NCHILD; ci++) {
			node.children[ci] = -1;
		}
	}
	node.box = box;
	node.active = true;
	node.parts = part_range;
	int myindex = nodes.size();
	nodes.push_back(node);
	return myindex;
}

int rockstar_groups_find(vector<rockstar_tree_node>& nodes, const vector<phase_t>& parts, vector<int>& groups, int self_index, vector<int> checklist,
		double link_len) {
	rockstar_tree_node& self = nodes[self_index];
	vector<int> nextlist;
	vector<int> leaflist;
	nextlist.resize(0);
	leaflist.resize(0);
	const auto self_box = self.box.pad(link_len * 1.001);
	const bool iamleaf = self.children[LEFT] == -1;
	do {
		for (int ci = 0; ci < checklist.size(); ci++) {
			const rockstar_tree_node& other = nodes[checklist[ci]];
			if (other.active) {
				if (self_box.intersection(other.box).volume() > 0) {
					if (other.children[LEFT] == -1) {
						leaflist.push_back(checklist[ci]);
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
	if (self.children[LEFT] == -1) {
		vector<phase_t> X;
		vector<int> G;
		if (leaflist.size()) {
			const auto my_rng = self.parts;
			const float link_len2 = sqr(link_len);
			bool found_any_link = false;
			int total_size = 0;
			for (int i = 0; i < leaflist.size(); i++) {
				const rockstar_tree_node& other = nodes[leaflist[i]];
				if (leaflist[i] != self_index) {
					const auto& other_rng = other.parts;
					const int other_size = other_rng.second - other_rng.first;
					total_size += other_size;
				}
			}
			X.reserve(total_size);
			G.reserve(total_size);
			for (int i = 0; i < leaflist.size(); i++) {
				const rockstar_tree_node& other = nodes[leaflist[i]];
				if (leaflist[i] != self_index) {
					const auto& other_rng = other.parts;
					int other_size = other_rng.second - other_rng.first;
					for (int i = other_rng.first; i < other_rng.second; i++) {
						if (self_box.contains(parts[i])) {
							X.push_back(parts[i]);
							G.push_back(groups[i]);
						}
					}
				}
			}
			for (int k = my_rng.first; k < my_rng.second; k++) {
				for (int j = 0; j < X.size(); j++) {
					float r2 = 0.0f;
					for (int dim = 0; dim < 2 * NDIM; dim++) {
						r2 += sqr(parts[k][dim] - X[j][dim]);
					}
					if (r2 < link_len2) {
						auto& grp = groups[k];
						if ((int) grp == ROCKSTAR_NO_GROUP) {
							grp = k;
							found_any_link = true;
						}
						if (grp > G[j]) {
							grp = G[j];
							found_any_link = true;
						}
					}
				}
			}
			bool found_link;
			do {
				found_link = false;
				for (int j = my_rng.first; j < my_rng.second; j++) {
					for (int k = j + 1; k < my_rng.second; k++) {
						float r2 = 0.0f;
						for (int dim = 0; dim < 2 * NDIM; dim++) {
							r2 += sqr(parts[j][dim] - parts[k][dim]);
						}
						if (r2 < link_len2) {
							auto& grpa = groups[k];
							auto& grpb = groups[j];
							if (grpa == ROCKSTAR_NO_GROUP) {
								grpa = k;
								found_link = true;
								found_any_link = true;
							}
							if (grpb == ROCKSTAR_NO_GROUP) {
								grpb = j;
								found_link = true;
								found_any_link = true;
							}
							if (grpa != grpb) {
								if (grpa < grpb) {
									grpb = grpa;
								} else {
									grpa = grpb;
								}
								found_link = true;
								found_any_link = true;
							}
						}
					}
				}
			} while (found_link);
			if (found_any_link) {
				self.active = true;
				return 1;
			} else {
				self.active = false;
				return 0;
			}
		} else {
			self.active = false;
			return 0;
		}
	} else {
		checklist.insert(checklist.end(), leaflist.begin(), leaflist.end());
		if (checklist.size()) {
			const int lcount = rockstar_groups_find(nodes, parts, groups, self.children[LEFT], checklist, link_len);
			const int rcount = rockstar_groups_find(nodes, parts, groups, self.children[RIGHT], std::move(checklist), link_len);
			const int tot = rcount + lcount;
			self.active = tot != 0;
			return tot;
		} else {
			self.active = false;
			return 0;
		}
	}
}

static pair<float> compute_norms(const vector<phase_t>& parts) {
	phase_t com;
	for (int dim = 0; dim < 2 * NDIM; dim++) {
		com[dim] = 0.0f;
	}
	for (int i = 0; i < parts.size(); i++) {
		for (int dim = 0; dim < 2 * NDIM; dim++) {
			com[dim] += parts[i][dim];
		}
	}
	for (int dim = 0; dim < 2 * NDIM; dim++) {
		com[dim] /= parts.size();
	}
	float x2 = 0.0;
	float v2 = 0.0;
	for (int i = 0; i < parts.size(); i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			x2 += sqr(parts[i][dim] - com[dim]);
		}
		for (int dim = NDIM; dim < 2 * NDIM; dim++) {
			v2 += sqr(parts[i][dim] - com[dim]);
		}
	}
	x2 /= parts.size();
	v2 /= parts.size();
	pair<float> rc;
	rc.first = sqrtf(x2);
	rc.second = sqrtf(v2);
	if (rc.second == 0.0) {
		rc.second = 1.0;
	}
	if (rc.first == 0.0) {
		rc.first = 1.0;
	}
	return rc;
}

static phase_range compute_root_box(const vector<phase_t>& parts) {
	phase_range box;
	for (int dim = 0; dim < 2 * NDIM; dim++) {
		box.begin[dim] = std::numeric_limits<float>::max();
		box.end[dim] = -std::numeric_limits<float>::max();
	}
	for (int i = 0; i < parts.size(); i++) {
		for (int dim = 0; dim < 2 * NDIM; dim++) {
			box.begin[dim] = std::min(box.begin[dim], parts[i][dim]);
			box.end[dim] = std::max(box.end[dim], parts[i][dim]);
		}
	}
	return box;
}

static void normalize(vector<phase_t>& parts, float sigma_x, float sigma_v) {
	const float sigma_x_inv = 1.0f / sigma_x;
	const float sigma_v_inv = 1.0f / sigma_v;
	for (int i = 0; i < parts.size(); i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			parts[i][dim] *= sigma_x_inv;
		}
		for (int dim = NDIM; dim < 2 * NDIM; dim++) {
			parts[i][dim] *= sigma_v_inv;
		}
	}
}

static double max_boxdist(phase_range box) {
	double r2 = 0.0;
	for (int dim = 0; dim < 2 * NDIM; dim++) {
		r2 += sqr(box.begin[dim] - box.end[dim]);
	}
	return sqrtf(r2);
}

static double fraction_in_groups(vector<int>& groups) {
	int ngroup = 0;
	for (int i = 0; i < groups.size(); i++) {
		if (groups[i] != ROCKSTAR_NO_GROUP) {
			ngroup++;
		}
	}
	return (double) ngroup / groups.size();
}

static vector<vector<phase_t>> separate_groups(const vector<phase_t>& parts, const vector<int>& groups) {
	std::unordered_map<int, vector<phase_t>> sep_parts;
	for (int i = 0; i < parts.size(); i++) {
		if (groups[i] != ROCKSTAR_NO_GROUP) {
			sep_parts[groups[i]].push_back(parts[i]);
		}
	}
	PRINT( "--- %i %i\n", sep_parts.size(), parts.size());
	vector<vector<phase_t>> rc;
	for (auto i = sep_parts.begin(); i != sep_parts.end(); i++) {
		PRINT( "%i %i\n", i->first, i->second.size());
		rc.push_back(std::move(i->second));
	}
	return rc;
}

vector<seed_halo> rockstar_seed_halos_find(vector<phase_t>& parts) {
	if (parts.size() >= ROCKSTAR_MIN_PARTS) {
		pair<float> norms = compute_norms(parts);
		const float sigma_x = norms.first;
		const float sigma_v = norms.second;
//		PRINT("%e %e\n", sigma_x, sigma_v);
		normalize(parts, sigma_x, sigma_v);
		vector<int> groups(parts.size());
		vector<rockstar_tree_node> nodes;
		pair<int> root_range;
		root_range.first = 0;
		root_range.second = parts.size();
		const auto root_box = compute_root_box(parts);
//		PRINT("%s\n", root_box.to_string().c_str());
//		PRINT("Creating tree\n");
		int root_index = rockstar_tree_create(nodes, parts, root_box, root_range);
//		PRINT("Tree created with %i nodes\n", root_index);
		double max_link_len = max_boxdist(root_box);
//		PRINT("max_link_len = %e\n", max_link_len);
		for (double link_len = 0.05 * max_link_len; link_len <= max_link_len * 1.001; link_len += 0.05 * max_link_len) {
			std::fill(groups.begin(), groups.end(), ROCKSTAR_NO_GROUP);
			for (auto& n : nodes) {
				n.active = true;
			}
			int nactive;
			do {
				nactive = rockstar_groups_find(nodes, parts, groups, root_index, vector<int>(1, root_index), link_len);
			} while (nactive > 0);
			const double frac = fraction_in_groups(groups);
			if (frac > ROCKSTAR_THRESHOLD) {
				max_link_len = link_len;
				break;
			}
		}
		double min_link_len = 0.0;
		double dif;
		double min_dif = std::numeric_limits<float>::max();
		bool done = false;
		do {
			double link_len = (max_link_len + min_link_len) * 0.5;
			std::fill(groups.begin(), groups.end(), ROCKSTAR_NO_GROUP);
			for (auto& n : nodes) {
				n.active = true;
			}
			int nactive;
			do {
				nactive = rockstar_groups_find(nodes, parts, groups, root_index, vector<int>(1, root_index), link_len);
			} while (nactive > 0);
			const double frac = fraction_in_groups(groups);
			dif = frac - ROCKSTAR_THRESHOLD;
			if (dif > 0.0) {
				max_link_len = link_len;
			} else {
				min_link_len = link_len;
			}
			if (min_dif == std::abs(dif)) {
				done = true;
			}
			min_dif = std::min(min_dif, std::abs(dif));
		} while (!done);
		auto subgroups = separate_groups(parts, groups);
		vector<seed_halo> halos;
		for (int i = 0; i < subgroups.size(); i++) {
			auto these_halos = rockstar_seed_halos_find(subgroups[i]);
			for (int j = 0; j < these_halos.size(); j++) {
				these_halos[j].normalize(sigma_x, sigma_v);
				halos.push_back(these_halos[j]);
			}
		}
		for (int i = 0; i < halos.size(); i++) {
			int j = i + 1;
			while (j < halos.size()) {
				if (halos[i].indistinguishable_from(halos[j])) {
					halos[i] += halos[j];
					halos[j] = halos.back();
					halos.pop_back();
				} else {
					j++;
				}
			}
		}
		return halos;
	} else {
		vector<seed_halo> halos(1, seed_halo(std::move(parts)));
		return halos;
	}
}

vector<seed_halo> rockstar_seed_halos(const vector<particle_data>& in_parts) {
	vector<phase_t> parts(in_parts.size());
	array<fixed32, NDIM> x0;
	for (int dim = 0; dim < NDIM; dim++) {
		x0[dim] = in_parts[0].x[dim];
	}
	for (int i = 0; i < parts.size(); i++) {
		for (int dim = 0; dim < NDIM; dim++) {
			parts[i][dim] = distance(in_parts[i].x[dim], x0[dim]);
		}
		for (int dim = 0; dim < NDIM; dim++) {
			parts[i][NDIM + dim] = in_parts[i].v[dim];
		}
	}
	auto halos = rockstar_seed_halos_find(parts);
	for (auto& halo : halos) {
		for (int dim = 0; dim < NDIM; dim++) {
			halo.x[dim] += x0[dim].to_double();
		}
	}
	return halos;
}

