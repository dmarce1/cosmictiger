#include <cosmictiger/rockstar.hpp>
#include <cosmictiger/math.hpp>

#define ROCKSTAR_BUCKET_SIZE 90

struct rockstar_tree {
	int part_begin;
	int part_end;
	array<int, NCHILD> children;
	range<float, 2 * NDIM> box;
};

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

vector<rockstar_particle> rockstar_seeds(vector<rockstar_particle>& parts) {
	constexpr float ff = 0.7;
	vector<rockstar_particle> seeds;

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

	float link_len2 = 3.0 * (sigma2_x + sigma2_v);
	int target_cnt = parts.size() * ff;
	/*	while (rockstar_find_subgroups(parts, link_len2) < target_cnt) {
	 link_len2 *= 2.0;
	 }
	 float link_len2_max = link_len2;
	 float link_len2_min = 0.0;
	 while (link_len2_max / link_len2_min > 1.01) {
	 float link_len2_mid = (link_len2_max + link_len2_min) * 0.5;
	 if ((rockstar_find_subgroups(parts, link_len2_mid) - target_cnt) * (rockstar_find_subgroups(parts, link_len2_max) - target_cnt)) {
	 link_len2_min = link_len2_mid;
	 } else {
	 link_len2_max = link_len2_mid;
	 }
	 }*/
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
