#include <cosmictiger/rockstar.hpp>
#include <cosmictiger/math.hpp>

static int rockstar_find_subgroups(vector<rockstar_particle>& parts, float link_len2) {
	int next_group = 1;
	for (int i = 0; i < parts.size(); i++) {
		parts[i].subgroup = 0;
	}
	bool change;
	do {
		change = false;
		for (int i = 0; i < parts.size(); i++) {
			for (int j = i + 1; j < parts.size(); j++) {
				const float dx2 = sqr(parts[i].x - parts[j].x);
				const float dy2 = sqr(parts[i].y - parts[j].y);
				const float dz2 = sqr(parts[i].z - parts[j].z);
				const float dvx2 = sqr(parts[i].vx - parts[j].vx);
				const float dvy2 = sqr(parts[i].vy - parts[j].vy);
				const float dvz2 = sqr(parts[i].vz - parts[j].vz);
				if (dx2 + dy2 + dz2 + dvx2 + dvy2 + dvz2 < link_len2) {
					if (parts[i].subgroup == 0) {
						parts[i].subgroup = next_group++;
						change = true;
					}
					if (parts[j].subgroup == 0) {
						change = true;
						parts[j].subgroup = next_group++;
					}
					if (parts[i].subgroup < parts[j].subgroup) {
						change = true;
						parts[j].subgroup = parts[i].subgroup;
					} else if (parts[i].subgroup > parts[j].subgroup) {
						change = true;
						parts[i].subgroup = parts[j].subgroup;
					}
				}
			}
		}
	} while (change);
	int cnt = 0;
	for (int i = 0; i < parts.size(); i++) {
		bool in_grp = false;
		for (int j = 0; j < parts.size(); j++) {
			if (i != j) {
				const float dx2 = sqr(parts[i].x - parts[j].x);
				const float dy2 = sqr(parts[i].y - parts[j].y);
				const float dz2 = sqr(parts[i].z - parts[j].z);
				const float dvx2 = sqr(parts[i].vx - parts[j].vx);
				const float dvy2 = sqr(parts[i].vy - parts[j].vy);
				const float dvz2 = sqr(parts[i].vz - parts[j].vz);
				if (dx2 + dy2 + dz2 + dvx2 + dvy2 + dvz2 < link_len2) {
					in_grp = true;
				}
			}
			if (in_grp) {
				break;
			}
		}
		if (in_grp) {
			cnt++;
		}
	}
	return cnt;
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
	while (rockstar_find_subgroups(parts, link_len2) < target_cnt) {
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
