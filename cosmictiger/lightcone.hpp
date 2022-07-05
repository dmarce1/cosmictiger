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

#pragma once

#include <cosmictiger/containers.hpp>
#include <cosmictiger/fixed.hpp>
#include <cosmictiger/cuda_unordered_map.hpp>
#include <cosmictiger/range.hpp>

using lc_real = fixed32;


#define LC_NO_GROUP (0x7FFFFFFFFFFFFFFFLL)
#define LC_EDGE_GROUP (0x0LL)

struct lc_tree_id {
	int pix;
	int index;
	CUDA_EXPORT
	bool operator!=(const lc_tree_id& other) const {
		return pix != other.pix || index != other.index;
	}
};


struct lc_tree_node {
	range<lc_real> box;
	array<lc_tree_id, NCHILD> children;
	pair<int> part_range;
	bool active;
	bool last_active;
	device_vector<lc_tree_id> neighbors;
	int pix;
};

struct lc_entry {
	fixed32 x, y, z;
	float vx, vy, vz;
};

using lc_group = long long;



struct lc_particle {
	array<lc_real, NDIM> pos;
	array<float, NDIM> vel;
	lc_group group;
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & pos;
		arc & vel;
		arc & group;
	}
};


cuda_unordered_map<device_vector<lc_particle>>& get_part_map();
cuda_unordered_map<device_vector<lc_tree_node>>& get_tree_map();
void lc_init(double, double);
int lc_add_particle(lc_real x0, lc_real y0, lc_real z0, lc_real x1, lc_real y1, lc_real z1, float vx, float vy, float vz, float t, float dt, vector<lc_particle>& this_part_buffer);
void lc_add_parts(vector<lc_particle>&&);
void lc_add_parts(const lc_entry* entries, int count);
void lc_buffer2homes();
size_t lc_time_to_flush(double, double);
void lc_particle_boundaries1();
void lc_particle_boundaries2();
void lc_form_trees(double tmax, double link_len);
size_t lc_find_groups();
void lc_groups2homes();
void lc_parts2groups(double a, double link_len);
void lc_save(FILE* fp);
void lc_load(FILE* fp);
vector<float> lc_flush_final();
size_t cuda_lightcone(const device_vector<lc_tree_id>& leaves);



