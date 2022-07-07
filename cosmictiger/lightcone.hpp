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

using lc_real = fixed<int,31>;

#define LC_NO_GROUP (0xFFFFFFFFFFFFFFULL)
#define LC_EDGE_GROUP (0x0ULL)

struct lc_tree_id {
	int pix;
	int index;
	CUDA_EXPORT
	bool operator!=(const lc_tree_id& other) const {
		return pix != other.pix || index != other.index;
	}
};


struct lc_tree_node {
	range<double> box;
	array<lc_tree_id, NCHILD> children;
	pair<int> part_range;
	bool active;
	bool last_active;
	device_vector<lc_tree_id> neighbors;
	int pix;
};

struct lc_entry {
	lc_real x, y, z;
	float vx, vy, vz;
};

using lc_group = unsigned long long;



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

using lc_part_map_type = cuda_unordered_map<device_vector<lc_particle>>;
using lc_tree_map_type = cuda_unordered_map<device_vector<lc_tree_node>>;

int lc_nside();
void lc_init(double, double);
int lc_add_particle(double x0, double y0, double z0, double x1, double y1, double z1, float vx, float vy, float vz, float t, float dt, vector<lc_particle>& this_part_buffer);
void lc_add_parts(vector<lc_particle>&&);
size_t lc_add_parts(const device_vector<device_vector<lc_entry>>& entries);
void lc_buffer2homes();
size_t lc_time_to_flush(double, double);
void lc_particle_boundaries1();
void lc_particle_boundaries2();
void lc_form_trees(double tmax, double link_len);
size_t lc_find_groups();
size_t lc_find_neighbors();
void lc_groups2homes();
void lc_parts2groups(double a, double link_len);
void lc_save(FILE* fp);
void lc_load(FILE* fp);
size_t cuda_lightcone(const device_vector<lc_tree_id>& leaves, lc_part_map_type* part_map_ptr, lc_tree_map_type* tree_map_ptr, lc_group* next_group);



