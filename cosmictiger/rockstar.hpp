#pragma once

#include <cosmictiger/defs.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/groups.hpp>
#include <cosmictiger/lightcone.hpp>

#define ROCKSTAR_BUCKET_SIZE 32
#define ROCKSTAR_NO_GROUP 0x7FFFFFFF
#define ROCKSTAR_HAS_GROUP -1
#define ROCKSTAR_FF 0.7
#define ROCKSTAR_MIN_GROUP 10
#define ROCKSTAR_MIN_BOUND 0.5
#define ROCKSTAR_TARGET_BLOCKS 1024


struct rockstar_particles {
	float* x;
	float* y;
	float* z;
	float* vx;
	float* vy;
	float* vz;
	int* subgroup;
};

struct rockstar_particle {
	union {
		array<float, 2 * NDIM> X;
		struct {
			float x;
			float y;
			float z;
			float vx;
			float vy;
			float vz;
		};
	};
	int subgroup;
	part_int index;
	float min_dist2;
};

struct rockstar_tree {
	int part_begin;
	int part_end;
	array<int, NCHILD> children;
	range<float, 2 * NDIM> box;
	int active_count;
	bool active;
	bool last_active;
};

struct subgroup {
	int id;
	int parent;
	vector<int> children;
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
	float min_xfactor;
	float r_dyn;
	float sigma2_v;
	float sigma2_x;
	float vcirc_max;
	float r_vir;
	int host_part_cnt;
	int depth;
	subgroup() {
		depth = -1;
		min_xfactor = std::numeric_limits<float>::max();
		parent = ROCKSTAR_NO_GROUP;
	}
};

void rockstar_find_subgroups(vector<rockstar_particle>& parts, float scale = 1.0);
vector<subgroup> rockstar_find_subgroups(const vector<particle_data>& parts, float scale);
vector<size_t> rockstar_find_subgroups_gpu(vector<rockstar_tree, pinned_allocator<rockstar_tree>>& trees, rockstar_particles parts, const vector<int>& selves,
		const vector<vector<int>>& checklists, float link_len, int& next_index);

vector<subgroup> rockstar_find_subgroups(const vector<lc_entry>& parts);
