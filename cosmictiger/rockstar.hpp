#pragma once

#include <cosmictiger/defs.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/groups.hpp>

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



void rockstar_find_subgroups(vector<rockstar_particle>& parts);
