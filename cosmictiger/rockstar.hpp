#pragma once

#include <cosmictiger/defs.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/groups.hpp>

struct rockstar_particle {
	float x;
	float y;
	float z;
	float vx;
	float vy;
	float vz;
	int subgroup;
};

vector<rockstar_particle> rockstar_seeds(vector<rockstar_particle>& parts);

