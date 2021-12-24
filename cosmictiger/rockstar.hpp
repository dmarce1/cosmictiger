#pragma once

#include <cosmictiger/defs.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/groups.hpp>

struct rockstar_particle {
	union {
		array<double, 2 * NDIM> X;
		struct {
			double x;
			double y;
			double z;
			double vx;
			double vy;
			double vz;
		};
	};
	int subgroup;
	int depth;
	part_int index;
	double min_dist2;
};



void rockstar_find_subgroups(vector<rockstar_particle>& parts, double scale = 1.0);
