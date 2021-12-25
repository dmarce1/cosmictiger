#pragma once

#include <cosmictiger/defs.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/groups.hpp>

#define ROCKSTAR_BUCKET_SIZE 64
#define ROCKSTAR_NO_GROUP 0
#define ROCKSTAR_HAS_GROUP -1
#define ROCKSTAR_FF 0.7
#define ROCKSTAR_MIN_GROUP 10
#define ROCKSTAR_MIN_BOUND 0.5

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
	part_int index;
	double min_dist2;
};


struct subgroup {
	int id;
	int parent;
	vector<int> children;
	vector<rockstar_particle> parts;
	union {
		array<double, NDIM * 2> X;
		struct {
			double x;
			double y;
			double z;
			double vx;
			double vy;
			double vz;
		};
	};
	double min_xfactor;
	double r_dyn;
	double sigma2_v;
	double sigma2_x;
	double vcirc_max;
	double r_vir;
	int host_part_cnt;
	int depth;
	subgroup() {
		depth = -1;
		min_xfactor = std::numeric_limits<double>::max();
		parent = ROCKSTAR_NO_GROUP;
	}
};




void rockstar_find_subgroups(vector<rockstar_particle>& parts, double scale = 1.0);
vector<subgroup> rockstar_find_subgroups(const vector<particle_data>& parts, double scale);
