#pragma once

#include <cosmictiger/defs.hpp>
#include <cosmictiger/containers.hpp>
#include <cosmictiger/groups.hpp>
#include <cosmictiger/lightcone.hpp>
#include <cosmictiger/fp16.hpp>

//#define ROCKSTAR_CPU_BUCKET_SIZE 32
#define ROCKSTAR_BUCKET_SIZE 64
#define ROCKSTAR_NO_GROUP 0x7FFFFFFF
#define ROCKSTAR_HAS_GROUP -1
#define ROCKSTAR_FF 0.7
#define ROCKSTAR_MIN_GROUP 10
#define ROCKSTAR_MIN_BOUND 0.5
#define ROCKSTAR_TARGET_BLOCKS (1024)

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
	device_vector<int> neighbors;
};

union rockstar_id {
	struct {
		unsigned long long fof_id :48;
		unsigned long long halo_id :15;
		unsigned long long sys :1;
	};
	unsigned long long id;
};

struct rockstar_record {
	rockstar_id id;
	double x;
	double y;
	double z;
	float mvir;
	float E;
	float Jx;
	float Jy;
	float Jz;
	float m200;
	float m500;
	float m2500;
	short pid;
	float16 Z;
	float16 rmax;
	float16 rvir;
	float16 rvmax;
	float16 rs;
	float16 rkly;
	float16 sig_x;
	float16 delta_x;
	float16 ToW;
	float16 sig_v;
	float16 sig_vb;
	float16 vcirc_max;
	float16 delta_v;
	float16 boa;
	float16 coa;
	float16 Ax;
	float16 Ay;
	float16 Az;
	float16 boa500;
	float16 coa500;
	float16 Ax500;
	float16 Ay500;
	float16 Az500;
	float16 lambda;
	float16 lambda_B;
	float16 vxc;
	float16 vyc;
	float16 vzc;
	float16 vxb;
	float16 vyb;
	float16 vzb;
};

struct subgroup {
	int id;
	int parent;
	vector<int> children;
	device_vector<rockstar_particle> parts;
	float x;
	float y;
	float z;
	float vxc;
	float vyc;
	float vzc;
	float vxb;
	float vyb;
	float vzb;
	float T;
	float W;
	float sigma_x_min;
	float r_dyn;
	float sigma2_v;
	float sigma2_x;
	float vcirc_max;
	float r_vir;
	int host_part_cnt;
	int depth;
	subgroup() {
		depth = -1;
		sigma_x_min = std::numeric_limits<float>::max();
		parent = -1;
	}
};

vector<subgroup> rockstar_find_subgroups(const vector<particle_data>& parts, float scale);
vector<rockstar_record> rockstar_find_subgroups(const vector<lc_entry>& parts, double);
void rockstar_find_subgroups_cuda(device_vector<rockstar_tree>& nodes, const device_vector<int>& leaves, device_vector<rockstar_particle>& parts,
		float link_len, int& next_id);
void rockstar_assign_linklen_cuda(const device_vector<rockstar_tree>& nodes, const device_vector<int>& leaves, device_vector<rockstar_particle>& parts,
		float link_len, bool phase);
bool rockstar_cuda_free();
