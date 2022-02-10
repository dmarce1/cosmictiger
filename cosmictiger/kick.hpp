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

#ifndef KICK_HPP_
#define KICK_HPP_

#include <cosmictiger/cuda_vector.hpp>
#include <cosmictiger/stack_vector.hpp>
#include <cosmictiger/tree.hpp>

#include <atomic>

struct cuda_kick_data {
#ifdef SPH_TOTAL_ENERGY
	float* sph_energy;
#endif
	tree_node* tree_nodes;
	fixed32* x;
	fixed32* y;
	fixed32* z;
	float* hsoft;
	float* vx;
	float* vy;
	float* vz;
	float* sph_gx;
	float* sph_gy;
	float* sph_gz;
	part_int* sph_index;
	char* type;
	char* sph;
	bool vsoft;
	char* rungs;
	float* gx;
	float* gy;
	float* gz;
	float* pot;
	int source_size;
	int sink_size;
	int tree_size;
	int rank;
};

struct kick_return;

#ifdef __CUDACC__
struct cuda_kick_shmem {
	array<fixed32, BUCKET_SIZE> sink_x;
	array<fixed32, BUCKET_SIZE> sink_y;
	array<fixed32, BUCKET_SIZE> sink_z;
	array<float, BUCKET_SIZE> sink_hsoft;
	struct {
		array<fixed32, KICK_PP_MAX> x;
		array<fixed32, KICK_PP_MAX> y;
		array<fixed32, KICK_PP_MAX> z;
		array<char, KICK_PP_MAX> sph;
		array<float, KICK_PP_MAX> hsoft;
	}src;
	array<float, BUCKET_SIZE> gx;
	array<float, BUCKET_SIZE> gy;
	array<float, BUCKET_SIZE> gz;
	array<float, BUCKET_SIZE> phi;
	array<part_int,BUCKET_SIZE> active;
	array<char,BUCKET_SIZE> rungs;
};
#endif

struct kick_return {
	char max_rung;
	double part_flops;
	double node_flops;
	double pot;
	double fx;
	double fy;
	double fz;
	double fnorm;
	double load;
	size_t nactive;CUDA_EXPORT
	kick_return() {
		max_rung = 0;
		part_flops = 0.0;
		node_flops = 0.0;
		pot = 0.0;
		fx = 0.0;
		fy = 0.0;
		fz = 0.0;
		fnorm = 0.0;
		nactive = 0;
		load = 0.0;
	}
	CUDA_EXPORT
	kick_return& operator+=(const kick_return& other) {
		if (other.max_rung > max_rung) {
			max_rung = other.max_rung;
		}
		part_flops += other.part_flops;
		node_flops += other.node_flops;
		pot += other.pot;
		fx += other.fx;
		fy += other.fy;
		fz += other.fz;
		fnorm += other.fnorm;
		return *this;

	}
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & load;
		arc & max_rung;
		arc & part_flops;
		arc & node_flops;
		arc & pot;
		arc & fx;
		arc & fy;
		arc & fz;
		arc & fnorm;
		arc & nactive;
	}
};

struct kick_params {
	int min_level;
	int min_rung;
	float a;
	float t0;
	float theta;
	float h;
	float eta;
	float GM;
	float dm_mass;
	float sph_mass;
	bool save_force;
	bool first_call;
	bool gpu;
	float node_load;
	kick_params() {
		dm_mass = get_options().dm_mass;
		sph_mass = get_options().sph_mass;
	}
	template<class A>
	void serialize(A && arc, unsigned) {
		arc & dm_mass;
		arc & sph_mass;
		arc & min_rung;
		arc & gpu;
		arc & a;
		arc & t0;
		arc & theta;
		arc & first_call;
		arc & h;
		arc & eta;
		arc & GM;
		arc & save_force;
		arc & node_load;
	}
};

struct kick_workitem {
	expansion<float> L;
	array<fixed32, NDIM> pos;
	tree_id self;
	vector<tree_id> dchecklist;
	vector<tree_id> echecklist;
};

struct kick_workspace;

#ifndef __CUDACC__
hpx::future<kick_return> kick(kick_params, expansion<float> L, array<fixed32, NDIM> pos, tree_id self, vector<tree_id> dchecklist, vector<tree_id> echecklist,
		std::shared_ptr<kick_workspace>);
#endif
void kick_show_timings();
#ifdef USE_CUDA
vector<kick_return> cuda_execute_kicks(kick_params params, fixed32*, fixed32*, fixed32*, float*, char*, tree_node*, vector<kick_workitem> workitems, cudaStream_t stream,
		int part_count, int ntrees, std::function<void()>, std::function<void()>);
#endif
int kick_block_count();
size_t kick_estimate_cuda_mem_usage(double theta, int nparts, int check_count);

#endif /* KICK_HPP_ */
