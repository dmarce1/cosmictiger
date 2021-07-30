/*
 * kick.hpp
 *
 *  Created on: Jul 19, 2021
 *      Author: dmarce1
 */

#ifndef KICK_HPP_
#define KICK_HPP_

#include <tigerfmm/cuda_vector.hpp>
#include <tigerfmm/stack_vector.hpp>
#include <tigerfmm/tree.hpp>

#include <atomic>

struct cuda_kick_data {
	tree_node* tree_nodes;
	fixed32* x;
	fixed32* y;
	fixed32* z;
	float* vx;
	float* vy;
	float* vz;
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
	array<fixed32, SINK_BUCKET_SIZE> sink_x;
	array<fixed32, SINK_BUCKET_SIZE> sink_y;
	array<fixed32, SINK_BUCKET_SIZE> sink_z;
	struct {
		array<fixed32, KICK_PP_MAX> x;
		array<fixed32, KICK_PP_MAX> y;
		array<fixed32, KICK_PP_MAX> z;
	}src;
	array<float, SINK_BUCKET_SIZE> gx;
	array<float, SINK_BUCKET_SIZE> gy;
	array<float, SINK_BUCKET_SIZE> gz;
	array<float, SINK_BUCKET_SIZE> phi;
	array<int,SINK_BUCKET_SIZE> active;
	array<char,SINK_BUCKET_SIZE> rungs;
	cuda_vector<int> nextlist;
	cuda_vector<int> multlist;
	cuda_vector<int> partlist;
	cuda_vector<int> leaflist;
	cuda_vector<expansion<float>> L;
	stack_vector<int> dchecks;
	stack_vector<int> echecks;
	cuda_vector<int> phase;
	cuda_vector<int> self;
	cuda_vector<kick_return> returns;
	cuda_vector<array<fixed32, NDIM>> Lpos;
};
#endif

struct kick_return {
	char max_rung;
	double flops;
	double pot;
	double fx;
	double fy;
	double fz;
	double fnorm;
	int nactive;CUDA_EXPORT
	kick_return() {
		max_rung = 0;
		flops = 0.0;
		pot = 0.0;
		fx = 0.0;
		fy = 0.0;
		fz = 0.0;
		fnorm = 0.0;
		nactive = 0;
	}
	CUDA_EXPORT
	kick_return& operator+=(const kick_return& other) {
		if (other.max_rung > max_rung) {
			max_rung = other.max_rung;
		}
		flops += other.flops;
		pot += other.pot;
		fx += other.fx;
		fy += other.fy;
		fz += other.fz;
		fnorm += other.fnorm;
		return *this;

	}
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & max_rung;
		arc & flops;
		arc & pot;
		arc & fx;
		arc & fy;
		arc & fz;
		arc & fnorm;
		arc & nactive;
	}
};

struct kick_params {
	int min_rung;
	float a;
	float t0;
	float theta;
	float h;
	float eta;
	float GM;
	bool save_force;
	bool first_call;
	template<class A>
	void serialize(A && arc, unsigned) {
		arc & min_rung;
		arc & a;
		arc & t0;
		arc & theta;
		arc & first_call;
		arc & h;
		arc & eta;
		arc & GM;
		arc & save_force;
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
vector<kick_return> cuda_execute_kicks(kick_params params, fixed32*, fixed32*, fixed32*, tree_node*, vector<kick_workitem> workitems, cudaStream_t stream,
		int part_count, int ntrees, std::atomic<int>&);
int kick_block_count();
void kick_reset_list_sizes();
void kick_reset_all_list_sizes();

#endif /* KICK_HPP_ */
