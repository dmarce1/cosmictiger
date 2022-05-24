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

#include <cosmictiger/stack_vector.hpp>
#include <cosmictiger/tree.hpp>

#include <atomic>

struct cuda_kick_data {
	tree_node* tree_nodes;
	fixed32* x;
	fixed32* y;
	fixed32* z;
	float* vx;
	float* vy;
	float* vz;
	fixed32* x_snk;
	fixed32* y_snk;
	fixed32* z_snk;
	char* rungs;
	float* gx;
	float* gy;
	float* gz;
	float* pot;
};

struct kick_return;

#ifdef __CUDACC__

struct cuda_kick_shmem {
	array<fixed32, KICK_PP_MAX> src_x;
	array<fixed32, KICK_PP_MAX> src_y;
	array<fixed32, KICK_PP_MAX> src_z;
	array<fixed32, MAX_BUCKET_SIZE> sink_x;
	array<fixed32, MAX_BUCKET_SIZE> sink_y;
	array<fixed32, MAX_BUCKET_SIZE> sink_z;
	device_vector<float> gx;
	device_vector<float> gy;
	device_vector<float> gz;
	device_vector<float> phi;
	stack_vector<int> echecks;
	stack_vector<int> dchecks;
	device_vector<int> leaflist;
	device_vector<int> cplist;
	device_vector<int> cclist;
	device_vector<int> pclist;
	device_vector<expansion<float>> L;
	device_vector<int> nextlist;
	device_vector<kick_return> returns;
	device_vector<array<fixed32, NDIM>> Lpos;
	device_vector<int> phase;
	device_vector<int> self;

};
#endif

struct kick_return {
	char max_rung;
	double part_flops;
	double node_flops;
	double pot;
	double kin;
	double load;
	double xmom;
	double ymom;
	double zmom;
	double nmom;
	double total_time;
	double parts_processed;
	size_t nactive;

	CUDA_EXPORT
	kick_return() {
		max_rung = 0;
		part_flops = 0.0;
		node_flops = 0.0;
		pot = 0.0;
		kin = 0.0;
		xmom = ymom = zmom = nmom = 0.0;
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
		kin += other.kin;
		xmom += other.xmom;
		ymom += other.ymom;
		zmom += other.zmom;
		nmom += other.nmom;
		return *this;

	}
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & load;
		arc & max_rung;
		arc & part_flops;
		arc & node_flops;
		arc & pot;
		arc & kin;
		arc & xmom;
		arc & ymom;
		arc & zmom;
		arc & nmom;
		arc & nactive;
		arc & total_time;
		arc & parts_processed;

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
	bool save_force;
	bool first_call;
	bool gpu;
	float node_load;
	float max_dt;
	bool ascending;
	bool descending;
	bool top;
	bool do_phi;
	kick_params() {
		max_dt = 1e30;
		do_phi = true;
	}
	template<class A>
	void serialize(A && arc, unsigned) {
		arc & ascending;
		arc & descending;
		arc & top;
		arc & max_dt;
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
		arc & min_level;
		arc & do_phi;
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
vector<kick_return> cuda_execute_kicks(kick_params params, fixed32*, fixed32*, fixed32*, tree_node*, vector<kick_workitem> workitems, cudaStream_t stream);
#endif
int kick_block_count();
size_t kick_estimate_cuda_mem_usage(double theta, int nparts, int check_count);

#endif /* KICK_HPP_ */
