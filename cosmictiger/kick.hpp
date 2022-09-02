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

#include <cosmictiger/tree.hpp>

#include <atomic>

#define KICK_C_MAX 32

struct cuda_kick_data {
	tree_node* tree_nodes;
	fixed32* x;
	fixed32* y;
	fixed32* z;
	array<float, NDIM>* vel;
	char* rungs;
	float* gx;
	float* gy;
	float* gz;
	float* pot;
	float rs;
};

struct kick_return;

#ifdef __CUDACC__

#include <cosmictiger/stack_vector.hpp>

#define KICK_PP_MAX (32*29)

struct expansion_type {
	array<fixed32, NDIM> pos;
	expansion<float> expansion;
};

struct search_params {
	int self;
	int phase;
};

struct force_type {
	float gx;
	float gy;
	float gz;
	float phi;
};

struct cuda_kick_shmem {
	union {
		struct {
			array<fixed32, KICK_PP_MAX> x;
			array<fixed32, KICK_PP_MAX> y;
			array<fixed32, KICK_PP_MAX> z;
		}X;
		array<multi_pos, KICK_C_MAX> mpos;
	};
	device_vector<force_type> f;
	stack_vector<int> echecks;
	stack_vector<int> dchecks;
	device_vector<int> leaflist;
	device_vector<int> closelist;
	device_vector<int> cplist;
	device_vector<int> cclist;
	device_vector<int> pclist;
	device_vector<int> nextlist;
	device_vector<expansion_type> L;
	device_vector<search_params> params;
	barrier_type barrier;

};
#endif

struct kick_return {
	int max_rung;
	double pot;
	double kin;
	double dkin;
	double xmom;
	double ymom;
	double zmom;
	double nmom;
	double err;
	CUDA_EXPORT
	kick_return() {
		max_rung = 0;
		pot = 0.0;
		kin = 0.0;
		xmom = ymom = zmom = nmom = 0.0;
		dkin = 0.0;
	}
	CUDA_EXPORT
	kick_return& operator+=(const kick_return& other) {
		if (other.max_rung > max_rung) {
			max_rung = other.max_rung;
		}
		pot += other.pot;
		kin += other.kin;
		xmom += other.xmom;
		ymom += other.ymom;
		zmom += other.zmom;
		nmom += other.nmom;
		dkin += other.dkin;
		return *this;

	}
	template<class A>
	void serialize(A&& arc, unsigned) {
		arc & max_rung;
		arc & pot;
		arc & kin;
		arc & xmom;
		arc & ymom;
		arc & zmom;
		arc & nmom;
		arc & dkin;
	}
};

struct kick_params {
	float sign;
	int min_rung;
	float a;
	float t0;
	float theta;
	float h;
	float eta;
	float GM;
	bool vel_init;
	bool save_force;
	bool first_call;
	bool gpu;
	float node_load;
	float max_dt;
	bool ascending;
	bool descending;
	bool top;
	bool do_phi;
#ifdef TREEPM
	float rs;
#endif
	kick_params() {
		max_dt = 1e30;
		do_phi = true;
		vel_init = false;
		sign = 1.0;
	}
	template<class A>
	void serialize(A && arc, unsigned) {
		arc & rs;
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
		arc & sign;
		arc & save_force;
		arc & node_load;
		arc & do_phi;
		arc & vel_init;
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
kick_return kick(kick_params, expansion<float> L, array<fixed32, NDIM> pos, tree_id self, vector<tree_id> dchecklist, vector<tree_id> echecklist,
		std::shared_ptr<kick_workspace>);
#endif
void kick_show_timings();
void kick_set_rc(kick_return kr);
#ifdef USE_CUDA
kick_return cuda_execute_kicks(kick_params params, fixed32*, fixed32*, fixed32*, tree_node*, vector<kick_workitem> workitems);
#endif
int kick_block_count();
size_t kick_estimate_cuda_mem_usage(double theta, int nparts, int check_count);

#endif /* KICK_HPP_ */
