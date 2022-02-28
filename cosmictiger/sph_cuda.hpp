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

#ifndef SPH_CUDA_HPP_
#define SPH_CUDA_HPP_

#include <cosmictiger/defs.hpp>
#include <cosmictiger/sph.hpp>
#include <cosmictiger/sph_tree.hpp>
#include <cosmictiger/chemistry.hpp>
#include <cosmictiger/sph_particles.hpp>

struct sph_run_cuda_data {
	fixed32* x;
	fixed32* y;
	fixed32* z;
	float def_gamma;
	bool conduction;
	bool gravity;
	bool chem;
	float* fpot;
	float gcentral;
	float hcentral;
	float* fpot_snk;
	float* mmw;
	float hsoft_min;
	float G;
	float t0;
	float kappa0;
	float rho0_c;
	float h0;
	float* alpha;
	float rho0_b;
//	float* Z;
//	float* Y;
	float Y0;
	float* T;
	float* kappa;
	float* colog;
	float* lambda_e;
	float* gx;
	float* gy;
	char* oldrung;
	float* gz;
	float* vx;
	float* vy;
	float* vz;
	float* eint;
	float* f0;
	float* gamma;
	float* fvel;
	float* h;
	char* rungs;
	float* h_snk;
	float* tcool_snk;
	float* tdyn_snk;
	dif_vector* vec0_snk;
	float* eint_snk;
//	float* dchem_snk;
	float* difco;
	float* difco_snk;
	dif_vector* dif_vec;
	float* deint_con;
	float* dvx_con;
	float* dvy_con;
	float* dvz_con;
	float* deint_pred;
	dif_vector* dvec_snk;
	float* dvx_pred;
	float* kappa_snk;
	float* dvy_pred;
	float* dvz_pred;
	float* gx_snk;
	float* gy_snk;
	float* gz_snk;
	float* alpha_snk;
	float* f0_snk;
	float* fvel_snk;
	float* Z_snk;
	float code_dif_to_cgs;
	float hstar0;
	float N;
	char* sa_snk;
//	float* Yform_snk;
//	float* Zform_snk;
	float* divv_snk;
	int nselfs;
	sph_tree_node* trees;
	int* selfs;
	int* neighbors;
	float m;
	float eta;
};

sph_run_return sph_run_cuda(sph_run_params params, sph_run_cuda_data data, cudaStream_t stream);

#endif /* SPH_CUDA_HPP_ */
