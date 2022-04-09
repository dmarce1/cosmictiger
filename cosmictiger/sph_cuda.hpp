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
	float* cold_frac;
	float gsoft;
	float* dalpha;
	array<float, NCHEMFRACS>* chem;
	array<float, NCHEMFRACS>* dchem;
	float* divv;
	float* mmw;
	float* gradT;
	float* shearv;
	float* curlv_snk;
	float* divv0_snk;
	float* h_snk;
	float* entr_snk;
	float* gx_snk;
	float* gy_snk;
	float* gz_snk;
	float* dcold_mass;
	float* alpha_snk;
	char* oldrung_snk;
	float* divv_snk;
	float def_gamma;
	bool conduction;
	bool gravity;
	bool chemistry;
	char* converged_snk;
	float gcentral;
	float hcentral;
	float G;
	float t0;
	float kappa0;
	float rho0_c;
	float* alpha;
	float* alpha0;
	float rho0_b;
//	float* Z;
//	float* Y;
	float Y0;
	float* gx;
	float* gy;
	float* gz;
	float* vx;
	float* vy;
	float* vz;
	float* entr0;
	float* entr;
	float* fpre1;
	float* fpre2;
	float* pre;
	float* h;
	char* rungs;
	char* rungs_snk;
	sph_record1* rec1_snk;
	part_int* dm_index_snk;
	float* dentr;
	float* dentr_diss;
	float* dvx;
	float* dvy;
	float* dvz;
	float code_dif_to_cgs;
	float N;
	int nselfs;
	sph_tree_node* trees;
	int* selfs;
	int* neighbors;
	float m;
	float eta;
};

sph_run_return sph_run_cuda(sph_run_params params, sph_run_cuda_data data, cudaStream_t stream);

#endif /* SPH_CUDA_HPP_ */
