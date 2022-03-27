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
	array<float, NCHEMFRACS>* chem;
	array<float, NCHEMFRACS>* chem0;
	array<float, NCHEMFRACS>* dchem_snk;
	float* divv;
	float* mmw;
	float* gradT;
	float* shearv;
	float* fpot_snk;
	float* balsara_snk;
	char* oldrung_snk;
	float* gradT_snk;
	float* crsv_snk;
	float* shearv_snk;
	float* divv0_snk;
	float* xvx_snk;
	float* xvy_snk;
	float* xvz_snk;
	float* balsara;
	char* sa_snk;
	float* h_snk;
	float* eint_snk;
	float* gx_snk;
	float* gy_snk;
	float* gz_snk;
	float* alpha_snk;
	float* fpre_snk;
	float* divv_snk;
	float def_gamma;
	bool conduction;
	bool gravity;
	bool chemistry;
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
	float* eint0;
	float* eint;
	float* deint_snk;
	float* eavg_snk;
	float* fpre;
	float* gamma;
	float* h;
	char* rungs;
	float* deint_con;
	float* dvx_con;
	float* dvy_con;
	float* dvz_con;
	float* deint_pred;
	float* dvx_pred;
	float* dvy_pred;
	float* dvz_pred;
	float code_dif_to_cgs;
	float hstar0;
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
