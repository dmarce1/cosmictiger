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

struct sph_run_cuda_data {
	fixed32* x;
	fixed32* y;
	fixed32* z;
	char* rungs;
	float* h_snk;
	int nselfs;
	sph_tree_node* trees;
	int* selfs;
	int* neighbors;
};


sph_run_return sph_run_cuda(sph_run_params params, sph_run_cuda_data data, cudaStream_t stream);

#endif /* SPH_CUDA_HPP_ */
