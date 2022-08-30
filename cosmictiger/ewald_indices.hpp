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

#pragma once

#include <cosmictiger/containers.hpp>
#include <cosmictiger/cuda.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/tensor.hpp>


#define NREAL 178
#define NFOUR 92


struct ewald_const {
	CUDA_EXPORT static int nfour();
	CUDA_EXPORT static int nreal();
	static void init();
	static void init_gpu();
	CUDA_EXPORT static const array<float,NDIM>& real_index(int i);
	CUDA_EXPORT static const array<float,NDIM>& four_index(int i);
	CUDA_EXPORT static const tensor_trless_sym<float,LORDER>& four_expansion(int i);
	CUDA_EXPORT static const tensor_sym<float,LORDER> D0();
};


struct ewald_constants {
	array<array<float, NDIM>, NREAL> real_indices;
	array<array<float, NDIM>, NFOUR> four_indices;
	array<tensor_trless_sym<float, LORDER>, NFOUR> four_expanse;
	tensor_sym<float,LORDER> D0;
};
