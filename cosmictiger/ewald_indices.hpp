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
#define EWALD_TABLE_SIZE 16

using ewald_type = double;

using ewald_table_t = array<array<array<array<float, EWALD_TABLE_SIZE>, EWALD_TABLE_SIZE>, EWALD_TABLE_SIZE>, NDIM + 1>;

struct ewald_const {
	CUDA_EXPORT
	static int nfour();CUDA_EXPORT
	static int nreal();
	static void init();
	static void init_gpu();CUDA_EXPORT
	static const array<ewald_type, NDIM>& real_index(int i);CUDA_EXPORT
	static const array<ewald_type, NDIM>& four_index(int i);CUDA_EXPORT
	static const tensor_trless_sym<ewald_type, PM_ORDER>& four_expansion(int i);CUDA_EXPORT
	static const tensor_sym<ewald_type, PM_ORDER> D0();
	CUDA_EXPORT void table_interp(float& pot, float& fx, float& fy, float& fz, float x, float y, float z, bool do_pot);
};

struct ewald_constants {
	ewald_table_t table;
	array<array<ewald_type, NDIM>, NREAL> real_indices;
	array<array<ewald_type, NDIM>, NFOUR> four_indices;
	array<tensor_trless_sym<ewald_type, PM_ORDER>, NFOUR> four_expanse;
	tensor_sym<ewald_type, PM_ORDER> D0;
};



double high_precision_ewald(const array<double, NDIM>& X);
