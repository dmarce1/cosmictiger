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

#define PERSISTENT_KILL 0
#define PERSISTENT_DO_KICK 1
#define PERSISTENT_KICK_DONE 2

#include <cosmictiger/fmm_kernels.hpp>
#include <cosmictiger/fixed.hpp>
#include <cosmictiger/device_vector.hpp>
#include <cosmictiger/tree.hpp>

struct persistent_msg {
	int type;
	void* data;
};


struct persistent_kick_params {
	array<fixed32, NDIM> Lpos;
	expansion<float> L;
	int self;
	device_vector<int> dchecks;
	device_vector<int> echecks;
};
void persistent_do_kick(array<fixed32, NDIM> Lpos, expansion<float> L, int self, const vector<tree_id>& dchecks,const vector<tree_id>& echecks);
void persistent_kernel_launch();
void persistent_kernel_terminate();
void persistent_init();
