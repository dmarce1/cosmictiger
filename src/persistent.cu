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

#include <cosmictiger/persistent.hpp>
#include <cosmictiger/lockfree_queue.hpp>

#define QSIZE (1024*1024)

using q_type = lockfree_queue<persistent_msg, QSIZE>;

static __managed__ q_type* cpu2gpuq;
static __managed__ q_type* gpu2cpuq;

persistent_msg* msg;

void persistent_init() {
	cpu2gpuq = (q_type*) cuda_malloc(sizeof(q_type));
	gpu2cpuq = (q_type*) cuda_malloc(sizeof(q_type));
}

void persistent_do_kick(array<fixed32, NDIM> Lpos, expansion<float> L, int self, device_vector<int> dchecks, device_vector<int> echecks) {
	auto* msg = (persistent_msg*) cuda_malloc(sizeof(persistent_msg));
	auto* kick_params = (persistent_kick_params*) cuda_malloc(sizeof(persistent_kick_params));
	new (kick_params) persistent_kick_params;
	msg->type = PERSISTENT_DO_KICK;
	msg->data = kick_params;
	kick_params->Lpos = Lpos;
	kick_params->L = L;
	kick_params->self = self;
	kick_params->dchecks = std::move(dchecks);
	kick_params->echecks = std::move(echecks);
	cpu2gpuq->push(msg);
}

__global__ void persistent_kernel() {
	const int& tid = threadIdx.x;
	__shared__ persistent_msg* msg;
	while (1) {
		if (tid == 0) {
			do {
				msg = cpu2gpuq->pop();
			} while (msg == nullptr);
		}
		if (msg->type == PERSISTENT_KILL) {
			cuda_free(msg);
			break;
		} else {

		}
	}
}
