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
#include <cosmictiger/kick.hpp>

#define QSIZE (1024*1024)

using q_type = lockfree_queue<persistent_msg, QSIZE>;

static __managed__ q_type* cpu2gpuq;
static __managed__ int kicks_in_progress;

persistent_msg* msg;

void persistent_init() {
	kicks_in_progress = 0;
	cpu2gpuq = (q_type*) cuda_malloc(sizeof(q_type));
}

void persistent_do_kick(array<fixed32, NDIM> Lpos, expansion<float> L, int self, const vector<tree_id>& dchecks, const vector<tree_id>& echecks) {
	auto* msg = (persistent_msg*) cuda_malloc(sizeof(persistent_msg));
	auto* kick_params = (persistent_kick_params*) cuda_malloc(sizeof(persistent_kick_params));
	new (kick_params) persistent_kick_params;
	msg->type = PERSISTENT_DO_KICK;
	msg->data = kick_params;
	kick_params->Lpos = Lpos;
	kick_params->L = L;
	kick_params->self = self;
	kick_params->dchecks.resize(dchecks.size());
	kick_params->echecks.resize(echecks.size());
	for (int i = 0; i < dchecks.size(); i++) {
		kick_params->dchecks[i] = dchecks[i].index;
	}
	for (int i = 0; i < echecks.size(); i++) {
		kick_params->echecks[i] = echecks[i].index;
	}
	atomic_add(&kicks_in_progress, 1);
	cpu2gpuq->push(msg);
}

static int nblocks;
static __managed__ kick_return* global_kick_return;

__global__ void persistent_kernel() {

	const int& tid = threadIdx.x;
	__shared__ persistent_msg* msg;
	extern __shared__ int shmem_ptr[];
	cuda_kick_shmem& shmem = *(cuda_kick_shmem*) shmem_ptr;
	new (shmem_ptr) cuda_kick_shmem;
	auto& barrier = shmem.barrier;
	auto group = cooperative_groups::this_thread_block();
	if (group.thread_rank() == 0) {
		init(&barrier, group.size());
	}
	group.sync();
	kick_return kr;
	while (1) {
		if (tid == 0) {
			do {
				msg = cpu2gpuq->pop();
			} while (msg == nullptr);
		}
		__syncthreads();
		if (msg->type == PERSISTENT_KILL) {
			if (tid == 0) {
				cuda_free(msg);
			}
			__syncthreads();
			break;
		} else if (msg->type == PERSISTENT_DO_KICK) {
			persistent_kick_params* params_ptr = (persistent_kick_params*) msg->data;
			__syncthreads();
			kr += cuda_kick(*params_ptr);
			params_ptr->~persistent_kick_params();
			if (tid == 0) {
				cuda_free(msg);
				cuda_free(params_ptr);
				persistent_msg* msg = (persistent_msg*) cuda_malloc(sizeof(persistent_msg));
				msg->type = PERSISTENT_KICK_DONE;
				atomic_add(&kicks_in_progress, -1);
			}
			__syncthreads();
		} else {
			ALWAYS_ASSERT(false);
		}
	}

	if (tid == 0) {
		atomicAdd(&global_kick_return->kin, kr.kin);
		atomicAdd(&global_kick_return->pot, kr.pot);
		atomicAdd(&global_kick_return->xmom, kr.xmom);
		atomicAdd(&global_kick_return->ymom, kr.ymom);
		atomicAdd(&global_kick_return->zmom, kr.zmom);
		atomicAdd(&global_kick_return->nmom, kr.nmom);
		atomicMax(&global_kick_return->max_rung, kr.max_rung);
	}
	((cuda_kick_shmem*) shmem_ptr)->~cuda_kick_shmem();

}

void persistent_kernel_launch() {

	cuda_set_device();

	global_kick_return = (kick_return*) cuda_malloc(sizeof(kick_return));
	new (global_kick_return) kick_return;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, (const void*) persistent_kernel, WARP_SIZE, sizeof(cuda_kick_shmem)));
	nblocks *= cuda_smp_count();
	PRINT("---nblocks = %i\n", nblocks);
	persistent_kernel<<<nblocks,WARP_SIZE,sizeof(cuda_kick_shmem)>>>();
}

void persistent_kernel_terminate() {
	cuda_set_device();
	while ((int) kicks_in_progress) {
		usleep(1000);
	}
	for (int i = 0; i < nblocks; i++) {
		auto* msg = (persistent_msg*) cuda_malloc(sizeof(persistent_msg));
		msg->type = PERSISTENT_KILL;
		cpu2gpuq->push(msg);
	}
	CUDA_CHECK(cudaDeviceSynchronize());
	kick_set_rc(*global_kick_return);
	PRINT( "%i\n", global_kick_return->max_rung);
	global_kick_return->~kick_return();
	cuda_free(global_kick_return);
	PRINT( "Kernel terminated\n");
}
