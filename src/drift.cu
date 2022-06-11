/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2022  Dominic C. Marcello

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

#include <cosmictiger/defs.hpp>
#include <cosmictiger/drift.hpp>
#include <cosmictiger/fixed.hpp>
#include <cosmictiger/particles.hpp>

__global__ void cuda_drift_kernel(fixed32* const __restrict__ x, fixed32* const __restrict__ y, fixed32* const __restrict__ z,
		const array<float, NDIM>* const vels, const char* const rungs, part_int count, char rung, float a, float dt) {
	const auto& tid = threadIdx.x;
	const auto& block_size = blockDim.x;
	const auto& bid = blockIdx.x;
	const auto& nblocks = gridDim.x;
	const part_int begin = (size_t) bid * count / nblocks;
	const part_int end = (size_t)(bid + 1) * count / nblocks;
	const float float2fixed = 1.f / fixed2float;
	const float float2fixeddtainv = float2fixed * dt / a;
	for (part_int i = begin + tid; i < end; i += block_size) {
		const int this_rung = rungs[i];
		if (this_rung == rung) {
			const float* const vel= vels[i].data();
			x[i].raw() += (unsigned) ((int)roundf(vel[XDIM] * float2fixeddtainv));
			y[i].raw() += (unsigned) ((int)roundf(vel[YDIM] * float2fixeddtainv));
			z[i].raw() += (unsigned) ((int)roundf(vel[ZDIM] * float2fixeddtainv));
		}
	}
}

void cuda_drift(char rung, float a, float dt) {
	int nblocks;
	int nthreads;
	cudaFuncAttributes attr;
	CUDA_CHECK(cudaFuncGetAttributes(&attr, (const void*) cuda_drift_kernel));
	nthreads = attr.maxThreadsPerBlock;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, (const void*) cuda_drift_kernel, nthreads, 0));
//	PRINT("Drifting with %i blocks per SM and %i threads per block\n", nblocks, nthreads);
	nblocks *= cuda_smp_count();
	const auto rng = particles_current_range();
	nblocks = std::min(nblocks, std::max((rng.second - rng.first) / nthreads, 1));
	const auto cnt = rng.second - rng.first;
	fixed32* x = &particles_pos(XDIM, rng.first);
	fixed32* y = &particles_pos(YDIM, rng.first);
	fixed32* z = &particles_pos(ZDIM, rng.first);
	const auto* vels = particles_vel_data() + rng.first;
	const auto* rungs = &particles_rung(rng.first);
	cuda_drift_kernel<<<nblocks,nthreads>>>(x, y, z, vels, rungs, cnt, rung, a, dt);
	CUDA_CHECK(cudaDeviceSynchronize());
}
