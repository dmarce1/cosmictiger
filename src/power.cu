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

#include <cosmictiger/domain.hpp>
#include <cosmictiger/fft.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/cuda_mem.hpp>

#include <cosmictiger/power.hpp>

#define BLOCK_SIZE 256

__global__ void accumulate_density_kernel(fixed32* X, fixed32* Y, fixed32* Z, float* rho, int Mfold, size_t Nparts, int Ndim, range<int64_t> intbox) {
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	const int gsz = gridDim.x;
	const part_int ibegin = (size_t) bid * Nparts / (size_t) gsz;
	const part_int iend = (size_t) (bid + 1) * Nparts / (size_t) gsz;
	for (int i = ibegin + tid; i < iend; i += BLOCK_SIZE) {
		double x = X[i].to_double();
		double y = Y[i].to_double();
		double z = Z[i].to_double();
		const int j = x * Ndim;
		if (j >= intbox.begin[XDIM] && j < intbox.end[XDIM]) {
			const int k = y * Ndim;
			if (k >= intbox.begin[YDIM] && k < intbox.end[YDIM]) {
				const int l = z * Ndim;
				if (l >= intbox.begin[ZDIM] && l < intbox.end[ZDIM]) {
					atomicAdd(rho + intbox.index(j, k, l), 1.0f);
				}
			}
		}
	}
}

vector<float> accumulate_density_cuda(int M, int Ndim, range<int64_t> intbox) {
	int nblocks;
	const auto N3 = intbox.volume();
	vector<float> rho(N3);
	float* dev_rho;
	dev_rho = (float*) cuda_malloc(N3 * sizeof(float));
	for( int i = 0; i < N3; i++) {
		dev_rho[i] = 0.f;
	}
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, (const void*) accumulate_density_kernel, BLOCK_SIZE, 0));
	nblocks *= cuda_smp_count();
	accumulate_density_kernel<<<nblocks, BLOCK_SIZE>>>(&particles_pos(0,XDIM), &particles_pos(0,YDIM), &particles_pos(0,ZDIM), dev_rho, M, particles_size(), Ndim, intbox);
	CUDA_CHECK(cudaDeviceSynchronize());
	memcpy(rho.data(), dev_rho, sizeof(float) * N3);
	cuda_free(dev_rho);
	return rho;
}
