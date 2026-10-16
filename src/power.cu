/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

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

#include <cosmictiger/cuda_mem.hpp>
#include <cosmictiger/domain.hpp>
#include <cosmictiger/fft.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/power.hpp>

#define BLOCK_SIZE 256

__global__ void accumulate_density_kernel(fixed32* X, fixed32* Y, fixed32* Z, float* rho, float const* masses, int Mfold, size_t Nparts, int Ndim,
										  range<int64_t> intbox) {
	int const tid = threadIdx.x;
	int const bid = blockIdx.x;
	int const gsz = gridDim.x;
	part_int const ibegin = (size_t)bid * Nparts / (size_t)gsz;
	part_int const iend = (size_t)(bid + 1) * Nparts / (size_t)gsz;
	for (part_int i = ibegin + tid; i < iend; i += BLOCK_SIZE) {
		ALWAYS_ASSERT(i < Nparts);
		float x = fmodf(Mfold * X[i].to_float(), 1.0f);
		float y = fmodf(Mfold * Y[i].to_float(), 1.0f);
		float z = fmodf(Mfold * Z[i].to_float(), 1.0f);
		int const j = (x * Ndim);
		if (j >= intbox.begin[XDIM] + 1 && j < intbox.end[XDIM] - 1) {
			int const k = (y * Ndim);
			if (k >= intbox.begin[YDIM] + 1 && k < intbox.end[YDIM] - 1) {
				int const l = (z * Ndim);
				if (l >= intbox.begin[ZDIM] + 1 && l < intbox.end[ZDIM] - 1) {
					float const x0 = x * Ndim - j;
					float const y0 = y * Ndim - k;
					float const z0 = z * Ndim - l;
					for (int j0 = CLOUD_MIN; j0 <= CLOUD_MAX; j0++) {
						float wt_x = cloud_weight(x0 - j0);
						for (int k0 = CLOUD_MIN; k0 <= CLOUD_MAX; k0++) {
							float wt_y = cloud_weight(y0 - k0);
							for (int l0 = CLOUD_MIN; l0 <= CLOUD_MAX; l0++) {
								float wt_z = cloud_weight(z0 - l0);
								float const mass = masses == nullptr ? 1.0f : masses[i];
								float const wt = wt_x * wt_y * wt_z * mass;
								atomicAdd(rho + intbox.index(j0 + j, k0 + k, l0 + l), wt);
							}
						}
					}
				}
			}
		}
	}
}

vector<float> accumulate_density_cuda(int M, int Ndim, range<int64_t> intbox) {
	int nblocks;
	auto const N3 = intbox.volume();
	vector<float> rho(N3, 0.0f);
	if (particles_size() == 0) {
		return rho;
	}
	float* dev_rho;
	float* masses = nullptr;
	if (get_options().sph) {
		masses = static_cast<float*>(cuda_malloc(particles_size() * sizeof(float)));
		for (part_int i = 0; i < particles_size(); i++) {
			masses[i] = particles_mass(i);
		}
	}
	CUDA_CHECK(cudaMallocManaged(&dev_rho, N3 * sizeof(float)));
	for (int i = 0; i < N3; i++) {
		dev_rho[i] = 0.f;
	}
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, (void const*)accumulate_density_kernel, BLOCK_SIZE, 0));
	nblocks *= cuda_smp_count();
	accumulate_density_kernel<<<nblocks, BLOCK_SIZE>>>(&particles_pos(XDIM, 0), &particles_pos(YDIM, 0), &particles_pos(ZDIM, 0), dev_rho, masses, M,
													   particles_size(), Ndim, intbox);
	CUDA_CHECK(cudaDeviceSynchronize());
	memcpy(rho.data(), dev_rho, sizeof(float) * N3);
	CUDA_CHECK(cudaFree(dev_rho));
	if (masses != nullptr) {
		cuda_free(masses);
	}
	return rho;
}
