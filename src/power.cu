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
	const part_int iend = (size_t)(bid + 1) * Nparts / (size_t) gsz;
	for (part_int i = ibegin + tid; i < iend; i += BLOCK_SIZE) {
		ALWAYS_ASSERT(i < Nparts);
		double x = fmod(Mfold * X[i].to_double(), 1.0);
		double y = fmod(Mfold * Y[i].to_double(), 1.0);
		double z = fmod(Mfold * Z[i].to_double(), 1.0);
		const int j = (x * Ndim);
		if (j >= intbox.begin[XDIM] + 1 && j < intbox.end[XDIM] - 1) {
			const int k = (y * Ndim);
			if (k >= intbox.begin[YDIM] + 1 && k < intbox.end[YDIM] - 1) {
				const int l = (z * Ndim);
				if (l >= intbox.begin[ZDIM] + 1 && l < intbox.end[ZDIM] - 1) {
					const int jp = j + 1;
					const int kp = k + 1;
					const int lp = l + 1;
					const int jm = j - 1;
					const int km = k - 1;
					const int lm = l - 1;
					const double x0 = x * Ndim - j - 0.5;
					const double y0 = y * Ndim - k - 0.5;
					const double z0 = z * Ndim - l - 0.5;
					const double wx0 = 0.25 * (3.0 - 4.0 * sqr(x0));
					const double wy0 = 0.25 * (3.0 - 4.0 * sqr(y0));
					const double wz0 = 0.25 * (3.0 - 4.0 * sqr(z0));
					const double wxp = 0.125 * (9.0 - 12.0 * (-x0 + 1.0) + 4.0 * sqr(-x0 + 1.0));
					const double wyp = 0.125 * (9.0 - 12.0 * (-y0 + 1.0) + 4.0 * sqr(-y0 + 1.0));
					const double wzp = 0.125 * (9.0 - 12.0 * (-z0 + 1.0) + 4.0 * sqr(-z0 + 1.0));
					const double wxm = 1.0 - wxp - wx0;
					const double wym = 1.0 - wyp - wy0;
					const double wzm = 1.0 - wzp - wz0;
					const double w000 = wx0 * wy0 * wz0;
					const double w00p = wx0 * wy0 * wzp;
					const double w00m = wx0 * wy0 * wzm;
					const double w0p0 = wx0 * wyp * wz0;
					const double w0pp = wx0 * wyp * wzp;
					const double w0pm = wx0 * wyp * wzm;
					const double w0m0 = wx0 * wym * wz0;
					const double w0mp = wx0 * wym * wzp;
					const double w0mm = wx0 * wym * wzm;
					const double wp00 = wxp * wy0 * wz0;
					const double wp0p = wxp * wy0 * wzp;
					const double wp0m = wxp * wy0 * wzm;
					const double wpp0 = wxp * wyp * wz0;
					const double wppp = wxp * wyp * wzp;
					const double wppm = wxp * wyp * wzm;
					const double wpm0 = wxp * wym * wz0;
					const double wpmp = wxp * wym * wzp;
					const double wpmm = wxp * wym * wzm;
					const double wm00 = wxm * wy0 * wz0;
					const double wm0p = wxm * wy0 * wzp;
					const double wm0m = wxm * wy0 * wzm;
					const double wmp0 = wxm * wyp * wz0;
					const double wmpp = wxm * wyp * wzp;
					const double wmpm = wxm * wyp * wzm;
					const double wmm0 = wxm * wym * wz0;
					const double wmmp = wxm * wym * wzp;
					const double wmmm = wxm * wym * wzm;
					const int& j0 = j;
					const int& k0 = k;
					const int& l0 = l;
					atomicAdd(rho + intbox.index(j0, k0, l0), w000);
					atomicAdd(rho + intbox.index(j0, k0, lp), w00p);
					atomicAdd(rho + intbox.index(j0, k0, lm), w00m);
					atomicAdd(rho + intbox.index(j0, km, l0), w0m0);
					atomicAdd(rho + intbox.index(j0, km, lp), w0mp);
					atomicAdd(rho + intbox.index(j0, km, lm), w0mm);
					atomicAdd(rho + intbox.index(j0, kp, l0), w0p0);
					atomicAdd(rho + intbox.index(j0, kp, lp), w0pp);
					atomicAdd(rho + intbox.index(j0, kp, lm), w0pm);
					atomicAdd(rho + intbox.index(jp, k0, l0), wp00);
					atomicAdd(rho + intbox.index(jp, k0, lp), wp0p);
					atomicAdd(rho + intbox.index(jp, k0, lm), wp0m);
					atomicAdd(rho + intbox.index(jp, km, l0), wpm0);
					atomicAdd(rho + intbox.index(jp, km, lp), wpmp);
					atomicAdd(rho + intbox.index(jp, km, lm), wpmm);
					atomicAdd(rho + intbox.index(jp, kp, l0), wpp0);
					atomicAdd(rho + intbox.index(jp, kp, lp), wppp);
					atomicAdd(rho + intbox.index(jp, kp, lm), wppm);
					atomicAdd(rho + intbox.index(jm, k0, l0), wm00);
					atomicAdd(rho + intbox.index(jm, k0, lp), wm0p);
					atomicAdd(rho + intbox.index(jm, k0, lm), wm0m);
					atomicAdd(rho + intbox.index(jm, km, l0), wmm0);
					atomicAdd(rho + intbox.index(jm, km, lp), wmmp);
					atomicAdd(rho + intbox.index(jm, km, lm), wmmm);
					atomicAdd(rho + intbox.index(jm, kp, l0), wmp0);
					atomicAdd(rho + intbox.index(jm, kp, lp), wmpp);
					atomicAdd(rho + intbox.index(jm, kp, lm), wmpm);
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
	CUDA_CHECK(cudaMallocManaged(&dev_rho, N3 * sizeof(float)));
	for (int i = 0; i < N3; i++) {
		dev_rho[i] = 0.f;
	}
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblocks, (const void*) accumulate_density_kernel, BLOCK_SIZE, 0));
	nblocks *= cuda_smp_count();
	accumulate_density_kernel<<<nblocks, BLOCK_SIZE>>>(&particles_pos(XDIM, 0), &particles_pos(YDIM, 0), &particles_pos(ZDIM, 0), dev_rho, M, particles_size(), Ndim, intbox);
	CUDA_CHECK(cudaDeviceSynchronize());
	memcpy(rho.data(), dev_rho, sizeof(float) * N3);
	CUDA_CHECK(cudaFree(dev_rho));
	return rho;
}
