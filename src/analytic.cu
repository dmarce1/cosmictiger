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
#include <cosmictiger/cuda.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/options.hpp>

__global__ void analytic_gravity_kernel(fixed32* sinkx, fixed32* sinky, fixed32* sinkz, fixed32* sourcex, fixed32* sourcey, fixed32* sourcez, int Nsource,
		double* rphi, double* rgx, double* rgy, double*rgz, float hsoft);

#define ANALYTIC_BLOCK_SIZE 256

std::pair<vector<double>, array<vector<double>, NDIM>> gravity_analytic_call_kernel(const vector<fixed32>& sinkx, const vector<fixed32>& sinky,
		const vector<fixed32>& sinkz) {
	cuda_set_device();
	std::pair<vector<double>, array<vector<double>, NDIM>> rc;
	fixed32* dev_sinkx;
	fixed32* dev_sinky;
	fixed32* dev_sinkz;
	fixed32* dev_srcx;
	fixed32* dev_srcy;
	fixed32* dev_srcz;
	double* dev_phi;
	double* dev_gx;
	double* dev_gy;
	double* dev_gz;
	const int Nsinks = sinkx.size();
	(CUDA_MALLOC(&dev_sinkx, Nsinks * sizeof(fixed32)));
	(CUDA_MALLOC(&dev_sinky, Nsinks * sizeof(fixed32)));
	(CUDA_MALLOC(&dev_sinkz, Nsinks * sizeof(fixed32)));
	(CUDA_MALLOC(&dev_phi, Nsinks * sizeof(double)));
	(CUDA_MALLOC(&dev_gx, Nsinks * sizeof(double)));
	(CUDA_MALLOC(&dev_gy, Nsinks * sizeof(double)));
	(CUDA_MALLOC(&dev_gz, Nsinks * sizeof(double)));
	vector<double> zero(Nsinks, 0.0);
	if (hpx_rank() == 0) {
		vector<double> self_phi1(Nsinks, 0.0);
		CUDA_CHECK(cudaMemcpy(dev_phi, self_phi1.data(), Nsinks * sizeof(double), cudaMemcpyHostToDevice));
	} else {
		CUDA_CHECK(cudaMemcpy(dev_phi, zero.data(), Nsinks * sizeof(double), cudaMemcpyHostToDevice));
	}
	CUDA_CHECK(cudaMemcpy(dev_gx, zero.data(), Nsinks * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dev_gy, zero.data(), Nsinks * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dev_gz, zero.data(), Nsinks * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dev_sinkx, sinkx.data(), Nsinks * sizeof(fixed32), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dev_sinky, sinky.data(), Nsinks * sizeof(fixed32), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dev_sinkz, sinkz.data(), Nsinks * sizeof(fixed32), cudaMemcpyHostToDevice));
	PRINT("%li free\n", cuda_free_mem());
	const size_t parts_per_loop = (size_t) cuda_free_mem() / (NDIM * sizeof(fixed32)) * 85 / 100;
	int occupancy;
	CUDA_CHECK(
			cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, analytic_gravity_kernel, ANALYTIC_BLOCK_SIZE,
					sizeof(double) * (NDIM + 1) * ANALYTIC_BLOCK_SIZE));
	int num_kernels = std::max((int) (occupancy * cuda_smp_count() / Nsinks), 1);
	vector<cudaStream_t> streams(num_kernels);
	for (int i = 0; i < num_kernels; i++) {
		cudaStreamCreate(&streams[i]);
	}
	PRINT("%li particles per loop, %li kernels\n", parts_per_loop, num_kernels);
	for (size_t i = 0; i < particles_size(); i += parts_per_loop) {
		const int total_size = std::min(size_t(particles_size()), size_t(i) + size_t(parts_per_loop)) - size_t(i);
		(CUDA_MALLOC(&dev_srcx, total_size * sizeof(fixed32)));
		(CUDA_MALLOC(&dev_srcy, total_size * sizeof(fixed32)));
		(CUDA_MALLOC(&dev_srcz, total_size * sizeof(fixed32)));
		CUDA_CHECK(cudaMemcpy(dev_srcx, &particles_pos(0, i), total_size * sizeof(fixed32), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(dev_srcy, &particles_pos(1, i), total_size * sizeof(fixed32), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(dev_srcz, &particles_pos(2, i), total_size * sizeof(fixed32), cudaMemcpyHostToDevice));
		for (int j = 0; j < num_kernels; j++) {
			const int begin = size_t(j) * size_t(total_size) / size_t(num_kernels);
			const int end = size_t(j + 1) * size_t(total_size) / size_t(num_kernels);
			analytic_gravity_kernel<<<Nsinks,ANALYTIC_BLOCK_SIZE,0,streams[i]>>>(dev_sinkx, dev_sinky, dev_sinkz, dev_srcx + begin, dev_srcy + begin,
					dev_srcz + begin, end - begin, dev_phi, dev_gx, dev_gy, dev_gz, get_options().hsoft);
		}
		for (int i = 0; i < num_kernels; i++) {
			cuda_stream_synchronize(streams[i]);
		}
		CUDA_CHECK(cudaFree(dev_srcx));
		CUDA_CHECK(cudaFree(dev_srcy));
		CUDA_CHECK(cudaFree(dev_srcz));
	}
	for (int i = 0; i < num_kernels; i++) {
		cudaStreamDestroy(streams[i]);
	}
	rc.first.resize(Nsinks);
	for (int dim = 0; dim < NDIM; dim++) {
		rc.second[dim].resize(Nsinks);
	}
	CUDA_CHECK(cudaMemcpy(rc.first.data(), dev_phi, sizeof(double) * Nsinks, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(rc.second[XDIM].data(), dev_gx, sizeof(double) * Nsinks, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(rc.second[YDIM].data(), dev_gy, sizeof(double) * Nsinks, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(rc.second[ZDIM].data(), dev_gz, sizeof(double) * Nsinks, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(dev_phi));
	CUDA_CHECK(cudaFree(dev_gx));
	CUDA_CHECK(cudaFree(dev_gy));
	CUDA_CHECK(cudaFree(dev_gz));
	CUDA_CHECK(cudaFree(dev_sinkx));
	CUDA_CHECK(cudaFree(dev_sinky));
	CUDA_CHECK(cudaFree(dev_sinkz));
	return rc;
}

__global__ void analytic_gravity_kernel(fixed32* sinkx, fixed32* sinky, fixed32* sinkz, fixed32* sourcex, fixed32* sourcey, fixed32* sourcez, int Nsource,
		double* rphi, double* rgx, double* rgy, double*rgz, float h) {

	__shared__ double phi[ANALYTIC_BLOCK_SIZE];
	__shared__ double gx[ANALYTIC_BLOCK_SIZE];
	__shared__ double gy[ANALYTIC_BLOCK_SIZE];
	__shared__ double gz[ANALYTIC_BLOCK_SIZE];

	const int& tid = threadIdx.x;
	const int& bid = blockIdx.x;
	const float alpha = 1.1f;
	const float rmax = 4.15 / alpha + 0.5;
	const float i2max = sqr(rmax + 0.5);
	const int imax = sqrt(i2max) + 0.999999;
	const float h2max = sqr(1.26 * alpha + 0.5);
	const int hmax = sqrt(h2max) + 0.999999;
	const fixed32 x = sinkx[bid];
	const fixed32 y = sinky[bid];
	const fixed32 z = sinkz[bid];
	const float cons1 = float(2.0f / sqrtf(M_PI));
	float h2 = h * h;
	float hinv = 1.0 / (h);
	float h2inv = 1.0 / h2;
	float h3inv = hinv * hinv * hinv;
	phi[tid] = gx[tid] = gy[tid] = gz[tid] = 0.0f;
	for (int sourcei = tid; sourcei < Nsource; sourcei += ANALYTIC_BLOCK_SIZE) {
		const float X = distance(x, sourcex[sourcei]);
		const float Y = distance(y, sourcey[sourcei]);
		const float Z = distance(z, sourcez[sourcei]);
		const float R2 = sqr(X, Y, Z);
		float fx = 0.f;
		float fy = 0.f;
		float fz = 0.f;
		float pot = 0.f;
		if (R2 > 0.f) {
			for (int xi = -imax; xi <= +imax; xi++) {
				for (int yi = -imax; yi <= +imax; yi++) {
					for (int zi = -imax; zi <= +imax; zi++) {
						const float dx = X - xi;
						const float dy = Y - yi;
						const float dz = Z - zi;
						const float r2 = sqr(dx, dy, dz);
						if (r2 < rmax) {
							const float r = sqrt(r2);
							const float rinv = 1.f / r;
							const float r2inv = rinv * rinv;
							const float r3inv = r2inv * rinv;
							float exp0;
							float erfc0;
							erfcexp(alpha * r, &erfc0, &exp0);
							const float expfactor = alpha * cons1 * r * exp0;
							const float d0 = -erfc0 * rinv;
							const float d1 = (expfactor + erfc0) * r3inv;
							pot += d0;
							fx -= dx * d1;
							fy -= dy * d1;
							fz -= dz * d1;
						}
					}
				}
			}
			phi[tid] += float(M_PI / sqr(alpha));
			for (int xi = -hmax; xi <= +hmax; xi++) {
				for (int yi = -hmax; yi <= +hmax; yi++) {
					for (int zi = -hmax; zi <= +hmax; zi++) {
						const float hx = xi;
						const float hy = yi;
						const float hz = zi;
						const float h2 = sqr(hx, hy, hz);
						if (h2 > 0.0f && h2 <= h2max) {
							const float hdotx = X * hx + Y * hy + Z * hz;
							const float omega = float(2.0 * M_PI) * hdotx;
							float c, s;
							sincosf(omega, &s, &c);
							const float c0 = -1.0f / h2 * expf(float(-M_PI * M_PI) * h2 / sqr(alpha)) * float(1.f / M_PI);
							const float c1 = -s * 2.0 * M_PI * c0;
							pot += c0 * c;
							fx -= c1 * hx;
							fy -= c1 * hy;
							fz -= c1 * hz;
						}
					}
				}
			}
		}
		if (R2 == 0.f) {
			phi[tid] += 2.837291f;
			//		phi[tid] -= 15.0f / 8.0f * hinv;
		} else if (R2 < h2) {
			const float q2 = R2;
			float rinv1 = rsqrt(R2);
			float rinv3 = sqr(rinv1) * rinv1;
			fx += X * rinv3;
			fy += Y * rinv3;
			fz += Z * rinv3;
			pot += rinv1;
			gsoft(rinv3, rinv1, q2, hinv, h2inv, h3inv, true);
			fx -= X * rinv3;
			fy -= Y * rinv3;
			fz -= Z * rinv3;
			pot -= rinv1;
		}
		phi[tid] += pot;
		gx[tid] += fx;
		gy[tid] += fy;
		gz[tid] += fz;
	}
	__syncthreads();
	for (int P = ANALYTIC_BLOCK_SIZE / 2; P >= 1; P /= 2) {
		if (tid < P) {
			gx[tid] += gx[tid + P];
			gy[tid] += gy[tid + P];
			gz[tid] += gz[tid + P];
			phi[tid] += phi[tid + P];
		}
		__syncthreads();
	}
	if (tid == 0) {
		atomicAdd(rphi + bid, phi[0]);
		atomicAdd(rgx + bid, gx[0]);
		atomicAdd(rgy + bid, gy[0]);
		atomicAdd(rgz + bid, gz[0]);
	}

}
