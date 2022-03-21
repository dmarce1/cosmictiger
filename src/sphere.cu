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

#include <cosmictiger/math.hpp>
#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/cuda.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/cuda_mem.hpp>

#define alpha 2.0f
#define eta 0.25f

#define BLOCK_SIZE 512

__global__ void sphere_surface_kernel(float** x_main, float** y_main, float** z_main, int* N_main) {
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	float* x = x_main[bid];
	float* y = y_main[bid];
	float* z = z_main[bid];
	const int N = N_main[bid];
	constexpr unsigned a0 = 1103515245;
	constexpr unsigned c0 = 12345;
	constexpr unsigned m0 = 1UL << 31;
	unsigned rnd_num = tid;
	__shared__ device_vector<float> vx;
	__shared__ device_vector<float> vy;
	__shared__ device_vector<float> vz;
	__shared__ device_vector<float> gx;
	__shared__ device_vector<float> gy;
	__shared__ device_vector<float> gz;
	new(&vx) device_vector<float>(N);
	new(&vy) device_vector<float>(N);
	new(&vz) device_vector<float>(N);
	new(&gx) device_vector<float>(N);
	new(&gy) device_vector<float>(N);
	new(&gz) device_vector<float>(N);
	float gmax;
	const float h = 0.9f * sqrt(4.f * float(M_PI) / N);
	const float h2 = h * h;
	const float hinv = 1.f / h;
	const float h3inv = hinv * hinv * hinv;
	const float m = 1.0f / N;
	__syncthreads();
	const float m0inv = 1.f / m0;
	for (int i = tid; i < N; i += BLOCK_SIZE) {
		vx[i] = 0.f;
		vy[i] = 0.f;
		vz[i] = 0.f;
		float n;
		rnd_num = (a0 * rnd_num + c0) % m0;
		n = (rnd_num + 0.5f) * m0inv;
		x[i] = 2.f * n - 1.f;
		rnd_num = (a0 * rnd_num + c0) % m0;
		n = (rnd_num + 0.5f) * m0inv;
		y[i] = 2.f * n - 1.f;
		rnd_num = (a0 * rnd_num + c0) % m0;
		n = (rnd_num + 0.5f) * m0inv;
		z[i] = 2.f * n - 1.f;
	}
	__syncthreads();
	int iter = 0;
	float error;
	do {
		gmax = 0.f;
		__syncthreads();
		float phi = 0.f;
		for (int i = tid; i < N; i += BLOCK_SIZE) {
			float fx = 0.f;
			float fy = 0.f;
			float fz = 0.f;
			for (int j = 0; j < N; j++) {
				if (i != j) {
					const float dx = x[i] - x[j];
					const float dy = y[i] - y[j];
					const float dz = z[i] - z[j];
					const float r2 = sqr(dx, dy, dz);
					float r3inv, rinv;
					if (r2 > h2) {
						rinv = rsqrtf(r2);
						r3inv = rinv * rinv * rinv;
					} else {
						const float r = sqrtf(r2);
						const float q = r * hinv;
						rinv = (1.5f - 0.5f * q * q) * hinv;
						r3inv = h3inv;
					}
					fx = fmaf(m * r3inv, dx, fx);
					fy = fmaf(m * r3inv, dy, fy);
					fz = fmaf(m * r3inv, dz, fz);
					phi += m * rinv;
				}
			}
			gx[i] = fx;
			gy[i] = fy;
			gz[i] = fz;
			const float g = sqrt(fmaf(fx, fx, fmaf(fy, fy, fz * fz)));
			gmax = fmaxf(gmax, g);
		}
		shared_reduce_max<float, BLOCK_SIZE>(gmax);
		shared_reduce_add<float, BLOCK_SIZE>(phi);
		__syncthreads();
		const float dt = eta * fminf(sqrtf(h / gmax), 1.f / alpha);
		const float halfdt = 0.5f * dt;
		const float deninv = 1.f / (1.f + halfdt * alpha);
		__syncthreads();
		for (int i = tid; i < N; i += BLOCK_SIZE) {
			vx[i] = (fmaf(halfdt, gx[i], vx[i])) * deninv;
			vy[i] = (fmaf(halfdt, gy[i], vy[i])) * deninv;
			vz[i] = (fmaf(halfdt, gz[i], vz[i])) * deninv;
			x[i] = fmaf(vx[i], dt, x[i]);
			y[i] = fmaf(vy[i], dt, y[i]);
			z[i] = fmaf(vz[i], dt, z[i]);
			vx[i] = (fmaf(halfdt, gx[i], vx[i])) * deninv;
			vy[i] = (fmaf(halfdt, gy[i], vy[i])) * deninv;
			vz[i] = (fmaf(halfdt, gz[i], vz[i])) * deninv;
		}
		__syncthreads();
		float kin = 0.f;
		for (int i = tid; i < N; i += BLOCK_SIZE) {
			const float norm = sqrt(sqr(x[i], y[i], z[i]));
			const float inv = 1.f / norm;
			x[i] *= inv;
			y[i] *= inv;
			z[i] *= inv;
			const float r2 = fmaf(x[i], x[i], fmaf(y[i], y[i], z[i] * z[i]));
			const float rinv = rsqrt(r2);
			const float nx = x[i] * rinv;
			const float ny = y[i] * rinv;
			const float nz = z[i] * rinv;
			const float vr = (nx * vx[i] + ny * vy[i] + nz * vz[i]);
			vx[i] -= nx * vr;
			vy[i] -= ny * vr;
			vz[i] -= nz * vr;
			kin += fmaf(vx[i], vx[i], fmaf(vy[i], vy[i], vz[i] * vz[i])) * 0.5f * m;
		}
		shared_reduce_add<float, BLOCK_SIZE>(kin);
		error = kin / phi;
		iter++;
	} while (error > 1e-10f || iter < 100);
	if( tid == 0 ) {
		PRINT("Solved for %i\n", N);
	}
	__syncthreads();
	(&vx)->~device_vector<float>();
	(&vy)->~device_vector<float>();
	(&vz)->~device_vector<float>();
	(&gx)->~device_vector<float>();
	(&gy)->~device_vector<float>();
	(&gz)->~device_vector<float>();

}

void solve_sphere_surface_problem(vector<array<vector<float>, NDIM>>& X) {
	float** dev_x;
	float** dev_y;
	float** dev_z;
	int* dev_N;
	CUDA_CHECK(cudaMallocManaged(&dev_x, sizeof(float*) * X.size()));
	CUDA_CHECK(cudaMallocManaged(&dev_y, sizeof(float*) * X.size()));
	CUDA_CHECK(cudaMallocManaged(&dev_z, sizeof(float*) * X.size()));
	CUDA_CHECK(cudaMallocManaged(&dev_N, sizeof(int) * X.size()));
	for (int i = 0; i < X.size(); i++) {
		const int N = X[i][XDIM].size();
		CUDA_CHECK(cudaMallocManaged(&dev_x[i], sizeof(float) * N));
		CUDA_CHECK(cudaMallocManaged(&dev_y[i], sizeof(float) * N));
		CUDA_CHECK(cudaMallocManaged(&dev_z[i], sizeof(float) * N));
		dev_N[i] = X[i][XDIM].size();
	}
	auto stream = cuda_get_stream();
	sphere_surface_kernel<<<X.size(),BLOCK_SIZE,0,stream>>>(dev_x,dev_y,dev_z,dev_N);
	while (cudaStreamSynchronize(stream) != cudaSuccess) {
		hpx_yield();
	}
	cuda_end_stream(stream);
	for (int i = 0; i < X.size(); i++) {
		memcpy(X[i][XDIM].data(), dev_x[i], sizeof(float) * dev_N[i]);
		memcpy(X[i][YDIM].data(), dev_y[i], sizeof(float) * dev_N[i]);
		memcpy(X[i][ZDIM].data(), dev_z[i], sizeof(float) * dev_N[i]);
		CUDA_CHECK(cudaFree(dev_x[i]));
		CUDA_CHECK(cudaFree(dev_y[i]));
		CUDA_CHECK(cudaFree(dev_z[i]));
	}
	CUDA_CHECK(cudaFree(dev_x));
	CUDA_CHECK(cudaFree(dev_y));
	CUDA_CHECK(cudaFree(dev_z));
	CUDA_CHECK(cudaFree(dev_N));

}

void sphere_surface_test() {
	/*	const int N = 10000;
	 vector<float> x(N);
	 vector<float> y(N);
	 vector<float> z(N);
	 solve_sphere_surface_problem(x.data(), y.data(), z.data(), N);*/
}
