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

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/cuda_mem.hpp>
#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/device_vector.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/math.hpp>

#define alpha 2.0f
#define eta 0.25f

#define BLOCK_SIZE 512

__global__ void sphereSurfaceKernel(float** xMain, float** yMain, float** zMain, int* nMain) {
    int const tid = threadIdx.x;
    int const bid = blockIdx.x;
    float* x = xMain[bid];
    float* y = yMain[bid];
    float* z = zMain[bid];
    int const nValue = nMain[bid];
    constexpr unsigned kA0 = 1103515245;
    constexpr unsigned kC0 = 12345;
    constexpr unsigned kM0 = 1UL << 31;
    unsigned rndNum = tid;
    __shared__ DeviceVector<float> vx;
    __shared__ DeviceVector<float> vy;
    __shared__ DeviceVector<float> vz;
    __shared__ DeviceVector<float> gx;
    __shared__ DeviceVector<float> gy;
    __shared__ DeviceVector<float> gz;
    new (&vx) DeviceVector<float>(nValue);
    new (&vy) DeviceVector<float>(nValue);
    new (&vz) DeviceVector<float>(nValue);
    new (&gx) DeviceVector<float>(nValue);
    new (&gy) DeviceVector<float>(nValue);
    new (&gz) DeviceVector<float>(nValue);
    float gmax;
    float const h = 0.9f * sqrt(4.f * float(M_PI) / nValue);
    float const h2 = h * h;
    float const hinv = 1.f / h;
    float const h3inv = hinv * hinv * hinv;
    float const m = 1.0f / nValue;
    __syncthreads();
    float const m0inv = 1.f / kM0;
    for (int i = tid; i < nValue; i += BLOCK_SIZE) {
        vx[i] = 0.f;
        vy[i] = 0.f;
        vz[i] = 0.f;
        float n;
        rndNum = (kA0 * rndNum + kC0) % kM0;
        n = (rndNum + 0.5f) * m0inv;
        x[i] = 2.f * n - 1.f;
        rndNum = (kA0 * rndNum + kC0) % kM0;
        n = (rndNum + 0.5f) * m0inv;
        y[i] = 2.f * n - 1.f;
        rndNum = (kA0 * rndNum + kC0) % kM0;
        n = (rndNum + 0.5f) * m0inv;
        z[i] = 2.f * n - 1.f;
    }
    __syncthreads();
    int iter = 0;
    float error;
    do {
        gmax = 0.f;
        __syncthreads();
        float phi = 0.f;
        for (int i = tid; i < nValue; i += BLOCK_SIZE) {
            float fx = 0.f;
            float fy = 0.f;
            float fz = 0.f;
            for (int j = 0; j < nValue; j++) {
                if (i != j) {
                    float const dx = x[i] - x[j];
                    float const dy = y[i] - y[j];
                    float const dz = z[i] - z[j];
                    float const r2 = sqr(dx, dy, dz);
                    float r3inv, rinv;
                    if (r2 > h2) {
                        rinv = rsqrtf(r2);
                        r3inv = rinv * rinv * rinv;
                    } else {
                        float const r = sqrtf(r2);
                        float const q = r * hinv;
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
            float const g = sqrt(fmaf(fx, fx, fmaf(fy, fy, fz * fz)));
            gmax = fmaxf(gmax, g);
        }
        sharedReduceMax<BLOCK_SIZE>(gmax);
        sharedReduceAdd<float, BLOCK_SIZE>(phi);
        __syncthreads();
        float const dt = eta * fminf(sqrtf(h / gmax), 1.f / alpha);
        float const halfdt = 0.5f * dt;
        float const deninv = 1.f / (1.f + halfdt * alpha);
        __syncthreads();
        for (int i = tid; i < nValue; i += BLOCK_SIZE) {
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
        for (int i = tid; i < nValue; i += BLOCK_SIZE) {
            float const norm = sqrt(sqr(x[i], y[i], z[i]));
            float const inv = 1.f / norm;
            x[i] *= inv;
            y[i] *= inv;
            z[i] *= inv;
            float const r2 = fmaf(x[i], x[i], fmaf(y[i], y[i], z[i] * z[i]));
            float const rinv = rsqrt(r2);
            float const nx = x[i] * rinv;
            float const ny = y[i] * rinv;
            float const nz = z[i] * rinv;
            float const vr = (nx * vx[i] + ny * vy[i] + nz * vz[i]);
            vx[i] -= nx * vr;
            vy[i] -= ny * vr;
            vz[i] -= nz * vr;
            kin += fmaf(vx[i], vx[i], fmaf(vy[i], vy[i], vz[i] * vz[i])) * 0.5f * m;
        }
        sharedReduceAdd<float, BLOCK_SIZE>(kin);
        error = kin / phi;
        iter++;
    } while (error > 1e-10f || iter < 100);
    if (tid == 0) {
        PRINT("Solved for %i\n", nValue);
    }
    __syncthreads();
    (&vx)->~DeviceVector<float>();
    (&vy)->~DeviceVector<float>();
    (&vz)->~DeviceVector<float>();
    (&gx)->~DeviceVector<float>();
    (&gy)->~DeviceVector<float>();
    (&gz)->~DeviceVector<float>();
}

void solveSphereSurfaceProblem(Vector<Array<Vector<float>, NDIM>>& xValue) {
    float** devX;
    float** devY;
    float** devZ;
    int* devN;
    CUDA_CHECK(cudaMallocManaged(&devX, sizeof(float*) * xValue.size()));
    CUDA_CHECK(cudaMallocManaged(&devY, sizeof(float*) * xValue.size()));
    CUDA_CHECK(cudaMallocManaged(&devZ, sizeof(float*) * xValue.size()));
    CUDA_CHECK(cudaMallocManaged(&devN, sizeof(int) * xValue.size()));
    for (int i = 0; i < xValue.size(); i++) {
        int const nValue = xValue[i][XDIM].size();
        CUDA_CHECK(cudaMallocManaged(&devX[i], sizeof(float) * nValue));
        CUDA_CHECK(cudaMallocManaged(&devY[i], sizeof(float) * nValue));
        CUDA_CHECK(cudaMallocManaged(&devZ[i], sizeof(float) * nValue));
        devN[i] = xValue[i][XDIM].size();
    }
    auto stream = cudaGetStream();
    sphereSurfaceKernel<<<xValue.size(), BLOCK_SIZE, 0, stream>>>(devX, devY, devZ, devN);
    while (cudaStreamSynchronize(stream) != cudaSuccess) {
        hpxYield();
    }
    cudaEndStream(stream);
    for (int i = 0; i < xValue.size(); i++) {
        memcpy(xValue[i][XDIM].data(), devX[i], sizeof(float) * devN[i]);
        memcpy(xValue[i][YDIM].data(), devY[i], sizeof(float) * devN[i]);
        memcpy(xValue[i][ZDIM].data(), devZ[i], sizeof(float) * devN[i]);
        CUDA_CHECK(cudaFree(devX[i]));
        CUDA_CHECK(cudaFree(devY[i]));
        CUDA_CHECK(cudaFree(devZ[i]));
    }
    CUDA_CHECK(cudaFree(devX));
    CUDA_CHECK(cudaFree(devY));
    CUDA_CHECK(cudaFree(devZ));
    CUDA_CHECK(cudaFree(devN));
}

void sphereSurfaceTest() {
    /*	const int N = 10000;
     vector<float> x(N);
     vector<float> y(N);
     vector<float> z(N);
     solve_sphere_surface_problem(x.data(), y.data(), z.data(), N);*/
}
