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

#ifndef GRAVITY_HPP_
#define GRAVITY_HPP_

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/cuda_memory.hpp>
#include <cosmictiger/fixed.hpp>
#include <cosmictiger/fixed_capacity_vector.hpp>
#include <cosmictiger/kick.hpp>
#include <cosmictiger/tree.hpp>

struct ForceVectors {
    Vector<float> phi;
    Vector<float> gx;
    Vector<float> gy;
    Vector<float> gz;
    ForceVectors() = default;
    ForceVectors(int sz) {
        phi.resize(sz);
        gx.resize(sz);
        gy.resize(sz);
        gz.resize(sz);
    }
};

using GravityCcType = int;
constexpr GravityCcType kGravityDirect = 0;
constexpr GravityCcType kGravityEwald = 1;

void cpuGravityCc(GravityCcType, Expansion<float>&, Vector<TreeId> const&, TreeId, bool doPhi);
void cpuGravityCp(Expansion<float>&, Vector<TreeId> const&, TreeId, bool doPhi);
void cpuGravityPc(ForceVectors&, int, TreeId, Vector<TreeId> const&);
void cpuGravityPp(ForceVectors&, int, TreeId, Vector<TreeId> const&, float h);

void showTimings();

#ifdef __CUDACC__
__device__ void cudaGravityCcDirect(
    CudaKickData const&, Expansion<float>&, TreeNode const&, DeviceVector<int> const&, bool doPhi);
__device__ void cudaGravityCpDirect(
    CudaKickData const&, Expansion<float>&, TreeNode const&, DeviceVector<int> const&, bool doPhi);
__device__ void cudaGravityPcDirect(
    CudaKickData const& data, TreeNode const&, DeviceVector<int> const&, bool);
__device__ void cudaGravityCcEwald(
    CudaKickData const&, Expansion<float>&, TreeNode const&, DeviceVector<int> const&, bool doPhi);
__device__ void cudaGravityPpDirect(
    CudaKickData const& data, TreeNode const&, DeviceVector<int> const&, float h, bool);
#endif
#endif /* GRAVITY_HPP_ */

void resetGravityCounters();
void getGravityCounters(double& close, double& direct);
void setGravityCounterUse(bool code);

#define FERRER_N 4
#define NFERRER 5

template <class T>
CUDA_EXPORT inline void gravitySofteningKernel(
    T& f, T& phi, T q2, T hinv, T h2inv, T h3inv, bool doPhi) {
    constexpr float kAf[NFERRER][NFERRER] = {{1.0}, {2.5, -1.5},
        {35.0 / 8.0, -21.0 / 4.0, 15.0 / 8.0},
        {105.0 / 16.0, -189.0 / 16.0, 135.0 / 16.0, -35.0 / 16.0},
        {1155. / 128., -693. / 32., 1485. / 64., -385. / 32., 315. / 128.}};
    constexpr float kAphi[NFERRER][NFERRER + 1] = {{1.5, -0.5}, {15.0 / 8.0, -5.0 / 4.0, 3.0 / 8.0},
        {35.0 / 16.0, -35.0 / 16.0, 21.0 / 16.0, -5.0 / 16.0},
        {315.0 / 128.0, -105.0 / 32.0, 189.0 / 64.0, -45.0 / 32.0, 35.0 / 128.0},
        {693. / 256., -1155. / 256., 693. / 128., -495. / 128., 385. / 256., -63 / 256.}};
    q2 *= h2inv;
    f = kAf[FERRER_N][FERRER_N];
    for (int n = FERRER_N - 1; n >= 0; n--) {
        f = fmaf(f, q2, kAf[FERRER_N][n]);
    }
    f *= h3inv;
    if (doPhi) {
        phi = kAphi[FERRER_N][FERRER_N + 1];
        for (int n = FERRER_N; n >= 0; n--) {
            phi = fmaf(phi, q2, kAphi[FERRER_N][n]);
        }
        phi *= hinv;    // 1
    }
}

CUDA_EXPORT inline float selfPhi() {
    float f, phi;
    gravitySofteningKernel(f, phi, 0.0f, 1.f, 1.f, 1.f, true);
    return phi;
}
