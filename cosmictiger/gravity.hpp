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

#include <cmath>

#include <cosmictiger/cuda.hpp>
#include <cosmictiger/cuda_memory.hpp>
#include <cosmictiger/fixed.hpp>
#include <cosmictiger/fixed_capacity_vector.hpp>
#include <cosmictiger/kernel.hpp>
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

void resetGravityCounters();
void getGravityCounters(double& close, double& direct);
void setGravityCounterUse(bool code);

template <class T>
CUDA_EXPORT inline void gravitySofteningKernel(T& force, T& phi, T radiusSquared, T inverseSupport,
    T inverseSupportSquared, T inverseSupportCubed, bool doPhi) {
    T const q = simdSqrt(radiusSquared * inverseSupportSquared);
    auto const& kernel = selectedGravityKernel();
    force = radial_kernel::forceFactor(kernel, q) * inverseSupportCubed;
    if (doPhi) {
        phi = radial_kernel::potential(kernel, q) * inverseSupport;
    } else {
        phi = simdBroadcast<T>(0);
    }
}

CUDA_EXPORT inline float selfPhi() {
    return selectedGravityKernel().centralPotential;
}

#endif /* GRAVITY_HPP_ */
