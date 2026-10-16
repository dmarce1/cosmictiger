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

#include <cosmictiger/monaghan_sph.hpp>
#include <cosmictiger/sph_cuda.hpp>

namespace detail {

struct HydroRecord {
    Fixed32 x;
    Fixed32 y;
    Fixed32 z;
    float h;
    float vx;
    float vy;
    float vz;
    float energy;
    float rho;
};

struct HydroWorkspace {
    DeviceVector<HydroRecord> neighbors;
};

}    // namespace detail

// Monaghan (1992), equations (4.1)-(4.2) and (6.4)-(6.5).
// Positions and density are comoving; velocity and energy store a*v and a^2*u.
// Stored h is the kernel support radius, twice the paper's smoothing length.
__global__ void sphCudaHydro(SphRunParameters params, SphRunCudaData data, SphReduction* reduce) {
    int const tid = threadIdx.x;
    int const blockSize = blockDim.x;
    __shared__ int index;
    __shared__ detail::HydroWorkspace workspace;
    new (&workspace) detail::HydroWorkspace();
    if (tid == 0) {
        index = atomicAdd(&reduce->counter, 1);
    }
    __syncthreads();
    float const ainv = 1.0f / params.a;
    float const gamma = data.defGamma;
    float const massOverA = data.m * ainv;
    int maxRungHydro = 0;
    int maxRungGrav = 0;
    int maxRung = 0;
    float maxSignal = 0.0f;
    while (index < data.numberOfTargetNodes) {
        __syncthreads();
        SphTreeNode const& self = data.trees[data.selfs[index]];
        workspace.neighbors.resize(0);
        // The union of both supports is required by the symmetric gradient.
        for (int ni = self.neighborRange.first; ni < self.neighborRange.second; ni++) {
            auto const& other = data.trees[data.neighbors[ni]];
            int const end = other.particleRange.first +
                roundUp(other.particleRange.second - other.particleRange.first, blockSize);
            for (int pi = other.particleRange.first + tid; pi < end; pi += blockSize) {
                bool contains = false;
                Array<Fixed32, NDIM> position;
                if (pi < other.particleRange.second) {
                    position[XDIM] = data.x[pi];
                    position[YDIM] = data.y[pi];
                    position[ZDIM] = data.z[pi];
                    contains = self.outerBox.contains(position);
                    if (!contains) {
                        contains = sphSupportIntersects(self.innerBox, position, data.h[pi]);
                    }
                }
                int slot = contains;
                int total;
                computeIndices<HYDRO_BLOCK_SIZE>(slot, total);
                int const offset = workspace.neighbors.size();
                __syncthreads();
                workspace.neighbors.resize(offset + total);
                if (contains) {
                    workspace.neighbors[offset + slot] = {position[XDIM], position[YDIM],
                        position[ZDIM], data.h[pi], data.vx[pi], data.vy[pi], data.vz[pi],
                        data.eint[pi], data.rho[pi]};
                }
            }
        }
        for (int i = self.particleRange.first; i < self.particleRange.second; i++) {
            __syncthreads();
            int const sink = self.sinkParticleRange.first - self.particleRange.first + i;
            int const previousRung = data.rungsSnk[data.sinkParticleIndices[sink]];
            if (previousRung < params.minRung) {
                continue;
            }
            float const hI = data.h[i];
            float const rhoI = data.rho[i];
            float const energyI = data.eint[i];
            ALWAYS_ASSERT(hI > 0.0f && rhoI > 0.0f && energyI > 0.0f);
            float const pressureI = (gamma - 1.0f) * rhoI * energyI;
            float const soundI = sqrtf(gamma * (gamma - 1.0f) * energyI);
            float ax = 0.0f;
            float ay = 0.0f;
            float az = 0.0f;
            float energyRate = 0.0f;
            float divergence = 0.0f;
            float maxCompression = 0.0f;
            for (int j = tid; j < workspace.neighbors.size(); j += blockSize) {
                auto const& neighbor = workspace.neighbors[j];
                float const dx = distance(data.x[i], neighbor.x);
                float const dy = distance(data.y[i], neighbor.y);
                float const dz = distance(data.z[i], neighbor.z);
                float const r2 = sqr(dx, dy, dz);
                if (r2 == 0.0f || r2 >= fmaxf(sqr(hI), sqr(neighbor.h))) {
                    continue;
                }
                float const radius = sqrtf(r2);
                float const invHI = 1.0f / hI;
                float const invHJ = 1.0f / neighbor.h;
                float const gradientI =
                    monaghan_sph::kernelDerivative(radius * invHI) * sqr(sqr(invHI)) / radius;
                float const gradientJ =
                    monaghan_sph::kernelDerivative(radius * invHJ) * sqr(sqr(invHJ)) / radius;
                float const gradient = 0.5f * (gradientI + gradientJ);
                float const vx = data.vx[i] - neighbor.vx;
                float const vy = data.vy[i] - neighbor.vy;
                float const vz = data.vz[i] - neighbor.vz;
                float const peculiarDotR = vx * dx + vy * dy + vz * dz;
                // Include Hubble motion only in physical pair approach/viscous heating.
                float const physicalDotR = peculiarDotR + params.adot * r2;
                float const mu = monaghan_sph::viscosityMu(
                    r2, physicalDotR, hI, neighbor.h, params.viscosityEpsilon);
                float const pressureJ = (gamma - 1.0f) * neighbor.rho * neighbor.energy;
                float const soundJ = sqrtf(gamma * (gamma - 1.0f) * neighbor.energy);
                float const viscosity = monaghan_sph::artificialViscosity(
                    rhoI, neighbor.rho, soundI, soundJ, mu, params.alpha0, params.beta);
                float const pressurePair =
                    monaghan_sph::pressureCoefficient(rhoI, neighbor.rho, pressureI, pressureJ);
                float const force = massOverA * (pressurePair + viscosity) * gradient;
                ax -= force * dx;
                ay -= force * dy;
                az -= force * dz;
                energyRate += massOverA *
                    monaghan_sph::pairEnergyRate(
                        pressurePair, viscosity, peculiarDotR * gradient, physicalDotR * gradient);
                divergence -= massOverA / rhoI * peculiarDotR * gradientI;
                maxCompression = fmaxf(maxCompression, -mu);
            }
            sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(ax);
            sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(ay);
            sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(az);
            sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(energyRate);
            sharedReduceAdd<float, HYDRO_BLOCK_SIZE>(divergence);
            sharedReduceMax<HYDRO_BLOCK_SIZE>(maxCompression);
            if (tid == 0) {
                float const gx = data.gravity ? data.gxSnk[sink] : 0.0f;
                float const gy = data.gravity ? data.gySnk[sink] : 0.0f;
                float const gz = data.gravity ? data.gzSnk[sink] : 0.0f;
                ax += gx;
                ay += gy + params.yAcceleration;
                az += gz;
                if (data.centralAcceleration != 0.0f) {
                    float const dx = distance(data.x[i], Fixed32(0.5));
                    float const dy = distance(data.y[i], Fixed32(0.5));
                    float const dz = distance(data.z[i], Fixed32(0.5));
                    float const radius = sqrtf(sqr(dx, dy, dz));
                    float const hinv = 1.0f / data.hcentral;
                    float const force =
                        data.centralAcceleration * kernelFqinv(radius * hinv) * hinv * hinv * hinv;
                    ax -= force * dx;
                    ay -= force * dy;
                    az -= force * dz;
                }
                ax -= params.damping * data.vx[i];
                ay -= params.damping * data.vy[i];
                az -= params.damping * data.vz[i];
                data.rec6Snk[sink].dvel[XDIM] = ax;
                data.rec6Snk[sink].dvel[YDIM] = ay;
                data.rec6Snk[sink].dvel[ZDIM] = az;
                // Homogeneous adiabatic expansion of a^2*u for arbitrary gamma.
                energyRate += (5.0f - 3.0f * gamma) * params.adot * ainv * energyI;
                data.deint2Snk[sink] = energyRate;
                data.rec1Snk[sink].alpha = params.alpha0;
                if (params.phase == 1) {
                    data.deint1Snk[sink] = 0.0f;
                    data.divvSnk[sink] = divergence;
                    float const smoothingLength = 0.5f * hI;
                    // Section 10.3: max(-mu) is positive for approaching particles.
                    float const dtinvCfl = soundI / smoothingLength;
                    float const dtinvVisc = 0.6f *
                        (params.alpha0 * soundI + params.beta * maxCompression) / smoothingLength;
                    float const dtinvDivv = params.a * fabsf(divergence) / 3.0f;
                    float const acceleration = sqrtf(sqr(ax, ay, az));
                    float const dtinvAcc =
                        params.cfl * sqrtf(params.a * acceleration / smoothingLength) / data.eta;
                    float const dtinv = fmaxf(dtinvCfl + dtinvVisc, fmaxf(dtinvDivv, dtinvAcc));
                    float hydroDt =
                        fminf(params.maxDt, params.cfl * params.a / fmaxf(dtinv, 1.0e-30f));
                    // Bound fractional thermal change in the existing explicit integrator.
                    if (energyRate != 0.0f) {
                        hydroDt = fminf(hydroDt, params.cfl * energyI / fabsf(energyRate));
                    }
                    float const gravityAcceleration = sqrtf(sqr(gx, gy, gz));
                    float const gravityDt = fminf(params.maxDt,
                        data.eta *
                            sqrtf(params.a * data.gravitySofteningLength /
                                fmaxf(gravityAcceleration, 1.0e-30f)));
                    ALWAYS_ASSERT(isfinite(hydroDt) && hydroDt > 0.0f);
                    int const hydroRung = max(1, int(ceilf(log2f(params.t0 / hydroDt))));
                    int const gravityRung = max(1, int(ceilf(log2f(params.t0 / gravityDt))));
                    int const nextRung = max(max(params.minRung, previousRung - 1),
                        max(hydroRung, data.gravity ? gravityRung : 1));
                    ALWAYS_ASSERT(nextRung < MAX_RUNG);
                    data.rungsSnk[data.sinkParticleIndices[sink]] = nextRung;
                    maxRungHydro = max(maxRungHydro, hydroRung);
                    maxRungGrav = max(maxRungGrav, gravityRung);
                    maxRung = max(maxRung, nextRung);
                    maxSignal = fmaxf(maxSignal, (dtinvCfl + dtinvVisc) * smoothingLength);
                    atomicMax(&reduce->dtinvCfl, dtinvCfl);
                    atomicMax(&reduce->dtinvVisc, dtinvVisc);
                    atomicMax(&reduce->dtinvAcc, dtinvAcc);
                    atomicMax(&reduce->dtinvDivv, dtinvDivv);
                }
            }
        }
        __syncthreads();
        if (tid == 0) {
            index = atomicAdd(&reduce->counter, 1);
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicMax(&reduce->vsigMax, maxSignal);
        atomicMax(&reduce->maxRung, maxRung);
        atomicMax(&reduce->maxRungHydro, maxRungHydro);
        atomicMax(&reduce->maxRungGrav, maxRungGrav);
    }
    (&workspace)->~HydroWorkspace();
}
