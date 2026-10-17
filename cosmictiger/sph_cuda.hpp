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

#ifndef SPH_CUDA123_HPP_
#define SPH_CUDA123_HPP_

#include <cosmictiger/chemistry.hpp>
#include <cosmictiger/constants.hpp>
#include <cosmictiger/cuda_memory.hpp>
#include <cosmictiger/cuda_reduce.hpp>
#include <cosmictiger/defs.hpp>
#include <cosmictiger/device_vector.hpp>
#include <cosmictiger/kernel.hpp>
#include <cosmictiger/math.hpp>
#include <cosmictiger/sph.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/sph_tree.hpp>
#include <cosmictiger/timer.hpp>

#define ETA 0.01f

struct SphRunCudaData {
    Fixed32* x;
    Fixed32* y;
    Fixed32* z;
    char* stars;
#ifdef ENTROPY
    float* dentr1Snk;
    float* dentr2Snk;
    float* entr0Snk;
    float* entr;
#else
    float* deint1Snk;
    float* deint2Snk;
    float* eint0Snk;
    float* eint;
#endif
    float* kappa;
    float* coldFrac;
    float gravityEffectiveRadius;    // Effective gravitational smoothing radius, h = 2 sigma.
    Array<float, NCHEMFRACS>* chem;
    float* divvSnk;
    float* rhoSnk;
    float* mmw;
    float* gradT;
    float* shearv;
    float* coldMassSnk;
    float* dcoldMass;
    char* saSnk;
    float defGamma;
    bool conduction;
    float* rho;
    bool gravity;
    bool chemistry;
    char* convergedSnk;
    float centralAcceleration;
    float centralSupportRadius;
    float g;
    float t0;
    float kappa0;
    float rho0C;
    float* alpha;
    float* alpha0;
    float rho0B;
    //	float* Z;
    //	float* Y;
    float y0;
    float* gx;
    float* gy;
    float* gz;
    float* vx;
    float* vy;
    float* vz;
    float* omega;
    float* omegaP;
    float* pre;
    float* h;
    char* rungs;
    char* rungsSnk;
    float* omegaSnk;
    float* omegaPSnk;
    float* preSnk;
    float* shearSnk;
    SphRecord1* rec1Snk;
    SphRecord2* rec2Snk;
    ParticleInteger* sinkParticleIndices;
    float* gxSnk;
    float* gySnk;
    float* gzSnk;
    float* kapSnk;
    SphRecord5* rec5Snk;
    SphRecord6* rec6Snk;
    float codeDifToCgs;
    float n;
    int numberOfTargetNodes;
    SphTreeNode* trees;
    int* selfs;
    int* neighbors;
    float m;
    float eta;
};

#define PREHYDRO1_BLOCK_SIZE 160
#define PREHYDRO2_BLOCK_SIZE 64
#define HYDRO_BLOCK_SIZE 96
#define AUX_BLOCK_SIZE 64
#define RUNGS_BLOCK_SIZE 256

#define COND_INIT_BLOCK_SIZE 32
#define CONDUCTION_BLOCK_SIZE 32

#define SPH_DIFFUSION_C 0.03f
#define MAX_RUNG_DIF 1
#define SPH_SMOOTHLEN_TOLER float(1.0e-4)

struct SphReduction {
    int counter;
    int flag;
    float hmin;
    float hmax;
    float vsigMax;
    double flops;
    int maxRungHydro;
    int maxRungGrav;
    int maxRung;
    float dtinvCfl;
    float dtinvVisc;
    float dtinvDiff;
    float dtinvCond;
    float dtinvAcc;
    float dtinvDivv;
    float dtinvOmega;
};

SphRunReturn sphRunCuda(SphRunParameters params, SphRunCudaData data, cudaStream_t stream);

#ifdef __CUDACC__

__global__ void sphCudaPrehydro1(
    SphRunParameters params, SphRunCudaData data, SphReduction* reduce);
__global__ void sphCudaPrehydro2(
    SphRunParameters params, SphRunCudaData data, SphReduction* reduce);
__global__ void sphCudaAux(SphRunParameters params, SphRunCudaData data, SphReduction* reduce);
__global__ void sphCudaCondInit(SphRunParameters params, SphRunCudaData data, SphReduction* reduce);
__global__ void sphCudaHydro(SphRunParameters params, SphRunCudaData data, SphReduction* reduce);
__global__ void sphCudaConduction(
    SphRunParameters params, SphRunCudaData data, SphReduction* reduce);
__global__ void sphCudaRungs(SphRunParameters params, SphRunCudaData data, SphReduction* reduce);
#endif

struct SoftlensRecord {
    Fixed32 x;
    Fixed32 y;
    Fixed32 z;
    char type;
};

// A source can reach any sink in this box through a periodic image. Expanding
// the unwrapped interval before testing avoids inconsistent endpoint wrapping.
CUDA_EXPORT inline bool sphSupportIntersects(
    Fixed32Range const& sinkBox, Array<Fixed32, NDIM> const& sourcePosition, float sourceSupport) {
    Fixed32Range expanded = sinkBox;
    for (int dim = 0; dim < NDIM; dim++) {
        expanded.begin[dim] -= RangeFixed(sourceSupport);
        expanded.end[dim] += RangeFixed(sourceSupport);
    }
    return expanded.contains(sourcePosition);
}

#ifdef __CUDACC__
template <int blockSize>
inline __device__ bool computeSoftlens(float& h, float hmin, float hmax, float n,
    DeviceVector<SoftlensRecord> const& rec, Array<Fixed32, NDIM> const& x,
    Fixed32Range const& obox, int typeI, float& wcount) {
    int const tid = threadIdx.x;
    int const runtimeBlockSize = blockDim.x;
    ALWAYS_ASSERT(hmin > 0.0f && hmax >= hmin && n > 0.0f);
    if (tid == 0) {
        h = fminf(fmaxf(h, hmin), hmax);
    }
    __syncthreads();
    float lower = hmin;
    float upper = hmax;
    float const target = n * float(3.0 / (4.0 * M_PI));
    // rho H^3 / m = sum W(r/H) is held fixed, giving H proportional to rho^-1/3.
    // hmin and hmax bound the support radius without modifying the density sum.
    for (int iteration = 0; iteration < 128; iteration++) {
        int boxExceeded = false;
        for (int dim = 0; dim < NDIM; dim++) {
            if (obox.end[dim] < RangeFixed(x[dim]) + RangeFixed(h) + RangeFixed::min() ||
                RangeFixed(x[dim]) < obox.begin[dim] + RangeFixed(h) + RangeFixed::min()) {
                boxExceeded = true;
            }
        }
        if (boxExceeded) {
            return false;
        }
        float const hinv = 1.0f / h;
        float sum = 0.0f;
        float derivative = 0.0f;
        for (int j = tid; j < rec.size(); j += runtimeBlockSize) {
            auto const& neighbor = rec[j];
            if (neighbor.type == typeI) {
                float const dx = distance(x[XDIM], neighbor.x);
                float const dy = distance(x[YDIM], neighbor.y);
                float const dz = distance(x[ZDIM], neighbor.z);
                float const q = sqrtf(sqr(dx, dy, dz)) * hinv;
                if (q < 1.0f) {
                    sum += kernelW(q);
                    derivative -= q * dkernelWDq(q) * hinv;
                }
            }
        }
        sharedReduceAdd<float, blockSize>(sum);
        sharedReduceAdd<float, blockSize>(derivative);
        wcount = float(4.0 * M_PI / 3.0) * sum;
        float const residual = sum - target;
        if (fabsf(residual) <= SPH_SMOOTHLEN_TOLER * target || (h <= hmin && residual >= 0.0f) ||
            (h >= hmax && residual <= 0.0f)) {
            return true;
        }
        if (residual > 0.0f) {
            upper = h;
        } else {
            lower = h;
        }
        float change = derivative > 0.0f ? -residual / derivative : 0.2f * h;
        change = fminf(fmaxf(change, -0.2f * h), 0.2f * h);
        float next = fminf(fmaxf(h + change, hmin), hmax);
        if (next < lower || next > upper) {
            next = 0.5f * (lower + upper);
        }
        if (next == h) {
            return true;
        }
        __syncthreads();
        if (tid == 0) {
            h = next;
        }
        __syncthreads();
    }
    if (tid == 0) {
        PRINT("SPH smoothing-length solve did not converge: h=%e bounds=[%e,%e] target=%e\n", h,
            hmin, hmax, n);
        __trap();
    }
    return false;
}
#endif

#endif /* SPH_CUDA_HPP_ */
