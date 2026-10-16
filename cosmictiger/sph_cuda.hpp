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
    float gsoft;
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
    float gcentral;
    float hcentral;
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
    ParticleInteger* dmIndexSnk;
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

#ifdef __CUDACC__
template <int blockSize>
inline __device__ bool computeSoftlens(float& h, float hmin, float hmax, float n,
    DeviceVector<SoftlensRecord> const& rec, Array<Fixed32, NDIM> const& x,
    Fixed32Range const& obox, int typeI, float& wcount) {
    int const tid = threadIdx.x;
    int const runtimeBlockSize = blockDim.x;
    if (h < hmin) {
        if (tid == 0) {
            h = hmin * 1.1f;
        }
        __syncthreads();
    }
    if (h > hmax) {
        if (tid == 0) {
            //			h = hmax * 0.5f;
        }
        __syncthreads();
    }
    float const hinv = 1.f / h;
    __syncthreads();
    int count;
    float f;
    float dfdh;
    int boxXceeded = false;
    int iter = 0;
    float dh;
    float error;
    float sgn;
    float lastSgn = 0.f;
    float w = 1.0f;
    do {
        float maxDh = h * 0.1f;
        float const hinv = 1.f / h;    // 4
        float const h2 = sqr(h);       // 1
        count = 0;
        f = 0.f;
        dfdh = 0.f;
        auto const& xI = x[XDIM];
        auto const& yI = x[YDIM];
        auto const& zI = x[ZDIM];
        for (int j = tid; j < rec.size(); j += runtimeBlockSize) {
            auto const& r = rec[j];
            auto const& typeJ = r.type;
            if (typeJ == typeI) {
                auto const& xJ = r.x;
                auto const& yJ = r.y;
                auto const& zJ = r.z;
                float const xIj = distance(xI, xJ);
                float const yIj = distance(yI, yJ);
                float const zIj = distance(zI, zJ);
                float const r2 = sqr(xIj, yIj, zIj);                 // 2
                float const r = sqrt(r2);                            // 4
                float const q = r * hinv;                            // 1
                if (q < 1.f) {                                       // 1
                    float const w = kernelW(q);                      // 4
                    float const dwdh = -q * dkernelWDq(q) * hinv;    // 3
                    f += w;                                          // 1
                    dfdh += dwdh;                                    // 1
                    count++;
                }
            }
        }
        sharedReduceAdd<float, blockSize>(f);
        sharedReduceAdd<int, blockSize>(count);
        wcount = float(4 * M_PI / 3) * f;
        sharedReduceAdd<float, blockSize>(dfdh);
        float xValue, dXdh;
        dsmoothXDh(h, hmin, hmax, xValue, dXdh);
        dh = 0.2f * h;
        if (count > 1) {
            dfdh = dfdh + dXdh * f;
            f *= xValue;
            f -= n * float(3.0 / (4.0 * M_PI));
            dh = -f / dfdh;
            sgn = copysignf(1.f, dh);
            if (sgn * lastSgn < 0.f) {
                w = fmaxf(0.5f * w, 0.001f);
            } else {
                w = fminf(1.f, w * 1.1f);
            }
            lastSgn = sgn;
            dh = fminf(fmaxf(dh, -maxDh), maxDh);
        }
        error = fabsf(dh / h);
        __syncthreads();
        if (tid == 0) {
            h += w * dh;
            if (iter >= 100) {
                PRINT("over iteration on h solve - %i %e %e %e %e %i\n", iter, h, dh, w, error,
                    count);
            }
        }
        __syncthreads();
        for (int dim = 0; dim < NDIM; dim++) {
            if (obox.end[dim] < RangeFixed(x[dim] + Fixed32(h)) + RangeFixed::min()) {
                boxXceeded = true;
                break;
            }
            if (RangeFixed(x[dim]) < obox.begin[dim] + RangeFixed(h) + RangeFixed::min()) {
                boxXceeded = true;
                break;
            }
        }
        iter++;
        sharedReduceAdd<int, blockSize>(boxXceeded);
    } while (error > 1e-4f && !boxXceeded);
    if (tid == 0 && h <= 0.f) {
        PRINT("Less than ZERO H! sph.cu %e\n", h);
        __trap();
    }
    return !boxXceeded;
}
#endif

#endif /* SPH_CUDA_HPP_ */
