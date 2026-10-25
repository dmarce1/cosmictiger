/* CosmicTiger - global FMM periodic CUDA policy; BSL-1.0. */
#pragma once

#include <cosmictiger/gravity/dehnen/ewald_kernels.hpp>
#include <cosmictiger/gravity/kick_cuda_kernel.hpp>

template <int order>
__device__ FlopCount cudaGravityCcEwald(CudaKickData const& data, FmmExpansion<float, order>& lacc,
    TreeNode const& self, DeviceVector<int> const& multlist, bool doPhi) {
    int const& tid = threadIdx.x;
    auto const& treeNodes = data.treeNodes;
    FlopCounter<FlopCount> flops = 0;
    if (multlist.size()) {
        FmmExpansion<float, order> l;
        FmmExpansion<float, order> d;
        for (int i = 0; i < FmmOrder<order>::kLocalCoefficients_; i++) {
            l[i] = 0.0f;
        }
        for (int i = tid; i < multlist.size(); i += WARP_SIZE) {
            TreeNode const& other = treeNodes[multlist[i]];
            FmmMultipole<float, order> const m = fmmMultiPos<order>(other.mpos).multi;
            Array<float, NDIM> dx;
            Array<float, NDIM> unwrappedPosition;
            for (int dim = 0; dim < NDIM; dim++) {
                dx[dim] = distance(self.mpos->pos[dim], other.mpos->pos[dim]);
                unwrappedPosition[dim] =
                    unwrappedDistance(self.mpos->pos[dim], other.mpos->pos[dim]);
            }
            flops += 6 + ewaldGreensFunction(d, dx, unwrappedPosition);
            flops += m2L(l, m, d, doPhi);
        }
        sharedReduceAddArray(l);
        for (int i = tid; i < FmmOrder<order>::kLocalCoefficients_; i += WARP_SIZE) {
            lacc[i] += l[i];
        }
        // Five butterfly additions per coefficient in every lane, followed
        // by one distributed addition of each coefficient into lacc.
        flops += 5 * FmmOrder<order>::kLocalCoefficients_;
        if (tid == 0) {
            flops += FmmOrder<order>::kLocalCoefficients_;
        }
        __syncwarp();
    }
    return flops;
}

template <int order>
__device__ __forceinline__ FlopCount cudaApplyPeriodicPolicy(CudaKickData const& data,
    KickParameters const& globalParams, FmmExpansion<float, order>& expansion, TreeNode const& self,
    StackVector<int>& echecks, DeviceVector<int>& nextlist, DeviceVector<int>& cclist,
    DeviceVector<int>& leaflist, DeviceVector<int>& bothlist) {
    int const tid = threadIdx.x;
    auto const& treeNodes = data.treeNodes;
    Array<float, NDIM> dx;
    FlopCounter<FlopCount> flops = 0;
    nextlist.resize(0);
    cclist.resize(0);
    leaflist.resize(0);
    bothlist.resize(0);
    constexpr int kCc = 0;
    constexpr int kLeaf = 1;
    constexpr int kNext = 2;
    constexpr int kBoth = 3;
    constexpr int kNlists = 4;
    Array<bool, kNlists> sw;
    DeviceVector<int>* lists[kNlists] = {&cclist, &leaflist, &nextlist, &bothlist};

    auto& checks = echecks;
    float const thetainv = fmmInverseOpeningAngle(globalParams.theta);
    flops += 1;
    do {
        int const maxi = roundUp(checks.size(), WARP_SIZE);
        for (int i = tid; i < maxi; i += WARP_SIZE) {
            for (int n = 0; n < kNlists; n++) {
                sw[n] = false;
            }
            if (i < checks.size()) {
                TreeNode const& other = treeNodes[checks[i]];
                if (other.numberOfParticles()) {
                    for (int dim = 0; dim < NDIM; dim++) {
                        dx[dim] = distance(self.pos[dim], other.pos[dim]);    // 3
                    }
                    float r2 = sqr(dx[XDIM], dx[YDIM], dx[ZDIM]);
                    r2 = fmaxf(r2, sqr(fmaxf(0.5f - (self.radius + other.radius), 0.f)));
                    float const dcc = (self.radius + other.radius) * thetainv;
                    sw[kCc] = r2 > sqr(dcc);
                    flops += 14;
                    if (!sw[kCc]) {
                        ALWAYS_ASSERT(!other.leaf || !self.leaf);
                        cudaOpenOrDefer(globalParams.openLargerCell, self, other, sw[kNext],
                            sw[kLeaf], sw[kBoth]);
                    }
                }
            }
            {
                Array<int, kNlists> l, total;
                for (int n = 0; n < kNlists; n++) {
                    l[n] = sw[n];
                }
                computeIndicesArray(l, total);
                for (int n = 0; n < kNlists; n++) {
                    int const start = lists[n]->size();
                    __syncwarp();
                    lists[n]->resize(start + total[n]);
                    if (sw[n]) {
                        (*lists[n])[l[n] + start] = checks[i];
                    }
                }
            }
        }
        __syncwarp();
        checks.resize(NCHILD * nextlist.size());
        for (int i = tid; i < nextlist.size(); i += WARP_SIZE) {
            auto const& node = treeNodes[nextlist[i]];
            auto const& children = node.children;
            checks[NCHILD * i + LEFT] = children[LEFT].index;
            checks[NCHILD * i + RIGHT] = children[RIGHT].index;
        }
        nextlist.resize(0);
        __syncwarp();
    } while (checks.size() && (self.leaf || globalParams.openLargerCell));
    flops += cudaGravityCcEwald<order>(data, expansion, self, cclist, globalParams.doPhi);
    if (!self.leaf) {
        int const start = checks.size();
        int const bothStart = start + leaflist.size();
        checks.resize(bothStart + NCHILD * bothlist.size());
        for (int i = tid; i < leaflist.size(); i += WARP_SIZE) {
            checks[start + i] = leaflist[i];
        }
        for (int i = tid; i < bothlist.size(); i += WARP_SIZE) {
            auto const& children = treeNodes[bothlist[i]].children;
            checks[bothStart + NCHILD * i + LEFT] = children[LEFT].index;
            checks[bothStart + NCHILD * i + RIGHT] = children[RIGHT].index;
        }
        __syncwarp();
    }
    return flops;
}
