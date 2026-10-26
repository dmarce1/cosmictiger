/* CosmicTiger - global FMM periodic CUDA policy; BSL-1.0. */
#pragma once

#include <cosmictiger/gravity/fmm_kernels/cartesian_ewald_operators.hpp>
#include <cosmictiger/gravity/kick_cuda_kernel.hpp>

template <int order>
__device__ FlopCount cudaGravityCcEwald(CudaKickData const& data,
    cosmictiger::gravity::cartesian::Expansion<float, order>& lacc, TreeNode const& self,
    DeviceVector<int> const& multlist, bool doPhi) {
    int const& tid = threadIdx.x;
    auto const& treeNodes = data.treeNodes;
    FlopCounter<FlopCount> flops = 0;
    if (multlist.size()) {
        cosmictiger::gravity::cartesian::Expansion<float, order> l;
        cosmictiger::gravity::cartesian::Green<float, order> d{};
        for (int i = 0; i < FmmOrder<order>::kLocalCoefficients_; i++) {
            l[i] = 0.0f;
        }
        for (int i = tid; i < multlist.size(); i += kWarpSize) {
            TreeNode const& other = treeNodes[multlist[i]];
            cosmictiger::gravity::cartesian::Multipole<float, order> const m =
                fmmMultiPos<order>(other.mpos).multi;
            Array<float, kDimensions> dx;
            Array<float, kDimensions> unwrappedPosition;
            for (int dim = 0; dim < kDimensions; dim++) {
                dx[dim] = distance(self.mpos->pos[dim], other.mpos->pos[dim]);
                unwrappedPosition[dim] =
                    unwrappedDistance(self.mpos->pos[dim], other.mpos->pos[dim]);
            }
            flops +=
                6 + cosmictiger::gravity::cartesian::evaluateEwaldGreen(d, dx, unwrappedPosition);
            flops += cosmictiger::gravity::cartesian::M2L(l, m, d, doPhi);
        }
        sharedReduceAddArray(l);
        for (int i = tid; i < FmmOrder<order>::kLocalCoefficients_; i += kWarpSize) {
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
    KickParameters const& globalParams,
    cosmictiger::gravity::cartesian::Expansion<float, order>& expansion, TreeNode const& self,
    StackVector<int>& echecks, DeviceVector<int>& nextlist, DeviceVector<int>& cclist,
    DeviceVector<int>& leaflist, DeviceVector<int>& bothlist) {
    int const tid = threadIdx.x;
    auto const& treeNodes = data.treeNodes;
    Array<float, kDimensions> dx;
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
        int const maxi = roundUp(checks.size(), kWarpSize);
        for (int i = tid; i < maxi; i += kWarpSize) {
            for (int n = 0; n < kNlists; n++) {
                sw[n] = false;
            }
            if (i < checks.size()) {
                TreeNode const& other = treeNodes[checks[i]];
                if (other.numberOfParticles()) {
                    for (int dim = 0; dim < kDimensions; dim++) {
                        dx[dim] = distance(self.pos[dim], other.pos[dim]);    // 3
                    }
                    float r2 = sqr(dx[kXDimension], dx[kYDimension], dx[kZDimension]);
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
        checks.resize(kChildCount * nextlist.size());
        for (int i = tid; i < nextlist.size(); i += kWarpSize) {
            auto const& node = treeNodes[nextlist[i]];
            auto const& children = node.children;
            checks[kChildCount * i + kLeftChild] = children[kLeftChild].index;
            checks[kChildCount * i + kRightChild] = children[kRightChild].index;
        }
        nextlist.resize(0);
        __syncwarp();
    } while (checks.size() && (self.leaf || globalParams.openLargerCell));
    flops += cudaGravityCcEwald<order>(data, expansion, self, cclist, globalParams.doPhi);
    if (!self.leaf) {
        int const start = checks.size();
        int const bothStart = start + leaflist.size();
        checks.resize(bothStart + kChildCount * bothlist.size());
        for (int i = tid; i < leaflist.size(); i += kWarpSize) {
            checks[start + i] = leaflist[i];
        }
        for (int i = tid; i < bothlist.size(); i += kWarpSize) {
            auto const& children = treeNodes[bothlist[i]].children;
            checks[bothStart + kChildCount * i + kLeftChild] = children[kLeftChild].index;
            checks[bothStart + kChildCount * i + kRightChild] = children[kRightChild].index;
        }
        __syncwarp();
    }
    return flops;
}
