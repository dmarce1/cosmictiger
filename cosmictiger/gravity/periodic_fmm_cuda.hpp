/* CosmicTiger - periodic near-field CUDA walk; BSL-1.0. */
#pragma once

#include <cosmictiger/cuda/cuda_reduce.hpp>
#include <cosmictiger/gravity/kick_cuda.hpp>
#include <cosmictiger/gravity/periodic_fmm.hpp>

namespace detail {
template <int order>
__device__ FlopCount cudaPeriodicFmm(CudaKickData const& data, TreeNode const& sink,
    CudaKickControl<order>& control, float theta, float smoothingLength, bool doPotential,
    cosmictiger::gravity::cartesian::Expansion<float>& expansion) {
    int const tid = threadIdx.x;
    // echecks retain both parts of the periodic force. A separately accepted
    // correction transfers its source to dchecks for the Newtonian walk;
    // a joint acceptance consumes both parts with the same image selection.
    auto& checks = control.echecks;
    auto& accepted = control.cclist;
    auto& coupled = control.pclist;
    auto& deferred = control.leaflist;
    auto& next = control.nextlist;
    accepted.resize(0);
    coupled.resize(0);
    deferred.resize(0);
    next.resize(0);
    float const inverseTheta = fmmInverseOpeningAngle(theta);
    FlopCount flops = 1;
    do {
        int const count = roundUp(checks.size(), kWarpSize);
        for (int i = tid; i < count; i += kWarpSize) {
            Array<bool, 4> selected{};
            if (i < checks.size()) {
                auto const& source = data.treeNodes[checks[i]];
                if (source.numberOfParticles()) {
                    Vector3<float> displacement;
                    for (int dim = 0; dim < kDimensions; ++dim) {
                        displacement[dim] = distance(sink.pos[dim], source.pos[dim]);
                    }
                    FlopCount acceptanceFlops;
                    bool const separated = periodicFmmAccepted(displacement, sink.radius,
                        source.radius, inverseTheta, smoothingLength, &acceptanceFlops);
                    flops += 3 + acceptanceFlops;
                    if (separated) {
                        selected[0] =
                            fmmSinglePeriodicImage(displacement, sink.radius, source.radius);
                        flops += 7;
                        if (!selected[0]) {
                            selected[1] = periodicFmmDirectAccepted(displacement, sink.radius,
                                source.radius, inverseTheta, smoothingLength, &acceptanceFlops);
                            flops += acceptanceFlops;
                        }
                    }
                    bool const unresolved = !selected[0] && !selected[1];
                    selected[2] =
                        unresolved && !source.leaf && (sink.leaf || source.radius >= sink.radius);
                    selected[3] = unresolved && !selected[2];
                    ALWAYS_ASSERT(!unresolved || !sink.leaf || !source.leaf);
                }
            }
            Array<int, 4> offsets, totals;
            for (int list = 0; list < 4; ++list) {
                offsets[list] = selected[list];
            }
            computeIndicesArray(offsets, totals);
            DeviceVector<int>* lists[4] = {&accepted, &coupled, &next, &deferred};
            for (int list = 0; list < 4; ++list) {
                int const start = lists[list]->size();
                __syncwarp();
                lists[list]->resize(start + totals[list]);
                if (selected[list]) {
                    (*lists[list])[start + offsets[list]] = checks[i];
                }
            }
        }
        __syncwarp();
        checks.resize(kChildCount * next.size());
        for (int i = tid; i < next.size(); i += kWarpSize) {
            auto const& children = data.treeNodes[next[i]].children;
            checks[kChildCount * i] = children[kLeftChild].index;
            checks[kChildCount * i + 1] = children[kRightChild].index;
        }
        __syncwarp();
        next.resize(0);
    } while (checks.size());
    if (accepted.size() || coupled.size()) {
        cosmictiger::gravity::cartesian::Expansion<float> local{};
        int const count = accepted.size() + coupled.size();
        for (int i = tid; i < count; i += kWarpSize) {
            bool const includeNewtonian = i >= accepted.size();
            int const sourceIndex = includeNewtonian ? coupled[i - accepted.size()] : accepted[i];
            auto const& source = data.treeNodes[sourceIndex];
            Vector3<float> displacement;
            for (int dim = 0; dim < kDimensions; ++dim) {
                displacement[dim] = distance(sink.pos[dim], source.pos[dim]);
            }
            flops += 3 +
                periodicFmmM2L(fmmMultiPos(source.mpos).multi, displacement, doPotential, local,
                    includeNewtonian);
        }
        sharedReduceAddArray(local);
        for (int i = tid; i < local.size(); i += kWarpSize) {
            expansion[i] += local[i];
            ++flops;
        }
        flops += 5 * local.size();
    }
    if (accepted.size()) {
        auto& directChecks = control.dchecks;
        int const directStart = directChecks.size();
        directChecks.resize(directStart + accepted.size());
        for (int i = tid; i < accepted.size(); i += kWarpSize) {
            directChecks[directStart + i] = accepted[i];
        }
    }
    ALWAYS_ASSERT(!sink.leaf || deferred.size() == 0);
    checks.resize(deferred.size());
    for (int i = tid; i < deferred.size(); i += kWarpSize) {
        checks[i] = deferred[i];
    }
    __syncwarp();
    return flops;
}
}    // namespace detail
