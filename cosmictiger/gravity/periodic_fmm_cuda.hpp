/* CosmicTiger - periodic near-field CUDA walk; BSL-1.0. */
#pragma once

#include <cosmictiger/cuda/cuda_reduce.hpp>
#include <cosmictiger/gravity/kick_cuda.hpp>
#include <cosmictiger/gravity/periodic_fmm.hpp>

namespace detail {
template <int order>
__device__ FlopCount cudaPeriodicFmm(CudaKickData const& data, TreeNode const& sink,
    CudaKickControl<order>& control, float theta, float smoothingLength, bool doPotential,
    cosmictiger::gravity::cartesian::Expansion<float>& expansion, float& curvature) {
    int const tid = threadIdx.x;
    auto& checks = control.echecks;
    auto& accepted = control.cclist;
    auto& deferred = control.leaflist;
    auto& next = control.nextlist;
    accepted.resize(0);
    deferred.resize(0);
    next.resize(0);
    float const inverseTheta = fmmInverseOpeningAngle(theta);
    FlopCount flops = 1;
    do {
        int const count = roundUp(checks.size(), kWarpSize);
        for (int i = tid; i < count; i += kWarpSize) {
            Array<bool, 3> selected{};
            if (i < checks.size()) {
                auto const& source = data.treeNodes[checks[i]];
                if (source.numberOfParticles()) {
                    Vector3<float> displacement;
                    for (int dim = 0; dim < kDimensions; ++dim) {
                        displacement[dim] = distance(sink.pos[dim], source.pos[dim]);
                    }
                    FlopCount acceptanceFlops;
                    selected[0] = periodicFmmAccepted(displacement, sink.radius, source.radius,
                        inverseTheta, smoothingLength, &acceptanceFlops);
                    selected[1] =
                        !selected[0] && !source.leaf && (sink.leaf || source.radius >= sink.radius);
                    selected[2] = !selected[0] && !selected[1];
                    flops += 3 + acceptanceFlops;
                }
            }
            Array<int, 3> offsets, totals;
            for (int list = 0; list < 3; ++list) {
                offsets[list] = selected[list];
            }
            computeIndicesArray(offsets, totals);
            DeviceVector<int>* lists[3] = {&accepted, &next, &deferred};
            for (int list = 0; list < 3; ++list) {
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
    if (accepted.size()) {
        cosmictiger::gravity::cartesian::Expansion<float> local{};
        float localCurvature = 0;
        for (int i = tid; i < accepted.size(); i += kWarpSize) {
            auto const& source = data.treeNodes[accepted[i]];
            Vector3<float> displacement;
            for (int dim = 0; dim < kDimensions; ++dim) {
                displacement[dim] = distance(sink.pos[dim], source.pos[dim]);
            }
            flops += 3 +
                periodicFmmM2L(fmmMultiPos(source.mpos).multi, source.radialTrace, displacement,
                    doPotential, local, localCurvature);
        }
        sharedReduceAddArray(local);
        sharedReduceAdd(localCurvature);
        curvature += localCurvature;
        for (int i = tid; i < local.size(); i += kWarpSize) {
            expansion[i] += local[i];
            ++flops;
        }
        flops += 5 * local.size() + 6;
    }
    checks.resize(deferred.size());
    for (int i = tid; i < deferred.size(); i += kWarpSize) {
        checks[i] = deferred[i];
    }
    __syncwarp();
    return flops;
}

template <int order>
__device__ FlopCount cudaPeriodicPoints(CudaKickData const& data, TreeNode const& sink,
    CudaKickControl<order>& control, CudaKickForceTile const& force, int sinkBegin, int sinkCount,
    bool doPotential) {
    FlopCount flops = 0;
    for (int i = threadIdx.x; i < sinkCount; i += kWarpSize) {
        ParticleInteger const target = sink.particleRange.first + sinkBegin + i;
        float potential = 0, gx = 0, gy = 0, gz = 0;
        for (int check = 0; check < control.echecks.size(); ++check) {
            auto const& source = data.treeNodes[control.echecks[check]];
            bool const ghost = source.sourceStore == FmmParticleStore::Ghost;
            auto const* x = ghost ? data.ghostSources.positions[0] : data.x;
            auto const* y = ghost ? data.ghostSources.positions[1] : data.y;
            auto const* z = ghost ? data.ghostSources.positions[2] : data.z;
            auto const* masses = ghost ? data.ghostSources.masses : data.sourceMasses;
            auto const* types = ghost ? data.ghostSources.types : data.type;
            for (ParticleInteger j = source.particleRange.first; j < source.particleRange.second;
                ++j) {
                Vector3<float> displacement;
                displacement[0] = distance(data.x[target], x[j]);
                displacement[1] = distance(data.y[target], y[j]);
                displacement[2] = distance(data.z[target], z[j]);
                float const mass = masses != nullptr ?
                    masses[j] :
                    (!data.sph ?
                            1.0f :
                            (types[j] == kDarkMatterType ? data.darkMatterMass : data.sphMass));
                flops +=
                    3 + periodicFmmPoint(displacement, mass, doPotential, potential, gx, gy, gz);
            }
        }
        if (control.echecks.size()) {
            force.gx[i] += gx;
            force.gy[i] += gy;
            force.gz[i] += gz;
            if (doPotential) {
                force.phi[i] += potential;
            }
            flops += 3 + int(doPotential);
        }
    }
    __syncwarp();
    return flops;
}
}    // namespace detail
