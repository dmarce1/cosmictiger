/*
 CosmicTiger - A cosmological N-Body code
 Cooperative CUDA MFM support solves, density sums, and checked inverse moments.
 Copyright (C) 2026 Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <cmath>
#include <memory>
#include <new>
#include <type_traits>

#include <cosmictiger/cuda/cuda_reduce.hpp>
#include <cosmictiger/math/matrix.hpp>
#include <cosmictiger/mfm/cuda.hpp>
#include <cosmictiger/mfm/geometry.hpp>
#include <cosmictiger/mfm/kernel.hpp>
#include <cosmictiger/sph/neighbor_buffer.hpp>

namespace mfm {
namespace detail {

    inline constexpr int kSmoothingBlockSize = 160;
    inline constexpr int kDensityBlockSize = 64;
    inline constexpr int kSmoothingSharedBytes = 16 * 1024;
    inline constexpr int kDensitySharedBytes = 12 * 1024;

    template <typename Real, int dimensions>
    struct NeighborPosition {
        math::Vector<Position, dimensions> position;    // Periodic source coordinates in box units.
    };

    template <typename Real, int dimensions>
    struct DensityNeighbor {
        math::Vector<Position, dimensions> position;    // Periodic source coordinates in box units.
        Real smoothingLength;    // Frozen source support radius in box units.
        char rung;               // Frozen source timestep rung.
    };

    template <typename Real, int dimensions>
    struct SmoothingWorkspace {
        ::detail::SharedNeighborBuffer<NeighborPosition<Real, dimensions>,
            kSmoothingSharedBytes / sizeof(NeighborPosition<Real, dimensions>)>
            neighbors;    // Shared cache with collective CUDA-pool overflow storage.
    };

    template <typename Real, int dimensions>
    struct DensityWorkspace {
        ::detail::SharedNeighborBuffer<DensityNeighbor<Real, dimensions>,
            kDensitySharedBytes / sizeof(DensityNeighbor<Real, dimensions>)>
            neighbors;    // Positions and activity in one cache sized for the actual scalar type.
    };

    /// Build a source cache collectively, retaining sources that reach the target's box.
    template <typename Real, int dimensions, int blockSize, bool symmetric, typename Neighbors>
    __device__ void cacheNeighbors(RunData<Real, dimensions> const& data,
        SphTreeNode const& targetNode, Neighbors& neighbors) {
        neighbors.resize(0);
        for (int neighborNodeIndex = targetNode.neighborRange.first;
            neighborNodeIndex < targetNode.neighborRange.second; ++neighborNodeIndex) {
            auto const& neighborNode = data.treeNodes[data.neighborNodeIndices[neighborNodeIndex]];
            auto const end = neighborNode.particleRange.first +
                roundUp(neighborNode.particleRange.second - neighborNode.particleRange.first,
                    ParticleInteger(blockSize));
            for (ParticleInteger sourceParticleIndex =
                     neighborNode.particleRange.first + threadIdx.x;
                sourceParticleIndex < end; sourceParticleIndex += blockSize) {
                bool contains = false;
                if (sourceParticleIndex < neighborNode.particleRange.second) {
                    auto const paddedPosition = data.sourcePaddedPosition(sourceParticleIndex);
                    contains = targetNode.outerBox.contains(paddedPosition);
                    if constexpr (symmetric) {
                        if (!contains) {
                            contains = supportIntersects<Real>(targetNode.innerBox, paddedPosition,
                                data.sourceSmoothingLength[sourceParticleIndex]);
                        }
                    }
                }
                int slot = contains;
                int total;
                computeIndices<blockSize>(slot, total);
                int const offset = neighbors.size();
                __syncthreads();
                neighbors.resize(offset + total);
                if (contains) {
                    auto& neighbor = neighbors[offset + slot];
                    neighbor.position = data.sourcePosition(sourceParticleIndex);
                    if constexpr (symmetric) {
                        neighbor.smoothingLength = data.sourceSmoothingLength[sourceParticleIndex];
                        neighbor.rung = data.sourceRungs[sourceParticleIndex];
                    }
                }
            }
        }
        __syncthreads();
    }

    // clang-format off
/*
 * Hopkins (2015), section 2.2 and equations (6)-(7), arXiv:1409.7395:
 *   n𞁑 = Σ𞁓 W(x𞁑-x𞁓,h𞁑),  V𞁑 = 1/n𞁑.
 * Derived support constraint in D dimensions, using the full compact-support radius:
 *   ΣW = Σ𞁓 w(q𞁑𞁓) = Nₛₑₜ/V_D,
 *   dΣW/dh𞁑 = -Σ𞁓 q𞁑𞁓 w′(q𞁑𞁓)/h𞁑, q𞁑𞁓 = |x𞁑-x𞁓|/h𞁑.
 * Bounded Newton steps retain the existing SPH search-box retry behavior.
 * Symbol directory: h𞁑 → sink support; hₘᵢₙ,hₘₐₓ → support bounds; x𞁑 → sink position;
 *   Nₛₑₜ → weighted neighbor target; V_D → unit-ball volume; ΣWₛₑₜ → target sum;
 *   q𞁑𞁓 → normalized separation; ΣW → kernel sum; dΣW/dh𞁑 → support derivative.
 * Code names: ih𞁑 = h𞁑⁻¹; dΣW_dh𞁑 = dΣW/dh𞁑; abs_q𞁑𞁓 = |q𞁑𞁓|.
 */
    // clang-format on
    template <typename Real, int dimensions, int blockSize, typename Neighbors>
    __device__ bool solveSmoothingLength(Real& h𞁑, PrehydroParameters<Real> const& parameters,
        RunData<Real, dimensions> const& data, ParticleInteger sinkIndex,
        Neighbors const& neighbors, math::Vector<Position, dimensions> const& x𞁑,
        Vector3<Position> const& paddedPosition, SphBox const& neighborSearchBox) {
        Real const hₘᵢₙ = parameters.minimumSmoothingLength;
        Real const hₘₐₓ = parameters.maximumSmoothingLength;
        Real const Nₛₑₜ = data.targetNeighborCount;
        ALWAYS_ASSERT(finite(h𞁑) && finite(hₘᵢₙ) && finite(hₘₐₓ) && finite(Nₛₑₜ) &&
            hₘᵢₙ > Real(0) && hₘₐₓ >= hₘᵢₙ && hₘₐₓ <= Real(1) / Real(2) && Nₛₑₜ > Real(0) &&
            finite(parameters.smoothingLengthTolerance) &&
            parameters.smoothingLengthTolerance > Real(0));
        if (threadIdx.x == 0) {
            h𞁑 = math::min(math::max(h𞁑, hₘᵢₙ), hₘₐₓ);
        }
        __syncthreads();
        Real lower = hₘᵢₙ;
        Real upper = hₘₐₓ;
        Real const ΣWₛₑₜ = targetKernelSum<Real, dimensions>(Nₛₑₜ);
        for (int iteration = 0; iteration < kMaximumSmoothingIterations; ++iteration) {
            for (int dim = 0; dim < kDimensions; ++dim) {
                if (neighborSearchBox.end[dim] <
                        RangeFixed(paddedPosition[dim]) + RangeFixed(h𞁑) + RangeFixed::min() ||
                    RangeFixed(paddedPosition[dim]) <
                        neighborSearchBox.begin[dim] + RangeFixed(h𞁑) + RangeFixed::min()) {
                    return false;
                }
            }
            Real const ih𞁑 = Real(1) / h𞁑;
            Real ΣW = Real(0);
            Real dΣW_dh𞁑 = Real(0);
            for (int neighborIndex = threadIdx.x; neighborIndex < neighbors.size();
                neighborIndex += blockSize) {
                Real const abs_q𞁑𞁓 =
                    separation<Real, dimensions>(x𞁑, neighbors[neighborIndex].position).norm() *
                    ih𞁑;
                if (abs_q𞁑𞁓 < Real(1)) {
                    ΣW += kernelWeight<Real, dimensions>(abs_q𞁑𞁓);
                    dΣW_dh𞁑 -= abs_q𞁑𞁓 * kernelDerivative<Real, dimensions>(abs_q𞁑𞁓) * ih𞁑;
                }
            }
            sharedReduceAdd<Real, blockSize>(ΣW);
            sharedReduceAdd<Real, blockSize>(dΣW_dh𞁑);
            Real const residual = ΣW - ΣWₛₑₜ;
            Real const tolerance = parameters.smoothingLengthTolerance * ΣWₛₑₜ;
            if (hₘₐₓ >= Real(1) / Real(2) && h𞁑 >= hₘₐₓ && residual < -tolerance) {
                if (threadIdx.x == 0) {
                    printf("MFM smoothing length exceeds half the periodic box: h=%e sum=%e "
                           "target=%e\n",
                        double(h𞁑), double(ΣW), double(ΣWₛₑₜ));
                    __trap();
                }
            }
            bool const converged = std::abs(residual) <= tolerance ||
                (h𞁑 <= hₘᵢₙ && residual >= Real(0)) || (h𞁑 >= hₘₐₓ && residual <= Real(0));
            if (residual > Real(0)) {
                upper = h𞁑;
            } else {
                lower = h𞁑;
            }
            Real const maximumChange = h𞁑 / Real(5);
            Real change = dΣW_dh𞁑 > Real(0) ? -residual / dΣW_dh𞁑 : maximumChange;
            change = math::min(math::max(change, -maximumChange), maximumChange);
            Real next = math::min(math::max(h𞁑 + change, hₘᵢₙ), hₘₐₓ);
            if (next < lower || next > upper) {
                next = (lower + upper) / Real(2);
            }
            if (converged || next == h𞁑) {
                if (threadIdx.x == 0 && parameters.initializeDensity) {
                    data.sinkDensity[sinkIndex] =
                        data.gasMass * inverseVolume<Real, dimensions>(ih𞁑) * ΣW;
                }
                return true;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                h𞁑 = next;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            printf("MFM smoothing-length solve did not converge: h=%e bounds=[%e,%e] target=%e\n",
                double(h𞁑), double(hₘᵢₙ), double(hₘₐₓ), double(Nₛₑₜ));
            __trap();
        }
        return false;
    }

    /// Discard geometry before changing its support or attempting a new density pass.
    template <typename Real, int dimensions>
    __device__ void invalidateGeometryDevice(
        RunData<Real, dimensions> const& data, ParticleInteger sinkIndex) {
        if (threadIdx.x == 0) {
            data.sinkInverseMoment[sinkIndex] = math::SymmetricMatrix<Real, dimensions>{};
            data.sinkGeometryValid[sinkIndex] = 0;
        }
    }

    /// Solve one target leaf per block, retaining the neighbor allocation between leaves.
    template <typename Real, int dimensions>
    __global__ void smoothingLengthsKernel(PrehydroParameters<Real> parameters,
        RunData<Real, dimensions> data, PrehydroResult<Real>* nodeResults) {
        using Workspace = SmoothingWorkspace<Real, dimensions>;
        alignas(Workspace) __shared__ unsigned char workspaceStorage[sizeof(Workspace)];
        // Every lane participates in construction. Default initialization avoids an
        // implicit shared-memory zero-fill before the collective buffer constructors.
        auto& workspace = *new (workspaceStorage) Workspace;
        for (int targetNodeIndex = blockIdx.x; targetNodeIndex < data.numberOfTargetNodes;
            targetNodeIndex += gridDim.x) {
            if (threadIdx.x == 0) {
                nodeResults[targetNodeIndex] = PrehydroResult<Real>{};
            }
            auto const& targetNode = data.treeNodes[data.targetNodeIndices[targetNodeIndex]];
            cacheNeighbors<Real, dimensions, kSmoothingBlockSize, false>(
                data, targetNode, workspace.neighbors);
            for (ParticleInteger particleIndex = targetNode.particleRange.first;
                particleIndex < targetNode.particleRange.second; ++particleIndex) {
                __syncthreads();
                auto const sinkIndex = targetNode.sinkParticleRange.first -
                    targetNode.particleRange.first + particleIndex;
                bool const active = parameters.isActiveRung(
                    data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]]);
                if (active && !data.sinkFlags[sinkIndex].converged()) {
                    invalidateGeometryDevice(data, sinkIndex);
                    Real& h𞁑 = data.sinkSmoothingLength[sinkIndex];
                    bool const converged =
                        solveSmoothingLength<Real, dimensions, kSmoothingBlockSize>(h𞁑, parameters,
                            data, sinkIndex, workspace.neighbors,
                            data.sourcePosition(particleIndex),
                            data.sourcePaddedPosition(particleIndex), targetNode.outerBox);
                    if (threadIdx.x == 0) {
                        data.sinkFlags[sinkIndex].setConverged(converged);
                        auto& result = nodeResults[targetNodeIndex];
                        if (converged) {
                            result.minimumSmoothingLength =
                                math::min(result.minimumSmoothingLength, h𞁑);
                            result.maximumSmoothingLength =
                                math::max(result.maximumSmoothingLength, h𞁑);
                        } else {
                            result.needsAnotherPass = true;
                        }
                    }
                }
            }
            __syncthreads();
        }
        workspace.~Workspace();
    }

    // clang-format off
/*
 * Hopkins (2015), equations (6)-(7), (12)-(14), arXiv:1409.7395:
 *   ψ𞁓(x𞁑) = W𞁑𞁓/Σ𞁞W𞁑𞁞,
 *   E𞁑 = Σ𞁓 (x𞁓-x𞁑)⊗(x𞁓-x𞁑) ψ𞁓(x𞁑), B𞁑 = E𞁑⁻¹.
 * Derived accumulation with q𞁑𞁓=(x𞁑-x𞁓)/h𞁑 and equal gas masses:
 *   ΣW = Σ𞁓 w(|q𞁑𞁓|), M𞁑 = Σ𞁓 w(|q𞁑𞁓|) q𞁑𞁓⊗q𞁑𞁓,
 *   ρ𞁑 = m h𞁑⁻ᴰ ΣW, E𞁑 = h𞁑² M𞁑/ΣW.
 * Symbol directory: h𞁑 → sink support; x𞁑 → sink position; m → equal gas mass;
 *   q𞁑𞁓 → dimensionless separation; ρ𞁑 → mass density; M𞁑 → unnormalized moment;
 *   E𞁑 → physical normalized moment; B𞁑 → its checked inverse; ΣW → kernel sum.
 * Code names: ih𞁑 = h𞁑⁻¹; abs_q𞁑𞁓 = |q𞁑𞁓|; w_q𞁑𞁓 = w(|q𞁑𞁓|).
 */
    // clang-format on
    template <typename Real, int dimensions>
    __global__ void densityAndGeometryKernel(PrehydroParameters<Real> parameters,
        RunData<Real, dimensions> data, PrehydroResult<Real>* nodeResults) {
        using Workspace = DensityWorkspace<Real, dimensions>;
        alignas(Workspace) __shared__ unsigned char workspaceStorage[sizeof(Workspace)];
        // The buffer constructor/destructor synchronize the whole block, including
        // when the cache spills into and later releases CUDA-pool storage.
        auto& workspace = *new (workspaceStorage) Workspace;
        for (int targetNodeIndex = blockIdx.x; targetNodeIndex < data.numberOfTargetNodes;
            targetNodeIndex += gridDim.x) {
            if (threadIdx.x == 0) {
                nodeResults[targetNodeIndex] = PrehydroResult<Real>{};
            }
            auto const& targetNode = data.treeNodes[data.targetNodeIndices[targetNodeIndex]];
            cacheNeighbors<Real, dimensions, kDensityBlockSize, true>(
                data, targetNode, workspace.neighbors);
            for (ParticleInteger particleIndex = targetNode.particleRange.first;
                particleIndex < targetNode.particleRange.second; ++particleIndex) {
                __syncthreads();
                auto const sinkIndex = targetNode.sinkParticleRange.first -
                    targetNode.particleRange.first + particleIndex;
                bool const active = parameters.isActiveRung(
                    data.sinkParticleRungs[data.sinkParticleIndices[sinkIndex]]);
                auto const x𞁑 = data.sourcePosition(particleIndex);
                Real& h𞁑 = data.sinkSmoothingLength[sinkIndex];
                int semiactive = active;
                if (!active) {
                    for (int neighborIndex = threadIdx.x;
                        neighborIndex < workspace.neighbors.size();
                        neighborIndex += kDensityBlockSize) {
                        auto const& neighbor = workspace.neighbors[neighborIndex];
                        if (parameters.isActiveRung(neighbor.rung)) {
                            auto const r𞁑𞁓 = separation<Real, dimensions>(x𞁑, neighbor.position);
                            Real const h𞁓 = neighbor.smoothingLength;
                            if (r𞁑𞁓.dot(r𞁑𞁓) < math::max(h𞁑 * h𞁑, h𞁓 * h𞁓)) {
                                ++semiactive;
                            }
                        }
                    }
                    sharedReduceAdd<int, kDensityBlockSize>(semiactive);
                }
                if (threadIdx.x == 0) {
                    data.sinkFlags[sinkIndex].setSemiactive(semiactive != 0);
                }
                // Finish the byte containing both flags before any lane reads convergence.
                __syncthreads();
                if (!semiactive) {
                    continue;
                }
                invalidateGeometryDevice(data, sinkIndex);
                bool converged = true;
                if (!active && !data.sinkFlags[sinkIndex].converged()) {
                    converged = solveSmoothingLength<Real, dimensions, kDensityBlockSize>(h𞁑,
                        parameters, data, sinkIndex, workspace.neighbors, x𞁑,
                        data.sourcePaddedPosition(particleIndex), targetNode.outerBox);
                }
                if (!converged) {
                    if (threadIdx.x == 0) {
                        data.sinkFlags[sinkIndex].setConverged(false);
                        nodeResults[targetNodeIndex].needsAnotherPass = true;
                    }
                    continue;
                }
                if (!finite(h𞁑) || !(h𞁑 > Real(0))) {
                    if (threadIdx.x == 0) {
                        ++nodeResults[targetNodeIndex].geometryFailures;
                    }
                    continue;
                }
                Real const ih𞁑 = Real(1) / h𞁑;
                Real ΣW = Real(0);
                math::SymmetricMatrix<Real, dimensions> M𞁑;
                for (int neighborIndex = threadIdx.x; neighborIndex < workspace.neighbors.size();
                    neighborIndex += kDensityBlockSize) {
                    auto const q𞁑𞁓 = separation<Real, dimensions>(
                                         x𞁑, workspace.neighbors[neighborIndex].position) *
                        ih𞁑;
                    Real const abs_q𞁑𞁓 = q𞁑𞁓.norm();
                    if (abs_q𞁑𞁓 < Real(1)) {
                        Real const w_q𞁑𞁓 = kernelWeight<Real, dimensions>(abs_q𞁑𞁓);
                        ΣW += w_q𞁑𞁓;
                        M𞁑 += math::symmetricOuterProduct(q𞁑𞁓) * w_q𞁑𞁓;
                    }
                }
                sharedReduceAdd<Real, kDensityBlockSize>(ΣW);
                for (int row = 0; row < dimensions; ++row) {
                    for (int column = 0; column <= row; ++column) {
                        sharedReduceAdd<Real, kDensityBlockSize>(M𞁑(row, column));
                    }
                }
                if (threadIdx.x == 0) {
                    Real const ρ𞁑 = data.gasMass * inverseVolume<Real, dimensions>(ih𞁑) * ΣW;
                    data.sinkDensity[sinkIndex] = ρ𞁑;
                    bool const valid = finite(ρ𞁑) && ρ𞁑 > Real(0) &&
                        invertMoment<Real, dimensions>(M𞁑, ΣW, h𞁑,
                            parameters.maximumMomentCondition, data.sinkInverseMoment[sinkIndex]);
                    data.sinkGeometryValid[sinkIndex] = valid;
                    data.sinkFlags[sinkIndex].setConverged(true);
                    auto& result = nodeResults[targetNodeIndex];
                    result.geometryFailures += !valid;
                    result.minimumSmoothingLength = math::min(result.minimumSmoothingLength, h𞁑);
                    result.maximumSmoothingLength = math::max(result.maximumSmoothingLength, h𞁑);
                }
            }
            __syncthreads();
        }
        workspace.~Workspace();
    }

    /// Launch typed prehydro work and combine per-node diagnostics without Real atomics.
    template <typename Real, int dimensions, bool density>
    PrehydroResult<Real> launchPrehydro(PrehydroParameters<Real> const& parameters,
        RunData<Real, dimensions> const& data, CudaStream& stream) {
        static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>);
        static_assert(dimensions >= 1 && dimensions <= kDimensions);
        PrehydroResult<Real> result;
        if (data.numberOfTargetNodes <= 0) {
            return result;
        }
        PrehydroResult<Real>* nodeResults = nullptr;
        CUDA_CHECK(cudaMallocManaged(
            &nodeResults, sizeof(PrehydroResult<Real>) * size_t(data.numberOfTargetNodes)));
        auto const releaseResults = [&stream](PrehydroResult<Real>* pointer) noexcept {
            try {
                stream.wait();
            } catch (...) {
            }
            cudaFree(pointer);
        };
        std::unique_ptr<PrehydroResult<Real>, decltype(releaseResults)> owner(
            nodeResults, releaseResults);
        int blocksPerMultiprocessor = 0;
        if constexpr (density) {
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerMultiprocessor,
                densityAndGeometryKernel<Real, dimensions>, kDensityBlockSize, 0));
            int const blockCount = std::min(
                data.numberOfTargetNodes, std::max(1, blocksPerMultiprocessor * cudaSmpCount()));
            densityAndGeometryKernel<Real, dimensions>
                <<<blockCount, kDensityBlockSize, 0, stream.nativeHandle()>>>(
                    parameters, data, nodeResults);
        } else {
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerMultiprocessor,
                smoothingLengthsKernel<Real, dimensions>, kSmoothingBlockSize, 0));
            int const blockCount = std::min(
                data.numberOfTargetNodes, std::max(1, blocksPerMultiprocessor * cudaSmpCount()));
            smoothingLengthsKernel<Real, dimensions>
                <<<blockCount, kSmoothingBlockSize, 0, stream.nativeHandle()>>>(
                    parameters, data, nodeResults);
        }
        stream.checkLaunch();
        stream.wait();
        for (int index = 0; index < data.numberOfTargetNodes; ++index) {
            result += nodeResults[index];
        }
        return result;
    }

}    // namespace detail

template <typename Real, int dimensions>
PrehydroResult<Real> smoothingLengthsCuda(PrehydroParameters<Real> const& parameters,
    RunData<Real, dimensions> const& data, CudaStream& stream) {
    return detail::launchPrehydro<Real, dimensions, false>(parameters, data, stream);
}

template <typename Real, int dimensions>
PrehydroResult<Real> densityAndGeometryCuda(PrehydroParameters<Real> const& parameters,
    RunData<Real, dimensions> const& data, CudaStream& stream) {
    return detail::launchPrehydro<Real, dimensions, true>(parameters, data, stream);
}

#define COSMICTIGER_INSTANTIATE_MFM_CUDA(Real, dimensions)                               \
    template PrehydroResult<Real> smoothingLengthsCuda<Real, dimensions>(                \
        PrehydroParameters<Real> const&, RunData<Real, dimensions> const&, CudaStream&); \
    template PrehydroResult<Real> densityAndGeometryCuda<Real, dimensions>(              \
        PrehydroParameters<Real> const&, RunData<Real, dimensions> const&, CudaStream&)

COSMICTIGER_INSTANTIATE_MFM_CUDA(float, 1);
COSMICTIGER_INSTANTIATE_MFM_CUDA(float, 2);
COSMICTIGER_INSTANTIATE_MFM_CUDA(float, 3);
COSMICTIGER_INSTANTIATE_MFM_CUDA(double, 1);
COSMICTIGER_INSTANTIATE_MFM_CUDA(double, 2);
COSMICTIGER_INSTANTIATE_MFM_CUDA(double, 3);

#undef COSMICTIGER_INSTANTIATE_MFM_CUDA

}    // namespace mfm
