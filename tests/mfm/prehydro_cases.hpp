/* CosmicTiger - shared CPU/CUDA MFM geometry fixtures and independent references; BSL-1.0. */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <new>
#include <numbers>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/mfm/cpu.hpp>

namespace detail {

template <typename Real, int dimensions>
class MfmFixture {
    std::vector<std::function<void()>> allocations_;    // Destructors for owned staged arrays.

    template <typename Value>
    Value* allocate(size_t count) {
        auto* pointer = static_cast<Value*>(allocateManagedMemory(count * sizeof(Value)));
        for (size_t i = 0; i < count; ++i) {
            new (pointer + i) Value{};
        }
        allocations_.push_back([pointer, count]() {
            for (size_t i = 0; i < count; ++i) {
                pointer[i].~Value();
            }
            freeManagedMemory(pointer);
        });
        return pointer;
    }

public:
    static constexpr int kSourceOffset = 3;
    static constexpr int kSinkOffset = 2;
    static constexpr int kParticleOffset = 5;
    int count_;    // Number of real source/sink particles, excluding offset sentinels.
    mfm::PrehydroParameters<Real> parameters_;    // Independent preparation controls.
    mfm::RunData<Real, dimensions> data_;         // View of immutable sources and mutable sinks.
    Position* positions_[kDimensions];    // Mutable fixture owners of the staged coordinates.
    Real* sourceSupports_;                // Mutable fixture owner of frozen source supports.
    char* sourceRungs_;                   // Mutable fixture owner of frozen source rungs.
    SphTreeNode* nodes_;                  // Two leaves, deliberately dispatched in reverse order.

    explicit MfmFixture(int count = 37)
      : count_(count) {
        constexpr int kNodeCount = 2;
        size_t const sourceCount = count + kSourceOffset;
        size_t const sinkCount = count + kSinkOffset;
        for (int axis = 0; axis < kDimensions; ++axis) {
            positions_[axis] = allocate<Position>(sourceCount);
        }
        data_.sourcePositionX = positions_[kXi];
        data_.sourcePositionY = positions_[kYi];
        data_.sourcePositionZ = positions_[kZi];
        data_.sourceSmoothingLength = sourceSupports_ = allocate<Real>(sourceCount);
        data_.sourceRungs = sourceRungs_ = allocate<char>(sourceCount);
        data_.sinkDensity = allocate<Real>(sinkCount);
        data_.sinkSmoothingLength = allocate<Real>(sinkCount);
        data_.sinkInverseMoment = allocate<math::SymmetricMatrix<Real, dimensions>>(sinkCount);
        data_.sinkGeometryValid = allocate<uint8_t>(sinkCount);
        data_.sinkFlags = allocate<SphParticleFlags>(sinkCount);
        data_.sinkParticleIndices = allocate<ParticleInteger>(sinkCount);
        data_.sinkParticleRungs = allocate<char>(count + kParticleOffset);
        data_.treeNodes = nodes_ = allocate<SphTreeNode>(kNodeCount);
        auto* targets = allocate<int>(kNodeCount);
        auto* neighbors = allocate<int>(kNodeCount);
        data_.targetNodeIndices = targets;
        data_.neighborNodeIndices = neighbors;
        data_.numberOfTargetNodes = kNodeCount;
        data_.gasMass = Real(1) / Real(count);
        data_.targetNeighborCount = Real(16);
        parameters_.minimumRung = 1;
        parameters_.minimumSmoothingLength = Real(0.05);
        parameters_.maximumSmoothingLength = Real(0.4);
        for (int i = 0; i < count; ++i) {
            for (int axis = 0; axis < kDimensions; ++axis) {
                positions_[axis][source(i)] =
                    axis < dimensions ? 0.5 + 0.08 * std::sin(double((i + 1) * (axis + 2))) : 0.5;
            }
            sourceSupports_[source(i)] = data_.sinkSmoothingLength[sink(i)] = Real(0.3);
            sourceRungs_[source(i)] = data_.sinkParticleRungs[particle(i)] = 1;
            data_.sinkParticleIndices[sink(i)] = particle(i);
            data_.sinkFlags[sink(i)].setConverged(true);
        }
        for (int node = 0; node < kNodeCount; ++node) {
            targets[node] = kNodeCount - node - 1;
            neighbors[node] = node;
            nodes_[node].particleRange = {
                source(node * count / kNodeCount), source((node + 1) * count / kNodeCount)};
            nodes_[node].sinkParticleRange = {
                sink(node * count / kNodeCount), sink((node + 1) * count / kNodeCount)};
            nodes_[node].neighborRange = {0, kNodeCount};
            for (int axis = 0; axis < kDimensions; ++axis) {
                nodes_[node].innerBox.begin[axis] = 0.0;
                nodes_[node].innerBox.end[axis] = 1.0;
                nodes_[node].outerBox.begin[axis] = -1.0;
                nodes_[node].outerBox.end[axis] = 2.0;
            }
        }
    }

    MfmFixture(MfmFixture const&) = delete;
    MfmFixture& operator=(MfmFixture const&) = delete;

    ~MfmFixture() {
        for (auto const& release : allocations_) {
            release();
        }
    }

    int source(int i) const {
        return kSourceOffset + i;
    }
    int sink(int i) const {
        return kSinkOffset + i;
    }
    int particle(int i) const {
        return kParticleOffset + count_ - i - 1;
    }

    mfm::PrehydroResult<Real> densityCpu() {
        mfm::PrehydroResult<Real> result;
        for (int node = 0; node < data_.numberOfTargetNodes; ++node) {
            result += mfm::densityAndGeometryCpu(parameters_, data_, node);
        }
        return result;
    }

    mfm::PrehydroResult<Real> smoothingCpu() {
        mfm::PrehydroResult<Real> result;
        for (int node = 0; node < data_.numberOfTargetNodes; ++node) {
            result += mfm::smoothingLengthsCpu(parameters_, data_, node);
        }
        return result;
    }
};

inline void requireMfmClose(double actual, double expected, double tolerance) {
    if (!mfm::finite(actual) || !mfm::finite(expected) ||
        std::abs(actual - expected) > tolerance * std::max(1.0, std::abs(expected))) {
        throw std::runtime_error("MFM reference mismatch: " + std::to_string(actual) + " versus " +
            std::to_string(expected));
    }
}

// Independent piecewise reference in long double; it does not call the specialized kernel.
template <int dimensions>
long double referenceMfmWeight(long double radius) {
    long double const normalization = dimensions == 1 ?
        4.0L / 3.0L :
        (dimensions == 2 ? 40.0L / (7.0L * std::numbers::pi_v<long double>) :
                           8.0L / std::numbers::pi_v<long double>);
    if (radius >= 1) {
        return 0;
    }
    if (radius <= 0.5L) {
        return normalization * (1 - 6 * radius * radius * (1 - radius));
    }
    return normalization * 2 * std::pow(1 - radius, 3);
}

/// Check rho and E B = I using independently reconstructed periodic kernel moments.
template <typename Real, int dimensions>
void checkMfmReference(MfmFixture<Real, dimensions> const& fixture) {
    double const tolerance = std::is_same_v<Real, float> ? 2e-5 : 2e-12;
    for (int i = 0; i < fixture.count_; ++i) {
        int const sink = fixture.sink(i);
        if (!fixture.data_.sinkGeometryValid[sink]) {
            throw std::runtime_error("MFM rejected a well-conditioned cloud");
        }
        long double weightSum = 0;
        long double moment[dimensions][dimensions]{};
        long double const support = fixture.data_.sinkSmoothingLength[sink];
        for (int j = 0; j < fixture.count_; ++j) {
            long double displacement[dimensions]{};
            long double radiusSquared = 0;
            for (int axis = 0; axis < dimensions; ++axis) {
                // Fixed-point subtraction is evaluated in double before any Real rounding.
                displacement[axis] = doubleDistance(fixture.positions_[axis][fixture.source(i)],
                    fixture.positions_[axis][fixture.source(j)]);
                radiusSquared += displacement[axis] * displacement[axis];
            }
            long double const weight =
                referenceMfmWeight<dimensions>(std::sqrt(radiusSquared) / support);
            weightSum += weight;
            for (int row = 0; row < dimensions; ++row) {
                for (int column = 0; column < dimensions; ++column) {
                    moment[row][column] += weight * displacement[row] * displacement[column];
                }
            }
        }
        requireMfmClose(fixture.data_.sinkDensity[sink],
            fixture.data_.gasMass * weightSum / std::pow(support, dimensions), tolerance);
        auto const& inverse = fixture.data_.sinkInverseMoment[sink];
        for (int row = 0; row < dimensions; ++row) {
            for (int column = 0; column < dimensions; ++column) {
                long double product = 0;
                for (int inner = 0; inner < dimensions; ++inner) {
                    product += moment[row][inner] / weightSum * inverse(inner, column);
                }
                requireMfmClose(product, row == column ? 1.0 : 0.0, tolerance);
            }
        }
    }
}

}    // namespace detail
