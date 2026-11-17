/*
 CosmicTiger - compact persistent MFM inverse moments and geometry validity.
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <cosmictiger/cuda/cuda.hpp>
#include <cosmictiger/math/matrix.hpp>

namespace mfm {

namespace detail {

    /// Check IEEE exponent bits even in builds that assume finite floating arithmetic.
    template <typename Real>
    bool finiteStorageValue(Real value) {
        static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>);
        if constexpr (std::is_same_v<Real, float>) {
            constexpr uint32_t kExponentMask = 0x7f800000U;
            return (std::bit_cast<uint32_t>(value) & kExponentMask) != kExponentMask;
        } else {
            constexpr uint64_t kExponentMask = 0x7ff0000000000000ULL;
            return (std::bit_cast<uint64_t>(value) & kExponentMask) != kExponentMask;
        }
    }

}    // namespace detail

/// Select one exact compact dimension without allocating padded particle records.
template <typename Function>
decltype(auto) dispatchDimensions(int dimensions, Function&& function) {
    switch (dimensions) {
    case 1:
        return function.template operator()<1>();
    case 2:
        return function.template operator()<2>();
    case 3:
        return function.template operator()<3>();
    default:
        throw std::invalid_argument("MFM particle dimensions must be 1, 2, or 3");
    }
}

/// Own a gas-indexed array of B=E^-1 records and an explicit geometry-valid byte.
/// This owner is also usable independently of the production float specialization.
template <typename Real, int dimensions>
class ParticleStorage {
    static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>);
    static_assert(dimensions >= 1 && dimensions <= kDimensions);

public:
    using InverseMoment = math::SymmetricMatrix<Real, dimensions>;
    static constexpr size_t kComponents_ =
        dimensions * (dimensions + 1) / 2;    // Independent entries per packed matrix.

private:
    static_assert(sizeof(InverseMoment) == kComponents_ * sizeof(Real));
    static_assert(std::is_trivially_copyable_v<InverseMoment>);
    static_assert(std::numeric_limits<Real>::is_iec559);
    InverseMoment* inverseMoments_ = nullptr;    // Packed inverse moments, one record per gas.
    uint8_t* geometryValid_ = nullptr;           // One only after successful geometry formation.
    size_t size_ = 0;                            // Number of live gas records.
    size_t capacity_ = 0;                        // Number of allocated records in both arrays.

    /// Validate transport data without inferring validity from matrix values.
    void validate() const {
        for (size_t index = 0; index < size_; ++index) {
            if (geometryValid_[index] > 1) {
                throw std::runtime_error("Invalid MFM geometry-valid byte");
            }
            for (int row = 0; row < dimensions; ++row) {
                for (int column = 0; column <= row; ++column) {
                    if (!detail::finiteStorageValue(inverseMoments_[index](row, column))) {
                        throw std::runtime_error("Nonfinite MFM inverse moment");
                    }
                }
            }
        }
    }

    /// Record native precision, packed lower-triangle layout, and spatial extent.
    static constexpr std::array<uint32_t, 7> checkpointHeader() {
        return {0x43544d46U, 1U, dimensions, sizeof(Real), std::is_same_v<Real, float> ? 1U : 2U,
            1U, kComponents_};
    }

public:
    ParticleStorage() = default;
    ParticleStorage(ParticleStorage const&) = delete;
    ParticleStorage& operator=(ParticleStorage const&) = delete;
    ~ParticleStorage() {
        clear();
    }

    /// Grow with managed storage; every newly live geometry starts zero and invalid.
    void resize(size_t count, size_t reserve = 0) {
        size_t const requestedCapacity = std::max(count, reserve);
        if (requestedCapacity > capacity_) {
            if (requestedCapacity > std::numeric_limits<size_t>::max() / sizeof(InverseMoment)) {
                throw std::length_error("MFM particle allocation is too large");
            }
            auto* moments = static_cast<InverseMoment*>(
                allocateManagedMemory(requestedCapacity * sizeof(InverseMoment)));
            uint8_t* valid = nullptr;
            try {
                valid = static_cast<uint8_t*>(allocateManagedMemory(requestedCapacity));
            } catch (...) {
                freeManagedMemory(moments);
                throw;
            }
            for (size_t index = 0; index < requestedCapacity; ++index) {
                ::new (static_cast<void*>(moments + index)) InverseMoment{};
                valid[index] = 0;
            }
            for (size_t index = 0; index < size_; ++index) {
                moments[index] = inverseMoments_[index];
                valid[index] = geometryValid_[index];
            }
            freeManagedMemory(inverseMoments_);
            freeManagedMemory(geometryValid_);
            inverseMoments_ = moments;
            geometryValid_ = valid;
            capacity_ = requestedCapacity;
        }
        for (size_t index = size_; index < count; ++index) {
            inverseMoments_[index] = InverseMoment{};
            geometryValid_[index] = 0;
        }
        size_ = count;
    }

    /// Keep both fields attached to their gas record during sorting and removal.
    void swap(size_t first, size_t second) {
        ALWAYS_ASSERT(first < size_ && second < size_);
        std::swap(inverseMoments_[first], inverseMoments_[second]);
        std::swap(geometryValid_[first], geometryValid_[second]);
    }

    /// Release storage; a disabled Hopkins run owns no MFM arrays.
    void clear() noexcept {
        freeManagedMemory(inverseMoments_);
        freeManagedMemory(geometryValid_);
        inverseMoments_ = nullptr;
        geometryValid_ = nullptr;
        size_ = 0;
        capacity_ = 0;
    }

    size_t size() const {
        return size_;
    }
    size_t capacity() const {
        return capacity_;
    }
    InverseMoment* inverseMoments() {
        return inverseMoments_;
    }
    InverseMoment const* inverseMoments() const {
        return inverseMoments_;
    }
    uint8_t* geometryValid() {
        return geometryValid_;
    }
    uint8_t const* geometryValid() const {
        return geometryValid_;
    }

    /// Append an explicitly versioned native checkpoint extension after the SPH payload.
    void save(FILE* file) const {
        validate();
        auto const header = checkpointHeader();
        uint64_t const count = size_;
        if (fwrite(header.data(), sizeof(uint32_t), header.size(), file) != header.size() ||
            fwrite(&count, sizeof(count), 1, file) != 1 ||
            (size_ != 0 &&
                (fwrite(inverseMoments_, sizeof(InverseMoment), size_, file) != size_ ||
                    fwrite(geometryValid_, sizeof(uint8_t), size_, file) != size_))) {
            throw std::runtime_error("Unable to write MFM particle checkpoint");
        }
    }

    /// Reject layout/precision mismatches before consuming any matrix records.
    void load(FILE* file, size_t expectedCount) {
        std::array<uint32_t, 7> header{};
        uint64_t count = 0;
        if (fread(header.data(), sizeof(uint32_t), header.size(), file) != header.size() ||
            fread(&count, sizeof(count), 1, file) != 1) {
            throw std::runtime_error("Truncated MFM particle checkpoint header");
        }
        if (header != checkpointHeader() || count != expectedCount) {
            throw std::runtime_error(
                "Incompatible MFM checkpoint dimension, precision, layout, or count");
        }
        ParticleStorage restored;
        restored.resize(expectedCount, capacity_);
        if (expectedCount != 0 &&
            (fread(restored.inverseMoments_, sizeof(InverseMoment), expectedCount, file) !=
                    expectedCount ||
                fread(restored.geometryValid_, sizeof(uint8_t), expectedCount, file) !=
                    expectedCount)) {
            throw std::runtime_error("Truncated MFM particle checkpoint data");
        }
        restored.validate();
        std::swap(inverseMoments_, restored.inverseMoments_);
        std::swap(geometryValid_, restored.geometryValid_);
        std::swap(size_, restored.size_);
        std::swap(capacity_, restored.capacity_);
    }
};

/// Access the single local owner for this precision and compact runtime dimension.
template <typename Real, int dimensions>
ParticleStorage<Real, dimensions>& particles() {
    static ParticleStorage<Real, dimensions> storage;
    return storage;
}

template <typename Real, int dimensions>
math::SymmetricMatrix<Real, dimensions>* inverseMoments() {
    return particles<Real, dimensions>().inverseMoments();
}

template <typename Real, int dimensions>
uint8_t* geometryValid() {
    return particles<Real, dimensions>().geometryValid();
}

template <typename Real>
void resizeParticles(size_t count, size_t capacity, int dimensions, bool enabled);
template <typename Real>
void swapParticles(size_t first, size_t second, int dimensions);
template <typename Real>
void freeParticles();
template <typename Real>
void saveParticles(FILE* file, int dimensions);
template <typename Real>
void loadParticles(FILE* file, size_t count, int dimensions);

/// Temporary transport record; only the active packed entries enter an MPI payload.
/// Persistent particle storage above always has exactly 1, 3, or 6 Real entries.
template <typename Real>
struct MigrationState {
    std::array<Real, kDimensions*(kDimensions + 1) / 2> inverseMoment{};    // Lower triangle.
    uint8_t geometryValid = 0;    // One when the transported inverse is usable.

    template <typename Archive>
    void serialize(Archive& archive, int dimensions) {
        if (dimensions < 1 || dimensions > kDimensions) {
            throw std::invalid_argument("Invalid MFM migration dimension count");
        }
        int const components = dimensions * (dimensions + 1) / 2;
        for (int component = 0; component < components; ++component) {
            archive& inverseMoment[component];
        }
        archive & geometryValid;
    }
};

template <typename Real>
MigrationState<Real> packParticle(size_t index, int dimensions);
template <typename Real>
void unpackParticle(MigrationState<Real> const& record, size_t index, int dimensions);

}    // namespace mfm
