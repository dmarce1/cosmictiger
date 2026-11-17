/*
 CosmicTiger - runtime dispatch for compact persistent MFM particle geometry.
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <cosmictiger/mfm/particles.hpp>

namespace mfm {

template <typename Real>
void freeParticles() {
    particles<Real, 1>().clear();
    particles<Real, 2>().clear();
    particles<Real, 3>().clear();
}

template <typename Real>
void resizeParticles(size_t count, size_t capacity, int dimensions, bool enabled) {
    if (!enabled) {
        freeParticles<Real>();
        return;
    }
    if (dimensions != 1) {
        particles<Real, 1>().clear();
    }
    if (dimensions != 2) {
        particles<Real, 2>().clear();
    }
    if (dimensions != 3) {
        particles<Real, 3>().clear();
    }
    dispatchDimensions(
        dimensions, [&]<int dimension>() { particles<Real, dimension>().resize(count, capacity); });
}

template <typename Real>
void swapParticles(size_t first, size_t second, int dimensions) {
    dispatchDimensions(
        dimensions, [&]<int dimension>() { particles<Real, dimension>().swap(first, second); });
}

template <typename Real>
void saveParticles(FILE* file, int dimensions) {
    dispatchDimensions(
        dimensions, [&]<int dimension>() { particles<Real, dimension>().save(file); });
}

template <typename Real>
void loadParticles(FILE* file, size_t count, int dimensions) {
    dispatchDimensions(
        dimensions, [&]<int dimension>() { particles<Real, dimension>().load(file, count); });
}

template <typename Real>
MigrationState<Real> packParticle(size_t index, int dimensions) {
    return dispatchDimensions(dimensions, [&]<int dimension>() {
        auto const& storage = particles<Real, dimension>();
        ALWAYS_ASSERT(index < storage.size());
        MigrationState<Real> result;
        size_t component = 0;
        for (int row = 0; row < dimension; ++row) {
            for (int column = 0; column <= row; ++column) {
                result.inverseMoment[component++] = storage.inverseMoments()[index](row, column);
            }
        }
        result.geometryValid = storage.geometryValid()[index];
        return result;
    });
}

template <typename Real>
void unpackParticle(MigrationState<Real> const& record, size_t index, int dimensions) {
    dispatchDimensions(dimensions, [&]<int dimension>() {
        auto& storage = particles<Real, dimension>();
        ALWAYS_ASSERT(index < storage.size());
        if (record.geometryValid > 1) {
            throw std::runtime_error("Invalid migrated MFM geometry-valid byte");
        }
        math::SymmetricMatrix<Real, dimension> inverseMoment;
        size_t component = 0;
        for (int row = 0; row < dimension; ++row) {
            for (int column = 0; column <= row; ++column) {
                Real const value = record.inverseMoment[component++];
                if (!detail::finiteStorageValue(value)) {
                    throw std::runtime_error("Nonfinite migrated MFM inverse moment");
                }
                inverseMoment(row, column) = value;
            }
        }
        storage.inverseMoments()[index] = inverseMoment;
        storage.geometryValid()[index] = record.geometryValid;
    });
}

#define INSTANTIATE_MFM_STORAGE(Real)                               \
    template void resizeParticles<Real>(size_t, size_t, int, bool); \
    template void swapParticles<Real>(size_t, size_t, int);         \
    template void freeParticles<Real>();                            \
    template void saveParticles<Real>(FILE*, int);                  \
    template void loadParticles<Real>(FILE*, size_t, int);          \
    template MigrationState<Real> packParticle<Real>(size_t, int);  \
    template void unpackParticle<Real>(MigrationState<Real> const&, size_t, int)

INSTANTIATE_MFM_STORAGE(float);
INSTANTIATE_MFM_STORAGE(double);

#undef INSTANTIATE_MFM_STORAGE

}    // namespace mfm
