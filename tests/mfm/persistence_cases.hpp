/* CosmicTiger - inverse-moment identity checks across gas storage operations; BSL-1.0. */
#pragma once

#include <stdexcept>

#include <cosmictiger/core/options.hpp>
#include <cosmictiger/mfm/particles.hpp>

namespace detail {

/// Dispatch fixture access to the actual compact matrix stride.
template <typename Function>
void withMfmFixtureGeometry(Function&& function) {
    if (!getOptions().hydro.hopkins) {
        return;
    }
    switch (spatialDimensions()) {
    case 1:
        function.template operator()<1>();
        break;
    case 2:
        function.template operator()<2>();
        break;
    default:
        function.template operator()<3>();
        break;
    }
}

/// Assign distinct, exactly representable entries tied to a particle identity.
inline void setMfmFixtureGeometry(ParticleInteger gas, int identity) {
    withMfmFixtureGeometry([&]<int dimensions>() {
        auto& matrix = mfm::inverseMoments<float, dimensions>()[gas];
        for (int row = 0; row < dimensions; ++row) {
            for (int column = 0; column <= row; ++column) {
                matrix(row, column) = float(100 * identity + 10 * row + column + 1);
            }
        }
        mfm::geometryValid<float, dimensions>()[gas] = identity % 2;
    });
}

/// Check all packed entries and validity after a permutation, transfer, or restart.
inline void requireMfmFixtureGeometry(ParticleInteger gas, int identity) {
    withMfmFixtureGeometry([&]<int dimensions>() {
        auto const& matrix = mfm::inverseMoments<float, dimensions>()[gas];
        for (int row = 0; row < dimensions; ++row) {
            for (int column = 0; column <= row; ++column) {
                if (matrix(row, column) != float(100 * identity + 10 * row + column + 1)) {
                    throw std::runtime_error("Gas persistence detached an inverse moment matrix");
                }
            }
        }
        if (mfm::geometryValid<float, dimensions>()[gas] != identity % 2) {
            throw std::runtime_error("Gas persistence changed inverse moment validity");
        }
    });
}

}    // namespace detail
