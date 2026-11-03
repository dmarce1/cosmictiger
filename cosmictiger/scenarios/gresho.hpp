#pragma once

#include <array>
#include <cmath>

#include <cosmictiger/scenarios/scenario.hpp>

namespace scenarios {

struct GreshoState {
    double pressure;
    double azimuthalVelocity;
};

// The four sites of a conventional FCC cell, shifted away from the box faces.
inline std::array<double, 3> greshoLatticePosition(
    int ix, int iy, int iz, int phase, int side, bool closePack) {
    constexpr std::array<std::array<double, 3>, 4> kFccOffsets{{
        {{0.25, 0.25, 0.25}},
        {{0.25, 0.75, 0.75}},
        {{0.75, 0.25, 0.75}},
        {{0.75, 0.75, 0.25}},
    }};
    auto const offset = closePack ? kFccOffsets[phase] : std::array<double, 3>{0.5, 0.5, 0.5};
    return {(ix + offset[0]) / side, (iy + offset[1]) / side, (iz + offset[2]) / side};
}

// Standard Gresho-Chan vortex, rho=1; Hopkins (2015), section 4.2.3.
// https://doi.org/10.1093/mnras/stv195
inline GreshoState greshoState(double radius) {
    if (radius < 0.2) {
        return {5.0 + 12.5 * radius * radius, 5.0 * radius};
    } else if (radius < 0.4) {
        return {9.0 + 12.5 * radius * radius - 20.0 * radius + 4.0 * std::log(5.0 * radius),
            2.0 - 5.0 * radius};
    }
    return {3.0 + 4.0 * std::log(2.0), 0.0};
}

ScenarioPlan initializeGresho();

}    // namespace scenarios
