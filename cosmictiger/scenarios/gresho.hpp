#pragma once

#include <cmath>

#include <cosmictiger/scenarios/scenario.hpp>

namespace scenarios {

struct GreshoState {
    double pressure;
    double azimuthalVelocity;
};

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
