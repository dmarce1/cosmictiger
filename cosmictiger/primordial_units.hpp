/* CosmicTiger - primordial chemistry host units; GPL-2.0-or-later. */
#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include <boost/units/dimensionless_type.hpp>

#include <cosmictiger/primordial_types.hpp>
#include <cosmictiger/units.hpp>

namespace primordial {
using Dimensionless = units::Quantity<boost::units::dimensionless_type>;
using NumberDensity =
    units::Quantity<boost::units::derived_dimension<boost::units::length_base_dimension, -3>::type>;
// Exact SI electron volt, converted from joules to erg (NIST CODATA 2022):
// https://physics.nist.gov/cuu/pdf/all.pdf . Match the chemistry rate/EOS
// convention rather than the application's older, rounded global constant.
inline constexpr auto kElectronVoltEnergy = units::Energy::from_value(1.602176634e-12);
static_assert(static_cast<float>(kElectronVoltEnergy.value()) == kElectronVolt);
// Typed CGS quantities stop here. The device payload uses eV/H, cm^-3, K
// and seconds; tiny CGS energy densities never enter a float calculation.
inline float checkedScalar(double value, char const* name) {
    if (!std::isfinite(value) || std::fabs(value) > std::numeric_limits<float>::max() ||
        (value != 0.0 && std::fabs(value) < std::numeric_limits<float>::min())) {
        throw std::range_error(std::string("Chemistry boundary cannot represent ") + name);
    }
    return static_cast<float>(value);
}
inline float energyPerHydrogen(units::PhysicalSpecificEnergy energy, double hydrogenMassFraction) {
    units::detail::requirePositiveFinite(hydrogenMassFraction, "hydrogen mass fraction");
    Dimensionless const scaled =
        energy.quantity() * units::kHydrogenMass / (hydrogenMassFraction * kElectronVoltEnergy);
    return checkedScalar(scaled.value(), "thermal energy in eV/H");
}
inline units::PhysicalSpecificEnergy physicalEnergy(
    float energyEvPerHydrogen, double hydrogenMassFraction) {
    units::detail::requirePositiveFinite(hydrogenMassFraction, "hydrogen mass fraction");
    return units::PhysicalSpecificEnergy(double(energyEvPerHydrogen) * hydrogenMassFraction *
        kElectronVoltEnergy / units::kHydrogenMass);
}
inline float hydrogenNumberDensity(units::ProperDensity density, double hydrogenMassFraction) {
    units::detail::requirePositiveFinite(hydrogenMassFraction, "hydrogen mass fraction");
    NumberDensity const numberDensity =
        density.quantity() * hydrogenMassFraction / units::kHydrogenMass;
    return checkedScalar(numberDensity.value(), "proper hydrogen number density in cm^-3");
}
inline float physicalSeconds(units::PhysicalTime interval) {
    return checkedScalar(interval.quantity().value(), "physical interval in seconds");
}
}    // namespace primordial
