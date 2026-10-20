/* CosmicTiger - typed primordial boundary regression; BSL-1.0. */
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <cosmictiger/chemistry/primordial_units.hpp>

int main() {
    auto require = [](bool condition) {
        if (!condition) {
            throw std::runtime_error("primordial unit boundary regression");
        }
    };
    double const hydrogen = 0.76;
    // An absolute anchor catches a mutually consistent but obsolete conversion.
    auto const oneElectronVolt = primordial::physicalEnergy(1.0f, hydrogen);
    require(std::abs(oneElectronVolt.quantity().value() * units::kHydrogenMass.value() / hydrogen -
                1.602176634e-12) < 1.0e-27);
    require(primordial::energyPerHydrogen(
                units::PhysicalSpecificEnergy(units::SpecificEnergy::from_value(
                    1.602176634e-12 * hydrogen / units::kHydrogenMass.value())),
                hydrogen) == 1.0f);
    units::UnitScales const scales(3.0e24, 2.0e42, 1.0e16);
    for (double const a : {0.001, 0.02, 0.5, 1.0}) {
        units::CosmologicalContext const context(a, 0.67);
        for (float const e : {1.0e-3f, 0.1f, 100.0f, 1.0e4f}) {
            auto const physical = primordial::physicalEnergy(e, hydrogen);
            auto const stored = units::toStoredSpecificEnergy(physical, scales, context);
            auto const restored = units::toPhysicalSpecificEnergy(stored, scales, context);
            require(
                std::abs(primordial::energyPerHydrogen(restored, hydrogen) / e - 1.0f) < 3.0e-7f);
        }
        auto const density = units::ProperDensity(units::Density::from_value(1.0e-29));
        require(std::abs(primordial::hydrogenNumberDensity(density, hydrogen) /
                        float(1.0e-29 * hydrogen / units::kHydrogenMass.value()) -
                    1.0f) < 3.0e-7f);
        auto const physicalTime = units::toPhysicalInterval(
            units::ConformalTime(units::Time::from_value(1.0e12)), context);
        require(primordial::physicalSeconds(physicalTime) == float(a * 1.0e12));
    }
    for (double const value : {std::numeric_limits<double>::infinity(), 1.0e100, 1.0e-100}) {
        bool rejected = false;
        try {
            (void) primordial::checkedScalar(value, "test");
        } catch (std::range_error const&) {
            rejected = true;
        }
        require(rejected);
    }
    std::cout << "Primordial typed boundary round trips and range rejection passed\n";
}
