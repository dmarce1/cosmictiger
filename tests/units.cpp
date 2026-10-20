/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <cosmictiger/math/units.hpp>

namespace detail {

template <typename Left, typename Right>
concept Addable = requires(Left left, Right right) { left + right; };

static_assert(Addable<units::Length, units::Length>);
static_assert(!Addable<units::Length, units::Time>);
static_assert(!Addable<units::ComovingLength, units::ProperLength>);
static_assert(!std::is_convertible_v<units::ComovingLength, units::ProperLength>);
static_assert(!std::is_constructible_v<units::ProperLength, units::ComovingLength>);
static_assert(!std::is_convertible_v<units::StoredVelocity, units::PeculiarVelocity>);
static_assert(!std::is_constructible_v<units::StoredSpecificEnergy, units::StoredPotential>);
static_assert(!std::is_constructible_v<units::PhysicalTime, units::ConformalTime>);
static_assert(!std::is_constructible_v<units::PhysicalSpecificEnergy, units::PeculiarPotential>);
static_assert(std::is_same_v<decltype(units::Length{} / units::Time{}), units::Velocity>);
static_assert(
    std::is_same_v<decltype(units::Density{} * units::SpecificEnergy{}), units::Pressure>);
static_assert(std::is_same_v<decltype(units::SpecificEnergy{} /
                                 (units::kBoltzmann * units::kParticlesPerGram)),
    units::Temperature>);

void require(bool condition, char const* label) {
    if (!condition) {
        throw std::runtime_error(label);
    }
}

void requireClose(double actual, double expected, char const* label) {
    auto const bits = std::bit_cast<uint64_t>(actual);
    constexpr uint64_t kExponentMask = uint64_t(0x7ff) << 52;
    double const tolerance = 2.0e-14 * std::max(std::abs(expected), 1.0e-100);
    if ((bits & kExponentMask) == kExponentMask || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(std::string(label) + ": got " + std::to_string(actual) +
            ", expected " + std::to_string(expected));
    }
}

template <typename Function>
void requireInvalid(Function function, char const* label) {
    bool rejected = false;
    try {
        function();
    } catch (std::invalid_argument const&) {
        rejected = true;
    }
    require(rejected, label);
}

// A storage probe deliberately forbids floating conversion except toDouble().
// Production Fixed32 instantiation is exercised by the particle output integration.
class FixedStorageProbe {
    uint32_t raw_;

public:
    explicit FixedStorageProbe(double value)
      : raw_(uint32_t(value * 4294967296.0)) {}

    double toDouble() const {
        return double(raw_) / 4294967296.0;
    }
    uint32_t raw() const {
        return raw_;
    }
    float toFloat() const = delete;
    explicit operator float() const = delete;
};

void testFixedStorage() {
    units::UnitScales const scales(8.0, 32.0, 2.0);
    units::CosmologicalContext const context(0.25, 0.7);
    double const coordinate = 0.5 + 1.0 / 4294967296.0;
    FixedStorageProbe const fixed(coordinate);
    units::ComovingPosition<FixedStorageProbe> const position(fixed);
    static_assert(std::is_same_v<decltype(position.storage()), FixedStorageProbe const&>);
    require(position.storage().raw() == uint32_t(0x80000001), "retained fixed low bit");
    require(position.storage().toDouble() == coordinate, "no float conversion in wrapper");
    auto const comoving = units::toComovingLength(position, scales);
    auto const proper = units::toProperLength(position, scales, context);
    require(comoving.quantity().value() == 4.0 + 1.0 / 536870912.0,
        "comoving conversion preserves low fixed bit");
    require(proper.quantity().value() == 1.0 + 1.0 / 2147483648.0,
        "proper conversion preserves low fixed bit");
    require(units::toComovingPosition<FixedStorageProbe>(comoving, scales).storage().raw() ==
            fixed.raw(),
        "comoving fixed-point round trip");
    require(units::toComovingPosition<FixedStorageProbe>(proper, scales, context).storage().raw() ==
            fixed.raw(),
        "proper fixed-point round trip");
}

void testInversePositionBounds() {
    units::UnitScales const scales(8.0, 32.0, 2.0);
    units::CosmologicalContext const context(0.25, 0.7);
    units::ComovingLength const zero(units::Length::from_value(0.0));
    require(units::toComovingPosition<FixedStorageProbe>(zero, scales).storage().raw() == 0,
        "zero normalized position is valid");
    units::ComovingLength const last(units::Length::from_value(8.0 * (1.0 - 1.0 / 4294967296.0)));
    require(units::toComovingPosition<FixedStorageProbe>(last, scales).storage().raw() ==
            uint32_t(0xffffffff),
        "last fixed-point position is valid");
    for (double const invalid : {1.0, 1.25, -0.25, std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        units::ComovingLength const comoving(units::Length::from_value(invalid * 8.0));
        units::ProperLength const proper(units::Length::from_value(invalid * 2.0));
        requireInvalid([&] { units::toComovingPosition<FixedStorageProbe>(comoving, scales); },
            "reject invalid normalized comoving position before fixed constructor");
        requireInvalid(
            [&] { units::toComovingPosition<FixedStorageProbe>(proper, scales, context); },
            "reject invalid normalized proper position before fixed constructor");
    }
}

void testAnalyticConversions() {
    units::UnitScales const scales(8.0, 32.0, 2.0);
    units::CosmologicalContext const context(0.25, 0.7);
    requireClose(scales.velocityScale().value(), 4.0, "velocity scale");
    requireClose(scales.densityScale().value(), 0.0625, "density scale");
    requireClose(scales.energyScale().value(), 512.0, "energy scale");
    requireClose(scales.pressureScale().value(), 1.0, "pressure scale");
    requireClose(scales.accelerationScale().value(), 2.0, "acceleration scale");
    requireClose(scales.frequencyScale().value(), 0.5, "frequency scale");

    auto const length = units::toProperLength(units::StoredLength(2.0), scales, context);
    requireClose(length.quantity().value(), 4.0, "proper smoothing length");
    requireClose(units::toStoredLength(length, scales, context).value(), 2.0, "length round trip");
    auto const mass = units::toMass(units::StoredMass(5.0), scales);
    requireClose(mass.value(), 160.0, "CGS particle mass");
    requireClose(units::toStoredMass(mass, scales).value(), 5.0, "mass round trip");
    auto const velocity = units::toPeculiarVelocity(units::StoredVelocity(-3.0), scales, context);
    requireClose(velocity.quantity().value(), -48.0, "CGS peculiar velocity");
    requireClose(
        units::toStoredVelocity(velocity, scales, context).value(), -3.0, "velocity round trip");
    require(
        units::toPeculiarVelocity(units::StoredVelocity(0.0), scales, context).quantity().value() ==
            0.0,
        "zero peculiar velocity has no Hubble contribution");
    auto const density = units::toProperDensity(units::StoredDensity(3.0), scales, context);
    requireClose(density.quantity().value(), 12.0, "proper density scales as a^-3");
    requireClose(
        units::toStoredDensity(density, scales, context).value(), 3.0, "density round trip");
    auto const comovingDensity = units::toComovingDensity(units::StoredDensity(3.0), scales);
    requireClose(comovingDensity.quantity().value(), 0.1875, "comoving density");
    requireClose(units::toStoredDensity(comovingDensity, scales).value(), 3.0,
        "comoving density round trip");
    auto const energy =
        units::toPhysicalSpecificEnergy(units::StoredSpecificEnergy(5.0), scales, context);
    requireClose(energy.quantity().value(), 1280.0, "specific energy scales as a^-2");
    requireClose(units::toStoredSpecificEnergy(energy, scales, context).value(), 5.0,
        "specific energy round trip");
    auto const pressure = units::toPhysicalPressure(units::StoredPressure(7.0), scales, context);
    requireClose(pressure.quantity().value(), 7168.0, "pressure scales as a^-5");
    requireClose(
        units::toStoredPressure(pressure, scales, context).value(), 7.0, "pressure round trip");
    auto const potential =
        units::toPeculiarPotential(units::StoredPotential(-3.0), scales, context);
    requireClose(potential.quantity().value(), -192.0, "potential scales as a^-1");
    requireClose(
        units::toStoredPotential(potential, scales, context).value(), -3.0, "potential round trip");
    auto const acceleration = units::toPeculiarGravitationalAcceleration(
        units::StoredAcceleration(-2.0), scales, context);
    requireClose(acceleration.quantity().value(), -64.0, "gravity scales as a^-2");
    requireClose(units::toStoredAcceleration(acceleration, scales, context).value(), -2.0,
        "gravity round trip");
    units::ConformalTime const conformal(units::Time::from_value(8.0));
    auto const physical = units::toPhysicalInterval(conformal, context);
    requireClose(physical.quantity().value(), 2.0, "physical time interval");
    requireClose(units::toConformalInterval(physical, context).quantity().value(), 8.0,
        "time interval round trip");
}

void testCosmologicalConversions() {
    units::CosmologicalContext const context(0.25, 0.7);
    auto const scales = units::cosmologicalScales(context);
    auto const comovingLength = units::toComovingLength(units::StoredLength(2.0), scales);
    auto const properLength = units::toProperLength(units::StoredLength(2.0), scales, context);
    requireClose(units::toCosmologicalValue(comovingLength, context), 2.0, "comoving Mpc/h");
    requireClose(units::toCosmologicalValue(properLength, context), 0.5, "proper Mpc/h");
    requireClose(
        units::toCosmologicalValue(units::Length::from_value(constants::kMpcToCm), context), 0.7,
        "one Mpc equals h Mpc/h");
    requireClose(units::toCosmologicalValue(units::Mass::from_value(constants::kM0), context), 0.7,
        "one solar mass equals h Msun/h");
    auto const mass = units::toMass(units::StoredMass(3.0), scales);
    requireClose(units::toCosmologicalValue(mass, context), 3.0, "particle Msun/h");
    auto const velocity = units::toPeculiarVelocity(units::StoredVelocity(6.0), scales, context);
    requireClose(units::toCosmologicalValue(velocity, context), 24.0, "peculiar km/s");
    auto const density = units::toProperDensity(units::StoredDensity(7.0), scales, context);
    requireClose(
        units::toCosmologicalValue(density, context), 448.0, "density in (Msun/h)/(Mpc/h)^3");
    auto const energy =
        units::toPhysicalSpecificEnergy(units::StoredSpecificEnergy(2.0), scales, context);
    requireClose(units::toCosmologicalValue(energy, context), 32.0, "specific energy in (km/s)^2");
    auto const pressure = units::toPhysicalPressure(units::StoredPressure(5.0), scales, context);
    requireClose(units::toCosmologicalValue(pressure, context), 5120.0,
        "pressure in (Msun/h)/(Mpc/h)^3*(km/s)^2");
    auto const restoredLength = units::fromCosmologicalValue<units::ProperLength>(0.5, context);
    requireClose(restoredLength.quantity().value(), properLength.quantity().value(),
        "proper cosmological inverse");
    auto const restoredVelocity =
        units::fromCosmologicalValue<units::PeculiarVelocity>(24.0, context);
    requireClose(units::toStoredVelocity(restoredVelocity, scales, context).value(), 6.0,
        "cosmological velocity round trip");
    auto const restoredMass = units::quantityFromCosmologicalValue<units::Mass>(3.0, context);
    requireClose(restoredMass.value(), mass.value(), "cosmological mass inverse");

    // A second context ensures h changes units independently of the frame's a.
    units::CosmologicalContext const otherContext(0.5, 0.4);
    requireClose(
        units::toCosmologicalValue(units::Length::from_value(constants::kMpcToCm), otherContext),
        0.4, "independent h normalization");
    requireClose(units::toCosmologicalValue(velocity, otherContext), 24.0,
        "km/s independent of a and h after physical conversion");
    requireClose(units::toCosmologicalValue(units::Temperature::from_value(10.0), context), 10.0,
        "kelvin unchanged");
}

void testNeutralGasEnergy() {
    units::UnitScales const scales(8.0, 32.0, 2.0);
    units::CosmologicalContext const context(0.25, 0.7);
    double const neutralParticleFraction = 0.76 + 0.24 / 4.0;
    auto const temperature = units::Temperature::from_value(37.0);
    units::SpecificEnergy const physicalEnergy =
        1.5 * units::kBoltzmann * units::kParticlesPerGram * neutralParticleFraction * temperature;
    auto const stored = units::toStoredSpecificEnergy(
        units::PhysicalSpecificEnergy(physicalEnergy), scales, context);
    // Independent scalar equation: E_stored = a^2 (3/2) kB N_A (X + Y/4) T / (L/T)^2.
    double const expectedStored =
        (0.25 * 0.25) * 1.5 * 1.380658e-16 * 6.0221409e23 * 0.82 * 37.0 / 16.0;
    requireClose(stored.value(), expectedStored, "neutral gas stored energy normalization");
    auto const restored = units::toPhysicalSpecificEnergy(stored, scales, context);
    units::Temperature const restoredTemperature = restored.quantity() /
        (1.5 * units::kBoltzmann * units::kParticlesPerGram * neutralParticleFraction);
    requireClose(restoredTemperature.value(), 37.0, "neutral gas temperature round trip");
}

void testValidationAndConstants() {
    for (double const invalid : {0.0, -0.0, -1.0, std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        requireInvalid(
            [=] { units::UnitScales(invalid, 1.0, 1.0); }, "reject invalid length scale");
        requireInvalid([=] { units::UnitScales(1.0, invalid, 1.0); }, "reject invalid mass scale");
        requireInvalid([=] { units::UnitScales(1.0, 1.0, invalid); }, "reject invalid time scale");
        requireInvalid([=] { units::CosmologicalContext(invalid, 0.7); }, "reject invalid a");
        requireInvalid([=] { units::CosmologicalContext(0.25, invalid); }, "reject invalid h");
    }
    requireInvalid([] { units::UnitScales(1.0e200, 1.0, 1.0e-200); }, "reject overflowing scales");
    require(units::kLightSpeed.value() == constants::kC, "unchanged light speed");
    require(
        units::kGravitationalConstant.value() == constants::kG, "unchanged gravitational constant");
    require(units::kBoltzmann.value() == constants::kB, "unchanged Boltzmann constant");
    require(
        units::kParticlesPerGram.value() == constants::kAvo, "unchanged particle normalization");
    require(units::kSolarMass.value() == constants::kM0, "unchanged solar mass");
    require(units::kMegaparsec.value() == constants::kMpcToCm, "unchanged Mpc");
    require(units::kHubble100.value() == constants::kH0, "unchanged H0 normalization");
    require(units::kYear.value() == constants::kSpyr, "unchanged year");
}

}    // namespace detail

int main() {
    try {
        detail::testFixedStorage();
        detail::testInversePositionBounds();
        detail::testAnalyticConversions();
        detail::testCosmologicalConversions();
        detail::testNeutralGasEnergy();
        detail::testValidationAndConstants();
    } catch (std::exception const& error) {
        std::fprintf(stderr, "unit conversion test failed: %s\n", error.what());
        return 1;
    }
    std::puts("unit conversion tests passed");
    return 0;
}
