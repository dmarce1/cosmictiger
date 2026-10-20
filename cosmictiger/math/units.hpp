/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <bit>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <boost/units/base_units/cgs/centimeter.hpp>
#include <boost/units/base_units/cgs/gram.hpp>
#include <boost/units/base_units/si/kelvin.hpp>
#include <boost/units/base_units/si/second.hpp>
#include <boost/units/derived_dimension.hpp>
#include <boost/units/make_system.hpp>
#include <boost/units/physical_dimensions/acceleration.hpp>
#include <boost/units/physical_dimensions/action.hpp>
#include <boost/units/physical_dimensions/area.hpp>
#include <boost/units/physical_dimensions/energy.hpp>
#include <boost/units/physical_dimensions/frequency.hpp>
#include <boost/units/physical_dimensions/length.hpp>
#include <boost/units/physical_dimensions/mass.hpp>
#include <boost/units/physical_dimensions/mass_density.hpp>
#include <boost/units/physical_dimensions/pressure.hpp>
#include <boost/units/physical_dimensions/specific_energy.hpp>
#include <boost/units/physical_dimensions/temperature.hpp>
#include <boost/units/physical_dimensions/time.hpp>
#include <boost/units/physical_dimensions/velocity.hpp>
#include <boost/units/quantity.hpp>
#include <boost/units/unit.hpp>

#include <cosmictiger/math/constants.hpp>

// Host-side dimensional and frame interfaces. Particle storage and device payloads keep
// their existing representations, including the full fixed-point position value.
namespace units {

using CgsSystem = boost::units::make_system<boost::units::cgs::centimeter_base_unit,
    boost::units::cgs::gram_base_unit, boost::units::si::second_base_unit,
    boost::units::si::kelvin_base_unit>::type;

template <typename Dimension>
using Quantity = boost::units::quantity<boost::units::unit<Dimension, CgsSystem>, double>;

using Length = Quantity<boost::units::length_dimension>;
using Mass = Quantity<boost::units::mass_dimension>;
using Time = Quantity<boost::units::time_dimension>;
using Velocity = Quantity<boost::units::velocity_dimension>;
using Density = Quantity<boost::units::mass_density_dimension>;
using Pressure = Quantity<boost::units::pressure_dimension>;
using SpecificEnergy = Quantity<boost::units::specific_energy_dimension>;
using Energy = Quantity<boost::units::energy_dimension>;
using Acceleration = Quantity<boost::units::acceleration_dimension>;
using Frequency = Quantity<boost::units::frequency_dimension>;
using Temperature = Quantity<boost::units::temperature_dimension>;
using Area = Quantity<boost::units::area_dimension>;
using Action = Quantity<boost::units::action_dimension>;
using GravitationalConstant =
    Quantity<boost::units::derived_dimension<boost::units::length_base_dimension, 3,
        boost::units::mass_base_dimension, -1, boost::units::time_base_dimension, -2>::type>;
using BoltzmannConstant =
    Quantity<boost::units::derived_dimension<boost::units::length_base_dimension, 2,
        boost::units::mass_base_dimension, 1, boost::units::time_base_dimension, -2,
        boost::units::temperature_base_dimension, -1>::type>;
using InverseMass =
    Quantity<boost::units::derived_dimension<boost::units::mass_base_dimension, -1>::type>;
using StefanBoltzmannConstant =
    Quantity<boost::units::derived_dimension<boost::units::mass_base_dimension, 1,
        boost::units::time_base_dimension, -3, boost::units::temperature_base_dimension, -4>::type>;

// Preserve the established numerical values; adopting units does not update constants.
inline constexpr auto kLightSpeed = Velocity::from_value(constants::kC);
inline constexpr auto kGravitationalConstant = GravitationalConstant::from_value(constants::kG);
inline constexpr auto kPlanckConstant = Action::from_value(constants::kH);
inline constexpr auto kReducedPlanckConstant = Action::from_value(constants::kHbar);
inline constexpr auto kElectronMass = Mass::from_value(constants::kMe);
inline constexpr auto kHydrogenMass = Mass::from_value(constants::kMh);
inline constexpr auto kSolarMass = Mass::from_value(constants::kM0);
inline constexpr auto kMegaparsec = Length::from_value(constants::kMpcToCm);
inline constexpr auto kYear = Time::from_value(constants::kSpyr);
inline constexpr auto kBoltzmann = BoltzmannConstant::from_value(constants::kB);
inline constexpr auto kStefanBoltzmannConstant =
    StefanBoltzmannConstant::from_value(constants::kSigma);
inline constexpr auto kThomsonCrossSection = Area::from_value(constants::kSigmaT);
inline constexpr auto kCmbTemperature = Temperature::from_value(constants::kTcmb);
inline constexpr auto kHubble100 = Frequency::from_value(constants::kH0);
// Historical particles-per-gram normalization used by the gas temperature conversion.
inline constexpr auto kParticlesPerGram = InverseMass::from_value(constants::kAvo);
inline constexpr auto kElectronVolt = Energy::from_value(constants::kEvtoerg);
inline constexpr auto kElectronVoltTemperature = Temperature::from_value(constants::kEvtoK);

namespace detail {

    inline void requirePositiveFinite(double value, char const* name) {
        // Host compilation uses -ffast-math, which can discard std::isfinite checks.
        static_assert(sizeof(double) == sizeof(uint64_t) && std::numeric_limits<double>::is_iec559,
            "unit scales require IEEE binary64 doubles");
        auto const bits = std::bit_cast<uint64_t>(value);
        constexpr uint64_t kSignMask = uint64_t(1) << 63;
        constexpr uint64_t kExponentMask = uint64_t(0x7ff) << 52;
        if ((bits & kSignMask) != 0 || (bits & kExponentMask) == kExponentMask || bits == 0) {
            throw std::invalid_argument(std::string(name) + " must be finite and positive");
        }
    }

    inline void requireNormalizedPosition(double value) {
        // Positive binary64 values have the same ordering as their unsigned bits.
        // This check also rejects NaN and infinity when -ffast-math is enabled.
        auto const bits = std::bit_cast<uint64_t>(value);
        constexpr uint64_t kSignMask = uint64_t(1) << 63;
        constexpr uint64_t kMagnitudeMask = ~kSignMask;
        constexpr uint64_t kOne = uint64_t(0x3ff) << 52;
        auto const magnitude = bits & kMagnitudeMask;
        if (magnitude >= kOne || ((bits & kSignMask) != 0 && magnitude != 0)) {
            throw std::invalid_argument("normalized position must be finite and in [0, 1)");
        }
    }

}    // namespace detail

class UnitScales {
    Length lengthScale_;
    Mass massScale_;
    Time timeScale_;

public:
    UnitScales(double lengthCm, double massG, double timeSeconds)
      : lengthScale_(Length::from_value(lengthCm))
      , massScale_(Mass::from_value(massG))
      , timeScale_(Time::from_value(timeSeconds)) {
        detail::requirePositiveFinite(lengthCm, "length scale");
        detail::requirePositiveFinite(massG, "mass scale");
        detail::requirePositiveFinite(timeSeconds, "time scale");
        detail::requirePositiveFinite(velocityScale().value(), "velocity scale");
        detail::requirePositiveFinite(densityScale().value(), "density scale");
        detail::requirePositiveFinite(specificEnergyScale().value(), "specific energy scale");
        detail::requirePositiveFinite(pressureScale().value(), "pressure scale");
    }

    Length lengthScale() const {
        return lengthScale_;
    }
    Mass massScale() const {
        return massScale_;
    }
    Time timeScale() const {
        return timeScale_;
    }
    Velocity velocityScale() const {
        return lengthScale_ / timeScale_;
    }
    Density densityScale() const {
        return massScale_ / lengthScale_ / lengthScale_ / lengthScale_;
    }
    SpecificEnergy specificEnergyScale() const {
        return velocityScale() * velocityScale();
    }
    Energy energyScale() const {
        return massScale_ * specificEnergyScale();
    }
    Pressure pressureScale() const {
        return densityScale() * specificEnergyScale();
    }
    Acceleration accelerationScale() const {
        return velocityScale() / timeScale_;
    }
    Frequency frequencyScale() const {
        return Frequency::from_value(1.0 / timeScale_.value());
    }
};

// h = H0 / (100 km/s/Mpc); it is independent of the time-dependent scale factor a.
class CosmologicalContext {
    double scaleFactor_;
    double hubbleParameter_;

public:
    CosmologicalContext(double scaleFactor, double hubbleParameter)
      : scaleFactor_(scaleFactor)
      , hubbleParameter_(hubbleParameter) {
        detail::requirePositiveFinite(scaleFactor, "scale factor");
        detail::requirePositiveFinite(hubbleParameter, "Hubble parameter h");
    }

    double scaleFactor() const {
        return scaleFactor_;
    }
    double hubbleParameter() const {
        return hubbleParameter_;
    }
};

// A distinct tag is required even when two quantities have identical dimensions.
// There are deliberately no implicit conversions or cross-frame arithmetic operators.
template <typename Tag, typename QuantityTypeT>
class FramedQuantity {
    QuantityTypeT quantity_;

public:
    using QuantityType = QuantityTypeT;

    explicit constexpr FramedQuantity(QuantityType quantity)
      : quantity_(quantity) {}

    constexpr QuantityType quantity() const {
        return quantity_;
    }
};

template <typename Tag>
class StoredScalar {
    double value_;

public:
    explicit constexpr StoredScalar(double value)
      : value_(value) {}

    constexpr double value() const {
        return value_;
    }
};

struct ComovingLengthTag {};
struct ProperLengthTag {};
struct PeculiarVelocityTag {};
struct ComovingDensityTag {};
struct ProperDensityTag {};
struct PhysicalSpecificEnergyTag {};
struct PhysicalPressureTag {};
struct PeculiarPotentialTag {};
struct PeculiarGravitationalAccelerationTag {};
struct PhysicalTimeTag {};
struct ConformalTimeTag {};
struct StoredLengthTag {};
struct StoredMassTag {};
struct StoredVelocityTag {};
struct StoredDensityTag {};
struct StoredSpecificEnergyTag {};
struct StoredPressureTag {};
struct StoredPotentialTag {};
struct StoredAccelerationTag {};

using ComovingLength = FramedQuantity<ComovingLengthTag, Length>;
using ProperLength = FramedQuantity<ProperLengthTag, Length>;
using PeculiarVelocity = FramedQuantity<PeculiarVelocityTag, Velocity>;
using ComovingDensity = FramedQuantity<ComovingDensityTag, Density>;
using ProperDensity = FramedQuantity<ProperDensityTag, Density>;
using PhysicalSpecificEnergy = FramedQuantity<PhysicalSpecificEnergyTag, SpecificEnergy>;
using PhysicalPressure = FramedQuantity<PhysicalPressureTag, Pressure>;
using PeculiarPotential = FramedQuantity<PeculiarPotentialTag, SpecificEnergy>;
using PeculiarGravitationalAcceleration =
    FramedQuantity<PeculiarGravitationalAccelerationTag, Acceleration>;
using PhysicalTime = FramedQuantity<PhysicalTimeTag, Time>;
using ConformalTime = FramedQuantity<ConformalTimeTag, Time>;
using StoredLength = StoredScalar<StoredLengthTag>;
using StoredMass = StoredScalar<StoredMassTag>;
using StoredVelocity = StoredScalar<StoredVelocityTag>;
using StoredDensity = StoredScalar<StoredDensityTag>;
using StoredSpecificEnergy = StoredScalar<StoredSpecificEnergyTag>;
using StoredPressure = StoredScalar<StoredPressureTag>;
using StoredPotential = StoredScalar<StoredPotentialTag>;
using StoredAcceleration = StoredScalar<StoredAccelerationTag>;

template <typename Storage>
class ComovingPosition {
    Storage storage_;

public:
    explicit ComovingPosition(Storage storage)
      : storage_(std::move(storage)) {}

    Storage const& storage() const {
        return storage_;
    }
};

inline ComovingLength toComovingLength(StoredLength length, UnitScales const& scales) {
    return ComovingLength(length.value() * scales.lengthScale());
}

template <typename Storage>
ComovingLength toComovingLength(
    ComovingPosition<Storage> const& position, UnitScales const& scales) {
    return ComovingLength(position.storage().toDouble() * scales.lengthScale());
}

inline ProperLength toProperLength(ComovingLength length, CosmologicalContext const& context) {
    return ProperLength(length.quantity() * context.scaleFactor());
}

inline ComovingLength toComovingLength(ProperLength length, CosmologicalContext const& context) {
    return ComovingLength(length.quantity() / context.scaleFactor());
}

inline ProperLength toProperLength(
    StoredLength length, UnitScales const& scales, CosmologicalContext const& context) {
    return toProperLength(toComovingLength(length, scales), context);
}

template <typename Storage>
ProperLength toProperLength(ComovingPosition<Storage> const& position, UnitScales const& scales,
    CosmologicalContext const& context) {
    return toProperLength(toComovingLength(position, scales), context);
}

inline StoredLength toStoredLength(ComovingLength length, UnitScales const& scales) {
    return StoredLength((length.quantity() / scales.lengthScale()).value());
}

inline StoredLength toStoredLength(
    ProperLength length, UnitScales const& scales, CosmologicalContext const& context) {
    return toStoredLength(toComovingLength(length, context), scales);
}

// Inverse coordinate conversion quantizes into Storage after checking the normalized box
// [0, 1). Out-of-box coordinates are rejected; use existing fixed arithmetic for periodic
// wrapping. Direct access through storage() avoids any conversion or quantization at all.
template <typename Storage>
ComovingPosition<Storage> toComovingPosition(ComovingLength length, UnitScales const& scales) {
    double const normalized = toStoredLength(length, scales).value();
    detail::requireNormalizedPosition(normalized);
    return ComovingPosition<Storage>(Storage(normalized));
}

template <typename Storage>
ComovingPosition<Storage> toComovingPosition(
    ProperLength length, UnitScales const& scales, CosmologicalContext const& context) {
    return toComovingPosition<Storage>(toComovingLength(length, context), scales);
}

inline Mass toMass(StoredMass mass, UnitScales const& scales) {
    return mass.value() * scales.massScale();
}

inline StoredMass toStoredMass(Mass mass, UnitScales const& scales) {
    return StoredMass((mass / scales.massScale()).value());
}

// Stored w = a v_peculiar. No position-dependent Hubble flow is added.
inline PeculiarVelocity toPeculiarVelocity(
    StoredVelocity velocity, UnitScales const& scales, CosmologicalContext const& context) {
    return PeculiarVelocity(velocity.value() * scales.velocityScale() / context.scaleFactor());
}

inline StoredVelocity toStoredVelocity(
    PeculiarVelocity velocity, UnitScales const& scales, CosmologicalContext const& context) {
    return StoredVelocity(
        (velocity.quantity() / scales.velocityScale()).value() * context.scaleFactor());
}

inline ComovingDensity toComovingDensity(StoredDensity density, UnitScales const& scales) {
    return ComovingDensity(density.value() * scales.densityScale());
}

inline ProperDensity toProperDensity(ComovingDensity density, CosmologicalContext const& context) {
    double const a = context.scaleFactor();
    return ProperDensity(density.quantity() / a / a / a);
}

inline ComovingDensity toComovingDensity(
    ProperDensity density, CosmologicalContext const& context) {
    double const a = context.scaleFactor();
    return ComovingDensity(density.quantity() * a * a * a);
}

inline ProperDensity toProperDensity(
    StoredDensity density, UnitScales const& scales, CosmologicalContext const& context) {
    return toProperDensity(toComovingDensity(density, scales), context);
}

inline StoredDensity toStoredDensity(ComovingDensity density, UnitScales const& scales) {
    return StoredDensity((density.quantity() / scales.densityScale()).value());
}

inline StoredDensity toStoredDensity(
    ProperDensity density, UnitScales const& scales, CosmologicalContext const& context) {
    return toStoredDensity(toComovingDensity(density, context), scales);
}

// Stored E = a^2 u, where u is physical specific internal energy.
inline PhysicalSpecificEnergy toPhysicalSpecificEnergy(
    StoredSpecificEnergy energy, UnitScales const& scales, CosmologicalContext const& context) {
    double const a = context.scaleFactor();
    return PhysicalSpecificEnergy(energy.value() * scales.specificEnergyScale() / a / a);
}

inline StoredSpecificEnergy toStoredSpecificEnergy(
    PhysicalSpecificEnergy energy, UnitScales const& scales, CosmologicalContext const& context) {
    double const a = context.scaleFactor();
    return StoredSpecificEnergy((energy.quantity() / scales.specificEnergyScale()).value() * a * a);
}

// Stored P = (gamma - 1) rho_c E = a^5 P_proper.
inline PhysicalPressure toPhysicalPressure(
    StoredPressure pressure, UnitScales const& scales, CosmologicalContext const& context) {
    double const a = context.scaleFactor();
    return PhysicalPressure(pressure.value() * scales.pressureScale() / a / a / a / a / a);
}

inline StoredPressure toStoredPressure(
    PhysicalPressure pressure, UnitScales const& scales, CosmologicalContext const& context) {
    double const a = context.scaleFactor();
    return StoredPressure(
        (pressure.quantity() / scales.pressureScale()).value() * a * a * a * a * a);
}

// The gravity solver stores a phi_proper and a^2 g_proper in code units.
inline PeculiarPotential toPeculiarPotential(
    StoredPotential potential, UnitScales const& scales, CosmologicalContext const& context) {
    return PeculiarPotential(
        potential.value() * scales.specificEnergyScale() / context.scaleFactor());
}

inline StoredPotential toStoredPotential(
    PeculiarPotential potential, UnitScales const& scales, CosmologicalContext const& context) {
    return StoredPotential(
        (potential.quantity() / scales.specificEnergyScale()).value() * context.scaleFactor());
}

// Gravitational force per mass only; dv_pec/dt also contains cosmological drag.
inline PeculiarGravitationalAcceleration toPeculiarGravitationalAcceleration(
    StoredAcceleration acceleration, UnitScales const& scales, CosmologicalContext const& context) {
    double const a = context.scaleFactor();
    return PeculiarGravitationalAcceleration(
        acceleration.value() * scales.accelerationScale() / a / a);
}

inline StoredAcceleration toStoredAcceleration(PeculiarGravitationalAcceleration acceleration,
    UnitScales const& scales, CosmologicalContext const& context) {
    double const a = context.scaleFactor();
    return StoredAcceleration(
        (acceleration.quantity() / scales.accelerationScale()).value() * a * a);
}

// These convert infinitesimal/time-step intervals at a fixed a, not elapsed epoch times.
inline PhysicalTime toPhysicalInterval(ConformalTime interval, CosmologicalContext const& context) {
    return PhysicalTime(interval.quantity() * context.scaleFactor());
}

inline ConformalTime toConformalInterval(
    PhysicalTime interval, CosmologicalContext const& context) {
    return ConformalTime(interval.quantity() / context.scaleFactor());
}

// Astronomical display basis: L=Mpc/h, M=Msun/h, V=km/s, temperature=K.
// The time unit is consequently (Mpc/h)/(km/s). Frame conversion is separate.
inline UnitScales cosmologicalScales(CosmologicalContext const& context) {
    double const lengthCm = constants::kMpcToCm / context.hubbleParameter();
    double const massG = constants::kM0 / context.hubbleParameter();
    return UnitScales(lengthCm, massG, lengthCm / 1.0e5);
}

template <typename QuantityType>
QuantityType cosmologicalScale(CosmologicalContext const& context) {
    auto const scales = cosmologicalScales(context);
    if constexpr (std::is_same_v<QuantityType, Length>) {
        return scales.lengthScale();
    } else if constexpr (std::is_same_v<QuantityType, Mass>) {
        return scales.massScale();
    } else if constexpr (std::is_same_v<QuantityType, Time>) {
        return scales.timeScale();
    } else if constexpr (std::is_same_v<QuantityType, Velocity>) {
        return scales.velocityScale();
    } else if constexpr (std::is_same_v<QuantityType, Density>) {
        return scales.densityScale();
    } else if constexpr (std::is_same_v<QuantityType, SpecificEnergy>) {
        return scales.specificEnergyScale();
    } else if constexpr (std::is_same_v<QuantityType, Energy>) {
        return scales.energyScale();
    } else if constexpr (std::is_same_v<QuantityType, Pressure>) {
        return scales.pressureScale();
    } else if constexpr (std::is_same_v<QuantityType, Acceleration>) {
        return scales.accelerationScale();
    } else if constexpr (std::is_same_v<QuantityType, Frequency>) {
        return scales.frequencyScale();
    } else if constexpr (std::is_same_v<QuantityType, Temperature>) {
        return Temperature::from_value(1.0);
    } else {
        static_assert(
            !std::is_same_v<QuantityType, QuantityType>, "unsupported cosmological quantity");
    }
}

template <typename Dimension>
double toCosmologicalValue(Quantity<Dimension> quantity, CosmologicalContext const& context) {
    return (quantity / cosmologicalScale<Quantity<Dimension>>(context)).value();
}

template <typename Tag, typename QuantityType>
double toCosmologicalValue(
    FramedQuantity<Tag, QuantityType> quantity, CosmologicalContext const& context) {
    return toCosmologicalValue(quantity.quantity(), context);
}

template <typename FramedType>
FramedType fromCosmologicalValue(double value, CosmologicalContext const& context) {
    return FramedType(value * cosmologicalScale<typename FramedType::QuantityType>(context));
}

template <typename QuantityType>
QuantityType quantityFromCosmologicalValue(double value, CosmologicalContext const& context) {
    return value * cosmologicalScale<QuantityType>(context);
}

}    // namespace units
