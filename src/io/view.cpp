/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <variant>
#include <vector>

#if COSMICTIGER_WITH_GRAVITY
#include <chealpix.h>
#endif
#include <hpx/serialization/vector.hpp>
#include <hpx/threading_base/annotated_function.hpp>
#include <silo.h>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/options_metadata.hpp>
#include <cosmictiger/core/profiler.hpp>
#include <cosmictiger/core/terminal.hpp>
#include <cosmictiger/math/vector3.hpp>
#if COSMICTIGER_WITH_GRAVITY
#include <cosmictiger/cosmology/cosmology.hpp>
#endif
#include <cosmictiger/io/output.hpp>
#include <cosmictiger/io/view.hpp>
#include <cosmictiger/math/constants.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/units.hpp>
#include <cosmictiger/particles/particles.hpp>
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/sph/sph.hpp>
#if COSMICTIGER_ENABLE_CHEMISTRY
#include <cosmictiger/chemistry/sph.hpp>
#endif
#include <cosmictiger/sph/particles.hpp>
#if COSMICTIGER_ENABLE_STAR_FORMATION
#include <cosmictiger/stars/star_formation.hpp>
#endif
#endif

#if COSMICTIGER_WITH_GRAVITY
HPX_PLAIN_ACTION(outputView, OutputViewAction);
#endif

static Vector<Range<double>> viewBoxes;

struct DarkMatterParticleInfo {
    Position x;
    Position y;
    Position z;
    float vx;
    float vy;
    float vz;
    int rung;
    uint64_t id{};
    double formationTime = -1.0;
};

#if COSMICTIGER_WITH_SPH
struct StarParticleInfo : public DarkMatterParticleInfo {};

struct SphParticleInfo : public DarkMatterParticleInfo {
    double entropy;
    float rho;
    float smoothingLength;
    float alpha;
    double divergence;
#if COSMICTIGER_ENABLE_CHEMISTRY
    std::array<float, 9> chemistryFields{};
#endif
#if COSMICTIGER_ENABLE_STAR_FORMATION
    std::array<double, 3> starFormationFields{};
#endif
};

#endif
struct ViewSnapshot {
#if COSMICTIGER_WITH_SPH
    Vector<SphParticleInfo> gas;
    Vector<StarParticleInfo> stars;
#endif
    Vector<DarkMatterParticleInfo> darkMatter;
};

// A box selects immutable snapshot records by address. Frames share those same
// records and cached physical diagnostics instead of duplicating the population.
template <typename Particle>
class ParticleSelection {
    Vector<Particle const*> particles_;

public:
    class Iterator {
        typename Vector<Particle const*>::const_iterator iterator_;

    public:
        explicit Iterator(typename Vector<Particle const*>::const_iterator iterator)
          : iterator_(iterator) {}
        Particle const& operator*() const {
            return **iterator_;
        }
        Iterator& operator++() {
            ++iterator_;
            return *this;
        }
        bool operator!=(Iterator const& other) const {
            return iterator_ != other.iterator_;
        }
    };
    size_t size() const {
        return particles_.size();
    }
    bool empty() const {
        return particles_.empty();
    }
    void push_back(Particle const& particle) {
        particles_.push_back(&particle);
    }
    Iterator begin() const {
        return Iterator(particles_.begin());
    }
    Iterator end() const {
        return Iterator(particles_.end());
    }
};

struct ViewParticles {
#if COSMICTIGER_WITH_SPH
    ParticleSelection<SphParticleInfo> gas;
    ParticleSelection<StarParticleInfo> stars;
#endif
    ParticleSelection<DarkMatterParticleInfo> darkMatter;
};

namespace detail {

void checkViewWrite(int result, char const* name) {
    if (result < 0) {
        THROW_ERROR("Cannot write converted particle Silo object %s\n", name);
    }
}

void writeViewText(DBfile* db, char const* name, char const* value) {
    int const size = std::strlen(value) + 1;
    checkViewWrite(DBWrite(db, name, value, &size, 1, DB_CHAR), name);
}

void writeViewScalar(DBfile* db, char const* name, double value) {
    int const size = 1;
    checkViewWrite(DBWrite(db, name, &value, &size, 1, DB_DOUBLE), name);
}

void writeViewInteger(DBfile* db, char const* name, int value) {
    int const size = 1;
    checkViewWrite(DBWrite(db, name, &value, &size, 1, DB_INT), name);
}

void writeViewCommandLine(DBfile* db) {
    checkViewWrite(DBMkDir(db, "cli"), "cli");
    checkViewWrite(DBSetDir(db, "cli"), "cli");
    writeViewText(db, "value_convention",
        "Effective application settings at this snapshot in CLI units, including defaults, "
        "configuration files, and runtime overrides. Original process arguments are in argv.");
    auto const& arguments = getCommandLineArguments();
    writeViewInteger(db, "argc", arguments.size());
    checkViewWrite(DBMkDir(db, "argv"), "argv");
    for (size_t i = 0; i < arguments.size(); i++) {
        auto const name = "argv/arg_" + std::to_string(i);
        writeViewText(db, name.c_str(), arguments[i].c_str());
    }
    std::set<std::string> groups;
    for (auto const& entry : getOptionsMetadata(getOptions())) {
        std::string name = entry.name;
        size_t separator = name.find('.');
        while (separator != std::string::npos) {
            // PDB treats dots as structure-member syntax. Represent every
            // option component as a directory, including gravity.method.far.
            name[separator] = '/';
            auto const group = name.substr(0, separator);
            if (groups.insert(group).second) {
                checkViewWrite(DBMkDir(db, group.c_str()), group.c_str());
            }
            separator = name.find('.', separator + 1);
        }
        std::visit(
            [db, &name](auto const& value) {
                using Value = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<Value, std::string>) {
                    writeViewText(db, name.c_str(), value.c_str());
                } else if constexpr (std::is_same_v<Value, size_t>) {
                    long long const count = static_cast<long long>(value);
                    int const dimension = 1;
                    checkViewWrite(DBWrite(db, name.c_str(), &count, &dimension, 1, DB_LONG_LONG),
                        name.c_str());
                } else if constexpr (std::is_same_v<Value, double>) {
                    writeViewScalar(db, name.c_str(), value);
                } else {
                    writeViewInteger(db, name.c_str(), int(value));
                }
            },
            entry.value);
    }
    checkViewWrite(DBSetDir(db, ".."), "..");
}

class ViewUnitConversion {
    units::UnitScales scales_;
    units::CosmologicalContext context_;
    bool cosmological_;
    bool lab_;

    template <typename Quantity>
    double displayValue(Quantity const& quantity) const {
        return cosmological_ ? units::toCosmologicalValue(quantity, context_) :
                               quantity.quantity().value();
    }

public:
    ViewUnitConversion(double a, bool cosmological, bool lab)
      : scales_(
            getOptions().units.lengthCm, getOptions().units.massG, getOptions().units.timeSeconds)
      , context_(a, getOptions().cosmology.hubble)
      , cosmological_(cosmological)
      , lab_(lab) {}

    double position(Position const& value) const {
        // Keep every fixed-point bit until the explicit physical-output boundary.
        units::ComovingPosition<Position> const position(value);
        return lab_ ? displayValue(units::toProperLength(position, scales_, context_)) :
                      displayValue(units::toComovingLength(position, scales_));
    }

    double length(double value) const {
        units::StoredLength const length(value);
        return lab_ ? displayValue(units::toProperLength(length, scales_, context_)) :
                      displayValue(units::toComovingLength(length, scales_));
    }

    double velocity(double value) const {
        return displayValue(
            units::toPeculiarVelocity(units::StoredVelocity(value), scales_, context_));
    }

    double density(double value) const {
        if constexpr (kSpatialDimensions < 3) {
            return value * getOptions().units.massG /
                std::pow(getOptions().units.lengthCm, kSpatialDimensions);
        }
        units::StoredDensity const density(value);
        return lab_ ? displayValue(units::toProperDensity(density, scales_, context_)) :
                      displayValue(units::toComovingDensity(density, scales_));
    }

    double entropy(double value) const {
        double energyScale = scales_.specificEnergyScale().value();
        double densityScale = scales_.densityScale().value() *
            std::pow(getOptions().units.lengthCm, 3 - kSpatialDimensions);
        if (cosmological_) {
            energyScale *= 1e-10;
            densityScale *= std::pow(constants::kMpcToCm, 3) /
                (constants::kM0 * context_.hubbleParameter() * context_.hubbleParameter());
        }
        // A = P_physical / rho_physical^gamma: no scale-factor or frame conversion.
        return value * energyScale * std::pow(densityScale, 1.0 - getOptions().hydro.gamma);
    }

    double frequency(double value) const {
        auto const frequency = value * scales_.frequencyScale();
        return cosmological_ ? units::toCosmologicalValue(frequency, context_) : frequency.value();
    }

    double mass(double value) const {
        auto const mass = units::toMass(units::StoredMass(value), scales_);
        return cosmological_ ? units::toCosmologicalValue(mass, context_) : mass.value();
    }
};

struct ViewDirectoryMetadata {
    std::string path;
    bool gas = false;
    bool darkMatter = false;
    bool stars = false;
    std::vector<std::string> gasVariables;
    std::vector<std::string> darkMatterVariables;
    std::vector<std::string> starVariables;
    template <typename Archive>
    void serialize(Archive& archive, unsigned) {
        archive & path & gas & darkMatter & stars & gasVariables & darkMatterVariables &
            starVariables;
    }
};

std::string viewDirectory(DBfile* db) {
    char path[4096];
    checkViewWrite(DBGetDir(db, path), "current directory");
    return path;
}

void enterViewSpecies(DBfile* db, char const* species) {
    checkViewWrite(DBMkDir(db, species), species);
    checkViewWrite(DBSetDir(db, species), species);
}

void writeGasMassExpression(DBfile* db, double mass, char const* unit) {
    // VisIt takes expression names and references literally, including their directories.
    std::string const path = viewDirectory(db).substr(1);
    std::string const name = path + "/mass";
    char value[64];
    auto const converted = std::to_chars(value, value + sizeof(value), mass,
        std::chars_format::general, std::numeric_limits<double>::max_digits10);
    if (converted.ec != std::errc{}) {
        throw std::runtime_error("Cannot format SPH mass expression");
    }
    std::string const expression =
        "nodal_constant(<" + path + "/mesh>, " + std::string(value, converted.ptr) + ")";
    char const* names[] = {name.c_str()};
    char const* expressions[] = {expression.c_str()};
    int const types[] = {DB_VARTYPE_SCALAR};
    checkViewWrite(DBPutDefvars(db, "mass_expression", 1, names, types, expressions, nullptr),
        "mass_expression");
    writeViewText(db, "mass_unit", unit);
}

template <typename Particle, typename Position>
void writeParticlePointmesh(DBfile* db, ViewDirectoryMetadata& metadata, char const* mesh,
    ParticleSelection<Particle> const& particles, Position position, DBoptlist* options) {
    if (particles.size() > size_t(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Too many particles for a Silo point mesh");
    }
    std::array<Vector<double>, kDimensions> coordinates;
    for (auto& axis : coordinates) {
        axis.reserve(particles.size());
    }
    for (auto const& particle : particles) {
        coordinates[0].push_back(position(particle.x));
        coordinates[1].push_back(position(particle.y));
        coordinates[2].push_back(position(particle.z));
    }
    double* coordinatePointers[] = {
        coordinates[0].data(), coordinates[1].data(), coordinates[2].data()};
    checkViewWrite(DBPutPointmesh(db, mesh, kDimensions, coordinatePointers, int(particles.size()),
                       DB_DOUBLE, options),
        mesh);
#if COSMICTIGER_WITH_SPH
    if constexpr (std::is_same_v<Particle, SphParticleInfo>) {
        metadata.gas = true;
    } else if constexpr (std::is_same_v<Particle, StarParticleInfo>) {
        metadata.stars = true;
    } else
#endif
    {
        metadata.darkMatter = true;
    }
}

template <typename Particle, typename Value>
void writeViewField(DBfile* db, ViewDirectoryMetadata& metadata, char const* name, char const* mesh,
    ParticleSelection<Particle> const& particles, char const* unitName, Value value) {
    Vector<double> values;
    values.reserve(particles.size());
    for (auto const& particle : particles) {
        values.push_back(value(particle));
    }
    auto* options = DBMakeOptlist(1);
    DBAddOption(options, DBOPT_UNITS, const_cast<char*>(unitName));
    checkViewWrite(
        DBPutPointvar1(db, name, mesh, values.data(), int(values.size()), DB_DOUBLE, options),
        name);
#if COSMICTIGER_WITH_SPH
    if constexpr (std::is_same_v<Particle, SphParticleInfo>) {
        metadata.gasVariables.emplace_back(name);
    } else if constexpr (std::is_same_v<Particle, StarParticleInfo>) {
        metadata.starVariables.emplace_back(name);
    } else
#endif
    {
        metadata.darkMatterVariables.emplace_back(name);
    }
    DBFreeOptlist(options);
}

template <typename Particle>
void writeStarFormationFields(DBfile* db, ViewDirectoryMetadata& metadata, char const* mesh,
    char const* prefix, ParticleSelection<Particle> const& particles) {
    if (!getOptions().starFormation.enabled) {
        return;
    }
    std::string const name(prefix);
    // Double point fields preserve each 32-bit limb exactly, including IDs > 2^53.
    writeViewField(db, metadata, (name + "id_low").c_str(), mesh, particles, "1",
        [](Particle const& particle) { return double(uint32_t(particle.id)); });
    writeViewField(db, metadata, (name + "id_high").c_str(), mesh, particles, "1",
        [](Particle const& particle) { return double(particle.id >> 32); });
    writeViewField(db, metadata, (name + "formation_time").c_str(), mesh, particles,
        getOptions().run.scenario == "cosmology" ? "1" : "s",
        [](Particle const& particle) { return particle.formationTime; });
}

template <typename Particle>
void writeViewSpecies(DBfile* db, ViewDirectoryMetadata& metadata, char const* species,
    ParticleSelection<Particle> const& particles, ViewUnitConversion const& conversion,
    char const* lengthUnit, char const* velocityUnit, char const* massUnit, double mass,
    double time, int cycle) {
    enterViewSpecies(db, species);
    char const* mesh = "mesh";
    auto* options = DBMakeOptlist(5);
    DBAddOption(options, DBOPT_XUNITS, const_cast<char*>(lengthUnit));
    DBAddOption(options, DBOPT_YUNITS, const_cast<char*>(lengthUnit));
    DBAddOption(options, DBOPT_ZUNITS, const_cast<char*>(lengthUnit));
    DBAddOption(options, DBOPT_DTIME, &time);
    DBAddOption(options, DBOPT_CYCLE, &cycle);
    writeParticlePointmesh(
        db, metadata, mesh, particles,
        [&conversion](Position const& value) { return conversion.position(value); }, options);
    DBFreeOptlist(options);
    writeViewField(db, metadata, "vx", mesh, particles, velocityUnit,
        [&conversion](Particle const& particle) { return conversion.velocity(particle.vx); });
    writeViewField(db, metadata, "vy", mesh, particles, velocityUnit,
        [&conversion](Particle const& particle) { return conversion.velocity(particle.vy); });
    writeViewField(db, metadata, "vz", mesh, particles, velocityUnit,
        [&conversion](Particle const& particle) { return conversion.velocity(particle.vz); });
    writeViewField(db, metadata, "rung", mesh, particles, "1",
        [](Particle const& particle) { return double(particle.rung); });
#if COSMICTIGER_WITH_SPH
    if constexpr (std::is_same_v<Particle, SphParticleInfo>) {
        writeGasMassExpression(db, conversion.mass(mass), massUnit);
    } else
#endif
    {
        writeViewField(db, metadata, "mass", mesh, particles, massUnit,
            [&conversion, mass](Particle const&) { return conversion.mass(mass); });
    }
    writeStarFormationFields(db, metadata, mesh, "", particles);
    checkViewWrite(DBSetDir(db, ".."), "..");
}

#if COSMICTIGER_WITH_SPH
void writeViscosityField(
    DBfile* db, ViewDirectoryMetadata& metadata, ParticleSelection<SphParticleInfo> const& gas) {
    if (getOptions().hydro.viscosityEvolution != "none") {
        writeViewField(db, metadata, "alpha", "mesh", gas, "1",
            [](SphParticleInfo const& particle) { return double(particle.alpha); });
    }
}

#if COSMICTIGER_ENABLE_STAR_FORMATION
void writeStarFormationGasFields(
    DBfile* db, ViewDirectoryMetadata& metadata, ParticleSelection<SphParticleInfo> const& gas) {
    if (!getOptions().starFormation.enabled) {
        return;
    }
    char const* names[] = {"cold_fraction", "hot_temperature", "specific_sfr"};
    char const* unitNames[] = {"1", "K", "1/s"};
    for (size_t field = 0; field < 3; ++field) {
        writeViewField(db, metadata, names[field], "mesh", gas, unitNames[field],
            [field](
                SphParticleInfo const& particle) { return particle.starFormationFields[field]; });
    }
}

#endif

#if COSMICTIGER_ENABLE_CHEMISTRY
void writeChemistryFields(
    DBfile* db, ViewDirectoryMetadata& metadata, ParticleSelection<SphParticleInfo> const& gas) {
    if (!getOptions().chemistry.enabled) {
        return;
    }
    char const* names[] = {"chem_HI", "chem_HII", "chem_HeI", "chem_HeII", "chem_HeIII",
        "chem_electrons", "temperature", "mean_molecular_weight", "frozen_gamma"};
    for (size_t field = 0; field < 9; ++field) {
        writeViewField(db, metadata, names[field], "mesh", gas, field == 6 ? "K" : "1",
            [field](SphParticleInfo const& particle) {
                return double(particle.chemistryFields[field]);
            });
    }
    writeViewText(db, "cooling_model", "GADGET-4 equilibrium atomic H/He; X=0.76; gamma=5/3");
    writeViewText(db, "cooling_abundance_convention",
        "Instantaneous equilibrium species counts per H nucleus; the saved electron "
        "abundance is an iteration guess, not a time-evolved ionization history.");
}
#endif

#endif
void writeConvertedView(DBfile* db, ViewDirectoryMetadata& metadata, ViewParticles const& particles,
    double a, double time, bool cosmological, bool lab, int cycle) {
    char const* lengthUnit = cosmological ? "Mpc/h" : "cm";
    char const* velocityUnit = cosmological ? "km/s" : "cm/s";
    char const* massUnit = cosmological ? "Msun/h" : "g";
    char const* densityUnit = cosmological ?
        "(Msun/h)/(Mpc/h)^3" :
        (kSpatialDimensions == 1 ? "g/cm" : (kSpatialDimensions == 2 ? "g/cm^2" : "g/cm^3"));
    char const* entropyUnit = cosmological ?
        "(km/s)^2/((Msun/h)/(Mpc/h)^3)^(gamma-1)" :
        (kSpatialDimensions == 1 ? "(erg/g)/(g/cm)^(gamma-1)" :
                                   (kSpatialDimensions == 2 ? "(erg/g)/(g/cm^2)^(gamma-1)" :
                                                              "(erg/g)/(g/cm^3)^(gamma-1)"));
    ViewUnitConversion const conversion(a, cosmological, lab);
    writeViewText(db, "unit_system", cosmological ? "cosmological" : "cgs");
    writeViewText(db, "frame", lab ? "lab" : "comoving");
    writeViewText(db, "position_convention", lab ? "proper" : "comoving");
    writeViewText(db, "density_convention", lab ? "proper" : "comoving");
    writeViewText(db, "velocity_convention", "peculiar");
    writeViewText(db, "hubble_flow", "excluded");
    writeViewText(db, "entropy_convention", "A = P_physical / rho_physical^gamma");
    writeViewText(db, "integration_state", "stored_staggered");
    writeViewText(db, "temporal_convention",
        "Positions are at the snapshot epoch; velocities and entropy retain the "
        "integrator's stored kick stage; density retains the last SPH density evaluation. "
        "Unit conversion does not synchronize these quantities in time.");
    writeViewText(db, "length_unit", lengthUnit);
    writeViewText(db, "velocity_unit", velocityUnit);
    writeViewText(db, "mass_unit", massUnit);
    writeViewText(db, "density_unit", densityUnit);
    writeViewText(db, "entropy_unit", entropyUnit);
    writeViewScalar(db, "adiabatic_index", getOptions().hydro.gamma);
    writeViewText(db, "time_unit", "Gyr");
    writeViewText(db, "time_convention", "cosmological_age");
    writeViewScalar(db, "time", time);
    writeViewScalar(db, "scale_factor", a);
    writeViewScalar(db, "hubble_parameter", getOptions().cosmology.hubble);
    writeViewScalar(db, "code_length_cm", getOptions().units.lengthCm);
    writeViewScalar(db, "code_mass_g", getOptions().units.massG);
    writeViewScalar(db, "code_time_s", getOptions().units.timeSeconds);

    if (!particles.darkMatter.empty()) {
        double const mass = getOptions().hydro.enabled ? getOptions().initial.darkMatterMass : 1.0;
        writeViewSpecies(db, metadata, "dark_matter", particles.darkMatter, conversion, lengthUnit,
            velocityUnit, massUnit, mass, time, cycle);
    }
#if COSMICTIGER_WITH_SPH
    if (!particles.stars.empty()) {
        writeViewSpecies(db, metadata, "stars", particles.stars, conversion, lengthUnit,
            velocityUnit, massUnit, getOptions().initial.gasMass, time, cycle);
    }
    auto const& gas = particles.gas;
    if (!gas.empty()) {
        writeViewSpecies(db, metadata, "gas", gas, conversion, lengthUnit, velocityUnit, massUnit,
            getOptions().initial.gasMass, time, cycle);
        checkViewWrite(DBSetDir(db, "gas"), "gas");
#if COSMICTIGER_ENABLE_CHEMISTRY
        writeChemistryFields(db, metadata, gas);
#endif
#if COSMICTIGER_ENABLE_STAR_FORMATION
        writeStarFormationGasFields(db, metadata, gas);
#endif
        writeViscosityField(db, metadata, gas);
        if (getOptions().hydro.viscosityGradientSwitch()) {
            writeViewField(db, metadata, "divv", "mesh", gas,
                cosmological ? "(km/s)/(Mpc/h)" : "1/s",
                [&conversion](SphParticleInfo const& particle) {
                    return conversion.frequency(particle.divergence);
                });
            writeViewText(db, "divv_convention",
                "Stored peculiar velocity gradient G = div_x(a v_pec) = a_sample^2 div_r(v_pec); "
                "current output-coordinate sample, excluding Hubble flow.");
        }
        writeViewField(db, metadata, "rho", "mesh", gas, densityUnit,
            [&conversion](
                SphParticleInfo const& particle) { return conversion.density(particle.rho); });
        writeViewField(db, metadata, "A", "mesh", gas, entropyUnit,
            [&conversion](
                SphParticleInfo const& particle) { return conversion.entropy(particle.entropy); });
        writeViewField(db, metadata, "r", "mesh", gas, lengthUnit,
            [&conversion](SphParticleInfo const& particle) {
                return conversion.length(particle.smoothingLength);
            });
        checkViewWrite(DBSetDir(db, ".."), "..");
    }
#endif
}

std::mutex siloViewMutex;

ViewSnapshot collectViewParticles(
    double a, double viscosityBaseTimestep, double viscosityGlobalTimestep) {
    ViewSnapshot snapshot;
#if COSMICTIGER_WITH_SPH && COSMICTIGER_ENABLE_STAR_FORMATION
    std::optional<star_formation_backend::Parameters> parameters;
    if (getOptions().starFormation.enabled) {
        parameters = starFormationParameters(a);
    }
#endif
    for (ParticleInteger i = 0; i < particlesSize(); ++i) {
        DarkMatterParticleInfo info{};
        info.x = particlesPos(kXDimension, i);
        info.y = particlesPos(kYDimension, i);
        info.z = particlesPos(kZDimension, i);
        info.vx = particlesVel(kXDimension, i);
        info.vy = particlesVel(kYDimension, i);
        info.vz = particlesVel(kZDimension, i);
        info.rung = particlesRung(i);
        if (getOptions().starFormation.enabled) {
            info.id = particlesStarFormationStates[i].id;
            info.formationTime = particlesStarFormationStates[i].formationTime;
        }
        if (particlesType(i) == kDarkMatterType) {
            snapshot.darkMatter.push_back(info);
        }
#if COSMICTIGER_WITH_SPH
        else if (particlesType(i) == kStarType) {
            StarParticleInfo star{};
            static_cast<DarkMatterParticleInfo&>(star) = info;
            snapshot.stars.push_back(star);
        } else if (particlesType(i) == kSphType) {
            SphParticleInfo gas{};
            static_cast<DarkMatterParticleInfo&>(gas) = info;
            int const index = particlesCatIndex(i);
            gas.entropy = sphParticlesEntropyInitialized ?
                sphParticlesEntropy(index) :
                sph_entropy::fromStoredEnergy(sphParticlesInternalEnergy(index),
                    sphParticlesDensity(index), getOptions().hydro.gamma, a);
            gas.rho = sphParticlesDensity(index);
            gas.smoothingLength = sphParticlesSmoothingLength(index);
            if (getOptions().hydro.viscosityEvolution != "none") {
                gas.alpha = sphParticlesBulkViscosityCoefficients[index];
            }
            if (getOptions().hydro.viscosityGradientSwitch()) {
                auto const& history = sphScaledDivergence[index];
                double const sampleTimestep =
                    sphViscosityStoredSampleTimestep(sphParticlesRung(index), viscosityBaseTimestep,
                        viscosityGlobalTimestep, getOptions().run.timeIntegrator == "global");
                if (!(sampleTimestep > 0.0)) {
                    THROW_ERROR("SPH divergence output requires its sample timestep\n");
                }
                gas.divergence = sphDecodeDivergence(history, sampleTimestep);
            }
#if COSMICTIGER_ENABLE_CHEMISTRY
            if (getOptions().chemistry.enabled) {
                gas.chemistryFields = sphChemistryOutput(sphParticlesElectronFractions[index],
                    sphParticlesInternalEnergy(index), gas.rho, a);
            }
#endif
#if COSMICTIGER_ENABLE_STAR_FORMATION
            if (parameters) {
                star_formation_backend::Input input{};
                input.state = particlesStarFormationStates[i];
                input.electronFraction = sphParticlesElectronFractions[index];
                input.energy = sphParticlesInternalEnergy(index);
                input.density = gas.rho;
                auto const phase = star_formation_backend::diagnostics(input, *parameters);
                if (phase.status != 0) {
                    throw std::runtime_error("Cannot evaluate GADGET-4 gas phase diagnostics");
                }
                double hotTemperature = 0.0;
                if (phase.starForming) {
                    double electrons = input.electronFraction;
                    double const density = double(gas.rho) * parameters->densityScale / (a * a * a);
                    if (gadget4::temperatureFromEnergy(phase.hotEnergy, density,
                            parameters->cooling, electrons,
                            hotTemperature) != gadget4::Status::Success) {
                        throw std::runtime_error("Cannot evaluate GADGET-4 hot-phase temperature");
                    }
                }
                gas.starFormationFields = {
                    phase.coldFraction, hotTemperature, input.state.specificStarFormationRate};
            }
#endif
            snapshot.gas.push_back(gas);
        }
#endif
    }
    return snapshot;
}

// Finish every locality before propagating the first transport or writer error.
template <typename Result>
std::vector<Result> consumeViewFutures(
    std::vector<hpx::future<Result>>& futures, std::exception_ptr error = {}) {
    std::vector<Result> results(futures.size());
    hpx::wait_all(futures);
    for (size_t i = 0; i < futures.size(); ++i) {
        try {
            results[i] = futures[i].get();
        } catch (...) {
            if (!error) {
                error = std::current_exception();
            }
        }
    }
    if (error) {
        std::rethrow_exception(error);
    }
    return results;
}

template <typename Action, typename Result, typename... Arguments>
auto onViewLocalities(Arguments const&... arguments) {
    std::vector<hpx::future<Result>> futures;
    std::exception_ptr error;
    try {
        for (auto const& locality : hpxLocalities()) {
            futures.push_back(hpx::async<Action>(locality, arguments...));
        }
    } catch (...) {
        error = std::current_exception();
    }
    return consumeViewFutures(futures, error);
}

struct ViewLeafMetadata {
    std::vector<std::vector<ViewDirectoryMetadata>> boxes;
    template <typename Archive>
    void serialize(Archive& archive, unsigned) {
        archive & boxes;
    }
};

struct ViewFrame {
    std::string path;
    bool cosmological = false;
    bool lab = false;
};

std::vector<ViewFrame> particleViewFrames() {
    if (getOptions().output.units == "code") {
        return {{"", false, false}};
    }
    std::vector<ViewFrame> frames;
    for (bool cosmological : {false, true}) {
        std::string const system = cosmological ? "cosmological" : "cgs";
        if (getOptions().output.units != "both" && getOptions().output.units != system) {
            continue;
        }
        for (bool lab : {false, true}) {
            std::string const frame = lab ? "lab" : "comoving";
            if (getOptions().output.frame == "both" || getOptions().output.frame == frame) {
                frames.push_back({system + "/" + frame, cosmological, lab});
            }
        }
    }
    return frames;
}

void enterViewDirectory(DBfile* db, std::string const& path) {
    checkViewWrite(DBSetDir(db, "/"), "/");
    size_t begin = 0;
    while (begin < path.size()) {
        auto const end = path.find('/', begin);
        auto const part = path.substr(begin, end - begin);
        if (!part.empty()) {
            auto const* contents = DBGetToc(db);
            if (!contents) {
                throw std::runtime_error("Cannot inspect Silo directory");
            }
            bool exists = false;
            for (int i = 0; i < contents->ndir; ++i) {
                exists = exists || part == contents->dir_names[i];
            }
            if (!exists) {
                checkViewWrite(DBMkDir(db, part.c_str()), part.c_str());
            }
            checkViewWrite(DBSetDir(db, part.c_str()), part.c_str());
        }
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
}

std::filesystem::path particleMasterPath(size_t box, int cycle) {
    return std::filesystem::path(getOptions().output.directory) /
        ("view." + std::to_string(box) + "." + std::to_string(cycle) + ".silo");
}

using ViewFile = std::unique_ptr<DBfile, decltype(&DBClose)>;

ViewFile createParticleViewFile(std::filesystem::path const& path) {
    ViewFile file(
        DBCreateReal(path.c_str(), DB_NOCLOBBER, DB_LOCAL, "Distributed particle output", DB_PDB),
        &DBClose);
    if (!file) {
        throw std::runtime_error("Cannot create particle Silo file " + path.string());
    }
    writeViewCommandLine(file.get());
    return file;
}

void writeParticleMetadata(DBfile* db) {
    if (getOptions().starFormation.enabled) {
        writeViewText(db, "particle_id_convention",
            "Persistent unsigned ID = id_high * 4294967296 + id_low; both limbs are exact.");
        writeViewText(db, "formation_time_convention",
            getOptions().run.scenario == "cosmology" ?
                "Scale factor at formation, as in GADGET-4; -1 for nonstellar particles." :
                "Physical seconds since simulation start at formation; -1 for nonstellar "
                "particles.");
        writeViewText(db, "star_formation_phase_convention",
            "GADGET-4 algebraic phases evaluated at the stored density and output epoch. "
            "cold_fraction and hot_temperature describe that equilibrium target, not "
            "persistent reservoirs. specific_sfr is the saved post-cooling long-lived "
            "stellar mass rate per gas mass per physical second used by the next conversion.");
    }
#if COSMICTIGER_WITH_SPH
    writeViewText(db, "gas_geometry", "SPH particle point mesh");
    writeViewText(db, "gas_field_convention", "Node-centered SPH particle values.");
    writeViewInteger(db, "dimension_count", kSpatialDimensions);
    writeViewText(db, "gas_entropy_convention", "A = P_physical / rho_physical^gamma");
    writeViewText(db, "gas_rung_convention",
        "Committed integration rung after hydro, enabled gravity and timestep limits.");
    writeViewText(db, "gas_mass_convention",
        "gas/mass is a constant expression for the shared SPH particle mass.");
#endif
}

void writeCodeParticleView(DBfile* db, ViewDirectoryMetadata& metadata,
    ViewParticles const& particles, double a, double time, int cycle) {
    auto* options = DBMakeOptlist(2);
    DBAddOption(options, DBOPT_DTIME, &time);
    DBAddOption(options, DBOPT_CYCLE, &cycle);
    auto const& darkMatter = particles.darkMatter;
    if (!darkMatter.empty()) {
        enterViewSpecies(db, "dark_matter");
        writeParticlePointmesh(
            db, metadata, "mesh", darkMatter,
            [](Position const& value) { return value.toDouble(); }, options);
        writeViewField(
            db, metadata, "vx", "mesh", darkMatter, "code", [](auto const& p) { return p.vx; });
        writeViewField(
            db, metadata, "vy", "mesh", darkMatter, "code", [](auto const& p) { return p.vy; });
        writeViewField(
            db, metadata, "vz", "mesh", darkMatter, "code", [](auto const& p) { return p.vz; });
        writeViewField(
            db, metadata, "rung", "mesh", darkMatter, "1", [](auto const& p) { return p.rung; });
        writeStarFormationFields(db, metadata, "mesh", "", darkMatter);
        checkViewWrite(DBSetDir(db, ".."), "..");
    }
#if COSMICTIGER_WITH_SPH
    auto const& stars = particles.stars;
    if (!stars.empty()) {
        enterViewSpecies(db, "stars");
        writeParticlePointmesh(
            db, metadata, "mesh", stars, [](Position const& value) { return value.toDouble(); },
            options);
        writeViewField(
            db, metadata, "vx", "mesh", stars, "code", [](auto const& p) { return p.vx; });
        writeViewField(
            db, metadata, "vy", "mesh", stars, "code", [](auto const& p) { return p.vy; });
        writeViewField(
            db, metadata, "vz", "mesh", stars, "code", [](auto const& p) { return p.vz; });
        writeViewField(
            db, metadata, "rung", "mesh", stars, "1", [](auto const& p) { return p.rung; });
        writeViewField(db, metadata, "mass", "mesh", stars, "code",
            [](auto const&) { return getOptions().initial.gasMass; });
        writeStarFormationFields(db, metadata, "mesh", "", stars);
        checkViewWrite(DBSetDir(db, ".."), "..");
    }
    auto const& gas = particles.gas;
    if (!gas.empty()) {
        enterViewSpecies(db, "gas");
        writeParticlePointmesh(
            db, metadata, "mesh", gas, [](Position const& value) { return value.toDouble(); },
            options);
        writeViewField(db, metadata, "vx", "mesh", gas, "code", [](auto const& p) { return p.vx; });
        writeViewField(db, metadata, "vy", "mesh", gas, "code", [](auto const& p) { return p.vy; });
        writeViewField(db, metadata, "vz", "mesh", gas, "code", [](auto const& p) { return p.vz; });
        writeViewField(
            db, metadata, "rung", "mesh", gas, "1", [](auto const& p) { return p.rung; });
        writeGasMassExpression(db, getOptions().initial.gasMass, "code");
        writeStarFormationFields(db, metadata, "mesh", "", gas);
        writeViewField(
            db, metadata, "rho", "mesh", gas, "code", [](auto const& p) { return p.rho; });
        writeViewField(
            db, metadata, "A", "mesh", gas, "code", [](auto const& p) { return p.entropy; });
        writeViewField(db, metadata, "r", "mesh", gas, "code",
            [](auto const& p) { return p.smoothingLength; });
#if COSMICTIGER_ENABLE_CHEMISTRY
        writeChemistryFields(db, metadata, gas);
#endif
#if COSMICTIGER_ENABLE_STAR_FORMATION
        writeStarFormationGasFields(db, metadata, gas);
#endif
        writeViscosityField(db, metadata, gas);
        if (getOptions().hydro.viscosityGradientSwitch()) {
            writeViewField(db, metadata, "divv", "mesh", gas, "1/code_time",
                [](SphParticleInfo const& particle) { return double(particle.divergence); });
            writeViewText(db, "divv_convention",
                "Stored peculiar velocity gradient G = div_x(a v_pec) = a_sample^2 div_r(v_pec); "
                "current output-coordinate sample, excluding Hubble flow.");
        }
        checkViewWrite(DBSetDir(db, ".."), "..");
    }
#endif
    DBFreeOptlist(options);
}

ViewLeafMetadata writeParticleLeaf(Vector<Range<double>> const& boxes, int cycle, double a,
    double time, std::string const& generation, double viscosityBaseTimestep,
    double viscosityGlobalTimestep) {
    auto const snapshot = collectViewParticles(a, viscosityBaseTimestep, viscosityGlobalTimestep);
    ViewLeafMetadata metadata;
    metadata.boxes.resize(boxes.size());
    auto const frames = particleViewFrames();
    std::string const locality = "locality_" + std::to_string(hpxRank());
    // Each locality writes only its owned particles and serializes its Silo calls.
    std::lock_guard fileLock(siloViewMutex);
    for (size_t box = 0; box < boxes.size(); ++box) {
        auto const directory =
            std::filesystem::path(particleMasterPath(box, cycle).string() + ".data");
        auto const filename = directory / generation / (locality + ".silo");
        auto file = createParticleViewFile(filename);
        ViewParticles particles;
        auto selectParticles = [&](auto const& source, auto& selected) {
            for (auto const& particle : source) {
                Vector3<double> position;
                position[0] = particle.x.toDouble();
                position[1] = particle.y.toDouble();
                position[2] = particle.z.toDouble();
                if (boxes[box].contains(position)) {
                    selected.push_back(particle);
                }
            }
        };
#if COSMICTIGER_WITH_SPH
        selectParticles(snapshot.gas, particles.gas);
        selectParticles(snapshot.stars, particles.stars);
#endif
        selectParticles(snapshot.darkMatter, particles.darkMatter);
        for (auto const& frame : frames) {
            ViewDirectoryMetadata directoryMetadata;
            directoryMetadata.path = frame.path;
            enterViewDirectory(file.get(), locality + (frame.path.empty() ? "" : "/" + frame.path));
            if (frame.path.empty()) {
                writeCodeParticleView(file.get(), directoryMetadata, particles, a, time, cycle);
            } else {
                writeConvertedView(file.get(), directoryMetadata, particles, a, time,
                    frame.cosmological, frame.lab, cycle);
            }
            writeParticleMetadata(file.get());
            metadata.boxes[box].push_back(std::move(directoryMetadata));
        }
        checkViewWrite(DBClose(file.release()), filename.c_str());
    }
    return metadata;
}

}    // namespace detail

HPX_PLAIN_ACTION(detail::writeParticleLeaf, WriteParticleLeafAction);

namespace detail {

void writeParticleMaster(size_t box, int cycle, double a, double time,
    std::vector<ViewLeafMetadata> const& leaves, std::string const& generation) {
    auto const filename = particleMasterPath(box, cycle);
    auto const temporary = std::filesystem::path(filename.string() + ".tmp." + generation);
    auto file = createParticleViewFile(temporary);
    writeViewText(file.get(), "leaf_generation", generation.c_str());
    auto const frames = particleViewFrames();
    ViewParticles empty;
    for (size_t frameIndex = 0; frameIndex < frames.size(); ++frameIndex) {
        auto const& frame = frames[frameIndex];
        enterViewDirectory(file.get(), frame.path);
        if (!frame.path.empty()) {
            ViewDirectoryMetadata emptyMetadata;
            writeConvertedView(
                file.get(), emptyMetadata, empty, a, time, frame.cosmological, frame.lab, cycle);
        }
        writeParticleMetadata(file.get());
        for (int species = 0; species < 3; ++species) {
            char const* speciesName =
                species == 0 ? "dark_matter" : (species == 1 ? "gas" : "stars");
            char const* mesh = "mesh";
            std::vector<std::string> prefixes(leaves.size());
            std::set<std::string> variables;
            for (size_t rank = 0; rank < leaves.size(); ++rank) {
                auto const& directory = leaves[rank].boxes.at(box).at(frameIndex);
                if (directory.path != frame.path) {
                    throw std::runtime_error("Inconsistent particle leaf directory metadata");
                }
                bool const present = species == 0 ?
                    directory.darkMatter :
                    (species == 1 ? directory.gas : directory.stars);
                if (present) {
                    std::string const locality = "locality_" + std::to_string(rank);
                    prefixes[rank] = filename.filename().string() + ".data/" + generation + "/" +
                        locality + ".silo:/" + locality +
                        (frame.path.empty() ? "" : "/" + frame.path) + "/" + speciesName + "/";
                    auto const& names = species == 0 ?
                        directory.darkMatterVariables :
                        (species == 1 ? directory.gasVariables : directory.starVariables);
                    variables.insert(names.begin(), names.end());
                }
            }
            if (std::all_of(prefixes.begin(), prefixes.end(),
                    [](std::string const& prefix) { return prefix.empty(); })) {
                continue;
            }
            if (prefixes.size() > size_t(std::numeric_limits<int>::max())) {
                throw std::runtime_error("Too many particle Silo blocks");
            }
            enterViewSpecies(file.get(), speciesName);
            auto putMulti = [&](char const* name, bool variable) {
                std::vector<std::string> objects;
                std::vector<char const*> names;
                std::vector<int> types(prefixes.size(), variable ? DB_POINTVAR : DB_POINTMESH);
                for (auto const& prefix : prefixes) {
                    objects.push_back(prefix.empty() ? "EMPTY" : prefix + name);
                }
                for (auto const& object : objects) {
                    names.push_back(object.c_str());
                }
                auto* options = DBMakeOptlist(5);
                DBAddOption(options, DBOPT_DTIME, &time);
                DBAddOption(options, DBOPT_CYCLE, &cycle);
                if (variable) {
                    DBAddOption(options, DBOPT_MMESH_NAME, const_cast<char*>(mesh));
                    checkViewWrite(DBPutMultivar(file.get(), name, names.size(), names.data(),
                                       types.data(), options),
                        name);
                } else {
                    checkViewWrite(DBPutMultimesh(file.get(), name, names.size(), names.data(),
                                       types.data(), options),
                        name);
                }
                DBFreeOptlist(options);
            };
            putMulti(mesh, false);
            for (auto const& variable : variables) {
                putMulti(variable.c_str(), true);
            }
#if COSMICTIGER_WITH_SPH
            if (species == 1) {
                double const mass = getOptions().initial.gasMass;
                ViewUnitConversion const conversion(a, frame.cosmological, frame.lab);
                writeGasMassExpression(file.get(),
                    frame.path.empty() ? mass : conversion.mass(mass),
                    frame.path.empty() ? "code" : (frame.cosmological ? "Msun/h" : "g"));
            }
#endif
            checkViewWrite(DBSetDir(file.get(), ".."), "..");
        }
    }
    checkViewWrite(DBClose(file.release()), temporary.c_str());
    // Only a fully closed master referencing fully closed, immutable leaves is published.
    std::filesystem::rename(temporary, filename);
}

void writeDistributedParticleViews(int cycle, double a, double time, double viscosityBaseTimestep,
    double viscosityGlobalTimestep) {
    static std::atomic<uint64_t> sequence{0};
    auto const ticks = std::chrono::system_clock::now().time_since_epoch().count();
    std::string const generation =
        "generation_" + std::to_string(ticks) + "_" + std::to_string(sequence.fetch_add(1));
    for (size_t box = 0; box < viewBoxes.size(); ++box) {
        auto const directory =
            std::filesystem::path(particleMasterPath(box, cycle).string() + ".data");
        std::filesystem::create_directories(directory);
        if (!std::filesystem::create_directory(directory / generation)) {
            throw std::runtime_error("Particle output generation already exists");
        }
    }
    auto const leaves = [&]() {
        profile::ScopedPhase profilePhase("output.particles.leaves");
        return onViewLocalities<WriteParticleLeafAction, ViewLeafMetadata>(
            viewBoxes, cycle, a, time, generation, viscosityBaseTimestep, viscosityGlobalTimestep);
    }();
    profile::ScopedPhase profilePhase("output.particles.master");
    for (size_t box = 0; box < viewBoxes.size(); ++box) {
        writeParticleMaster(box, cycle, a, time, leaves, generation);
    }
}

}    // namespace detail

void viewOutputViews(int cycle, double a, bool force, double physicalYears,
    double viscosityBaseTimestep, double viscosityGlobalTimestep) {
    if (!getOptions().output.silo || viewBoxes.empty() ||
        (!force && cycle % getOptions().output.siloInterval != 0)) {
        return;
    }
    double time = physicalYears >= 0.0 ? physicalYears / 1.0e9 : 0.0;
#if COSMICTIGER_WITH_GRAVITY
    if (physicalYears < 0.0) {
        time =
            cosmosTime(1e-6, a) * getOptions().units.timeSeconds * constants::kSecondsToYears / 1e9;
    }
#endif
    detail::writeDistributedParticleViews(
        cycle, a, time, viscosityBaseTimestep, viscosityGlobalTimestep);
}

void viewConfigure() {
    viewBoxes.clear();
    auto const& opts = getOptions();
    if (opts.run.test.empty()) {
        // Diagnostics and checkpoints also need the directory when Silo is disabled.
        outputPath(".");
    }
    if (!opts.output.silo) {
        if (terminal::verbose() || !opts.run.test.empty()) {
            std::cout << "Silo particle output: disabled (output.silo=false)\n" << std::flush;
        }
        return;
    }
    std::error_code directoryError;
    std::filesystem::create_directories(opts.output.directory, directoryError);
    if (directoryError) {
        THROW_ERROR("Cannot create output.directory %s: %s\n", opts.output.directory.c_str(),
            directoryError.message().c_str());
    }
    Range<double> viewBox;
    viewBox.begin[kXDimension] = opts.output.xMin;
    viewBox.end[kXDimension] = opts.output.xMax;
    viewBox.begin[kYDimension] = opts.output.yMin;
    viewBox.end[kYDimension] = opts.output.yMax;
    viewBox.begin[kZDimension] = opts.output.zMin;
    viewBox.end[kZDimension] = opts.output.zMax;
    viewBoxes.push_back(viewBox);
    if (terminal::verbose() || !opts.run.test.empty()) {
        std::cout << std::format("Silo particle output: x [{:g}, {:g}], y [{:g}, {:g}], z [{:g}, "
                                 "{:g}], every {:d} synchronized coarse steps\n",
                         opts.output.xMin, opts.output.xMax, opts.output.yMin, opts.output.yMax,
                         opts.output.zMin, opts.output.zMax, opts.output.siloInterval)
                  << std::flush;
    }
    if (terminal::verbose() || !opts.run.test.empty()) {
        std::cout << std::format(
                         "Silo particle output directory: {:s}\n", opts.output.directory.c_str())
                  << std::flush;
    }
}

#if COSMICTIGER_WITH_GRAVITY
Vector<float> outputView(int number, double time) {
    Vector<hpx::future<void>> futs;
    Vector<hpx::future<Vector<float>>> valFuts;
    for (auto const& c : hpxChildren()) {
        valFuts.push_back(hpx::async<OutputViewAction>(c, number, time));
    }
    int const numberOfThreads = hpxHardwareConcurrency();
    int const nside = getOptions().analysis.viewSize;
    int const npix = 12 * sqr(nside);
    Vector<std::atomic<float>> values(npix);
    for (auto& v : values) {
        v = 0.0f;
    }
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async(hpx::annotated_function(
            [numberOfThreads, proc, nside, &values]() {
                ParticleInteger const begin = (size_t) proc * particlesSize() / numberOfThreads;
                ParticleInteger const end = (size_t) (proc + 1) * particlesSize() / numberOfThreads;
                for (int i = begin; i < end; i++) {
                    for (int xi = -1; xi <= 0; xi++) {
                        for (int yi = -1; yi <= 0; yi++) {
                            for (int zi = -1; zi <= 0; zi++) {
                                Vector3<double> vec;
                                long int ipix;
                                vec[kXDimension] = particlesPos(kXDimension, i).toFloat() + xi;
                                vec[kYDimension] = particlesPos(kYDimension, i).toFloat() + yi;
                                vec[kZDimension] = particlesPos(kZDimension, i).toFloat() + zi;
                                float const r2 =
                                    sqr(vec[kXDimension], vec[kYDimension], vec[kZDimension]);
                                if (r2 < 1.0 && r2 > 0.0) {
                                    vec2pix_ring(nside, vec.data(), &ipix);
                                    atomicAdd(values[ipix], particlesMass(i) / r2);
                                }
                            }
                        }
                    }
                }
            },
            "project_view_mass")));
    }
    hpx::wait_all(futs.begin(), futs.end());
    Vector<float> results(values.size());
    for (int i = 0; i < results.size(); i++) {
        results[i] = values[i];
    }
    for (auto& valFut : valFuts) {
        auto const vals = valFut.get();
        for (int i = 0; i < vals.size(); i++) {
            results[i] += vals[i];
        }
    }
    if (hpxRank() == 0) {
        std::string const filename = outputPath("view." + std::to_string(number) + ".dat");
        FILE* fp = fopen(filename.c_str(), "wb");
        if (fp == NULL) {
            THROW_ERROR("unable to open %s for writing\n", filename.c_str());
        }
        bool const writeSucceeded = fwrite(&nside, sizeof(int), 1, fp) == 1 &&
            fwrite(&npix, sizeof(int), 1, fp) == 1 &&
            fwrite(results.data(), sizeof(float), npix, fp) == size_t(npix) &&
            fwrite(&number, sizeof(int), 1, fp) == 1 && fwrite(&time, sizeof(double), 1, fp) == 1;
        int const closeStatus = fclose(fp);
        if (!writeSucceeded || closeStatus != 0) {
            THROW_ERROR("Unable to complete HEALPix view output %s\n", filename.c_str());
        }
    }
    return results;
}
#endif
