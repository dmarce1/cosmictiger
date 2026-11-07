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
#include <cosmictiger/chemistry/gadget4_cooling.hpp>
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

// VisIt's Silo reader GetTime() queries file-level dtime (double), then time
// (float). Mesh DBOPT_TIME/DBOPT_DTIME alone do not set the database time slider.
// https://github.com/visit-dav/visit/blob/develop/src/databases/Silo/avtSiloFileFormat.C
void writeViewClock(DBfile* db, double time, int cycle) {
    int const size = 1;
    float const singleTime = static_cast<float>(time);
    checkViewWrite(DBWrite(db, "time", &singleTime, &size, 1, DB_FLOAT), "time");
    writeViewScalar(db, "dtime", time);
    writeViewInteger(db, "cycle", cycle);
    bool const codeClock =
        getOptions().output.units == "code" && getOptions().run.scenario != "cosmology";
    writeViewText(db, "time_unit", codeClock ? "code" : "Gyr");
    writeViewText(db, "time_convention", codeClock ? "elapsed_code_time" : "cosmological_age");
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
    double time{};
    int cycle{};
    bool gas = false;
    bool darkMatter = false;
    bool stars = false;
    std::vector<std::string> gasVariables;
    std::vector<std::string> darkMatterVariables;
    std::vector<std::string> starVariables;
    template <typename Archive>
    void serialize(Archive& archive, unsigned) {
        archive & path & gas & darkMatter & stars & gasVariables & darkMatterVariables &
            starVariables & time & cycle;
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

std::string viewExpressionNumber(double value) {
    char buffer[64];
    auto const result = std::to_chars(buffer, buffer + sizeof(buffer), value,
        std::chars_format::general, std::numeric_limits<double>::max_digits10);
    if (result.ec != std::errc{}) {
        throw std::runtime_error("Cannot format Silo expression constant");
    }
    return std::string(buffer, result.ptr);
}

// Silo stores VisIt expressions, not additional particle arrays. Full paths keep
// the same definitions valid in locality files and their distributed masters.
// https://silo.readthedocs.io/latest/objects.html#dbputdefvars
void writeViewExpression(
    DBfile* db, std::string const& name, std::string const& expression, char const* unit) {
    auto const parent = std::filesystem::path(name).parent_path().string();
    if (!parent.empty()) {
        checkViewWrite(DBMkDirP(db, parent.c_str()), parent.c_str());
    }
    std::string const fullName = viewDirectory(db).substr(1) + "/" + name;
    char const* names[] = {fullName.c_str()};
    char const* definitions[] = {expression.c_str()};
    int const types[] = {DB_VARTYPE_SCALAR};
    auto* options = DBMakeOptlist(1);
    DBAddOption(options, DBOPT_UNITS, const_cast<char*>(unit));
    DBoptlist const* optionLists[] = {options};
    std::string const object = name + "_expression";
    checkViewWrite(DBPutDefvars(db, object.c_str(), 1, names, types, definitions, optionLists),
        object.c_str());
    DBFreeOptlist(options);
    writeViewText(db, (name + "_unit").c_str(), unit);
}

void writeParticleExpressions(
    DBfile* db, bool gas, double mass, double a, bool converted, bool cosmological, bool lab) {
    ViewUnitConversion const conversion(a, cosmological, lab);
    std::string const path = viewDirectory(db).substr(1) + "/";
    auto const field = [&path](std::string const& name) { return "<" + path + name + ">"; };
    auto const number = viewExpressionNumber;
    auto const& options = getOptions();
    writeViewExpression(db, "mass",
        "nodal_constant(" + field("mesh") + ", " +
            number(converted ? conversion.mass(mass) : mass) + ")",
        converted ? (cosmological ? "Msun/h" : "g") : "code");
    if (gas) {
        double const γ = options.hydro.gamma;
        // Derived from the stored-state definitions in docs/units.md:
        //   ρc = a³ ρphys, A = Pphys/ρphys^γ, Pstored = a⁵ Pphys.
        //   Pstored = A ρc^γ a^(5−3γ); Pphys = A ρview^γ a^(−3γ)
        // for a comoving density view, or A ρview^γ for a proper density view.
        // A → entropy tracer; ρview → density; γ → adiabatic index; a → scale factor.
        // Existing cosmological factors remain 3D; lower-dimensional SPH has a=1.
        double const pressureFactor =
            converted ? (lab ? 1.0 : std::pow(a, -3.0 * γ)) : std::pow(a, 5.0 - 3.0 * γ);
        char const* pressureUnit = !converted ? "code" :
            cosmological                      ? "(km/s)^2*(Msun/h)/(Mpc/h)^3" :
            kSpatialDimensions == 1           ? "erg/cm" :
            kSpatialDimensions == 2           ? "erg/cm^2" :
                                                "erg/cm^3";
        writeViewExpression(db, "pressure",
            field("entropyTracer") + " * (" + field("density") + " ^ " + number(γ) + ") * " +
                number(pressureFactor),
            pressureUnit);
        writeViewExpression(
            db, "adiabaticIndex", "nodal_constant(" + field("mesh") + ", " + number(γ) + ")", "1");
        writeViewText(db, "pressure_convention",
            converted ? "Physical pressure in either frame, reconstructed from stored entropy and "
                        "density." :
                        "Stored pressure = (gamma-1)*rho_c*E = a^5*physical pressure.");
        if (!options.chemistry.enabled) {
            // Derived neutral H/He ideal gas law, using the same normalization as
            // src/cosmology/initialize.cpp: T = (γ−1)u/(kB NA (1−3Y/4)).
            // Equivalently T = Pphys/ρphys /(kB NA (1−3Y/4)).
            // Y → helium fraction; 1−3Y/4 → neutral particle fraction;
            // Pphys/ρphys → pressure/density times the frame/unit conversion.
            // Code names: one_minus_3Y_over_4 = 1−3Y/4.
            double const one_minus_3Y_over_4 = 1.0 - 0.75 * options.cosmology.heliumFraction;
            double const energyFactor = converted ?
                (cosmological ? 1.0e10 : 1.0) * (lab ? 1.0 : a * a * a) :
                std::pow(options.units.lengthCm / (options.units.timeSeconds * a), 2);
            double const temperatureFactor =
                energyFactor / (constants::kB * constants::kAvo * one_minus_3Y_over_4);
            writeViewExpression(db, "temperature",
                field("pressure") + " / " + field("density") + " * " + number(temperatureFactor),
                "K");
            writeViewExpression(db, "meanMolecularWeight",
                "nodal_constant(" + field("mesh") + ", " + number(1.0 / one_minus_3Y_over_4) + ")",
                "1");
        }
#if COSMICTIGER_WITH_SPH && COSMICTIGER_ENABLE_CHEMISTRY
        if (options.chemistry.enabled) {
            // Exact equilibrium identities used by findAbundances/finishGas in
            // cosmictiger/chemistry/gadget4_cooling_impl.hpp (up to float output rounding):
            //   nₑ/nH = nH⁺/nH + nHe⁺/nH + 2 nHe²⁺/nH,
            //   μ = (1 + 4y)/(1 + y + nₑ/nH), y = (1−X)/(4X), X = 0.76.
            // nₑ/nH → electron abundance; y → helium number ratio;
            // μ → mean molecular weight; ion counts → corresponding abundance fields.
            // Retain individual species: subtraction can erase small trace abundances.
            double const y = gadget4::kHeliumNumberRatio;
            writeViewExpression(db, "electronAbundance",
                field("ionizedHydrogenAbundance") + " + " + field("singlyIonizedHeliumAbundance") +
                    " + 2 * " + field("doublyIonizedHeliumAbundance"),
                "1");
            writeViewExpression(db, "meanMolecularWeight",
                number(1.0 + 4.0 * y) + " / (" + number(1.0 + y) + " + " +
                    field("electronAbundance") + ")",
                "1");
        }
#endif
        writeViewText(db, "temperature_convention",
            options.chemistry.enabled ?
                "Equilibrium chemistry temperature at the stored thermal stage, in kelvin." :
                "Neutral H/He ideal-gas temperature in kelvin; composition uses "
                "cosmology.heliumFraction.");
    }
}

void writeParticleExpressions(DBfile* db, ViewParticles const& particles, double a,
    bool converted = false, bool cosmological = false, bool lab = false) {
    auto const write = [&](char const* species, bool gas, double mass) {
        checkViewWrite(DBSetDir(db, species), species);
        writeParticleExpressions(db, gas, mass, a, converted, cosmological, lab);
        checkViewWrite(DBSetDir(db, ".."), "..");
    };
    if (!particles.darkMatter.empty()) {
        write("darkMatter", false,
            getOptions().hydro.enabled ? getOptions().initial.darkMatterMass : 1.0);
    }
#if COSMICTIGER_WITH_SPH
    if (!particles.gas.empty()) {
        write("gas", true, getOptions().initial.gasMass);
    }
    if (!particles.stars.empty()) {
        write("stars", false, getOptions().initial.gasMass);
    }
#endif
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
    auto const parent = std::filesystem::path(name).parent_path().string();
    if (!parent.empty()) {
        checkViewWrite(DBMkDirP(db, parent.c_str()), parent.c_str());
    }
    std::string const meshPath = viewDirectory(db) + "/" + mesh;
    Vector<double> values;
    values.reserve(particles.size());
    for (auto const& particle : particles) {
        values.push_back(value(particle));
    }
    auto* options = DBMakeOptlist(4);
    float singleTime = static_cast<float>(metadata.time);
    DBAddOption(options, DBOPT_UNITS, const_cast<char*>(unitName));
    DBAddOption(options, DBOPT_TIME, &singleTime);
    DBAddOption(options, DBOPT_DTIME, &metadata.time);
    DBAddOption(options, DBOPT_CYCLE, &metadata.cycle);
    checkViewWrite(DBPutPointvar1(db, name, meshPath.c_str(), values.data(), int(values.size()),
                       DB_DOUBLE, options),
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
    writeViewField(db, metadata, (name + "particleIdLow").c_str(), mesh, particles, "1",
        [](Particle const& particle) { return double(uint32_t(particle.id)); });
    writeViewField(db, metadata, (name + "particleIdHigh").c_str(), mesh, particles, "1",
        [](Particle const& particle) { return double(particle.id >> 32); });
    writeViewField(db, metadata, (name + "formationTime").c_str(), mesh, particles,
        getOptions().run.scenario == "cosmology" ? "1" : "s",
        [](Particle const& particle) { return particle.formationTime; });
}

template <typename Particle>
void writeViewSpecies(DBfile* db, ViewDirectoryMetadata& metadata, char const* species,
    ParticleSelection<Particle> const& particles, ViewUnitConversion const& conversion,
    char const* lengthUnit, char const* velocityUnit, double time, int cycle) {
    enterViewSpecies(db, species);
    char const* mesh = "mesh";
    auto* options = DBMakeOptlist(6);
    float singleTime = static_cast<float>(time);
    DBAddOption(options, DBOPT_TIME, &singleTime);
    DBAddOption(options, DBOPT_XUNITS, const_cast<char*>(lengthUnit));
    DBAddOption(options, DBOPT_YUNITS, const_cast<char*>(lengthUnit));
    DBAddOption(options, DBOPT_ZUNITS, const_cast<char*>(lengthUnit));
    DBAddOption(options, DBOPT_DTIME, &time);
    DBAddOption(options, DBOPT_CYCLE, &cycle);
    writeParticlePointmesh(
        db, metadata, mesh, particles,
        [&conversion](Position const& value) { return conversion.position(value); }, options);
    DBFreeOptlist(options);
    writeViewField(db, metadata, "xVelocity", mesh, particles, velocityUnit,
        [&conversion](Particle const& particle) { return conversion.velocity(particle.vx); });
    if constexpr (kSpatialDimensions > 1) {
        writeViewField(db, metadata, "yVelocity", mesh, particles, velocityUnit,
            [&conversion](Particle const& particle) { return conversion.velocity(particle.vy); });
    }
    if constexpr (kSpatialDimensions > 2) {
        writeViewField(db, metadata, "zVelocity", mesh, particles, velocityUnit,
            [&conversion](Particle const& particle) { return conversion.velocity(particle.vz); });
    }
    writeViewField(db, metadata, "timestepRung", mesh, particles, "1",
        [](Particle const& particle) { return double(particle.rung); });
    writeStarFormationFields(db, metadata, mesh, "", particles);
    checkViewWrite(DBSetDir(db, ".."), "..");
}

#if COSMICTIGER_WITH_SPH
void writeViscosityField(
    DBfile* db, ViewDirectoryMetadata& metadata, ParticleSelection<SphParticleInfo> const& gas) {
    if (getOptions().hydro.viscosityEvolution != "none") {
        writeViewField(db, metadata, "bulkViscosityCoefficient", "mesh", gas, "1",
            [](SphParticleInfo const& particle) { return double(particle.alpha); });
    }
}

#if COSMICTIGER_ENABLE_STAR_FORMATION
void writeStarFormationGasFields(
    DBfile* db, ViewDirectoryMetadata& metadata, ParticleSelection<SphParticleInfo> const& gas) {
    if (!getOptions().starFormation.enabled) {
        return;
    }
    char const* names[] = {"coldGasMassFraction", "hotGasTemperature", "specificStarFormationRate"};
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
    char const* names[] = {"neutralHydrogenAbundance", "ionizedHydrogenAbundance",
        "neutralHeliumAbundance", "singlyIonizedHeliumAbundance", "doublyIonizedHeliumAbundance",
        "electronAbundance", "temperature"};
    for (size_t field = 0; field < std::size(names); ++field) {
        if (field == 5) {
            continue;    // Electron abundance is derived from the stored ions.
        }
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
    metadata.time = time;
    metadata.cycle = cycle;
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
    writeViewText(db, "entropy_convention", "entropyTracer = P_physical / rho_physical^gamma");
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
    writeViewClock(db, time, cycle);
    writeViewScalar(db, "scale_factor", a);
    writeViewScalar(db, "hubble_parameter", getOptions().cosmology.hubble);
    writeViewScalar(db, "code_length_cm", getOptions().units.lengthCm);
    writeViewScalar(db, "code_mass_g", getOptions().units.massG);
    writeViewScalar(db, "code_time_s", getOptions().units.timeSeconds);

    if (!particles.darkMatter.empty()) {
        writeViewSpecies(db, metadata, "darkMatter", particles.darkMatter, conversion, lengthUnit,
            velocityUnit, time, cycle);
    }
#if COSMICTIGER_WITH_SPH
    if (!particles.stars.empty()) {
        writeViewSpecies(db, metadata, "stars", particles.stars, conversion, lengthUnit,
            velocityUnit, time, cycle);
    }
    auto const& gas = particles.gas;
    if (!gas.empty()) {
        writeViewSpecies(
            db, metadata, "gas", gas, conversion, lengthUnit, velocityUnit, time, cycle);
        checkViewWrite(DBSetDir(db, "gas"), "gas");
#if COSMICTIGER_ENABLE_CHEMISTRY
        writeChemistryFields(db, metadata, gas);
#endif
#if COSMICTIGER_ENABLE_STAR_FORMATION
        writeStarFormationGasFields(db, metadata, gas);
#endif
        writeViscosityField(db, metadata, gas);
        writeViewField(db, metadata, "density", "mesh", gas, densityUnit,
            [&conversion](
                SphParticleInfo const& particle) { return conversion.density(particle.rho); });
        writeViewField(db, metadata, "entropyTracer", "mesh", gas, entropyUnit,
            [&conversion](
                SphParticleInfo const& particle) { return conversion.entropy(particle.entropy); });
        writeViewField(db, metadata, "kernelSupportRadius", "mesh", gas, lengthUnit,
            [&conversion](SphParticleInfo const& particle) {
                return conversion.length(particle.smoothingLength);
            });
        checkViewWrite(DBSetDir(db, ".."), "..");
    }
#endif
    writeParticleExpressions(db, particles, a, true, cosmological, lab);
}

std::mutex siloViewMutex;

ViewSnapshot collectViewParticles(double a) {
    ViewSnapshot snapshot;
#if COSMICTIGER_WITH_SPH && COSMICTIGER_ENABLE_STAR_FORMATION
    std::optional<star_formation_backend::Parameters> parameters;
    if (getOptions().starFormation.enabled) {
        parameters = starFormationParameters(a);
    }
#endif
    for (ParticleInteger i = 0; i < particlesSize(); ++i) {
        DarkMatterParticleInfo info{};
        info.x = particlesPositionValue(kXi, i);
        info.y = particlesPositionValue(kYi, i);
        info.z = particlesPositionValue(kZi, i);
        info.vx = particlesVelocityValue(kXi, i);
        info.vy = particlesVelocityValue(kYi, i);
        info.vz = particlesVelocityValue(kZi, i);
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
            "Persistent unsigned ID = particleIdHigh * 4294967296 + particleIdLow; both limbs are "
            "exact.");
        writeViewText(db, "formation_time_convention",
            getOptions().run.scenario == "cosmology" ?
                "Scale factor at formation, as in GADGET-4; -1 for nonstellar particles." :
                "Physical seconds since simulation start at formation; -1 for nonstellar "
                "particles.");
        writeViewText(db, "star_formation_phase_convention",
            "GADGET-4 algebraic phases evaluated at the stored density and output epoch. "
            "coldGasMassFraction and hotGasTemperature describe that equilibrium target, not "
            "persistent reservoirs. specificStarFormationRate is the saved post-cooling long-lived "
            "stellar mass rate per gas mass per physical second used by the next conversion.");
    }
#if COSMICTIGER_WITH_SPH
    writeViewText(db, "gas_geometry", "SPH particle point mesh");
    writeViewText(db, "gas_field_convention", "Node-centered SPH particle values.");
    writeViewInteger(db, "dimension_count", kSpatialDimensions);
    writeViewText(db, "gas_entropy_convention", "entropyTracer = P_physical / rho_physical^gamma");
    writeViewText(db, "gas_rung_convention",
        "Committed integration timestepRung after hydro, enabled gravity and timestep limits.");
    writeViewText(db, "gas_mass_convention",
        "gas/mass is a constant expression for the shared SPH particle mass.");
#endif
}

void writeCodeParticleView(DBfile* db, ViewDirectoryMetadata& metadata,
    ViewParticles const& particles, double a, double time, int cycle) {
    metadata.time = time;
    metadata.cycle = cycle;
    writeViewClock(db, time, cycle);
    auto* options = DBMakeOptlist(3);
    float singleTime = static_cast<float>(time);
    DBAddOption(options, DBOPT_TIME, &singleTime);
    DBAddOption(options, DBOPT_DTIME, &time);
    DBAddOption(options, DBOPT_CYCLE, &cycle);
    auto const& darkMatter = particles.darkMatter;
    if (!darkMatter.empty()) {
        enterViewSpecies(db, "darkMatter");
        writeParticlePointmesh(
            db, metadata, "mesh", darkMatter,
            [](Position const& value) { return value.toDouble(); }, options);
        writeViewField(db, metadata, "xVelocity", "mesh", darkMatter, "code",
            [](auto const& p) { return p.vx; });
        if constexpr (kSpatialDimensions > 1) {
            writeViewField(db, metadata, "yVelocity", "mesh", darkMatter, "code",
                [](auto const& p) { return p.vy; });
        }
        if constexpr (kSpatialDimensions > 2) {
            writeViewField(db, metadata, "zVelocity", "mesh", darkMatter, "code",
                [](auto const& p) { return p.vz; });
        }
        writeViewField(db, metadata, "timestepRung", "mesh", darkMatter, "1",
            [](auto const& p) { return p.rung; });
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
            db, metadata, "xVelocity", "mesh", stars, "code", [](auto const& p) { return p.vx; });
        if constexpr (kSpatialDimensions > 1) {
            writeViewField(db, metadata, "yVelocity", "mesh", stars, "code",
                [](auto const& p) { return p.vy; });
        }
        if constexpr (kSpatialDimensions > 2) {
            writeViewField(db, metadata, "zVelocity", "mesh", stars, "code",
                [](auto const& p) { return p.vz; });
        }
        writeViewField(
            db, metadata, "timestepRung", "mesh", stars, "1", [](auto const& p) { return p.rung; });
        writeStarFormationFields(db, metadata, "mesh", "", stars);
        checkViewWrite(DBSetDir(db, ".."), "..");
    }
    auto const& gas = particles.gas;
    if (!gas.empty()) {
        enterViewSpecies(db, "gas");
        writeParticlePointmesh(
            db, metadata, "mesh", gas, [](Position const& value) { return value.toDouble(); },
            options);
        writeViewField(
            db, metadata, "xVelocity", "mesh", gas, "code", [](auto const& p) { return p.vx; });
        if constexpr (kSpatialDimensions > 1) {
            writeViewField(
                db, metadata, "yVelocity", "mesh", gas, "code", [](auto const& p) { return p.vy; });
        }
        if constexpr (kSpatialDimensions > 2) {
            writeViewField(
                db, metadata, "zVelocity", "mesh", gas, "code", [](auto const& p) { return p.vz; });
        }
        writeViewField(
            db, metadata, "timestepRung", "mesh", gas, "1", [](auto const& p) { return p.rung; });
        writeStarFormationFields(db, metadata, "mesh", "", gas);
        writeViewField(
            db, metadata, "density", "mesh", gas, "code", [](auto const& p) { return p.rho; });
        writeViewField(db, metadata, "entropyTracer", "mesh", gas, "code",
            [](auto const& p) { return p.entropy; });
        writeViewField(db, metadata, "kernelSupportRadius", "mesh", gas, "code",
            [](auto const& p) { return p.smoothingLength; });
#if COSMICTIGER_ENABLE_CHEMISTRY
        writeChemistryFields(db, metadata, gas);
#endif
#if COSMICTIGER_ENABLE_STAR_FORMATION
        writeStarFormationGasFields(db, metadata, gas);
#endif
        writeViscosityField(db, metadata, gas);
        checkViewWrite(DBSetDir(db, ".."), "..");
    }
#endif
    DBFreeOptlist(options);
    writeParticleExpressions(db, particles, a);
}

ViewLeafMetadata writeParticleLeaf(Vector<Range<double>> const& boxes, int cycle, double a,
    double time, std::string const& generation) {
    auto const snapshot = collectViewParticles(a);
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
        writeViewClock(file.get(), time, cycle);
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
    writeViewClock(file.get(), time, cycle);
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
                species == 0 ? "darkMatter" : (species == 1 ? "gas" : "stars");
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
                float singleTime = static_cast<float>(time);
                DBAddOption(options, DBOPT_TIME, &singleTime);
                DBAddOption(options, DBOPT_DTIME, &time);
                DBAddOption(options, DBOPT_CYCLE, &cycle);
                if (variable) {
                    auto const parent = std::filesystem::path(name).parent_path().string();
                    if (!parent.empty()) {
                        checkViewWrite(DBMkDirP(file.get(), parent.c_str()), parent.c_str());
                    }
                    std::string const meshPath = viewDirectory(file.get()) + "/" + mesh;
                    DBAddOption(options, DBOPT_MMESH_NAME, const_cast<char*>(meshPath.c_str()));
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
            double const mass = species == 0 ?
                (getOptions().hydro.enabled ? getOptions().initial.darkMatterMass : 1.0) :
                getOptions().initial.gasMass;
            writeParticleExpressions(file.get(), species == 1, mass, a, !frame.path.empty(),
                frame.cosmological, frame.lab);
            checkViewWrite(DBSetDir(file.get(), ".."), "..");
        }
    }
    checkViewWrite(DBClose(file.release()), temporary.c_str());
    // Only a fully closed master referencing fully closed, immutable leaves is published.
    std::filesystem::rename(temporary, filename);
}

void writeDistributedParticleViews(int cycle, double a, double time) {
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
            viewBoxes, cycle, a, time, generation);
    }();
    profile::ScopedPhase profilePhase("output.particles.master");
    for (size_t box = 0; box < viewBoxes.size(); ++box) {
        writeParticleMaster(box, cycle, a, time, leaves, generation);
    }
}

}    // namespace detail

void viewOutputViews(int cycle, double a, bool force, double physicalYears) {
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
    if (getOptions().output.units == "code" && getOptions().run.scenario != "cosmology") {
        time = physicalYears >= 0.0 ?
            physicalYears / (getOptions().units.timeSeconds * constants::kSecondsToYears) :
            0.0;
    }
    detail::writeDistributedParticleViews(cycle, a, time);
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
    viewBox.begin[kXi] = opts.output.xMin;
    viewBox.end[kXi] = opts.output.xMax;
    viewBox.begin[kYi] = opts.output.yMin;
    viewBox.end[kYi] = opts.output.yMax;
    viewBox.begin[kZi] = opts.output.zMin;
    viewBox.end[kZi] = opts.output.zMax;
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
                                vec[kXi] = particlesPositionValue(kXi, i).toFloat() + xi;
                                vec[kYi] = particlesPositionValue(kYi, i).toFloat() + yi;
                                vec[kZi] = particlesPositionValue(kZi, i).toFloat() + zi;
                                float const r2 = sqr(vec[kXi], vec[kYi], vec[kZi]);
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
