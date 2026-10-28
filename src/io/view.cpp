/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <variant>
#include <vector>

#include <chealpix.h>
#include <hpx/serialization/vector.hpp>
#include <hpx/threading_base/annotated_function.hpp>
#include <silo.h>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/options_metadata.hpp>
#include <cosmictiger/core/terminal.hpp>
#include <cosmictiger/cosmology/cosmology.hpp>
#include <cosmictiger/io/output.hpp>
#include <cosmictiger/io/view.hpp>
#include <cosmictiger/math/constants.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/units.hpp>
#include <cosmictiger/particles/particles.hpp>
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/sph_chemistry.hpp>
#include <cosmictiger/sph/sph_particles.hpp>
#endif

HPX_PLAIN_ACTION(outputView, OutputViewAction);

static Vector<Range<double>> viewBoxes;

struct DarkMatterParticleInfo {
    Position x;
    Position y;
    Position z;
    float vx;
    float vy;
    float vz;
    int rung;
};

#if COSMICTIGER_WITH_SPH
struct SphParticleInfo : public DarkMatterParticleInfo {
    float eint;
    float rho;
    float h;
    float alpha;
    float divergence;
    primordial::State chemistry;
};

#endif
struct ViewParticles {
#if COSMICTIGER_WITH_SPH
    Vector<SphParticleInfo> gas;
#endif
    Vector<DarkMatterParticleInfo> darkMatter;
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
        auto const separator = name.find('.');
        if (separator != std::string::npos) {
            auto const group = name.substr(0, separator);
            if (groups.insert(group).second) {
                checkViewWrite(DBMkDir(db, group.c_str()), group.c_str());
            }
            name[separator] = '/';
        }
        std::visit(
            [db, &name](auto const& value) {
                using Value = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<Value, std::string>) {
                    writeViewText(db, name.c_str(), value.c_str());
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
        units::StoredDensity const density(value);
        return lab_ ? displayValue(units::toProperDensity(density, scales_, context_)) :
                      displayValue(units::toComovingDensity(density, scales_));
    }

    double specificEnergy(double value) const {
        return displayValue(
            units::toPhysicalSpecificEnergy(units::StoredSpecificEnergy(value), scales_, context_));
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
    std::vector<std::string> gasVariables;
    std::vector<std::string> darkMatterVariables;
    template <typename Archive>
    void serialize(Archive& archive, unsigned) {
        archive & path & gas & darkMatter & gasVariables & darkMatterVariables;
    }
};

template <typename Particle, typename Position>
void writeParticlePointmesh(DBfile* db, ViewDirectoryMetadata& metadata, char const* mesh,
    Vector<Particle> const& particles, Position position, DBoptlist* options) {
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
    } else
#endif
    {
        metadata.darkMatter = true;
    }
}

template <typename Particle, typename Value>
void writeViewField(DBfile* db, ViewDirectoryMetadata& metadata, char const* name, char const* mesh,
    Vector<Particle> const& particles, char const* unitName, Value value) {
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
    } else
#endif
    {
        metadata.darkMatterVariables.emplace_back(name);
    }
    DBFreeOptlist(options);
}

template <typename Particle>
void writeViewSpecies(DBfile* db, ViewDirectoryMetadata& metadata, char const* mesh,
    char const* prefix, Vector<Particle> const& particles, ViewUnitConversion const& conversion,
    char const* lengthUnit, char const* velocityUnit, char const* massUnit, double mass,
    double time, int cycle) {
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
    std::string const fieldPrefix(prefix);
    writeViewField(db, metadata, (fieldPrefix + "vx").c_str(), mesh, particles, velocityUnit,
        [&conversion](Particle const& particle) { return conversion.velocity(particle.vx); });
    writeViewField(db, metadata, (fieldPrefix + "vy").c_str(), mesh, particles, velocityUnit,
        [&conversion](Particle const& particle) { return conversion.velocity(particle.vy); });
    writeViewField(db, metadata, (fieldPrefix + "vz").c_str(), mesh, particles, velocityUnit,
        [&conversion](Particle const& particle) { return conversion.velocity(particle.vz); });
    writeViewField(db, metadata, (fieldPrefix + "rung").c_str(), mesh, particles, "1",
        [](Particle const& particle) { return double(particle.rung); });
    writeViewField(db, metadata, (fieldPrefix + "mass").c_str(), mesh, particles, massUnit,
        [&conversion, mass](Particle const&) { return conversion.mass(mass); });
}

#if COSMICTIGER_WITH_SPH
void writeViscosityField(
    DBfile* db, ViewDirectoryMetadata& metadata, Vector<SphParticleInfo> const& gas) {
    if (getOptions().hydro.viscosityEvolution != "none") {
        writeViewField(db, metadata, "alpha", "gas", gas, "1",
            [](SphParticleInfo const& particle) { return double(particle.alpha); });
    }
}

void writeChemistryFields(
    DBfile* db, ViewDirectoryMetadata& metadata, Vector<SphParticleInfo> const& gas, double a) {
    if (!getOptions().chemistry.enabled) {
        return;
    }
    char const* names[] = {"chem_HI", "chem_HII", "chem_HeI", "chem_HeII", "chem_HeIII",
        "chem_Hminus", "chem_H2", "chem_H2plus", "chem_electrons", "temperature",
        "mean_molecular_weight", "frozen_gamma"};
    std::vector<std::array<float, 12>> fields;
    fields.reserve(gas.size());
    for (auto const& particle : gas) {
        fields.push_back(sphChemistryOutput(particle.chemistry, particle.eint, a));
    }
    for (size_t field = 0; field < 12; ++field) {
        size_t index = 0;
        writeViewField(db, metadata, names[field], "gas", gas, field == 9 ? "K" : "1",
            [&](SphParticleInfo const&) { return double(fields[index++][field]); });
    }
    // Preserve the minority and branch bit even when displaying the dominant
    // abundance as an ordinary floating value would round it to one.
    for (int species = 0; species < primordial::kAbundanceCount; ++species) {
        std::string const name = "chem_signed_fraction_" + std::to_string(species);
        writeViewField(db, metadata, name.c_str(), "gas", gas, "1",
            [species](SphParticleInfo const& particle) {
                return double(particle.chemistry.abundances[species].raw());
            });
    }
    writeViewText(db, "chemistry_fraction_convention",
        "species counts per H nucleus; signed fractions preserve minority branches");
    writeViewText(db, "chemistry_network", getOptions().chemistry.network.c_str());
}

#endif
void writeConvertedView(DBfile* db, ViewDirectoryMetadata& metadata, ViewParticles const& particles,
    double a, double time, bool cosmological, bool lab, int cycle) {
    char const* lengthUnit = cosmological ? "Mpc/h" : "cm";
    char const* velocityUnit = cosmological ? "km/s" : "cm/s";
    char const* massUnit = cosmological ? "Msun/h" : "g";
    char const* densityUnit = cosmological ? "(Msun/h)/(Mpc/h)^3" : "g/cm^3";
    char const* energyUnit = cosmological ? "(km/s)^2" : "erg/g";
    ViewUnitConversion const conversion(a, cosmological, lab);
    writeViewText(db, "unit_system", cosmological ? "cosmological" : "cgs");
    writeViewText(db, "frame", lab ? "lab" : "comoving");
    writeViewText(db, "position_convention", lab ? "proper" : "comoving");
    writeViewText(db, "density_convention", lab ? "proper" : "comoving");
    writeViewText(db, "velocity_convention", "peculiar");
    writeViewText(db, "hubble_flow", "excluded");
    writeViewText(db, "specific_energy_convention", "physical");
    writeViewText(db, "integration_state", "stored_staggered");
    writeViewText(db, "temporal_convention",
        "Positions are at the snapshot epoch; velocities and internal energy retain the "
        "integrator's stored kick stage; density retains the last SPH density evaluation. "
        "Unit conversion does not synchronize these quantities in time.");
    writeViewText(db, "length_unit", lengthUnit);
    writeViewText(db, "velocity_unit", velocityUnit);
    writeViewText(db, "mass_unit", massUnit);
    writeViewText(db, "density_unit", densityUnit);
    writeViewText(db, "specific_energy_unit", energyUnit);
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
        writeViewSpecies(db, metadata, "dark_matter", "", particles.darkMatter, conversion,
            lengthUnit, velocityUnit, massUnit, mass, time, cycle);
    }
#if COSMICTIGER_WITH_SPH
    auto const& gas = particles.gas;
    if (!gas.empty()) {
        writeViewSpecies(db, metadata, "gas", "hydro_", gas, conversion, lengthUnit, velocityUnit,
            massUnit, getOptions().initial.gasMass, time, cycle);
        writeChemistryFields(db, metadata, gas, a);
        writeViscosityField(db, metadata, gas);
        if (getOptions().hydro.viscosityEvolution == "cullenAndDehnen") {
            writeViewField(db, metadata, "divv", "gas", gas,
                cosmological ? "(km/s)/(Mpc/h)" : "1/s",
                [&conversion](SphParticleInfo const& particle) {
                    return conversion.frequency(particle.divergence);
                });
            writeViewText(db, "divv_convention",
                "Physical velocity divergence including Hubble flow; last SPH evaluation.");
        }
        writeViewField(db, metadata, "rho", "gas", gas, densityUnit,
            [&conversion](
                SphParticleInfo const& particle) { return conversion.density(particle.rho); });
        writeViewField(db, metadata, "eint", "gas", gas, energyUnit,
            [&conversion](SphParticleInfo const& particle) {
                return conversion.specificEnergy(particle.eint);
            });
        writeViewField(db, metadata, "kernel_support_radius", "gas", gas, lengthUnit,
            [&conversion](
                SphParticleInfo const& particle) { return conversion.length(particle.h); });
    }
#endif
}

std::mutex siloViewMutex;

ViewParticles collectViewParticles() {
    ViewParticles snapshot;
    for (ParticleInteger i = 0; i < particlesSize(); ++i) {
        DarkMatterParticleInfo info{};
        info.x = particlesPos(kXDimension, i);
        info.y = particlesPos(kYDimension, i);
        info.z = particlesPos(kZDimension, i);
        info.vx = particlesVel(kXDimension, i);
        info.vy = particlesVel(kYDimension, i);
        info.vz = particlesVel(kZDimension, i);
        info.rung = particlesRung(i);
        if (particlesType(i) == kDarkMatterType) {
            snapshot.darkMatter.push_back(info);
        }
#if COSMICTIGER_WITH_SPH
        else if (particlesType(i) == kSphType) {
            SphParticleInfo gas{};
            static_cast<DarkMatterParticleInfo&>(gas) = info;
            int const index = particlesCatIndex(i);
            gas.eint = sphParticlesInternalEnergy(index);
            gas.rho = sphParticlesDensity(index);
            gas.h = sphParticlesSupportRadius(index);
            if (getOptions().hydro.viscosityEvolution != "none") {
                gas.alpha = sphParticlesViscosityStates[index].alpha;
            }
            if (getOptions().hydro.viscosityEvolution == "cullenAndDehnen") {
                gas.divergence = sphParticlesCullenDehnenStates[index].divergence;
            }
            if (getOptions().chemistry.enabled) {
                gas.chemistry = sphParticlesChemistryStates[index];
            }
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
#if COSMICTIGER_WITH_SPH
    writeViewText(db, "gas_geometry", "SPH particle point mesh");
    writeViewText(db, "gas_field_convention", "Node-centered SPH particle values.");
    writeViewText(db, "gas_mass_convention", "hydro_mass is the mass of each SPH particle.");
#endif
}

void writeCodeParticleView(DBfile* db, ViewDirectoryMetadata& metadata,
    ViewParticles const& particles, double a, double time, int cycle) {
    auto* options = DBMakeOptlist(2);
    DBAddOption(options, DBOPT_DTIME, &time);
    DBAddOption(options, DBOPT_CYCLE, &cycle);
    auto const& darkMatter = particles.darkMatter;
    if (!darkMatter.empty()) {
        writeParticlePointmesh(
            db, metadata, "dark_matter", darkMatter,
            [](Position const& value) { return value.toDouble(); }, options);
        writeViewField(db, metadata, "vx", "dark_matter", darkMatter, "code",
            [](auto const& p) { return p.vx; });
        writeViewField(db, metadata, "vy", "dark_matter", darkMatter, "code",
            [](auto const& p) { return p.vy; });
        writeViewField(db, metadata, "vz", "dark_matter", darkMatter, "code",
            [](auto const& p) { return p.vz; });
        writeViewField(db, metadata, "rung", "dark_matter", darkMatter, "1",
            [](auto const& p) { return p.rung; });
    }
#if COSMICTIGER_WITH_SPH
    auto const& gas = particles.gas;
    if (!gas.empty()) {
        writeParticlePointmesh(
            db, metadata, "gas", gas, [](Position const& value) { return value.toDouble(); },
            options);
        writeViewField(
            db, metadata, "hydro_vx", "gas", gas, "code", [](auto const& p) { return p.vx; });
        writeViewField(
            db, metadata, "hydro_vy", "gas", gas, "code", [](auto const& p) { return p.vy; });
        writeViewField(
            db, metadata, "hydro_vz", "gas", gas, "code", [](auto const& p) { return p.vz; });
        writeViewField(
            db, metadata, "hydro_rung", "gas", gas, "1", [](auto const& p) { return p.rung; });
        writeViewField(db, metadata, "hydro_mass", "gas", gas, "code",
            [](auto const&) { return getOptions().initial.gasMass; });
        writeViewField(
            db, metadata, "rho", "gas", gas, "code", [](auto const& p) { return p.rho; });
        writeViewField(
            db, metadata, "eint", "gas", gas, "code", [](auto const& p) { return p.eint; });
        writeViewField(db, metadata, "kernel_support_radius", "gas", gas, "code",
            [](auto const& p) { return p.h; });
        writeChemistryFields(db, metadata, gas, a);
        writeViscosityField(db, metadata, gas);
        if (getOptions().hydro.viscosityEvolution == "cullenAndDehnen") {
            writeViewField(db, metadata, "divv", "gas", gas, "1/code_time",
                [](SphParticleInfo const& particle) { return double(particle.divergence); });
            writeViewText(db, "divv_convention",
                "Physical velocity divergence including Hubble flow; last SPH evaluation.");
        }
    }
#endif
    DBFreeOptlist(options);
}

ViewLeafMetadata writeParticleLeaf(Vector<Range<double>> const& boxes, int cycle, double a,
    double time, std::string const& generation) {
    auto const snapshot = collectViewParticles();
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
                Array<double, kDimensions> position;
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
        for (bool gas : {false, true}) {
            char const* mesh = gas ? "gas" : "dark_matter";
            std::vector<std::string> prefixes(leaves.size());
            std::set<std::string> variables;
            for (size_t rank = 0; rank < leaves.size(); ++rank) {
                auto const& directory = leaves[rank].boxes.at(box).at(frameIndex);
                if (directory.path != frame.path) {
                    throw std::runtime_error("Inconsistent particle leaf directory metadata");
                }
                if (gas ? directory.gas : directory.darkMatter) {
                    std::string const locality = "locality_" + std::to_string(rank);
                    prefixes[rank] = filename.filename().string() + ".data/" + generation + "/" +
                        locality + ".silo:/" + locality +
                        (frame.path.empty() ? "" : "/" + frame.path) + "/";
                    auto const& names =
                        gas ? directory.gasVariables : directory.darkMatterVariables;
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
    auto const leaves = onViewLocalities<WriteParticleLeafAction, ViewLeafMetadata>(
        viewBoxes, cycle, a, time, generation);
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
    double const time = physicalYears >= 0.0 ?
        physicalYears / 1.0e9 :
        cosmosTime(1e-6, a) * getOptions().units.timeSeconds * constants::kSecondsToYears / 1e9;
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
            PRINT("Silo particle output: disabled (output.silo=false)\n");
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
        PRINT("Silo particle output: x [%g, %g], y [%g, %g], z [%g, %g], every %d synchronized "
              "coarse "
              "steps\n",
            opts.output.xMin, opts.output.xMax, opts.output.yMin, opts.output.yMax,
            opts.output.zMin, opts.output.zMax, opts.output.siloInterval);
    }
    if (terminal::verbose() || !opts.run.test.empty()) {
        PRINT("Silo particle output directory: %s\n", opts.output.directory.c_str());
    }
}

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
                                double vec[kDimensions];
                                long int ipix;
                                vec[kXDimension] = particlesPos(kXDimension, i).toFloat() + xi;
                                vec[kYDimension] = particlesPos(kYDimension, i).toFloat() + yi;
                                vec[kZDimension] = particlesPos(kZDimension, i).toFloat() + zi;
                                float const r2 =
                                    sqr(vec[kXDimension], vec[kYDimension], vec[kZDimension]);
                                if (r2 < 1.0 && r2 > 0.0) {
                                    vec2pix_ring(nside, vec, &ipix);
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
        fwrite(&nside, sizeof(int), 1, fp);
        fwrite(&npix, sizeof(int), 1, fp);
        fwrite(results.data(), sizeof(float), npix, fp);
        fwrite(&number, sizeof(int), 1, fp);
        fwrite(&time, sizeof(double), 1, fp);
        fclose(fp);
    }
    return results;
}
