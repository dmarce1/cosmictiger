/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <set>
#include <span>
#include <string>
#include <system_error>
#include <type_traits>
#include <variant>

#include <chealpix.h>
#include <hpx/threading_base/annotated_function.hpp>
#include <silo.h>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/options_metadata.hpp>
#include <cosmictiger/core/terminal.hpp>
#include <cosmictiger/cosmology/cosmology.hpp>
#include <cosmictiger/io/output.hpp>
#include <cosmictiger/io/silo_spheres.hpp>
#include <cosmictiger/io/view.hpp>
#include <cosmictiger/math/constants.hpp>
#include <cosmictiger/math/math.hpp>
#include <cosmictiger/math/units.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/sph_chemistry.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

HPX_PLAIN_ACTION(outputView, OutputViewAction);

static Vector<Range<double>> viewBoxes;

struct DarkMatterParticleInfo {
    Fixed32 x;
    Fixed32 y;
    Fixed32 z;
    float vx;
    float vy;
    float vz;
    int rung;
    template <typename A>
    void serialize(A&& arc, unsigned) {
        arc & x;
        arc & y;
        arc & z;
        arc & vx;
        arc & vy;
        arc & vz;
        arc & rung;
    }
};

struct SphParticleInfo : public DarkMatterParticleInfo {
    float eint;
    float rho;
    float h;
    primordial::State chemistry;
    template <typename A>
    void serialize(A&& arc, unsigned ver) {
        DarkMatterParticleInfo::serialize(arc, ver);
        arc & eint;
        arc & rho;
        arc & h;
        if (getOptions().chemistry.enabled) {
            for (auto& abundance : chemistry.abundances) {
                float raw = abundance.raw();
                arc & raw;
                abundance.setRaw(raw);
            }
        }
    }
};

struct ViewReturn {
    Vector<Vector<SphParticleInfo>> hydro;
    Vector<Vector<DarkMatterParticleInfo>> darkMatter;
    template <typename A>
    void serialize(A&& arc, unsigned) {
        arc & hydro;
        arc & darkMatter;
    }
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

    double position(Fixed32 const& value) const {
        // Keep every fixed-point bit until the explicit physical-output boundary.
        units::ComovingPosition<Fixed32> const position(value);
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

    double mass(double value) const {
        auto const mass = units::toMass(units::StoredMass(value), scales_);
        return cosmological_ ? units::toCosmologicalValue(mass, context_) : mass.value();
    }
};

template <typename Position, typename Radius>
void writeGasSpheres(DBfile* db, Vector<SphParticleInfo> const& particles, Position position,
    Radius radius, DBoptlist* options) {
    Vector<double> coefficients;
    coefficients.reserve(4 * particles.size());
    for (auto const& particle : particles) {
        coefficients.push_back(position(particle.x));
        coefficients.push_back(position(particle.y));
        coefficients.push_back(position(particle.z));
        coefficients.push_back(radius(particle.h));
    }
    writeSiloSpheres(db, "gas", "gas_zones",
        std::span<double const>(coefficients.data(), coefficients.size()), options);
}

void writeCodeGasField(
    DBfile* db, char const* name, Vector<float> const& values, DBoptlist* options) {
    writeSiloSphereVariable(
        db, name, "gas", std::span<float const>(values.data(), values.size()), options);
}

template <typename Particle, typename Value>
void writeViewField(DBfile* db, char const* name, char const* mesh,
    Vector<Particle> const& particles, char const* unitName, Value value) {
    Vector<double> values;
    values.reserve(particles.size());
    for (auto const& particle : particles) {
        values.push_back(value(particle));
    }
    auto* options = DBMakeOptlist(1);
    DBAddOption(options, DBOPT_UNITS, const_cast<char*>(unitName));
    if constexpr (std::is_same_v<Particle, SphParticleInfo>) {
        writeSiloSphereVariable(
            db, name, mesh, std::span<double const>(values.data(), values.size()), options);
    } else {
        checkViewWrite(
            DBPutPointvar1(db, name, mesh, values.data(), values.size(), DB_DOUBLE, options), name);
    }
    DBFreeOptlist(options);
}

template <typename Particle>
void writeViewSpecies(DBfile* db, char const* mesh, char const* prefix,
    Vector<Particle> const& particles, ViewUnitConversion const& conversion, char const* lengthUnit,
    char const* velocityUnit, char const* massUnit, double mass, double time) {
    auto* options = DBMakeOptlist(4);
    DBAddOption(options, DBOPT_XUNITS, const_cast<char*>(lengthUnit));
    DBAddOption(options, DBOPT_YUNITS, const_cast<char*>(lengthUnit));
    DBAddOption(options, DBOPT_ZUNITS, const_cast<char*>(lengthUnit));
    DBAddOption(options, DBOPT_DTIME, &time);
    if constexpr (std::is_same_v<Particle, SphParticleInfo>) {
        writeGasSpheres(
            db, particles,
            [&conversion](Fixed32 const& value) { return conversion.position(value); },
            [&conversion](float value) { return conversion.length(value); }, options);
    } else {
        Vector<double> x, y, z;
        x.reserve(particles.size());
        y.reserve(particles.size());
        z.reserve(particles.size());
        for (auto const& particle : particles) {
            x.push_back(conversion.position(particle.x));
            y.push_back(conversion.position(particle.y));
            z.push_back(conversion.position(particle.z));
        }
        double* coordinates[NDIM] = {x.data(), y.data(), z.data()};
        checkViewWrite(
            DBPutPointmesh(db, mesh, NDIM, coordinates, x.size(), DB_DOUBLE, options), mesh);
    }
    DBFreeOptlist(options);
    std::string const fieldPrefix(prefix);
    writeViewField(db, (fieldPrefix + "vx").c_str(), mesh, particles, velocityUnit,
        [&conversion](Particle const& particle) { return conversion.velocity(particle.vx); });
    writeViewField(db, (fieldPrefix + "vy").c_str(), mesh, particles, velocityUnit,
        [&conversion](Particle const& particle) { return conversion.velocity(particle.vy); });
    writeViewField(db, (fieldPrefix + "vz").c_str(), mesh, particles, velocityUnit,
        [&conversion](Particle const& particle) { return conversion.velocity(particle.vz); });
    writeViewField(db, (fieldPrefix + "rung").c_str(), mesh, particles, "1",
        [](Particle const& particle) { return double(particle.rung); });
    writeViewField(db, (fieldPrefix + "mass").c_str(), mesh, particles, massUnit,
        [&conversion, mass](Particle const&) { return conversion.mass(mass); });
}

void writeChemistryFields(DBfile* db, Vector<SphParticleInfo> const& gas, double a) {
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
        writeViewField(db, names[field], "gas", gas, field == 9 ? "K" : "1",
            [&](SphParticleInfo const&) { return double(fields[index++][field]); });
    }
    // Preserve the minority and branch bit even when displaying the dominant
    // abundance as an ordinary floating value would round it to one.
    for (int species = 0; species < primordial::kAbundanceCount; ++species) {
        std::string const name = "chem_signed_fraction_" + std::to_string(species);
        writeViewField(
            db, name.c_str(), "gas", gas, "1", [species](SphParticleInfo const& particle) {
                return double(particle.chemistry.abundances[species].raw());
            });
    }
    writeViewText(db, "chemistry_fraction_convention",
        "species counts per H nucleus; signed fractions preserve minority branches");
    writeViewText(db, "chemistry_network", getOptions().chemistry.network.c_str());
}

void writeConvertedView(DBfile* db, ViewReturn const& particles, int box, double a, double time,
    bool cosmological, bool lab) {
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

    if (!particles.darkMatter[box].empty()) {
        double const mass = getOptions().hydro.enabled ? getOptions().initial.darkMatterMass : 1.0;
        writeViewSpecies(db, "dark_matter", "", particles.darkMatter[box], conversion, lengthUnit,
            velocityUnit, massUnit, mass, time);
    }
    auto const& gas = particles.hydro[box];
    if (!gas.empty()) {
        writeViewSpecies(db, "gas", "hydro_", gas, conversion, lengthUnit, velocityUnit, massUnit,
            getOptions().initial.gasMass, time);
        writeChemistryFields(db, gas, a);
        writeViewField(
            db, "rho", "gas", gas, densityUnit, [&conversion](SphParticleInfo const& particle) {
                return conversion.density(particle.rho);
            });
        writeViewField(
            db, "eint", "gas", gas, energyUnit, [&conversion](SphParticleInfo const& particle) {
                return conversion.specificEnergy(particle.eint);
            });
        writeViewField(db, "kernel_support_radius", "gas", gas, lengthUnit,
            [&conversion](
                SphParticleInfo const& particle) { return conversion.length(particle.h); });
    }
}

void writeConvertedViews(DBfile* db, ViewReturn const& particles, int box, double a) {
    double const time =
        cosmosTime(1e-6, a) * getOptions().units.timeSeconds * constants::kSecondsToYears / 1e9;
    for (bool cosmological : {false, true}) {
        char const* systemName = cosmological ? "cosmological" : "cgs";
        if (getOptions().output.units != "both" && getOptions().output.units != systemName) {
            continue;
        }
        checkViewWrite(DBMkDir(db, systemName), systemName);
        checkViewWrite(DBSetDir(db, systemName), systemName);
        for (bool lab : {false, true}) {
            char const* frameName = lab ? "lab" : "comoving";
            if (getOptions().output.frame != "both" && getOptions().output.frame != frameName) {
                continue;
            }
            checkViewWrite(DBMkDir(db, frameName), frameName);
            checkViewWrite(DBSetDir(db, frameName), frameName);
            writeConvertedView(db, particles, box, a, time, cosmological, lab);
            checkViewWrite(DBSetDir(db, ".."), "..");
        }
        checkViewWrite(DBSetDir(db, ".."), "..");
    }
}

}    // namespace detail

ViewReturn viewGetParticles(Vector<Range<double>> boxes);

HPX_PLAIN_ACTION(viewGetParticles, ViewGetParticlesAction);

ViewReturn viewGetParticles(Vector<Range<double>> boxes = Vector<Range<double>>()) {
    if (hpxRank() == 0) {
        boxes = viewBoxes;
    }
    ViewReturn rc;
    rc.hydro.resize(boxes.size());
    rc.darkMatter.resize(boxes.size());
    Vector<hpx::future<ViewReturn>> futs;
    for (auto& c : hpxChildren()) {
        futs.push_back(hpx::async<ViewGetParticlesAction>(c, boxes));
    }
    int const numberOfThreads = hpxHardwareConcurrency();
    for (int proc = 0; proc < numberOfThreads; proc++) {
        futs.push_back(hpx::async(hpx::annotated_function(
            [proc, numberOfThreads, boxes]() {
                ViewReturn rc;
                rc.hydro.resize(boxes.size());
                rc.darkMatter.resize(boxes.size());
                ParticleInteger const b = (size_t) proc * particlesSize() / numberOfThreads;
                ParticleInteger const e = (size_t) (proc + 1) * particlesSize() / numberOfThreads;
                for (ParticleInteger i = b; i < e; i++) {
                    Array<double, NDIM> x;
                    x[XDIM] = particlesPos(XDIM, i).toDouble();
                    x[YDIM] = particlesPos(YDIM, i).toDouble();
                    x[ZDIM] = particlesPos(ZDIM, i).toDouble();
                    for (int j = 0; j < boxes.size(); j++) {
                        if (boxes[j].contains(x)) {
                            ParticleInteger const k = particlesType(i);
                            switch (k) {
                            case DARK_MATTER_TYPE: {
                                DarkMatterParticleInfo info{};
                                info.x = particlesPos(XDIM, i);
                                info.y = particlesPos(YDIM, i);
                                info.z = particlesPos(ZDIM, i);
                                info.vx = particlesVel(XDIM, i);
                                info.vy = particlesVel(YDIM, i);
                                info.vz = particlesVel(ZDIM, i);
                                info.rung = particlesRung(i);
                                rc.darkMatter[j].push_back(info);
                                break;
                            }
                            case SPH_TYPE: {
                                SphParticleInfo info{};
                                int const l = particlesCatIndex(i);
                                info.x = particlesPos(XDIM, i);
                                info.y = particlesPos(YDIM, i);
                                info.z = particlesPos(ZDIM, i);
                                info.vx = particlesVel(XDIM, i);
                                info.vy = particlesVel(YDIM, i);
                                info.vz = particlesVel(ZDIM, i);
                                info.eint = sphParticlesInternalEnergy(l);
                                info.rho = sphParticlesDensity(l);
                                info.h = sphParticlesSupportRadius(l);
                                if (getOptions().chemistry.enabled) {
                                    info.chemistry = sphParticlesChemistryStates[l];
                                }
                                info.rung = particlesRung(i);
                                rc.hydro[j].push_back(info);
                                break;
                            }
                            }
                        }
                    }
                }
                return rc;
            },
            "collect_view_particles")));
    }

    for (auto& f : futs) {
        auto tmp = f.get();
        for (int i = 0; i < boxes.size(); i++) {
            rc.hydro[i].insert(rc.hydro[i].begin(), tmp.hydro[i].begin(), tmp.hydro[i].end());
            rc.darkMatter[i].insert(
                rc.darkMatter[i].begin(), tmp.darkMatter[i].begin(), tmp.darkMatter[i].end());
        }
    }
    return rc;
}

void viewOutputViews(int cycle, double a, bool force) {
    if (!getOptions().output.silo || viewBoxes.empty() ||
        (!force && cycle % getOptions().output.siloInterval != 0)) {
        return;
    }
    profilerEnter("view_output_views");
    double const codeToVelocity = 1.;
    double const codeToEnergy = 1.;
    double const codeToDensity = 1.;
    for (int bi = 0; bi < viewBoxes.size(); bi++) {
        terminal::detailPrint("Writing view for box (%e %e) (%e %e) (%e %e)\n",
            viewBoxes[bi].begin[XDIM], viewBoxes[bi].end[XDIM], viewBoxes[bi].begin[YDIM],
            viewBoxes[bi].end[YDIM], viewBoxes[bi].begin[ZDIM], viewBoxes[bi].end[ZDIM]);
        auto const filename = std::filesystem::path(getOptions().output.directory) /
            ("view." + std::to_string(bi) + "." + std::to_string(cycle) + ".silo");
        DBfile* db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "Meshless", DB_PDB);
        if (db == nullptr) {
            THROW_ERROR("Cannot create particle Silo file %s\n", filename.c_str());
        }
        detail::writeViewCommandLine(db);
        ViewReturn parts;
        parts = viewGetParticles();
        if (getOptions().output.units != "code") {
            detail::writeConvertedViews(db, parts, bi, a);
            detail::checkViewWrite(DBClose(db), filename.c_str());
            continue;
        }
        Vector<float> x, y, z;
        auto opts = DBMakeOptlist(1);
        float z0 = 1.0 / a - 1.0;
        float tm =
            cosmosTime(1e-6, a) * getOptions().units.timeSeconds * constants::kSecondsToYears / 1e9;
        DBAddOption(opts, DBOPT_TIME, &tm);
        if (parts.darkMatter[bi].size()) {
            x.resize(0);
            y.resize(0);
            z.resize(0);
            for (int i = 0; i < parts.darkMatter[bi].size(); i++) {
                x.push_back(parts.darkMatter[bi][i].x.toFloat());
                y.push_back(parts.darkMatter[bi][i].y.toFloat());
                z.push_back(parts.darkMatter[bi][i].z.toFloat());
            }
            float* coords1[NDIM] = {x.data(), y.data(), z.data()};
            DBPutPointmesh(db, "dark_matter", NDIM, coords1, x.size(), DB_FLOAT, opts);
            x.resize(0);
            y.resize(0);
            z.resize(0);
            for (int i = 0; i < parts.darkMatter[bi].size(); i++) {
                x.push_back(parts.darkMatter[bi][i].vx * codeToVelocity);
                y.push_back(parts.darkMatter[bi][i].vy * codeToVelocity);
                z.push_back(parts.darkMatter[bi][i].vz * codeToVelocity);
            }
            DBPutPointvar1(db, "vx", "dark_matter", x.data(), x.size(), DB_FLOAT, opts);
            DBPutPointvar1(db, "vy", "dark_matter", y.data(), x.size(), DB_FLOAT, opts);
            DBPutPointvar1(db, "vz", "dark_matter", z.data(), x.size(), DB_FLOAT, opts);
            x.resize(0);
            for (int i = 0; i < parts.darkMatter[bi].size(); i++) {
                x.push_back(parts.darkMatter[bi][i].rung);
            }
            DBPutPointvar1(db, "rung", "dark_matter", x.data(), x.size(), DB_FLOAT, opts);
        }
        if (parts.hydro[bi].size()) {
            detail::writeGasSpheres(
                db, parts.hydro[bi], [](Fixed32 const& value) { return value.toDouble(); },
                [](float value) { return double(value); }, opts);
            x.resize(0);
            y.resize(0);
            z.resize(0);
            for (int i = 0; i < parts.hydro[bi].size(); i++) {
                x.push_back(parts.hydro[bi][i].vx * codeToVelocity);
                y.push_back(parts.hydro[bi][i].vy * codeToVelocity);
                z.push_back(parts.hydro[bi][i].vz * codeToVelocity);
            }
            detail::writeCodeGasField(db, "hydro_vx", x, opts);
            detail::writeCodeGasField(db, "hydro_vy", y, opts);
            detail::writeCodeGasField(db, "hydro_vz", z, opts);
            x.resize(0);
            for (int i = 0; i < parts.hydro[bi].size(); i++) {
                x.push_back(parts.hydro[bi][i].rung);
            }
            detail::writeCodeGasField(db, "hydro_rung", x, opts);
            x.resize(0);
            y.resize(0);
            z.resize(0);
            for (int i = 0; i < parts.hydro[bi].size(); i++) {
                x.push_back(parts.hydro[bi][i].rho * codeToDensity);
                y.push_back(parts.hydro[bi][i].eint * codeToEnergy);
                z.push_back(parts.hydro[bi][i].h);
            }
            detail::writeCodeGasField(db, "rho", x, opts);
            detail::writeCodeGasField(db, "eint", y, opts);
            detail::writeCodeGasField(db, "kernel_support_radius", z, opts);
            detail::writeChemistryFields(db, parts.hydro[bi], a);
            x.resize(0);
            y.resize(0);
            z.resize(0);
        }
        DBFreeOptlist(opts);
        DBClose(db);
    }
    profilerExit();
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
    viewBox.begin[XDIM] = opts.output.xMin;
    viewBox.end[XDIM] = opts.output.xMax;
    viewBox.begin[YDIM] = opts.output.yMin;
    viewBox.end[YDIM] = opts.output.yMax;
    viewBox.begin[ZDIM] = opts.output.zMin;
    viewBox.end[ZDIM] = opts.output.zMax;
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
    profilerEnter("output_view");
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
                                double vec[NDIM];
                                long int ipix;
                                vec[XDIM] = particlesPos(XDIM, i).toFloat() + xi;
                                vec[YDIM] = particlesPos(YDIM, i).toFloat() + yi;
                                vec[ZDIM] = particlesPos(ZDIM, i).toFloat() + zi;
                                float const r2 = sqr(vec[XDIM], vec[YDIM], vec[ZDIM]);
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
    profilerExit();
    return results;
}
