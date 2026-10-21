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
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <silo.h>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/io/view.hpp>
#include <cosmictiger/math/constants.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

namespace detail {

void requireUnitsOutput(bool condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error("units output: " + message);
    }
}

void requireUnitsOutputClose(double actual, double expected, std::string const& message) {
    double const tolerance = 2.0e-12 * std::abs(expected);
    auto const bits = std::bit_cast<uint64_t>(actual);
    constexpr uint64_t kExponentMask = uint64_t(0x7ff) << 52;
    requireUnitsOutput(
        (bits & kExponentMask) != kExponentMask && std::abs(actual - expected) <= tolerance,
        message);
}

using UnitsOutputFile = std::unique_ptr<DBfile, decltype(&DBClose)>;
using UnitsOutputMesh = std::unique_ptr<DBpointmesh, decltype(&DBFreePointmesh)>;
using UnitsOutputField = std::unique_ptr<DBpointvar, decltype(&DBFreePointvar)>;

UnitsOutputFile openUnitsOutput(int cycle, int rank) {
    auto const filename = std::filesystem::path(getOptions().output.directory) /
        ("view.0." + std::to_string(cycle) + ".silo");
    UnitsOutputFile master(DBOpen(filename.c_str(), DB_UNKNOWN, DB_READ), DBClose);
    requireUnitsOutput(bool(master), "open master " + filename.string());
    int const length = DBGetVarLength(master.get(), "leaf_generation");
    requireUnitsOutput(length > 1, "master leaf generation");
    std::vector<char> generation(length);
    requireUnitsOutput(DBReadVar(master.get(), "leaf_generation", generation.data()) >= 0 &&
            generation.back() == '\0',
        "read master leaf generation");
    auto const leaf = std::filesystem::path(filename.string() + ".data") / generation.data() /
        ("locality_" + std::to_string(rank) + ".silo");
    UnitsOutputFile file(DBOpen(leaf.c_str(), DB_UNKNOWN, DB_READ), DBClose);
    requireUnitsOutput(bool(file), "open locality output " + leaf.string());
    auto const directory = "/locality_" + std::to_string(rank);
    requireUnitsOutput(DBSetDir(file.get(), directory.c_str()) >= 0, "locality directory");
    return file;
}

std::string readUnitsOutputText(DBfile* file, std::string const& name) {
    requireUnitsOutput(DBGetVarType(file, name.c_str()) == DB_CHAR, name + " text datatype");
    int const length = DBGetVarLength(file, name.c_str());
    requireUnitsOutput(length > 0, name + " text length");
    std::vector<char> value(length);
    requireUnitsOutput(DBReadVar(file, name.c_str(), value.data()) >= 0, name + " read");
    requireUnitsOutput(value.back() == '\0', name + " terminator");
    return value.data();
}

double readUnitsOutputScalar(DBfile* file, std::string const& name) {
    requireUnitsOutput(DBGetVarType(file, name.c_str()) == DB_DOUBLE, name + " scalar datatype");
    requireUnitsOutput(DBGetVarLength(file, name.c_str()) == 1, name + " scalar length");
    double value;
    requireUnitsOutput(DBReadVar(file, name.c_str(), &value) >= 0, name + " read");
    return value;
}

int readUnitsOutputInteger(DBfile* file, std::string const& name) {
    requireUnitsOutput(DBGetVarType(file, name.c_str()) == DB_INT, name + " integer datatype");
    requireUnitsOutput(DBGetVarLength(file, name.c_str()) == 1, name + " scalar length");
    int value;
    requireUnitsOutput(DBReadVar(file, name.c_str(), &value) >= 0, name + " read");
    return value;
}

void checkUnitsOutputCommandLine(DBfile* file) {
    requireUnitsOutputClose(
        readUnitsOutputScalar(file, "/cli/hydro/viscosityAlpha"), 1.75, "scalar viscosity alpha");
    requireUnitsOutputClose(
        readUnitsOutputScalar(file, "/cli/gravity/softening"), 0.125, "CLI softening units");
    requireUnitsOutputClose(
        readUnitsOutputScalar(file, "/cli/gravity/timestepEta"), 0.3, "CLI timestep eta");
    requireUnitsOutputClose(
        readUnitsOutputScalar(file, "/cli/hydro/minSupportRadius"), 0.2, "CLI minimum support");
    requireUnitsOutputClose(
        readUnitsOutputScalar(file, "/cli/hydro/maxSupportRadius"), 0.4, "CLI maximum support");
    requireUnitsOutputClose(readUnitsOutputScalar(file, "/cli/units/massSolar"),
        3.0e33 / constants::kM0, "CLI solar mass units");
    requireUnitsOutput(readUnitsOutputInteger(file, "/cli/hydro/enabled") == 1,
        "boolean option stored as scalar integer");
    requireUnitsOutput(readUnitsOutputInteger(file, "/cli/output/siloInterval") == 1,
        "integer option stored as scalar integer");
    requireUnitsOutput(readUnitsOutputText(file, "/cli/configuration") == "input with spaces.ini",
        "configuration path preserves spaces");
    requireUnitsOutput(
        readUnitsOutputText(file, "/cli/output/units") == getOptions().output.units &&
            readUnitsOutputText(file, "/cli/output/frame") == getOptions().output.frame,
        "effective current output options");
    requireUnitsOutput(
        readUnitsOutputText(file, "/cli/initial/gadget4Restart").empty(), "empty string option");
    int const argc = readUnitsOutputInteger(file, "/cli/argc");
    requireUnitsOutput(argc > 1, "original process argument count");
    bool foundHpxThreads = false;
    bool foundTest = false;
    for (int i = 0; i < argc; i++) {
        auto const argument = readUnitsOutputText(file, "/cli/argv/arg_" + std::to_string(i));
        foundHpxThreads = foundHpxThreads || argument == "--hpx:threads=2";
        foundTest = foundTest || argument == "--run.test=unitsOutput";
    }
    requireUnitsOutput(foundHpxThreads && foundTest, "original HPX and application arguments");
}

void checkUnitsOutputField(
    DBfile* file, std::string const& name, int count, double expected, char const* unit = nullptr) {
    UnitsOutputField field(DBGetPointvar(file, name.c_str()), DBFreePointvar);
    requireUnitsOutput(bool(field), name + " field exists");
    requireUnitsOutput(field->datatype == DB_DOUBLE, name + " double datatype");
    requireUnitsOutput(field->nels == count && field->nvals == 1, name + " shape");
    if (unit) {
        requireUnitsOutput(field->units && std::string(field->units) == unit, name + " units");
    }
    bool const gas = name.find("hydro_") != std::string::npos || name.ends_with("rho") ||
        name.ends_with("eint") || name.ends_with("kernel_support_radius");
    requireUnitsOutput(field->meshname &&
            std::filesystem::path(field->meshname).filename() == (gas ? "gas" : "dark_matter"),
        name + " attaches to the correct particle species");
    auto const* values = static_cast<double const*>(field->vals[0]);
    for (int i = 0; i < field->nels; i++) {
        requireUnitsOutputClose(values[i], expected, name + " value");
    }
}

void checkUnitsOutputGas(DBfile* file, std::string const& name, int count, double coordinateScale,
    char const* lengthUnit = nullptr, double time = 0.0) {
    UnitsOutputMesh mesh(DBGetPointmesh(file, name.c_str()), DBFreePointmesh);
    requireUnitsOutput(bool(mesh), name + " point mesh exists");
    requireUnitsOutput(mesh->ndims == NDIM && mesh->datatype == DB_DOUBLE && mesh->nels == count,
        name + " one point per particle");
    if (lengthUnit) {
        requireUnitsOutputClose(mesh->dtime, time, name + " mesh time metadata");
    }
    for (int dim = 0; dim < NDIM; dim++) {
        requireUnitsOutput(mesh->coords[dim], name + " particle coordinates");
        if (lengthUnit) {
            requireUnitsOutput(mesh->units[dim] && std::string(mesh->units[dim]) == lengthUnit,
                name + " coordinate unit");
        }
        auto const* coordinates = static_cast<double const*>(mesh->coords[dim]);
        double const expected =
            (0.5 + (dim == XDIM ? std::ldexp(1.0, -32) : 0.0)) * coordinateScale;
        for (int i = 0; i < count; i++) {
            requireUnitsOutputClose(coordinates[i], expected, name + " particle position");
        }
    }
}

void checkConvertedUnitsOutput(DBfile* file, bool cosmological, bool lab) {
    std::string const path =
        std::string(cosmological ? "cosmological/" : "cgs/") + (lab ? "lab/" : "comoving/");
    requireUnitsOutput(
        !DBInqVarExists(file, (path + "alpha").c_str()), "converted output omits particle alpha");
    double const a = 0.25;
    double const hubble = 0.7;
    double const length = 2.0e24;
    double const mass = 3.0e33;
    double const time = 5.0e15;
    double const coordinateScale =
        length * (lab ? a : 1.0) * (cosmological ? hubble / constants::kMpcToCm : 1.0);
    double const velocityScale = length / time / a * (cosmological ? 1.0e-5 : 1.0);
    double const energyScale =
        length * length / (time * time * a * a) * (cosmological ? 1.0e-10 : 1.0);
    double const densityScale = mass / (length * length * length) / (lab ? a * a * a : 1.0) *
        (cosmological ? std::pow(constants::kMpcToCm, 3) / (constants::kM0 * hubble * hubble) :
                        1.0);
    double const massScale = mass * (cosmological ? hubble / constants::kM0 : 1.0);
    char const* lengthUnit = cosmological ? "Mpc/h" : "cm";
    char const* velocityUnit = cosmological ? "km/s" : "cm/s";
    char const* massUnit = cosmological ? "Msun/h" : "g";
    char const* densityUnit = cosmological ? "(Msun/h)/(Mpc/h)^3" : "g/cm^3";
    char const* energyUnit = cosmological ? "(km/s)^2" : "erg/g";
    requireUnitsOutput(
        readUnitsOutputText(file, path + "unit_system") == (cosmological ? "cosmological" : "cgs"),
        "unit system metadata");
    requireUnitsOutput(
        readUnitsOutputText(file, path + "frame") == (lab ? "lab" : "comoving"), "frame metadata");
    requireUnitsOutput(readUnitsOutputText(file, path + "velocity_convention") == "peculiar",
        "peculiar velocity convention");
    requireUnitsOutput(
        readUnitsOutputText(file, path + "hubble_flow") == "excluded", "Hubble flow excluded");
    requireUnitsOutput(readUnitsOutputText(file, path + "specific_energy_convention") == "physical",
        "physical energy convention");
    requireUnitsOutput(readUnitsOutputText(file, path + "integration_state") == "stored_staggered",
        "explicit staggered snapshot convention");
    requireUnitsOutput(
        readUnitsOutputText(file, path + "temporal_convention").find("does not synchronize") !=
            std::string::npos,
        "temporal limitation metadata");
    requireUnitsOutputClose(readUnitsOutputScalar(file, path + "scale_factor"), a, "a metadata");
    requireUnitsOutputClose(
        readUnitsOutputScalar(file, path + "hubble_parameter"), hubble, "h metadata");
    requireUnitsOutputClose(readUnitsOutputScalar(file, path + "code_length_cm"), length,
        "length normalization metadata");
    requireUnitsOutputClose(
        readUnitsOutputScalar(file, path + "code_mass_g"), mass, "mass normalization metadata");
    requireUnitsOutputClose(
        readUnitsOutputScalar(file, path + "code_time_s"), time, "time normalization metadata");
    requireUnitsOutput(readUnitsOutputText(file, path + "time_unit") == "Gyr", "time unit");
    requireUnitsOutput(readUnitsOutputText(file, path + "time_convention") == "cosmological_age",
        "cosmological age convention");
    double const epoch = readUnitsOutputScalar(file, path + "time");
    requireUnitsOutput(epoch > 0.0, "positive cosmological age");
    UnitsOutputMesh mesh(DBGetPointmesh(file, (path + "dark_matter").c_str()), DBFreePointmesh);
    requireUnitsOutput(bool(mesh) && mesh->datatype == DB_DOUBLE && mesh->nels == 2,
        "dark matter mesh shape and datatype");
    requireUnitsOutputClose(mesh->dtime, epoch, "mesh time metadata");
    for (int dim = 0; dim < NDIM; dim++) {
        requireUnitsOutput(
            mesh->units[dim] && std::string(mesh->units[dim]) == lengthUnit, "coordinate unit");
        auto const* coordinates = static_cast<double const*>(mesh->coords[dim]);
        auto const bounds = std::minmax_element(coordinates, coordinates + mesh->nels);
        double const lo = *bounds.first;
        double const hi = *bounds.second;
        requireUnitsOutputClose(lo, 0.5 * coordinateScale, "first fixed position");
        requireUnitsOutputClose(hi,
            (0.5 + (dim == XDIM ? std::ldexp(1.0, -32) : 0.0)) * coordinateScale,
            "second fixed position");
        if (dim == XDIM) {
            requireUnitsOutput(hi > lo, "adjacent fixed-point positions stay distinct");
        }
    }
    checkUnitsOutputGas(file, path + "gas", 1, coordinateScale, lengthUnit, epoch);
    checkUnitsOutputField(file, path + "vx", 2, 0.0, velocityUnit);
    checkUnitsOutputField(file, path + "vy", 2, 0.0, velocityUnit);
    checkUnitsOutputField(file, path + "vz", 2, 0.0, velocityUnit);
    checkUnitsOutputField(file, path + "mass", 2, 0.75 * massScale, massUnit);
    checkUnitsOutputField(file, path + "hydro_vx", 1, 0.125 * velocityScale, velocityUnit);
    checkUnitsOutputField(file, path + "hydro_vy", 1, 0.25 * velocityScale, velocityUnit);
    checkUnitsOutputField(file, path + "hydro_vz", 1, -0.5 * velocityScale, velocityUnit);
    checkUnitsOutputField(file, path + "hydro_mass", 1, 0.25 * massScale, massUnit);
    checkUnitsOutputField(file, path + "rho", 1, 8.0 * densityScale, densityUnit);
    checkUnitsOutputField(file, path + "eint", 1, 4.0 * energyScale, energyUnit);
    checkUnitsOutputField(
        file, path + "kernel_support_radius", 1, 0.03125 * coordinateScale, lengthUnit);
}

}    // namespace detail

void unitsOutputParticleFixture(bool initialize);
HPX_PLAIN_ACTION(unitsOutputParticleFixture, UnitsOutputParticleFixtureAction);

void unitsOutputParticleFixture(bool initialize) {
    Vector<hpx::future<void>> pending;
    for (auto const& child : hpxChildren()) {
        pending.push_back(hpx::async<UnitsOutputParticleFixtureAction>(child, initialize));
    }
    if (initialize) {
        particlesResize(3);
        sphParticlesResize(1, false);
        for (ParticleInteger i = 0; i < 3; i++) {
            for (int dim = 0; dim < NDIM; dim++) {
                particlesPos(dim, i).setInteger(0x80000000U + (i != 0 && dim == XDIM ? 1U : 0U));
                particlesVel(dim, i) = i == 2 ? (dim == XDIM        ? 0.125 :
                                                        dim == YDIM ? 0.25 :
                                                                      -0.5) :
                                                0.0;
            }
            particlesRung(i) = 1;
            particlesType(i) = i == 2 ? SPH_TYPE : DARK_MATTER_TYPE;
            particlesCatIndex(i) = i == 2 ? 0 : NO_INDEX;
        }
        sphParticlesParticleIndex(0) = 2;
        sphParticlesSupportRadius(0) = 0.03125;
        sphParticlesDensity(0) = 8.0;
        sphParticlesThermodynamicStates[0].internalEnergy = 4.0;
    } else {
        detail::requireUnitsOutput(particlesPos(XDIM, 0).raw() == 0x80000000U &&
                particlesPos(XDIM, 1).raw() == 0x80000001U,
            "particle fixed-point storage unchanged on every locality");
    }
    for (auto& future : pending) {
        future.get();
    }
}

void unitsOutputTest() {
    using namespace detail;
    auto const originalOptions = getOptions();
    auto options = originalOptions;
    options.hydro.enabled = true;

    options.output.silo = true;
    options.output.siloInterval = 1;
    options.output.directory = "units_output_fixture";
    options.output.xMin = options.output.yMin = options.output.zMin = 0.0;
    options.output.xMax = options.output.yMax = options.output.zMax = 1.0;
    options.units.lengthCm = 2.0e24;
    options.units.massG = 3.0e33;
    options.units.timeSeconds = 5.0e15;
    options.cosmology.hubble = 0.7;
    options.initial.darkMatterMass = 0.75;
    options.initial.gasMass = 0.25;
    options.output.units = "code";
    options.output.frame = "comoving";
    options.hydro.viscosityAlpha = 1.75;
    options.gravity.supportRadius = 0.125 *
        selectedGravityKernel().supportRadiusPerEffectiveRadius /
        std::pow(double(options.initial.numberOfParticles), 1.0 / NDIM);
    options.gravity.timestepEta = 0.3;
    options.hydro.minSupportRadius = 0.2 / options.initial.particlesPerDimension;
    options.hydro.maxSupportRadius = 0.4 / options.initial.particlesPerDimension;
    options.configuration = "input with spaces.ini";
    setOptions(options);
    unitsOutputParticleFixture(true);
    viewConfigure();
    viewOutputViews(0, 0.25, true);
    for (int rank = 0; rank < hpxSize(); rank++) {
        auto file = openUnitsOutput(0, rank);
        checkUnitsOutputCommandLine(file.get());
        requireUnitsOutput(
            !DBInqVarExists(file.get(), "alpha"), "code-unit output omits particle alpha");
        UnitsOutputMesh mesh(DBGetPointmesh(file.get(), "dark_matter"), DBFreePointmesh);
        requireUnitsOutput(bool(mesh) && mesh->datatype == DB_DOUBLE && mesh->nels == 2,
            "code-unit dark matter mesh schema");
        auto const* positions = static_cast<double const*>(mesh->coords[XDIM]);
        auto const bounds = std::minmax_element(positions, positions + mesh->nels);
        requireUnitsOutputClose(*bounds.first, 0.5, "first code-unit dark matter position");
        requireUnitsOutputClose(
            *bounds.second, 0.5 + std::ldexp(1.0, -32), "second code-unit dark matter position");
        requireUnitsOutput(*bounds.second > *bounds.first,
            "code-unit output preserves adjacent fixed-point positions");
        checkUnitsOutputGas(file.get(), "gas", 1, 1.0);
        checkUnitsOutputField(file.get(), "hydro_vx", 1, 0.125);
        checkUnitsOutputField(file.get(), "hydro_rung", 1, 1.0);
        checkUnitsOutputField(file.get(), "rho", 1, 8.0);
        checkUnitsOutputField(file.get(), "eint", 1, 4.0);
        checkUnitsOutputField(file.get(), "kernel_support_radius", 1, 0.03125);
        requireUnitsOutput(
            !DBInqVarExists(file.get(), "cgs/lab/gas"), "code-unit output selection");
    }
    options.output.units = "both";
    options.output.frame = "both";
    setOptions(options);
    viewOutputViews(1, 0.25, true);
    for (int rank = 0; rank < hpxSize(); rank++) {
        auto file = openUnitsOutput(1, rank);
        checkUnitsOutputCommandLine(file.get());
        for (bool cosmological : {false, true}) {
            for (bool lab : {false, true}) {
                checkConvertedUnitsOutput(file.get(), cosmological, lab);
            }
        }
        requireUnitsOutput(
            !DBInqVarExists(file.get(), "gas"), "converted selection omits code mesh");
    }
    options.output.units = "cgs";
    options.output.frame = "lab";
    setOptions(options);
    viewOutputViews(2, 0.25, true);
    for (int rank = 0; rank < hpxSize(); rank++) {
        auto file = openUnitsOutput(2, rank);
        checkUnitsOutputCommandLine(file.get());
        checkConvertedUnitsOutput(file.get(), false, true);
        requireUnitsOutput(!DBInqVarExists(file.get(), "cgs/comoving/gas") &&
                !DBInqVarExists(file.get(), "cosmological/lab/gas"),
            "CGS lab selection");
    }
    options.output.units = "cosmological";
    options.output.frame = "comoving";
    setOptions(options);
    viewOutputViews(3, 0.25, true);
    for (int rank = 0; rank < hpxSize(); rank++) {
        auto file = openUnitsOutput(3, rank);
        checkUnitsOutputCommandLine(file.get());
        checkConvertedUnitsOutput(file.get(), true, false);
        requireUnitsOutput(!DBInqVarExists(file.get(), "cosmological/lab/gas") &&
                !DBInqVarExists(file.get(), "cgs/comoving/gas"),
            "cosmological comoving selection");
    }
    unitsOutputParticleFixture(false);
    setOptions(originalOptions);
    viewConfigure();
    PRINT("Boost.Units Silo output checks passed (%i localities)\n", hpxSize());
}
