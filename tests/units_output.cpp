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
#include <cstring>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <silo.h>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/io/view.hpp>
#include <cosmictiger/math/constants.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/particles.hpp>
#if COSMICTIGER_ENABLE_CHEMISTRY
#include <cosmictiger/chemistry/sph.hpp>
#endif
#if COSMICTIGER_ENABLE_STAR_FORMATION
#include <cosmictiger/stars/star_formation.hpp>
#endif

namespace detail {

constexpr float kUnitsOutputViscosityAlpha = 0.375f;
constexpr double kUnitsOutputPhysicalYears = 4.2e9;

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

double readUnitsOutputClock(DBfile* file, std::string const& path) {
    double const time = readUnitsOutputScalar(file, path + "dtime");
    std::string const name = path + "time";
    requireUnitsOutput(DBGetVarType(file, name.c_str()) == DB_FLOAT, "VisIt float time datatype");
    requireUnitsOutput(DBGetVarLength(file, name.c_str()) == 1, "VisIt float time length");
    float singleTime = 0.0f;
    requireUnitsOutput(DBReadVar(file, name.c_str(), &singleTime) >= 0, "VisIt float time read");
    requireUnitsOutput(singleTime == static_cast<float>(time), "VisIt float/double clocks agree");
    if (!path.empty()) {
        requireUnitsOutputClose(readUnitsOutputScalar(file, "/dtime"), time, "leaf root clock");
    }
    return time;
}

int readUnitsOutputInteger(DBfile* file, std::string const& name) {
    requireUnitsOutput(DBGetVarType(file, name.c_str()) == DB_INT, name + " integer datatype");
    requireUnitsOutput(DBGetVarLength(file, name.c_str()) == 1, name + " scalar length");
    int value;
    requireUnitsOutput(DBReadVar(file, name.c_str(), &value) >= 0, name + " read");
    return value;
}

void checkUnitsOutputCommandLine(DBfile* file) {
    requireUnitsOutputClose(readUnitsOutputScalar(file, "/cli/hydro/viscosityCoefficientMax"), 1.75,
        "scalar viscosity maximumBulkViscosityCoefficient");
    requireUnitsOutputClose(readUnitsOutputScalar(file, "/cli/gravity/effectiveRadius"), 0.125,
        "CLI effective radius units");
    requireUnitsOutputClose(
        readUnitsOutputScalar(file, "/cli/gravity/timestepEta"), 0.3, "CLI timestep eta");
    requireUnitsOutput(readUnitsOutputInteger(file, "/cli/gravity/expansionDegree") ==
            getOptions().gravity.fmmOrder - 1,
        "CLI potential expansion degree");
    requireUnitsOutputClose(
        readUnitsOutputScalar(file, "/cli/hydro/minSmoothingLength"), 0.2, "CLI minimum support");
    requireUnitsOutputClose(
        readUnitsOutputScalar(file, "/cli/hydro/maxSmoothingLength"), 0.4, "CLI maximum support");
    requireUnitsOutputClose(readUnitsOutputScalar(file, "/cli/units/massSolar"),
        3.0e33 / constants::kM0, "CLI solar mass units");
    requireUnitsOutputClose(readUnitsOutputScalar(file, "/cli/units/boxLength"),
        2.0e24 * 0.7 / constants::kMpcToCm, "resolved comoving box length in Mpc/h");
    requireUnitsOutput(readUnitsOutputInteger(file, "/cli/hydro/enabled") == 1,
        "boolean option stored as scalar integer");
    requireUnitsOutput(readUnitsOutputInteger(file, "/cli/run/numberOfOutputs") ==
            getOptions().run.numberOfOutputs,
        "integer option stored as scalar integer");
    requireUnitsOutput(readUnitsOutputText(file, "/cli/configuration") == "input with spaces.ini",
        "configuration path preserves spaces");
    requireUnitsOutput(
        readUnitsOutputText(file, "/cli/output/units") == getOptions().output.units &&
            readUnitsOutputText(file, "/cli/output/frame") == getOptions().output.frame,
        "effective current output options");
    requireUnitsOutput(readUnitsOutputText(file, "/cli/scenario/cosmology/gadget4Restart").empty(),
        "empty string option");
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
    requireUnitsOutput(
        field->meshname && std::filesystem::path(field->meshname).filename() == "mesh",
        name + " attaches to the species mesh");
    auto const* values = static_cast<double const*>(field->vals[0]);
    for (int i = 0; i < field->nels; i++) {
        requireUnitsOutputClose(values[i], expected, name + " value");
    }
}

void checkUnitsOutputMassExpression(DBfile* file, std::string const& path, double expected,
    char const* unit, std::string const& species = "gas") {
    std::string const gas = path + species + "/";
    requireUnitsOutput(!DBInqVarExists(file, (gas + "mass").c_str()),
        "SPH mass is not stored as a particle array");
    std::unique_ptr<DBdefvars, decltype(&DBFreeDefvars)> definition(
        DBGetDefvars(file, (gas + "mass_expression").c_str()), DBFreeDefvars);
    char directory[4096];
    requireUnitsOutput(DBGetDir(file, directory) >= 0, "current output directory");
    std::string const fullPath = std::string(directory + 1) + "/" + gas;
    requireUnitsOutput(definition && definition->ndefs == 1 &&
            definition->types[0] == DB_VARTYPE_SCALAR &&
            std::string(definition->names[0]) == fullPath + "mass",
        "mass is a single scalar expression with the species path");
    std::string const expression(definition->defns[0]);
    std::string const begin = "nodal_constant(<" + fullPath + "mesh>, ";
    requireUnitsOutput(expression.starts_with(begin) && expression.ends_with(")"),
        "mass expression is node centered on the gas mesh");
    requireUnitsOutputClose(
        std::stod(expression.substr(begin.size())), expected, "shared SPH mass value");
    requireUnitsOutput(readUnitsOutputText(file, gas + "mass_unit") == unit, "mass units");
}

void checkUnitsOutputExpression(DBfile* file, std::string const& path, char const* name,
    char const* unit, char const* dependency) {
    std::string const field = path + "gas/" + name;
    requireUnitsOutput(
        !DBInqVarExists(file, field.c_str()), "derived quantity has no particle array");
    std::unique_ptr<DBdefvars, decltype(&DBFreeDefvars)> expression(
        DBGetDefvars(file, (field + "_expression").c_str()), DBFreeDefvars);
    char directory[4096];
    requireUnitsOutput(DBGetDir(file, directory) >= 0, "expression directory");
    std::string const fullName = std::string(directory + 1) + "/" + field;
    requireUnitsOutput(expression && expression->ndefs == 1 &&
            std::string(expression->names[0]) == fullName &&
            expression->types[0] == DB_VARTYPE_SCALAR,
        "fully qualified scalar expression " + field);
    requireUnitsOutput(std::string(expression->defns[0]).find(dependency) != std::string::npos,
        "expression dependency " + field);
    requireUnitsOutput(readUnitsOutputText(file, field + "_unit") == unit, "expression units");
}

void checkUnitsOutputSourceFields(DBfile* file, std::string const& path, int rank) {
#if COSMICTIGER_ENABLE_CHEMISTRY
    auto const chemistry = sphChemistryOutput(sphParticlesElectronFractions[0],
        sphParticlesInternalEnergy(0), sphParticlesDensity(0), 0.25);
    char const* names[] = {"neutralHydrogenAbundance", "ionizedHydrogenAbundance",
        "neutralHeliumAbundance", "singlyIonizedHeliumAbundance", "doublyIonizedHeliumAbundance",
        "electronAbundance", "temperature"};
    for (size_t field = 0; field < std::size(names); ++field) {
        if (field == 5) {
            continue;
        }
        checkUnitsOutputField(
            file, path + "gas/" + names[field], 1, chemistry[field], field == 6 ? "K" : "1");
    }
    checkUnitsOutputExpression(file, path, "electronAbundance", "1", "ionizedHydrogenAbundance");
    checkUnitsOutputExpression(file, path, "meanMolecularWeight", "1", "electronAbundance");
#endif
#if COSMICTIGER_ENABLE_STAR_FORMATION
    auto const phase = starFormationDiagnostics(0, 0.25);
    double hotTemperature = 0.0;
    if (phase.starForming) {
        auto const parameters = starFormationParameters(0.25);
        double electrons = sphParticlesElectronFractions[0];
        double const density =
            double(sphParticlesDensity(0)) * parameters.densityScale / (0.25 * 0.25 * 0.25);
        requireUnitsOutput(
            gadget4::temperatureFromEnergy(phase.hotEnergy, density, parameters.cooling, electrons,
                hotTemperature) == gadget4::Status::Success,
            "reference hot-phase temperature");
    }
    checkUnitsOutputField(file, path + "gas/coldGasMassFraction", 1, phase.coldFraction, "1");
    checkUnitsOutputField(file, path + "gas/hotGasTemperature", 1, hotTemperature, "K");
    checkUnitsOutputField(file, path + "gas/specificStarFormationRate", 1,
        particlesStarFormationStates[2].specificStarFormationRate, "1/s");
    uint64_t const identity = (uint64_t{1} << 60) + 3 * uint64_t(rank) + 3;
    checkUnitsOutputField(file, path + "gas/particleIdLow", 1, uint32_t(identity), "1");
    checkUnitsOutputField(file, path + "gas/particleIdHigh", 1, identity >> 32, "1");
#endif
}

void checkUnitsOutputViscosity(DBfile* file, std::string const& path) {
    std::string const name = path + "gas/bulkViscosityCoefficient";
    checkUnitsOutputField(file, name, 1, double(kUnitsOutputViscosityAlpha), "1");
}

void checkUnitsOutputGas(DBfile* file, std::string const& name, int count, double coordinateScale,
    char const* lengthUnit = nullptr, double time = 0.0) {
    UnitsOutputMesh mesh(DBGetPointmesh(file, name.c_str()), DBFreePointmesh);
    requireUnitsOutput(bool(mesh), name + " point mesh exists");
    requireUnitsOutput(
        mesh->ndims == kDimensions && mesh->datatype == DB_DOUBLE && mesh->nels == count,
        name + " one point per particle");
    if (lengthUnit) {
        requireUnitsOutputClose(mesh->dtime, time, name + " mesh time metadata");
    }
    for (int dim = 0; dim < kDimensions; dim++) {
        requireUnitsOutput(mesh->coords[dim], name + " particle coordinates");
        if (lengthUnit) {
            requireUnitsOutput(mesh->units[dim] && std::string(mesh->units[dim]) == lengthUnit,
                name + " coordinate unit");
        }
        auto const* coordinates = static_cast<double const*>(mesh->coords[dim]);
        double const expected = (0.5 + (dim == kXi ? std::ldexp(1.0, -32) : 0.0)) * coordinateScale;
        for (int i = 0; i < count; i++) {
            requireUnitsOutputClose(coordinates[i], expected, name + " particle position");
        }
    }
}

void checkConvertedUnitsOutput(DBfile* file, bool cosmological, bool lab) {
    std::string const path =
        std::string(cosmological ? "cosmological/" : "cgs/") + (lab ? "lab/" : "comoving/");
    checkUnitsOutputViscosity(file, path);
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
    char const* entropyUnit =
        cosmological ? "(km/s)^2/((Msun/h)/(Mpc/h)^3)^(gamma-1)" : "(erg/g)/(g/cm^3)^(gamma-1)";
    requireUnitsOutput(
        readUnitsOutputText(file, path + "unit_system") == (cosmological ? "cosmological" : "cgs"),
        "unit system metadata");
    requireUnitsOutput(
        readUnitsOutputText(file, path + "frame") == (lab ? "lab" : "comoving"), "frame metadata");
    requireUnitsOutput(readUnitsOutputText(file, path + "velocity_convention") == "peculiar",
        "peculiar velocity convention");
    requireUnitsOutput(
        readUnitsOutputText(file, path + "hubble_flow") == "excluded", "Hubble flow excluded");
    requireUnitsOutput(readUnitsOutputText(file, path + "entropy_convention") ==
            "entropyTracer = P_physical / rho_physical^gamma",
        "physical entropy convention");
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
    double const epoch = readUnitsOutputClock(file, path);
    requireUnitsOutput(epoch > 0.0, "positive cosmological age");
    double const entropyScale = energyScale * a * a *
        std::pow(
            densityScale * (lab ? a * a * a : 1.0), 1.0 - getOptions().hydro.heatCapacityRatio);
    UnitsOutputMesh mesh(DBGetPointmesh(file, (path + "darkMatter/mesh").c_str()), DBFreePointmesh);
    requireUnitsOutput(bool(mesh) && mesh->datatype == DB_DOUBLE && mesh->nels == 2,
        "dark matter mesh shape and datatype");
    requireUnitsOutputClose(mesh->dtime, epoch, "mesh time metadata");
    for (int dim = 0; dim < kDimensions; dim++) {
        requireUnitsOutput(
            mesh->units[dim] && std::string(mesh->units[dim]) == lengthUnit, "coordinate unit");
        auto const* coordinates = static_cast<double const*>(mesh->coords[dim]);
        auto const bounds = std::minmax_element(coordinates, coordinates + mesh->nels);
        double const lo = *bounds.first;
        double const hi = *bounds.second;
        requireUnitsOutputClose(lo, 0.5 * coordinateScale, "first fixed position");
        requireUnitsOutputClose(hi,
            (0.5 + (dim == kXi ? std::ldexp(1.0, -32) : 0.0)) * coordinateScale,
            "second fixed position");
        if (dim == kXi) {
            requireUnitsOutput(hi > lo, "adjacent fixed-point positions stay distinct");
        }
    }
    checkUnitsOutputGas(file, path + "gas/mesh", 1, coordinateScale, lengthUnit, epoch);
    checkUnitsOutputField(file, path + "darkMatter/xVelocity", 2, 0.0, velocityUnit);
    checkUnitsOutputField(file, path + "darkMatter/yVelocity", 2, 0.0, velocityUnit);
    checkUnitsOutputField(file, path + "darkMatter/zVelocity", 2, 0.0, velocityUnit);
    checkUnitsOutputMassExpression(file, path, 0.75 * massScale, massUnit, "darkMatter");
    checkUnitsOutputField(file, path + "gas/xVelocity", 1, 0.125 * velocityScale, velocityUnit);
    checkUnitsOutputField(file, path + "gas/yVelocity", 1, 0.25 * velocityScale, velocityUnit);
    checkUnitsOutputField(file, path + "gas/zVelocity", 1, -0.5 * velocityScale, velocityUnit);
    checkUnitsOutputMassExpression(file, path, 0.25 * massScale, massUnit);
    checkUnitsOutputField(file, path + "gas/density", 1, 8.0 * densityScale, densityUnit);
    checkUnitsOutputField(file, path + "gas/entropyTracer", 1, 4.0 * entropyScale, entropyUnit);
    checkUnitsOutputField(
        file, path + "gas/kernelSupportRadius", 1, 0.03125 * coordinateScale, lengthUnit);
    checkUnitsOutputExpression(file, path, "pressure",
        cosmological ? "(km/s)^2*(Msun/h)/(Mpc/h)^3" : "erg/cm^3", "entropyTracer");
    checkUnitsOutputExpression(file, path, "adiabaticIndex", "1", "nodal_constant");
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
            for (int dim = 0; dim < kDimensions; dim++) {
                particlesPos(dim, i).setInteger(
                    (PositionInteger(0x80000000U) + (i != 0 && dim == kXi ? 1U : 0U))
                    << (kPositionBits - 32));
                particlesVel(dim, i) = i == 2 ? (dim == kXi        ? 0.125 :
                                                        dim == kYi ? 0.25 :
                                                                     -0.5) :
                                                0.0;
            }
            particlesRung(i) = 1;
            particlesType(i) = i == 2 ? kSphType : kDarkMatterType;
            particlesCatIndex(i) = i == 2 ? 0 : kNoParticleIndex;
#if COSMICTIGER_ENABLE_STAR_FORMATION
            if (getOptions().starFormation.enabled) {
                particlesStarFormationStates[i] = {};
                particlesStarFormationStates[i].id =
                    (uint64_t{1} << 60) + 3 * uint64_t(hpxRank()) + i + 1;
            }
#endif
        }
#if COSMICTIGER_ENABLE_CHEMISTRY
        if (getOptions().chemistry.enabled) {
            sphParticlesElectronFractions[0] = {0.25f};
        }
#endif
#if COSMICTIGER_ENABLE_STAR_FORMATION
        if (getOptions().starFormation.enabled) {
            particlesStarFormationStates[2].specificStarFormationRate = 0.125;
        }
#endif
        sphParticlesParticleIndex(0) = 2;
        sphParticlesSmoothingLength(0) = 0.03125;
        sphParticlesDensity(0) = 8.0;
        sphParticlesEntropyInitialized = true;
        sphParticlesThermalScaleFactor = 0.25;
        sphParticlesEntropy(0) = 4.0;
        // Distinct from the scalar CLI alpha; unchanged by unit/frame conversion.
        sphParticlesBulkViscosityCoefficients[0] = detail::kUnitsOutputViscosityAlpha;
    } else {
        detail::requireUnitsOutput(
            particlesPos(kXi, 0).raw() == (PositionInteger(0x80000000U) << (kPositionBits - 32)) &&
                particlesPos(kXi, 1).raw() ==
                    (PositionInteger(0x80000001U) << (kPositionBits - 32)),
            "particle fixed-point storage unchanged on every locality");
    }
    for (auto& future : pending) {
        future.get();
    }
}

#if COSMICTIGER_ENABLE_STAR_FORMATION
void unitsOutputSourceFixture(int snapshot);
HPX_PLAIN_ACTION(unitsOutputSourceFixture, UnitsOutputSourceFixtureAction);
void unitsOutputSourceFixture(int snapshot) {
    Vector<hpx::future<void>> pending;
    for (auto const& child : hpxChildren()) {
        pending.push_back(hpx::async<UnitsOutputSourceFixtureAction>(child, snapshot));
    }
    auto const parameters = starFormationParameters(0.25);
    double const density = std::max(parameters.model.thresholdDensity,
        parameters.model.overdensityThreshold / (0.25 * 0.25 * 0.25));
    sphParticlesDensity(0) =
        float((10.0 + snapshot) * density * (0.25 * 0.25 * 0.25) / parameters.densityScale);
    sphParticlesSetInternalEnergy(
        0, float((snapshot + 1.0) * 1.e12 * (0.25 * 0.25) / parameters.energyScale));
    particlesStarFormationStates[2].specificStarFormationRate = 0.125 * (snapshot + 1);
    detail::requireUnitsOutput(starFormationDiagnostics(0, 0.25).starForming,
        "dense output fixture has a multiphase target");
    for (auto& future : pending) {
        future.get();
    }
}
#endif

void unitsOutputTest() {
    using namespace detail;
    auto const originalOptions = getOptions();
    bool const originalEntropyInitialized = sphParticlesEntropyInitialized;
    double const originalThermalScaleFactor = sphParticlesThermalScaleFactor;
    auto options = originalOptions;
    options.hydro.enabled = true;
#if COSMICTIGER_ENABLE_CHEMISTRY
    options.chemistry.enabled = true;
#endif
#if COSMICTIGER_ENABLE_STAR_FORMATION
    options.starFormation.enabled = true;
#endif

    options.output.silo = true;

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
    options.hydro.maximumBulkViscosityCoefficient = 1.75;
    options.gravity.smoothingLength = 0.125 *
        selectedGravityKernel().smoothingLengthPerEffectiveRadius /
        std::pow(double(options.initial.numberOfParticles), 1.0 / kDimensions);
    options.gravity.timestepEta = 0.3;
    options.hydro.minSmoothingLength = 0.2 / std::cbrt(double(options.run.gasParticleCount));
    options.hydro.maxSmoothingLength = 0.4 / std::cbrt(double(options.run.gasParticleCount));
    options.configuration = "input with spaces.ini";
    setOptions(options);
    unitsOutputParticleFixture(true);
    viewConfigure();
    viewOutputViews(0, 0.25, true, kUnitsOutputPhysicalYears);
    for (int rank = 0; rank < hpxSize(); rank++) {
        auto file = openUnitsOutput(0, rank);
        checkUnitsOutputCommandLine(file.get());
        checkUnitsOutputViscosity(file.get(), "");
        checkUnitsOutputSourceFields(file.get(), "", rank);
        UnitsOutputMesh mesh(DBGetPointmesh(file.get(), "darkMatter/mesh"), DBFreePointmesh);
        requireUnitsOutput(bool(mesh) && mesh->datatype == DB_DOUBLE && mesh->nels == 2,
            "code-unit dark matter mesh schema");
        auto const* positions = static_cast<double const*>(mesh->coords[kXi]);
        auto const bounds = std::minmax_element(positions, positions + mesh->nels);
        requireUnitsOutputClose(*bounds.first, 0.5, "first code-unit dark matter position");
        requireUnitsOutputClose(
            *bounds.second, 0.5 + std::ldexp(1.0, -32), "second code-unit dark matter position");
        requireUnitsOutput(*bounds.second > *bounds.first,
            "code-unit output preserves adjacent fixed-point positions");
        checkUnitsOutputGas(file.get(), "gas/mesh", 1, 1.0);
        checkUnitsOutputField(file.get(), "gas/xVelocity", 1, 0.125);
        checkUnitsOutputMassExpression(file.get(), "", 0.25, "code");
        requireUnitsOutput(
            !DBInqVarExists(file.get(), "gas/eint") && !DBInqVarExists(file.get(), "gas/etot"),
            "output stores entropy, not energy");
        checkUnitsOutputField(file.get(), "gas/timestepRung", 1, 1.0);
        checkUnitsOutputField(file.get(), "gas/density", 1, 8.0);
        checkUnitsOutputField(file.get(), "gas/entropyTracer", 1, 4.0);
        checkUnitsOutputField(file.get(), "gas/kernelSupportRadius", 1, 0.03125);
        checkUnitsOutputExpression(file.get(), "", "pressure", "code", "entropyTracer");
        checkUnitsOutputExpression(file.get(), "", "adiabaticIndex", "1", "nodal_constant");
        requireUnitsOutput(
            !DBInqVarExists(file.get(), "cgs/lab/gas/mesh"), "code-unit output selection");
    }
    options.output.units = "both";
    options.output.frame = "both";
    setOptions(options);
    viewOutputViews(1, 0.25, true, kUnitsOutputPhysicalYears);
    for (int rank = 0; rank < hpxSize(); rank++) {
        auto file = openUnitsOutput(1, rank);
        checkUnitsOutputCommandLine(file.get());
        for (bool cosmological : {false, true}) {
            for (bool lab : {false, true}) {
                checkConvertedUnitsOutput(file.get(), cosmological, lab);
                std::string const path = std::string(cosmological ? "cosmological/" : "cgs/") +
                    (lab ? "lab/" : "comoving/");
                checkUnitsOutputSourceFields(file.get(), path, rank);
            }
        }
        requireUnitsOutput(
            !DBInqVarExists(file.get(), "gas/mesh"), "converted selection omits code mesh");
    }
    options.output.units = "cgs";
    options.output.frame = "lab";
    setOptions(options);
    viewOutputViews(2, 0.25, true, kUnitsOutputPhysicalYears);
    for (int rank = 0; rank < hpxSize(); rank++) {
        auto file = openUnitsOutput(2, rank);
        checkUnitsOutputCommandLine(file.get());
        checkConvertedUnitsOutput(file.get(), false, true);
        requireUnitsOutput(!DBInqVarExists(file.get(), "cgs/comoving/gas/mesh") &&
                !DBInqVarExists(file.get(), "cosmological/lab/gas/mesh"),
            "CGS lab selection");
    }
    options.output.units = "cosmological";
    options.output.frame = "comoving";
    setOptions(options);
    viewOutputViews(3, 0.25, true, kUnitsOutputPhysicalYears);
    for (int rank = 0; rank < hpxSize(); rank++) {
        auto file = openUnitsOutput(3, rank);
        checkUnitsOutputCommandLine(file.get());
        checkConvertedUnitsOutput(file.get(), true, false);
        requireUnitsOutput(!DBInqVarExists(file.get(), "cosmological/lab/gas/mesh") &&
                !DBInqVarExists(file.get(), "cgs/comoving/gas/mesh"),
            "cosmological comoving selection");
    }
    unitsOutputParticleFixture(false);
#if COSMICTIGER_ENABLE_STAR_FORMATION
    options.output.units = "both";
    options.output.frame = "both";
    setOptions(options);
    for (int snapshot = 0; snapshot < 2; ++snapshot) {
        unitsOutputSourceFixture(snapshot);
        viewOutputViews(4 + snapshot, 0.25, true, kUnitsOutputPhysicalYears);
        for (int rank = 0; rank < hpxSize(); ++rank) {
            auto file = openUnitsOutput(4 + snapshot, rank);
            for (auto const* path :
                {"cgs/comoving/", "cgs/lab/", "cosmological/comoving/", "cosmological/lab/"}) {
                checkUnitsOutputSourceFields(file.get(), path, rank);
            }
        }
    }
#endif

    options.output.units = "code";
    options.output.frame = "comoving";
    options.run.scenario = "hydroWave";
    setOptions(options);
    double const elapsedCodeTime = 0.123456789;
    double const elapsedYears =
        elapsedCodeTime * options.units.timeSeconds * constants::kSecondsToYears;
    viewOutputViews(6, 1.0, true, elapsedYears);
    for (int rank = 0; rank < hpxSize(); ++rank) {
        auto file = openUnitsOutput(6, rank);
        requireUnitsOutputClose(readUnitsOutputClock(file.get(), ""), elapsedCodeTime,
            "native problem scalar time is elapsed code time");
        requireUnitsOutput(
            readUnitsOutputText(file.get(), "time_unit") == "code", "native problem time unit");
        UnitsOutputMesh mesh(DBGetPointmesh(file.get(), "gas/mesh"), DBFreePointmesh);
        requireUnitsOutput(bool(mesh), "native problem gas mesh");
        requireUnitsOutputClose(mesh->dtime, elapsedCodeTime, "native mesh double time");
        requireUnitsOutputClose(
            mesh->time, static_cast<float>(elapsedCodeTime), "native mesh float time");
        requireUnitsOutput(mesh->cycle == 6, "native mesh cycle");
        UnitsOutputField variable(DBGetPointvar(file.get(), "gas/density"), DBFreePointvar);
        requireUnitsOutput(bool(variable), "native problem gas density");
        requireUnitsOutputClose(variable->dtime, elapsedCodeTime, "native field double time");
        requireUnitsOutputClose(
            variable->time, static_cast<float>(elapsedCodeTime), "native field float time");
        requireUnitsOutput(variable->cycle == 6, "native field cycle");
    }

    // Temperature must also be available in builds/runs without chemistry.
    options.chemistry.enabled = false;
    options.starFormation.enabled = false;
    setOptions(options);
    unitsOutputParticleFixture(true);
    for (int converted = 0; converted < 2; ++converted) {
        options.output.units = converted ? "both" : "code";
        options.output.frame = converted ? "both" : "comoving";
        setOptions(options);
        viewOutputViews(7 + converted, 0.25, true, kUnitsOutputPhysicalYears);
        for (int rank = 0; rank < hpxSize(); ++rank) {
            auto file = openUnitsOutput(7 + converted, rank);
            for (auto const* path :
                {"", "cgs/comoving/", "cgs/lab/", "cosmological/comoving/", "cosmological/lab/"}) {
                if (converted == (std::strlen(path) == 0)) {
                    continue;
                }
                checkUnitsOutputExpression(file.get(), path, "temperature", "K", "pressure");
                checkUnitsOutputExpression(
                    file.get(), path, "meanMolecularWeight", "1", "nodal_constant");
            }
        }
    }

    sphParticlesEntropyInitialized = originalEntropyInitialized;
    sphParticlesThermalScaleFactor = originalThermalScaleFactor;
    setOptions(originalOptions);
    viewConfigure();
    std::cout << std::format("Boost.Units Silo output checks passed ({:d} localities)\n", hpxSize())
              << std::flush;
}
