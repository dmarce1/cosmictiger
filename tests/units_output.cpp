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
constexpr float kUnitsOutputStoredDivergence = -1.25f;
constexpr double kUnitsOutputBaseTimestep = 2.0;
constexpr double kUnitsOutputGlobalTimestep = 0.375;
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
    requireUnitsOutput(
        readUnitsOutputText(file, "/cli/gravity/method/far") == getOptions().gravity.method &&
            readUnitsOutputText(file, "/cli/gravity/method/near") ==
                getOptions().gravity.pmmmNearMethod,
        "nested gravity method options use directories");
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
    requireUnitsOutput(
        field->meshname && std::filesystem::path(field->meshname).filename() == "mesh",
        name + " attaches to the species mesh");
    auto const* values = static_cast<double const*>(field->vals[0]);
    for (int i = 0; i < field->nels; i++) {
        requireUnitsOutputClose(values[i], expected, name + " value");
    }
}

void checkUnitsOutputMassExpression(
    DBfile* file, std::string const& path, double expected, char const* unit) {
    std::string const gas = path + "gas/";
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

void checkUnitsOutputSourceFields(DBfile* file, std::string const& path, int rank) {
#if COSMICTIGER_ENABLE_CHEMISTRY
    auto const chemistry = sphChemistryOutput(sphParticlesElectronFractions[0],
        sphParticlesInternalEnergy(0), sphParticlesDensity(0), 0.25);
    char const* names[] = {"chem_HI", "chem_HII", "chem_HeI", "chem_HeII", "chem_HeIII",
        "chem_electrons", "temperature", "mean_molecular_weight", "frozen_gamma"};
    for (size_t field = 0; field < chemistry.size(); ++field) {
        checkUnitsOutputField(
            file, path + "gas/" + names[field], 1, chemistry[field], field == 6 ? "K" : "1");
    }
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
    checkUnitsOutputField(file, path + "gas/cold_fraction", 1, phase.coldFraction, "1");
    checkUnitsOutputField(file, path + "gas/hot_temperature", 1, hotTemperature, "K");
    checkUnitsOutputField(file, path + "gas/specific_sfr", 1,
        particlesStarFormationStates[2].specificStarFormationRate, "1/s");
    uint64_t const identity = (uint64_t{1} << 60) + 3 * uint64_t(rank) + 3;
    checkUnitsOutputField(file, path + "gas/id_low", 1, uint32_t(identity), "1");
    checkUnitsOutputField(file, path + "gas/id_high", 1, identity >> 32, "1");
#endif
}

void checkUnitsOutputViscosity(DBfile* file, std::string const& path) {
    std::string const name = path + "gas/alpha";
    if (getOptions().hydro.viscosityEvolution == "none") {
        requireUnitsOutput(
            !DBInqVarExists(file, name.c_str()), name + " is absent for constant viscosity");
    } else {
        checkUnitsOutputField(file, name, 1, double(kUnitsOutputViscosityAlpha), "1");
    }
    if (getOptions().hydro.viscosityGradientSwitch()) {
        // The sampled quantity is G = ∇_x·w. Its normalization is removed using
        // the sample's duration; output must not divide by the current a².
        double expected = kUnitsOutputStoredDivergence;
        char const* unit = "1/code_time";
        if (path.starts_with("cgs/")) {
            expected /= 5.0e15;
            unit = "1/s";
        } else if (path.starts_with("cosmological/")) {
            expected *= constants::kMpcToCm / (5.0e15 * 0.7 * 1.0e5);
            unit = "(km/s)/(Mpc/h)";
        }
        checkUnitsOutputField(file, path + "gas/divv", 1, expected, unit);
    }
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
        double const expected =
            (0.5 + (dim == kXDimension ? std::ldexp(1.0, -32) : 0.0)) * coordinateScale;
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
            "A = P_physical / rho_physical^gamma",
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
    double const epoch = readUnitsOutputScalar(file, path + "time");
    requireUnitsOutput(epoch > 0.0, "positive cosmological age");
    double const entropyScale = energyScale * a * a *
        std::pow(densityScale * (lab ? a * a * a : 1.0), 1.0 - getOptions().hydro.gamma);
    UnitsOutputMesh mesh(
        DBGetPointmesh(file, (path + "dark_matter/mesh").c_str()), DBFreePointmesh);
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
            (0.5 + (dim == kXDimension ? std::ldexp(1.0, -32) : 0.0)) * coordinateScale,
            "second fixed position");
        if (dim == kXDimension) {
            requireUnitsOutput(hi > lo, "adjacent fixed-point positions stay distinct");
        }
    }
    checkUnitsOutputGas(file, path + "gas/mesh", 1, coordinateScale, lengthUnit, epoch);
    checkUnitsOutputField(file, path + "dark_matter/vx", 2, 0.0, velocityUnit);
    checkUnitsOutputField(file, path + "dark_matter/vy", 2, 0.0, velocityUnit);
    checkUnitsOutputField(file, path + "dark_matter/vz", 2, 0.0, velocityUnit);
    checkUnitsOutputField(file, path + "dark_matter/mass", 2, 0.75 * massScale, massUnit);
    checkUnitsOutputField(file, path + "gas/vx", 1, 0.125 * velocityScale, velocityUnit);
    checkUnitsOutputField(file, path + "gas/vy", 1, 0.25 * velocityScale, velocityUnit);
    checkUnitsOutputField(file, path + "gas/vz", 1, -0.5 * velocityScale, velocityUnit);
    checkUnitsOutputMassExpression(file, path, 0.25 * massScale, massUnit);
    checkUnitsOutputField(file, path + "gas/rho", 1, 8.0 * densityScale, densityUnit);
    checkUnitsOutputField(file, path + "gas/A", 1, 4.0 * entropyScale, entropyUnit);
    checkUnitsOutputField(file, path + "gas/r", 1, 0.03125 * coordinateScale, lengthUnit);
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
                    (PositionInteger(0x80000000U) + (i != 0 && dim == kXDimension ? 1U : 0U))
                    << (kPositionBits - 32));
                particlesVel(dim, i) = i == 2 ? (dim == kXDimension        ? 0.125 :
                                                        dim == kYDimension ? 0.25 :
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
        if (getOptions().hydro.viscosityEvolution != "none") {
            // Distinct from the scalar CLI alpha; unchanged by unit/frame conversion.
            sphParticlesBulkViscosityCoefficients[0] = detail::kUnitsOutputViscosityAlpha;
        }
        if (getOptions().hydro.viscosityGradientSwitch()) {
            auto& history = sphScaledDivergence[0];
            // History normalization follows the live rung.

            double const sampleTimestep = usesGlobalIntegration(getOptions().run) ?
                detail::kUnitsOutputGlobalTimestep :
                std::ldexp(detail::kUnitsOutputBaseTimestep, -sphParticlesRung(0));
            detail::requireUnitsOutput(
                sphEncodeDivergence(history, detail::kUnitsOutputStoredDivergence, sampleTimestep),
                "representable canonical divergence fixture");
        }
    } else {
        detail::requireUnitsOutput(particlesPos(kXDimension, 0).raw() ==
                    (PositionInteger(0x80000000U) << (kPositionBits - 32)) &&
                particlesPos(kXDimension, 1).raw() ==
                    (PositionInteger(0x80000001U) << (kPositionBits - 32)),
            "particle fixed-point storage unchanged on every locality");
        if (getOptions().hydro.viscosityGradientSwitch()) {
            auto const& history = sphScaledDivergence[0];
            double const sampleTimestep = usesGlobalIntegration(getOptions().run) ?
                detail::kUnitsOutputGlobalTimestep :
                std::ldexp(detail::kUnitsOutputBaseTimestep, -sphParticlesRung(0));
            detail::requireUnitsOutput(sphDecodeDivergence(history, sampleTimestep) ==
                    detail::kUnitsOutputStoredDivergence,
                "output leaves the endpoint divergence unchanged");
        }
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
    options.hydro.bulkViscosityCoefficient = 1.75;
    options.gravity.smoothingLength = 0.125 *
        selectedGravityKernel().smoothingLengthPerEffectiveRadius /
        std::pow(double(options.initial.numberOfParticles), 1.0 / kDimensions);
    options.gravity.timestepEta = 0.3;
    options.hydro.minSmoothingLength = 0.2 / options.initial.particlesPerDimension;
    options.hydro.maxSmoothingLength = 0.4 / options.initial.particlesPerDimension;
    options.configuration = "input with spaces.ini";
    setOptions(options);
    unitsOutputParticleFixture(true);
    viewConfigure();
    viewOutputViews(0, 0.25, true, kUnitsOutputPhysicalYears, kUnitsOutputBaseTimestep,
        kUnitsOutputGlobalTimestep);
    for (int rank = 0; rank < hpxSize(); rank++) {
        auto file = openUnitsOutput(0, rank);
        checkUnitsOutputCommandLine(file.get());
        checkUnitsOutputViscosity(file.get(), "");
        checkUnitsOutputSourceFields(file.get(), "", rank);
        UnitsOutputMesh mesh(DBGetPointmesh(file.get(), "dark_matter/mesh"), DBFreePointmesh);
        requireUnitsOutput(bool(mesh) && mesh->datatype == DB_DOUBLE && mesh->nels == 2,
            "code-unit dark matter mesh schema");
        auto const* positions = static_cast<double const*>(mesh->coords[kXDimension]);
        auto const bounds = std::minmax_element(positions, positions + mesh->nels);
        requireUnitsOutputClose(*bounds.first, 0.5, "first code-unit dark matter position");
        requireUnitsOutputClose(
            *bounds.second, 0.5 + std::ldexp(1.0, -32), "second code-unit dark matter position");
        requireUnitsOutput(*bounds.second > *bounds.first,
            "code-unit output preserves adjacent fixed-point positions");
        checkUnitsOutputGas(file.get(), "gas/mesh", 1, 1.0);
        checkUnitsOutputField(file.get(), "gas/vx", 1, 0.125);
        checkUnitsOutputMassExpression(file.get(), "", 0.25, "code");
        requireUnitsOutput(
            !DBInqVarExists(file.get(), "gas/eint") && !DBInqVarExists(file.get(), "gas/etot"),
            "output stores entropy, not energy");
        checkUnitsOutputField(file.get(), "gas/rung", 1, 1.0);
        checkUnitsOutputField(file.get(), "gas/rho", 1, 8.0);
        checkUnitsOutputField(file.get(), "gas/A", 1, 4.0);
        checkUnitsOutputField(file.get(), "gas/r", 1, 0.03125);
        requireUnitsOutput(
            !DBInqVarExists(file.get(), "cgs/lab/gas/mesh"), "code-unit output selection");
    }
    options.output.units = "both";
    options.output.frame = "both";
    setOptions(options);
    viewOutputViews(1, 0.25, true, kUnitsOutputPhysicalYears, kUnitsOutputBaseTimestep,
        kUnitsOutputGlobalTimestep);
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
    viewOutputViews(2, 0.25, true, kUnitsOutputPhysicalYears, kUnitsOutputBaseTimestep,
        kUnitsOutputGlobalTimestep);
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
    viewOutputViews(3, 0.25, true, kUnitsOutputPhysicalYears, kUnitsOutputBaseTimestep,
        kUnitsOutputGlobalTimestep);
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
        viewOutputViews(4 + snapshot, 0.25, true, kUnitsOutputPhysicalYears,
            kUnitsOutputBaseTimestep, kUnitsOutputGlobalTimestep);
        for (int rank = 0; rank < hpxSize(); ++rank) {
            auto file = openUnitsOutput(4 + snapshot, rank);
            for (auto const* path :
                {"cgs/comoving/", "cgs/lab/", "cosmological/comoving/", "cosmological/lab/"}) {
                checkUnitsOutputSourceFields(file.get(), path, rank);
            }
        }
    }
#endif
    sphParticlesEntropyInitialized = originalEntropyInitialized;
    sphParticlesThermalScaleFactor = originalThermalScaleFactor;
    setOptions(originalOptions);
    viewConfigure();
    std::cout << std::format("Boost.Units Silo output checks passed ({:d} localities)\n", hpxSize())
              << std::flush;
}
