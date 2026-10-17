/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
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

#include <cosmictiger/constants.hpp>
#include <cosmictiger/gravity.hpp>
#include <cosmictiger/hpx.hpp>
#include <cosmictiger/options.hpp>
#include <cosmictiger/particles.hpp>
#include <cosmictiger/sph_particles.hpp>
#include <cosmictiger/view.hpp>

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
using UnitsOutputCsgMesh = std::unique_ptr<DBcsgmesh, decltype(&DBFreeCsgmesh)>;
using UnitsOutputCsgField = std::unique_ptr<DBcsgvar, decltype(&DBFreeCsgvar)>;

UnitsOutputFile openUnitsOutput(int cycle) {
    auto const filename = std::filesystem::path(getOptions().outputDirectory) /
        ("view.0." + std::to_string(cycle) + ".silo");
    UnitsOutputFile file(DBOpen(filename.c_str(), DB_UNKNOWN, DB_READ), DBClose);
    requireUnitsOutput(bool(file), "open " + filename.string());
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
        readUnitsOutputScalar(file, "cli/hydro/viscosityAlpha"), 1.75, "scalar viscosity alpha");
    requireUnitsOutputClose(
        readUnitsOutputScalar(file, "cli/gravity/softening"), 0.125, "CLI softening units");
    requireUnitsOutputClose(
        readUnitsOutputScalar(file, "cli/gravity/timestepEta"), 0.3, "CLI timestep eta");
    requireUnitsOutputClose(
        readUnitsOutputScalar(file, "cli/hydro/minSupportRadius"), 0.2, "CLI minimum support");
    requireUnitsOutputClose(
        readUnitsOutputScalar(file, "cli/hydro/maxSupportRadius"), 0.4, "CLI maximum support");
    requireUnitsOutputClose(readUnitsOutputScalar(file, "cli/units/massSolar"),
        3.0e33 / constants::kM0, "CLI solar mass units");
    requireUnitsOutput(readUnitsOutputInteger(file, "cli/hydro/enabled") == 1,
        "boolean option stored as scalar integer");
    requireUnitsOutput(readUnitsOutputInteger(file, "cli/output/siloInterval") == 1,
        "integer option stored as scalar integer");
    requireUnitsOutput(readUnitsOutputText(file, "cli/configuration") == "input with spaces.ini",
        "configuration path preserves spaces");
    requireUnitsOutput(readUnitsOutputText(file, "cli/output/units") == getOptions().outputUnits &&
            readUnitsOutputText(file, "cli/output/frame") == getOptions().outputFrame,
        "effective current output options");
    requireUnitsOutput(
        readUnitsOutputText(file, "cli/initial/gadget4Restart").empty(), "empty string option");
    int const argc = readUnitsOutputInteger(file, "cli/argc");
    requireUnitsOutput(argc > 1, "original process argument count");
    bool foundHpxThreads = false;
    bool foundTest = false;
    for (int i = 0; i < argc; i++) {
        auto const argument = readUnitsOutputText(file, "cli/argv/arg_" + std::to_string(i));
        foundHpxThreads = foundHpxThreads || argument == "--hpx:threads=2";
        foundTest = foundTest || argument == "--run.test=unitsOutput";
    }
    requireUnitsOutput(foundHpxThreads && foundTest, "original HPX and application arguments");
}

void checkUnitsOutputField(
    DBfile* file, std::string const& name, int count, double expected, char const* unit) {
    UnitsOutputField field(DBGetPointvar(file, name.c_str()), DBFreePointvar);
    requireUnitsOutput(bool(field), name + " field exists");
    requireUnitsOutput(field->datatype == DB_DOUBLE, name + " double datatype");
    requireUnitsOutput(field->nels == count * hpxSize() && field->nvals == 1, name + " shape");
    requireUnitsOutput(field->units && std::string(field->units) == unit, name + " units");
    auto const* values = static_cast<double const*>(field->vals[0]);
    for (int i = 0; i < field->nels; i++) {
        requireUnitsOutputClose(values[i], expected, name + " value");
    }
}

void checkUnitsOutputSpheres(DBfile* file, std::string const& name, int count,
    double coordinateScale, char const* lengthUnit = nullptr, double time = 0.0) {
    UnitsOutputCsgMesh mesh(DBGetCsgmesh(file, name.c_str()), DBFreeCsgmesh);
    requireUnitsOutput(bool(mesh), name + " CSG mesh exists");
    requireUnitsOutput(mesh->ndims == NDIM && mesh->datatype == DB_DOUBLE, name + " CSG shape");
    requireUnitsOutput(mesh->nbounds == count && mesh->lcoeffs == 4 * count,
        name + " one sphere boundary per particle");
    requireUnitsOutput(mesh->coeffs && mesh->typeflags, name + " sphere coefficients");
    requireUnitsOutput(mesh->zones && mesh->zones->nregs == count && mesh->zones->nzones == count,
        name + " one CSG zone per particle");
    requireUnitsOutput(mesh->zones->typeflags && mesh->zones->leftids && mesh->zones->zonelist,
        name + " CSG zone topology");
    if (lengthUnit) {
        requireUnitsOutputClose(mesh->dtime, time, name + " mesh time metadata");
        for (int dim = 0; dim < NDIM; dim++) {
            requireUnitsOutput(mesh->units[dim] && std::string(mesh->units[dim]) == lengthUnit,
                name + " coordinate unit");
        }
    }
    for (int i = 0; i < count; i++) {
        requireUnitsOutput(mesh->typeflags[i] == DBCSG_SPHERE_PR, name + " sphere boundary type");
        requireUnitsOutput(mesh->zones->typeflags[i] == DBCSG_INNER &&
                mesh->zones->leftids[i] == i && mesh->zones->zonelist[i] == i,
            name + " sphere interior zone");
        for (int component = 0; component < 4; component++) {
            int const index = 4 * i + component;
            double const value = static_cast<double const*>(mesh->coeffs)[index];
            double const expected = component == 3 ?
                0.03125 * coordinateScale :
                (0.5 + (component == XDIM ? std::ldexp(1.0, -32) : 0.0)) * coordinateScale;
            requireUnitsOutputClose(value, expected, name + " sphere center and radius");
        }
    }
}

void checkUnitsOutputCsgField(DBfile* file, std::string const& name, int count, int datatype,
    double expected, char const* unit = nullptr) {
    UnitsOutputCsgField field(DBGetCsgvar(file, name.c_str()), DBFreeCsgvar);
    requireUnitsOutput(bool(field), name + " CSG field exists");
    requireUnitsOutput(field->datatype == datatype && field->centering == DB_ZONECENT,
        name + " zone-centered datatype");
    requireUnitsOutput(field->nels == count && field->nvals == 1, name + " CSG field shape");
    requireUnitsOutput(field->vals && field->vals[0], name + " CSG values");
    if (unit) {
        requireUnitsOutput(field->units && std::string(field->units) == unit, name + " units");
    }
    for (int i = 0; i < field->nels; i++) {
        double const value = datatype == DB_FLOAT ? static_cast<float const*>(field->vals[0])[i] :
                                                    static_cast<double const*>(field->vals[0])[i];
        requireUnitsOutputClose(value, expected, name + " value");
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
    requireUnitsOutput(bool(mesh) && mesh->datatype == DB_DOUBLE && mesh->nels == 2 * hpxSize(),
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
    checkUnitsOutputSpheres(file, path + "gas", hpxSize(), coordinateScale, lengthUnit, epoch);
    checkUnitsOutputField(file, path + "vx", 2, 0.0, velocityUnit);
    checkUnitsOutputField(file, path + "vy", 2, 0.0, velocityUnit);
    checkUnitsOutputField(file, path + "vz", 2, 0.0, velocityUnit);
    checkUnitsOutputField(file, path + "mass", 2, 0.75 * massScale, massUnit);
    checkUnitsOutputCsgField(
        file, path + "hydro_vx", hpxSize(), DB_DOUBLE, 0.125 * velocityScale, velocityUnit);
    checkUnitsOutputCsgField(
        file, path + "hydro_vy", hpxSize(), DB_DOUBLE, 0.25 * velocityScale, velocityUnit);
    checkUnitsOutputCsgField(
        file, path + "hydro_vz", hpxSize(), DB_DOUBLE, -0.5 * velocityScale, velocityUnit);
    checkUnitsOutputCsgField(
        file, path + "hydro_mass", hpxSize(), DB_DOUBLE, 0.25 * massScale, massUnit);
    checkUnitsOutputCsgField(
        file, path + "rho", hpxSize(), DB_DOUBLE, 8.0 * densityScale, densityUnit);
    checkUnitsOutputCsgField(
        file, path + "eint", hpxSize(), DB_DOUBLE, 4.0 * energyScale, energyUnit);
    checkUnitsOutputCsgField(file, path + "kernel_support_radius", hpxSize(), DB_DOUBLE,
        0.03125 * coordinateScale, lengthUnit);
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
        sphParticlesIsstar(0) = false;
        sphParticlesSmoothingLength(0) = 0.03125;
        sphParticlesRho(0) = 8.0;
        sphParticlesAlpha(0) = 1.0;
#ifdef ENTROPY
        sphParticlesR2[0].A =
            4.0 * (getOptions().gamma - 1.0) / std::pow(8.0, getOptions().gamma - 1.0);
#else
        sphParticlesR2[0].eint = 4.0;
#endif
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
    options.sph = true;
    options.stars = false;
    options.chem = false;
    options.outputSilo = true;
    options.outputSiloInterval = 1;
    options.outputDirectory = "units_output_fixture";
    options.outputXMin = options.outputYMin = options.outputZMin = 0.0;
    options.outputXMax = options.outputYMax = options.outputZMax = 1.0;
    options.codeToCm = 2.0e24;
    options.codeToG = 3.0e33;
    options.codeToS = 5.0e15;
    options.hubble = 0.7;
    options.darkMatterMass = 0.75;
    options.sphMass = 0.25;
    options.outputUnits = "code";
    options.outputFrame = "comoving";
    options.alpha0 = 1.75;
    options.gravitySupportRadius = 0.125 * selectedGravityKernel().supportRadiusPerEffectiveRadius /
        std::pow(double(options.numberOfParticles), 1.0 / NDIM);
    options.eta = 0.3;
    options.hydroMinSupportRadius = 0.2 / options.particlesPerDimension;
    options.hydroMaxSupportRadius = 0.4 / options.particlesPerDimension;
    options.configFile = "input with spaces.ini";
    setOptions(options);
    unitsOutputParticleFixture(true);
    viewConfigure();
    viewOutputViews(0, 0.25, true);
    {
        auto file = openUnitsOutput(0);
        checkUnitsOutputCommandLine(file.get());
        requireUnitsOutput(
            !DBInqVarExists(file.get(), "alpha"), "code-unit output omits particle alpha");
        UnitsOutputMesh mesh(DBGetPointmesh(file.get(), "dark_matter"), DBFreePointmesh);
        requireUnitsOutput(bool(mesh) && mesh->datatype == DB_FLOAT && mesh->nels == 2 * hpxSize(),
            "code-unit dark matter mesh schema");
        auto const* positions = static_cast<float const*>(mesh->coords[XDIM]);
        for (int i = 0; i < mesh->nels; i++) {
            requireUnitsOutput(positions[i] == 0.5f, "code-unit dark matter float position");
        }
        checkUnitsOutputSpheres(file.get(), "gas", hpxSize(), 1.0);
        checkUnitsOutputCsgField(file.get(), "hydro_vx", hpxSize(), DB_FLOAT, 0.125);
        checkUnitsOutputCsgField(file.get(), "hydro_rung", hpxSize(), DB_FLOAT, 1.0);
        checkUnitsOutputCsgField(file.get(), "rho", hpxSize(), DB_FLOAT, 8.0);
        checkUnitsOutputCsgField(file.get(), "eint", hpxSize(), DB_FLOAT, 4.0);
        checkUnitsOutputCsgField(file.get(), "kernel_support_radius", hpxSize(), DB_FLOAT, 0.03125);
        requireUnitsOutput(
            !DBInqVarExists(file.get(), "cgs/lab/gas"), "code-unit output selection");
    }
    options.outputUnits = "both";
    options.outputFrame = "both";
    setOptions(options);
    viewOutputViews(1, 0.25, true);
    {
        auto file = openUnitsOutput(1);
        checkUnitsOutputCommandLine(file.get());
        for (bool cosmological : {false, true}) {
            for (bool lab : {false, true}) {
                checkConvertedUnitsOutput(file.get(), cosmological, lab);
            }
        }
        requireUnitsOutput(
            !DBInqVarExists(file.get(), "gas"), "converted selection omits code mesh");
    }
    options.outputUnits = "cgs";
    options.outputFrame = "lab";
    setOptions(options);
    viewOutputViews(2, 0.25, true);
    {
        auto file = openUnitsOutput(2);
        checkUnitsOutputCommandLine(file.get());
        checkConvertedUnitsOutput(file.get(), false, true);
        requireUnitsOutput(!DBInqVarExists(file.get(), "cgs/comoving/gas") &&
                !DBInqVarExists(file.get(), "cosmological/lab/gas"),
            "CGS lab selection");
    }
    options.outputUnits = "cosmological";
    options.outputFrame = "comoving";
    setOptions(options);
    viewOutputViews(3, 0.25, true);
    {
        auto file = openUnitsOutput(3);
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
