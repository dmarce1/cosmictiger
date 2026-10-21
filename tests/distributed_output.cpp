/*
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt.
 */

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <silo.h>

#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/io/view.hpp>
#include <cosmictiger/math/constants.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

namespace detail {

constexpr int kDistributedFixtureCount = 64;

void requireDistributedOutput(bool condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error("Distributed output: " + message);
    }
}

void nearDistributedOutput(double actual, double expected, std::string const& message) {
    constexpr uint64_t kExponentMask = uint64_t(0x7ff) << 52;
    requireDistributedOutput((std::bit_cast<uint64_t>(actual) & kExponentMask) != kExponentMask &&
            std::abs(actual - expected) <= 5e-11 * std::max(std::abs(expected), 1e-100),
        message);
}

std::vector<int> distributedFixtureIndices(int rank, int activeRanks) {
    std::vector<int> indices;
    for (int global = 0; global < kDistributedFixtureCount; global++) {
        if (global % activeRanks == rank) {
            indices.push_back(global);
        }
    }
    return indices;
}

std::array<double, 3> distributedFixturePosition(int global) {
    return {0.125 + 0.25 * (global / 16), 0.125 + 0.25 * ((global / 4) % 4),
        0.125 + 0.25 * (global % 4)};
}

using DistributedOutputFile = std::unique_ptr<DBfile, decltype(&DBClose)>;
using DistributedOutputMultimesh = std::unique_ptr<DBmultimesh, decltype(&DBFreeMultimesh)>;
using DistributedOutputMultivar = std::unique_ptr<DBmultivar, decltype(&DBFreeMultivar)>;
using DistributedOutputMesh = std::unique_ptr<DBpointmesh, decltype(&DBFreePointmesh)>;
using DistributedOutputVariable = std::unique_ptr<DBpointvar, decltype(&DBFreePointvar)>;

std::filesystem::path distributedMasterPath(int cycle) {
    return std::filesystem::path(getOptions().output.directory) /
        ("view.0." + std::to_string(cycle) + ".silo");
}

double distributedNumericValue(DBpointvar const* variable, int index) {
    requireDistributedOutput(
        variable && index >= 0 && index < variable->nels && variable->vals && variable->vals[0],
        "read valid scalar particle field");
    switch (variable->datatype) {
    case DB_FLOAT:
        return static_cast<float const*>(variable->vals[0])[index];
    case DB_DOUBLE:
        return static_cast<double const*>(variable->vals[0])[index];
    case DB_INT:
        return static_cast<int const*>(variable->vals[0])[index];
    case DB_LONG:
        return static_cast<long const*>(variable->vals[0])[index];
    case DB_LONG_LONG:
        return static_cast<long long const*>(variable->vals[0])[index];
    default:
        throw std::runtime_error("Distributed output: unexpected particle datatype");
    }
}

struct DistributedOutputScales {
    double length = 1.0;
    double density = 1.0;
    double velocity = 1.0;
    double energy = 1.0;
    double mass = 1.0;
};

DistributedOutputScales distributedOutputScales(bool cosmological, bool lab) {
    double const a = 0.25;
    double const hubble = 0.7;
    double const length = 2.0e24;
    double const mass = 3.0e33;
    double const time = 5.0e15;
    DistributedOutputScales scales;
    scales.length = length * (lab ? a : 1.0) * (cosmological ? hubble / constants::kMpcToCm : 1.0);
    scales.velocity = length / time / a * (cosmological ? 1e-5 : 1.0);
    scales.energy = length * length / (time * time * a * a) * (cosmological ? 1e-10 : 1.0);
    scales.mass = mass * (cosmological ? hubble / constants::kM0 : 1.0);
    scales.density = mass / (length * length * length) / (lab ? a * a * a : 1.0) *
        (cosmological ? std::pow(constants::kMpcToCm, 3) / (constants::kM0 * hubble * hubble) :
                        1.0);
    return scales;
}

std::string checkDistributedOutput(int cycle, std::string const& prefix, int activeRanks,
    bool cropped, DistributedOutputScales const& scales = {}, bool segregated = false) {
    auto const masterPath = distributedMasterPath(cycle);
    DistributedOutputFile master(DBOpen(masterPath.c_str(), DB_UNKNOWN, DB_READ), DBClose);
    requireDistributedOutput(bool(master), "open master " + masterPath.string());
    int const generationLength = DBGetVarLength(master.get(), "leaf_generation");
    requireDistributedOutput(generationLength > 1, "master records immutable leaf generation");
    std::vector<char> generation(generationLength);
    requireDistributedOutput(DBReadVar(master.get(), "leaf_generation", generation.data()) >= 0 &&
            generation.back() == '\0',
        "read immutable leaf generation");
    DistributedOutputMultimesh multimesh(
        DBGetMultimesh(master.get(), (prefix + "gas").c_str()), DBFreeMultimesh);
    requireDistributedOutput(
        multimesh && multimesh->nblocks == hpxSize(), "gas multimesh has one block per locality");
    std::unique_ptr<int, decltype(&std::free)> masterCycle(
        static_cast<int*>(DBGetComponent(master.get(), (prefix + "gas").c_str(), "cycle")),
        &std::free);
    std::unique_ptr<double, decltype(&std::free)> masterTime(
        static_cast<double*>(DBGetComponent(master.get(), (prefix + "gas").c_str(), "dtime")),
        &std::free);
    requireDistributedOutput(
        masterCycle && *masterCycle == cycle && masterTime && *masterTime > 0.0,
        "master carries cycle and time");
    std::vector<std::string> const fields = {"rho", "eint", "hydro_vx", "hydro_vy", "hydro_vz",
        "hydro_rung", "hydro_mass", "kernel_support_radius"};
    std::vector<DistributedOutputMultivar> multivars;
    for (auto const& field : fields) {
        multivars.emplace_back(
            DBGetMultivar(master.get(), (prefix + field).c_str()), DBFreeMultivar);
        requireDistributedOutput(multivars.back() && multivars.back()->nvars == hpxSize(),
            "multivar exists for " + prefix + field);
        requireDistributedOutput(multivars.back()->mmesh_name &&
                std::filesystem::path(multivars.back()->mmesh_name).filename() == "gas",
            "gas multivar attaches to the gas multimesh: " + field);
    }
    double totalMass = 0.0;
    std::set<int> representedParticles;
    for (int rank = 0; rank < hpxSize(); rank++) {
        auto localGlobals = distributedFixtureIndices(rank, activeRanks);
        if (cropped) {
            std::erase_if(localGlobals, [](int global) { return global < 16 || global >= 32; });
        }
        auto const leafPath = std::filesystem::path(masterPath.string() + ".data") /
            generation.data() / ("locality_" + std::to_string(rank) + ".silo");
        requireDistributedOutput(std::filesystem::exists(leafPath), "one Silo file per locality");
        std::string const directory = "/locality_" + std::to_string(rank) + "/" + prefix;
        std::string const meshReference(multimesh->meshnames[rank]);
        if (localGlobals.empty()) {
            requireDistributedOutput(meshReference == "EMPTY", "empty gas locality marked EMPTY");
            for (auto const& variable : multivars) {
                requireDistributedOutput(std::string(variable->varnames[rank]) == "EMPTY",
                    "empty mesh and variable entries match");
            }
            continue;
        }
        auto const colon = meshReference.find(':');
        requireDistributedOutput(colon != std::string::npos, "multimesh references external leaf");
        auto const referencedFile =
            (masterPath.parent_path() / meshReference.substr(0, colon)).lexically_normal();
        requireDistributedOutput(referencedFile == leafPath.lexically_normal(),
            "mesh reference names correct locality file");
        requireDistributedOutput(meshReference.substr(colon + 1) == directory + "gas",
            "mesh reference names locality and unit subdirectories");
        requireDistributedOutput(
            multimesh->meshtypes[rank] == DB_POINTMESH, "gas block is a particle point mesh");
        DistributedOutputFile leaf(DBOpen(leafPath.c_str(), DB_UNKNOWN, DB_READ), DBClose);
        requireDistributedOutput(bool(leaf) && DBSetDir(leaf.get(), directory.c_str()) >= 0,
            "open locality output subdirectory");
        DistributedOutputMesh mesh(DBGetPointmesh(leaf.get(), "gas"), DBFreePointmesh);
        int const count = int(localGlobals.size());
        requireDistributedOutput(mesh && mesh->datatype == DB_DOUBLE && mesh->ndims == 3 &&
                mesh->nels == count && mesh->coords[0] && mesh->coords[1] && mesh->coords[2],
            "one point for each local selected particle");
        requireDistributedOutput(mesh->cycle == cycle, "leaf mesh cycle metadata");
        nearDistributedOutput(mesh->dtime, *masterTime, "master and leaf time agree");
        std::map<std::string, DistributedOutputVariable> variables;
        for (size_t field = 0; field < fields.size(); field++) {
            auto const& name = fields[field];
            std::string const expectedReference =
                meshReference.substr(0, colon + 1) + directory + name;
            requireDistributedOutput(multivars[field]->vartypes[rank] == DB_POINTVAR &&
                    std::string(multivars[field]->varnames[rank]) == expectedReference,
                "multivar refers to matching leaf field " + name);
            DistributedOutputVariable variable(
                DBGetPointvar(leaf.get(), name.c_str()), DBFreePointvar);
            requireDistributedOutput(variable && variable->nels == count && variable->nvals == 1 &&
                    variable->datatype == DB_DOUBLE && variable->meshname &&
                    std::filesystem::path(variable->meshname).filename() == "gas",
                "field attaches to gas particles: " + name);
            variables.emplace(name, std::move(variable));
        }
        for (int point = 0; point < count; point++) {
            auto const value = [&](char const* name) {
                return distributedNumericValue(variables.at(name).get(), point);
            };
            // Recover the fixture identity from the stored density rather than assume
            // collection order; each owned particle must appear exactly once.
            int const global = int(std::llround(value("rho") / scales.density)) - 1000;
            requireDistributedOutput(
                std::find(localGlobals.begin(), localGlobals.end(), global) != localGlobals.end(),
                "particle remains with its owning locality and inside the selected crop");
            requireDistributedOutput(
                representedParticles.insert(global).second, "each selected particle appears once");
            auto const position = distributedFixturePosition(global);
            for (int dim = 0; dim < 3; dim++) {
                auto const* coordinates = static_cast<double const*>(mesh->coords[dim]);
                nearDistributedOutput(coordinates[point], position[dim] * scales.length,
                    "point position follows the particle");
            }
            nearDistributedOutput(
                value("rho"), (1000.0 + global) * scales.density, "SPH density association");
            nearDistributedOutput(
                value("eint"), (10.0 + global) * scales.energy, "internal energy association");
            nearDistributedOutput(value("hydro_vx"), (0.125 + global / 256.0) * scales.velocity,
                "velocity association");
            nearDistributedOutput(
                value("hydro_vy"), -0.25 * scales.velocity, "y velocity conversion");
            nearDistributedOutput(
                value("hydro_vz"), 0.5 * scales.velocity, "z velocity conversion");
            nearDistributedOutput(value("hydro_rung"), global % 5, "rung association");
            double const supportRadius = (0.03125 + global / 4096.0) * scales.length;
            nearDistributedOutput(value("kernel_support_radius"), supportRadius,
                "SPH support radius remains a field");
            double const mass = value("hydro_mass") / scales.mass;
            nearDistributedOutput(mass, 0.25, "each output particle retains its full mass");
            totalMass += mass;
        }
    }
    nearDistributedOutput(totalMass, cropped ? 4.0 : 16.0, "selected particle mass conservation");
    requireDistributedOutput(representedParticles.size() == (cropped ? 16 : 64),
        "all selected particles represented exactly once");
    if (cropped) {
        requireDistributedOutput(!DBInqVarExists(master.get(), (prefix + "dark_matter").c_str()),
            "species outside crop is omitted from master");
        for (char const* name : {"vx", "vy", "vz", "rung"}) {
            requireDistributedOutput(!DBInqVarExists(master.get(), (prefix + name).c_str()),
                "cropped-out dark matter contributes no multivars");
        }
    } else {
        DistributedOutputMultimesh darkMatter(
            DBGetMultimesh(master.get(), (prefix + "dark_matter").c_str()), DBFreeMultimesh);
        requireDistributedOutput(
            darkMatter && darkMatter->nblocks == hpxSize(), "dark matter multimesh");
        std::vector<std::string> darkFields = {"vx", "vy", "vz", "rung"};
        if (!prefix.empty()) {
            darkFields.push_back("mass");
        }
        for (int rank = 0; rank < hpxSize(); rank++) {
            bool const empty = segregated ? rank == 0 : rank >= activeRanks;
            std::string const meshReference(darkMatter->meshnames[rank]);
            auto const colon = meshReference.find(':');
            auto const leafPath = std::filesystem::path(masterPath.string() + ".data") /
                generation.data() / ("locality_" + std::to_string(rank) + ".silo");
            std::string const directory = "/locality_" + std::to_string(rank) + "/" + prefix;
            if (empty) {
                requireDistributedOutput(
                    meshReference == "EMPTY", "empty dark-matter locality marked EMPTY");
            } else {
                requireDistributedOutput(darkMatter->meshtypes[rank] == DB_POINTMESH &&
                        colon != std::string::npos &&
                        meshReference.substr(colon + 1) == directory + "dark_matter",
                    "dark matter references its own point mesh");
            }
            DistributedOutputFile leaf(DBOpen(leafPath.c_str(), DB_UNKNOWN, DB_READ), DBClose);
            requireDistributedOutput(leaf && DBSetDir(leaf.get(), directory.c_str()) >= 0,
                "open dark matter locality directory");
            for (auto const& name : darkFields) {
                DistributedOutputMultivar variable(
                    DBGetMultivar(master.get(), (prefix + name).c_str()), DBFreeMultivar);
                requireDistributedOutput(variable && variable->nvars == hpxSize() &&
                        variable->mmesh_name &&
                        std::filesystem::path(variable->mmesh_name).filename() == "dark_matter",
                    "dark matter field attaches to its own multimesh: " + name);
                if (empty) {
                    requireDistributedOutput(std::string(variable->varnames[rank]) == "EMPTY" &&
                            !DBInqVarExists(leaf.get(), name.c_str()),
                        "missing dark matter has no local fields");
                } else {
                    std::string const expectedReference =
                        meshReference.substr(0, colon + 1) + directory + name;
                    requireDistributedOutput(variable->vartypes[rank] == DB_POINTVAR &&
                            std::string(variable->varnames[rank]) == expectedReference,
                        "dark matter multivar refers to its locality field");
                    DistributedOutputVariable field(
                        DBGetPointvar(leaf.get(), name.c_str()), DBFreePointvar);
                    requireDistributedOutput(field && field->nels == 2 && field->meshname &&
                            std::filesystem::path(field->meshname).filename() == "dark_matter",
                        "dark matter leaf field attaches to its own particles");
                    for (int point = 0; point < field->nels; point++) {
                        double const expected = name == "mass" ? 0.75 * scales.mass : 0.0;
                        nearDistributedOutput(distributedNumericValue(field.get(), point), expected,
                            "dark matter fields do not contain gas values");
                    }
                }
            }
            if (segregated && rank > 0) {
                for (auto const& name : fields) {
                    requireDistributedOutput(!DBInqVarExists(leaf.get(), name.c_str()),
                        "dark-matter-only locality has no gas fields");
                }
            }
        }
    }
    return generation.data();
}

}    // namespace detail

void distributedOutputParticleFixture(int activeRanks, bool verify, bool segregated);
HPX_PLAIN_ACTION(distributedOutputParticleFixture, DistributedOutputParticleFixtureAction);

void distributedOutputParticleFixture(int activeRanks, bool verify, bool segregated) {
    Vector<hpx::future<void>> pending;
    for (auto const& child : hpxChildren()) {
        pending.push_back(hpx::async<DistributedOutputParticleFixtureAction>(
            child, activeRanks, verify, segregated));
    }
    auto const globals = detail::distributedFixtureIndices(hpxRank(), activeRanks);
    int const gasCount = int(globals.size());
    int const darkCount = (segregated ? hpxRank() > 0 : hpxRank() < activeRanks) ? 2 : 0;
    if (!verify) {
        particlesResize(gasCount + darkCount);
        sphParticlesResize(gasCount, false);
    }
    for (int i = 0; i < gasCount; i++) {
        int const global = globals[i];
        auto const position = detail::distributedFixturePosition(global);
        if (verify) {
            for (int dim = 0; dim < 3; dim++) {
                detail::requireDistributedOutput(particlesPos(dim, i).toDouble() == position[dim],
                    "output leaves physical particle coordinates unchanged");
            }
            detail::requireDistributedOutput(sphParticlesDensity(i) == 1000.0f + global &&
                    sphParticlesInternalEnergy(i) == 10.0f + global,
                "output leaves gas state unchanged");
            continue;
        }
        for (int dim = 0; dim < 3; dim++) {
            particlesPos(dim, i) = position[dim];
            particlesVel(dim, i) = dim == 0 ? 0.125 + global / 256.0 : dim == 1 ? -0.25 : 0.5;
        }
        particlesRung(i) = global % 5;
        particlesType(i) = SPH_TYPE;
        particlesCatIndex(i) = i;
        sphParticlesParticleIndex(i) = i;
        sphParticlesSupportRadius(i) = 0.03125 + global / 4096.0;
        sphParticlesDensity(i) = 1000.0f + global;
        sphParticlesThermodynamicStates[i].internalEnergy = 10.0f + global;
    }
    if (!verify) {
        for (int i = 0; i < darkCount; i++) {
            int const particle = gasCount + i;
            for (int dim = 0; dim < 3; dim++) {
                particlesPos(dim, particle) = 0.5 + i * 0.125;
                particlesVel(dim, particle) = 0.0;
            }
            particlesRung(particle) = 0;
            particlesType(particle) = DARK_MATTER_TYPE;
            particlesCatIndex(particle) = NO_INDEX;
        }
    }
    for (auto& future : pending) {
        future.get();
    }
}

void distributedOutputTest() {
    auto const originalOptions = getOptions();
    auto options = originalOptions;
    options.compute.cuda = false;
    options.hydro.enabled = true;
    options.chemistry.enabled = false;
    options.output.silo = true;
    options.output.siloInterval = 1;
    options.output.directory = "distributed_output_fixture/run_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    options.output.units = "code";
    options.output.frame = "comoving";
    options.output.xMin = options.output.yMin = options.output.zMin = 0.0;
    options.output.xMax = options.output.yMax = options.output.zMax = 1.0;
    options.units.lengthCm = 2.0e24;
    options.units.massG = 3.0e33;
    options.units.timeSeconds = 5.0e15;
    options.cosmology.hubble = 0.7;
    options.initial.darkMatterMass = 0.75;
    options.initial.gasMass = 0.25;
    setOptions(options);
    distributedOutputParticleFixture(hpxSize(), false, false);
    viewConfigure();
    viewOutputViews(0, 0.25);
    auto const initialGeneration = detail::checkDistributedOutput(0, "", hpxSize(), false);
    viewOutputViews(0, 0.25);
    auto const replacementGeneration = detail::checkDistributedOutput(0, "", hpxSize(), false);
    detail::requireDistributedOutput(initialGeneration != replacementGeneration,
        "republishing a cycle uses a new immutable generation");
    for (int rank = 0; rank < hpxSize(); rank++) {
        auto const previousLeaf =
            std::filesystem::path(detail::distributedMasterPath(0).string() + ".data") /
            initialGeneration / ("locality_" + std::to_string(rank) + ".silo");
        detail::requireDistributedOutput(std::filesystem::exists(previousLeaf),
            "republishing retains previously published leaves");
    }

    options.output.xMin = 0.2;
    options.output.xMax = 0.45;
    setOptions(options);
    viewConfigure();
    viewOutputViews(1, 0.25);
    detail::checkDistributedOutput(1, "", hpxSize(), true);

    options.output.xMin = 0.0;
    options.output.xMax = 1.0;
    options.output.units = "both";
    options.output.frame = "both";
    setOptions(options);
    viewConfigure();
    viewOutputViews(2, 0.25);
    for (bool const cosmological : {false, true}) {
        for (bool const lab : {false, true}) {
            std::string const prefix =
                std::string(cosmological ? "cosmological/" : "cgs/") + (lab ? "lab/" : "comoving/");
            detail::checkDistributedOutput(
                2, prefix, hpxSize(), false, detail::distributedOutputScales(cosmological, lab));
        }
    }
    distributedOutputParticleFixture(hpxSize(), true, false);

    options.output.units = "code";
    options.output.frame = "comoving";
    options.output.siloInterval = 3;
    setOptions(options);
    viewConfigure();
    auto const skipped = detail::distributedMasterPath(4);
    detail::requireDistributedOutput(
        !std::filesystem::exists(skipped), "cadence probe starts unpublished");
    viewOutputViews(4, 0.25);
    detail::requireDistributedOutput(
        !std::filesystem::exists(skipped), "cadence skips nonoutput cycle");
    viewOutputViews(4, 0.25, true);
    detail::checkDistributedOutput(4, "", hpxSize(), false);

    int const activeRanks = std::max(1, hpxSize() - 1);
    distributedOutputParticleFixture(activeRanks, false, false);
    viewOutputViews(6, 0.25);
    detail::checkDistributedOutput(6, "", activeRanks, false);
    distributedOutputParticleFixture(activeRanks, true, false);
    options.output.xMin = 0.2;
    options.output.xMax = 0.3;
    setOptions(options);
    viewConfigure();
    viewOutputViews(9, 0.25);
    {
        auto const masterPath = detail::distributedMasterPath(9);
        detail::DistributedOutputFile master(
            DBOpen(masterPath.c_str(), DB_UNKNOWN, DB_READ), DBClose);
        detail::requireDistributedOutput(master && !DBInqVarExists(master.get(), "gas") &&
                !DBInqVarExists(master.get(), "dark_matter"),
            "view containing no particle centers publishes an empty master");
    }
    if (hpxSize() > 1) {
        options.output.xMin = 0.0;
        options.output.xMax = 1.0;
        setOptions(options);
        viewConfigure();
        distributedOutputParticleFixture(1, false, true);
        viewOutputViews(12, 0.25);
        detail::checkDistributedOutput(12, "", 1, false, {}, true);
        distributedOutputParticleFixture(1, true, true);
    }
    setOptions(originalOptions);
    viewConfigure();
    PRINT("Distributed particle Silo output checks passed (%i localities)\n", hpxSize());
}
