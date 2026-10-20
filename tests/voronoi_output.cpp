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
#include <cosmictiger/io/silo_voronoi.hpp>
#include <cosmictiger/io/view.hpp>
#include <cosmictiger/math/constants.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

namespace detail {

constexpr int kVoronoiFixtureCount = 64;

void requireVoronoiOutput(bool condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error("Voronoi output: " + message);
    }
}

void nearVoronoiOutput(double actual, double expected, std::string const& message) {
    constexpr uint64_t kExponentMask = uint64_t(0x7ff) << 52;
    requireVoronoiOutput((std::bit_cast<uint64_t>(actual) & kExponentMask) != kExponentMask &&
            std::abs(actual - expected) <= 5e-11 * std::max(std::abs(expected), 1e-100),
        message);
}

std::vector<int> voronoiFixtureIndices(int rank, int activeRanks) {
    std::vector<int> indices;
    for (int global = 0; global < kVoronoiFixtureCount; global++) {
        if (global % activeRanks == rank) {
            indices.push_back(global);
        }
    }
    return indices;
}

std::array<double, 3> voronoiFixturePosition(int global) {
    return {0.125 + 0.25 * (global / 16), 0.125 + 0.25 * ((global / 4) % 4),
        0.125 + 0.25 * (global % 4)};
}

using VoronoiOutputFile = std::unique_ptr<DBfile, decltype(&DBClose)>;
using VoronoiOutputMultimesh = std::unique_ptr<DBmultimesh, decltype(&DBFreeMultimesh)>;
using VoronoiOutputMultivar = std::unique_ptr<DBmultivar, decltype(&DBFreeMultivar)>;
using VoronoiOutputMesh = std::unique_ptr<DBucdmesh, decltype(&DBFreeUcdmesh)>;
using VoronoiOutputVariable = std::unique_ptr<DBucdvar, decltype(&DBFreeUcdvar)>;

std::filesystem::path voronoiMasterPath(int cycle) {
    return std::filesystem::path(getOptions().output.directory) /
        ("view.0." + std::to_string(cycle) + ".silo");
}

double voronoiNumericValue(DBucdvar const* variable, int index) {
    requireVoronoiOutput(
        variable && index >= 0 && index < variable->nels && variable->vals && variable->vals[0],
        "read valid scalar zone field");
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
        throw std::runtime_error("Voronoi output: unexpected zone datatype");
    }
}

uint64_t voronoiGeneratorId(DBucdvar const* variable, int index) {
    if (variable->datatype == DB_LONG_LONG) {
        return uint64_t(static_cast<long long const*>(variable->vals[0])[index]);
    }
    requireVoronoiOutput(variable->datatype == DB_LONG && sizeof(long) == sizeof(uint64_t),
        "generator identities retain 64-bit integer precision");
    return uint64_t(static_cast<long const*>(variable->vals[0])[index]);
}

struct VoronoiOutputScales {
    double length = 1.0;
    double density = 1.0;
    double velocity = 1.0;
    double energy = 1.0;
    double mass = 1.0;
    bool converted = false;
};

VoronoiOutputScales voronoiOutputScales(bool cosmological, bool lab) {
    double const a = 0.25;
    double const hubble = 0.7;
    double const length = 2.0e24;
    double const mass = 3.0e33;
    double const time = 5.0e15;
    VoronoiOutputScales scales;
    scales.converted = true;
    scales.length = length * (lab ? a : 1.0) * (cosmological ? hubble / constants::kMpcToCm : 1.0);
    scales.velocity = length / time / a * (cosmological ? 1e-5 : 1.0);
    scales.energy = length * length / (time * time * a * a) * (cosmological ? 1e-10 : 1.0);
    scales.mass = mass * (cosmological ? hubble / constants::kM0 : 1.0);
    scales.density = mass / (length * length * length) / (lab ? a * a * a : 1.0) *
        (cosmological ? std::pow(constants::kMpcToCm, 3) / (constants::kM0 * hubble * hubble) :
                        1.0);
    return scales;
}

void checkVoronoiOutput(int cycle, std::string const& prefix, int activeRanks, bool cropped,
    VoronoiOutputScales const& scales = {}) {
    auto const masterPath = voronoiMasterPath(cycle);
    VoronoiOutputFile master(DBOpen(masterPath.c_str(), DB_UNKNOWN, DB_READ), DBClose);
    requireVoronoiOutput(bool(master), "open master " + masterPath.string());
    int const generationLength = DBGetVarLength(master.get(), "leaf_generation");
    requireVoronoiOutput(generationLength > 1, "master records immutable leaf generation");
    std::vector<char> generation(generationLength);
    requireVoronoiOutput(DBReadVar(master.get(), "leaf_generation", generation.data()) >= 0 &&
            generation.back() == '\0',
        "read immutable leaf generation");
    VoronoiOutputMultimesh multimesh(
        DBGetMultimesh(master.get(), (prefix + "gas").c_str()), DBFreeMultimesh);
    requireVoronoiOutput(
        multimesh && multimesh->nblocks == hpxSize(), "gas multimesh has one block per locality");
    std::vector<std::string> fields = {"rho", "eint", "hydro_vx", "hydro_vy", "hydro_vz",
        "hydro_rung", "hydro_mass", "kernel_support_radius"};
    for (char const* name : voronoi::kOutputDiagnosticNames) {
        fields.emplace_back(name);
    }
    std::vector<VoronoiOutputMultivar> multivars;
    for (auto const& field : fields) {
        multivars.emplace_back(
            DBGetMultivar(master.get(), (prefix + field).c_str()), DBFreeMultivar);
        requireVoronoiOutput(multivars.back() && multivars.back()->nvars == hpxSize(),
            "multivar exists for " + prefix + field);
    }
    double totalVolume = 0.0;
    double totalMass = 0.0;
    std::set<int> representedGenerators;
    for (int rank = 0; rank < hpxSize(); rank++) {
        auto const localGlobals = voronoiFixtureIndices(rank, activeRanks);
        auto const leafPath = std::filesystem::path(masterPath.string() + ".data") /
            generation.data() / ("locality_" + std::to_string(rank) + ".silo");
        requireVoronoiOutput(std::filesystem::exists(leafPath), "one Silo file per locality");
        std::string const directory = "/locality_" + std::to_string(rank) + "/" + prefix;
        std::string const meshReference(multimesh->meshnames[rank]);
        bool const empty = std::none_of(localGlobals.begin(), localGlobals.end(),
            [cropped](int global) { return !cropped || global < 32; });
        if (empty) {
            requireVoronoiOutput(meshReference == "EMPTY", "empty gas locality marked EMPTY");
            for (auto const& variable : multivars) {
                requireVoronoiOutput(std::string(variable->varnames[rank]) == "EMPTY",
                    "empty mesh and variable entries match");
            }
            continue;
        }
        auto const colon = meshReference.find(':');
        requireVoronoiOutput(colon != std::string::npos, "multimesh references external leaf");
        auto const referencedFile =
            (masterPath.parent_path() / meshReference.substr(0, colon)).lexically_normal();
        requireVoronoiOutput(referencedFile == leafPath.lexically_normal(),
            "mesh reference names correct locality file");
        requireVoronoiOutput(meshReference.substr(colon + 1) == directory + "gas",
            "mesh reference names locality and unit subdirectories");
        requireVoronoiOutput(multimesh->meshtypes[rank] == DB_UCDMESH, "gas block is native UCD");
        VoronoiOutputFile leaf(DBOpen(leafPath.c_str(), DB_UNKNOWN, DB_READ), DBClose);
        requireVoronoiOutput(bool(leaf) && DBSetDir(leaf.get(), directory.c_str()) >= 0,
            "open locality output subdirectory");
        VoronoiOutputMesh mesh(DBGetUcdmesh(leaf.get(), "gas"), DBFreeUcdmesh);
        requireVoronoiOutput(mesh && mesh->datatype == DB_DOUBLE && mesh->ndims == 3 &&
                mesh->phzones && mesh->phzones->nzones > 0,
            "native 3-D polyhedral gas mesh");
        int const zones = mesh->phzones->nzones;
        requireVoronoiOutput(mesh->phzones->lo_offset == 0 && mesh->phzones->hi_offset == zones - 1,
            "all output fragments are real zones");
        for (int dim = 0; dim < 3; dim++) {
            double const lower = cropped && dim == 0 ? 0.2 : 0.0;
            double const upper = cropped && dim == 0 ? 0.3 : 1.0;
            auto const* coordinates = static_cast<double const*>(mesh->coords[dim]);
            for (int node = 0; node < mesh->nnodes; node++) {
                double const coordinate = coordinates[node] / scales.length;
                requireVoronoiOutput(coordinate >= lower - 2e-13 && coordinate <= upper + 2e-13,
                    "all nodes respect configured view bounds and unit scale");
            }
        }
        std::map<std::string, VoronoiOutputVariable> variables;
        for (size_t field = 0; field < fields.size(); field++) {
            auto const& name = fields[field];
            std::string const expectedReference =
                meshReference.substr(0, colon + 1) + directory + name;
            requireVoronoiOutput(multivars[field]->vartypes[rank] == DB_UCDVAR &&
                    std::string(multivars[field]->varnames[rank]) == expectedReference,
                "multivar refers to matching leaf field " + name);
            VoronoiOutputVariable variable(DBGetUcdvar(leaf.get(), name.c_str()), DBFreeUcdvar);
            requireVoronoiOutput(variable && variable->nels == zones && variable->nvals == 1 &&
                    variable->centering == DB_ZONECENT,
                "field follows fragment count: " + name);
            variables.emplace(name, std::move(variable));
        }
        requireVoronoiOutput(variables.at("rho")->datatype == DB_DOUBLE,
            "Voronoi gas scalar output uses double precision");
        for (int zone = 0; zone < zones; zone++) {
            auto const value = [&](char const* name) {
                return voronoiNumericValue(variables.at(name).get(), zone);
            };
            uint64_t const generatorId =
                voronoiGeneratorId(variables.at("generator_id").get(), zone);
            size_t const localIndex = generatorId & 0xffffffffULL;
            requireVoronoiOutput(
                generatorId >> 32 == uint64_t(rank) && localIndex < localGlobals.size(),
                "fragment remains with its generator's locality");
            int const global = localGlobals[localIndex];
            representedGenerators.insert(global);
            nearVoronoiOutput(value("rho"), (1000.0 + global) * scales.density,
                "SPH density remains distinct from mass/cell volume");
            nearVoronoiOutput(
                value("eint"), (10.0 + global) * scales.energy, "internal energy association");
            nearVoronoiOutput(value("hydro_vx"), (0.125 + global / 256.0) * scales.velocity,
                "velocity parent association");
            nearVoronoiOutput(value("hydro_vy"), -0.25 * scales.velocity, "y velocity conversion");
            nearVoronoiOutput(value("hydro_vz"), 0.5 * scales.velocity, "z velocity conversion");
            nearVoronoiOutput(value("hydro_rung"), global % 5, "rung parent association");
            nearVoronoiOutput(value("kernel_support_radius"),
                (0.03125 + global / 4096.0) * scales.length, "SPH support radius remains a field");
            double const volumeScale = scales.length * scales.length * scales.length;
            double const fullVolume = value("cell_volume") / volumeScale;
            double const fragmentVolume = value("fragment_volume") / volumeScale;
            nearVoronoiOutput(fullVolume, 1.0 / 64.0, "physical Voronoi cell volume");
            double const mass = value("hydro_mass") / scales.mass;
            nearVoronoiOutput(mass, 0.25 * fragmentVolume / fullVolume,
                "extensive mass apportioned to clipped fragments");
            if (cropped) {
                double const generatorX = voronoiFixturePosition(global)[0];
                requireVoronoiOutput(generatorX < 0.2 || generatorX > 0.3,
                    "crop includes cells whose generators lie outside view");
            }
            totalVolume += fragmentVolume;
            totalMass += mass;
        }
    }
    nearVoronoiOutput(
        totalVolume, cropped ? 0.1 : 1.0, "distributed fragments cover requested view");
    nearVoronoiOutput(
        totalMass, (cropped ? 0.1 : 1.0) * 16.0, "distributed fragment mass conservation");
    requireVoronoiOutput(representedGenerators.size() == (cropped ? 32 : 64),
        "all intersecting physical cells represented exactly by parent identity");

    if (!cropped) {
        VoronoiOutputMultimesh darkMatter(
            DBGetMultimesh(master.get(), (prefix + "dark_matter").c_str()), DBFreeMultimesh);
        requireVoronoiOutput(
            darkMatter && darkMatter->nblocks == hpxSize(), "dark matter multimesh");
        for (int rank = 0; rank < hpxSize(); rank++) {
            if (rank >= activeRanks) {
                requireVoronoiOutput(std::string(darkMatter->meshnames[rank]) == "EMPTY",
                    "empty dark-matter locality marked EMPTY");
            } else {
                requireVoronoiOutput(darkMatter->meshtypes[rank] == DB_POINTMESH,
                    "dark matter remains a point mesh");
            }
        }
    }
}

}    // namespace detail

void voronoiOutputParticleFixture(int activeRanks, bool verify);
HPX_PLAIN_ACTION(voronoiOutputParticleFixture, VoronoiOutputParticleFixtureAction);

void voronoiOutputParticleFixture(int activeRanks, bool verify) {
    Vector<hpx::future<void>> pending;
    for (auto const& child : hpxChildren()) {
        pending.push_back(
            hpx::async<VoronoiOutputParticleFixtureAction>(child, activeRanks, verify));
    }
    auto const globals = detail::voronoiFixtureIndices(hpxRank(), activeRanks);
    int const gasCount = int(globals.size());
    int const darkCount = hpxRank() < activeRanks ? 2 : 0;
    if (!verify) {
        particlesResize(gasCount + darkCount);
        sphParticlesResize(gasCount, false);
    }
    for (int i = 0; i < gasCount; i++) {
        int const global = globals[i];
        auto const position = detail::voronoiFixturePosition(global);
        if (verify) {
            for (int dim = 0; dim < 3; dim++) {
                detail::requireVoronoiOutput(particlesPos(dim, i).toDouble() == position[dim],
                    "output leaves physical generator coordinates unchanged");
            }
            detail::requireVoronoiOutput(sphParticlesDensity(i) == 1000.0f + global &&
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

void voronoiOutputTest() {
    auto const originalOptions = getOptions();
    auto options = originalOptions;
    options.compute.cuda = false;
    options.hydro.enabled = true;
    options.chemistry.enabled = false;
    options.output.silo = true;
    options.output.gasMesh = "voronoi";
    options.output.siloInterval = 1;
    options.output.directory = "voronoi_output_fixture/run_" +
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
    voronoiOutputParticleFixture(hpxSize(), false);
    viewConfigure();
    viewOutputViews(0, 0.25);
    detail::checkVoronoiOutput(0, "", hpxSize(), false);

    options.output.xMin = 0.2;
    options.output.xMax = 0.3;
    setOptions(options);
    viewConfigure();
    viewOutputViews(1, 0.25);
    detail::checkVoronoiOutput(1, "", hpxSize(), true);

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
            detail::checkVoronoiOutput(
                2, prefix, hpxSize(), false, detail::voronoiOutputScales(cosmological, lab));
        }
    }
    voronoiOutputParticleFixture(hpxSize(), true);

    options.output.units = "code";
    options.output.frame = "comoving";
    options.output.siloInterval = 3;
    setOptions(options);
    viewConfigure();
    auto const skipped = detail::voronoiMasterPath(4);
    detail::requireVoronoiOutput(
        !std::filesystem::exists(skipped), "cadence probe starts unpublished");
    viewOutputViews(4, 0.25);
    detail::requireVoronoiOutput(
        !std::filesystem::exists(skipped), "cadence skips nonoutput cycle");
    viewOutputViews(4, 0.25, true);
    detail::checkVoronoiOutput(4, "", hpxSize(), false);

    int const activeRanks = std::max(1, hpxSize() - 1);
    voronoiOutputParticleFixture(activeRanks, false);
    viewOutputViews(6, 0.25);
    detail::checkVoronoiOutput(6, "", activeRanks, false);
    voronoiOutputParticleFixture(activeRanks, true);
    setOptions(originalOptions);
    viewConfigure();
    PRINT("Distributed Voronoi Silo output checks passed (%i localities)\n", hpxSize());
}
