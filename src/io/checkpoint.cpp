/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

#if COSMICTIGER_WITH_GRAVITY
#include <cosmictiger/analysis/lightcone.hpp>
#endif
#include <cosmictiger/core/driver.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/terminal.hpp>
#include <cosmictiger/io/checkpoint.hpp>
#include <cosmictiger/io/output.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/particles/particles.hpp>
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/sph_chemistry.hpp>
#include <cosmictiger/sph/sph_particles.hpp>
#endif

HPX_PLAIN_ACTION(writeCheckpoint, WriteCheckpointAction);
HPX_PLAIN_ACTION(readCheckpoint, ReadCheckpointAction);

void validateCheckpointOutput() {
    auto const& options = getOptions();
    if (hpxRank() != 0 || !options.run.test.empty() || options.checkpoint.readIndex >= 0) {
        return;
    }
    std::filesystem::path const directory(options.output.directory);
    if (!std::filesystem::exists(directory)) {
        return;
    }
    for (auto const& entry : std::filesystem::directory_iterator(directory)) {
        auto const name = entry.path().filename().string();
        std::string_view index(name);
        if (!index.starts_with("checkpoint.")) {
            continue;
        }
        index.remove_prefix(sizeof("checkpoint.") - 1);
        if (index.ends_with(".incomplete")) {
            index.remove_suffix(sizeof(".incomplete") - 1);
        }
        if (index.empty() || index.find_first_not_of("0123456789") != std::string_view::npos) {
            continue;
        }
        throw std::runtime_error("Cannot start a fresh run: output.directory contains an existing "
                                 "checkpoint path " +
            entry.path().string() +
            ". Choose a new --output.directory, or use --checkpoint.readIndex to restart from "
            "a completed checkpoint. Preserve or move the previous run before reusing its "
            "output directory.");
    }
}

Vector<double> readCheckpointList() {
    Vector<double> chkpts;
    FILE* fp = fopen("checkpoint.txt", "rt");
    if (fp != nullptr) {
        float z;
        PRINT("Checkpoints will be written at z = \n");
        while (fscanf(fp, "%f\n", &z) == 1) {
            PRINT("%f\n", z);
            chkpts.push_back(z);
        }

        fclose(fp);
    }
    return chkpts;
}

namespace detail {

void checkpointRedshiftAlias(int iteration, double redshift) {
    char aliasName[128];
    snprintf(aliasName, sizeof(aliasName), "checkpoint.z.%.1f", redshift);
    auto const aliasPath = outputPath(aliasName);
    std::error_code error;
    auto const status = std::filesystem::symlink_status(aliasPath, error);
    if (!error && status.type() != std::filesystem::file_type::not_found) {
        PRINT("Redshift checkpoint alias %s already exists; restart from checkpoint.%i\n",
            aliasName, iteration);
        return;
    }
    error.clear();
    std::filesystem::create_directory_symlink(
        "checkpoint." + std::to_string(iteration), aliasPath, error);
    if (error) {
        PRINT("Unable to create redshift checkpoint alias %s; restart from checkpoint.%i\n",
            aliasName, iteration);
    }
}

}    // namespace detail

namespace detail {

constexpr uint64_t kCheckpointMagic = 0x4354535048303031ULL;
constexpr uint32_t kLegacyDarkMatterCheckpointVersion = 1;
constexpr uint32_t kCheckpointVersion = 3;
constexpr uint32_t kCheckpointManifestVersion = 1;
constexpr uint32_t kScenarioManifestVersion = 2;
constexpr uint32_t kMonaghanSph = 1;
constexpr uint32_t kCompactSphLayout = 1U << 24;
constexpr uint32_t kZPlaneAcceleration = 1U << 12;
constexpr uint32_t kBalsaraSwitch = 1U << 13;
constexpr uint32_t kVariableViscosity = 1U << 14;
constexpr uint32_t kFixed64Positions = 1U << 15;
// Retired PM checkpoints are recognized only to reject them before reading payloads.
constexpr uint32_t kRetiredParticleMesh = 1U << 16;
constexpr uint32_t kParticleMultipoleMesh = 1U << 17;
constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

namespace fs = std::filesystem;

struct CheckpointFileInfo {
    uint64_t size;
    uint64_t checksum;
};

struct CheckpointFileCloser {
    void operator()(FILE* file) const {
        if (file != nullptr) {
            fclose(file);
        }
    }
};

using CheckpointFile = std::unique_ptr<FILE, CheckpointFileCloser>;

fs::path checkpointDirectory(int iteration) {
    return outputPath("checkpoint." + std::to_string(iteration));
}

fs::path checkpointStagingDirectory(int iteration) {
    return outputPath("checkpoint." + std::to_string(iteration) + ".incomplete");
}

fs::path checkpointRankFile(fs::path const& directory, int rank) {
    return directory / ("checkpoint." + std::to_string(rank) + ".dat");
}

void syncDirectory(fs::path const& directory) {
    int const fd = open(directory.c_str(), O_RDONLY | O_DIRECTORY);
    if (fd < 0) {
        THROW_ERROR("Unable to open checkpoint directory %s for sync\n", directory.c_str());
    }
    int const result = fsync(fd);
    int const closeResult = close(fd);
    if (result != 0 || closeResult != 0) {
        THROW_ERROR("Unable to sync checkpoint directory %s\n", directory.c_str());
    }
}

void flushAndCloseCheckpointFile(CheckpointFile& file, fs::path const& path) {
    if (ferror(file.get()) != 0 || fflush(file.get()) != 0 || fsync(fileno(file.get())) != 0) {
        THROW_ERROR("Unable to flush checkpoint file %s\n", path.c_str());
    }
    FILE* rawFile = file.release();
    if (fclose(rawFile) != 0) {
        THROW_ERROR("Unable to close checkpoint file %s\n", path.c_str());
    }
}

CheckpointFileInfo checkpointFileInfo(fs::path const& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        THROW_ERROR("Unable to inspect checkpoint file %s\n", path.c_str());
    }
    std::array<char, 65536> buffer{};
    CheckpointFileInfo info{0, kFnvOffsetBasis};
    while (file) {
        file.read(buffer.data(), buffer.size());
        auto const bytes = file.gcount();
        info.size += static_cast<uint64_t>(bytes);
        for (std::streamsize i = 0; i < bytes; i++) {
            info.checksum ^= static_cast<unsigned char>(buffer[i]);
            info.checksum *= kFnvPrime;
        }
    }
    if (!file.eof()) {
        THROW_ERROR("Unable to read checkpoint file %s\n", path.c_str());
    }
    return info;
}

uint32_t checkpointFileVersion(fs::path const& path) {
    std::ifstream file(path, std::ios::binary);
    uint64_t magic = 0;
    uint32_t version = 0;
    if (!file.read(reinterpret_cast<char*>(&magic), sizeof(magic)) ||
        !file.read(reinterpret_cast<char*>(&version), sizeof(version)) ||
        magic != kCheckpointMagic) {
        return 0;
    }
    return version;
}

void writeCheckpointScenario(fs::path const& directory) {
    auto const& options = getOptions();
    if (!options.run.test.empty()) {
        return;
    }
    auto const path = directory / "scenario.meta";
    CheckpointFile file(fopen(path.c_str(), "wb"));
    if (!file ||
        fprintf(file.get(), "COSMICTIGER_SCENARIO 1\n%s\n%.17g\n%d\n", options.run.scenario.c_str(),
            options.run.endTime, int(options.gravity.enabled)) < 0) {
        THROW_ERROR("Unable to write checkpoint scenario metadata %s\n", path.c_str());
    }
    flushAndCloseCheckpointFile(file, path);
}

void validateCheckpointScenario(fs::path const& directory) {
    if (!getOptions().run.test.empty()) {
        return;
    }
    auto const path = directory / "scenario.meta";
    std::ifstream file(path);
    if (!file) {
        THROW_ERROR("Checkpoint %s has no scenario metadata; use its original executable to "
                    "resume a checkpoint written before first-class scenarios\n",
            directory.c_str());
    }
    std::string label;
    std::string scenario;
    int version = 0;
    int gravity = -1;
    double endTime = -1.0;
    if (!(file >> label >> version >> scenario >> endTime >> gravity) ||
        label != "COSMICTIGER_SCENARIO" || version != 1 || scenario != getOptions().run.scenario ||
        endTime != getOptions().run.endTime || gravity != int(getOptions().gravity.enabled)) {
        THROW_ERROR("Checkpoint scenario, run.endTime or gravity.enabled differs from the "
                    "requested run: %s\n",
            path.c_str());
    }
    file >> std::ws;
    if (file.peek() != std::char_traits<char>::eof()) {
        THROW_ERROR("Checkpoint scenario metadata %s has unexpected trailing data\n", path.c_str());
    }
}

void writeCheckpointManifest(fs::path const& directory, int iteration) {
    fs::path const path = directory / "checkpoint.meta";
    CheckpointFile file(fopen(path.c_str(), "wb"));
    if (!file) {
        THROW_ERROR("Unable to create checkpoint manifest %s\n", path.c_str());
    }
    auto const manifestVersion =
        getOptions().run.test.empty() ? kScenarioManifestVersion : kCheckpointManifestVersion;
    if (fprintf(file.get(), "COSMICTIGER_CHECKPOINT %u %u %i %i\n", manifestVersion,
            kCheckpointVersion, iteration, hpxSize()) < 0) {
        THROW_ERROR("Unable to write checkpoint manifest %s\n", path.c_str());
    }
    for (int rank = 0; rank < hpxSize(); rank++) {
        auto const rankPath = checkpointRankFile(directory, rank);
        if (checkpointFileVersion(rankPath) != kCheckpointVersion) {
            THROW_ERROR("Checkpoint rank %i has an unexpected format version\n", rank);
        }
        auto const info = checkpointFileInfo(rankPath);
        if (fprintf(file.get(), "%i %llu %016llx\n", rank,
                static_cast<unsigned long long>(info.size),
                static_cast<unsigned long long>(info.checksum)) < 0) {
            THROW_ERROR("Unable to write checkpoint manifest %s\n", path.c_str());
        }
    }
    if (manifestVersion == kScenarioManifestVersion) {
        auto const info = checkpointFileInfo(directory / "scenario.meta");
        if (fprintf(file.get(), "SCENARIO %llu %016llx\n",
                static_cast<unsigned long long>(info.size),
                static_cast<unsigned long long>(info.checksum)) < 0) {
            THROW_ERROR("Unable to write scenario checksum to %s\n", path.c_str());
        }
    }
    flushAndCloseCheckpointFile(file, path);
}

void validateCheckpointDirectory(fs::path const& directory, int iteration) {
    if (!fs::is_directory(directory)) {
        THROW_ERROR("Checkpoint directory %s does not exist\n", directory.c_str());
    }
    validateCheckpointScenario(directory);
    fs::path const manifestPath = directory / "checkpoint.meta";
    if (!fs::exists(manifestPath)) {
        // Checkpoints written before format 3 have no manifest or integrity hashes.
        for (int rank = 0; rank < hpxSize(); rank++) {
            auto const rankPath = checkpointRankFile(directory, rank);
            if (!fs::is_regular_file(rankPath)) {
                THROW_ERROR("Legacy checkpoint is missing rank file %s\n", rankPath.c_str());
            }
            if (checkpointFileVersion(rankPath) == kCheckpointVersion) {
                THROW_ERROR("Format 3 checkpoint %s is missing its manifest\n", directory.c_str());
            }
        }
        PRINT("Reading legacy checkpoint without integrity manifest\n");
        return;
    }
    std::ifstream manifest(manifestPath);
    std::string label;
    uint32_t manifestVersion = 0;
    uint32_t payloadVersion = 0;
    int savedIteration = -1;
    int rankCount = 0;
    if (!(manifest >> label >> manifestVersion >> payloadVersion >> savedIteration >> rankCount) ||
        label != "COSMICTIGER_CHECKPOINT" ||
        (manifestVersion != kCheckpointManifestVersion &&
            manifestVersion != kScenarioManifestVersion) ||
        (getOptions().run.test.empty() && manifestVersion != kScenarioManifestVersion) ||
        payloadVersion != kCheckpointVersion || savedIteration != iteration ||
        rankCount != hpxSize()) {
        THROW_ERROR("Checkpoint manifest %s has an incompatible version, iteration or locality "
                    "count\n",
            manifestPath.c_str());
    }
    for (int rank = 0; rank < rankCount; rank++) {
        int savedRank = -1;
        uint64_t savedSize = 0;
        uint64_t savedChecksum = 0;
        if (!(manifest >> std::dec >> savedRank >> savedSize >> std::hex >> savedChecksum) ||
            savedRank != rank) {
            THROW_ERROR("Checkpoint manifest %s has an invalid rank entry\n", manifestPath.c_str());
        }
        auto const rankPath = checkpointRankFile(directory, rank);
        if (checkpointFileVersion(rankPath) != kCheckpointVersion) {
            THROW_ERROR(
                "Checkpoint rank file %s has an incompatible format version\n", rankPath.c_str());
        }
        auto const actual = checkpointFileInfo(rankPath);
        if (actual.size != savedSize || actual.checksum != savedChecksum) {
            THROW_ERROR(
                "Checkpoint rank file %s failed size or checksum validation\n", rankPath.c_str());
        }
    }
    if (manifestVersion == kScenarioManifestVersion) {
        std::string scenarioLabel;
        uint64_t savedSize = 0;
        uint64_t savedChecksum = 0;
        if (!(manifest >> scenarioLabel >> std::dec >> savedSize >> std::hex >> savedChecksum) ||
            scenarioLabel != "SCENARIO") {
            THROW_ERROR(
                "Checkpoint manifest %s has an invalid scenario entry\n", manifestPath.c_str());
        }
        auto const actual = checkpointFileInfo(directory / "scenario.meta");
        if (actual.size != savedSize || actual.checksum != savedChecksum) {
            THROW_ERROR("Checkpoint scenario metadata failed size or checksum validation\n");
        }
    }
    manifest >> std::ws;
    if (manifest.peek() != std::char_traits<char>::eof()) {
        THROW_ERROR("Checkpoint manifest %s has unexpected trailing data\n", manifestPath.c_str());
    }
}

struct CheckpointHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t flags;
    double darkMatterMass;
    double gasMass;
    double gamma;
    double softening;
    double gravitationalMass;
    double omegaMatter;
    double omegaRadiation;
    double omegaLambda;
    double unitLength;
    double unitTime;
    double unitMass;
    uint64_t particlesPerSpecies;
    int32_t nominalSteps;
    uint32_t reserved;
};

// The pre-SPH main branch wrote this native layout directly.
struct LegacyEnergies {
    double pot;
    double kin;
    double tckin;
    double xmom;
    double ymom;
    double cosmic;
    double zmom;
    double nmom;
};

struct LegacyDriver {
    double a;
    double adot;
    double dummy;
    double tau;
    double tauMax;
    double energy0;
    LegacyEnergies energies;
    int maxRung;
    int iter;
    int minRung0;
    int step;
    int bucketSize;
    size_t totalProcessed;
    double flops;
    double runtime;
    double years;
    TimeType itime;
};

uint32_t checkpointFlags(Options const& opts) {
    constexpr uint32_t kSph = 1U << 0;
    constexpr uint32_t kHierarchical = 1U << 6;
    constexpr uint32_t kLightcone = 1U << 7;
    constexpr uint32_t kGroups = 1U << 8;
    constexpr uint32_t kTracers = 1U << 9;
    constexpr uint32_t kChemistry = 1U << 11;
    // Leave retired option bits unused so old incompatible headers remain rejected.
    return (opts.hydro.enabled ? kSph : 0U) | (opts.run.hierarchicalGravity ? kHierarchical : 0U) |
        (opts.lightcone.enabled ? kLightcone : 0U) | (opts.analysis.groups ? kGroups : 0U) |
        (opts.analysis.tracers ? kTracers : 0U) | (opts.chemistry.enabled ? kChemistry : 0U) |
        (opts.hydro.enabled && opts.hydro.zPlaneAcceleration != 0.0 ? kZPlaneAcceleration : 0U) |
        (opts.hydro.enabled && opts.hydro.balsaraSwitch ? kBalsaraSwitch : 0U) |
        (opts.hydro.enabled && opts.hydro.viscosityEvolution != "none" ? kVariableViscosity : 0U) |
        (opts.compute.positionType == "fixed64" ? kFixed64Positions : 0U) |
        (opts.gravity.method == "pmmm" ? kParticleMultipoleMesh : 0U);
}

uint32_t checkpointKernelCode(std::string const& family, int degree) {
    uint32_t familyCode = 0;
    if (family == "spline") {
        familyCode = 1;
    } else if (family == "wendland") {
        familyCode = 2;
    } else if (family == "ferrers") {
        familyCode = 3;
    } else {
        THROW_ERROR("Unknown checkpoint kernel family: %s\n", family.c_str());
    }
    if (degree < 0 || degree > 15) {
        THROW_ERROR("Checkpoint kernel degree does not fit its selection field\n");
    }
    return (familyCode << 4) | static_cast<uint32_t>(degree);
}

uint32_t checkpointPhysicsCode(Options const& opts) {
    // Preserve the existing low-byte formulation and native header layout.
    // Historical default kernels encode zero in both new selection bytes.
    uint32_t const gravityCode =
        checkpointKernelCode(opts.gravity.kernelFamily, opts.gravity.kernelDegree) ^
        checkpointKernelCode("ferrers", 8);
    uint32_t const hydroCode =
        checkpointKernelCode(opts.hydro.kernelFamily, opts.hydro.kernelDegree) ^
        checkpointKernelCode("spline", 3);
    return (opts.hydro.enabled ? kMonaghanSph | kCompactSphLayout : 0U) | (gravityCode << 8) |
        (hydroCode << 16);
}

bool legacyCheckpointCompatible(Options const& opts) {
    return opts.gravity.method == "fmm" && opts.compute.positionType == "fixed32" &&
        !opts.hydro.enabled && opts.run.hierarchicalGravity &&
        opts.gravity.kernelFamily == "ferrers" && opts.gravity.kernelDegree == 8;
}

CheckpointHeader checkpointHeader(Options const& opts) {
    return {kCheckpointMagic, kCheckpointVersion, checkpointFlags(opts),
        opts.initial.darkMatterMass, opts.initial.gasMass, opts.hydro.gamma,
        opts.gravity.supportRadius, opts.gravity.gm, opts.cosmology.omegaM, opts.cosmology.omegaR,
        opts.cosmology.omegaLambda, opts.units.lengthCm, opts.units.timeSeconds, opts.units.massG,
        opts.initial.numberOfParticles, opts.run.numberOfSteps, checkpointPhysicsCode(opts)};
}

bool checkpointCompatible(CheckpointHeader const& header, Options const& opts) {
    bool const supportedVersion = header.version == kCheckpointVersion ||
        (opts.gravity.method == "fmm" && !opts.hydro.enabled &&
            opts.compute.positionType == "fixed32" &&
            header.version == kLegacyDarkMatterCheckpointVersion);
    return supportedVersion && (header.flags & kRetiredParticleMesh) == 0U &&
        header.flags == checkpointFlags(opts) && header.reserved == checkpointPhysicsCode(opts) &&
        header.darkMatterMass == opts.initial.darkMatterMass &&
        header.gasMass == opts.initial.gasMass && header.gamma == opts.hydro.gamma &&
        header.softening == opts.gravity.supportRadius &&
        header.gravitationalMass == opts.gravity.gm &&
        header.omegaMatter == opts.cosmology.omegaM &&
        header.omegaRadiation == opts.cosmology.omegaR &&
        header.omegaLambda == opts.cosmology.omegaLambda &&
        header.unitLength == opts.units.lengthCm && header.unitTime == opts.units.timeSeconds &&
        header.unitMass == opts.units.massG &&
        header.particlesPerSpecies == opts.initial.numberOfParticles &&
        header.nominalSteps == opts.run.numberOfSteps;
}

// Stored only for Monaghan SPH checkpoints; the dark-matter layout is unchanged.
std::array<double, 13> sphCheckpointParameters(Options const& opts) {
    return {opts.hydro.viscosityAlpha, opts.hydro.viscosityBeta, opts.hydro.viscosityEpsilon,
        opts.hydro.neighborNumber, opts.hydro.minSupportRadius, opts.hydro.maxSupportRadius,
        opts.hydro.cfl, opts.gravity.timestepEta, opts.hydro.yAcceleration,
        opts.hydro.centralAcceleration, opts.hydro.centralSupportRadius, opts.hydro.damping,
        opts.gravity.enabled ? 1.0 : 0.0};
}

std::array<double, 7> viscosityCheckpointParameters(Options const& opts) {
    double const model = opts.hydro.viscosityEvolution == "cullenAndDehnen" ?
        1.0 :
        (opts.hydro.viscosityEvolution == "priceEtAl" ? 2.0 : 0.0);
    return {opts.hydro.viscosityAlphaMin, opts.hydro.viscosityDecay, model,
        opts.hydro.viscosityAlphaMax, opts.hydro.viscosityDecayCD,
        opts.hydro.viscosityAlphaMaxPrice, opts.hydro.viscosityDecayPrice};
}

bool zPlaneCheckpointCompatible(double acceleration, Options const& opts) {
    // Inspect the exponent bits because this translation unit permits fast-math.
    uint64_t bits;
    std::memcpy(&bits, &acceleration, sizeof(bits));
    return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL &&
        acceleration == opts.hydro.zPlaneAcceleration;
}

std::array<uint64_t, 11> pmmmCheckpointParameters(Options const& opts) {
    uint64_t timestepEtaBits;
    std::memcpy(&timestepEtaBits, &opts.gravity.timestepEta, sizeof(timestepEtaBits));
    uint64_t openingAngleBits;
    std::memcpy(&openingAngleBits, &opts.gravity.openingAngle, sizeof(openingAngleBits));
    // Force convention 1: periodic spherical far field including its isotropic
    // trace, plus compact-softened primary interactions in the neighbor cube.
    // Runtime FFT plans and decompositions are reconstructed after restart.
    // Version two appends the near solver. Version one always used direct PP.
    // Version three records the PMMM order and geometry-derived opening angle
    // so a restart preserves the shared near/far order and accuracy settings.
    // Version four appends the maximum FFT prime factor. Earlier versions used
    // unrestricted mesh sizes, represented by zero. Preserve the requested base
    // mesh as well as the cap: both determine all future hierarchical meshes.
    return {4, static_cast<uint64_t>(opts.gravity.meshSize),
        static_cast<uint64_t>(opts.gravity.fmmOrder),
        static_cast<uint64_t>(opts.gravity.pmmmNearWidth), static_cast<uint64_t>(opts.run.minRung),
        timestepEtaBits, 1, opts.gravity.pmmmNearMethod == "fmm" ? 1U : 0U,
        static_cast<uint64_t>(opts.gravity.pmmmOrder), openingAngleBits,
        static_cast<uint64_t>(opts.gravity.pmmmMaxPrimeFactor)};
}

std::array<uint64_t, 11> readPmmmCheckpointParameters(FILE* fp) {
    std::array<uint64_t, 11> parameters{};
    FREAD(parameters.data(), sizeof(uint64_t), 7, fp);
    if (parameters[0] == 2) {
        FREAD(&parameters[7], sizeof(uint64_t), 1, fp);
    } else if (parameters[0] == 3) {
        FREAD(&parameters[7], sizeof(uint64_t), 3, fp);
    } else if (parameters[0] == 4) {
        FREAD(&parameters[7], sizeof(uint64_t), 4, fp);
    } else if (parameters[0] != 1) {
        THROW_ERROR("Unsupported PMMM checkpoint parameter version\n");
    }
    return parameters;
}

bool pmmmCheckpointCompatible(std::array<uint64_t, 11> parameters, Options const& opts) {
    auto const expected = pmmmCheckpointParameters(opts);
    if (parameters[0] == 1 || parameters[0] == 2) {
        if (parameters[0] == 1) {
            parameters[7] = 0;
        }
        // Historical checkpoints used one order for both solvers and did not
        // record theta. Their original order must still match both new orders.
        parameters[8] = parameters[2];
        parameters[9] = expected[9];
        parameters[0] = 3;
    }
    if (parameters[0] == 3) {
        parameters[10] = 0;
        parameters[0] = 4;
    }
    return parameters == expected;
}

void writeDriverState(FILE* fp, DriverParameters const& params) {
    auto const header = checkpointHeader(getOptions());
    if (fwrite(&header, sizeof(header), 1, fp) != 1) {
        THROW_ERROR("Unable to write checkpoint header\n");
    }
    if ((header.flags & kParticleMultipoleMesh) != 0U) {
        auto const meshParameters = pmmmCheckpointParameters(getOptions());
        if (fwrite(meshParameters.data(), sizeof(uint64_t), meshParameters.size(), fp) !=
            meshParameters.size()) {
            THROW_ERROR("Unable to write PMMM checkpoint parameters\n");
        }
    }
    if (getOptions().hydro.enabled) {
        auto const sphParameters = sphCheckpointParameters(getOptions());
        if (fwrite(sphParameters.data(), sizeof(double), sphParameters.size(), fp) !=
            sphParameters.size()) {
            THROW_ERROR("Unable to write SPH checkpoint parameters\n");
        }
        // Optional extension: preserve the historical block and zero-force layout.
        if ((header.flags & kZPlaneAcceleration) != 0U &&
            fwrite(&getOptions().hydro.zPlaneAcceleration, sizeof(double), 1, fp) != 1) {
            THROW_ERROR("Unable to write SPH z-plane acceleration\n");
        }
        if ((header.flags & kVariableViscosity) != 0U) {
            auto const viscosityParameters = viscosityCheckpointParameters(getOptions());
            if (fwrite(viscosityParameters.data(), sizeof(double), viscosityParameters.size(),
                    fp) != viscosityParameters.size()) {
                THROW_ERROR("Unable to write SPH variable-viscosity parameters\n");
            }
        }
    }
#if COSMICTIGER_WITH_SPH
    if (getOptions().chemistry.enabled) {
        auto const chemicalParameters = sphChemistryCheckpointParameters();
        uint64_t const radiationChecksum = sphChemistryRadiationChecksum();
        if (fwrite(chemicalParameters.data(), sizeof(double), chemicalParameters.size(), fp) !=
                chemicalParameters.size() ||
            fwrite(&radiationChecksum, sizeof(radiationChecksum), 1, fp) != 1) {
            THROW_ERROR("Unable to write chemistry checkpoint parameters\n");
        }
        // Checkpoints are taken after drift: the closing source half is pending.
        // tau=0 instead denotes the untouched initial state. Fixed rungs make the
        // interval recoverable without a per-particle chemical clock.
        std::array<double, 3> const sourcePhase = {params.tau == 0.0 ? 0.0 : 1.0,
            std::ldexp(params.tauMax / getOptions().run.numberOfSteps,
                -1 - getOptions().chemistry.synchronizedRung),
            sphChemistryPendingHeating()};
        if (fwrite(sourcePhase.data(), sizeof(double), sourcePhase.size(), fp) !=
            sourcePhase.size()) {
            THROW_ERROR("Unable to write chemistry source phase\n");
        }
    }
#endif
    if (fwrite(&params, sizeof(params), 1, fp) != 1) {
        THROW_ERROR("Unable to write checkpoint driver state\n");
    }
}

DriverParameters readDriverState(FILE* fp) {
    DriverParameters params{};
    CheckpointHeader header{};
    uint64_t magic = 0;
    FREAD(&magic, sizeof(magic), 1, fp);
    if (fseek(fp, 0, SEEK_SET) != 0) {
        THROW_ERROR("Unable to rewind checkpoint\n");
    }
    if (magic == kCheckpointMagic) {
        FREAD(&header, sizeof(header), 1, fp);
        if ((header.flags & kRetiredParticleMesh) != 0U) {
            THROW_ERROR("Ordinary PM gravity checkpoints are no longer supported\n");
        }
        if (!checkpointCompatible(header, getOptions())) {
            THROW_ERROR("Checkpoint physics, particle layout or integrator differs from the "
                        "requested run\n");
        }
        if ((header.flags & kParticleMultipoleMesh) != 0U) {
            auto const meshParameters = readPmmmCheckpointParameters(fp);
            if (meshParameters[0] < 4 && getOptions().gravity.pmmmMaxPrimeFactor != 0) {
                THROW_ERROR("This PMMM checkpoint predates gravity.pmmmMaxPrimeFactor and used "
                            "unrestricted FFT mesh sizes. Restart with "
                            "--gravity.pmmmMaxPrimeFactor=infinity to preserve its mesh policy.\n");
            }
            if (!pmmmCheckpointCompatible(meshParameters, getOptions())) {
                THROW_ERROR(
                    "PMMM checkpoint version, mesh size, near/far expansion orders, opening "
                    "angle, neighbor width/method, gravity.pmmmMaxPrimeFactor, timestep parameters "
                    "or force convention differs from the requested run\n");
            }
        }
        if (getOptions().hydro.enabled) {
            auto sphParameters = sphCheckpointParameters(getOptions());
            FREAD(sphParameters.data(), sizeof(double), sphParameters.size(), fp);
            if (sphParameters != sphCheckpointParameters(getOptions())) {
                THROW_ERROR("SPH checkpoint parameters differ from the requested run\n");
            }
            if ((header.flags & kZPlaneAcceleration) != 0U) {
                double acceleration;
                FREAD(&acceleration, sizeof(acceleration), 1, fp);
                if (!zPlaneCheckpointCompatible(acceleration, getOptions())) {
                    THROW_ERROR("SPH checkpoint hydro.zPlaneAcceleration differs from the "
                                "requested run\n");
                }
            }
            if ((header.flags & kVariableViscosity) != 0U) {
                auto viscosityParameters = viscosityCheckpointParameters(getOptions());
                FREAD(viscosityParameters.data(), sizeof(double), viscosityParameters.size(), fp);
                if (viscosityParameters != viscosityCheckpointParameters(getOptions())) {
                    THROW_ERROR("SPH checkpoint variable-viscosity parameters differ from the "
                                "requested run\n");
                }
            }
        }
#if COSMICTIGER_WITH_SPH
        std::array<double, 3> sourcePhase{};
        if (getOptions().chemistry.enabled) {
            auto chemicalParameters = sphChemistryCheckpointParameters();
            uint64_t radiationChecksum;
            FREAD(chemicalParameters.data(), sizeof(double), chemicalParameters.size(), fp);
            FREAD(&radiationChecksum, sizeof(radiationChecksum), 1, fp);
            FREAD(sourcePhase.data(), sizeof(double), sourcePhase.size(), fp);
            auto finiteCheckpointScalar = [](double value) {
                uint64_t bits;
                std::memcpy(&bits, &value, sizeof(bits));
                return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
            };
            for (double value : chemicalParameters) {
                if (!finiteCheckpointScalar(value)) {
                    THROW_ERROR("Checkpoint chemistry parameters are not finite\n");
                }
            }
            for (double value : sourcePhase) {
                if (!finiteCheckpointScalar(value)) {
                    THROW_ERROR("Checkpoint chemistry source phase is not finite\n");
                }
            }
            if (chemicalParameters != sphChemistryCheckpointParameters() ||
                radiationChecksum != sphChemistryRadiationChecksum()) {
                THROW_ERROR("Checkpoint chemistry network, rates, EOS, radiation or solver "
                            "settings differ\n");
            }
        }
        if (getOptions().chemistry.enabled) {
            // Inspect the exponent bits: this TU otherwise permits fast-math.
            uint64_t bits;
            std::memcpy(&bits, &sourcePhase[2], sizeof(bits));
            if ((bits & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL) {
                THROW_ERROR("Checkpoint chemistry energy ledger is not finite\n");
            }
            sphChemistryPendingHeating() = sourcePhase[2];
        }
#endif
        FREAD(&params, sizeof(params), 1, fp);
#if COSMICTIGER_WITH_SPH
        if (getOptions().chemistry.enabled &&
            (sourcePhase[0] != (params.tau == 0.0 ? 0.0 : 1.0) ||
                sourcePhase[1] !=
                    std::ldexp(params.tauMax / getOptions().run.numberOfSteps,
                        -1 - getOptions().chemistry.synchronizedRung))) {
            THROW_ERROR("Checkpoint chemistry source phase is inconsistent\n");
        }
#endif
    } else {
        if (!legacyCheckpointCompatible(getOptions())) {
            THROW_ERROR("Legacy checkpoints support only the original dark-matter hierarchical "
                        "mode with the degree-eight Ferrers gravity kernel\n");
        }
        LegacyDriver old{};
        FREAD(&old, sizeof(old), 1, fp);
        params.a = old.a;
        params.adot = old.adot;
        params.dummy = old.dummy;
        params.tau = old.tau;
        params.tauMax = old.tauMax;
        params.energy0 = old.energy0;
        params.energies.pot = old.energies.pot;
        params.energies.kin = old.energies.kin;
        params.energies.tckin = old.energies.tckin;
        params.energies.xmom = old.energies.xmom;
        params.energies.ymom = old.energies.ymom;
        params.energies.zmom = old.energies.zmom;
        params.energies.nmom = old.energies.nmom;
        params.energies.cosmic = old.energies.cosmic;
        params.maxRung = old.maxRung;
        params.iter = old.iter;
        params.minrung0 = old.minRung0;
        params.step = old.step;
        params.bucketSize = old.bucketSize;
        params.totalProcessed = old.totalProcessed;
        params.flops = old.flops;
        params.runtime = old.runtime;
        params.years = old.years;
        params.itime = old.itime;
    }
    return params;
}

}    // namespace detail

void checkpointStateTest() {
    constexpr double kScale = 0.5;
    constexpr double kClock = 0.125;
    constexpr double kThermal = 3.0;
    constexpr double kHeating = 0.25;
    constexpr int kRung = 4;
    FILE* fp = tmpfile();
    if (fp == nullptr) {
        THROW_ERROR("Cannot create checkpoint test stream\n");
    }
    DriverParameters input{};
    input.a = kScale;
    input.tau = kClock;
    input.tauMax = 2.0;
    input.itime = 128;
    input.maxRung = kRung;
    input.energies.therm = kThermal;
    input.energies.heating = kHeating;
    detail::writeDriverState(fp, input);
    rewind(fp);
    auto const restored = detail::readDriverState(fp);
    ALWAYS_ASSERT(
        restored.a == input.a && restored.tau == input.tau && restored.tauMax == input.tauMax);
    ALWAYS_ASSERT(restored.itime == input.itime && restored.maxRung == input.maxRung);
    ALWAYS_ASSERT(restored.energies.therm == kThermal && restored.energies.heating == kHeating);
    auto changed = getOptions();
    auto const header = detail::checkpointHeader(changed);
    changed.gravity.method = changed.gravity.method == "pmmm" ? "fmm" : "pmmm";
    ALWAYS_ASSERT(!detail::checkpointCompatible(header, changed));
    changed = getOptions();
    // Reject the old solver bit for either supported method rather than treating
    // its extension bytes as driver state after ordinary PM support is removed.
    for (auto const* method : {"fmm", "pmmm"}) {
        changed.gravity.method = method;
        auto retiredPmHeader = detail::checkpointHeader(changed);
        retiredPmHeader.flags &= ~detail::kParticleMultipoleMesh;
        retiredPmHeader.flags |= detail::kRetiredParticleMesh;
        ALWAYS_ASSERT(!detail::checkpointCompatible(retiredPmHeader, changed));
    }
    changed = getOptions();
    auto const pmmmParameters = detail::pmmmCheckpointParameters(changed);
    ALWAYS_ASSERT(pmmmParameters[0] == 4 &&
        pmmmParameters[10] == uint64_t(changed.gravity.pmmmMaxPrimeFactor));
    changed.gravity.meshSize += 1;
    ALWAYS_ASSERT(pmmmParameters != detail::pmmmCheckpointParameters(changed));
    changed = getOptions();
    changed.gravity.fmmOrder += 1;
    ALWAYS_ASSERT(pmmmParameters != detail::pmmmCheckpointParameters(changed));
    changed = getOptions();
    changed.gravity.pmmmOrder += 1;
    ALWAYS_ASSERT(!detail::pmmmCheckpointCompatible(pmmmParameters, changed));
    changed = getOptions();
    changed.gravity.openingAngle *= 0.5;
    ALWAYS_ASSERT(!detail::pmmmCheckpointCompatible(pmmmParameters, changed));
    changed = getOptions();
    changed.gravity.pmmmNearWidth = changed.gravity.pmmmNearWidth == 1 ? 2 : 1;
    ALWAYS_ASSERT(pmmmParameters != detail::pmmmCheckpointParameters(changed));
    changed = getOptions();
    changed.gravity.pmmmNearMethod = changed.gravity.pmmmNearMethod == "fmm" ? "pp" : "fmm";
    ALWAYS_ASSERT(pmmmParameters != detail::pmmmCheckpointParameters(changed));
    changed = getOptions();
    changed.gravity.pmmmMaxPrimeFactor = changed.gravity.pmmmMaxPrimeFactor == 0 ? 5 : 0;
    ALWAYS_ASSERT(!detail::pmmmCheckpointCompatible(pmmmParameters, changed));
    // A matching current mesh does not make two prime-factor policies equivalent:
    // their next rung-selected mesh may differ after the restart.
    changed.gravity.meshSize = 8;
    changed.gravity.pmmmMaxPrimeFactor = 5;
    auto const smoothPmmmParameters = detail::pmmmCheckpointParameters(changed);
    changed.gravity.pmmmMaxPrimeFactor = 3;
    ALWAYS_ASSERT(!detail::pmmmCheckpointCompatible(smoothPmmmParameters, changed));
    changed = getOptions();
    changed.gravity.pmmmOrder = changed.gravity.fmmOrder;
    changed.gravity.pmmmNearMethod = "pp";
    changed.gravity.pmmmMaxPrimeFactor = 0;
    auto legacyPmmmParameters = detail::pmmmCheckpointParameters(changed);
    legacyPmmmParameters[0] = 1;
    legacyPmmmParameters[7] = 0;
    legacyPmmmParameters[8] = legacyPmmmParameters[9] = 0;
    ALWAYS_ASSERT(detail::pmmmCheckpointCompatible(legacyPmmmParameters, changed));
    changed.gravity.pmmmNearMethod = "fmm";
    ALWAYS_ASSERT(!detail::pmmmCheckpointCompatible(legacyPmmmParameters, changed));
    legacyPmmmParameters[0] = 2;
    legacyPmmmParameters[7] = 1;
    ALWAYS_ASSERT(detail::pmmmCheckpointCompatible(legacyPmmmParameters, changed));
    changed.gravity.openingAngle *= 0.5;
    ALWAYS_ASSERT(detail::pmmmCheckpointCompatible(legacyPmmmParameters, changed));
    changed.gravity.pmmmOrder += 1;
    ALWAYS_ASSERT(!detail::pmmmCheckpointCompatible(legacyPmmmParameters, changed));
    changed = getOptions();
    changed.gravity.pmmmMaxPrimeFactor = 0;
    legacyPmmmParameters = detail::pmmmCheckpointParameters(changed);
    legacyPmmmParameters[0] = 3;
    ALWAYS_ASSERT(detail::pmmmCheckpointCompatible(legacyPmmmParameters, changed));
    changed.gravity.pmmmMaxPrimeFactor = 5;
    ALWAYS_ASSERT(!detail::pmmmCheckpointCompatible(legacyPmmmParameters, changed));
    // Decode each historical extension without consuming any following driver
    // bytes, and require the old mesh policy to remain explicitly unrestricted.
    constexpr std::array<size_t, 4> kPmmmParameterWords{7, 8, 10, 11};
    constexpr uint64_t kFollowingPayload = 0x0123456789abcdefULL;
    for (uint64_t version = 1; version <= 4; ++version) {
        changed = getOptions();
        changed.gravity.pmmmOrder = changed.gravity.fmmOrder;
        changed.gravity.pmmmNearMethod = "pp";
        changed.gravity.pmmmMaxPrimeFactor = version < 4 ? 0 : 5;
        auto serialized = detail::pmmmCheckpointParameters(changed);
        serialized[0] = version;
        detail::CheckpointFile stream(tmpfile());
        ALWAYS_ASSERT(stream != nullptr);
        size_t const words = kPmmmParameterWords[version - 1];
        ALWAYS_ASSERT(fwrite(serialized.data(), sizeof(uint64_t), words, stream.get()) == words);
        ALWAYS_ASSERT(fwrite(&kFollowingPayload, sizeof(uint64_t), 1, stream.get()) == 1);
        rewind(stream.get());
        auto const decoded = detail::readPmmmCheckpointParameters(stream.get());
        ALWAYS_ASSERT(detail::pmmmCheckpointCompatible(decoded, changed));
        uint64_t followingPayload = 0;
        ALWAYS_ASSERT(fread(&followingPayload, sizeof(uint64_t), 1, stream.get()) == 1 &&
            followingPayload == kFollowingPayload);
        if (version < 4) {
            ALWAYS_ASSERT(decoded[10] == 0);
            changed.gravity.pmmmMaxPrimeFactor = 5;
            ALWAYS_ASSERT(!detail::pmmmCheckpointCompatible(decoded, changed));
        }
    }
    changed = getOptions();
    changed.run.minRung += 1;
    ALWAYS_ASSERT(pmmmParameters != detail::pmmmCheckpointParameters(changed));
    changed = getOptions();
    changed.gravity.timestepEta *= 0.5;
    ALWAYS_ASSERT(pmmmParameters != detail::pmmmCheckpointParameters(changed));
    changed = getOptions();
    changed.compute.positionType =
        changed.compute.positionType == "fixed32" ? "fixed64" : "fixed32";
    ALWAYS_ASSERT(!detail::checkpointCompatible(header, changed));
    changed = getOptions();
    changed.output.units = "both";
    changed.output.frame = "lab";
    ALWAYS_ASSERT(detail::checkpointCompatible(header, changed));
    changed = getOptions();
    if (changed.hydro.enabled) {
        auto oldFormulation = header;
        oldFormulation.reserved = 0;
        oldFormulation.version = 1;
        ALWAYS_ASSERT(!detail::checkpointCompatible(oldFormulation, changed));
    }
    changed.hydro.enabled = !changed.hydro.enabled;
    ALWAYS_ASSERT(!detail::checkpointCompatible(header, changed));
    changed = getOptions();
    changed.run.hierarchicalGravity = !changed.run.hierarchicalGravity;
    ALWAYS_ASSERT(!detail::checkpointCompatible(header, changed));
    changed = getOptions();
    auto removedPhysics = header;
    removedPhysics.flags |= 1U << 1;
    ALWAYS_ASSERT(!detail::checkpointCompatible(removedPhysics, changed));
    removedPhysics = header;
    removedPhysics.flags |= 1U << 10;
    ALWAYS_ASSERT(!detail::checkpointCompatible(removedPhysics, changed));
    auto const differentKernelDegree = [](std::string const& family, int degree) {
        if (family == "spline") {
            return degree == 3 ? 5 : 3;
        }
        if (family == "wendland") {
            return degree == 2 ? 5 : 2;
        }
        return degree == 8 ? 6 : 8;
    };
    changed = getOptions();
    changed.gravity.kernelFamily = changed.gravity.kernelFamily == "spline" ? "ferrers" : "spline";
    changed.gravity.kernelDegree = 8;
    ALWAYS_ASSERT(!detail::checkpointCompatible(header, changed));
    auto const changedGravityHeader = detail::checkpointHeader(changed);
    ALWAYS_ASSERT(detail::checkpointCompatible(changedGravityHeader, changed));
    ALWAYS_ASSERT(!detail::checkpointCompatible(changedGravityHeader, getOptions()));
    changed = getOptions();
    changed.gravity.kernelDegree =
        differentKernelDegree(changed.gravity.kernelFamily, changed.gravity.kernelDegree);
    ALWAYS_ASSERT(!detail::checkpointCompatible(header, changed));
    changed = getOptions();
    changed.hydro.kernelFamily = changed.hydro.kernelFamily == "spline" ? "ferrers" : "spline";
    changed.hydro.kernelDegree = 8;
    ALWAYS_ASSERT(!detail::checkpointCompatible(header, changed));
    auto const changedHydroHeader = detail::checkpointHeader(changed);
    ALWAYS_ASSERT(detail::checkpointCompatible(changedHydroHeader, changed));
    ALWAYS_ASSERT(!detail::checkpointCompatible(changedHydroHeader, getOptions()));
    changed = getOptions();
    changed.hydro.kernelDegree =
        differentKernelDegree(changed.hydro.kernelFamily, changed.hydro.kernelDegree);
    ALWAYS_ASSERT(!detail::checkpointCompatible(header, changed));
    auto defaultKernelOptions = getOptions();
    defaultKernelOptions.gravity.kernelFamily = "ferrers";
    defaultKernelOptions.gravity.kernelDegree = 8;
    defaultKernelOptions.hydro.kernelFamily = "spline";
    defaultKernelOptions.hydro.kernelDegree = 3;
    ALWAYS_ASSERT(detail::checkpointHeader(defaultKernelOptions).reserved ==
        (defaultKernelOptions.hydro.enabled ? detail::kMonaghanSph | detail::kCompactSphLayout :
                                              0U));
    auto legacyKernelOptions = defaultKernelOptions;
    legacyKernelOptions.hydro.enabled = false;
    legacyKernelOptions.run.hierarchicalGravity = true;
    ALWAYS_ASSERT(detail::legacyCheckpointCompatible(legacyKernelOptions) ==
        (legacyKernelOptions.compute.positionType == "fixed32" &&
            legacyKernelOptions.gravity.method == "fmm"));
    legacyKernelOptions.gravity.kernelFamily = "spline";
    ALWAYS_ASSERT(!detail::legacyCheckpointCompatible(legacyKernelOptions));
    legacyKernelOptions.gravity.kernelFamily = "ferrers";
    legacyKernelOptions.gravity.kernelDegree = 6;
    ALWAYS_ASSERT(!detail::legacyCheckpointCompatible(legacyKernelOptions));
    auto legacySphOptions = getOptions();
    legacySphOptions.hydro.enabled = true;
    auto legacySphHeader = detail::checkpointHeader(legacySphOptions);
    ALWAYS_ASSERT(legacySphHeader.version == detail::kCheckpointVersion);
    ALWAYS_ASSERT(detail::checkpointCompatible(legacySphHeader, legacySphOptions));
    legacySphHeader.reserved &= ~detail::kCompactSphLayout;
    ALWAYS_ASSERT(!detail::checkpointCompatible(legacySphHeader, legacySphOptions));
    legacySphHeader.version = 2;
    ALWAYS_ASSERT(!detail::checkpointCompatible(legacySphHeader, legacySphOptions));
    auto legacyDarkMatterOptions = getOptions();
    legacyDarkMatterOptions.hydro.enabled = false;
    auto legacyDarkMatterHeader = detail::checkpointHeader(legacyDarkMatterOptions);
    legacyDarkMatterHeader.version = detail::kLegacyDarkMatterCheckpointVersion;
    ALWAYS_ASSERT(detail::checkpointCompatible(legacyDarkMatterHeader, legacyDarkMatterOptions) ==
        (legacyDarkMatterOptions.compute.positionType == "fixed32" &&
            legacyDarkMatterOptions.gravity.method == "fmm"));
    auto const sphParameters = detail::sphCheckpointParameters(getOptions());
    auto balsaraOptions = getOptions();
    balsaraOptions.hydro.enabled = true;
    balsaraOptions.hydro.balsaraSwitch = true;
    auto const balsaraHeader = detail::checkpointHeader(balsaraOptions);
    ALWAYS_ASSERT(detail::checkpointCompatible(balsaraHeader, balsaraOptions));
    balsaraOptions.hydro.balsaraSwitch = false;
    ALWAYS_ASSERT(!detail::checkpointCompatible(balsaraHeader, balsaraOptions));
    auto const unlimitedHeader = detail::checkpointHeader(balsaraOptions);
    ALWAYS_ASSERT(detail::checkpointCompatible(unlimitedHeader, balsaraOptions));
    balsaraOptions.hydro.balsaraSwitch = true;
    ALWAYS_ASSERT(!detail::checkpointCompatible(unlimitedHeader, balsaraOptions));
    auto viscosityOptions = getOptions();
    viscosityOptions.hydro.enabled = true;
    viscosityOptions.hydro.viscosityEvolution = "morrisAndMonaghan";
    auto const viscosityHeader = detail::checkpointHeader(viscosityOptions);
    auto const viscosityParameters = detail::viscosityCheckpointParameters(viscosityOptions);
    ALWAYS_ASSERT(detail::checkpointCompatible(viscosityHeader, viscosityOptions));
    viscosityOptions.hydro.viscosityEvolution = "none";
    ALWAYS_ASSERT(!detail::checkpointCompatible(viscosityHeader, viscosityOptions));
    auto const fixedViscosityHeader = detail::checkpointHeader(viscosityOptions);
    ALWAYS_ASSERT((fixedViscosityHeader.flags & detail::kVariableViscosity) == 0U);
    auto betaOptions = viscosityOptions;
    betaOptions.hydro.viscosityAlpha = 1.0;
    auto const fixedViscosityParameters = detail::sphCheckpointParameters(betaOptions);
    betaOptions.hydro.viscosityBeta += 0.5;
    ALWAYS_ASSERT(fixedViscosityParameters != detail::sphCheckpointParameters(betaOptions));
    viscosityOptions.hydro.viscosityEvolution = "morrisAndMonaghan";
    ALWAYS_ASSERT(!detail::checkpointCompatible(fixedViscosityHeader, viscosityOptions));
    for (int parameter = 0; parameter < 7; ++parameter) {
        auto altered = viscosityOptions;
        if (parameter == 0) {
            altered.hydro.viscosityAlphaMin += 0.125;
        } else if (parameter == 1) {
            altered.hydro.viscosityDecay += 0.125;
        } else if (parameter == 2) {
            altered.hydro.viscosityEvolution = "cullenAndDehnen";
        } else if (parameter == 3) {
            altered.hydro.viscosityEvolution = "priceEtAl";
        } else if (parameter == 4) {
            altered.hydro.viscosityAlphaMax += 0.125;
        } else if (parameter == 5) {
            altered.hydro.viscosityDecayCD += 0.125;
        } else {
            altered.hydro.viscosityAlphaMaxPrice += 0.125;
        }
        ALWAYS_ASSERT(viscosityParameters != detail::viscosityCheckpointParameters(altered));
    }
    auto planeOptions = getOptions();
    planeOptions.hydro.enabled = true;
    planeOptions.hydro.zPlaneAcceleration = 0.0;
    auto const noPlaneHeader = detail::checkpointHeader(planeOptions);
    ALWAYS_ASSERT((noPlaneHeader.flags & detail::kZPlaneAcceleration) == 0U);
    auto const noPlaneParameters = detail::sphCheckpointParameters(planeOptions);
    planeOptions.hydro.zPlaneAcceleration = 0.5;
    auto const planeHeader = detail::checkpointHeader(planeOptions);
    ALWAYS_ASSERT((planeHeader.flags & detail::kZPlaneAcceleration) != 0U);
    ALWAYS_ASSERT(!detail::checkpointCompatible(noPlaneHeader, planeOptions));
    ALWAYS_ASSERT(detail::checkpointCompatible(planeHeader, planeOptions));
    ALWAYS_ASSERT(noPlaneParameters == detail::sphCheckpointParameters(planeOptions));
    ALWAYS_ASSERT(detail::zPlaneCheckpointCompatible(0.5, planeOptions));
    ALWAYS_ASSERT(!detail::zPlaneCheckpointCompatible(0.25, planeOptions));
    planeOptions.hydro.zPlaneAcceleration = 0.0;
    ALWAYS_ASSERT(!detail::checkpointCompatible(planeHeader, planeOptions));
    ALWAYS_ASSERT(!detail::zPlaneCheckpointCompatible(0.5, planeOptions));
    changed = getOptions();
    changed.hydro.viscosityAlpha += 0.25;
    ALWAYS_ASSERT(sphParameters != detail::sphCheckpointParameters(changed));
    changed = getOptions();
    changed.hydro.neighborNumber += 1.0;
    ALWAYS_ASSERT(sphParameters != detail::sphCheckpointParameters(changed));
    changed = getOptions();
    changed.hydro.maxSupportRadius *= 2.0;
    ALWAYS_ASSERT(sphParameters != detail::sphCheckpointParameters(changed));
    changed = getOptions();
    changed.run.numberOfSteps++;
    ALWAYS_ASSERT(!detail::checkpointCompatible(header, changed));
    changed = getOptions();
    changed.initial.numberOfParticles++;
    ALWAYS_ASSERT(!detail::checkpointCompatible(header, changed));
    if (detail::legacyCheckpointCompatible(getOptions())) {
        detail::LegacyDriver legacy{};
        legacy.a = kScale;
        legacy.tau = kClock;
        legacy.maxRung = kRung;
        legacy.energies.pot = -1.0;
        rewind(fp);
        ALWAYS_ASSERT(fwrite(&legacy, sizeof(legacy), 1, fp) == 1);
        rewind(fp);
        auto const previous = detail::readDriverState(fp);
        ALWAYS_ASSERT(previous.a == kScale && previous.tau == kClock && previous.maxRung == kRung);
        ALWAYS_ASSERT(previous.energies.pot == -1.0 && previous.energies.therm == 0.0);
    }
    fclose(fp);
    if (!getOptions().hydro.enabled && !getOptions().compute.cuda) {
        // Exercise the actual particle payload with low coordinate bits that a
        // double round trip cannot preserve in the fixed64 backend.
        fp = tmpfile();
        ALWAYS_ASSERT(fp != nullptr);
        constexpr ParticleInteger kCount = 4;
        particlesResize(kCount);
        for (ParticleInteger particle = 0; particle < kCount; ++particle) {
            for (int dim = 0; dim < kDimensions; ++dim) {
                PositionInteger const raw = (PositionInteger(1) << (kPositionBits - 1)) +
                    PositionInteger(17 * particle + dim + 1);
                particlesPos(dim, particle).raw() = raw;
                particlesVel(dim, particle) = float(particle + dim);
            }
            particlesRung(particle) = char(particle);
            if (getOptions().analysis.groups) {
                particlesLastgroup(particle) = particle;
            }
            if (getOptions().analysis.tracers) {
                particlesTracer(particle) = char(particle);
            }
        }
        particlesSave(fp);
        for (ParticleInteger particle = 0; particle < kCount; ++particle) {
            for (int dim = 0; dim < kDimensions; ++dim) {
                particlesPos(dim, particle).raw() = 0;
            }
        }
        rewind(fp);
        particlesLoad(fp);
        ALWAYS_ASSERT(particlesSize() == kCount);
        for (ParticleInteger particle = 0; particle < kCount; ++particle) {
            for (int dim = 0; dim < kDimensions; ++dim) {
                PositionInteger const expected = (PositionInteger(1) << (kPositionBits - 1)) +
                    PositionInteger(17 * particle + dim + 1);
                ALWAYS_ASSERT(particlesPos(dim, particle).raw() == expected);
                ALWAYS_ASSERT(particlesVel(dim, particle) == float(particle + dim));
            }
            ALWAYS_ASSERT(particlesRung(particle) == char(particle));
        }
        fclose(fp);
        particlesFree();
        particlesDestroy();
        PRINT("Checkpoint position payload round trip passed (%i bits)\n", kPositionBits);
    }
    PRINT("Checkpoint state and mode validation passed\n");
}

void writeCheckpoint(DriverParameters params) {
    auto const directory = detail::checkpointDirectory(params.iter);
    auto const stagingDirectory = detail::checkpointStagingDirectory(params.iter);
    if (hpxRank() == 0) {
        terminal::detailPrint("Writing checkpoint %s\n", directory.c_str());
        if (detail::fs::exists(directory)) {
            THROW_ERROR("Refusing to replace existing checkpoint %s\n", directory.c_str());
        }
        if (!detail::fs::create_directory(stagingDirectory)) {
            THROW_ERROR("Checkpoint staging directory %s already exists; inspect the incomplete "
                        "checkpoint before retrying\n",
                stagingDirectory.c_str());
        }
    }
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<WriteCheckpointAction>(c, params));
    }
    auto const rankPath = detail::checkpointRankFile(stagingDirectory, hpxRank());
    detail::CheckpointFile file(fopen(rankPath.c_str(), "wb"));
    if (!file) {
        THROW_ERROR("Unable to open %s for writing\n", rankPath.c_str());
    }
    detail::writeDriverState(file.get(), params);
    particlesSave(file.get());
    domainsSave(file.get());
#if COSMICTIGER_WITH_GRAVITY
    if (getOptions().lightcone.enabled) {
        lcSave(file.get());
    }
#endif
    detail::flushAndCloseCheckpointFile(file, rankPath);
    for (auto& future : futs) {
        future.get();
    }
    if (hpxRank() == 0) {
        detail::writeCheckpointScenario(stagingDirectory);
        detail::writeCheckpointManifest(stagingDirectory, params.iter);
        detail::syncDirectory(stagingDirectory);
        if (detail::fs::exists(directory)) {
            THROW_ERROR("Refusing to replace existing checkpoint %s\n", directory.c_str());
        }
        detail::fs::rename(stagingDirectory, directory);
        detail::syncDirectory(directory.has_parent_path() ? directory.parent_path() : ".");
        PRINT("  Checkpoint saved: %s\n", directory.c_str());
    }
}

DriverParameters readCheckpoint() {
    DriverParameters params{};
    if (hpxRank() == 0) {
        PRINT("Reading checkpoint\n");
        detail::validateCheckpointDirectory(
            detail::checkpointDirectory(getOptions().checkpoint.readIndex),
            getOptions().checkpoint.readIndex);
    }
    Vector<hpx::future<DriverParameters>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<ReadCheckpointAction>(c));
    }
    int const iter = getOptions().checkpoint.readIndex;
    auto const rankPath = detail::checkpointRankFile(detail::checkpointDirectory(iter), hpxRank());
    detail::CheckpointFile file(fopen(rankPath.c_str(), "rb"));
    if (!file) {
        THROW_ERROR("Unable to open %s for reading.\n", rankPath.c_str());
    }
    params = detail::readDriverState(file.get());
    particlesLoad(file.get());
    domainsLoad(file.get());
#if COSMICTIGER_WITH_GRAVITY
    if (getOptions().lightcone.enabled) {
        lcLoad(file.get());
    }
#endif
    if (ferror(file.get()) != 0) {
        THROW_ERROR("Unable to read checkpoint file %s\n", rankPath.c_str());
    }
    FILE* rawFile = file.release();
    if (fclose(rawFile) != 0) {
        THROW_ERROR("Unable to close checkpoint file %s\n", rankPath.c_str());
    }
    for (auto& future : futs) {
        future.get();
    }
    if (hpxRank() == 0) {
        PRINT("Done reading checkpoint\n");
    }
    return params;
}
