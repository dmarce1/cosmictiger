/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <ios>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

#if COSMICTIGER_ENABLE_ROCKSTAR
#include <cosmictiger/analysis/lightcone.hpp>
#endif
#include <cosmictiger/core/driver.hpp>
#include <cosmictiger/core/file_handle.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/profiler.hpp>
#include <cosmictiger/core/terminal.hpp>
#include <cosmictiger/gravity/opening_schedule.hpp>
#if COSMICTIGER_WITH_GRAVITY
#include <cosmictiger/gravity/tree.hpp>
#endif
#include <cosmictiger/io/checkpoint.hpp>
#include <cosmictiger/io/output.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/particles/particles.hpp>
#if COSMICTIGER_WITH_SPH
#include <cosmictiger/sph/sph.hpp>
#if COSMICTIGER_ENABLE_CHEMISTRY
#include <cosmictiger/chemistry/sph.hpp>
#endif
#include <cosmictiger/sph/particles.hpp>
#include <cosmictiger/sph/time_integration.hpp>
#endif

HPX_PLAIN_ACTION(writeCheckpoint, WriteCheckpointAction);
HPX_PLAIN_ACTION(readCheckpoint, ReadCheckpointAction);

/// Prevent a fresh run from reusing an output directory containing checkpoint state without
/// an explicit restart.
void validateCheckpointOutput() {
    auto const& options = getOptions();
    if (hpxRank() != 0 || !options.run.test.empty()) {
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
        if (options.checkpoint.readIndex >= 0 && !name.ends_with(".incomplete") &&
            std::stoll(std::string(index)) <= options.checkpoint.readIndex) {
            continue;
        }
        throw std::runtime_error("Cannot start this run: output.directory contains a conflicting "
                                 "checkpoint path " +
            entry.path().string() +
            ". Choose a new --output.directory, or use --checkpoint.readIndex to restart from "
            "a completed checkpoint. Preserve or move the previous run before reusing its "
            "output directory.");
    }
}

/// Read optional requested checkpoint redshifts from checkpoint.txt and report the parsed
/// list.
Vector<double> readCheckpointList() {
    Vector<double> chkpts;
    FILE* fp = fopen("checkpoint.txt", "rt");
    if (fp != nullptr) {
        float z;
        std::cout << "Checkpoints will be written at z = \n" << std::flush;
        while (fscanf(fp, "%f\n", &z) == 1) {
            std::cout << std::format("{:f}\n", z) << std::flush;
            chkpts.push_back(z);
        }

        fclose(fp);
    }
    return chkpts;
}

namespace detail {

/// Create a non-overwriting redshift alias for a completed indexed checkpoint.
void checkpointRedshiftAlias(int iteration, double redshift) {
    char aliasName[128];
    snprintf(aliasName, sizeof(aliasName), "checkpoint.z.%.1f", redshift);
    auto const aliasPath = outputPath(aliasName);
    std::error_code error;
    auto const status = std::filesystem::symlink_status(aliasPath, error);
    if (!error && status.type() != std::filesystem::file_type::not_found) {
        std::cout
            << std::format(
                   "Redshift checkpoint alias {:s} already exists; restart from checkpoint.{:d}\n",
                   aliasName, iteration)
            << std::flush;
        return;
    }
    error.clear();
    std::filesystem::create_directory_symlink(
        "checkpoint." + std::to_string(iteration), aliasPath, error);
    if (error) {
        std::cout << std::format("Unable to create redshift checkpoint alias {:s}; restart from "
                                 "checkpoint.{:d}\n",
                         aliasName, iteration)
                  << std::flush;
    }
}

}    // namespace detail

namespace detail {

constexpr uint64_t kCheckpointMagic = 0x4354535048303031ULL;
constexpr uint32_t kLegacyDarkMatterCheckpointVersion = 1;
constexpr uint32_t kCheckpointVersion = 3;
constexpr uint32_t kCheckpointManifestVersion = 1;
constexpr uint32_t kScenarioManifestVersion = 2;
// The low byte of the physics code identifies the SPH pressure operator.
// Retained only to identify and reject retired Monaghan pressure checkpoints.
constexpr uint32_t kRetiredMonaghanSph = 1;
constexpr uint32_t kVariationalSph = 2;
constexpr uint32_t kSphFormationMask = 0xffU;
constexpr uint32_t kCompactSphLayout = 1U << 24;
// Bit 25 of the physics code marks gas state without the retired viscosity models. Earlier
// files carry a longer viscosity block, so they are rejected before it is read.
constexpr uint32_t kScalarViscosityLayout = 1U << 25;
// Bit 26 of the physics code marks the Rosswog-limited viscosity block: alpha min, alpha max,
// decay and the Balsara sound-speed coefficient, with no fixed-alpha parameter. Earlier files are
// rejected before that block is read.
constexpr uint32_t kRosswogViscosityLayout = 1U << 26;
constexpr uint32_t kArtificialConduction = 1U << 25;
constexpr uint32_t kIndependentParticleCounts = 1U << 26;
constexpr uint32_t kChemistry = 1U << 11;
constexpr uint32_t kZPlaneAcceleration = 1U << 12;
constexpr uint32_t kBalsaraSwitch = 1U << 13;
constexpr uint32_t kVariableViscosity = 1U << 14;
constexpr uint32_t kFixed64Positions = 1U << 15;
// Retired PM checkpoints are recognized only to reject them before reading payloads.
constexpr uint32_t kRetiredParticleMesh = 1U << 16;
constexpr uint32_t kParticleMultipoleMesh = 1U << 17;
constexpr uint32_t kRuntimeAlgorithms = 1U << 18;
constexpr uint32_t kGlobalIntegrator = 1U << 19;
constexpr uint32_t kStarFormation = 1U << 20;
// Identifies full-width rung boundaries as well as the particle index width.
// Older 64-bit-index files truncated these boundaries to sizeof(int).
constexpr uint32_t kWideParticleIndices = 1U << 21;
// Retired scalar particle-mesh solvers (pm, p3m) with their outer PM interval;
// recognized only to reject such checkpoints.
constexpr uint32_t kParticleMeshInterval = 1U << 22;
// Retired multipole-mesh interval bit, recognized only to reject old checkpoints.
constexpr uint32_t kMeshGravityInterval = 1U << 23;
constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

namespace fs = std::filesystem;

struct CheckpointFileInfo {
    uint64_t size;
    uint64_t checksum;
};

using CheckpointFile = UniqueFile;

/// Return the final output-directory path for an indexed checkpoint.
fs::path checkpointDirectory(int iteration) {
    return outputPath("checkpoint." + std::to_string(iteration));
}

/// Return the incomplete staging-directory path used while an indexed checkpoint is being
/// written.
fs::path checkpointStagingDirectory(int iteration) {
    return outputPath("checkpoint." + std::to_string(iteration) + ".incomplete");
}

/// Construct the checkpoint payload filename for one rank inside a supplied directory.
fs::path checkpointRankFile(fs::path const& directory, int rank) {
    return directory / ("checkpoint." + std::to_string(rank) + ".dat");
}

/// Synchronize directory metadata to storage and require successful descriptor closure.
void syncDirectory(fs::path const& directory) {
    int const fd = open(directory.c_str(), O_RDONLY | O_DIRECTORY);
    if (fd < 0) {
        throw std::ios_base::failure(
            std::format("Unable to open checkpoint directory {} for sync", directory.c_str()));
    }
    int const result = fsync(fd);
    int const closeResult = close(fd);
    if (result != 0 || closeResult != 0) {
        throw std::ios_base::failure(
            std::format("Unable to sync checkpoint directory {}", directory.c_str()));
    }
}

/// Check, flush, synchronize, and close a checkpoint file before publication.
void flushAndCloseCheckpointFile(CheckpointFile& file, fs::path const& path) {
    if (ferror(file.get()) != 0 || fflush(file.get()) != 0 || fsync(fileno(file.get())) != 0) {
        throw std::ios_base::failure(
            std::format("Unable to flush checkpoint file {}", path.c_str()));
    }
    FILE* rawFile = file.release();
    if (fclose(rawFile) != 0) {
        throw std::ios_base::failure(
            std::format("Unable to close checkpoint file {}", path.c_str()));
    }
}

/// Compute a checkpoint file's byte count and FNV checksum, rejecting read failures.
CheckpointFileInfo checkpointFileInfo(fs::path const& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::ios_base::failure(
            std::format("Unable to inspect checkpoint file {}", path.c_str()));
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
        throw std::ios_base::failure(
            std::format("Unable to read checkpoint file {}", path.c_str()));
    }
    return info;
}

/// Read the version from a recognized checkpoint header, returning zero for an unrecognized
/// file.
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

/// Write and synchronize the scenario identity, duration, and gravity mode for restart
/// validation.
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
        throw std::ios_base::failure(
            std::format("Unable to write checkpoint scenario metadata {}", path.c_str()));
    }
    flushAndCloseCheckpointFile(file, path);
}

/// Require scenario metadata to match the requested run and reject malformed or trailing
/// content.
void validateCheckpointScenario(fs::path const& directory) {
    if (!getOptions().run.test.empty()) {
        return;
    }
    auto const path = directory / "scenario.meta";
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error(
            std::format("Checkpoint {} has no scenario metadata; use its original executable to "
                        "resume a checkpoint written before first-class scenarios",
                directory.c_str()));
    }
    std::string label;
    std::string scenario;
    int version = 0;
    int gravity = -1;
    double endTime = -1.0;
    if (!(file >> label >> version >> scenario >> endTime >> gravity) ||
        label != "COSMICTIGER_SCENARIO" || version != 1 || scenario != getOptions().run.scenario ||
        endTime != getOptions().run.endTime || gravity != int(getOptions().gravity.enabled)) {
        throw std::runtime_error(std::format("Checkpoint scenario, run.endTime or gravity.enabled "
                                             "differs from the requested run: {}",
            path.c_str()));
    }
    file >> std::ws;
    if (file.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error(std::format(
            "Checkpoint scenario metadata {} has unexpected trailing data", path.c_str()));
    }
}

/// Write and synchronize payload sizes/checksums and scenario metadata for a completed
/// checkpoint set.
void writeCheckpointManifest(fs::path const& directory, int iteration) {
    fs::path const path = directory / "checkpoint.meta";
    CheckpointFile file(fopen(path.c_str(), "wb"));
    if (!file) {
        throw std::ios_base::failure(
            std::format("Unable to create checkpoint manifest {}", path.c_str()));
    }
    auto const manifestVersion =
        getOptions().run.test.empty() ? kScenarioManifestVersion : kCheckpointManifestVersion;
    if (fprintf(file.get(), "COSMICTIGER_CHECKPOINT %u %u %i %i\n", manifestVersion,
            kCheckpointVersion, iteration, hpxSize()) < 0) {
        throw std::ios_base::failure(
            std::format("Unable to write checkpoint manifest {}", path.c_str()));
    }
    for (int rank = 0; rank < hpxSize(); rank++) {
        auto const rankPath = checkpointRankFile(directory, rank);
        if (checkpointFileVersion(rankPath) != kCheckpointVersion) {
            throw std::runtime_error(
                std::format("Checkpoint rank {} has an unexpected format version", rank));
        }
        auto const info = checkpointFileInfo(rankPath);
        if (fprintf(file.get(), "%i %llu %016llx\n", rank,
                static_cast<unsigned long long>(info.size),
                static_cast<unsigned long long>(info.checksum)) < 0) {
            throw std::ios_base::failure(
                std::format("Unable to write checkpoint manifest {}", path.c_str()));
        }
    }
    if (manifestVersion == kScenarioManifestVersion) {
        auto const info = checkpointFileInfo(directory / "scenario.meta");
        if (fprintf(file.get(), "SCENARIO %llu %016llx\n",
                static_cast<unsigned long long>(info.size),
                static_cast<unsigned long long>(info.checksum)) < 0) {
            throw std::ios_base::failure(
                std::format("Unable to write scenario checksum to {}", path.c_str()));
        }
    }
    flushAndCloseCheckpointFile(file, path);
}

/// Validate checkpoint presence, scenario compatibility, rank files, and available manifest
/// integrity checks.
void validateCheckpointDirectory(fs::path const& directory, int iteration) {
    if (!fs::is_directory(directory)) {
        throw std::runtime_error(
            std::format("Checkpoint directory {} does not exist", directory.c_str()));
    }
    validateCheckpointScenario(directory);
    fs::path const manifestPath = directory / "checkpoint.meta";
    if (!fs::exists(manifestPath)) {
        // Checkpoints written before format 3 have no manifest or integrity hashes.
        for (int rank = 0; rank < hpxSize(); rank++) {
            auto const rankPath = checkpointRankFile(directory, rank);
            if (!fs::is_regular_file(rankPath)) {
                throw std::runtime_error(
                    std::format("Legacy checkpoint is missing rank file {}", rankPath.c_str()));
            }
            if (checkpointFileVersion(rankPath) == kCheckpointVersion) {
                throw std::runtime_error(std::format(
                    "Format 3 checkpoint {} is missing its manifest", directory.c_str()));
            }
        }
        std::cout << "Reading legacy checkpoint without integrity manifest\n" << std::flush;
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
        throw std::runtime_error(std::format(
            "Checkpoint manifest {} has an incompatible version, iteration or locality count",
            manifestPath.c_str()));
    }
    for (int rank = 0; rank < rankCount; rank++) {
        int savedRank = -1;
        uint64_t savedSize = 0;
        uint64_t savedChecksum = 0;
        if (!(manifest >> std::dec >> savedRank >> savedSize >> std::hex >> savedChecksum) ||
            savedRank != rank) {
            throw std::runtime_error(std::format(
                "Checkpoint manifest {} has an invalid rank entry", manifestPath.c_str()));
        }
        auto const rankPath = checkpointRankFile(directory, rank);
        if (checkpointFileVersion(rankPath) != kCheckpointVersion) {
            throw std::runtime_error(std::format(
                "Checkpoint rank file {} has an incompatible format version", rankPath.c_str()));
        }
        auto const actual = checkpointFileInfo(rankPath);
        if (actual.size != savedSize || actual.checksum != savedChecksum) {
            throw std::runtime_error(std::format(
                "Checkpoint rank file {} failed size or checksum validation", rankPath.c_str()));
        }
    }
    if (manifestVersion == kScenarioManifestVersion) {
        std::string scenarioLabel;
        uint64_t savedSize = 0;
        uint64_t savedChecksum = 0;
        if (!(manifest >> scenarioLabel >> std::dec >> savedSize >> std::hex >> savedChecksum) ||
            scenarioLabel != "SCENARIO") {
            throw std::runtime_error(std::format(
                "Checkpoint manifest {} has an invalid scenario entry", manifestPath.c_str()));
        }
        auto const actual = checkpointFileInfo(directory / "scenario.meta");
        if (actual.size != savedSize || actual.checksum != savedChecksum) {
            throw std::runtime_error(
                "Checkpoint scenario metadata failed size or checksum validation");
        }
    }
    manifest >> std::ws;
    if (manifest.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error(std::format(
            "Checkpoint manifest {} has unexpected trailing data", manifestPath.c_str()));
    }
}

struct CheckpointHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t flags;
    double darkMatterMass;
    double gasMass;
    double gamma;
    double gravitySmoothingLength;
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

/// Encode enabled persistent layouts and runtime modes into the checkpoint feature flags.
uint32_t checkpointFlags(Options const& opts) {
    constexpr uint32_t kSph = 1U << 0;
    constexpr uint32_t kHierarchical = 1U << 6;
    constexpr uint32_t kLightcone = 1U << 7;
    constexpr uint32_t kGroups = 1U << 8;
    constexpr uint32_t kTracers = 1U << 9;
    // Leave retired option bits unused so old incompatible headers remain rejected.
    return (opts.run.scenario == "cosmology" && opts.hydro.enabled &&
                       opts.run.darkMatterParticleCount != opts.run.gasParticleCount ?
                   kIndependentParticleCounts :
                   0U) |
        kRuntimeAlgorithms | (sizeof(ParticleInteger) == 8 ? kWideParticleIndices : 0U) |
        (usesGlobalIntegration(opts.run) ? kGlobalIntegrator : 0U) |
        (opts.hydro.enabled ? kSph : 0U) | (opts.run.hierarchicalGravity ? kHierarchical : 0U) |
        (opts.lightcone.enabled ? kLightcone : 0U) | (opts.analysis.groups ? kGroups : 0U) |
        (opts.analysis.tracers ? kTracers : 0U) | (opts.chemistry.enabled ? kChemistry : 0U) |
        (opts.hydro.enabled && opts.scenario.rayleighTaylor.zPlaneAcceleration != 0.0 ?
                kZPlaneAcceleration :
                0U) |
        (opts.hydro.enabled ? kBalsaraSwitch : 0U) |
        (opts.hydro.enabled && opts.hydro.conductionCoefficient > 0.0 ? kArtificialConduction :
                                                                        0U) |
        (opts.hydro.enabled ? kVariableViscosity : 0U) |
        (opts.compute.positionType == "fixed64" ? kFixed64Positions : 0U) |
        (opts.starFormation.enabled ? kStarFormation : 0U);
}

/// Encode star-formation model parameters and stochastic seed for exact restart
/// compatibility checks.
std::array<uint64_t, 11> starFormationCheckpointParameters(Options const& opts) {
    auto bits = [](double value) {
        uint64_t result;
        std::memcpy(&result, &value, sizeof(result));
        return result;
    };
    auto const& sf = opts.starFormation;
    // Version three: GADGET-4 full-step cooling and stored-SFR conversion ordering.
    // Unit scales are checked by CheckpointHeader. OmegaB and h separately set
    // the comoving overdensity threshold even at a fixed total matter density.
    return {3, bits(sf.beta), bits(sf.evaporationFactor), bits(sf.starFormationTimeGyr),
        bits(sf.coldTemperature), bits(sf.supernovaTemperature), bits(sf.criticalPhysicalDensity),
        bits(sf.criticalOverDensity), sf.seed, bits(opts.cosmology.omegaB),
        bits(opts.cosmology.hubble)};
}

/// Encode a supported kernel family and degree into the checkpoint selection field.
uint32_t checkpointKernelCode(std::string const& family, int degree) {
    uint32_t familyCode = 0;
    if (family == "spline") {
        familyCode = 1;
    } else if (family == "wendland") {
        familyCode = 2;
    } else if (family == "ferrers") {
        familyCode = 3;
    } else {
        throw std::runtime_error(
            std::format("Unknown checkpoint kernel family: {}", family.c_str()));
    }
    if (degree < 0 || degree > 15) {
        throw std::runtime_error("Checkpoint kernel degree does not fit its selection field");
    }
    return (familyCode << 4) | static_cast<uint32_t>(degree);
}

/// Encode hydro formulation, storage, kernel choices, and spatial dimension into the
/// reserved header field.
uint32_t checkpointPhysicsCode(Options const& opts) {
    // Preserve the existing low-byte formulation and native header layout.
    // Historical default kernels encode zero in both new selection bytes.
    uint32_t const gravityCode =
        checkpointKernelCode(opts.gravity.kernelFamily, opts.gravity.kernelDegree) ^
        checkpointKernelCode("ferrers", 8);
    uint32_t const hydroCode =
        checkpointKernelCode(opts.hydro.kernelFamily, opts.hydro.kernelDegree) ^
        checkpointKernelCode("spline", 3);
    // Bit 24 identifies compact gas storage.
    return (opts.hydro.enabled ? kVariationalSph | kCompactSphLayout | kScalarViscosityLayout |
                       kRosswogViscosityLayout :
                                 0U) |
        (gravityCode << 8) | (hydroCode << 16) | (uint32_t(3 - kSpatialDimensions) << 28) |
        (kSpatialDimensions < 3 ? (1U << 30) : 0U);
}

// Pre-header dark-matter checkpoints hold only particle state and the legacy
// driver record; they restart under the original dark-matter hierarchical mode
// with the degree-eight Ferrers kernel, whatever solver wrote them.
bool legacyCheckpointCompatible(Options const& opts) {
    return opts.compute.positionType == "fixed32" && !opts.hydro.enabled &&
        opts.run.hierarchicalGravity && opts.gravity.kernelFamily == "ferrers" &&
        opts.gravity.kernelDegree == 8;
}

/// Build the checkpoint header from the current physics, units, particle counts, and
/// integration schedule.
CheckpointHeader checkpointHeader(Options const& opts) {
    CheckpointHeader header;
    header.magic = kCheckpointMagic;
    header.version = kCheckpointVersion;
    header.flags = checkpointFlags(opts);
    header.darkMatterMass = opts.initial.darkMatterMass;
    header.gasMass = opts.initial.gasMass;
    header.gamma = opts.hydro.heatCapacityRatio;
    header.gravitySmoothingLength = opts.gravity.smoothingLength;
    header.gravitationalMass = opts.gravity.gm;
    header.omegaMatter = opts.cosmology.omegaM;
    header.omegaRadiation = opts.cosmology.omegaR;
    header.omegaLambda = opts.cosmology.omegaLambda;
    header.unitLength = opts.units.lengthCm;
    header.unitTime = opts.units.timeSeconds;
    header.unitMass = opts.units.massG;
    header.particlesPerSpecies = opts.initial.numberOfParticles;
    header.nominalSteps = opts.run.outputIntervalCount();
    header.reserved = checkpointPhysicsCode(opts);
    return header;
}

/// Check supported header versions and required physical/layout values against the current
/// options.
bool checkpointCompatible(CheckpointHeader const& header, Options const& opts) {
    if ((header.flags &
            (kParticleMultipoleMesh | kMeshGravityInterval | kRetiredParticleMesh |
                kParticleMeshInterval)) != 0U) {
        return false;
    }

    bool const supportedVersion = header.version == kCheckpointVersion ||
        (!opts.hydro.enabled && opts.compute.positionType == "fixed32" &&
            header.version == kLegacyDarkMatterCheckpointVersion);
    uint32_t const ignoredFlags = kRuntimeAlgorithms;
    return supportedVersion && (header.flags & kRetiredParticleMesh) == 0U &&
        (header.flags & kParticleMeshInterval) == 0U &&
        (header.flags & ~ignoredFlags) == (checkpointFlags(opts) & ~ignoredFlags) &&
        header.reserved == checkpointPhysicsCode(opts) &&
        header.darkMatterMass == opts.initial.darkMatterMass &&
        header.gasMass == opts.initial.gasMass && header.gamma == opts.hydro.heatCapacityRatio &&
        header.gravitySmoothingLength == opts.gravity.smoothingLength &&
        header.gravitationalMass == opts.gravity.gm &&
        header.omegaMatter == opts.cosmology.omegaM &&
        header.omegaRadiation == opts.cosmology.omegaR &&
        header.omegaLambda == opts.cosmology.omegaLambda &&
        header.unitLength == opts.units.lengthCm && header.unitTime == opts.units.timeSeconds &&
        header.unitMass == opts.units.massG &&
        header.particlesPerSpecies == opts.initial.numberOfParticles &&
        header.nominalSteps == opts.run.outputIntervalCount();
}

// Stored only for SPH checkpoints; the dark-matter layout is unchanged.
std::array<double, 12> sphCheckpointParameters(Options const& opts) {
    return {opts.hydro.vonNeumannRichtmyerCoefficient, opts.hydro.viscosityEpsilon,
        opts.hydro.neighborNumber, opts.hydro.minSmoothingLength, opts.hydro.maxSmoothingLength,
        opts.hydro.cfl, opts.gravity.timestepEta, opts.hydro.yAcceleration,
        opts.scenario.disc.centralAcceleration, opts.scenario.disc.centralSmoothingLength,
        opts.hydro.damping, opts.gravity.enabled ? 1.0 : 0.0};
}

/// Return viscosity bounds and model controls recorded for restart compatibility.
std::array<double, 4> viscosityCheckpointParameters(Options const& opts) {
    return {opts.hydro.minimumBulkViscosityCoefficient, opts.hydro.maximumBulkViscosityCoefficient,
        opts.hydro.viscosityDecay, opts.hydro.balsaraSoundSpeedCoefficient};
}

/// Require a finite saved plane acceleration equal to the current scenario setting.
bool zPlaneCheckpointCompatible(double acceleration, Options const& opts) {
    // Inspect the exponent bits because this translation unit permits fast-math.
    uint64_t bits;
    std::memcpy(&bits, &acceleration, sizeof(bits));
    return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL &&
        acceleration == opts.scenario.rayleighTaylor.zPlaneAcceleration;
}

// Active gravity retains its selected order; hydro-only files carry no solver.
// Far code 1 is global FMM; historical mesh solver codes are rejected.
// Global FMM has no separate near-field solver.
std::array<uint64_t, 5> checkpointAlgorithms(Options const& opts) {
    bool const gravityEnabled = opts.gravity.enabled;
    return {1, gravityEnabled ? 1U : 0U, 0U,
        static_cast<uint64_t>(gravityEnabled ? gravity::kBuildFmmOrder : 0),
        usesGlobalIntegration(opts.run) ? 2U : (usesHierarchicalIntegration(opts.run) ? 1U : 0U)};
}

/// Read and validate the runtime solver/integrator selection extension.
std::array<uint64_t, 5> readCheckpointAlgorithms(FILE* fp) {
    std::array<uint64_t, 5> algorithms{};
    FREAD(algorithms.data(), sizeof(uint64_t), algorithms.size(), fp);
    bool const nearExpected = algorithms[1] == 2 || algorithms[1] == 4;
    if (algorithms[0] != 1 || algorithms[1] > 4 || algorithms[2] > 2 || algorithms[3] > 9 ||
        algorithms[4] > 2 || (nearExpected ? algorithms[2] == 0 : algorithms[2] != 0) ||
        (algorithms[1] == 0 ? algorithms[3] != 0 : algorithms[3] < 3)) {
        throw std::runtime_error("Invalid checkpoint algorithm selection extension");
    }
    return algorithms;
}

// DriverParameters historically used its native layout. The new member is appended;
// preserve exactly the old prefix so existing version-three files remain readable.
constexpr size_t kHistoricalDriverBytes = offsetof(DriverParameters, globalTimestep);
static_assert(kHistoricalDriverBytes == offsetof(DriverParameters, itime) + sizeof(TimeType));

/// Write the versioned driver state and enabled physics/integration metadata extensions.
void writeDriverState(FILE* fp, DriverParameters const& params) {
    auto const header = checkpointHeader(getOptions());
    if (fwrite(&header, sizeof(header), 1, fp) != 1) {
        throw std::ios_base::failure("Unable to write checkpoint header");
    }
    auto const algorithms = checkpointAlgorithms(getOptions());
    if (fwrite(algorithms.data(), sizeof(uint64_t), algorithms.size(), fp) != algorithms.size()) {
        throw std::ios_base::failure("Unable to write checkpoint algorithm selection");
    }

    if ((header.flags & kIndependentParticleCounts) != 0U) {
        std::array<uint64_t, 2> const counts{
            getOptions().run.darkMatterParticleCount, getOptions().run.gasParticleCount};
        if (fwrite(counts.data(), sizeof(uint64_t), counts.size(), fp) != counts.size()) {
            throw std::ios_base::failure("Unable to write independent particle counts");
        }
    }
    if (getOptions().hydro.enabled) {
        auto const sphParameters = sphCheckpointParameters(getOptions());
        if (fwrite(sphParameters.data(), sizeof(double), sphParameters.size(), fp) !=
            sphParameters.size()) {
            throw std::ios_base::failure("Unable to write SPH checkpoint parameters");
        }
        // Zero conduction preserves the existing SPH checkpoint layout.
        if ((header.flags & kArtificialConduction) != 0U &&
            fwrite(&getOptions().hydro.conductionCoefficient, sizeof(double), 1, fp) != 1) {
            throw std::ios_base::failure("Unable to write SPH conduction coefficient");
        }
        // Optional extension: preserve the historical block and zero-force layout.
        if ((header.flags & kZPlaneAcceleration) != 0U &&
            fwrite(&getOptions().scenario.rayleighTaylor.zPlaneAcceleration, sizeof(double), 1,
                fp) != 1) {
            throw std::ios_base::failure("Unable to write SPH z-plane acceleration");
        }
        if ((header.flags & kVariableViscosity) != 0U) {
            auto const viscosityParameters = viscosityCheckpointParameters(getOptions());
            if (fwrite(viscosityParameters.data(), sizeof(double), viscosityParameters.size(),
                    fp) != viscosityParameters.size()) {
                throw std::ios_base::failure("Unable to write SPH variable-viscosity parameters");
            }
        }
    }
#if COSMICTIGER_WITH_SPH && COSMICTIGER_ENABLE_CHEMISTRY
    if (getOptions().chemistry.enabled) {
        auto const chemicalParameters = sphChemistryCheckpointParameters();
        if (fwrite(chemicalParameters.data(), sizeof(double), chemicalParameters.size(), fp) !=
            chemicalParameters.size()) {
            throw std::ios_base::failure(
                "Unable to write equilibrium-cooling checkpoint parameters");
        }
    }
#endif
    if (getOptions().starFormation.enabled) {
        auto const parameters = starFormationCheckpointParameters(getOptions());
        if (fwrite(parameters.data(), sizeof(uint64_t), parameters.size(), fp) !=
            parameters.size()) {
            throw std::ios_base::failure("Unable to write star-formation checkpoint parameters");
        }
    }
    if (fwrite(&params, kHistoricalDriverBytes, 1, fp) != 1) {
        throw std::ios_base::failure("Unable to write checkpoint driver state");
    }
    if ((header.flags & kGlobalIntegrator) != 0U) {
        uint64_t const version = 2;
        if (fwrite(&version, sizeof(version), 1, fp) != 1 ||
            fwrite(&params.globalTimestep, sizeof(params.globalTimestep), 1, fp) != 1 ||
            fwrite(&params.globalHydroTimestep, sizeof(params.globalHydroTimestep), 1, fp) != 1) {
            throw std::ios_base::failure("Unable to write global timestep checkpoint extension");
        }
    }
}

/// Validate and restore current or supported legacy driver state and its enabled
/// extensions.
DriverParameters readDriverState(FILE* fp) {
    DriverParameters params{};
    CheckpointHeader header{};
    uint64_t magic = 0;
    FREAD(&magic, sizeof(magic), 1, fp);
    if (fseek(fp, 0, SEEK_SET) != 0) {
        throw std::ios_base::failure("Unable to rewind checkpoint");
    }
    if (magic == kCheckpointMagic) {
        FREAD(&header, sizeof(header), 1, fp);
        unsigned const savedDimensions = 3 - ((header.reserved >> 28) & 3);
        if (savedDimensions != kSpatialDimensions) {
            throw std::runtime_error(
                std::format("Checkpoint spatial dimension {} does not match this build ({})",
                    savedDimensions, kSpatialDimensions));
        }
        bool const compactDimensions = (header.reserved & (1U << 30)) != 0U;
        if (compactDimensions != (kSpatialDimensions < 3)) {
            throw std::runtime_error("Checkpoint particle storage differs: lower-dimensional "
                                     "builds require compact D-component particle records");
        }
        if ((header.flags &
                (kRetiredParticleMesh | kParticleMeshInterval | kParticleMultipoleMesh |
                    kMeshGravityInterval)) != 0U) {
            throw std::runtime_error("Mesh-gravity checkpoints are no longer supported; restart "
                                     "requires a global FMM checkpoint");
        }
#if !COSMICTIGER_WITH_SPH || !COSMICTIGER_ENABLE_CHEMISTRY
        if ((header.flags & kChemistry) != 0U) {
            throw std::runtime_error("Chemistry checkpoints require COSMICTIGER_ENABLE_SPH=ON and "
                                     "COSMICTIGER_ENABLE_CHEMISTRY=ON");
        }
#endif
        if ((header.reserved & kCompactSphLayout) != 0U &&
            (header.reserved & kScalarViscosityLayout) == 0U) {
            throw std::runtime_error(
                "Checkpoint uses the retired SPH viscosity layout and cannot be restarted; only "
                "none and morrisAndMonaghan viscosity remain");
        }
        if (getOptions().hydro.enabled && (header.reserved & kCompactSphLayout) != 0U &&
            (header.reserved & kSphFormationMask) !=
                (checkpointPhysicsCode(getOptions()) & kSphFormationMask)) {
            uint32_t const saved = header.reserved & kSphFormationMask;
            throw std::runtime_error(
                std::format("Checkpoint SPH pressure operator ({}) is no longer supported; SPH "
                            "requires the variational pressure operator",
                    saved == kRetiredMonaghanSph ? "monaghan" : "unknown"));
        }
        if (!checkpointCompatible(header, getOptions())) {
            throw std::runtime_error(
                "Checkpoint physics, particle layout or integrator differs from the requested run");
        }
        if ((header.flags & kRuntimeAlgorithms) != 0U) {
            auto algorithms = readCheckpointAlgorithms(fp);
            if (getOptions().hydro.enabled && !getOptions().gravity.enabled) {
                // Accept old inactive solver selections only after the SPH
                // block below confirms the saved gravity-enabled flag is zero.
                algorithms[1] = algorithms[2] = algorithms[3] = 0;
            }
            if (algorithms != checkpointAlgorithms(getOptions())) {
                throw std::runtime_error("Checkpoint gravity methods, expansion degree, or time "
                                         "integrator differs from the requested run");
            }
        }

        if ((header.flags & kIndependentParticleCounts) != 0U) {
            std::array<uint64_t, 2> counts{};
            FREAD(counts.data(), sizeof(uint64_t), counts.size(), fp);
            if (counts !=
                std::array<uint64_t, 2>{
                    getOptions().run.darkMatterParticleCount, getOptions().run.gasParticleCount}) {
                throw std::runtime_error(
                    "Checkpoint dark-matter or gas particle count differs from the requested run");
            }
        }
        if (getOptions().hydro.enabled) {
            auto sphParameters = sphCheckpointParameters(getOptions());
            FREAD(sphParameters.data(), sizeof(double), sphParameters.size(), fp);
            if (sphParameters != sphCheckpointParameters(getOptions())) {
                throw std::runtime_error("SPH checkpoint parameters differ from the requested run");
            }
            if ((header.flags & kArtificialConduction) != 0U) {
                double conductionCoefficient;
                FREAD(&conductionCoefficient, sizeof(conductionCoefficient), 1, fp);
                if (conductionCoefficient != getOptions().hydro.conductionCoefficient) {
                    throw std::runtime_error("SPH checkpoint hydro.conductionCoefficient differs "
                                             "from the requested run");
                }
            }
            if ((header.flags & kZPlaneAcceleration) != 0U) {
                double acceleration;
                FREAD(&acceleration, sizeof(acceleration), 1, fp);
                if (!zPlaneCheckpointCompatible(acceleration, getOptions())) {
                    throw std::runtime_error(
                        "SPH checkpoint scenario.rayleighTaylor.zPlaneAcceleration differs from "
                        "the requested run");
                }
            }
            if ((header.flags & kVariableViscosity) != 0U) {
                auto viscosityParameters = viscosityCheckpointParameters(getOptions());
                FREAD(viscosityParameters.data(), sizeof(double), viscosityParameters.size(), fp);
                if (viscosityParameters != viscosityCheckpointParameters(getOptions())) {
                    throw std::runtime_error("SPH checkpoint variable-viscosity parameters differ "
                                             "from the requested run");
                }
            }
        }
#if COSMICTIGER_WITH_SPH && COSMICTIGER_ENABLE_CHEMISTRY
        if (getOptions().chemistry.enabled) {
            auto chemicalParameters = sphChemistryCheckpointParameters();
            FREAD(chemicalParameters.data(), sizeof(double), chemicalParameters.size(), fp);
            if (chemicalParameters[0] != sphChemistryCheckpointParameters()[0]) {
                throw std::runtime_error(
                    "Checkpoint particle cooling model is incompatible with GADGET-4 cooling");
            }
            auto finiteCheckpointScalar = [](double value) {
                uint64_t bits;
                std::memcpy(&bits, &value, sizeof(bits));
                return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
            };
            for (double value : chemicalParameters) {
                if (!finiteCheckpointScalar(value)) {
                    throw std::runtime_error(
                        "Checkpoint equilibrium-cooling parameters are not finite");
                }
            }
            if (chemicalParameters != sphChemistryCheckpointParameters()) {
                throw std::runtime_error(
                    "Checkpoint GADGET-4 cooling model or initialization settings differ");
            }
        }
#endif
        if (getOptions().starFormation.enabled) {
            auto parameters = starFormationCheckpointParameters(getOptions());
            FREAD(parameters.data(), sizeof(uint64_t), parameters.size(), fp);
            if (parameters != starFormationCheckpointParameters(getOptions())) {
                throw std::runtime_error("Checkpoint star-formation model, thresholds, or "
                                         "random-stream parameters differ from the requested run");
            }
        }
        FREAD(&params, kHistoricalDriverBytes, 1, fp);
        if ((header.flags & kGlobalIntegrator) != 0U) {
            uint64_t version;
            FREAD(&version, sizeof(version), 1, fp);
            FREAD(&params.globalTimestep, sizeof(params.globalTimestep), 1, fp);
            if (version == 2) {
                FREAD(&params.globalHydroTimestep, sizeof(params.globalHydroTimestep), 1, fp);
            }
            uint64_t bits;
            std::memcpy(&bits, &params.globalTimestep, sizeof(bits));
            if ((version != 1 && version != 2) ||
                (bits & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL ||
                params.globalTimestep < 0.0 || params.itime != 0 ||
                (params.iter > 0 && params.globalTimestep == 0.0)) {
                throw std::runtime_error("Invalid global timestep checkpoint extension");
            }
            std::memcpy(&bits, &params.globalHydroTimestep, sizeof(bits));
            if ((bits & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL ||
                params.globalHydroTimestep < 0.0) {
                throw std::runtime_error("Invalid global hydro timestep checkpoint extension");
            }
#if COSMICTIGER_WITH_SPH
            if (sphMidpointEnabled() && params.iter > 0 && params.globalHydroTimestep == 0.0) {
                throw std::runtime_error("Invalid global hydro timestep checkpoint extension");
            }
#endif
        }

    } else {
        if (!legacyCheckpointCompatible(getOptions())) {
            throw std::runtime_error(
                "Legacy checkpoints support only the original dark-matter hierarchical mode with "
                "the degree-eight Ferrers gravity kernel");
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

/// Exercise driver-state round trips and rejection of incompatible checkpoint modes and
/// layouts.
void checkpointStateTest() {
    constexpr double kScale = 0.5;
    constexpr double kClock = 0.125;
    constexpr double kThermal = 3.0;
    constexpr double kHeating = 0.25;
    constexpr int kRung = 4;
    FILE* fp = tmpfile();
    if (fp == nullptr) {
        throw std::ios_base::failure("Cannot create checkpoint test stream");
    }
    DriverParameters input{};
    input.a = kScale;
    input.tau = kClock;
    input.tauMax = 2.0;
    input.itime = usesGlobalIntegration(getOptions().run) ? 0 : 128;
    input.iter = 1;
    input.maxRung = kRung;
    input.energies.therm = kThermal;
    input.energies.heating = kHeating;
    input.globalTimestep = usesGlobalIntegration(getOptions().run) ? 0.03125 : 0.0;
    input.globalHydroTimestep = usesGlobalIntegration(getOptions().run) ? 0.0625 : 0.0;

    detail::writeDriverState(fp, input);
    rewind(fp);
    auto const restored = detail::readDriverState(fp);
    ALWAYS_ASSERT(
        restored.a == input.a && restored.tau == input.tau && restored.tauMax == input.tauMax);
    ALWAYS_ASSERT(restored.itime == input.itime && restored.maxRung == input.maxRung);
    ALWAYS_ASSERT(restored.energies.therm == kThermal && restored.energies.heating == kHeating);
    ALWAYS_ASSERT(restored.globalTimestep == input.globalTimestep);
    ALWAYS_ASSERT(restored.globalHydroTimestep == input.globalHydroTimestep);
    if (!usesGlobalIntegration(getOptions().run)) {
        // Strip only the new selection extension to reconstruct the exact old
        // version-three driver payload and exercise its unchanged decoder.
        long const length = ftell(fp);
        ALWAYS_ASSERT(length > long(sizeof(detail::CheckpointHeader) + 5 * sizeof(uint64_t)));
        std::vector<unsigned char> bytes(static_cast<size_t>(length));
        rewind(fp);
        ALWAYS_ASSERT(fread(bytes.data(), 1, bytes.size(), fp) == bytes.size());
        auto oldHeader = detail::checkpointHeader(getOptions());
        oldHeader.flags &= ~detail::kRuntimeAlgorithms;
        detail::CheckpointFile oldStream(tmpfile());
        ALWAYS_ASSERT(oldStream != nullptr);
        ALWAYS_ASSERT(fwrite(&oldHeader, sizeof(oldHeader), 1, oldStream.get()) == 1);
        size_t const offset = sizeof(oldHeader) + 5 * sizeof(uint64_t);
        ALWAYS_ASSERT(fwrite(bytes.data() + offset, 1, bytes.size() - offset, oldStream.get()) ==
            bytes.size() - offset);
        rewind(oldStream.get());
        auto const previous = detail::readDriverState(oldStream.get());
        ALWAYS_ASSERT(previous.tau == input.tau && previous.itime == input.itime &&
            previous.energies.heating == input.energies.heating && previous.globalTimestep == 0.0);
    }
    auto changed = getOptions();
    auto const header = detail::checkpointHeader(changed);
    // The algorithm extension distinguishes global gravity from no gravity.
    changed.gravity.enabled = !changed.gravity.enabled;
    ALWAYS_ASSERT(
        detail::checkpointAlgorithms(getOptions()) != detail::checkpointAlgorithms(changed));
    auto hydroOnlyOptions = getOptions();
    hydroOnlyOptions.gravity.enabled = false;
    hydroOnlyOptions.hydro.enabled = true;
    {
        auto const algorithms = detail::checkpointAlgorithms(hydroOnlyOptions);
        ALWAYS_ASSERT(algorithms[1] == 0 && algorithms[2] == 0 && algorithms[3] == 0);
        auto oldHydroOnlyHeader = detail::checkpointHeader(hydroOnlyOptions);
        ALWAYS_ASSERT((oldHydroOnlyHeader.flags & detail::kParticleMultipoleMesh) == 0U);
        oldHydroOnlyHeader.flags |= detail::kParticleMultipoleMesh;
        ALWAYS_ASSERT(!detail::checkpointCompatible(oldHydroOnlyHeader, hydroOnlyOptions));
    }
    changed = getOptions();
    // Reject the retired scalar particle-mesh bits rather than treating their
    // extension bytes as driver state.
    for (uint32_t const retiredFlag : {detail::kRetiredParticleMesh, detail::kParticleMeshInterval,
             detail::kParticleMultipoleMesh, detail::kMeshGravityInterval}) {
        auto retiredPmHeader = detail::checkpointHeader(changed);
        retiredPmHeader.flags &= ~detail::kParticleMultipoleMesh;
        retiredPmHeader.flags |= retiredFlag;
        ALWAYS_ASSERT(!detail::checkpointCompatible(retiredPmHeader, changed));
    }
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
    defaultKernelOptions.hydro.method = "sph";
    ALWAYS_ASSERT(detail::checkpointHeader(defaultKernelOptions).reserved ==
        (defaultKernelOptions.hydro.enabled ? detail::kVariationalSph | detail::kCompactSphLayout |
                    detail::kScalarViscosityLayout | detail::kRosswogViscosityLayout :
                                              0U));
    auto legacyKernelOptions = defaultKernelOptions;
    legacyKernelOptions.hydro.enabled = false;
    legacyKernelOptions.run.hierarchicalGravity = true;
    ALWAYS_ASSERT(detail::legacyCheckpointCompatible(legacyKernelOptions) ==
        (legacyKernelOptions.compute.positionType == "fixed32"));
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
        (legacyDarkMatterOptions.compute.positionType == "fixed32"));
    auto const sphParameters = detail::sphCheckpointParameters(getOptions());
    auto starOptions = getOptions();
    starOptions.hydro.enabled = true;
    starOptions.chemistry.enabled = true;
    starOptions.starFormation.enabled = true;
    auto const starHeader = detail::checkpointHeader(starOptions);
    ALWAYS_ASSERT(detail::checkpointCompatible(starHeader, starOptions));
    starOptions.starFormation.enabled = false;
    ALWAYS_ASSERT(!detail::checkpointCompatible(starHeader, starOptions));
    starOptions.starFormation.enabled = true;
    starOptions.chemistry.enabled = false;
    ALWAYS_ASSERT(!detail::checkpointCompatible(starHeader, starOptions));
    starOptions.chemistry.enabled = true;
    auto const starParameters = detail::starFormationCheckpointParameters(starOptions);
    ALWAYS_ASSERT(starParameters[0] == 3);
    double StarFormation::* const starScalars[] = {&StarFormation::beta,
        &StarFormation::evaporationFactor, &StarFormation::starFormationTimeGyr,
        &StarFormation::coldTemperature, &StarFormation::supernovaTemperature,
        &StarFormation::criticalPhysicalDensity, &StarFormation::criticalOverDensity};
    for (auto member : starScalars) {
        auto altered = starOptions;
        altered.starFormation.*member += 0.125;
        ALWAYS_ASSERT(starParameters != detail::starFormationCheckpointParameters(altered));
    }
    auto alteredStarOptions = starOptions;
    alteredStarOptions.cosmology.heliumFraction += 0.01;
    // The GADGET-4 cooling composition is fixed independently of background Y.
    ALWAYS_ASSERT(starParameters == detail::starFormationCheckpointParameters(alteredStarOptions));
    alteredStarOptions = starOptions;
    alteredStarOptions.cosmology.omegaB += 0.01;
    ALWAYS_ASSERT(starParameters != detail::starFormationCheckpointParameters(alteredStarOptions));
    alteredStarOptions = starOptions;
    alteredStarOptions.cosmology.hubble += 0.01;
    ALWAYS_ASSERT(starParameters != detail::starFormationCheckpointParameters(alteredStarOptions));
    alteredStarOptions = starOptions;
    alteredStarOptions.starFormation.seed ^= uint64_t{1} << 63;
    ALWAYS_ASSERT(starParameters != detail::starFormationCheckpointParameters(alteredStarOptions));
    auto conductionOptions = getOptions();
    conductionOptions.hydro.enabled = true;
    conductionOptions.hydro.conductionCoefficient = 0.0;
    auto const noConductionHeader = detail::checkpointHeader(conductionOptions);
    ALWAYS_ASSERT((noConductionHeader.flags & detail::kArtificialConduction) == 0U);
    conductionOptions.hydro.conductionCoefficient = 0.3;
    ALWAYS_ASSERT(!detail::checkpointCompatible(noConductionHeader, conductionOptions));
    auto const conductionHeader = detail::checkpointHeader(conductionOptions);
    ALWAYS_ASSERT(detail::checkpointCompatible(conductionHeader, conductionOptions));
    conductionOptions.hydro.conductionCoefficient = 0.0;
    ALWAYS_ASSERT(!detail::checkpointCompatible(conductionHeader, conductionOptions));
    // Every SPH checkpoint carries the Balsara factor and evolved alpha.
    auto balsaraOptions = getOptions();
    balsaraOptions.hydro.enabled = true;
    auto const balsaraHeader = detail::checkpointHeader(balsaraOptions);
    ALWAYS_ASSERT((balsaraHeader.flags & detail::kBalsaraSwitch) != 0U);
    ALWAYS_ASSERT((balsaraHeader.flags & detail::kVariableViscosity) != 0U);
    ALWAYS_ASSERT(detail::checkpointCompatible(balsaraHeader, balsaraOptions));
    auto unlimitedHeader = balsaraHeader;
    unlimitedHeader.flags &= ~(detail::kBalsaraSwitch | detail::kVariableViscosity);
    ALWAYS_ASSERT(!detail::checkpointCompatible(unlimitedHeader, balsaraOptions));
    // Programmatic updates validate exactly as CLI/INI parsing does.
    {
        auto const previousOptions = getOptions();
        auto invalidOptions = previousOptions;
        invalidOptions.hydro.minimumBulkViscosityCoefficient = 0.5;
        invalidOptions.hydro.maximumBulkViscosityCoefficient = 0.25;
        bool rejected = false;
        try {
            setOptions(invalidOptions);
        } catch (std::invalid_argument const& error) {
            rejected =
                std::string(error.what()).find("hydro.viscosityCoefficientMax must not be") !=
                std::string::npos;
        }
        ALWAYS_ASSERT(rejected);
        ALWAYS_ASSERT(getOptions().hydro.maximumBulkViscosityCoefficient ==
            previousOptions.hydro.maximumBulkViscosityCoefficient);
    }
    // Variational checkpoints round-trip; retired Monaghan pressure is never reinterpreted.
    auto operatorOptions = getOptions();
    operatorOptions.hydro.enabled = true;
    auto const variationalHeader = detail::checkpointHeader(operatorOptions);
    ALWAYS_ASSERT(
        (variationalHeader.reserved & detail::kSphFormationMask) == detail::kVariationalSph);
    ALWAYS_ASSERT(detail::checkpointCompatible(variationalHeader, operatorOptions));
    auto retiredHeader = variationalHeader;
    retiredHeader.reserved =
        (retiredHeader.reserved & ~detail::kSphFormationMask) | detail::kRetiredMonaghanSph;
    ALWAYS_ASSERT(!detail::checkpointCompatible(retiredHeader, operatorOptions));
    // Older SPH files carry retired viscosity state and must never reach the payload.
    ALWAYS_ASSERT((variationalHeader.reserved & detail::kScalarViscosityLayout) != 0U);
    auto retiredViscosityHeader = variationalHeader;
    retiredViscosityHeader.reserved &= ~detail::kScalarViscosityLayout;
    ALWAYS_ASSERT(!detail::checkpointCompatible(retiredViscosityHeader, operatorOptions));
    // Files from before the Rosswog limiter carry a fixed alpha and no alpha max.
    ALWAYS_ASSERT((variationalHeader.reserved & detail::kRosswogViscosityLayout) != 0U);
    auto unlimitedViscosityHeader = variationalHeader;
    unlimitedViscosityHeader.reserved &= ~detail::kRosswogViscosityLayout;
    ALWAYS_ASSERT(!detail::checkpointCompatible(unlimitedViscosityHeader, operatorOptions));
    auto viscosityOptions = getOptions();
    viscosityOptions.hydro.enabled = true;
    auto const viscosityParameters = detail::viscosityCheckpointParameters(viscosityOptions);
    {
        auto altered = viscosityOptions;
        altered.hydro.vonNeumannRichtmyerCoefficient += 0.5;
        ALWAYS_ASSERT(detail::sphCheckpointParameters(viscosityOptions) !=
            detail::sphCheckpointParameters(altered));
        altered = viscosityOptions;
        altered.hydro.minimumBulkViscosityCoefficient += 0.125;
        ALWAYS_ASSERT(viscosityParameters != detail::viscosityCheckpointParameters(altered));
        altered = viscosityOptions;
        altered.hydro.maximumBulkViscosityCoefficient += 0.125;
        ALWAYS_ASSERT(viscosityParameters != detail::viscosityCheckpointParameters(altered));
        altered = viscosityOptions;
        altered.hydro.viscosityDecay += 0.125;
        ALWAYS_ASSERT(viscosityParameters != detail::viscosityCheckpointParameters(altered));
        altered = viscosityOptions;
        altered.hydro.balsaraSoundSpeedCoefficient += 0.125;
        ALWAYS_ASSERT(viscosityParameters != detail::viscosityCheckpointParameters(altered));
    }
    auto planeOptions = getOptions();
    planeOptions.hydro.enabled = true;
    planeOptions.scenario.rayleighTaylor.zPlaneAcceleration = 0.0;
    auto const noPlaneHeader = detail::checkpointHeader(planeOptions);
    ALWAYS_ASSERT((noPlaneHeader.flags & detail::kZPlaneAcceleration) == 0U);
    auto const noPlaneParameters = detail::sphCheckpointParameters(planeOptions);
    planeOptions.scenario.rayleighTaylor.zPlaneAcceleration = 0.5;
    auto const planeHeader = detail::checkpointHeader(planeOptions);
    ALWAYS_ASSERT((planeHeader.flags & detail::kZPlaneAcceleration) != 0U);
    ALWAYS_ASSERT(!detail::checkpointCompatible(noPlaneHeader, planeOptions));
    ALWAYS_ASSERT(detail::checkpointCompatible(planeHeader, planeOptions));
    ALWAYS_ASSERT(noPlaneParameters == detail::sphCheckpointParameters(planeOptions));
    ALWAYS_ASSERT(detail::zPlaneCheckpointCompatible(0.5, planeOptions));
    ALWAYS_ASSERT(!detail::zPlaneCheckpointCompatible(0.25, planeOptions));
    planeOptions.scenario.rayleighTaylor.zPlaneAcceleration = 0.0;
    ALWAYS_ASSERT(!detail::checkpointCompatible(planeHeader, planeOptions));
    ALWAYS_ASSERT(!detail::zPlaneCheckpointCompatible(0.5, planeOptions));
    changed = getOptions();
    changed.hydro.vonNeumannRichtmyerCoefficient += 0.25;
    ALWAYS_ASSERT(sphParameters != detail::sphCheckpointParameters(changed));
    changed = getOptions();
    changed.hydro.neighborNumber += 1.0;
    ALWAYS_ASSERT(sphParameters != detail::sphCheckpointParameters(changed));
    changed = getOptions();
    changed.hydro.maxSmoothingLength *= 2.0;
    ALWAYS_ASSERT(sphParameters != detail::sphCheckpointParameters(changed));
    changed = getOptions();
    changed.run.numberOfOutputs++;
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
    if (!getOptions().hydro.enabled && !usesCuda(getOptions())) {
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
                ALWAYS_ASSERT(particlesVelocityValue(dim, particle) == float(particle + dim));
            }
            ALWAYS_ASSERT(particlesRung(particle) == char(particle));
        }
        fclose(fp);
        particlesFree();
        particlesDestroy();
        std::cout << std::format("Checkpoint position payload round trip passed ({:d} bits)\n",
                         kPositionBits)
                  << std::flush;
    }
    std::cout << "Checkpoint state and mode validation passed\n" << std::flush;
}

/// Write and synchronize all rank payloads, then publish the complete checkpoint directory
/// atomically.
void writeCheckpoint(DriverParameters params) {
    auto const directory = detail::checkpointDirectory(params.iter);
    auto const stagingDirectory = detail::checkpointStagingDirectory(params.iter);
    if (hpxRank() == 0) {
        terminal::detailPrint("Writing checkpoint %s\n", directory.c_str());
        if (detail::fs::exists(directory)) {
            throw std::runtime_error(
                std::format("Refusing to replace existing checkpoint {}", directory.c_str()));
        }
        if (!detail::fs::create_directory(stagingDirectory)) {
            throw std::runtime_error(
                std::format("Checkpoint staging directory {} already exists; inspect the "
                            "incomplete checkpoint before retrying",
                    stagingDirectory.c_str()));
        }
    }
    Vector<hpx::future<void>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<WriteCheckpointAction>(c, params));
    }
    {
        profile::ScopedPhase profilePhase("writeCheckpoint.local", hpxRank() == 0);
        auto const rankPath = detail::checkpointRankFile(stagingDirectory, hpxRank());
        detail::CheckpointFile file(fopen(rankPath.c_str(), "wb"));
        if (!file) {
            throw std::ios_base::failure(
                std::format("Unable to open {} for writing", rankPath.c_str()));
        }
        detail::writeDriverState(file.get(), params);
        particlesSave(file.get());
        domainsSave(file.get());
#if COSMICTIGER_ENABLE_ROCKSTAR
        if (getOptions().lightcone.enabled) {
            lcSave(file.get());
        }
#endif
        detail::flushAndCloseCheckpointFile(file, rankPath);
    }
    {
        profile::ScopedPhase profilePhase("writeCheckpoint.wait", hpxRank() == 0);
        for (auto& future : futs) {
            future.get();
        }
    }
    if (hpxRank() == 0) {
        profile::ScopedPhase profilePhase("writeCheckpoint.publish");
        detail::writeCheckpointScenario(stagingDirectory);
        detail::writeCheckpointManifest(stagingDirectory, params.iter);
        detail::syncDirectory(stagingDirectory);
        if (detail::fs::exists(directory)) {
            throw std::runtime_error(
                std::format("Refusing to replace existing checkpoint {}", directory.c_str()));
        }
        detail::fs::rename(stagingDirectory, directory);
        detail::syncDirectory(directory.has_parent_path() ? directory.parent_path() : ".");
        std::cout << std::format("  Checkpoint saved: {:s}\n", directory.c_str()) << std::flush;
    }
}

/// Validate the checkpoint set and restore distributed driver, particle, domain, and
/// optional analysis state.
DriverParameters readCheckpoint() {
    DriverParameters params{};
    if (hpxRank() == 0) {
        std::cout << "Reading checkpoint\n" << std::flush;
        profile::ScopedPhase profilePhase("readCheckpoint.validate");
        detail::validateCheckpointDirectory(
            detail::checkpointDirectory(getOptions().checkpoint.readIndex),
            getOptions().checkpoint.readIndex);
    }
    Vector<hpx::future<DriverParameters>> futs;
    for (auto const& c : hpxChildren()) {
        futs.push_back(hpx::async<ReadCheckpointAction>(c));
    }
    {
        profile::ScopedPhase profilePhase("readCheckpoint.local", hpxRank() == 0);
        int const iter = getOptions().checkpoint.readIndex;
        auto const rankPath =
            detail::checkpointRankFile(detail::checkpointDirectory(iter), hpxRank());
        detail::CheckpointFile file(fopen(rankPath.c_str(), "rb"));
        if (!file) {
            throw std::ios_base::failure(
                std::format("Unable to open {} for reading.", rankPath.c_str()));
        }
        params = detail::readDriverState(file.get());
        particlesLoad(file.get());
        domainsLoad(file.get());
#if COSMICTIGER_ENABLE_ROCKSTAR
        if (getOptions().lightcone.enabled) {
            lcLoad(file.get());
        }
#endif
        if (ferror(file.get()) != 0) {
            throw std::ios_base::failure(
                std::format("Unable to read checkpoint file {}", rankPath.c_str()));
        }
        FILE* rawFile = file.release();
        if (fclose(rawFile) != 0) {
            throw std::ios_base::failure(
                std::format("Unable to close checkpoint file {}", rankPath.c_str()));
        }
    }
    {
        profile::ScopedPhase profilePhase("readCheckpoint.wait", hpxRank() == 0);
        for (auto& future : futs) {
            auto const child = future.get();
        }
    }
    if (hpxRank() == 0) {
        std::cout << "Done reading checkpoint\n" << std::flush;
    }
    return params;
}
