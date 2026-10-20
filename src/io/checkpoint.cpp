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

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <system_error>

#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

#include <cosmictiger/analysis/lightcone.hpp>
#include <cosmictiger/core/driver.hpp>
#include <cosmictiger/core/hpx.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/core/profiler.hpp>
#include <cosmictiger/io/checkpoint.hpp>
#include <cosmictiger/io/output.hpp>
#include <cosmictiger/math/kernel.hpp>
#include <cosmictiger/particles/domain.hpp>
#include <cosmictiger/particles/particles.hpp>
#include <cosmictiger/sph/sph.hpp>
#include <cosmictiger/sph/sph_chemistry.hpp>
#include <cosmictiger/sph/sph_particles.hpp>

HPX_PLAIN_ACTION(writeCheckpoint, WriteCheckpointAction);
HPX_PLAIN_ACTION(readCheckpoint, ReadCheckpointAction);

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
        (opts.analysis.tracers ? kTracers : 0U) | (opts.chemistry.enabled ? kChemistry : 0U);
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
    return !opts.hydro.enabled && opts.run.hierarchicalGravity &&
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
        (!opts.hydro.enabled && header.version == kLegacyDarkMatterCheckpointVersion);
    return supportedVersion && header.flags == checkpointFlags(opts) &&
        header.reserved == checkpointPhysicsCode(opts) &&
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

void writeDriverState(FILE* fp, DriverParameters const& params) {
    auto const header = checkpointHeader(getOptions());
    if (fwrite(&header, sizeof(header), 1, fp) != 1) {
        THROW_ERROR("Unable to write checkpoint header\n");
    }
    if (getOptions().hydro.enabled) {
        auto const sphParameters = sphCheckpointParameters(getOptions());
        if (fwrite(sphParameters.data(), sizeof(double), sphParameters.size(), fp) !=
            sphParameters.size()) {
            THROW_ERROR("Unable to write SPH checkpoint parameters\n");
        }
    }
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
        if (!checkpointCompatible(header, getOptions())) {
            THROW_ERROR("Checkpoint physics, particle layout or integrator differs from the "
                        "requested run\n");
        }
        if (getOptions().hydro.enabled) {
            auto sphParameters = sphCheckpointParameters(getOptions());
            FREAD(sphParameters.data(), sizeof(double), sphParameters.size(), fp);
            if (sphParameters != sphCheckpointParameters(getOptions())) {
                THROW_ERROR("SPH checkpoint parameters differ from the requested run\n");
            }
        }
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
        FREAD(&params, sizeof(params), 1, fp);
        if (getOptions().chemistry.enabled &&
            (sourcePhase[0] != (params.tau == 0.0 ? 0.0 : 1.0) ||
                sourcePhase[1] !=
                    std::ldexp(params.tauMax / getOptions().run.numberOfSteps,
                        -1 - getOptions().chemistry.synchronizedRung))) {
            THROW_ERROR("Checkpoint chemistry source phase is inconsistent\n");
        }
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
    ALWAYS_ASSERT(detail::legacyCheckpointCompatible(legacyKernelOptions));
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
    ALWAYS_ASSERT(detail::checkpointCompatible(legacyDarkMatterHeader, legacyDarkMatterOptions));
    auto const sphParameters = detail::sphCheckpointParameters(getOptions());
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
    PRINT("Checkpoint state and mode validation passed\n");
}

void writeCheckpoint(DriverParameters params) {
    profilerEnter(__FUNCTION__);
    auto const directory = detail::checkpointDirectory(params.iter);
    auto const stagingDirectory = detail::checkpointStagingDirectory(params.iter);
    if (hpxRank() == 0) {
        PRINT("Writing checkpoint %s\n", directory.c_str());
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
    if (getOptions().lightcone.enabled) {
        lcSave(file.get());
    }
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
        PRINT("Done writing checkpoint %s\n", directory.c_str());
    }
    profilerExit();
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
    if (getOptions().lightcone.enabled) {
        lcLoad(file.get());
    }
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
