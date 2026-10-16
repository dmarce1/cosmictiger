/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026 Dominic C. Marcello
 Distributed under the GNU General Public License, version 2 or later.
 See LICENSE for details.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include <cosmictiger/legacy_options.hpp>

namespace detail {

// Frozen pre-unit-selection layout, independent of the current Options fields.
// Legacy files are native-ABI records; string bytes must never become live strings.
struct LegacyOptionsLayout {
    int bucketSize;
    int nfour;
    bool cuda;
    bool closePack;
    bool useGlass;
    bool saveForce;
    bool doLc;
    bool doPower;
    bool doGroups;
    bool doTracers;
    bool doSlice;
    bool doViews;
    bool outputSilo;
    int outputSiloInterval;
    std::string outputDirectory;
    double outputXMin;
    double outputXMax;
    double outputYMin;
    double outputYMax;
    double outputZMin;
    double outputZMax;
    bool twolpt;
    bool usePowerFile;
    bool plummer;
    double plummerR;
    bool createGlass;
    int readCheck;
    int tracerCount;
    int particlesPerDimension;
    int treeCacheLineSize;
    int treeAllocLineSize;
    int particleCacheLineSize;
    int checkFreq;
    int maxIter;
    int minrung;
    int viewSize;
    int lcMinGroup;
    int lcMapSize;
    int minGroup;
    int sliceRes;
    int numberOfSteps;
    double lcB;
    double sliceSize;
    double linkingLength;
    double hsoft;
    double gm;
    double eta;
    double codeToS;
    double codeToCm;
    double codeToG;
    double omegaM;
    double omegaR;
    double z0;
    double z1;
    double hubble;
    double sigma8;
    double theta;
    double omegaB;
    double omegaLam;
    double omegaC;
    double omegaGam;
    double omegaNu;
    double thetaCmb;
    double ns;
    double y0;
    double neff;
    double omegaK;
    size_t numberOfParticles;
    std::string configFile;
    std::string test;
    std::string lcDir;
    std::string gadget4Restart;

    bool htime{};
    bool sph{};
    bool yreflect{};
    int viscType{};
    bool stars{};
    bool chem{};
    bool conduction{};
    bool vsoft{};
    bool gravity{};
    int glass{};
    double kernel{};
    bool diffusion{};
    int sphBucketSize{};
    double alpha0{};
    double alpha1{};
    double beta{};
    double viscosityEpsilon{};
    double alphaDecay{};
    double darkMatterNeighborNumber{};
    double sphNeighborNumber{};
    double hmin{};
    double hmax{};
    double damping{};
    double sigma8C{};
    double darkMatterMass{};
    double sphMass{};
    double cfl{};
    double rho0B{};
    double rho0C{};
    double gamma{};
    double yAcceleration{};
    double centralAcceleration{};
    double hcentral{};
};

static_assert(sizeof(LegacyOptionsLayout) == legacy_options::kLightconeOptionsBytes);
static_assert(offsetof(LegacyOptionsLayout, codeToCm) == offsetof(Options, codeToCm));
static_assert(offsetof(LegacyOptionsLayout, codeToG) == offsetof(Options, codeToG));
static_assert(offsetof(LegacyOptionsLayout, codeToS) == offsetof(Options, codeToS));
static_assert(offsetof(LegacyOptionsLayout, hubble) == offsetof(Options, hubble));
static_assert(offsetof(LegacyOptionsLayout, omegaM) == offsetof(Options, omegaM));
static_assert(offsetof(LegacyOptionsLayout, hsoft) == offsetof(Options, hsoft));
static_assert(offsetof(LegacyOptionsLayout, lcB) == offsetof(Options, lcB));

void require(bool condition, char const* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct StreamCloser {
    void operator()(FILE* stream) const {
        std::fclose(stream);
    }
};

using Stream = std::unique_ptr<FILE, StreamCloser>;

Stream makeStream() {
    Stream stream(std::tmpfile());
    require(bool(stream), "cannot create legacy options test stream");
    return stream;
}

void testLegacyRead() {
    std::array<unsigned char, sizeof(LegacyOptionsLayout)> bytes;
    // Invalid pointer and string representations prove they are ignored rather
    // than deserialized, copied into live Options, or passed to destructors.
    bytes.fill(0xa5);
    auto const writeDouble = [&bytes](size_t offset, double value) {
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
    };
    writeDouble(offsetof(LegacyOptionsLayout, codeToCm), 8.0);
    writeDouble(offsetof(LegacyOptionsLayout, codeToG), 32.0);
    writeDouble(offsetof(LegacyOptionsLayout, codeToS), 2.0);
    writeDouble(offsetof(LegacyOptionsLayout, hubble), 0.7);
    writeDouble(offsetof(LegacyOptionsLayout, omegaM), 0.3);
    writeDouble(offsetof(LegacyOptionsLayout, hsoft), 0.125);
    writeDouble(offsetof(LegacyOptionsLayout, lcB), 0.28);
    auto stream = makeStream();
    constexpr int kGroups = 3;
    constexpr uint64_t kNextRecord = 0x123456789abcdef0ULL;
    require(std::fwrite(&kGroups, sizeof(kGroups), 1, stream.get()) == 1, "write group count");
    require(std::fwrite(bytes.data(), 1, bytes.size(), stream.get()) == bytes.size(),
        "write old header");
    require(
        std::fwrite(&kNextRecord, sizeof(kNextRecord), 1, stream.get()) == 1, "write next record");
    std::rewind(stream.get());
    int groups{};
    require(std::fread(&groups, sizeof(groups), 1, stream.get()) == 1 && groups == kGroups,
        "read group count");
    Options options{};
    require(legacy_options::readLightconeOptions(stream.get(), options), "read legacy header");
    require(options.codeToCm == 8.0 && options.codeToG == 32.0 && options.codeToS == 2.0,
        "restore base unit scales");
    require(options.hubble == 0.7 && options.omegaM == 0.3 && options.hsoft == 0.125 &&
            options.lcB == 0.28,
        "restore cosmology and linking parameters");
    require(options.outputDirectory.empty() && options.configFile.empty() && options.test.empty() &&
            options.lcDir.empty() && options.gadget4Restart.empty(),
        "ignore legacy raw string objects");
    require(options.outputUnits == "code" && options.outputFrame == "comoving",
        "new options retain initialized defaults");
    require(std::ftell(stream.get()) == long(sizeof(kGroups) + sizeof(LegacyOptionsLayout)),
        "consume exactly the old header size");
    uint64_t nextRecord{};
    require(std::fread(&nextRecord, sizeof(nextRecord), 1, stream.get()) == 1 &&
            nextRecord == kNextRecord,
        "leave the next compressed record intact");

    auto truncated = makeStream();
    require(std::fwrite(bytes.data(), 1, bytes.size() - 1, truncated.get()) == bytes.size() - 1,
        "write truncated header");
    std::rewind(truncated.get());
    options.outputUnits = "cgs";
    require(
        !legacy_options::readLightconeOptions(truncated.get(), options), "reject truncated header");
    require(options.codeToCm == 8.0 && options.outputUnits == "cgs",
        "failed read does not alter options");
}

}    // namespace detail

int main() {
    try {
        detail::testLegacyRead();
        std::puts("Legacy lightcone options layout, scalar read, and cursor checks passed");
        return 0;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
