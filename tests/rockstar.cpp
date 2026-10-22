#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <cosmictiger/analysis/rockstar.hpp>
#include <cosmictiger/core/options.hpp>
#include <cosmictiger/cosmology/cosmology.hpp>
#include <cosmictiger/gravity/gravity.hpp>
#include <cosmictiger/math/constants.hpp>

// These production stages are also exercised before catalogue finalization so
// that later unbinding or seed merging cannot hide a connectivity regression.
int rockstarFormTree(DeviceVector<RockstarTree>& trees, DeviceVector<RockstarParticle>& particles);
void rockstarFindSubgroups(DeviceVector<RockstarTree>& trees,
    DeviceVector<RockstarParticle>& particles, float linkingLength, int& nextId, bool gpu);
Vector<Subgroup> rockstarSeeds(DeviceVector<RockstarParticle> particles, int& nextId,
    float positionFactor, float velocityFactor, float scale, bool gpu, int depth);

namespace detail {

Vector<ParticleData> makeRockstarSnapshot(int count, bool planar);

bool rockstarTestFinite(double value) {
    return (std::bit_cast<uint64_t>(value) & UINT64_C(0x7ff0000000000000)) !=
        UINT64_C(0x7ff0000000000000);
}

void requireRockstar(bool condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error("ROCKSTAR: " + message);
    }
}

void requireRockstarClose(double actual, double expected, double relative,
    std::string const& message, double absolute = 1.e-12) {
    requireRockstar(rockstarTestFinite(actual) &&
            std::abs(actual - expected) <= absolute + relative * std::abs(expected),
        message + ": got " + std::to_string(actual) + ", expected " + std::to_string(expected));
}

struct RockstarHalfField {
    char const* name;
    Float16 RockstarRecord::* member;
};

constexpr std::array<RockstarHalfField, 31> kRockstarHalfFields{{
    {"zValue", &RockstarRecord::zValue},
    {"rmax", &RockstarRecord::rmax},
    {"rvir", &RockstarRecord::rvir},
    {"rvmax", &RockstarRecord::rvmax},
    {"rs", &RockstarRecord::rs},
    {"rkly", &RockstarRecord::rkly},
    {"sigX", &RockstarRecord::sigX},
    {"deltaX", &RockstarRecord::deltaX},
    {"toW", &RockstarRecord::toW},
    {"sigV", &RockstarRecord::sigV},
    {"sigVb", &RockstarRecord::sigVb},
    {"vcircMax", &RockstarRecord::vcircMax},
    {"deltaV", &RockstarRecord::deltaV},
    {"boa", &RockstarRecord::boa},
    {"coa", &RockstarRecord::coa},
    {"ax", &RockstarRecord::ax},
    {"ay", &RockstarRecord::ay},
    {"az", &RockstarRecord::az},
    {"boa500", &RockstarRecord::boa500},
    {"coa500", &RockstarRecord::coa500},
    {"ax500", &RockstarRecord::ax500},
    {"ay500", &RockstarRecord::ay500},
    {"az500", &RockstarRecord::az500},
    {"lambda", &RockstarRecord::lambda},
    {"lambdaB", &RockstarRecord::lambdaB},
    {"vxc", &RockstarRecord::vxc},
    {"vyc", &RockstarRecord::vyc},
    {"vzc", &RockstarRecord::vzc},
    {"vxb", &RockstarRecord::vxb},
    {"vyb", &RockstarRecord::vyb},
    {"vzb", &RockstarRecord::vzb},
}};

class RockstarCatalogueDirectory {
    std::filesystem::path path_;

public:
    RockstarCatalogueDirectory() {
        std::random_device random;
        for (int attempt = 0; attempt < 16; ++attempt) {
            auto const candidate = std::filesystem::temp_directory_path() /
                ("cosmictiger_rockstar_roundtrip_" + std::to_string(random()) + "_" +
                    std::to_string(random()));
            if (std::filesystem::create_directory(candidate)) {
                path_ = candidate;
                return;
            }
        }
        throw std::runtime_error("ROCKSTAR: unable to create catalogue test directory");
    }

    ~RockstarCatalogueDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    std::filesystem::path const& path() const {
        return path_;
    }
};

void requireRockstarRecordEqual(
    RockstarRecord const& actual, RockstarRecord const& expected, std::string const& context) {
    requireRockstar(actual.id.id == expected.id.id && actual.pid == expected.pid,
        context + ": catalogue or host ID changed");
    std::array<double RockstarRecord::*, 3> const positions{
        &RockstarRecord::x, &RockstarRecord::y, &RockstarRecord::z};
    for (auto const member : positions) {
        requireRockstar(
            std::bit_cast<uint64_t>(actual.*member) == std::bit_cast<uint64_t>(expected.*member),
            context + ": position changed");
    }
    std::array<float RockstarRecord::*, 8> const properties{&RockstarRecord::mvir,
        &RockstarRecord::e, &RockstarRecord::jx, &RockstarRecord::jy, &RockstarRecord::jz,
        &RockstarRecord::m200, &RockstarRecord::m500, &RockstarRecord::m2500};
    for (auto const member : properties) {
        requireRockstar(
            std::bit_cast<uint32_t>(actual.*member) == std::bit_cast<uint32_t>(expected.*member),
            context + ": float property changed");
    }
    for (auto const& field : kRockstarHalfFields) {
        auto const actualHalf = actual.*field.member;
        auto const expectedHalf = expected.*field.member;
        requireRockstar(actualHalf.bits == expectedHalf.bits,
            context + ": IEEE half bits changed for " + field.name);
        requireRockstar(std::bit_cast<uint64_t>(float162double(actualHalf)) ==
                std::bit_cast<uint64_t>(float162double(expectedHalf)),
            context + ": decoded IEEE half changed for " + field.name);
    }
}

void checkRockstarCatalogueRoundtrip(
    RockstarRecord const* records, size_t count, std::string const& context) {
    // These are the existing native-layout offsets, before the IEEE conversion.
    static_assert(sizeof(RockstarRecord) == 128);
    static_assert(offsetof(RockstarRecord, pid) == 64);
    static_assert(offsetof(RockstarRecord, zValue) == 66);
    static_assert(offsetof(RockstarRecord, vzb) == 126);
    RockstarCatalogueDirectory directory;
    auto const filename = (directory.path() / "catalogue.dat").string();
    rockstarWriteCatalogue(filename, records, count);
    requireRockstar(std::filesystem::file_size(filename) == count * sizeof(RockstarRecord),
        context + ": catalogue record count or native layout changed");
    std::unique_ptr<FILE, void (*)(FILE*)> input(
        std::fopen(filename.c_str(), "rb"), [](FILE* file) { std::fclose(file); });
    requireRockstar(bool(input), context + ": cannot reopen written catalogue");
    std::vector<RockstarRecord> restored(count);
    requireRockstar(count == 0 ||
            std::fread(restored.data(), sizeof(RockstarRecord), count, input.get()) == count,
        context + ": incomplete catalogue read");
    requireRockstar(std::fgetc(input.get()) == EOF && !std::ferror(input.get()),
        context + ": trailing bytes or catalogue read failure");
    for (size_t i = 0; i < count; ++i) {
        requireRockstarRecordEqual(
            restored[i], records[i], context + " record " + std::to_string(i));
    }
    std::ifstream format(filename + ".format");
    requireRockstar(bool(format), context + ": missing format metadata");
    std::map<std::string, std::string> metadata;
    std::string key;
    std::string value;
    while (format >> key) {
        requireRockstar(bool(format >> value), context + ": incomplete format metadata entry");
        requireRockstar(metadata.emplace(key, value).second,
            context + ": duplicate format metadata key " + key);
    }
    requireRockstar(format.eof() && !format.bad(), context + ": unreadable format metadata");
    requireRockstar(metadata.size() == 5 && metadata["format_version"] == "2" &&
            metadata["float16"] == "ieee754_binary16" &&
            metadata["record_size"] == std::to_string(sizeof(RockstarRecord)) &&
            metadata["record_count"] == std::to_string(count) &&
            metadata["byte_order"] ==
                (std::endian::native == std::endian::little ? "little" : "big"),
        context + ": incorrect IEEE catalogue format metadata");
}

void checkRockstarCataloguePublication() {
    RockstarCatalogueDirectory directory;
    auto const filename = (directory.path() / "catalogue.dat").string();
    auto const requireCount = [&](size_t count) {
        requireRockstar(std::filesystem::file_size(filename) == count * sizeof(RockstarRecord),
            "rewritten catalogue has stale records");
        std::ifstream format(filename + ".format");
        std::map<std::string, std::string> metadata;
        std::string key;
        std::string value;
        while (format >> key >> value) {
            metadata.emplace(key, value);
        }
        requireRockstar(metadata["format_version"] == "2" &&
                metadata["float16"] == "ieee754_binary16" &&
                metadata["record_count"] == std::to_string(count),
            "rewritten catalogue has missing or stale format metadata");
    };
    RockstarRecord const record{};
    rockstarWriteCatalogue(filename, &record, 1);
    requireCount(1);
    rockstarWriteCatalogue(filename, nullptr, 0);
    requireCount(0);
    requireRockstar(
        std::filesystem::remove(filename) && std::filesystem::create_directory(filename),
        "unable to prepare catalogue write-failure fixture");
    bool failed = false;
    try {
        rockstarWriteCatalogue(filename, &record, 1);
    } catch (std::runtime_error const&) {
        failed = true;
    }
    requireRockstar(failed, "catalogue write failure was not reported");
    requireRockstar(!std::filesystem::exists(filename + ".format"),
        "failed catalogue write left a stale format marker");
}

void checkRockstarCatalogueHalfBoundaries() {
    struct HalfCase {
        uint16_t bits;
        double value;
    };
    std::array<HalfCase, 13> const cases{{
        {0x0000, 0.0},
        {0x8000, -0.0},
        {0x0001, 0x1p-24},
        {0x8001, -0x1p-24},
        {0x03ff, 1023.0 * 0x1p-24},
        {0x0400, 0x1p-14},
        {0x3c00, 1.0},
        {0x3c01, 1.0 + 0x1p-10},
        {0xc000, -2.0},
        {0x7bff, 65504.0},
        {0xfbff, -65504.0},
        {0x7c00, std::numeric_limits<double>::infinity()},
        {0xfc00, -std::numeric_limits<double>::infinity()},
    }};
    std::array<RockstarRecord, 14> records{};
    for (size_t i = 0; i < records.size(); ++i) {
        auto& record = records[i];
        record.id.fofId = UINT64_C(0xabcdef123456) + i;
        record.id.haloId = 32767 - i;
        record.id.sys = i % 2;
        record.pid = int(i) - 1;
        record.x = 1.0 + i * 0x1p-40;
        record.y = -0.0;
        record.z = -3.25;
        record.mvir = 1.25f;
        record.e = -2.5f;
        record.jx = -0.0f;
        record.jy = 0.125f;
        record.jz = -0.25f;
        record.m200 = 2.0f;
        record.m500 = 1.5f;
        record.m2500 = 0.75f;
        for (size_t fieldIndex = 0; fieldIndex < kRockstarHalfFields.size(); ++fieldIndex) {
            auto const& field = kRockstarHalfFields[fieldIndex];
            size_t const caseIndex = (i + fieldIndex) % records.size();
            auto& half = record.*field.member;
            if (caseIndex < cases.size()) {
                half = double2float16(cases[caseIndex].value);
                requireRockstar(half.bits == cases[caseIndex].bits,
                    std::string("catalogue known IEEE encoding for ") + field.name);
                requireRockstar(std::bit_cast<uint64_t>(float162double(half)) ==
                        std::bit_cast<uint64_t>(cases[caseIndex].value),
                    std::string("catalogue known IEEE decoding for ") + field.name);
            } else {
                // Raw I/O preserves even a NaN payload; production packing rejects nonfinite
                // values.
                half.bits = 0x7e55;
            }
        }
    }
    checkRockstarCatalogueRoundtrip(records.data(), records.size(), "IEEE boundary catalogue");
    checkRockstarCatalogueRoundtrip(records.data(), 0, "empty catalogue");
    for (double value : {65505.0, -65505.0}) {
        bool overflow = false;
        try {
            rockstarEncodeHalf(value, "vxc");
        } catch (std::overflow_error const& error) {
            overflow = std::string(error.what()).find("vxc") != std::string::npos;
        }
        requireRockstar(overflow, "catalogue overflow must identify the offending field");
    }
    for (double value : {std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        bool rejected = false;
        try {
            rockstarEncodeHalf(value, "rvir");
        } catch (std::domain_error const& error) {
            rejected = std::string(error.what()).find("rvir") != std::string::npos;
        }
        requireRockstar(
            rejected, "catalogue nonfinite rejection must identify the offending field");
    }
}

RockstarParticle rockstarTestParticle(float x, float y, float z, int index) {
    RockstarParticle particle{};
    particle.x = x;
    particle.y = y;
    particle.z = z;
    particle.index = index;
    particle.identity = index + 1;
    particle.subgroup = 1;
    return particle;
}

Subgroup rockstarTestGroup(int count, float center, int id, float width = 1.e-4f) {
    Subgroup group;
    group.id = id;
    group.x = center;
    group.y = group.z = 0.0f;
    group.sigmaXMin = 0.0f;
    for (int i = 0; i < count; ++i) {
        auto particle = rockstarTestParticle(center, 0.0f, 0.0f, 1000 * id + i);
        particle.xValue[(i / 2) % 3] += (i % 2 ? -width : width);
        particle.subgroup = id;
        group.parts.pushBack(particle);
    }
    rockstarUpdateStatistics(group, 1.0f, 1.0f, 1.0f);
    return group;
}

double rockstarTestVirialDensity(double scale) {
    auto const& options = getOptions();
    double const omega = options.cosmology.omegaM /
        (options.cosmology.omegaM + options.cosmology.omegaLambda * scale * scale * scale);
    double const x = omega - 1.0;
    return options.initial.numberOfParticles * (18.0 * M_PI * M_PI + 82.0 * x - 39.0 * x * x) /
        omega;
}

double rockstarTestRadius(int count, double density) {
    return std::cbrt(3.0 * count / (4.0 * M_PI * density));
}

std::vector<double> directRockstarPotentials(Subgroup const& group) {
    auto const& options = getOptions();
    std::vector<double> phi(group.parts.size(), 0.0);
    for (int i = 0; i < group.parts.size(); ++i) {
        for (int j = 0; j < group.parts.size(); ++j) {
            if (i == j) {
                continue;
            }
            double r2 = 0.0;
            for (int dim = 0; dim < NDIM; ++dim) {
                double const dx = double(group.parts[i].xValue[dim]) - group.parts[j].xValue[dim];
                r2 += dx * dx;
            }
            phi[i] -= options.gravity.gm / options.gravity.supportRadius *
                radial_kernel::potential(
                    selectedGravityKernel(), std::sqrt(r2) / options.gravity.supportRadius);
        }
    }
    return phi;
}

void checkRockstarUnbindingOrder() {
    Subgroup group;
    group.id = 1;
    group.x = group.y = group.z = 0.0f;
    group.vxb = group.vyb = group.vzb = 0.0f;
    for (int pair = 0; pair < 12; ++pair) {
        float const radius = pair < 8 ? 0.0005f * (pair + 1) : 0.02f + 0.005f * (pair - 8);
        for (int sign : {-1, 1}) {
            group.parts.pushBack(rockstarTestParticle(sign * radius, 0, 0, group.parts.size()));
        }
    }
    auto const phi = directRockstarPotentials(group);
    std::set<int> expected;
    for (int i = 0; i < group.parts.size(); ++i) {
        bool const escaping = i < 2 || i >= 22;
        double const speed = std::sqrt(-2.0 * phi[i]) * (escaping ? 1.2 : 0.5);
        group.parts[i].vx = (i % 2 ? -speed : speed);
        if (!escaping) {
            expected.insert(i);
        }
    }
    Vector<RockstarParticle> removed;
    requireRockstar(rockstarHaloBound(group, removed, 1.0f), "bound majority rejected");
    std::set<int> actual;
    for (auto const& particle : group.parts) {
        actual.insert(particle.index);
    }
    requireRockstar(actual == expected && removed.size() == 4,
        "single-pass removal detached particle IDs from their frozen potentials");
}

void checkRockstarSphericalOverdensities() {
    auto makeGroup = [] {
        Subgroup group;
        group.id = 1;
        group.x = group.y = group.z = 0.0f;
        group.sigmaXMin = 0.0f;
        for (float radius : {0.0005f, 0.0013f, 0.0025f, 0.0041f, 0.02f}) {
            for (int i = 0; i < 12; ++i) {
                auto particle = rockstarTestParticle(0, 0, 0, group.parts.size());
                particle.xValue[(i / 2) % 3] = (i % 2 ? -radius : radius);
                group.parts.pushBack(particle);
            }
        }
        return group;
    };
    for (float scale : {1.0f, 0.5f}) {
        Vector<Subgroup> groups;
        groups.push_back(makeGroup());
        rockstarUpdateStatistics(groups[0], 1.0f, 1.0f, scale);
        rockstarFinalizeSubgroups(groups, scale);
        requireRockstar(groups.size() == 1, "analytic shell halo disappeared");
        auto const& group = groups[0];
        requireRockstarClose(group.massVir, 36.0, 0.0, "virial SO shell count");
        requireRockstarClose(group.mass200, scale == 1.0f ? 24.0 : 36.0, 0.0,
            "critical 200 SO shell count at epoch");
        requireRockstarClose(group.mass500, 24.0, 0.0, "critical 500 SO shell count");
        requireRockstarClose(group.mass2500, 12.0, 0.0, "critical 2500 SO shell count");
        requireRockstarClose(group.rVir, rockstarTestRadius(36, rockstarTestVirialDensity(scale)),
            2.e-6, "epoch-dependent virial radius");
        Array<double, NDIM> const origin{};
        auto const record = rockstarMakeRecord(groups, 0, scale, origin);
        requireRockstarClose(record.mvir, 36.0, 2.e-6, "record virial mass conversion");
        requireRockstarClose(record.m200, group.mass200, 2.e-6, "record critical SO mass");
        requireRockstarClose(
            float162double(record.zValue), 1.0 / scale - 1.0, 1.e-3, "record redshift");
    }
}

void checkRockstarCoreAperture() {
    auto group = rockstarTestGroup(100, 0.0f, 1, 0.001f);
    group.parts[0].x = group.parts[0].y = group.parts[0].z = 0.0f;
    for (int i = 0; i < group.parts.size(); ++i) {
        group.parts[i].vx = i == 0 ? 1.0f : -1.0f;
    }
    rockstarUpdateStatistics(group, 1.0f, 1.0f, 1.0f);
    requireRockstarClose(
        group.vxc, 1.0, 0.0, "core velocity must use 0.1 Rvir, not ten percent of particles");
}

void checkRockstarHierarchyAndHistory() {
    Vector<Subgroup> groups;
    groups.push_back(rockstarTestGroup(30, 0.0f, 1));
    groups.push_back(rockstarTestGroup(20, 0.02f, 2));
    groups.push_back(rockstarTestGroup(10, 0.019f, 3));
    rockstarFinalizeSubgroups(groups, 1.0f);
    requireRockstar(groups.size() == 3, "separate internal hierarchy halos disappeared");
    requireRockstar(groups[1].internalParent == 0 && groups[2].internalParent == 1,
        "internal hosts must use closest larger phase-space halo");
    requireRockstar(groups[1].parent == -1 && groups[2].parent == 1,
        "output SO hosts must be distinct from internal mass hierarchy");
    requireRockstarClose(
        groups[0].massVir, 30, 0.0, "distant internal children must remain outside host SO mass");

    groups.clear();
    groups.push_back(rockstarTestGroup(30, 0.0f, 1));
    groups.push_back(rockstarTestGroup(40, 0.0004f, 2));
    for (int i = 0; i < 2; ++i) {
        for (auto& particle : groups[i].parts) {
            particle.previousHalo = i == 0 ? 100 : 200;
            particle.previousHost = i == 0 ? NO_GROUP : 100;
        }
    }
    rockstarFinalizeSubgroups(groups, 1.0f);
    requireRockstar(groups.size() == 2, "temporal fixture halos disappeared");
    requireRockstar(groups[0].internalParent == -1 && groups[1].internalParent == 0,
        "previous host identity lost when exclusive particle counts reverse");
    Array<double, NDIM> const origin{};
    auto const entry = rockstarMakeGroupEntry(groups, 1, 1.0f, origin);
    requireRockstar(entry.parents.at(200) == 40, "particle overlap history was not retained");
}

void checkRockstarMajorMerger() {
    float const offset = 0.125f * getOptions().gravity.supportRadius;
    Vector<Subgroup> groups;
    groups.push_back(rockstarTestGroup(36, -offset, 1, 0.0f));
    groups.push_back(rockstarTestGroup(12, offset, 2, 0.0f));
    auto const phi = directRockstarPotentials(groups[1]);
    float const speed = std::sqrt(3.0 * -phi[0]);
    for (int i = 0; i < groups[1].parts.size(); ++i) {
        groups[1].parts[i].vx = i % 2 ? speed : -speed;
    }
    rockstarUpdateStatistics(groups[1], 1.0f, 1.0f, 1.0f);
    auto isolated = groups[1];
    Vector<RockstarParticle> removed;
    requireRockstar(!rockstarHaloBound(isolated, removed, 1.0f),
        "major-merger fixture is not unbound in its isolated potential");
    rockstarFinalizeSubgroups(groups, 1.0f);
    requireRockstar(groups.size() == 2 && groups[1].parts.size() == 12,
        "exact 1:3 merger must retain particles bound in the combined potential");

    // A historical host may now be the smaller halo. Reversing host labels
    // must not turn the same 1:4 mass ratio into a major merger.
    groups.clear();
    groups.push_back(rockstarTestGroup(12, -offset, 1, 0.0f));
    groups.push_back(rockstarTestGroup(48, offset, 2, 0.0f));
    auto const childPhi = directRockstarPotentials(groups[1]);
    float const escapingSpeed = std::sqrt(2.16 * -childPhi[0]);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < groups[i].parts.size(); ++j) {
            auto& particle = groups[i].parts[j];
            particle.previousHalo = i == 0 ? 100 : 200;
            particle.previousHost = i == 0 ? NO_GROUP : 100;
            if (i == 1) {
                particle.vx = j % 2 ? escapingSpeed : -escapingSpeed;
            }
        }
        rockstarUpdateStatistics(groups[i], 1.0f, 1.0f, 1.0f);
    }
    rockstarFinalizeSubgroups(groups, 1.0f);
    requireRockstar(groups.size() == 1 && groups[0].parts.size() == 60,
        "temporal host reversal incorrectly promoted a 1:4 pair to a major merger");
}

void checkRockstarPrunedDonation() {
    Vector<Subgroup> groups;
    groups.push_back(rockstarTestGroup(100, 0.0f, 100));
    groups.push_back(rockstarTestGroup(12, 0.002f, 2));
    groups.push_back(rockstarTestGroup(10, 0.001f, 3));
    // A stripped candidate has lost its central particles, but its surviving
    // exclusive material still lies inside the host. Preserve its former peak
    // when checking whether it remains a resolved SO object.
    groups[1].x = 0.02f;
    groups[1].sigmaXMin = 0.0f;
    groups[0].internalParent = -1;
    groups[1].internalParent = 0;
    groups[2].internalParent = 1;
    groups[0].children.push_back(1);
    groups[1].children.push_back(2);
    requireRockstar(rockstarGatherParticles(groups, 0).size() == 122,
        "invalid stripped-child donation fixture");
    rockstarPruneCatalogueGroups(groups, 1.0f);
    requireRockstar(groups.size() == 2, "unresolved middle child not pruned");
    requireRockstar(groups[0].parts.size() == 112 && groups[1].parts.size() == 10,
        "SO pruning lost bound particles or absorbed a surviving grandchild");
    requireRockstar(groups[1].internalParent == 0 && groups[0].children.size() == 1,
        "surviving grandchild did not reconnect to retained host");
    requireRockstarClose(groups[0].massVir, 122.0, 0.0, "SO pruning changed inclusive host mass");
    requireRockstar(
        groups[0].catalogueId == 2001, "host catalogue identity was not refreshed after donation");
}

void checkRockstarCenterUnits() {
    DeviceVector<RockstarParticle> original;
    std::mt19937 random(117);
    std::normal_distribution<float> gaussian(0.0f, 1.0f);
    for (int i = 0; i < 128; ++i) {
        auto particle = rockstarTestParticle(0, 0, 0, i);
        float const width = i < 32 ? 1.e-5f : 1.e-4f;
        for (int dim = 0; dim < NDIM; ++dim) {
            particle.xValue[dim] = width * gaussian(random);
            particle.xValue[dim + NDIM] = 1.e-5f * gaussian(random);
        }
        original.pushBack(particle);
    }
    auto scaled = original;
    for (auto& particle : scaled) {
        for (int dim = 0; dim < NDIM; ++dim) {
            particle.xValue[dim] *= 2.0f;
        }
    }
    int nextId = 1;
    auto const first = rockstarSeeds(original, nextId, 1.0f, 1.0f, 1.0f, false, 1);
    auto const options = getOptions();
    auto dilatedOptions = options;
    // Dilate the background number density with the coordinates so that every
    // dynamical assignment radius scales by two as well. This preserves halo
    // membership while independently checking the returned error's length unit.
    dilatedOptions.initial.numberOfParticles /= 8;
    setOptions(dilatedOptions);
    nextId = 1;
    auto const second = rockstarSeeds(scaled, nextId, 1.0f, 1.0f, 1.0f, false, 1);
    setOptions(options);
    requireRockstar(!first.empty() && first.size() == second.size(),
        "recursive center fixture changed halo count under spatial rescaling");
    for (size_t i = 0; i < first.size(); ++i) {
        requireRockstar(first[i].parts.size() == second[i].parts.size(),
            "recursive center fixture changed particle membership under rescaling");
        requireRockstar(first[i].sigmaXMin > 0.0f, "center fixture has no measurable uncertainty");
        requireRockstarClose(second[i].sigmaXMin, 2.0 * first[i].sigmaXMin, 2.e-5,
            "recursive Poisson center uncertainty did not return to physical coordinates", 1.e-12);
        requireRockstarClose(second[i].x, 2.0 * first[i].x, 2.e-5,
            "recursive halo center failed spatial rescaling", 1.e-10);
    }
}

void checkRockstarTiedShellBoundary() {
    double const boundary = rockstarTestRadius(12, rockstarTestVirialDensity(1.0));
    for (bool inside : {true, false}) {
        float const radius =
            std::nextafter(float(boundary), inside ? 0.0f : std::numeric_limits<float>::infinity());
        auto group = rockstarTestGroup(12, 0.0f, 1, radius);
        Vector<Subgroup> groups{group};
        Array<double, NDIM> const origin{};
        auto const record = rockstarMakeRecord(groups, 0, 1.0f, origin);
        requireRockstarClose(record.mvir, inside ? 12.0 : 0.0, 2.e-6,
            "tied shell must cross the SO boundary as a complete shell");
        if (inside) {
            requireRockstarClose(float162double(record.rmax), 1000.0 * radius, 1.e-3,
                "record aperture includes the complete accepted boundary shell");
        } else {
            requireRockstarClose(
                record.e, 0.0, 0.0, "rejected outer shell leaked into record energy aperture");
        }
    }
}

#ifdef USE_CUDA
std::vector<int> rockstarCanonicalPartition(DeviceVector<RockstarParticle> const& particles) {
    std::map<int, int> minimum;
    for (auto const& particle : particles) {
        if (particle.subgroup != ROCKSTAR_NO_GROUP) {
            auto [it, inserted] = minimum.emplace(particle.subgroup, particle.index);
            if (!inserted) {
                it->second = std::min(it->second, int(particle.index));
            }
        }
    }
    std::vector<int> result(particles.size());
    for (auto const& particle : particles) {
        result[particle.index] =
            particle.subgroup == ROCKSTAR_NO_GROUP ? particle.index : minimum.at(particle.subgroup);
    }
    return result;
}

void checkRockstarCudaConnectivity() {
    // Run once per backend, keeping allocation and dispatch modes fixed until
    // every fixture buffer has been destroyed.
    constexpr int kLeaves = 65;
    constexpr int kLeafSize = 64;
    DeviceVector<RockstarParticle> particles;
    DeviceVector<RockstarTree> manualNodes(kLeaves);
    DeviceVector<int> leaves;
    std::vector<int> expected(kLeaves * kLeafSize);
    for (int leaf = 0; leaf < kLeaves; ++leaf) {
        auto& node = manualNodes[leaf];
        node.particleBegin = particles.size();
        node.particleEnd = node.particleBegin + kLeafSize;
        node.children[LEFT] = node.children[RIGHT] = -1;
        node.neighbors.pushBack(leaf);
        leaves.pushBack(leaf);
        for (int lane = 0; lane < kLeafSize; ++lane) {
            int first;
            float coordinate;
            if (lane == 0 || lane == 32) {
                first = lane;
                coordinate = lane == 0 ? -10.0f : 20.0f;
            } else if (lane < 16) {
                first = 1;
                coordinate = 0.0f;
            } else if (lane < 32) {
                first = 16;
                coordinate = 10.0f;
            } else if (lane < 48) {
                first = 33;
                coordinate = 30.0f;
            } else {
                first = 48;
                coordinate = 40.0f;
            }
            int const index = particles.size();
            particles.pushBack(rockstarTestParticle(leaf * 100.0f + coordinate, 0, 0, index));
            expected[index] = leaf * kLeafSize + first;
        }
        for (int dim = 0; dim < 6; ++dim) {
            node.box.begin[dim] = dim == 0 ? leaf * 100.0f - 10.0f : 0.0f;
            node.box.end[dim] = dim == 0 ? leaf * 100.0f + 40.0f : 0.0f;
        }
    }
    // In each leaf, both lane-zero anchors (indices 0 and 32) are isolated.
    // The old kernel fragmented every real component in all other lanes.
    int nextId = 1;
    if (!getOptions().compute.cuda) {
        auto cpuParticles = particles;
        DeviceVector<RockstarTree> cpuTrees;
        rockstarFormTree(cpuTrees, cpuParticles);
        rockstarFindSubgroups(cpuTrees, cpuParticles, 1.0f, nextId, false);
        requireRockstar(rockstarCanonicalPartition(cpuParticles) == expected,
            "CPU component fixture disagrees with analytic phase-space components");
        return;
    }
    auto gpuParticles = particles;
    rockstarFindSubgroupsCuda(manualNodes, leaves, gpuParticles, 1.0f, nextId);
    requireRockstar(rockstarCanonicalPartition(gpuParticles) == expected,
        "CUDA multiwarp label propagation disagrees with CPU components");
    DeviceVector<RockstarTree> gpuTrees;
    gpuParticles = particles;
    rockstarFormTree(gpuTrees, gpuParticles);
    nextId = 1;
    rockstarFindSubgroups(gpuTrees, gpuParticles, 1.0f, nextId, true);
    requireRockstar(rockstarCanonicalPartition(gpuParticles) == expected,
        "CUDA production tree dispatch disagrees with CPU components");
    auto const coincident = rockstarFindSubgroups(makeRockstarSnapshot(4097, false), 1.0f);
    requireRockstar(coincident.size() == 1 && coincident[0].parts.size() == 4097,
        "CUDA nearest-neighbor and full finder lost coincident particles");
}
#endif

Vector<ParticleData> makeRockstarSnapshot(int count, bool planar) {
    Vector<ParticleData> particles(count);
    for (int i = 0; i < count; ++i) {
        auto& particle = particles[i];
        for (int dim = 0; dim < NDIM; ++dim) {
            particle.x[dim] = Position(0.25);
            particle.v[dim] = 0.0f;
        }
        if (planar) {
            particle.x[0] = Position(0.25 + 0.0002 * std::sin(double(i)));
            particle.x[1] = Position(0.25 + 0.0002 * std::cos(double(i)));
            particle.v[0] = 1.e-6f * std::sin(1.3f * i);
            particle.v[1] = 1.e-6f * std::cos(1.7f * i);
            particle.v[2] = 1.e-6f * std::sin(1.9f * i);
        }
        particle.index = i;
        particle.rank = 0;
        particle.identity = i + 1;
        particle.lastGroup = NO_GROUP;
        particle.lastHost = NO_GROUP;
    }
    return particles;
}

void checkRockstarFinderDegeneracies() {
    for (int count : {20, 130, 4097}) {
        auto const particles = makeRockstarSnapshot(count, count == 20);
        auto const groups = rockstarFindSubgroups(particles, 1.0f);
        std::set<int> retained;
        for (auto const& group : groups) {
            requireRockstar(rockstarTestFinite(group.x) && rockstarTestFinite(group.rDyn),
                "degenerate finder produced nonfinite coordinates");
            for (auto const& particle : group.parts) {
                retained.insert(particle.index);
            }
        }
        requireRockstar(retained.size() == size_t(count),
            "planar/coincident/cold CPU finder lost particles, N=" + std::to_string(count));
    }
    requireRockstar(!getOptions().compute.cuda, "large CPU regression enabled CUDA");
}

void checkRockstarRecordInvariance() {
    Subgroup group;
    group.id = 1;
    group.x = group.y = group.z = 0.0f;
    group.sigmaXMin = 0.0f;
    double expectedJz = 0.0;
    for (int i = 0; i < 24; ++i) {
        auto particle = rockstarTestParticle(0, 0, 0, i);
        int const axis = (i / 2) % 3;
        float const radius = axis == 0 ? 0.0006f : axis == 1 ? 0.0004f : 0.0002f;
        particle.xValue[axis] = i % 2 ? -radius : radius;
        particle.vx = -particle.y;
        particle.vy = particle.x;
        expectedJz += double(particle.x) * particle.x + double(particle.y) * particle.y;
        group.parts.pushBack(particle);
    }
    rockstarUpdateStatistics(group, 1.0f, 1.0f, 1.0f);
    Vector<Subgroup> groups{group};
    Array<double, NDIM> const origin{};
    auto const record = rockstarMakeRecord(groups, 0, 1.0f, origin);
    requireRockstarClose(record.jx, 0.0, 0.0, "rotation has no Jx");
    requireRockstarClose(record.jy, 0.0, 0.0, "rotation has no Jy");
    requireRockstarClose(record.jz, expectedJz, 2.e-6, "Jz cross product and conversion");
    requireRockstarClose(float162double(record.boa), 2.0 / 3.0, 1.e-3, "ellipsoid b/a");
    requireRockstarClose(float162double(record.coa), 1.0 / 3.0, 1.e-3, "ellipsoid c/a");
    requireRockstarClose(std::abs(float162double(record.ax)), 0.6 / std::sqrt(3.0), 1.e-3,
        "major-axis eigenvector amplitude");
    auto shifted = group;
    std::array<float, 3> const shift{0.125f, -0.25f, 0.0625f};
    std::array<float, 3> const boost{0.02f, -0.01f, 0.005f};
    shifted.x += shift[0];
    shifted.y += shift[1];
    shifted.z += shift[2];
    for (auto& particle : shifted.parts) {
        for (int dim = 0; dim < NDIM; ++dim) {
            particle.xValue[dim] += shift[dim];
            particle.xValue[dim + NDIM] += boost[dim];
        }
    }
    groups[0] = std::move(shifted);
    auto const translated = rockstarMakeRecord(groups, 0, 1.0f, origin);
    requireRockstarClose(translated.jz, record.jz, 5.e-4, "translation/boost invariant Jz");
    requireRockstarClose(translated.e, record.e, 1.e-3, "translation/boost invariant energy");
    requireRockstarClose(translated.mvir, record.mvir, 0.0, "translation invariant mass");
    requireRockstarClose(translated.x - record.x, shift[0], 2.e-6, "translated record position");
    requireRockstarClose(float162double(translated.boa), float162double(record.boa), 2.e-3,
        "translation invariant halo shape");
    groups[0] = group;
    auto const earlier = rockstarMakeRecord(groups, 0, 0.5f, origin);
    requireRockstarClose(
        earlier.jz, 0.5 * record.jz, 2.e-6, "physical angular momentum epoch factor");
    requireRockstarClose(float162double(earlier.vcircMax),
        std::sqrt(2.0) * float162double(record.vcircMax), 2.e-3,
        "physical circular velocity epoch factor");
    std::array<RockstarRecord, 3> const records{record, translated, earlier};
    checkRockstarCatalogueRoundtrip(records.data(), records.size(), "snapshot catalogue");
}

void checkRockstarLightconeEpoch() {
    Vector<LightconeEntry> particles(30);
    for (auto& particle : particles) {
        particle.x = LightconeReal(0.02);
        particle.y = particle.z = LightconeReal(0.0);
        particle.vx = particle.vy = particle.vz = 0.0f;
        particle.pix = 0;
    }
    double const initialScale = 1.0 / (1.0 + getOptions().cosmology.startRedshift);
    double const tauMax = particles[0].x.toDouble() + cosmosConformalTime(initialScale, 0.5);
    auto const records = rockstarFindSubgroups(particles, tauMax);
    requireRockstar(records.size() == 1, "lightcone coincident halo disappeared");
    requireRockstarClose(float162double(records[0].zValue), 1.0, 2.e-3,
        "lightcone epoch must come from conformal time");
    requireRockstarClose(float162double(records[0].rvir),
        1000.0 * rockstarTestRadius(30, rockstarTestVirialDensity(0.5)), 2.e-3,
        "lightcone halo properties must use its emission epoch");
    checkRockstarCatalogueRoundtrip(records.data(), records.size(), "lightcone catalogue");
}

}    // namespace detail

void rockstarTest() {
    auto const original = getOptions();
    auto options = original;
    options.compute.cuda = false;
    options.initial.numberOfParticles = 1000000;
    options.gravity.gm = 1.e-8;
    options.gravity.supportRadius = 1.e-4;
    options.cosmology.omegaM = 0.3;
    options.cosmology.omegaLambda = 0.7;
    options.cosmology.omegaR = options.cosmology.omegaK = 0.0;
    options.cosmology.hubble = 0.7;
    options.cosmology.startRedshift = 9.0;
    // Make the public M_sun/h, Mpc/h and km/s conversion factors unity.
    options.units.massG = constants::kM0 / options.cosmology.hubble;
    options.units.lengthCm = constants::kMpcToCm / options.cosmology.hubble;
    options.units.timeSeconds = options.units.lengthCm / 1.e5;
    setOptions(options);
    try {
        detail::checkRockstarCatalogueHalfBoundaries();
        detail::checkRockstarCataloguePublication();
        detail::checkRockstarUnbindingOrder();
        detail::checkRockstarSphericalOverdensities();
        detail::checkRockstarCoreAperture();
        detail::checkRockstarHierarchyAndHistory();
        detail::checkRockstarMajorMerger();
        detail::checkRockstarPrunedDonation();
        detail::checkRockstarCenterUnits();
        detail::checkRockstarTiedShellBoundary();
        detail::checkRockstarFinderDegeneracies();
        detail::checkRockstarRecordInvariance();
        detail::checkRockstarLightconeEpoch();
#ifdef USE_CUDA
        if (original.compute.cuda) {
            detail::checkRockstarCudaConnectivity();
            options.compute.cuda = true;
            setOptions(options);
            detail::checkRockstarCudaConnectivity();
            options.compute.cuda = false;
            setOptions(options);
        }
#endif
    } catch (...) {
        setOptions(original);
        throw;
    }
    setOptions(original);
    PRINT("ROCKSTAR production integration regressions passed\n");
}
