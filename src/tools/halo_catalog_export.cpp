#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <cosmictiger/analysis/group_header.hpp>
#include <cosmictiger/analysis/rockstar_record.hpp>

namespace detail {

namespace fs = std::filesystem;
constexpr std::uint64_t kNoGroup = 0x7fffffffffffffffULL;
using LocalKey = std::tuple<int, std::uint64_t, unsigned int>;

struct Halo {
    RockstarRecord record;
    std::uint64_t id;
    std::uint64_t host;
    int rank;
    size_t ordinal;
};

struct CatalogueFormat {
    size_t recordCount = 0;
    std::string virialDensity = "bryan_norman";
    std::string soDensityReference = "critical";
};

constexpr std::array<std::pair<char const*, Float16 RockstarRecord::*>, 31> kHalfFields{
    {{"zValue", &RockstarRecord::zValue}, {"rmax", &RockstarRecord::rmax},
        {"rvir", &RockstarRecord::rvir}, {"rvmax", &RockstarRecord::rvmax},
        {"rs", &RockstarRecord::rs}, {"rkly", &RockstarRecord::rkly},
        {"sigX", &RockstarRecord::sigX}, {"deltaX", &RockstarRecord::deltaX},
        {"toW", &RockstarRecord::toW}, {"sigV", &RockstarRecord::sigV},
        {"sigVb", &RockstarRecord::sigVb}, {"vcircMax", &RockstarRecord::vcircMax},
        {"deltaV", &RockstarRecord::deltaV}, {"boa", &RockstarRecord::boa},
        {"coa", &RockstarRecord::coa}, {"ax", &RockstarRecord::ax}, {"ay", &RockstarRecord::ay},
        {"az", &RockstarRecord::az}, {"boa500", &RockstarRecord::boa500},
        {"coa500", &RockstarRecord::coa500}, {"ax500", &RockstarRecord::ax500},
        {"ay500", &RockstarRecord::ay500}, {"az500", &RockstarRecord::az500},
        {"lambda", &RockstarRecord::lambda}, {"lambdaB", &RockstarRecord::lambdaB},
        {"vxc", &RockstarRecord::vxc}, {"vyc", &RockstarRecord::vyc}, {"vzc", &RockstarRecord::vzc},
        {"vxb", &RockstarRecord::vxb}, {"vyb", &RockstarRecord::vyb},
        {"vzb", &RockstarRecord::vzb}}};

[[noreturn]] void fail(fs::path const& path, std::string const& reason) {
    throw std::runtime_error(path.string() + ": " + reason);
}

std::ifstream openText(fs::path const& path) {
    std::ifstream input(path);
    if (!fs::is_regular_file(path) || !input) {
        fail(path, "cannot read file");
    }
    return input;
}

std::uint64_t unsignedInteger(std::string const& value, fs::path const& path) {
    std::uint64_t number = 0;
    auto const result = std::from_chars(value.data(), value.data() + value.size(), number);
    if (value.empty() || result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
        fail(path, "invalid unsigned integer " + value);
    }
    return number;
}

std::vector<std::string> tokens(std::string const& line) {
    std::istringstream input(line);
    std::vector<std::string> result;
    std::string value;
    while (input >> value) {
        result.push_back(value);
    }
    return result;
}

std::vector<std::string> oneLine(fs::path const& path, size_t count) {
    auto input = openText(path);
    std::string line;
    if (!std::getline(input, line)) {
        fail(path, "missing content");
    }
    auto values = tokens(line);
    if (values.size() != count) {
        fail(path, "unexpected number of fields");
    }
    while (std::getline(input, line)) {
        if (!tokens(line).empty()) {
            fail(path, "unexpected extra content");
        }
    }
    if (!input.eof()) {
        fail(path, "read error");
    }
    return values;
}

CatalogueFormat readFormat(fs::path const& path) {
    auto const formatPath = fs::path(path.string() + ".format");
    auto input = openText(formatPath);
    std::map<std::string, std::string> metadata;
    std::string line;
    while (std::getline(input, line)) {
        auto const values = tokens(line);
        if (values.size() != 2 || !metadata.emplace(values[0], values[1]).second) {
            fail(formatPath, "malformed or duplicate metadata field");
        }
    }
    if (!input.eof()) {
        fail(formatPath, "read error");
    }
    static_assert(
        std::endian::native == std::endian::little || std::endian::native == std::endian::big);
    std::string const byteOrder = std::endian::native == std::endian::little ? "little" : "big";
    bool const hasConvention =
        metadata.count("virial_density") == 1 && metadata.count("so_density_reference") == 1;
    if (metadata.size() != (hasConvention ? 7 : 5) || metadata["format_version"] != "2" ||
        metadata["float16"] != "ieee754_binary16" || metadata["byte_order"] != byteOrder ||
        metadata["record_size"] != std::to_string(sizeof(RockstarRecord)) ||
        metadata.count("record_count") != 1) {
        fail(formatPath,
            "requires format_version 2, IEEE binary16, matching native ABI and byte order");
    }
    std::uint64_t const count = unsignedInteger(metadata.at("record_count"), formatPath);
    if (count > std::numeric_limits<size_t>::max() / sizeof(RockstarRecord) ||
        count > std::numeric_limits<std::streamsize>::max() / sizeof(RockstarRecord) ||
        fs::file_size(path) != count * sizeof(RockstarRecord)) {
        fail(path, "record count or binary file size mismatch");
    }
    CatalogueFormat result;
    result.recordCount = static_cast<size_t>(count);
    if (hasConvention) {
        result.virialDensity = metadata.at("virial_density");
        result.soDensityReference = metadata.at("so_density_reference");
        bool const fitted = result.virialDensity == "bryan_norman_flat" ||
            result.virialDensity == "bryan_norman_open";
        bool const meanMatter = result.virialDensity == "200_mean_matter";
        if (!((fitted && result.soDensityReference == "critical") ||
                (meanMatter && result.soDensityReference == "mean_matter"))) {
            fail(formatPath, "unknown or inconsistent halo density convention");
        }
    }
    return result;
}

GroupHeader readHeader(fs::path const& directory) {
    auto const path = directory / "groups.dat";
    if (fs::file_size(path) != sizeof(GroupHeader)) {
        fail(path, "header size does not match this build's native ABI");
    }
    std::ifstream input(path, std::ios::binary);
    GroupHeader header{};
    if (!input.read(reinterpret_cast<char*>(&header), sizeof(header))) {
        fail(path, "cannot read group header");
    }
    if (header.number < 0 || !std::isfinite(header.time) || !std::isfinite(header.scale) ||
        header.scale <= 0.0 || !std::isfinite(header.boxLength) || header.boxLength <= 0.0) {
        fail(path, "invalid snapshot header values");
    }
    return header;
}

void validateProperties(Halo const& halo, fs::path const& path) {
    auto const& record = halo.record;
    std::array<double, 11> const values{{record.x, record.y, record.z, record.mvir, record.e,
        record.jx, record.jy, record.jz, record.m200, record.m500, record.m2500}};
    for (double const value : values) {
        if (!std::isfinite(value)) {
            fail(path, "nonfinite halo property at ordinal " + std::to_string(halo.ordinal));
        }
    }
    for (auto const& field : kHalfFields) {
        if (((record.*field.second).bits & 0x7c00u) == 0x7c00u) {
            fail(path,
                "nonfinite halo " + std::string(field.first) + " at ordinal " +
                    std::to_string(halo.ordinal));
        }
    }
    if (record.pid < -1) {
        fail(path, "invalid pid at ordinal " + std::to_string(halo.ordinal));
    }
}

std::vector<Halo> readHalos(fs::path const& directory, int rankCount, size_t expectedCount,
    CatalogueFormat& catalogueFormat) {
    std::vector<Halo> halos;
    std::map<std::uint64_t, size_t> byId;
    std::map<LocalKey, size_t> byLocal;
    for (int rank = 0; rank < rankCount; ++rank) {
        auto const path = directory / ("rockstar." + std::to_string(rank) + ".dat");
        auto const format = readFormat(path);
        if (rank == 0) {
            catalogueFormat = format;
        } else if (format.virialDensity != catalogueFormat.virialDensity ||
            format.soDensityReference != catalogueFormat.soDensityReference) {
            fail(path, "halo density convention differs between ranks");
        }
        size_t const count = format.recordCount;
        if (count > expectedCount - halos.size()) {
            fail(path, "rank record count exceeds completion manifest");
        }
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            fail(path, "cannot read file");
        }
        auto const hostsPath = directory / ("rockstar_hosts." + std::to_string(rank) + ".dat");
        auto hosts = openText(hostsPath);
        for (size_t ordinal = 0; ordinal < count; ++ordinal) {
            Halo halo{};
            halo.rank = rank;
            halo.ordinal = ordinal;
            if (!input.read(reinterpret_cast<char*>(&halo.record), sizeof(halo.record))) {
                fail(path, "cannot read record");
            }
            std::string line;
            if (!std::getline(hosts, line)) {
                fail(hostsPath, "host row count does not match record count");
            }
            auto const values = tokens(line);
            if (values.size() != 2) {
                fail(hostsPath, "malformed host row");
            }
            halo.id = unsignedInteger(values[0], hostsPath);
            halo.host = unsignedInteger(values[1], hostsPath);
            if (halo.id >= kNoGroup || halo.host > kNoGroup || halo.id == halo.host) {
                fail(hostsPath, "invalid halo or host identifier");
            }
            validateProperties(halo, path);
            if (!byId.emplace(halo.id, halos.size()).second) {
                fail(hostsPath, "duplicate persistent halo identifier");
            }
            LocalKey const key{rank, static_cast<std::uint64_t>(halo.record.id.fofId),
                static_cast<unsigned int>(halo.record.id.haloId)};
            if (!byLocal.emplace(key, halos.size()).second) {
                fail(path, "duplicate local FOF/halo key");
            }
            halos.push_back(halo);
        }
        std::string line;
        while (std::getline(hosts, line)) {
            if (!tokens(line).empty()) {
                fail(hostsPath, "host row count does not match record count");
            }
        }
        if (!hosts.eof()) {
            fail(hostsPath, "read error");
        }
    }
    if (halos.size() != expectedCount) {
        fail(directory, "completion manifest halo count mismatch");
    }
    std::vector<size_t> parents(halos.size(), halos.size());
    for (size_t i = 0; i < halos.size(); ++i) {
        auto const& halo = halos[i];
        if ((halo.record.pid == -1) != (halo.host == kNoGroup)) {
            fail(directory, "pid disagrees with persistent host mapping");
        }
        if (halo.host != kNoGroup) {
            auto const host = byId.find(halo.host);
            auto const local =
                byLocal.find(LocalKey{halo.rank, static_cast<std::uint64_t>(halo.record.id.fofId),
                    static_cast<unsigned int>(halo.record.pid)});
            if (host == byId.end() || local == byLocal.end() || host->second != local->second) {
                fail(directory, "missing host or pid/FOF/rank disagrees with host mapping");
            }
            parents[i] = host->second;
        }
    }
    std::vector<unsigned char> state(halos.size(), 0);
    for (size_t i = 0; i < halos.size(); ++i) {
        size_t current = i;
        std::vector<size_t> path;
        while (current != halos.size() && state[current] == 0) {
            state[current] = 1;
            path.push_back(current);
            current = parents[current];
        }
        if (current != halos.size() && state[current] == 1) {
            fail(directory, "cycle in host relationships");
        }
        for (size_t const visited : path) {
            state[visited] = 2;
        }
    }
    return halos;
}

void exportSnapshot(fs::path const& directory) {
    auto const manifestPath = directory / "rockstar_hosts.complete";
    auto const manifest = oneLine(manifestPath, 3);
    std::uint64_t const ranks = unsignedInteger(manifest[1], manifestPath);
    std::uint64_t const count = unsignedInteger(manifest[2], manifestPath);
    if (manifest[0] != "1" || ranks == 0 || ranks > std::numeric_limits<int>::max() ||
        count > std::numeric_limits<size_t>::max()) {
        fail(manifestPath, "invalid completion manifest version, rank count, or halo count");
    }
    // The marker certifies that descendant output was closed, even though this
    // viewer milestone does not ingest temporal relationships.
    auto descendants = openText(directory / "rockstar_descendants.dat");
    GroupHeader const header = readHeader(directory);
    CatalogueFormat format;
    auto const halos =
        readHalos(directory, static_cast<int>(ranks), static_cast<size_t>(count), format);
    std::cout << std::showpoint << std::setprecision(std::numeric_limits<double>::max_digits10)
              << "{\"type\":\"catalog\",\"schema_version\":1,\"snapshot\":" << header.number
              << ",\"scale_factor\":" << header.scale << ",\"time\":" << header.time
              << ",\"box_length_mpc\":" << header.boxLength << ",\"rank_count\":" << ranks
              << ",\"halo_count\":" << halos.size() << ",\"virial_density\":\""
              << format.virialDensity << "\",\"so_density_reference\":\""
              << format.soDensityReference << "\"}\n";
    for (auto const& halo : halos) {
        auto const& record = halo.record;
        std::cout << "{\"type\":\"halo\",\"rank\":" << halo.rank << ",\"ordinal\":" << halo.ordinal
                  << ",\"halo_id\":\"" << halo.id << "\",\"host_id\":";
        if (halo.host == kNoGroup) {
            std::cout << "null";
        } else {
            std::cout << '"' << halo.host << '"';
        }
        std::cout << ",\"fof_id\":\"" << record.id.fofId
                  << "\",\"local_halo_id\":" << record.id.haloId
                  << ",\"sys\":" << (record.id.sys ? "true" : "false") << ",\"pid\":" << record.pid
                  << ",\"x\":" << record.x << ",\"y\":" << record.y << ",\"z\":" << record.z
                  << ",\"mvir\":" << record.mvir << ",\"e\":" << record.e << ",\"jx\":" << record.jx
                  << ",\"jy\":" << record.jy << ",\"jz\":" << record.jz
                  << ",\"m200\":" << record.m200 << ",\"m500\":" << record.m500
                  << ",\"m2500\":" << record.m2500;
        for (auto const& field : kHalfFields) {
            std::cout << ",\"" << field.first << "\":" << float162double(record.*field.second);
        }
        std::cout << "}\n";
    }
    if (!std::cout) {
        throw std::runtime_error("cannot write JSONL output");
    }
}

}    // namespace detail

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: halo_catalog_export SNAPSHOT_DIRECTORY\n";
        return 2;
    }
    try {
        detail::exportSnapshot(argv[1]);
    } catch (std::exception const& error) {
        std::cerr << "halo_catalog_export: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
