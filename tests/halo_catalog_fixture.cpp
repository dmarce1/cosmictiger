#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cosmictiger/group_header.hpp>
#include <cosmictiger/rockstar_record.hpp>

namespace detail {

namespace fs = std::filesystem;
constexpr std::uint64_t kIdBase = 9007199254740993ULL;
constexpr std::uint64_t kNoGroup = 0x7fffffffffffffffULL;
constexpr size_t kHaloCount = 64;
constexpr int kRanks = 2;
constexpr std::array<Float16 RockstarRecord::*, 31> kHalfFields{{&RockstarRecord::zValue,
    &RockstarRecord::rmax, &RockstarRecord::rvir, &RockstarRecord::rvmax, &RockstarRecord::rs,
    &RockstarRecord::rkly, &RockstarRecord::sigX, &RockstarRecord::deltaX, &RockstarRecord::toW,
    &RockstarRecord::sigV, &RockstarRecord::sigVb, &RockstarRecord::vcircMax,
    &RockstarRecord::deltaV, &RockstarRecord::boa, &RockstarRecord::coa, &RockstarRecord::ax,
    &RockstarRecord::ay, &RockstarRecord::az, &RockstarRecord::boa500, &RockstarRecord::coa500,
    &RockstarRecord::ax500, &RockstarRecord::ay500, &RockstarRecord::az500, &RockstarRecord::lambda,
    &RockstarRecord::lambdaB, &RockstarRecord::vxc, &RockstarRecord::vyc, &RockstarRecord::vzc,
    &RockstarRecord::vxb, &RockstarRecord::vyb, &RockstarRecord::vzb}};

void writeBinary(fs::path const& path, void const* data, size_t bytes) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<char const*>(data), static_cast<std::streamsize>(bytes));
    output.close();
    if (!output) {
        throw std::runtime_error("Cannot write fixture " + path.string());
    }
}

void writeText(fs::path const& path, std::string const& text) {
    std::ofstream output(path);
    output << text;
    output.close();
    if (!output) {
        throw std::runtime_error("Cannot write fixture " + path.string());
    }
}

RockstarRecord makeRecord(size_t index) {
    RockstarRecord record{};
    unsigned int const local = static_cast<unsigned int>(index % 8);
    record.id.fofId = 100 + index / 8;
    record.id.haloId = local;
    record.id.sys = local <= 1;
    record.pid = local == 0 ? -1 : (local == 2 ? 1 : 0);
    record.x = 4.0 + 1.1 * index;
    record.y = 5.0 + (index * 7 % 64);
    record.z = 3.0 + (index * 11 % 64);
    record.mvir = 1.0e12 / (local + 1);
    record.e = -2.0e15 / (local + 1);
    record.jx = 1.0e8 * (index + 1);
    record.jy = -2.0e8 * (index + 1);
    record.jz = 3.0e8 * (index + 1);
    record.m200 = record.mvir * 0.9f;
    record.m500 = record.mvir * 0.7f;
    record.m2500 = record.mvir * 0.4f;
    for (size_t i = 0; i < kHalfFields.size(); ++i) {
        record.*kHalfFields[i] = double2float16(0.25 * (i + 1));
    }
    record.zValue = double2float16(1.0);
    record.rmax = double2float16(300.0 / (local + 1));
    record.rvir = double2float16(200.0 / (local + 1));
    record.rvmax = double2float16(100.0 / (local + 1));
    record.rs = index == 0 ? Float16{0x0001} : double2float16(20.0 / (local + 1));
    record.rkly = double2float16(20.0 / (local + 1));
    record.boa = double2float16(0.75);
    record.coa = double2float16(0.5);
    record.lambda = double2float16(0.01 * (local + 1));
    record.lambdaB = double2float16(0.005 * (local + 1));
    record.vcircMax = double2float16(100.0 + index * 5);
    record.vxc = double2float16(-100.0 + index);
    record.vyc = double2float16(40.0 - index);
    record.vzc = index == 0 ? Float16{0x8000} : double2float16(-10.0 + index);
    return record;
}

void writeFixture(fs::path const& directory) {
    fs::create_directories(directory);
    // Replacing a fixture in place must invalidate completion before any data.
    fs::remove(directory / "rockstar_hosts.complete");
    GroupHeader const header{7, 0.25, 100.0, 0.5};
    writeBinary(directory / "groups.dat", &header, sizeof(header));
    for (int rank = 0; rank < kRanks; ++rank) {
        std::vector<RockstarRecord> records;
        std::string hosts;
        for (size_t ordinal = 0; ordinal < kHaloCount / kRanks; ++ordinal) {
            size_t const index = rank * (kHaloCount / kRanks) + ordinal;
            records.push_back(makeRecord(index));
            auto const& record = records.back();
            std::uint64_t const host = record.pid == -1 ?
                kNoGroup :
                kIdBase + (index / 8) * 8 + static_cast<unsigned int>(record.pid);
            hosts += std::to_string(kIdBase + index) + " " + std::to_string(host) + "\n";
        }
        std::string const basename = "rockstar." + std::to_string(rank) + ".dat";
        writeBinary(directory / basename, records.data(), records.size() * sizeof(RockstarRecord));
        writeText(directory / (basename + ".format"),
            "format_version 2\nfloat16 ieee754_binary16\nrecord_size " +
                std::to_string(sizeof(RockstarRecord)) + "\nrecord_count " +
                std::to_string(records.size()) + "\nbyte_order " +
                (std::endian::native == std::endian::little ? "little" : "big") + "\n");
        writeText(directory / ("rockstar_hosts." + std::to_string(rank) + ".dat"), hosts);
    }
    writeText(directory / "rockstar_descendants.dat", "");
    writeText(directory / "rockstar_hosts.complete", "1 2 64\n");
}

}    // namespace detail

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: halo_catalog_fixture OUTPUT_DIRECTORY\n";
        return 2;
    }
    try {
        detail::writeFixture(argv[1]);
    } catch (std::exception const& error) {
        std::cerr << "halo_catalog_fixture: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
