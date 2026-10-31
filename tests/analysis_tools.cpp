#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

// Exercise the production parsers/converters with synthetic native inputs.
#define main groups2halofuncMain
#include "../src/tools/groups2halofunc.cpp"
#undef main
#define main healpix2siloMain
#include "../src/tools/healpix2silo.cpp"
#undef main

#include <cosmictiger/analysis/group_header.hpp>

namespace detail {

class AnalysisTools : public ::testing::Test {
protected:
    std::filesystem::path original_;
    std::filesystem::path directory_;

    void SetUp() override {
        original_ = std::filesystem::current_path();
        directory_ = std::filesystem::temp_directory_path() /
            ("cosmictiger_analysis_tools_" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directory(directory_);
        std::filesystem::current_path(directory_);
    }

    void TearDown() override {
        std::filesystem::current_path(original_);
        std::filesystem::remove_all(directory_);
    }

    void catalogue(std::vector<std::pair<float, bool>> const& masses, int ranks = 1) {
        std::filesystem::create_directory("groups.4");
        GroupHeader header{};
        header.number = 4;
        header.boxLength = 10.0;
        header.scale = 1.0;
        std::ofstream headerFile("groups.4/groups.dat", std::ios::binary);
        headerFile.write(reinterpret_cast<char const*>(&header), sizeof(header));
        headerFile.close();
        std::ofstream manifest("groups.4/rockstar_hosts.complete");
        manifest << 1 << ' ' << ranks << ' ' << masses.size() << '\n';
        manifest.close();
        for (int rank = 0; rank < ranks; ++rank) {
            std::string const suffix = std::to_string(rank) + ".dat";
            auto* records = std::fopen(("groups.4/groups.4." + suffix).c_str(), "wb");
            ASSERT_NE(records, nullptr);
            std::ofstream hosts("groups.4/rockstar_hosts." + suffix);
            for (size_t index = rank; index < masses.size(); index += ranks) {
                GroupEntry entry{};
                entry.id = index + 1;
                entry.mass = masses[index].first;
                entry.write(records);
                hosts << entry.id << ' '
                      << (masses[index].second ? std::numeric_limits<GroupInteger>::max() : 1)
                      << '\n';
            }
            ASSERT_EQ(std::fclose(records), 0);
        }
    }

    int massFunction(int bins = 2) {
        std::string command = "groups2halofunc";
        std::string snapshot = "--input.groupNumber=4";
        std::string binOption = "--analysis.numberOfBins=" + std::to_string(bins);
        char* arguments[] = {command.data(), snapshot.data(), binOption.data()};
        return haloFunction(3, arguments);
    }

    void map(int nside, int count, std::vector<float> const& values) {
        std::ofstream output("view.dat", std::ios::binary);
        output.write(reinterpret_cast<char const*>(&nside), sizeof(nside));
        output.write(reinterpret_cast<char const*>(&count), sizeof(count));
        output.write(reinterpret_cast<char const*>(values.data()), values.size() * sizeof(float));
    }

    int convertMap(std::string target = "view.silo") {
        std::string command = "healpix2silo";
        std::string source = "--input.file=view.dat";
        std::string destination = "--output.file=" + target;
        char* arguments[] = {command.data(), source.data(), destination.data()};
        return convert(3, arguments);
    }
};

}    // namespace detail

using detail::AnalysisTools;

TEST_F(AnalysisTools, MassFunctionUsesDistinctHalosAndCatalogueVolume) {
    catalogue({{100.0f, true}, {200.0f, false}, {10000.0f, true}}, 2);
    testing::internal::CaptureStdout();
    EXPECT_EQ(massFunction(), 0);
    auto const text = testing::internal::GetCapturedStdout();
    std::istringstream rows(text);
    double mass, density;
    ASSERT_TRUE(bool(rows >> mass >> density));
    EXPECT_DOUBLE_EQ(mass, 2.5);
    EXPECT_DOUBLE_EQ(density, 0.001);
    ASSERT_TRUE(bool(rows >> mass >> density));
    EXPECT_DOUBLE_EQ(mass, 3.5);
    EXPECT_DOUBLE_EQ(density, 0.001);
    EXPECT_FALSE(bool(rows >> mass));
}

TEST_F(AnalysisTools, EqualMassCatalogueHasOneFiniteBin) {
    catalogue({{100.0f, true}, {100.0f, true}});
    testing::internal::CaptureStdout();
    EXPECT_EQ(massFunction(), 0);
    auto const text = testing::internal::GetCapturedStdout();
    std::istringstream rows(text);
    double mass, density;
    ASSERT_TRUE(bool(rows >> mass >> density));
    EXPECT_DOUBLE_EQ(mass, 2.0);
    EXPECT_DOUBLE_EQ(density, 0.002);
    EXPECT_FALSE(bool(rows >> mass));
}

TEST_F(AnalysisTools, EmptyCatalogueAndMissingRanksAreExplicit) {
    catalogue({});
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    EXPECT_EQ(massFunction(), 0);
    EXPECT_TRUE(testing::internal::GetCapturedStdout().empty());
    EXPECT_NE(testing::internal::GetCapturedStderr().find("No distinct halos"), std::string::npos);
    catalogue({{100.0f, true}, {1000.0f, true}, {10000.0f, true}}, 3);
    std::filesystem::remove("groups.4/groups.4.1.dat");
    EXPECT_THROW(massFunction(), std::runtime_error);
}

TEST_F(AnalysisTools, ProjectionPlacesNorthAboveSouth) {
    std::vector<float> pixels(48, 0.0f);
    std::fill_n(pixels.begin(), 4, 1.0f);
    map(2, 48, pixels);
    ASSERT_EQ(convertMap(), 0);
    std::unique_ptr<DBfile, decltype(&DBClose)> file(
        DBOpen("view.silo", DB_UNKNOWN, DB_READ), &DBClose);
    ASSERT_TRUE(file);
    std::unique_ptr<DBquadvar, decltype(&DBFreeQuadvar)> field(
        DBGetQuadvar(file.get(), "intensity"), &DBFreeQuadvar);
    ASSERT_TRUE(field);
    ASSERT_EQ(field->datatype, DB_FLOAT);
    double north = 0.0;
    double south = 0.0;
    auto const* values = static_cast<float const*>(field->vals[0]);
    for (int y = 0; y < field->dims[1]; ++y) {
        for (int x = 0; x < field->dims[0]; ++x) {
            if (y < field->dims[1] / 2) {
                south += values[y * field->dims[0] + x];
            } else {
                north += values[y * field->dims[0] + x];
            }
        }
    }
    EXPECT_GT(north, 0.0);
    EXPECT_EQ(south, 0.0);
}

TEST_F(AnalysisTools, BadMapHeadersTruncationAndOutputFailuresAreRejected) {
    map(0, 0, {});
    EXPECT_THROW(convertMap(), std::runtime_error);
    map(8193, 0, {});
    EXPECT_THROW(convertMap(), std::runtime_error);
    map(1, 13, std::vector<float>(13));
    EXPECT_THROW(convertMap(), std::runtime_error);
    map(1, 12, std::vector<float>(11));
    EXPECT_THROW(convertMap(), std::runtime_error);
    map(1, 12, std::vector<float>(12));
    EXPECT_THROW(convertMap("missing/view.silo"), std::runtime_error);
}
