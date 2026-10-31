/* CosmicTiger - distinct-halo mass function; BSL-1.0. */

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <boost/program_options.hpp>

#include <cosmictiger/analysis/group_entry.hpp>
#include <cosmictiger/core/file_handle.hpp>

namespace detail {

bool positiveFinite(double value) {
    return value > 0.0 &&
        (std::bit_cast<uint64_t>(value) & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}

UniqueFile openInput(std::filesystem::path const& path) {
    UniqueFile input(fopen(path.c_str(), "rb"));
    if (!input) {
        throw std::runtime_error("Cannot open " + path.string());
    }
    return input;
}

int haloFunction(int argc, char* argv[]) {
    int snapshot;
    int numberOfBins;
    double boxSize = 0.0;
    namespace po = boost::program_options;
    po::options_description options("options");
    options.add_options()("help", "produce help message")("input.groupNumber",
        po::value<int>(&snapshot)->default_value(-1),
        "input number")("analysis.boxSize", po::value<double>(&boxSize),
        "override catalogue box length in Mpc")("analysis.numberOfBins",
        po::value<int>(&numberOfBins)->default_value(1000), "number of logarithmic mass bins");
    po::variables_map variables;
    po::store(po::parse_command_line(argc, argv, options), variables);
    po::notify(variables);
    if (variables.count("help")) {
        std::cout << options << '\n';
        return 0;
    }
    if (snapshot < 0 || numberOfBins <= 0) {
        throw std::invalid_argument(
            "A nonnegative input.groupNumber and positive analysis.numberOfBins are required");
    }
    auto const directory = std::filesystem::path("groups." + std::to_string(snapshot));
    auto headerFile = openInput(directory / "groups.dat");
    GroupHeader header{};
    if (fread(&header, sizeof(header), 1, headerFile.get()) != 1 || header.number != snapshot ||
        std::filesystem::file_size(directory / "groups.dat") != sizeof(header)) {
        throw std::runtime_error("Invalid native group catalogue header");
    }
    if (!variables.count("analysis.boxSize")) {
        boxSize = header.boxLength;
    }
    if (!positiveFinite(boxSize)) {
        throw std::invalid_argument("Catalogue box length must be finite and positive");
    }
    int version;
    int ranks;
    size_t expectedCount;
    std::ifstream manifest(directory / "rockstar_hosts.complete");
    if (!(manifest >> version >> ranks >> expectedCount) || version != 1 || ranks <= 0) {
        throw std::runtime_error("Missing or invalid ROCKSTAR completion manifest");
    }
    std::string extra;
    if (manifest >> extra) {
        throw std::runtime_error("Unexpected ROCKSTAR completion manifest fields");
    }
    std::vector<double> masses;
    size_t totalCount = 0;
    for (int rank = 0; rank < ranks; ++rank) {
        auto const hostPath = directory / ("rockstar_hosts." + std::to_string(rank) + ".dat");
        std::ifstream hosts(hostPath);
        if (!hosts) {
            throw std::runtime_error("Cannot open " + hostPath.string());
        }
        std::unordered_map<GroupInteger, GroupInteger> hostById;
        GroupInteger id;
        GroupInteger host;
        while (hosts >> id) {
            if (!(hosts >> host) || id < 0 || host < 0 || !hostById.emplace(id, host).second) {
                throw std::runtime_error("Malformed ROCKSTAR host catalogue: " + hostPath.string());
            }
        }
        if (!hosts.eof()) {
            throw std::runtime_error("Unable to read " + hostPath.string());
        }
        auto input = openInput(directory /
            ("groups." + std::to_string(snapshot) + "." + std::to_string(rank) + ".dat"));
        for (;;) {
            int const firstByte = fgetc(input.get());
            if (firstByte == EOF) {
                if (ferror(input.get())) {
                    throw std::runtime_error("Group catalogue read failure");
                }
                break;
            }
            ungetc(firstByte, input.get());
            GroupEntry entry;
            if (!entry.read(input.get())) {
                throw std::runtime_error("Truncated group catalogue identifier");
            }
            auto const found = hostById.find(entry.id);
            if (found == hostById.end()) {
                throw std::runtime_error("Group catalogue and ROCKSTAR host identifiers disagree");
            }
            // GroupEntry masses are M_sun (unlike native records' M_sun/h).
            // Joining persistent host IDs retains physical mass/volume units.
            if (found->second == std::numeric_limits<GroupInteger>::max()) {
                if (!positiveFinite(entry.mass)) {
                    throw std::runtime_error("Distinct halo has a nonpositive or nonfinite mass");
                }
                masses.push_back(entry.mass);
            }
            hostById.erase(found);
            ++totalCount;
        }
        if (!hostById.empty()) {
            throw std::runtime_error(
                "Group catalogue is missing entries named by its host catalogue");
        }
    }
    if (totalCount != expectedCount) {
        throw std::runtime_error("Catalogue count does not match its completion manifest");
    }
    if (masses.empty()) {
        std::cerr << "No distinct halos in catalogue\n";
        return 0;
    }
    auto const limits = std::minmax_element(masses.begin(), masses.end());
    double logmin = std::log10(*limits.first);
    double logmax = std::log10(*limits.second);
    if (logmin == logmax) {
        // A finite, explicitly one-dex bin for a single distinct mass.
        numberOfBins = 1;
        logmin -= 0.5;
        logmax += 0.5;
    }
    double const dlog = (logmax - logmin) / numberOfBins;
    std::vector<size_t> bins(numberOfBins, 0);
    for (double mass : masses) {
        int const bin = std::min(int((std::log10(mass) - logmin) / dlog), numberOfBins - 1);
        ++bins[bin];
    }
    double const volume = boxSize * boxSize * boxSize;
    if (!positiveFinite(volume)) {
        throw std::invalid_argument("Box volume is outside the finite positive range");
    }
    std::cout << std::scientific << std::setprecision(12);
    for (int i = 0; i < numberOfBins; ++i) {
        std::cout << (i + 0.5) * dlog + logmin << ' ' << bins[i] / dlog / volume << '\n';
    }
    if (!std::cout) {
        throw std::runtime_error("Unable to write halo mass function");
    }
    return 0;
}

}    // namespace detail

int main(int argc, char* argv[]) {
    try {
        return detail::haloFunction(argc, argv);
    } catch (std::exception const& error) {
        std::cerr << "groups2halofunc: " << error.what() << '\n';
        return 1;
    }
}
