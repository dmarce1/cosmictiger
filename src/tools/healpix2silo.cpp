/* CosmicTiger - HEALPix view to Mollweide Silo conversion; BSL-1.0. */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <chealpix.h>
#include <silo.h>

namespace detail {

template <class T>
void readValues(FILE* input, T* values, size_t count) {
    if (std::fread(values, sizeof(T), count, input) != count) {
        throw std::runtime_error("Truncated HEALPix input");
    }
}

void checkSilo(int status, char const* operation) {
    if (status < 0) {
        throw std::runtime_error(std::string("Silo conversion failed: ") + operation);
    }
}

bool cartToSph(double x, double y, double& latitude, double& longitude) {
    double const theta = std::asin(y);
    double const argument = (2.0 * theta + std::sin(2.0 * theta)) / std::numbers::pi;
    latitude = std::asin(std::clamp(argument, -1.0, 1.0));
    longitude = std::numbers::pi * x / (2.0 * std::cos(theta));
    return longitude >= -std::numbers::pi && longitude <= std::numbers::pi;
}

int convert(int argc, char** argv) {
    std::string infile, outfile;
    namespace po = boost::program_options;
    po::options_description options("options");
    options.add_options()("help", "produce help message")(
        "input.file", po::value<std::string>(&infile), "input view.N.dat file")(
        "output.file", po::value<std::string>(&outfile), "output Silo file");
    po::variables_map variables;
    po::store(po::parse_command_line(argc, argv, options), variables);
    po::notify(variables);
    if (variables.count("help")) {
        std::cout << options << '\n';
        return 0;
    }
    if (infile.empty() || outfile.empty()) {
        throw std::invalid_argument("Both input.file and output.file are required");
    }
    std::unique_ptr<FILE, decltype(&fclose)> input(fopen(infile.c_str(), "rb"), &fclose);
    if (!input) {
        throw std::runtime_error("Cannot open " + infile);
    }
    int nside;
    int size;
    readValues(input.get(), &nside, 1);
    readValues(input.get(), &size, 1);
    if (nside <= 0 || nside > 8192 || size != 12 * nside * nside) {
        throw std::runtime_error(
            "Invalid HEALPix header: require 1<=nside<=8192 and size=12*nside^2");
    }
    std::vector<float> healpixData(size);
    readValues(input.get(), healpixData.data(), healpixData.size());
    int cycle = 0;
    double time = 0.0;
    int const nextByte = fgetc(input.get());
    if (nextByte != EOF) {
        ungetc(nextByte, input.get());
        readValues(input.get(), &cycle, 1);
        readValues(input.get(), &time, 1);
        if (fgetc(input.get()) != EOF) {
            throw std::runtime_error("Unexpected data after HEALPix time metadata");
        }
    }
    if (ferror(input.get())) {
        throw std::runtime_error("Unable to read HEALPix input");
    }
    if (fclose(input.release()) != 0) {
        throw std::runtime_error("Unable to close HEALPix input");
    }
    int const resolution = int(nside * std::sqrt(1.5) + 0.5);
    constexpr int kOrder = 3;
    std::vector<float> mollweide;
    mollweide.reserve(size_t(resolution) * resolution * 8);
    for (int iy = -resolution; iy < resolution; ++iy) {
        for (int ix = -2 * resolution; ix < 2 * resolution; ++ix) {
            double value = 0.0;
            for (int nx = 0; nx < kOrder; ++nx) {
                for (int ny = 0; ny < kOrder; ++ny) {
                    double const x = (ix + (nx + 0.5) / kOrder) / resolution;
                    double const y = (iy + (ny + 0.5) / kOrder) / resolution;
                    double latitude, longitude;
                    if (cartToSph(x, y, latitude, longitude)) {
                        long pixel;
                        ang2pix_ring(nside, std::numbers::pi / 2.0 - latitude, longitude, &pixel);
                        value += healpixData[pixel];
                    }
                }
            }
            mollweide.push_back(value);
        }
    }
    std::unique_ptr<DBfile, decltype(&DBClose)> file(
        DBCreate(outfile.c_str(), DB_CLOBBER, DB_LOCAL, nullptr, DB_PDB), &DBClose);
    if (!file) {
        throw std::runtime_error("Cannot create Silo output " + outfile);
    }
    std::unique_ptr<DBoptlist, decltype(&DBFreeOptlist)> attributes(
        DBMakeOptlist(3), &DBFreeOptlist);
    if (!attributes) {
        throw std::runtime_error("Cannot allocate Silo options");
    }
    float ftime = time;
    checkSilo(DBAddOption(attributes.get(), DBOPT_CYCLE, &cycle), "cycle");
    checkSilo(DBAddOption(attributes.get(), DBOPT_TIME, &ftime), "time");
    checkSilo(DBAddOption(attributes.get(), DBOPT_DTIME, &time), "time");
    std::vector<float> x, y;
    for (int ix = -2 * resolution; ix <= 2 * resolution; ++ix) {
        x.push_back(double(ix) / resolution);
    }
    for (int iy = -resolution; iy <= resolution; ++iy) {
        y.push_back(double(iy) / resolution);
    }
    constexpr int kDimensions = 2;
    int const meshDimensions[kDimensions] = {4 * resolution + 1, 2 * resolution + 1};
    int const fieldDimensions[kDimensions] = {4 * resolution, 2 * resolution};
    float const* coordinates[kDimensions] = {x.data(), y.data()};
    char const* coordinateNames[kDimensions] = {"x", "y"};
    checkSilo(DBPutQuadmesh(file.get(), "mesh", coordinateNames, coordinates, meshDimensions,
                  kDimensions, DB_FLOAT, DB_COLLINEAR, attributes.get()),
        "mesh");
    checkSilo(DBPutQuadvar1(file.get(), "intensity", "mesh", mollweide.data(), fieldDimensions,
                  kDimensions, nullptr, 0, DB_FLOAT, DB_ZONECENT, attributes.get()),
        "intensity");
    checkSilo(DBClose(file.release()), "close");
    return 0;
}

}    // namespace detail

int main(int argc, char** argv) {
    try {
        return detail::convert(argc, argv);
    } catch (std::exception const& error) {
        std::cerr << "healpix2silo: " << error.what() << '\n';
        return 1;
    }
}
