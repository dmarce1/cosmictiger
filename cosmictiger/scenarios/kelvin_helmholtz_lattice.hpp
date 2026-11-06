#pragma once

#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>

#include <cosmictiger/math/cell_lattice.hpp>

namespace scenarios {

struct KelvinHelmholtzPacking {
    cell_lattice::CellLattice high;
    cell_lattice::CellLattice low;

    std::string name() const {
        return high.sites() + "/" + low.sites();
    }
};

// Site runs are case insensitive; any nonempty run of punctuation/whitespace
// separates them. Reject extra runs, unknown letters, digits and missing sides.
inline KelvinHelmholtzPacking parseKelvinHelmholtzPacking(std::string_view input, int dimension) {
    while (!input.empty() && std::isspace(static_cast<unsigned char>(input.front()))) {
        input.remove_prefix(1);
    }
    while (!input.empty() && std::isspace(static_cast<unsigned char>(input.back()))) {
        input.remove_suffix(1);
    }
    auto const isLetter = [](char value) {
        return std::isalpha(static_cast<unsigned char>(value));
    };
    size_t highEnd = 0;
    while (highEnd < input.size() && isLetter(input[highEnd])) {
        ++highEnd;
    }
    size_t lowBegin = highEnd;
    while (lowBegin < input.size() &&
        (std::ispunct(static_cast<unsigned char>(input[lowBegin])) ||
            std::isspace(static_cast<unsigned char>(input[lowBegin])))) {
        ++lowBegin;
    }
    size_t lowEnd = lowBegin;
    while (lowEnd < input.size() && isLetter(input[lowEnd])) {
        ++lowEnd;
    }
    if (highEnd == 0 || lowBegin == highEnd || lowBegin == lowEnd || lowEnd != input.size()) {
        throw std::invalid_argument("initial.kelvinHelmholz.packing requires two site strings, "
                                    "high/low (for example CV/C)");
    }
    KelvinHelmholtzPacking result{cell_lattice::CellLattice(input.substr(0, highEnd)),
        cell_lattice::CellLattice(input.substr(lowBegin))};
    for (auto const* lattice : {&result.high, &result.low}) {
        if (!lattice->appliesToDimension(dimension)) {
            throw std::invalid_argument("initial.kelvinHelmholz.packing: " + lattice->sites() +
                " is not applicable in " + std::to_string(dimension) + "D");
        }
        if (!lattice->isUniform(dimension)) {
            throw std::invalid_argument("initial.kelvinHelmholz.packing: " + lattice->sites() +
                " has nonuniform site neighborhoods in " + std::to_string(dimension) + "D");
        }
    }
    if (result.high.sitesPerCell(dimension) != 2 * result.low.sitesPerCell(dimension)) {
        throw std::invalid_argument(
            "initial.kelvinHelmholz.packing must have a 2:1 high/low site count");
    }
    return result;
}

}    // namespace scenarios
