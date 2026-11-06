#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <cosmictiger/core/particle_count.hpp>
#include <cosmictiger/math/cell_lattice.hpp>
#include <cosmictiger/scenarios/kelvin_helmholtz_lattice.hpp>

namespace detail {

void require(bool condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <class Operation>
void rejected(Operation&& operation) {
    bool threw = false;
    try {
        operation();
    } catch (std::invalid_argument const&) {
        threw = true;
    }
    require(threw, "Invalid lattice input was accepted");
}

// Independent radial-neighborhood check on a 7-cell stencil. All shells with
// r <= 2 cell widths are completely contained, regardless of the sink offset.
bool equalNeighborShells(cell_lattice::CellLattice const& lattice, int dimension) {
    std::vector<std::map<int, int>> shells;
    auto const offsets = lattice.offsets(dimension);
    for (auto const& sink : offsets) {
        std::map<int, int> histogram;
        for (int x = -3; x <= 3; ++x) {
            for (int y = dimension >= 2 ? -3 : 0; y <= (dimension >= 2 ? 3 : 0); ++y) {
                for (int z = dimension >= 3 ? -3 : 0; z <= (dimension >= 3 ? 3 : 0); ++z) {
                    for (auto const& source : offsets) {
                        std::array<int, 3> const cell{x, y, z};
                        int radiusSquared = 0;
                        for (int axis = 0; axis < dimension; ++axis) {
                            int const delta = 2 * cell[axis] + int(2 * (source[axis] - sink[axis]));
                            radiusSquared += delta * delta;
                        }
                        if (radiusSquared <= 16) {
                            ++histogram[radiusSquared];
                        }
                    }
                }
            }
        }
        shells.push_back(histogram);
    }
    return std::all_of(
        shells.begin(), shells.end(), [&](auto const& shell) { return shell == shells.front(); });
}

void allLattices() {
    for (unsigned mask = 1; mask < 16; ++mask) {
        std::string sites;
        for (int bit = 0; bit < 4; ++bit) {
            if (mask & (1U << bit)) {
                sites += "CFEV"[bit];
            }
        }
        cell_lattice::CellLattice const lattice(sites);
        std::reverse(sites.begin(), sites.end());
        require(cell_lattice::CellLattice(sites).sites() == lattice.sites(), "Site order matters");
        for (int dimension = 1; dimension <= 3; ++dimension) {
            bool const applicable = (dimension == 3 || sites.find('F') == std::string::npos) &&
                (dimension >= 2 || sites.find('E') == std::string::npos);
            require(lattice.appliesToDimension(dimension) == applicable,
                "Wrong dimensional applicability");
            if (!applicable) {
                require(!lattice.isUniform(dimension), "Inapplicable lattice reported uniform");
                rejected([&] { lattice.offsets(dimension); });
                continue;
            }
            size_t expected = 0;
            for (char site : sites) {
                expected += site == 'E' ? dimension : site == 'F' ? 3 : 1;
            }
            require(lattice.sitesPerCell(dimension) == expected, "Wrong shared-site count");
            require(lattice.isUniform(dimension) == equalNeighborShells(lattice, dimension),
                "Symmetry and independent neighbor shells disagree for " + sites);
            std::set<std::array<double, 3>> points;
            lattice.generate(dimension, {2, dimension >= 2 ? 2 : 1, dimension >= 3 ? 2 : 1}, 0.25,
                {0.5, 0.0, 0.0}, [&](auto const& point) {
                    require(point[0] >= 0.5 && point[0] < 1.0,
                        "Generated point left its half-open region");
                    require(points.insert(point).second, "Duplicate lattice point");
                });
            require(
                points.size() == expected * (size_t(1) << dimension), "Wrong generated population");
        }
    }
    require(cell_lattice::CellLattice("vCc").sites() == "CV", "Case normalization failed");
    require(!cell_lattice::CellLattice("C").appliesToDimension(0), "Accepted dimension zero");
    require(!cell_lattice::CellLattice("C").appliesToDimension(4), "Accepted dimension four");
    for (auto const* sites : {"", "X", "CV?", "C1", "C V"}) {
        rejected([&] { cell_lattice::CellLattice lattice(sites); });
    }
}

void packingPairs() {
    for (auto const* pair :
        {"CFEV/FV", "CFEV/CE", "VF/CV", "CE/CV", "CV/V", "CV/C", "FE/E", "FE/F"}) {
        auto const packing = scenarios::parseKelvinHelmholtzPacking(pair, 3);
        require(packing.high.sitesPerCell(3) == 2 * packing.low.sitesPerCell(3), "Wrong 3D ratio");
    }
    for (auto const* pair : {"CEV/CV", "CEV/E", "CV/C", "CV/V", "E/C", "E/V"}) {
        scenarios::parseKelvinHelmholtzPacking(pair, 2);
    }
    for (auto const* separator : {"/", "-", "_", ":", ",", " ", "|", " -> ", ".", "::", "=", "#"}) {
        auto const pair =
            scenarios::parseKelvinHelmholtzPacking(std::string(" veC") + separator + "cV ", 2);
        require(pair.name() == "CEV/CV", "Case/separator normalization failed");
    }
    for (auto const* pair : {"VE/CV", "CF/CV", "CV/E", "V/CV", "C/V", "CFEV/C", "X/V", "CV", "/CV",
             "CV/", "CV/V/C", "CV1V", "CV/V/", "CV/1", ""}) {
        rejected([&] { scenarios::parseKelvinHelmholtzPacking(pair, 3); });
    }
    rejected([] { scenarios::parseKelvinHelmholtzPacking("FV/CV", 2); });
    rejected([] { scenarios::parseKelvinHelmholtzPacking("CEV/CV", 3); });
    rejected([] { scenarios::parseKelvinHelmholtzPacking("E/V", 1); });
}

void rounding() {
    for (size_t sites : {1, 2, 3, 4}) {
        if (sites > (size_t(1) << kSpatialDimensions)) {
            continue;
        }
        for (size_t requested : {1, 47, 48, 49, 999, 1000, 1001, 262144}) {
            auto const layout = particle_count::roundKelvinHelmholtz(requested, sites);
            size_t const multiplicity = 3 * sites * (size_t(1) << (kSpatialDimensions - 1));
            require(layout.count >= requested, "Population undershot request");
            require(layout.count == multiplicity * particle_count::spatialPower(layout.side),
                "Wrong rounded count");
            require(layout.highCount == 2 * layout.lowCount && layout.count == 3 * layout.lowCount,
                "Wrong layer ratio");
            require(layout.side == 1 ||
                    multiplicity * particle_count::spatialPower(layout.side - 1) < requested,
                "Skipped smaller valid cell count");
            require(particle_count::roundKelvinHelmholtz(layout.count, sites).count == layout.count,
                "Rounding not idempotent");
        }
    }
    rejected([] { particle_count::roundKelvinHelmholtz(1000, 0); });
    bool overflowed = false;
    try {
        particle_count::roundKelvinHelmholtz(std::numeric_limits<size_t>::max(), 1);
    } catch (std::overflow_error const&) {
        overflowed = true;
    }
    require(overflowed, "Population overflow not rejected");
}

}    // namespace detail

int main() {
    try {
        detail::allLattices();
        detail::packingPairs();
        detail::rounding();
        std::cout << "All cell subsets, neighborhood symmetries, packing inputs and "
                  << kSpatialDimensions << "D counts passed\n";
    } catch (std::exception const& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
