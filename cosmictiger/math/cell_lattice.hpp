#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace cell_lattice {

// A periodic Cartesian cell, with each shared boundary site owned by the cell
// on its positive side. Offsets therefore contain only 0 and 1/2, never 1.
// C = cell center, F = face centers (3D), E = edge centers (2D/3D), V = vertices.
class CellLattice {
public:
    using Position = std::array<double, 3>;
    using CellCounts = std::array<int, 3>;

    /// Parse a nonempty set of center/face/edge/vertex site letters, accepting either
    /// case and rejecting unknown sites.
    explicit CellLattice(std::string_view sites) {
        if (sites.empty()) {
            throw std::invalid_argument("A cell lattice must contain at least one of C, F, E, V");
        }
        for (unsigned char letter : sites) {
            char const upper = char(std::toupper(letter));
            auto const index = std::string_view("CFEV").find(upper);
            if (index == std::string_view::npos) {
                throw std::invalid_argument(
                    "Unknown cell lattice site: " + std::string(1, char(letter)));
            }
            selected_[index] = true;
        }
    }

    /// Return the selected site types in canonical CFEV order.
    std::string sites() const {
        std::string result;
        for (size_t index = 0; index < selected_.size(); ++index) {
            if (selected_[index]) {
                result += "CFEV"[index];
            }
        }
        return result;
    }

    /// Check whether all selected site types exist in the requested one-, two-, or
    /// three-dimensional lattice.
    bool appliesToDimension(int dimension) const {
        return dimension >= 1 && dimension <= 3 && (!selected_[1] || dimension == 3) &&
            (!selected_[2] || dimension >= 2);
    }

    /// List selected lattice-site offsets at integer or half-cell coordinates.
    std::vector<Position> offsets(int dimension) const {
        auto const occupied = occupiedSites(dimension);
        std::vector<Position> result;
        for (unsigned site = 0; site < (1U << dimension); ++site) {
            if (occupied[site]) {
                Position offset{};
                for (int axis = 0; axis < dimension; ++axis) {
                    offset[axis] = ((site >> axis) & 1U) * 0.5;
                }
                result.push_back(offset);
            }
        }
        return result;
    }

    /// Count the distinct occupied sites in a lattice cell of the requested dimension.
    size_t sitesPerCell(int dimension) const {
        auto const occupied = occupiedSites(dimension);
        return std::count(occupied.begin(), occupied.end(), true);
    }

    /// Count occupied integer/half-cell planes along an axis, rejecting an invalid
    /// axis.
    int planesPerCell(int dimension, int axis) const {
        if (axis < 0 || axis >= dimension) {
            throw std::invalid_argument("Cell lattice axis is outside its dimension");
        }
        bool phases[2]{};
        for (auto const& offset : offsets(dimension)) {
            phases[int(2 * offset[axis])] = true;
        }
        return int(phases[0]) + int(phases[1]);
    }

    // Exact geometric uniformity: every site must be related to the first by
    // an isometry of the entire periodic point set. Enumerate axis permutations
    // and half-cell translations modulo the cell. Reflections have the same
    // action as the corresponding unsigned permutation on these binary offsets.
    // Consequently all radial neighbor shells, and radial SPH kernel sums at a
    // common support radius, agree. This does not test an interface or prescribe
    // the kernel's absolute density normalization.
    bool isUniform(int dimension) const {
        if (!appliesToDimension(dimension)) {
            return false;
        }
        auto const occupied = occupiedSites(dimension);
        unsigned const first =
            unsigned(std::find(occupied.begin(), occupied.end(), true) - occupied.begin());
        for (unsigned target = 0; target < (1U << dimension); ++target) {
            if (!occupied[target]) {
                continue;
            }
            std::array<int, 3> permutation{0, 1, 2};
            bool equivalent = false;
            do {
                auto const permute = [&](unsigned site) {
                    unsigned result = 0;
                    for (int axis = 0; axis < dimension; ++axis) {
                        result |= ((site >> permutation[axis]) & 1U) << axis;
                    }
                    return result;
                };
                unsigned const translation = target ^ permute(first);
                bool preservesLattice = true;
                for (unsigned site = 0; site < (1U << dimension); ++site) {
                    if (occupied[site] != occupied[permute(site) ^ translation]) {
                        preservesLattice = false;
                        break;
                    }
                }
                equivalent = preservesLattice;
            } while (!equivalent &&
                std::next_permutation(permutation.begin(), permutation.begin() + dimension));
            if (!equivalent) {
                return false;
            }
        }
        return true;
    }

    // Emit all points in a rectangular collection of whole cells. Inactive
    // coordinates retain origin[axis]; inactive cell counts must be one.
    template <class Emit>
    void generate(int dimension, CellCounts const& cells, double spacing, Position const& origin,
        Emit&& emit) const {
        auto const cellOffsets = offsets(dimension);
        if (!(spacing > 0.0) || !std::isfinite(spacing)) {
            throw std::invalid_argument("Cell lattice spacing must be finite and positive");
        }
        for (int axis = 0; axis < 3; ++axis) {
            if (cells[axis] <= 0 || (axis >= dimension && cells[axis] != 1) ||
                !std::isfinite(origin[axis])) {
                throw std::invalid_argument("Invalid cell lattice extent or origin");
            }
        }
        for (int x = 0; x < cells[0]; ++x) {
            for (int y = 0; y < cells[1]; ++y) {
                for (int z = 0; z < cells[2]; ++z) {
                    CellCounts const cell{x, y, z};
                    for (auto const& offset : cellOffsets) {
                        Position position = origin;
                        for (int axis = 0; axis < dimension; ++axis) {
                            position[axis] += (cell[axis] + offset[axis]) * spacing;
                        }
                        emit(position);
                    }
                }
            }
        }
    }

private:
    std::array<bool, 4> selected_{};

    /// Build the occupied half-cell site mask after validating dimensional
    /// applicability.
    std::array<bool, 8> occupiedSites(int dimension) const {
        if (!appliesToDimension(dimension)) {
            throw std::invalid_argument("Cell lattice " + sites() + " is not applicable in " +
                std::to_string(dimension) + "D (F requires 3D; E requires 2D or 3D)");
        }
        std::array<bool, 8> occupied{};
        for (unsigned site = 0; site < (1U << dimension); ++site) {
            int const centers = std::popcount(site);
            size_t const kind = centers == dimension ? 0 : centers == 0 ? 3 : centers == 1 ? 2 : 1;
            occupied[site] = selected_[kind];
        }
        return occupied;
    }
};

}    // namespace cell_lattice
