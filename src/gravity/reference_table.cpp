/* CosmicTiger - symmetry-compressed periodic gravity reference; BSL-1.0. */

#include <array>
#include <vector>

#include <cosmictiger/gravity/direct_reference.hpp>
#include <cosmictiger/gravity/reference_table.hpp>

namespace detail {

std::vector<std::array<double, 3>> const& gravityReferenceCorrectionTable() {
    static auto const values = [] {
        std::vector<std::array<double, 3>> table(kGravityReferenceTableSize);
        auto const& exact = gravityReferenceTables();
        constexpr double kSpacing = 0.5 / kGravityReferenceTableIntervals;
        for (int i = 0; i <= kGravityReferenceTableMaximumIndex; ++i) {
            for (int j = 0; j <= i; ++j) {
                for (int k = 0; k <= j; ++k) {
                    std::array<double, 3> const position{i * kSpacing, j * kSpacing, k * kSpacing};
                    auto force = gravityReferenceCorrection(position, exact);
                    gravityReferenceTableEqualComponents(position.data(), force);
                    for (int dim = 0; dim < 3; ++dim) {
                        if (position[dim] == 0) {
                            force[dim] = 0;
                        }
                    }
                    table[gravityReferenceTableIndex(i, j, k)] = force;
                }
            }
        }
        return table;
    }();
    return values;
}

}    // namespace detail
