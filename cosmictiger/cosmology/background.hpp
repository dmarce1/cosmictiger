#pragma once

#include <memory>
#include <optional>

#include <cosmictiger/cosmology/friedmann.hpp>

namespace cosmology {

// Immutable conformal-time history. Dense solver output preserves both branches
// of a recollapsing model without integrating again for individual lightcone hits.
class SimulationBackgroundTable {
    std::optional<FriedmannTrajectory> trajectory_;
    double fixedScale_{};
    double finalConformalTime_{};

public:
    SimulationBackgroundTable(
        FriedmannCosmology const& model, FriedmannState initialState, double finalConformalTime);
    SimulationBackgroundTable(double fixedScale, double finalConformalTime);

    FriedmannAdvance sample(double conformalTime) const;
    double finalConformalTime() const;
};

// Acquire once per output batch. A locality caches the complete original run
// history, so restarting at a later epoch keeps absolute emission times valid.
// Changed cosmological parameters or endpoint replace the cached table; readers
// holding an earlier table retain their immutable history.
std::shared_ptr<SimulationBackgroundTable const> simulationBackground(double finalConformalTime);

}    // namespace cosmology
