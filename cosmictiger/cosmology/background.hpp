// Cached cosmological background histories and their diagnostic plot output.
// Copyright (C) 2026 Dominic C. Marcello; BSL-1.0 (see LICENSE_1_0.txt).

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

// Write the full configured history, including the signed physical da/dt.
// Rendering failures leave the text samples available and report a warning.
void writeBackgroundPlot(double finalConformalTime);

}    // namespace cosmology
