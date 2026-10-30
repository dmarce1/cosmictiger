/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <string>

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/driver.hpp>

struct CheckpointSelection {
    std::string farMethod;
    std::string nearMethod;
    std::string timeIntegrator;
    int expansionOrder{};
};

// Read only algorithm selection before resolving CLI/INI defaults at startup.
CheckpointSelection readCheckpointSelection(std::string const& outputDirectory, int iteration);

// Reject stale checkpoint paths before a fresh simulation writes any output.
void validateCheckpointOutput();

// Checkpoint directories: one rank file per locality plus a manifest, written
// to a staging directory and renamed into place once every rank has synced.
void writeCheckpoint(DriverParameters params);
DriverParameters readCheckpoint();
// Redshifts listed in checkpoint.txt, one per line.
Vector<double> readCheckpointList();
// Round-trips the driver state through a checkpoint directory (--run.test).
void checkpointStateTest();

namespace detail {
// Symlink checkpoint.z.<redshift> to checkpoint.<iteration> on rank 0.
void checkpointRedshiftAlias(int iteration, double redshift);
}    // namespace detail
