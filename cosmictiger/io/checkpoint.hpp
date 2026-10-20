/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 Distributed under the Boost Software License, Version 1.0.
 See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.
 */

#pragma once

#include <cosmictiger/core/containers.hpp>
#include <cosmictiger/core/driver.hpp>

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
