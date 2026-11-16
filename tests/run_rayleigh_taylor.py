#!/usr/bin/env python3
# Exercise the Rayleigh-Taylor preset and scenario defaults without self-gravity.
# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
#
# Distributed under the Boost Software License, Version 1.0.
# See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.

"""Check two- and three-dimensional Rayleigh-Taylor application evolution."""

import argparse
import math
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True

from checkpoint_format import decode_checkpoint, require
from run_checkpoint import checkManifest, prepareCase, runSimulation


def checkOption(output, key, value, name):
    require(re.search(r"^" + re.escape(key) + r"\s*:\s*" + re.escape(str(value)) + r"\s*$",
                      output, re.MULTILINE) is not None,
            f"{name}: resolved {key} is not {value}")


def readDiagnostics(path, width):
    rows = [list(map(float, line.split())) for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    require(rows and all(len(row) == width and all(math.isfinite(value) for value in row)
                         for row in rows),
            f"{path}: missing, malformed, or nonfinite diagnostics")
    require(rows[-1][0] > rows[0][0], f"{path}: diagnostics did not advance in time")
    return rows


def checkState(checkpoint, iteration, dimension, duration, acceleration):
    checkManifest(checkpoint, iteration, scenario="rayleighTaylor", endTime=duration,
                  gravity=False)
    state = decode_checkpoint((checkpoint / "checkpoint.0.dat").read_bytes())
    require(state.hydro and state.count == state.gas_count > 0,
            f"{checkpoint}: checkpoint is not an all-gas simulation")
    require(state.dimension_count == dimension and state.storage_dimensions == dimension,
            f"{checkpoint}: wrong spatial dimension or padded particle storage")
    require(len(state.positions) == dimension and
            len(state.velocities) == dimension * state.count,
            f"{checkpoint}: wrong position or velocity extent")
    require(state.algorithms[1:4] == (0, 0, 0) and state.sph_parameters[-1] == 0.0,
            f"{checkpoint}: self-gravity is enabled in checkpoint metadata")
    require(getattr(state, "z_plane_acceleration", None) == acceleration,
            f"{checkpoint}: imposed acceleration was lost")
    require(state.driver.iteration == iteration and
            math.isfinite(state.driver.tau) and 0.0 < state.driver.tau <= duration,
            f"{checkpoint}: invalid evolved iteration or time")
    require(state.entropy_initialized == 1 and state.thermal_scale_factor == 1.0,
            f"{checkpoint}: invalid thermal state")
    require(all(math.isfinite(value) for value in state.velocities),
            f"{checkpoint}: nonfinite velocity")
    for field in ("densities", "entropies", "internal_energies", "smoothing_lengths"):
        require(all(math.isfinite(value) and value > 0.0 for value in getattr(state, field)),
                f"{checkpoint}: nonpositive or nonfinite {field}")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--dimension", type=int, choices=(2, 3), required=True)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    root = args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    executable = args.executable.resolve()
    preset = Path(__file__).resolve().parents[1] / "configurations" / "rayleigh_taylor.ini"
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    duration = 1.0e-5
    acceleration = 0.5
    particleCount = 384 if args.dimension == 2 else 768
    common = [f"--hpx:threads={args.threads}", "--hpx:bind=none",
              "--run.darkMatterParticleCount=0", f"--run.gasParticleCount={particleCount}",
              f"--run.endTime={duration}", "--run.maxIterations=2", "--run.numberOfOutputs=3",
              "--run.verbose=true", "--output.directory=.", "--output.silo=false",
              "--checkpoint.intervalSeconds=0", "--analysis.saveForce=true",
              f"--hydro.dimensions={args.dimension}", "--gravity.cuda=false",
              "--hydro.cuda=false", "--drift.cuda=false"]
    # Select CPU backends and the runtime dimension while leaving gravity.enabled
    # and hydro.enabled untouched so the preset and scenario select the physics.
    cases = (("preset", [f"--configuration={preset}"]),
             ("scenario_defaults", ["--run.scenario=rayleighTaylor",
                                     f"--scenario.rayleighTaylor.zPlaneAcceleration={acceleration}"]))
    for name, options in cases:
        directory = prepareCase(root, name)
        output = runSimulation([str(executable), *options, *common], directory, "run.log", timeout=180)
        for key, value in (("run.scenario", "rayleighTaylor"),
                           ("build.dimensionCount", args.dimension),
                           ("hydro.dimensions", args.dimension), ("gravity.enabled", "F"),
                           ("hydro.enabled", "T"), ("gravity.cuda", "F"),
                           ("hydro.cuda", "F"), ("drift.cuda", "F")):
            checkOption(output, key, value, name)
        require("Do kick\n" not in output, f"{name}: invoked a gravity kick")
        energy = readDiagnostics(directory / "energy.dat", 9)
        require(all(row[7] == 0.0 for row in energy),
                f"{name}: nonzero self-gravity potential energy")
        require(all(row[4] > 0.0 and row[5] > 0.0 for row in energy),
                f"{name}: nonpositive kinetic or thermal energy")
        density = readDiagnostics(directory / "rho.dat", 2)
        require(all(row[1] > 0.0 for row in density), f"{name}: nonpositive maximum density")
        first, final = [checkState(directory / f"checkpoint.{iteration}", iteration,
                                   args.dimension, duration, acceleration)
                        for iteration in (1, 2)]
        require(final.count == first.count and final.driver.tau > first.driver.tau,
                f"{name}: particle count changed or checkpoint time did not advance")
        # Compare unordered positions so a tree reorder cannot masquerade as drift.
        require(sorted(zip(*first.positions)) != sorted(zip(*final.positions)),
                f"{name}: particles did not move between evolved checkpoints")
        print(f"{name}: D={args.dimension}, {final.count} gas particles, "
              f"time={final.driver.tau:.6e}; imposed field retained, self-gravity disabled",
              flush=True)


if __name__ == "__main__":
    main()
