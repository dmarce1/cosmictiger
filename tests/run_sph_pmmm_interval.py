#!/usr/bin/env python3
#  CosmicTiger - A cosmological N-Body code
#  Copyright (C) 2026  Dominic C. Marcello
#
#  Distributed under the Boost Software License, Version 1.0.
#  See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.

"""Restart SPH with PMMM gravity inside an outer interval (docs/sph_parent_tree_plan.md, M8).

A small cosmology with gas and dark matter runs on the hierarchical integrator with
gravity.pmmmOuterInterval, on one locality and, with MPI, on two. A restart from a
checkpoint strictly inside an outer interval must finish that interval and cross the next
boundary exactly as the uninterrupted run does: the frozen keys, the long accelerations
and the interval clock come from the checkpoint. Fixed-time scenarios, which have no
outer-interval schedule, must reject the option at startup.
"""

import argparse
import json
from pathlib import Path
import re
import shutil
import sys

sys.dont_write_bytecode = True

from application_selection import cpu_backend_options
from checkpoint_format import decode_checkpoint, require
from run_checkpoint import prepareCase, runSimulation

BOUNDARY = re.compile(r"^PMMM boundary at tau=(\S+):", re.MULTILINE)


def load(directory, iteration, ranks):
    """Every particle of every rank, and the driver state of rank 0."""
    rows = []
    driver = None
    for rank in range(ranks):
        state = decode_checkpoint((directory / f"checkpoint.{iteration}" /
                                   f"checkpoint.{rank}.dat").read_bytes(), ranks=ranks)
        driver = driver or state.driver
        period = 1 << (8 * state.position_bytes)
        for particle in range(state.count):
            row = {"type": state.types[particle], "period": period,
                   "position": tuple(state.positions[axis][particle] for axis in range(3)),
                   "velocity": tuple(state.velocities[3 * particle:3 * particle + 3])}
            if row["type"] == 1:
                gas = state.gas_indices[particle]
                row.update(entropy=state.entropies[gas], smoothing_length=state.smoothing_lengths[gas],
                           density=state.densities[gas])
            rows.append(row)
    return rows, driver


def compare(label, reference, resumed):
    """Pair particles by species and position; a restart may reorder them."""
    require(len(reference) == len(resumed), f"{label}: particle counts differ")
    period = reference[0]["period"]
    cell = 1 << 12
    cells = period // cell
    grid = {}
    for index, row in enumerate(resumed):
        key = (row["type"],) + tuple(coordinate // cell for coordinate in row["position"])
        grid.setdefault(key, []).append(index)
    velocity_scale = max(abs(value) for row in reference for value in row["velocity"])
    worst = {"position_units": 0, "velocity": 0.0, "entropy": 0.0, "smoothing_length": 0.0,
             "density": 0.0}
    used = set()
    for row in reference:
        key = tuple(coordinate // cell for coordinate in row["position"])
        best = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbor = (row["type"], (key[0] + dx) % cells, (key[1] + dy) % cells,
                                (key[2] + dz) % cells)
                    for index in grid.get(neighbor, ()):
                        distance = max(min((a - b) % period, (b - a) % period)
                                       for a, b in zip(row["position"], resumed[index]["position"]))
                        if best is None or distance < best[0]:
                            best = (distance, index)
        require(best is not None and best[1] not in used, f"{label}: a particle has no counterpart")
        used.add(best[1])
        other = resumed[best[1]]
        worst["position_units"] = max(worst["position_units"], best[0])
        worst["velocity"] = max(worst["velocity"], max(
            abs(a - b) for a, b in zip(row["velocity"], other["velocity"])) / velocity_scale)
        if row["type"] == 1:
            for field in ("entropy", "smoothing_length", "density"):
                worst[field] = max(worst[field],
                                   abs(row[field] - other[field]) / max(abs(row[field]), 1e-30))
    # Pair sums run in a scheduling order, so velocities agree to float roundoff and a drift
    # may round a coordinate to a neighboring fixed-point unit (tests/run_checkpoint.py).
    require(worst["position_units"] <= 2, f"{label}: restart moved a position by "
            f"{worst['position_units']} fixed-point units")
    for field in ("velocity", "entropy", "smoothing_length", "density"):
        require(worst[field] <= 1.0e-5, f"{label}: restart changed {field} by {worst[field]:.3e}")
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--mpi-exec")
    parser.add_argument("--mpi-numproc-flag", default="-n")
    parser.add_argument("--mpi-preflag", action="append", default=[])
    parser.add_argument("--mpi-postflag", action="append", default=[])
    parser.add_argument("--hpx-argument", action="append", default=[])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    executable = args.executable.resolve()
    root = args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    configurations = Path(__file__).resolve().parents[1] / "configurations"
    iterations = 40
    common = {"configuration": configurations / "cosmology.ini",
              "run.timeIntegrator": "hierarchical", "gravity.method.near": "pp",
              "gravity.pmmmOuterInterval": "true", "gravity.meshSize": 16,
              "run.darkMatterParticleCount": 4096, "run.gasParticleCount": 4096,
              "chemistry.enabled": "false", "starFormation.enabled": "false",
              "output.silo": "false", "output.directory": ".", "checkpoint.intervalSeconds": 0,
              "run.verbose": "true",
              "analysis.views": "false", "analysis.power": "false", "analysis.groups": "false",
              "hpx:bind": "none", **cpu_backend_options()}

    def run(directory, options, ranks, success=True):
        command = [str(executable), *args.hpx_argument,
                   *[f"--{key}={value}" for key, value in options.items()],
                   f"--hpx:threads={max(1, args.threads // ranks)}"]
        if ranks > 1:
            command = [args.mpi_exec, args.mpi_numproc_flag, str(ranks), *args.mpi_preflag,
                       command[0], *args.mpi_postflag, *command[1:], f"--hpx:localities={ranks}"]
            if "--hpx:ini=hpx.parcel.mpi.enable=1" not in command:
                command.append("--hpx:ini=hpx.parcel.mpi.enable=1")
        return runSimulation(command, directory, "run.log", timeout=1500, expectSuccess=success)

    evidence = {}
    for ranks in [1] + ([2] if args.mpi_exec else []):
        label = f"ranks_{ranks}"
        reference = prepareCase(root, label)
        output = run(reference, dict(common, **{"run.maxIterations": iterations}), ranks)
        boundaries = [float(value) for value in BOUNDARY.findall(output)]
        require(len(boundaries) >= 2, f"{label}: the run crossed fewer than two outer boundaries")
        # The last checkpoint strictly inside the first interval, by the exact interval
        # clock (a nonzero integer time in interval 0; the checkpoint at the boundary has
        # integer time 0): the restart finishes that interval and crosses the next boundary.
        inside = None
        for iteration in range(1, iterations):
            _, driver = load(reference, iteration, ranks)
            if driver.outer_index == 0 and driver.integer_time != 0 and driver.outer_interval > 0.0:
                inside = (iteration, driver.tau)
        require(inside is not None, f"{label}: no checkpoint lies inside the first interval")
        restart, restart_tau = inside
        resumed = prepareCase(root, label + "_restart", reference / f"checkpoint.{restart}")
        output = run(resumed, dict(common, **{"run.maxIterations": iterations - restart,
                                              "checkpoint.readIndex": restart}), ranks)
        require("Done reading checkpoint" in output, f"{label}: the checkpoint was not read")
        final_reference, driver = load(reference, iterations, ranks)
        final_resumed, resumed_driver = load(resumed, iterations, ranks)
        require(resumed_driver.outer_index >= 1 and BOUNDARY.search(output),
                f"{label}: the resumed run crossed no outer boundary")
        for field in ("tau", "outer_start", "outer_interval", "outer_index"):
            require(getattr(driver, field) == getattr(resumed_driver, field),
                    f"{label}: restart changed the driver {field}")
        worst = compare(label, final_reference, final_resumed)
        gas = sum(row["type"] == 1 for row in final_reference)
        require(gas > 0 and gas < len(final_reference), f"{label}: the run lacks gas or dark matter")
        evidence[label] = dict(worst, restart=restart, boundaries=len(boundaries))
        print(f"{label}: restart at iteration {restart} (tau={restart_tau:.6g}) inside the first "
              f"interval; " + ", ".join(f"{key}={value:.3e}" if isinstance(value, float) else
                                        f"{key}={value}" for key, value in worst.items()),
              flush=True)

    rejected = prepareCase(root, "jeans_rejected")
    output = run(rejected, {"configuration": configurations / "jeans_wave.ini",
                            "run.timeIntegrator": "hierarchical", "gravity.method.near": "pp",
                            "gravity.pmmmOuterInterval": "true", "run.maxIterations": 1,
                            "output.directory": ".", "hpx:bind": "none", **cpu_backend_options()},
                 1, success=False)
    require("gravity.pmmmOuterInterval is not implemented for the fixed-time scenario" in output,
            "a fixed-time scenario did not reject the outer interval")
    (root / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print("SPH with PMMM gravity restarts inside an outer interval", flush=True)


if __name__ == "__main__":
    main()
