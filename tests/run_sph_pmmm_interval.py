#!/usr/bin/env python3
#  CosmicTiger - A cosmological N-Body code
#  Copyright (C) 2026  Dominic C. Marcello
#
#  Distributed under the Boost Software License, Version 1.0.
#  See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.

"""Restart SPH with PMMM gravity inside an outer interval (docs/sph_parent_tree_plan.md, M8).

Two runs use the hierarchical integrator with gravity.pmmmOuterInterval, on one locality
and, with MPI, on two: a small cosmology with gas and dark matter (cosmology driver) and
the linear Jeans wave (fixed-time scenario driver). A restart from a checkpoint strictly
inside an outer interval must finish that interval and cross the next boundary exactly as
the uninterrupted run does: the frozen keys, the long accelerations and the interval clock
come from the checkpoint.
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


def compare(label, reference, resumed, position_units, tolerances):
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
    require(worst["position_units"] <= position_units, f"{label}: restart moved a position by "
            f"{worst['position_units']} fixed-point units")
    for field in ("velocity", "entropy", "smoothing_length", "density"):
        require(worst[field] <= tolerances.get(field, 1.0e-5),
                f"{label}: restart changed {field} by {worst[field]:.3e}")
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
    base = {"run.timeIntegrator": "hierarchical", "gravity.method.near": "pp",
            "gravity.pmmmOuterInterval": "true", "gravity.meshSize": 16,
            "chemistry.enabled": "false", "starFormation.enabled": "false",
            "output.silo": "false", "output.directory": ".", "checkpoint.intervalSeconds": 0,
            "run.verbose": "true", "analysis.views": "false", "analysis.power": "false",
            "analysis.groups": "false", "hpx:bind": "none", **cpu_backend_options()}
    # (name, options, iterations, whether dark matter takes part, position units, tolerances).
    # Pair sums run in a scheduling order. The cosmology restart was bit-identical in every
    # run; a drift may still round a coordinate to a neighboring fixed-point unit
    # (tests/run_checkpoint.py). The Jeans wave's velocities are some 1e-8 of its sound
    # speed, so that roundoff is large against them: two uninterrupted runs differed by 228
    # position units and 3.4e-5 of the velocity scale (November 9, 2026). A lost frozen key or
    # long acceleration changes the velocities by order one, since gravity supplies half the
    # restoring force.
    cases = [("cosmology", dict(base, **{"configuration": configurations / "cosmology.ini",
                                         "run.darkMatterParticleCount": 4096,
                                         "run.gasParticleCount": 4096}), 40, True, 2, {}),
             ("jeans", dict(base, **{"configuration": configurations / "jeans_wave.ini",
                                     "run.gasParticleCount": 4096}), 30, False, 2048,
              {"velocity": 3.0e-4})]

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
    for name, options, iterations, dark_matter, position_units, tolerances in cases:
        for ranks in [1] + ([2] if args.mpi_exec else []):
            label = f"{name}_{ranks}"
            reference = prepareCase(root, label)
            output = run(reference, dict(options, **{"run.maxIterations": iterations}), ranks)
            boundaries = BOUNDARY.findall(output)
            require(len(boundaries) >= 2, f"{label}: the run crossed fewer than two outer boundaries")
            # The last checkpoint strictly inside an interval (nonzero integer time; a boundary
            # checkpoint has integer time 0) whose interval closes before the final iteration,
            # so the resumed run also performs the next boundary pass.
            clocks = [load(reference, iteration, ranks)[1] for iteration in range(1, iterations + 1)]
            inside = None
            for index, driver in enumerate(clocks[:-1]):
                closed = any(later.integer_time == 0 for later in clocks[index + 1:-1])
                if driver.integer_time != 0 and driver.outer_interval > 0.0 and closed:
                    inside = (index + 1, driver.tau, driver.outer_index)
            require(inside is not None, f"{label}: no checkpoint lies inside a closed interval")
            restart, restart_tau, restart_interval = inside
            resumed = prepareCase(root, label + "_restart", reference / f"checkpoint.{restart}")
            output = run(resumed, dict(options, **{"run.maxIterations": iterations - restart,
                                                   "checkpoint.readIndex": restart}), ranks)
            require("Done reading checkpoint" in output, f"{label}: the checkpoint was not read")
            final_reference, driver = load(reference, iterations, ranks)
            final_resumed, resumed_driver = load(resumed, iterations, ranks)
            require(resumed_driver.outer_index > restart_interval and BOUNDARY.search(output),
                    f"{label}: the resumed run crossed no outer boundary")
            for field in ("tau", "outer_start", "outer_interval", "outer_index", "integer_time"):
                require(getattr(driver, field) == getattr(resumed_driver, field),
                        f"{label}: restart changed the driver {field}")
            worst = compare(label, final_reference, final_resumed, position_units, tolerances)
            gas = sum(row["type"] == 1 for row in final_reference)
            require(gas > 0 and (gas < len(final_reference)) == dark_matter,
                    f"{label}: the run does not have the expected species")
            evidence[label] = dict(worst, restart=restart, boundaries=len(boundaries))
            print(f"{label}: restart at iteration {restart} (tau={restart_tau:.6g}) inside outer "
                  f"interval {restart_interval}; " + ", ".join(
                      f"{key}={value:.3e}" if isinstance(value, float) else f"{key}={value}"
                      for key, value in worst.items()), flush=True)

    (root / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print("SPH with PMMM gravity restarts inside an outer interval", flush=True)


if __name__ == "__main__":
    main()
