#!/usr/bin/env python3
#  CosmicTiger - A cosmological N-Body code
#  Copyright (C) 2026  Dominic C. Marcello
#
#  Distributed under the Boost Software License, Version 1.0.
#  See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.

"""Compare SPH evolution with ghost leaves against the remote fetch path.

Milestone 2 of docs/sph_parent_tree_plan.md. Each case runs the same scenario on 2, 3
and 4 localities twice, with compute.sphGhosts on and off. The ghosts carry exactly the
values the fetches read, but each locality discovers its neighbor leaves in a different
order, so staging order and with it float summation order differ. Every rank must hold
the same particles; positions, velocities and the gas state must agree to roundoff.
"""

import argparse
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True

from application_selection import application_arguments, compiled_algorithms, cpu_backend_options
from checkpoint_format import decode_checkpoint, require
from run_checkpoint import prepareCase, runSimulation


def states(directory, iteration, ranks):
    checkpoint = directory / f"checkpoint.{iteration}"
    return [decode_checkpoint((checkpoint / f"checkpoint.{rank}.dat").read_bytes(), ranks=ranks)
            for rank in range(ranks)]


def relative_difference(first, second):
    scale = max((abs(value) for value in list(first) + list(second)), default=0.0)
    error = max((abs(a - b) for a, b in zip(first, second)), default=0.0)
    return error / scale if scale > 0.0 else 0.0


GHOSTS = re.compile(r"^SPH ghosts: syncs=(\d+) refreshes=(\d+) leaves=(\d+) particles=(\d+) "
                    r"exported leaves=(\d+) record bytes=(\d+) field bytes=(\d+)$", re.MULTILINE)


def particles(states):
    """Every gas particle of every rank: fixed-point position, velocity and gas state."""
    rows = []
    for state in states:
        period = 1 << (8 * state.position_bytes)
        for particle in range(state.count):
            if state.types[particle] != 1:
                continue
            gas = state.gas_indices[particle]
            rows.append({"position": tuple(state.positions[axis][particle] for axis in range(3)),
                         "period": period,
                         "velocity": tuple(state.velocities[3 * particle:3 * particle + 3]),
                         "entropy": state.entropies[gas],
                         "smoothing_length": state.smoothing_lengths[gas],
                         "density": state.densities[gas]})
    return rows


def pair(label, fetched, ghosted, tolerance):
    """Pair each particle with the unique one within tolerance units in every axis.

    Ranks may order (and with the global integrator even own) particles differently
    after roundoff moves one across a sort or domain plane, so pair by position."""
    # A power-of-two cell divides the fixed-point period, so cell keys wrap exactly.
    cell = 1 << max(0, (4 * tolerance - 1).bit_length())
    grid = {}
    for index, row in enumerate(ghosted):
        key = tuple(coordinate // cell for coordinate in row["position"])
        grid.setdefault(key, []).append(index)
    pairs = []
    used = set()
    for row in fetched:
        period = row["period"]
        key = tuple(coordinate // cell for coordinate in row["position"])
        best = None
        cells = period // cell
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbor = ((key[0] + dx) % cells, (key[1] + dy) % cells, (key[2] + dz) % cells)
                    for index in grid.get(neighbor, ()):
                        steps = []
                        for old, new in zip(row["position"], ghosted[index]["position"]):
                            step = (new - old) % period
                            steps.append(min(step, period - step))
                        distance = max(steps)
                        if distance <= tolerance and (best is None or distance < best[0]):
                            best = (distance, index)
        require(best is not None, f"{label}: a particle has no counterpart within {tolerance} units")
        require(best[1] not in used, f"{label}: two particles share one counterpart")
        used.add(best[1])
        pairs.append((row, ghosted[best[1]], best[0]))
    require(len(used) == len(ghosted), f"{label}: particle populations differ")
    return pairs


def compare(label, fetched, ghosted):
    require(len(fetched) == len(ghosted), f"{label}: rank count differs")
    first, second = particles(fetched), particles(ghosted)
    require(len(first) == len(second), f"{label}: gas populations differ")
    period = first[0]["period"]
    # Float summation order alone: velocities differ by about 1e-7 of their scale, and a few
    # steps integrate that into about 1e-8 of the box. A real error is of order a smoothing
    # length, 1e-2 of the box.
    tolerance = int(1.0e-6 * period)
    pairs = pair(label, first, second, tolerance)
    worst = {"position_units": max(distance for _, _, distance in pairs)}
    velocity_scale = max(abs(value) for row in first + second for value in row["velocity"])
    worst["velocity"] = max(abs(a - b) for x, y, _ in pairs
                            for a, b in zip(x["velocity"], y["velocity"])) / velocity_scale
    for field in ("entropy", "smoothing_length", "density"):
        scale = max(abs(row[field]) for row in first + second)
        worst[field] = max(abs(x[field] - y[field]) for x, y, _ in pairs) / scale
    for field in ("velocity", "entropy", "smoothing_length", "density"):
        require(worst[field] <= 1.0e-4,
                f"{label}: {field} differs by {worst[field]:.3e} of its scale")
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--mpi-exec", required=True)
    parser.add_argument("--mpi-numproc-flag", default="-n")
    parser.add_argument("--mpi-preflag", action="append", default=[])
    parser.add_argument("--mpi-postflag", action="append", default=[])
    parser.add_argument("--hpx-argument", action="append", default=[])
    parser.add_argument("--ranks", type=int, action="append")
    args = parser.parse_args()
    executable = args.executable.resolve()
    root = args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    available = compiled_algorithms(str(executable))
    integrators = [name for name in ("traditional", "global") if name in available["integrator"]]
    require(integrators, "traditional or global integration is required")
    common = {"hpx:threads": 1, "hpx:bind": "none", **cpu_backend_options(),
              "hydro.enabled": "true", "gravity.enabled": "false",
              "chemistry.enabled": "false", "starFormation.enabled": "false",
              "analysis.groups": "false", "analysis.tracers": "false",
              "analysis.views": "false", "analysis.power": "false", "analysis.slice": "false",
              "lightcone.enabled": "false", "output.silo": "false", "output.directory": ".",
              "checkpoint.intervalSeconds": 0, "run.verbose": "true",
              "compute.sphBucketSize": 16}
    scenarios = {
        "wave": {"run.scenario": "hydroWave", "run.endTime": 0.5, "run.numberOfSteps": 4,
                 "run.maxIterations": 6, "run.darkMatterParticleCount": 512,
                 "run.gasParticleCount": 512},
        "sod": {"run.scenario": "sod", "run.numberOfSteps": 8, "run.maxIterations": 6,
                "run.darkMatterParticleCount": 2048, "run.gasParticleCount": 2048},
    }

    def run(directory, options, integrator, ranks):
        command = [str(executable), *args.hpx_argument,
                   *application_arguments(executable, integrator=integrator, hydro=True)]
        command += [f"--{key}={value}" for key, value in dict(common, **options).items()]
        command = [args.mpi_exec, args.mpi_numproc_flag, str(ranks), *args.mpi_preflag,
                   command[0], *args.mpi_postflag, *command[1:],
                   f"--hpx:localities={ranks}"]
        if "--hpx:ini=hpx.parcel.mpi.enable=1" not in command:
            command.append("--hpx:ini=hpx.parcel.mpi.enable=1")
        return runSimulation(command, directory, "run.log")

    evidence = {}
    for integrator in integrators:
        for name, options in scenarios.items():
            for ranks in args.ranks or [2, 3, 4]:
                label = f"{integrator}_{name}_{ranks}"
                directories = {}
                for ghosts in ("false", "true"):
                    directory = prepareCase(root, f"{label}_ghosts_{ghosts}")
                    output = run(directory, dict(options, **{"compute.sphGhosts": ghosts}),
                                 integrator, ranks)
                    require("SPH midpoint:" in output, f"{label}: no midpoint hydro event")
                    require(f"compute.sphGhosts                     : "
                            f"{'T' if ghosts == 'true' else 'F'}" in output,
                            f"{label}: compute.sphGhosts was not applied")
                    counters = GHOSTS.search(output)
                    if ghosts == "true":
                        require(counters and int(counters[2]) > 0 and int(counters[7]) > 0,
                                f"{label}: the ghost run pushed no ghost fields")
                        evidence[label + "_ghosts"] = {
                            "syncs": int(counters[1]), "refreshes": int(counters[2]),
                            "leaves": int(counters[3]), "particles": int(counters[4]),
                            "field_bytes": int(counters[7])}
                    else:
                        require(counters is None, f"{label}: the fetch run used ghosts")
                    directories[ghosts] = directory
                iteration = options["run.maxIterations"]
                worst = compare(label, states(directories["false"], iteration, ranks),
                                states(directories["true"], iteration, ranks))
                evidence[label] = worst
                print(f"{label}: " + ", ".join(
                    f"{key}={value:.3e}" if isinstance(value, float) else f"{key}={value}"
                    for key, value in worst.items()), flush=True)
    (root / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print("SPH ghost evolution matches the fetch path", flush=True)


if __name__ == "__main__":
    main()
