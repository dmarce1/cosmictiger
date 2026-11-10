#!/usr/bin/env python3
#  CosmicTiger - A cosmological N-Body code
#  Copyright (C) 2026  Dominic C. Marcello
#
#  Distributed under the Boost Software License, Version 1.0.
#  See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.

"""Compare distributed SPH evolution on ghost leaves against one locality.

Ghost leaves are the only distributed SPH path (docs/sph_parent_tree_plan.md, M5), so the
reference is the same scenario on a single locality, where every neighbor is local. Each
case then runs on 2, 3 and 4 localities. The particle pairs within each support are the
same, but the leaves, their order and the staging order differ, so float summation order
differs. The same particles must exist; positions, velocities and the gas state must agree
to roundoff.
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
                    r"exported leaves=(\d+) record bytes=(\d+) field bytes=(\d+) "
                    r"needed field bytes=(\d+) all field bytes=(\d+)$", re.MULTILINE)


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


def pair(label, reference, distributed, tolerance):
    """Pair each particle with the unique one within tolerance units in every axis.

    Ranks order and own particles differently, so pair by position."""
    # A power-of-two cell divides the fixed-point period, so cell keys wrap exactly.
    cell = 1 << max(0, (4 * tolerance - 1).bit_length())
    grid = {}
    for index, row in enumerate(distributed):
        key = tuple(coordinate // cell for coordinate in row["position"])
        grid.setdefault(key, []).append(index)
    pairs = []
    used = set()
    for row in reference:
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
                        for old, new in zip(row["position"], distributed[index]["position"]):
                            step = (new - old) % period
                            steps.append(min(step, period - step))
                        distance = max(steps)
                        if distance <= tolerance and (best is None or distance < best[0]):
                            best = (distance, index)
        require(best is not None, f"{label}: a particle has no counterpart within {tolerance} units")
        require(best[1] not in used, f"{label}: two particles share one counterpart")
        used.add(best[1])
        pairs.append((row, distributed[best[1]], best[0]))
    require(len(used) == len(distributed), f"{label}: particle populations differ")
    return pairs


def compare(label, reference, distributed, tolerances):
    first, second = particles(reference), particles(distributed)
    require(len(first) == len(second), f"{label}: gas populations differ")
    period = first[0]["period"]
    # Float summation order alone: velocities and gas fields differ by about 1e-7 of their
    # scale, and a few steps integrate that into about 1e-8 of the box. A real error is far
    # larger: gas fields paired with the wrong particle after the domain sort moved Sod
    # particles by 2e-4 of the box in one step.
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
        require(worst[field] <= tolerances.get(field, 1.0e-5),
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
    integrators = [name for name in ("hierarchical",)
                   if name in available["integrator"]]
    require(integrators, "an SPH-capable integrator is required")
    common = {"hpx:threads": 1, "hpx:bind": "none", **cpu_backend_options(),
              "hydro.enabled": "true", "gravity.enabled": "false",
              "chemistry.enabled": "false", "starFormation.enabled": "false",
              "analysis.groups": "false", "analysis.tracers": "false",
              "analysis.views": "false", "analysis.power": "false", "analysis.slice": "false",
              "lightcone.enabled": "false", "output.silo": "false", "output.directory": ".",
              "checkpoint.intervalSeconds": 0, "run.verbose": "true",
              "compute.sphBucketSize": 16}
    scenarios = {
        "wave": {"run.scenario": "hydroWave", "run.endTime": 0.5, "run.numberOfOutputs": 5,
                 "run.maxIterations": 6, "run.darkMatterParticleCount": 512,
                 "run.gasParticleCount": 512},
        "sod": {"run.scenario": "sod", "run.numberOfOutputs": 9, "run.maxIterations": 6,
                "run.darkMatterParticleCount": 2048, "run.gasParticleCount": 2048},
    }
    # SPH couples to PMMM self-gravity only on the hierarchical integrator, which needs gravity.
    preset = Path(__file__).resolve().parents[1] / "configurations" / "jeans_wave.ini"
    gravity_scenarios = {
        "jeans": {"configuration": preset, "gravity.enabled": "true",
                  "run.gasParticleCount": 4096, "gravity.meshSize": 16,
                  "run.maxIterations": 6},
    }
    # PMMM needs a power-of-two locality count, and its tree approximation depends on the
    # decomposition: velocities differ by about 5e-5 of their scale between any two counts.
    gravity_ranks = [ranks for ranks in args.ranks or [2, 4] if ranks & (ranks - 1) == 0]
    gravity_tolerances = {"velocity": 5.0e-4}

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
        cases = dict(scenarios, **gravity_scenarios) if "pmmm" in available["far"] else scenarios
        for name, options in cases.items():
            iteration = options["run.maxIterations"]
            label = f"{integrator}_{name}"
            directory = prepareCase(root, f"{label}_1")
            output = run(directory, options, integrator, 1)
            require("SPH midpoint:" in output, f"{label}_1: no midpoint hydro event")
            require(GHOSTS.search(output) is None, f"{label}_1: one locality used ghosts")
            reference = states(directory, iteration, 1)
            gravity = options.get("gravity.enabled") == "true"
            for ranks in gravity_ranks if gravity else args.ranks or [2, 3, 4]:
                label = f"{integrator}_{name}_{ranks}"
                directory = prepareCase(root, label)
                output = run(directory, options, integrator, ranks)
                require("SPH midpoint:" in output, f"{label}: no midpoint hydro event")
                counters = GHOSTS.search(output)
                require(counters and int(counters[2]) > 0 and int(counters[7]) > 0,
                        f"{label}: the run pushed no ghost fields")
                evidence[label + "_ghosts"] = {
                    "syncs": int(counters[1]), "refreshes": int(counters[2]),
                    "leaves": int(counters[3]), "particles": int(counters[4]),
                    "field_bytes": int(counters[7]), "needed_field_bytes": int(counters[8]),
                    "all_field_bytes": int(counters[9])}
                worst = compare(label, reference, states(directory, iteration, ranks),
                                gravity_tolerances if gravity else {})
                evidence[label] = worst
                print(f"{label}: " + ", ".join(
                    f"{key}={value:.3e}" if isinstance(value, float) else f"{key}={value}"
                    for key, value in worst.items()), flush=True)
    (root / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print("Distributed SPH evolution on ghost leaves matches one locality", flush=True)


if __name__ == "__main__":
    main()
