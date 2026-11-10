# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# SPDX-License-Identifier: BSL-1.0

"""Check the PMMM outer interval: frozen near membership, one transform per boundary,
the interval clock, restart from inside and from a boundary, identity rejection, and
energy and momentum behaviour against the shorter-interval PMMM reference."""

import argparse
import math
from pathlib import Path
import re
import struct
import subprocess
import sys

sys.dont_write_bytecode = True

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checkpoint_format import decode_checkpoint
from run_checkpoint import fnv1a64, prepareCase, runSimulation


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_checkpoint(path, ranks, mesh_size):
    iteration = int(path.name.split(".")[1])
    lines = (path / "checkpoint.meta").read_text().splitlines()
    rows = []
    state = None
    header = None
    for rank in range(ranks):
        data = (path / f"checkpoint.{rank}.dat").read_bytes()
        record = lines[rank + 1].split()
        require(record == [str(rank), str(len(data)), f"{fnv1a64(data):016x}"],
                f"{path}: rank {rank} size/checksum mismatch")
        decoded = decode_checkpoint(data, ranks=ranks)
        require(decoded.header.flags & (1 << 17), f"{path}: PMMM flag is missing")
        require(decoded.header.flags & (1 << 23), f"{path}: PMMM outer-interval flag is missing")
        require(not decoded.header.flags & (1 << 22), f"{path}: scalar PM flag set for PMMM")
        require(decoded.driver.iteration == iteration, f"{path}: iteration mismatch")
        require(len(decoded.frozen_cells) == decoded.count, f"{path}: frozen key payload size")
        require(all(key < mesh_size ** 3 for key in decoded.frozen_cells),
                f"{path}: a frozen cell key lies outside the mesh")
        if state is None:
            header, state = decoded.header, decoded.driver
        else:
            require((state.tau, state.integer_time, state.outer_interval, state.outer_start) ==
                    (decoded.driver.tau, decoded.driver.integer_time,
                     decoded.driver.outer_interval, decoded.driver.outer_start),
                    f"{path}: locality clocks differ")
        scale = 2 ** (8 * decoded.position_bytes)
        for index in range(decoded.count):
            rows.append({"x": tuple(decoded.positions[dim][index] / scale for dim in range(3)),
                         "v": decoded.velocities[3 * index:3 * index + 3],
                         "rung": decoded.rungs[index],
                         "cell": decoded.frozen_cells[index],
                         "key": tuple(decoded.positions[dim][index] for dim in range(3))})
    return header, state, rows


def phase(state):
    return math.ldexp(state.integer_time, -64)


def check_interval_clock(states, coarse):
    for header, state, _ in states:
        require(state.outer_interval > 0, "an evolving checkpoint has no outer interval")
        require(state.outer_interval <= coarse * (1 + 1e-12),
                "an outer interval exceeded the coarse interval")
        if state.integer_time != 0:
            expected = state.outer_start + state.outer_interval * phase(state)
        else:
            expected = state.outer_start + state.outer_interval
        require(math.isclose(state.tau, expected, rel_tol=1e-9, abs_tol=1e-15),
                f"interval clock mismatch: tau={state.tau} start={state.outer_start} "
                f"interval={state.outer_interval} phase={phase(state)}")
        start_position = state.outer_start / coarse
        end_position = (state.outer_start + state.outer_interval) / coarse
        require(math.floor(start_position + 1e-9) == math.ceil(end_position - 1e-9) - 1 or
                math.isclose(end_position, round(end_position), rel_tol=0, abs_tol=1e-9),
                "an outer interval crossed a coarse output boundary")


def is_dyadic_fraction(interval, coarse):
    ratio = interval / coarse
    if ratio <= 0:
        return False
    power = math.log2(ratio)
    return math.isclose(power, round(power), rel_tol=0, abs_tol=1e-9)


def match_rows(before, after):
    require(len(before) == len(after), "restart changed the particle count")
    by_key = {}
    for row in after:
        by_key.setdefault(row["key"], []).append(row)
    pairs = []
    for row in before:
        candidates = by_key.get(row["key"])
        require(candidates, "restart changed a particle position")
        pairs.append((row, candidates.pop()))
    return pairs


def compare_restart(reference, resumed):
    _, reference_state, reference_rows = reference
    _, resumed_state, resumed_rows = resumed
    require((reference_state.tau, reference_state.integer_time, reference_state.max_rung) ==
            (resumed_state.tau, resumed_state.integer_time, resumed_state.max_rung),
            "restart changed the common clock")
    require((reference_state.outer_interval, reference_state.outer_start,
             reference_state.outer_index) ==
            (resumed_state.outer_interval, resumed_state.outer_start, resumed_state.outer_index),
            "restart changed the outer interval state")
    for old, new in match_rows(reference_rows, resumed_rows):
        require(old["rung"] == new["rung"], "restart changed a particle rung")
        require(old["cell"] == new["cell"], "restart changed a frozen cell key")
        for a, b in zip(old["v"], new["v"]):
            require(math.isclose(a, b, rel_tol=2.0e-5, abs_tol=1.0e-8),
                    f"restart changed a velocity: {a} != {b}")


def frozen_keys_match_positions(rows, mesh_size):
    """Every frozen key equals the particle's current cell (true right after a refresh)."""
    for row in rows:
        cell = tuple(min(mesh_size - 1, int(coordinate * mesh_size)) for coordinate in row["x"])
        key = (cell[0] * mesh_size + cell[1]) * mesh_size + cell[2]
        if key != row["cell"]:
            return False
    return True


def read_energy(directory):
    rows = []
    for line in (directory / "energy.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        rows.append([float(value) for value in line.split()])
    require(rows, f"{directory}: energy.txt is empty")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--mpi-exec")
    parser.add_argument("--mpi-numproc-flag", default="-n")
    parser.add_argument("--mpi-preflag", action="append", default=[])
    parser.add_argument("--mpi-postflag", action="append", default=[])
    parser.add_argument("--disable-tcp", action="store_true")
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--position-type", choices=("fixed32", "fixed64"), default="fixed32")
    parser.add_argument("--order", type=int, choices=range(3, 10), required=True)
    parser.add_argument("--time-integrator", choices=("hierarchical",),
                        required=True)
    parser.add_argument("--near-method", choices=("pp", "fmm"), default="fmm")
    args = parser.parse_args()
    executable = args.executable.resolve()
    root = args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    ranks = args.ranks if args.mpi_exec else 1
    mesh_size = 16
    common = {"hpx:threads": 2, "hpx:bind": "none",
              "gravity.method.near": args.near_method,
              "gravity.meshSize": mesh_size, "gravity.pmmmNearWidth": 2,
              "gravity.pmmmOuterInterval": "true",
              "gravity.expansionDegree": args.order - 1,
              "run.timeIntegrator": args.time_integrator, "gravity.enabled": "true",
              "gravity.cuda": "false", "drift.cuda": "false", "hydro.cuda": "false",
              "hydro.enabled": "false", "run.scenario": "plummer",
              "compute.positionType": args.position_type,
              "run.darkMatterParticleCount": 512, "run.gasParticleCount": 512,
              "initial.closePack": "false", "scenario.cosmology.useGlass": "false",
              "chemistry.enabled": "false",
              # One coarse step of 0.005: with eta 0.04 the Plummer far acceleration
              # bound (about 4.1e-3) selects a non-dyadic first interval and the
              # remainder forms the second, while the near force picks the rungs.
              "run.endTime": 0.005, "run.numberOfSteps": 1, "gravity.timestepEta": 0.04,
              "analysis.groups": "false", "analysis.tracers": "false",
              "analysis.views": "false", "analysis.power": "false", "lightcone.enabled": "false",
              "output.silo": "false", "output.directory": ".", "checkpoint.intervalSeconds": 0,
              "run.verbose": "true"}
    if ranks > 1:
        common["hpx:localities"] = ranks

    def run(directory, options, success=True):
        command = [str(executable)] + [f"--{key}={value}" for key, value in options.items()]
        if args.mpi_exec:
            command = ([args.mpi_exec, args.mpi_numproc_flag, str(ranks)] + args.mpi_preflag
                       + command[:1] + args.mpi_postflag + command[1:])
        if ranks > 1:
            command.append("--hpx:ini=hpx.parcel.mpi.enable=1")
            if args.disable_tcp:
                command.append("--hpx:ini=hpx.parcel.tcp.enable=0")
        return runSimulation(command, directory, "run.log", expectSuccess=success)

    label = f"pmmm_interval_{args.time_integrator}_{args.near_method}"
    coarse = common["run.endTime"] / common["run.numberOfSteps"]
    iterations = 400
    baseline = prepareCase(root, label + "_baseline")
    log = run(baseline, dict(common, **{"run.maxIterations": iterations}))
    for key, expected in (("run.timeIntegrator", args.time_integrator),
                          ("gravity.method.near", args.near_method),
                          ("gravity.pmmmOuterInterval", "T")):
        require(re.search(r"^" + re.escape(key) + r"\s*:\s*" + expected + r"\s*$",
                          log, re.MULTILINE), f"{label}: resolved options lack {key}={expected}")
    summary = re.search(r"PMMM outer intervals completed: (\d+); mesh transforms this run: (\d+); "
                        r"force passes this run: (\d+)", log)
    require(summary, f"{label}: missing PMMM outer-interval summary line")
    intervals, transforms, passes = map(int, summary.groups())
    boundaries = len(re.findall(r"^PMMM boundary at tau=", log, re.MULTILINE))
    require(2 <= passes < iterations, f"{label}: expected fewer than {iterations} passes, saw {passes}")
    iterations = passes
    finished = "reached the requested end time" in log
    require(finished, f"{label}: the run did not reach the end time within the iteration budget")
    # One transform per boundary reached and one for the terminal energy pass,
    # none inside an interval: every completed interval opened at a boundary.
    require(boundaries == transforms, f"{label}: {boundaries} boundaries but {transforms} transforms")
    require(transforms == intervals + 1,
            f"{label}: {transforms} transforms for {intervals} completed intervals")
    require(intervals >= 2, f"{label}: expected at least two outer intervals, saw {intervals}")
    require(transforms < passes, f"{label}: no force pass ran inside an interval")
    paths = sorted((path for path in baseline.glob("checkpoint.*") if path.is_dir() and
                    re.fullmatch(r"checkpoint\.[0-9]+", path.name)),
                   key=lambda path: int(path.name.split(".")[1]))
    require([int(path.name.split(".")[1]) for path in paths] == list(range(1, iterations + 1)),
            f"{label}: expected one checkpoint per force pass")
    states = [read_checkpoint(path, ranks, mesh_size) for path in paths]
    check_interval_clock(states, coarse)
    starts = sorted({state.outer_start for _, state, _ in states})
    require(len(starts) == intervals or len(starts) + 1 == intervals,
            f"{label}: {len(starts)} interval starts but {intervals} intervals")
    intervals_seen = {state.outer_interval for _, state, _ in states}
    require(any(not is_dyadic_fraction(interval, coarse) for interval in intervals_seen),
            f"{label}: every outer interval was a dyadic fraction of the coarse interval")
    rung_sets = [{row["rung"] for row in rows} for _, _, rows in states]
    require(any(len(rungs) > 1 or min(rungs) > 0 for rungs in rung_sets),
            f"{label}: the near force never selected an inner rung")
    inner = [state for _, state, _ in states if state.integer_time != 0]
    require(inner, f"{label}: no checkpoint fell inside an outer interval")
    boundary_states = [(index, state) for index, (_, state, _) in enumerate(states, start=1)
                       if state.integer_time == 0 and state.iteration > 0]
    require(boundary_states, f"{label}: no checkpoint fell on an outer boundary")
    # A boundary checkpoint, taken before its closing pass, holds the whole old
    # interval's membership; particles may sit outside their frozen cells there.
    stale = 0
    for _, state, rows in states:
        if state.integer_time == 0 and state.iteration > 0:
            stale += not frozen_keys_match_positions(rows, mesh_size)
    print(f"{label}: {len(boundary_states)} boundary checkpoints, {stale} with particles outside "
          "their frozen cells (membership kept until the closing pass)")
    # Restart from inside an interval and from a boundary.
    boundary_iteration = boundary_states[0][0]
    for source_iteration in sorted({2, boundary_iteration} - {iterations}):
        source = baseline / f"checkpoint.{source_iteration}"
        resumed = prepareCase(root, f"{label}_resumed_{source_iteration}", source)
        log = run(resumed, dict(common, **{"checkpoint.readIndex": source_iteration,
                                         "run.maxIterations": iterations - source_iteration}))
        require("Done reading checkpoint" in log, f"{label}: restart did not load its checkpoint")
        compare_restart(states[-1], read_checkpoint(resumed / f"checkpoint.{iterations}", ranks,
                                                    mesh_size))
    # The frozen-membership checkpoint is not a per-substep PMMM checkpoint.
    source = baseline / "checkpoint.2"
    rejected = prepareCase(root, label + "_reject_mode", source)
    log = run(rejected, dict(common, **{"gravity.pmmmOuterInterval": "false",
                                      "checkpoint.readIndex": 2, "run.maxIterations": 0}), False)
    require("long/short gravity splitting is mandatory" in log,
            f"{label}: switching the outer interval off was not rejected on restart")
    # Compare to the same split scheme with a tighter coarse interval. Every
    # comparison run retains mandatory splitting and its own frozen membership.
    reference = prepareCase(root, label + "_reference")
    reference_log = run(reference, dict(common, **{"run.numberOfSteps": 4,
                                                 "run.maxIterations": 1000}))
    frozen_energy = read_energy(baseline)
    reference_energy = read_energy(reference)
    require("reached the requested end time" in reference_log,
            f"{label}: the shorter-interval reference did not reach the end time")
    frozen_error = abs(frozen_energy[-1][8])
    reference_error = abs(reference_energy[-1][8])
    frozen_momentum = max(abs(value) for value in frozen_energy[-1][2:5])
    reference_momentum = max(abs(value) for value in reference_energy[-1][2:5])
    print(f"{label}: energy error {frozen_error:.3e} (shorter-interval reference {reference_error:.3e}), "
          f"momentum/sum|p| {frozen_momentum:.3e} (reference {reference_momentum:.3e}), "
          f"{transforms} transforms over {passes} passes against "
          f"{len(reference_energy)} reference rows")
    require(frozen_error <= max(4.0 * reference_error, 0.02),
            f"{label}: frozen-membership energy error {frozen_error} exceeds the reference bound")
    require(frozen_momentum <= max(100.0 * reference_momentum, 1.0e-4),
            f"{label}: frozen-membership momentum drift {frozen_momentum} exceeds the bound")
    print(f"{label}: {transforms} transforms over {passes} passes, interval clock, inner rungs, "
          "frozen keys, restart inside and at a boundary, and mode identity checks passed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, struct.error, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
