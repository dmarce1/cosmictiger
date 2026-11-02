# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# SPDX-License-Identifier: BSL-1.0

"""Check pm/p3m evolution: outer PM intervals, inner rungs, mesh-solve count, restart and identity."""

import argparse
import math
from pathlib import Path
import re
import struct
import subprocess
import sys

sys.dont_write_bytecode = True

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from checkpoint_format import decode_checkpoint
from run_checkpoint import fnv1a64, prepareCase, runSimulation


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_checkpoint(path, ranks):
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
        require(decoded.header.flags & (1 << 22), f"{path}: particle-mesh flag is missing")
        require(decoded.particle_mesh, f"{path}: particle-mesh parameter block is missing")
        require(decoded.driver.iteration == iteration, f"{path}: iteration mismatch")
        if state is None:
            header, state = decoded.header, decoded.driver
        else:
            require((state.tau, state.integer_time, state.outer_interval, state.outer_start) ==
                    (decoded.driver.tau, decoded.driver.integer_time,
                     decoded.driver.outer_interval, decoded.driver.outer_start),
                    f"{path}: locality clocks differ")
        scale = 2 ** (8 * decoded.position_bytes)
        for index, kind in enumerate(decoded.types):
            gas = decoded.gas[decoded.gas_indices[index]] if kind == 1 else ()
            rows.append({"x": tuple(decoded.positions[dim][index] / scale for dim in range(3)),
                         "v": decoded.velocities[3 * index:3 * index + 3],
                         "rung": decoded.rungs[index], "type": kind, "gas": gas,
                         "key": tuple(decoded.positions[dim][index] for dim in range(3))})
    return header, state, rows, decoded.particle_mesh


def phase(state):
    return math.ldexp(state.integer_time, -64)


def check_interval_clock(states, coarse):
    """The conformal clock equals the interval start plus the interval-local phase."""
    for header, state, _, _ in states:
        require(state.outer_interval > 0, "an evolving checkpoint has no outer PM interval")
        require(state.outer_interval <= coarse * (1 + 1e-12),
                "an outer PM interval exceeded the coarse interval")
        if state.integer_time != 0:
            expected = state.outer_start + state.outer_interval * phase(state)
        else:
            expected = state.outer_start + state.outer_interval
        require(math.isclose(state.tau, expected, rel_tol=1e-9, abs_tol=1e-15),
                f"interval clock mismatch: tau={state.tau} start={state.outer_start} "
                f"interval={state.outer_interval} phase={phase(state)}")
        # The coarse clock keeps outputs on coarse boundaries: no interval crosses one.
        start_position = state.outer_start / coarse
        end_position = (state.outer_start + state.outer_interval) / coarse
        require(math.floor(start_position + 1e-9) == math.ceil(end_position - 1e-9) - 1 or
                math.isclose(end_position, round(end_position), rel_tol=0, abs_tol=1e-9),
                "an outer PM interval crossed a coarse output boundary")


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
    _, reference_state, reference_rows, _ = reference
    _, resumed_state, resumed_rows, _ = resumed
    require((reference_state.tau, reference_state.integer_time, reference_state.max_rung) ==
            (resumed_state.tau, resumed_state.integer_time, resumed_state.max_rung),
            "restart changed the common clock")
    require((reference_state.outer_interval, reference_state.outer_start,
             reference_state.outer_index) ==
            (resumed_state.outer_interval, resumed_state.outer_start, resumed_state.outer_index),
            "restart changed the outer PM interval state")
    for old, new in match_rows(reference_rows, resumed_rows):
        require(old["rung"] == new["rung"] and len(old["gas"]) == len(new["gas"]),
                "restart changed particle/rung layout")
        for a, b in zip((*old["v"], *old["gas"]), (*new["v"], *new["gas"])):
            require(math.isclose(a, b, rel_tol=2.0e-5, abs_tol=1.0e-8),
                    f"restart changed velocity or SPH state: {a} != {b}")


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
    parser.add_argument("--far-method", choices=("pm", "p3m"), required=True)
    parser.add_argument("--near-method", choices=("none", "pp", "fmm"), required=True)
    parser.add_argument("--order", type=int, choices=range(3, 10), required=True)
    parser.add_argument("--time-integrator", choices=("traditional", "hierarchical"), required=True)
    parser.add_argument("--with-sph", action="store_true")
    args = parser.parse_args()
    executable = args.executable.resolve()
    root = args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    ranks = args.ranks if args.mpi_exec else 1
    split = args.far_method == "p3m"
    mesh_size = 32 if split else 16
    common = {"hpx:threads": 2, "hpx:bind": "none",
              "gravity.method.far": args.far_method, "gravity.meshSize": mesh_size,
              "gravity.cloud": "pcs", "gravity.expansionDegree": args.order - 1,
              "run.timeIntegrator": args.time_integrator, "gravity.enabled": "true",
              "gravity.cuda": "false", "drift.cuda": "false", "hydro.cuda": "false",
              "compute.positionType": args.position_type,
              "initial.particlesPerDimension": 8, "initial.closePack": "false",
              "initial.useGlass": "false", "chemistry.enabled": "false",
              # A small eta makes the PM acceleration bound select the interval of
              # the Plummer case; the split long force is smoother, so p3m keeps
              # a longer coarse interval instead of a smaller inner prefactor.
              "run.endTime": 0.005, "run.numberOfSteps": 2 if split else 4,
              "gravity.timestepEta": 0.05,
              "analysis.groups": "false", "analysis.tracers": "false",
              "analysis.views": "false", "analysis.power": "false", "lightcone.enabled": "false",
              "output.silo": "false", "output.directory": ".", "checkpoint.intervalSeconds": 0,
              "run.verbose": "true"}
    if split:
        common.update({"gravity.method.near": args.near_method, "gravity.pmmmNearWidth": 2,
                       "gravity.p3mSplitCells": 1.25, "gravity.p3mCutoff": 4.5})
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

    def help_text():
        return subprocess.run([str(executable), "--help", "--hpx:threads=1", "--hpx:bind=none"],
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              timeout=60).stdout

    label = f"{args.far_method}_{args.near_method}_{args.time_integrator}"
    coarse = common["run.endTime"] / common["run.numberOfSteps"]
    cases = [("dark_matter", False)]
    if args.with_sph and args.time_integrator == "traditional":
        cases.append(("mixed", True))
    for case, hydro in cases:
        name = f"{label}_{case}"
        options = dict(common, **{"hydro.enabled": str(hydro).lower(),
                                  "run.scenario": "cosmology" if hydro else "plummer"})
        # Short-range rungs subdivide a P3M interval into tens of substeps.
        iterations = 80 if split else 6
        baseline = prepareCase(root, name + "_baseline")
        log = run(baseline, dict(options, **{"run.maxIterations": iterations}))
        for key, expected in (("run.timeIntegrator", args.time_integrator),
                              ("gravity.method.far", args.far_method),
                              ("gravity.cloud", "pcs")):
            require(re.search(r"^" + re.escape(key) + r"\s*:\s*" + expected + r"\s*$",
                              log, re.MULTILINE), f"{name}: resolved options lack {key}={expected}")
        summary = re.search(r"Particle-mesh intervals completed: (\d+); mesh solves this run: (\d+); "
                            r"force passes this run: (\d+)", log)
        require(summary, f"{name}: missing particle-mesh summary line")
        intervals, solves, passes = map(int, summary.groups())
        boundaries = len(re.findall(r"^PM boundary at tau=", log, re.MULTILINE))
        # A run may reach the end time before the iteration limit.
        require(2 <= passes <= iterations, f"{name}: expected at most {iterations} force passes, saw {passes}")
        iterations = passes
        # One mesh solve per outer boundary reached (intervals started), none inside.
        require(boundaries == solves, f"{name}: {boundaries} boundaries but {solves} mesh solves")
        require(solves <= passes, f"{name}: more mesh solves than force passes")
        paths = sorted((path for path in baseline.glob("checkpoint.*") if path.is_dir() and
                        re.fullmatch(r"checkpoint\.[0-9]+", path.name)),
                       key=lambda path: int(path.name.split(".")[1]))
        require([int(path.name.split(".")[1]) for path in paths] == list(range(1, iterations + 1)),
                f"{name}: expected one checkpoint per force pass")
        states = [read_checkpoint(path, ranks) for path in paths]
        check_interval_clock(states, coarse)
        starts = sorted({state.outer_start for _, state, _, _ in states})
        require(len(starts) == solves or len(starts) + 1 == solves,
                f"{name}: {len(starts)} interval starts but {solves} mesh solves")
        intervals_seen = {state.outer_interval for _, state, _, _ in states}
        # The cosmological mixed case at z=49 is limited by the coarse interval.
        if not hydro:
            require(any(not is_dyadic_fraction(interval, coarse) for interval in intervals_seen),
                    f"{name}: every outer interval was a dyadic fraction of the coarse interval")
        rung_sets = [{row["rung"] for row in rows} for _, _, rows, _ in states]
        if not hydro and split:
            require(any(len(rungs) > 1 or min(rungs) > 0 for rungs in rung_sets),
                    f"{name}: the short force never selected an inner rung")
        inner = [state for _, state, _, _ in states if state.integer_time != 0]
        # Pure PM without hydro has no inner force: the hierarchical ladder keeps
        # every particle on rung zero and each interval is one force pass.
        if split or hydro or args.time_integrator == "traditional":
            require(inner, f"{name}: no checkpoint fell inside an outer interval")
        else:
            require(not inner and solves == passes,
                    f"{name}: pure PM without hydro took inner substeps")
        require(any(state.integer_time == 0 and state.iteration > 0 for _, state, _, _ in states),
                f"{name}: no checkpoint fell on an outer boundary")
        if hydro:
            gas = [row for row in states[-1][2] if row["type"] == 1]
            require(gas and all(all(math.isfinite(value) for value in row["gas"]) for row in gas),
                    f"{name}: nonfinite gas state")
        # Restart from a mid-interval checkpoint and from the boundary one.
        boundary_iteration = next(index for index, (_, state, _, _) in enumerate(states, start=1)
                                  if state.integer_time == 0 and state.iteration > 0)
        for source_iteration in sorted({2, boundary_iteration} - {iterations}):
            source = baseline / f"checkpoint.{source_iteration}"
            resumed = prepareCase(root, f"{name}_resumed_{source_iteration}", source)
            log = run(resumed, dict(options, **{"checkpoint.readIndex": source_iteration,
                                              "run.maxIterations": iterations - source_iteration}))
            require("Done reading checkpoint" in log, f"{name}: restart did not load its checkpoint")
            compare_restart(states[-1], read_checkpoint(resumed / f"checkpoint.{iterations}", ranks))
        if hydro:
            print(f"{name}: mixed gas/dark-matter evolution across outer PM intervals with restart passed")
            continue
        source = baseline / "checkpoint.2"
        changes = [("mesh", {"gravity.meshSize": 2 * mesh_size}), ("cloud", {"gravity.cloud": "tsc"}),
                   ("eta", {"gravity.timestepEta": 0.1})]
        if split:
            changes += [("split", {"gravity.p3mSplitCells": 1.5}), ("cutoff", {"gravity.p3mCutoff": 4.0}),
                        ("near", {"gravity.method.near": "fmm" if args.near_method == "pp" else "pp"})]
        for change_name, change in changes:
            if change_name == "near":
                compiled = re.search(r"P3M short solvers:([^\n]*)", help_text())
                if compiled is None or change["gravity.method.near"] not in compiled[1].split():
                    continue
            rejected = prepareCase(root, f"{name}_reject_{change_name}", source)
            log = run(rejected, dict(options, **change, **{"checkpoint.readIndex": 2,
                                                          "run.maxIterations": 0}), False)
            require("Particle-mesh checkpoint" in log or "Checkpoint gravity methods" in log,
                    f"{name}: {change_name} mismatch lacked the particle-mesh identity diagnostic")
        other_far = "pm" if split else "p3m"
        compiled = re.search(r"Gravity solvers:([^\n]*)", help_text())
        if compiled and other_far in compiled[1].split():
            rejected = prepareCase(root, f"{name}_reject_far", source)
            far_options = {key: value for key, value in options.items()
                           if key not in ("gravity.method.near", "gravity.pmmmNearWidth",
                                          "gravity.p3mSplitCells", "gravity.p3mCutoff")}
            change = {"gravity.method.far": other_far, "gravity.meshSize": 32}
            if other_far == "p3m":
                change["gravity.method.near"] = "pp"
                change.update({"gravity.pmmmNearWidth": 2, "gravity.p3mSplitCells": 1.25,
                               "gravity.p3mCutoff": 4.5})
            log = run(rejected, dict(far_options, **change, **{"checkpoint.readIndex": 2,
                                                              "run.maxIterations": 0}), False)
            require("Checkpoint physics" in log or "Checkpoint gravity methods" in log or
                    "Particle-mesh checkpoint" in log,
                    f"{name}: far solver mismatch lacked a compatibility diagnostic")
        print(f"{name}: {solves} mesh solves over {passes} force passes, non-dyadic changing "
              f"outer intervals, interval clock, inner rungs, restart and identity checks passed")
    print(f"Particle-mesh evolution passed ({ranks} rank(s), {args.position_type}, {label})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, struct.error, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
