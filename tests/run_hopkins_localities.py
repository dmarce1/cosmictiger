#!/usr/bin/env python3
# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# SPDX-License-Identifier: BSL-1.0

"""Run gravity-free Hopkins MFM fixtures on one and on two localities and require
per-particle bitwise agreement.

Stable identifiers are assigned at creation, before the first domain rebound, so
both runs number the gas identically. Every per-cell result is a function of the
cell's record and its identifier-ordered stencil, and pair deposits are applied
in pair-key order, so the decomposition must not change a single bit of the
persistent particle state. Pair-cache records are compared by value: a
first-order lane inside a mixed SIMD batch may turn a -0.0 into +0.0, which does
not change any state it is later applied to. Gravity is disabled because the
gravity solvers' floating-point reductions depend on the decomposition.
"""

import argparse
import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

sys.dont_write_bytecode = True

from run_hopkins import particle_state


STEP_FIELDS = ("active", "total", "minRung", "maxRung", "pairs", "cached", "refreshed",
               "woken", "settled", "stars", "switched", "terminal")


def checkpoints(directory):
    return sorted((path for path in (directory / "output").glob("checkpoint.*") if path.is_dir()),
                  key=lambda path: int(path.suffix[1:]))


def step_rows(output):
    rows = []
    for match in re.finditer(r"Hopkins step: ([^\n]+)", output):
        fields = dict(re.findall(r"([A-Za-z]+)=([^\s]+)", match.group(1)))
        rows.append((fields["time"],) + tuple(int(fields[key]) for key in STEP_FIELDS))
    return rows


def exact(values):
    """Floats as their IEEE bit patterns, so -0.0 and +0.0 differ."""
    return tuple(struct.pack("=d", value) if isinstance(value, float) else value
                 for value in values)


def run(command, values, directory, timeout):
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    invocation = command + [f"--{key}={value}" for key, value in values.items()]
    (directory / "command.json").write_text(json.dumps(invocation, indent=2) + "\n")
    result = subprocess.run(invocation, cwd=directory, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=timeout)
    (directory / "run.log").write_text(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"{directory.name} failed with exit code {result.returncode}; "
                           f"see {directory / 'run.log'}")
    return result.stdout


def compare(label, single, double):
    paths = [checkpoints(single[1]), checkpoints(double[1])]
    if not paths[0] or len(paths[0]) != len(paths[1]):
        raise RuntimeError(f"{label}: checkpoint counts differ: {len(paths[0])} vs {len(paths[1])}")
    rows = [step_rows(single[0]), step_rows(double[0])]
    if not rows[0] or rows[0] != rows[1]:
        for index, (left, right) in enumerate(zip(*rows)):
            if left != right:
                raise RuntimeError(f"{label}: Hopkins step row {index} differs:\n  1: {left}\n  2: {right}")
        raise RuntimeError(f"{label}: Hopkins step row counts differ: {len(rows[0])} vs {len(rows[1])}")
    crossing = 0
    for one, two in zip(*paths):
        states = [particle_state(one, "mfm"), particle_state(two, "mfm")]
        if states[0]["clock"] != states[1]["clock"]:
            raise RuntimeError(f"{label}: {one.name} clocks differ")
        records = [state["physical_records"] for state in states]
        if records[0].keys() != records[1].keys():
            raise RuntimeError(f"{label}: {one.name} identifier sets differ")
        for identifier in records[0]:
            if records[0][identifier] != records[1][identifier]:
                raise RuntimeError(f"{label}: {one.name} physical record of {identifier} differs")
        integration = [{state[0]: exact(state) for state in each["integration_state"]}
                       for each in states]
        for identifier, values in integration[0].items():
            if integration[1].get(identifier) != values:
                raise RuntimeError(f"{label}: {one.name} integration state of {identifier} differs")
        pairs = [{(record[0], record[1]): record for record in each["pair_cache"]} for each in states]
        if pairs[0].keys() != pairs[1].keys():
            raise RuntimeError(f"{label}: {one.name} live pair sets differ")
        for key, record in pairs[0].items():
            if record != pairs[1][key]:
                raise RuntimeError(f"{label}: {one.name} pair {key} differs by value")
        ranks = states[1]["identifier_ranks"]
        crossing += sum(ranks[key[0]] != ranks[key[1]] for key in pairs[1])
    if crossing == 0:
        raise RuntimeError(f"{label}: no live pair joined cells of different localities")
    return {"checkpoints": len(paths[0]), "steps": len(rows[0]), "crossingPairs": crossing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--mpi-exec", type=Path, required=True)
    parser.add_argument("--mpi-numproc-flag", default="-n")
    parser.add_argument("--mpi-preflag", action="append", default=[])
    parser.add_argument("--mpi-postflag", action="append", default=[])
    parser.add_argument("--hpx-argument", action="append", default=[])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    executable = args.executable.resolve()
    base = [str(executable), f"--hpx:threads={args.threads}", "--hpx:bind=none"]
    commands = {1: base, 2: [str(args.mpi_exec.resolve()), args.mpi_numproc_flag, "2"] +
                args.mpi_preflag + base + args.mpi_postflag + args.hpx_argument +
                ["--hpx:localities=2"]}
    common = {"run.numberOfSteps": 2, "run.maxIterations": 4096, "run.verbose": "true",
              "initial.particlesPerDimension": 8, "hydro.enabled": "true",
              "hydro.method": "mfm", "hydro.cuda": "false", "gravity.enabled": "false",
              "gravity.cuda": "false", "drift.cuda": "false", "chemistry.enabled": "false",
              "starFormation.enabled": "false", "analysis.groups": "false",
              "analysis.tracers": "false", "analysis.power": "false",
              "analysis.views": "false", "lightcone.enabled": "false",
              "output.silo": "false", "output.directory": "output",
              "checkpoint.intervalSeconds": 0, "run.scenario": "blast", "run.endTime": 0.004}
    help_text = subprocess.run(base + ["--help"], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True).stdout
    integrators = [name for name in ("traditional", "global", "hierarchical")
                   if re.search(rf"\b{name}\b", help_text)]
    report = {}
    for integrator in integrators:
        values = dict(common, **{"run.timeIntegrator": integrator})
        outcome = {}
        for localities in (1, 2):
            directory = args.workdir / f"blast_{integrator}_{localities}"
            outcome[localities] = (run(commands[localities], values, directory, args.timeout),
                                   directory)
        report[integrator] = compare(f"blast {integrator}", outcome[1], outcome[2])
        print(f"blast {integrator}: one and two localities agree bitwise "
              f"({report[integrator]})", flush=True)
    (args.workdir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
