# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# SPDX-License-Identifier: BSL-1.0

"""Check fine-grained coordinator profiles in small sequential CPU simulations."""

import argparse
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True

from application_selection import application_options, compiled_algorithms


DURATION = r"\d+:[0-5]\d:[0-5]\d\.\d{2}"
ROW = re.compile(rf"(\S+)\s+(\d+)\s+({DURATION})\s+({DURATION})\s+(\d+\.\d{{2}})%")
TOTAL = re.compile(rf"Total\s+({DURATION})")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def duration_seconds(value):
    hours, minutes, seconds = value.split(":")
    result = 3600 * int(hours) + 60 * int(minutes) + float(seconds)
    require(math.isfinite(result) and result >= 0.0, f"invalid duration: {value}")
    return result


def read_profile(path, output):
    require(path.is_file(), f"missing phase profile: {path}")
    report = path.read_text()
    lines = report.splitlines()
    require(len(lines) >= 4 and lines[0].startswith("Profile:") and
            lines[1].split() == ["Phase", "Calls", "Total", "Mean", "Share"],
            f"invalid profile header: {path}")
    # Compare the actual printed section, including every row and its spacing.
    starts = list(re.finditer(r"^Profile:[^\n]*\n", output, re.MULTILINE))
    require(len(starts) == 1, f"expected one terminal profile section: {path}")
    terminal_end = re.search(rf"^Total\s+{DURATION}\n", output[starts[0].start():], re.MULTILINE)
    require(terminal_end is not None, f"missing terminal profile total: {path}")
    printed = output[starts[0].start():starts[0].start() + terminal_end.end()]
    require(printed == report, f"terminal and saved phase profiles differ: {path}")

    total_match = TOTAL.fullmatch(lines[-1])
    require(total_match is not None, f"invalid profile total: {path}")
    total = duration_seconds(total_match[1])
    rows = {}
    for line in lines[2:-1]:
        match = ROW.fullmatch(line)
        require(match is not None, f"invalid profile row in {path}: {line}")
        name, calls, elapsed, mean, share = match.groups()
        calls = int(calls)
        elapsed, mean, share = duration_seconds(elapsed), duration_seconds(mean), float(share)
        require(name not in rows and calls > 0, f"duplicate phase or invalid calls: {line}")
        require(math.isfinite(share) and 0.0 <= share <= 100.0, f"invalid share: {line}")
        # Each displayed duration independently rounds to the nearest 0.01 s.
        require(abs(elapsed - calls * mean) <= 0.005 * (calls + 1) + 1.e-9,
                f"profile mean disagrees with total/calls: {line}")
        rows[name] = {"calls": calls, "elapsed": elapsed, "share": share}
    require(bool(rows), f"empty phase profile: {path}")
    require(abs(math.fsum(row["elapsed"] for row in rows.values()) - total) <=
            0.005 * (len(rows) + 1) + 1.e-9,
            f"exclusive phase durations do not sum to the total: {path}")
    require(abs(math.fsum(row["share"] for row in rows.values()) - 100.0) <=
            0.005 * len(rows) + 1.e-9,
            f"phase shares do not sum to 100 percent: {path}")
    return rows


def require_phases(rows, names, label):
    missing = sorted(set(names) - rows.keys())
    require(not missing, f"{label}: missing executed child phases: {missing}")


def check_sph(rows, integrator):
    require_phases(rows, (
        "hydro.tree.create", "hydro.tree.destroy", "hydro.predictor", "hydro.corrector",
        "hydro.density", "hydro.density.reset", "hydro.density.activeSolve",
        "hydro.density.neighborSolve", "hydro.neighbors.initial", "hydro.neighbors.boxes",
        "hydro.neighbors.refresh", "hydro.force.close", "hydro.force.open", "hydro.kick.open",
        "hydro.caches.clear", "hydro.caches.clearHydro", "hydro.diagnostics.energies",
        "hydro.diagnostics.timesteps"), integrator)
    calls = lambda name: rows[name]["calls"]
    for name in ("hydro.tree.destroy", "hydro.predictor", "hydro.corrector", "hydro.density",
                 "hydro.neighbors.initial", "hydro.force.close", "hydro.diagnostics.energies"):
        require(calls(name) == calls("hydro.tree.create"),
                f"{integrator}: {name} call count differs from SPH force events")
    require(calls("hydro.force.open") == calls("hydro.kick.open"),
            f"{integrator}: opening force/kick counts differ")
    require(calls("hydro.density.reset") == 2 * calls("hydro.density"),
            f"{integrator}: missing density-stage reset")
    for name in ("hydro.density.activeSolve", "hydro.density.neighborSolve"):
        require(calls(name) >= calls("hydro.density"), f"{integrator}: missing density solve")
    density_passes = calls("hydro.density.activeSolve") + calls("hydro.density.neighborSolve")
    require(calls("hydro.neighbors.boxes") == density_passes and
            calls("hydro.neighbors.refresh") == density_passes,
            f"{integrator}: missing neighbor refresh for a density pass")
    if integrator == "traditional":
        require_phases(rows, ("hydro.rungs.limit",), integrator)
    else:
        require("hydro.rungs.limit" not in rows, "global integration unexpectedly limited rungs")


def check_hopkins(rows, label):
    require_phases(rows, (
        "hydro.hopkins.sources.correct", "hydro.hopkins.tree.create",
        "hydro.hopkins.tree.destroy", "hydro.hopkins.gather", "hydro.hopkins.scatter",
        "hydro.hopkins.neighbors.refresh", "hydro.hopkins.support.solve",
        "hydro.hopkins.geometry.solve", "hydro.hopkins.gradients",
        "hydro.hopkins.geometryRates", "hydro.hopkins.timesteps", "hydro.hopkins.flux.solve",
        "hydro.hopkins.flux.integrate", "hydro.hopkins.corrector",
        "hydro.hopkins.diagnostics.energies", "hydro.hopkins.diagnostics.soundWave"), label)
    require(rows["hydro.hopkins.tree.create"]["calls"] ==
            rows["hydro.hopkins.tree.destroy"]["calls"], f"{label}: tree lifecycle counts differ")
    require(rows["hydro.hopkins.flux.solve"]["calls"] ==
            rows["hydro.hopkins.flux.integrate"]["calls"] ==
            rows["hydro.hopkins.corrector"]["calls"], f"{label}: flux/corrector counts differ")


def run_case(executable, directory, options, deadline):
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    options = dict(options, **{"output.directory": str(directory / "simulation")})
    command = [str(executable), "--hpx:threads=2", "--hpx:bind=none"]
    command += [f"--{key}={value}" for key, value in options.items()]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    remaining = deadline - time.monotonic()
    require(remaining > 0.0, "phase-profile regression exceeded its total 115 s budget")
    log_path = directory / "run.log"
    with log_path.open("w") as log:
        with subprocess.Popen(command, cwd=directory, env=environment, text=True, stdout=log,
                              stderr=subprocess.STDOUT, start_new_session=True) as process:
            try:
                result = process.wait(timeout=min(30.0, remaining))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError(f"phase-profile fixture timed out; see {log_path}")
    output = log_path.read_text(errors="replace")
    require(result == 0, f"fixture exited {result}; see {log_path}\n{output[-6000:]}")
    rows = read_profile(directory / "simulation/profile.txt", output)
    require_phases(rows, ("initialize", "writeCheckpoint", "drift"), directory.name)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--hopkins", action="store_true", help="also check compiled Hopkins MFM/MFV")
    parser.add_argument("--gravity", action="store_true", help="also check available FMM/PMMM modes")
    args = parser.parse_args()
    executable, base = args.executable.resolve(), args.workdir.resolve()
    deadline = time.monotonic() + 115.0
    compiled = compiled_algorithms(str(executable))
    integrators = [name for name in ("traditional", "global") if name in compiled["integrator"]]
    require(bool(integrators), "phase-profile regression requires a hydro-capable integrator")
    common = {"compute.cuda": "false", "initial.particlesPerDimension": 8,
              "initial.useGlass": "false", "initial.closePack": "false",
              "gravity.enabled": "false", "gravity.meshSize": 8,
              "hydro.enabled": "true", "hydro.method": "sph",
              "chemistry.enabled": "false", "starFormation.enabled": "false",
              "run.verbose": "false", "run.scenario": "hydroWave", "run.endTime": 1.e-5,
              "run.numberOfSteps": 1, "run.maxIterations": 1, "run.minRung": 0,
              "checkpoint.readIndex": -1, "checkpoint.intervalSeconds": 0,
              "output.silo": "false", "analysis.groups": "false", "analysis.power": "false",
              "analysis.views": "false", "analysis.slice": "false", "analysis.tracers": "false",
              "lightcone.enabled": "false"}
    count = 0
    for integrator in integrators:
        selections = application_options(executable, integrator=integrator, hydro=True)
        label = f"sph_{integrator}"
        rows = run_case(executable, base / label, dict(common, **selections), deadline)
        check_sph(rows, integrator)
        print(f"{label}: child phases, call counts, totals and terminal/file equality passed")
        count += 1
        if args.hopkins:
            for method in ("mfm", "mfv"):
                label = f"{method}_{integrator}"
                options = dict(common, **selections)
                options.update({"run.scenario": "soundWave", "hydro.method": method,
                                "initial.particlesPerDimension": 16, "hydro.neighborNumber": 4})
                rows = run_case(executable, base / label, options, deadline)
                check_hopkins(rows, label)
                print(f"{label}: Hopkins child phases and report accounting passed")
                count += 1
    if args.gravity:
        for far in (name for name in compiled["far"] if name != "none"):
            for near in (compiled["near"] if far == "pmmm" else [None]):
                label = far + (f"_{near}" if near is not None else "")
                selections = application_options(executable, far=far, near=near,
                                                  integrator=integrators[0])
                options = dict(common, **selections)
                options.update({"run.scenario": "plummer", "gravity.enabled": "true",
                                "hydro.enabled": "false", "initial.particlesPerDimension": 4})
                rows = run_case(executable, base / label, options, deadline)
                if far == "fmm":
                    require_phases(rows, ("treeCreate.allocate", "treeCreate.build",
                                          "kick.walk", "kick.background"), label)
                else:
                    prefix = "pmmmSolve" if integrators[0] == "global" else "pmmmKick"
                    phases = ["particleCounts", "prepare", "deposit", "transform", "diagnostics"]
                    phases += (["evaluate"] if near == "pp" else
                               ["planGhosts", "startNearPipeline", "finishNearPipeline", "cleanup"])
                    require_phases(rows, (f"{prefix}.{name}" for name in phases), label)
                print(f"{label}: gravity child phases and report accounting passed")
                count += 1
    print(f"Phase profiling passed for {count} sequential CPU fixtures")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
