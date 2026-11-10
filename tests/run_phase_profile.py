# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# SPDX-License-Identifier: BSL-1.0

"""Check coordinator and gravity work profiles in small sequential simulations."""

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
WORK_ROW = re.compile(rf"(\S+)\s+(\d+)\s+({DURATION})\s+(\d+\.\d{{3}})")
WORK_SECTIONS = {
    "Profile detail: rank-0 local host elapsed intervals (overlap; not CPU time)":
        "Intervals may overlap and are not additive to the coordinator total.",
    "Profile detail: rank-0 CUDA device elapsed intervals (overlap possible)":
        "Device intervals are not additive to host or coordinator totals.",
}
DEVICE_SECTION = (
    "Profile device cycles: rank-0 summed CUDA block cycles (overlapping; not wall time)")
DEVICE_EXPLANATION = (
    "Shares divide summed instrumented block cycles; no conversion to wall time.")
DEVICE_ROW = re.compile(r"(kick\.device\.\S+)\s+(\d+)\s+(\d+)\s+(\d+\.\d{3})\s+(\d+\.\d{2})%")
DEVICE_STATS = re.compile(
    r"Recorded (blocks|launches): (\d+); cycles min/mean/max: (\d+) / (\d+\.\d{3}) / (\d+)")
DEVICE_PHASES = tuple(f"kick.device.{name}" for name in (
    "prepare", "traversal", "M2L", "P2L", "M2P", "P2P", "leafKick",
    "copyWait", "schedule", "cleanup"))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def duration_seconds(value):
    hours, minutes, seconds = value.split(":")
    result = 3600 * int(hours) + 60 * int(minutes) + float(seconds)
    require(math.isfinite(result) and result >= 0.0, f"invalid duration: {value}")
    return result


def read_device_cycles(lines, index, path, detail_rows):
    cycle_rows = {}
    while index < len(lines) and lines[index].startswith("kick.device."):
        line = lines[index]
        match = DEVICE_ROW.fullmatch(line)
        require(match is not None, f"invalid device-cycle row in {path}: {line}")
        name, calls, cycles, mean, share = match.groups()
        calls, cycles, mean, share = int(calls), int(cycles), float(mean), float(share)
        require(name in DEVICE_PHASES and name not in detail_rows and calls > 0,
                f"unknown/duplicate device phase or invalid calls: {line}")
        require(math.isfinite(mean) and mean >= 0.0 and
                math.isfinite(share) and 0.0 <= share <= 100.0,
                f"invalid device-cycle mean or share: {line}")
        # Cycles are exact integers; only the printed mean is rounded.
        require(abs(mean - cycles / calls) <= 0.0005 + 2 * math.ulp(mean),
                f"device-cycle mean disagrees with cycles/calls: {line}")
        row = {"calls": calls, "cycles": cycles, "mean": mean, "share": share}
        cycle_rows[name] = row
        detail_rows[name] = row
        index += 1
    require(bool(cycle_rows), f"empty device-cycle profile: {path}")
    cycles = sum(row["cycles"] for row in cycle_rows.values())
    for name, row in cycle_rows.items():
        expected_share = 100.0 * row["cycles"] / cycles if cycles else 0.0
        require(abs(row["share"] - expected_share) <= 0.005 + 1.e-9,
                f"{name}: device share disagrees with summed cycles: {path}")
    require(abs(math.fsum(row["share"] for row in cycle_rows.values()) -
                (100.0 if cycles else 0.0)) <= 0.005 * len(cycle_rows) + 1.e-9,
            f"device-cycle shares do not sum correctly: {path}")
    stats = {}
    for label in ("blocks", "launches"):
        require(index < len(lines), f"missing device-cycle {label} statistics: {path}")
        match = DEVICE_STATS.fullmatch(lines[index])
        require(match is not None and match[1] == label,
                f"invalid device-cycle {label} statistics: {path}")
        samples, minimum, mean, maximum = (
            int(match[2]), int(match[3]), float(match[4]), int(match[5]))
        require(samples > 0 and math.isfinite(mean) and minimum <= mean <= maximum,
                f"invalid device-cycle {label} range or sample count: {path}")
        require(minimum * samples <= cycles <= maximum * samples and
                abs(mean - cycles / samples) <= 0.0005 + 2 * math.ulp(mean),
                f"device-cycle {label} statistics disagree with phase totals: {path}")
        stats[label] = samples
        index += 1
    require(stats["launches"] <= stats["blocks"] <=
            sum(row["calls"] for row in cycle_rows.values()),
            f"inconsistent device launch/block/call counts: {path}")
    return index, stats


def read_profile(path, output):
    require(path.is_file(), f"missing phase profile: {path}")
    report = path.read_text()
    lines = report.splitlines()
    require(len(lines) >= 4 and lines[0].startswith("Profile:") and
            lines[1].split() == ["Phase", "Calls", "Total", "Mean", "Share"],
            f"invalid profile header: {path}")
    # Compare the entire printed report, including appended work tables and spacing.
    starts = list(re.finditer(r"^Profile:[^\n]*\n", output, re.MULTILINE))
    require(len(starts) == 1, f"expected one terminal profile section: {path}")
    printed = output[starts[0].start():starts[0].start() + len(report)]
    require(printed == report, f"terminal and saved phase profiles differ: {path}")

    totals = [(index, TOTAL.fullmatch(line)) for index, line in enumerate(lines)]
    totals = [(index, match) for index, match in totals if match is not None]
    require(len(totals) == 1, f"expected one coordinator profile total: {path}")
    coordinator_end, total_match = totals[0]
    total = duration_seconds(total_match[1])
    rows = {}
    for line in lines[2:coordinator_end]:
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
    detail_rows = {}
    detail_sections = set()
    device_stats = None
    index = coordinator_end + 1
    while index < len(lines):
        require(lines[index] == "" and len(lines) - index >= 5,
                f"invalid work-profile section: {path}")
        title, explanation, header = lines[index + 1:index + 4]
        if title == DEVICE_SECTION:
            require(title not in detail_sections and explanation == DEVICE_EXPLANATION and
                    header.split() == ["Work", "Calls", "Cycles", "Mean(cycles)", "Share"],
                    f"invalid or duplicate device-cycle header: {path}")
            detail_sections.add(title)
            index, device_stats = read_device_cycles(lines, index + 4, path, detail_rows)
            continue
        require(title in WORK_SECTIONS and title not in detail_sections,
                f"unknown or duplicate work-profile section: {title}")
        require(explanation == WORK_SECTIONS[title] and
                header.split() == ["Work", "Calls", "Total", "Mean(us)"],
                f"invalid work-profile header: {path}")
        detail_sections.add(title)
        index += 4
        section_start = index
        while index < len(lines) and lines[index]:
            line = lines[index]
            match = WORK_ROW.fullmatch(line)
            require(match is not None, f"invalid work-profile row in {path}: {line}")
            name, calls, elapsed, mean = match.groups()
            calls, elapsed, mean = int(calls), duration_seconds(elapsed), float(mean)
            require(name not in detail_rows and calls > 0,
                    f"duplicate work phase or invalid calls: {line}")
            require(math.isfinite(mean) and mean >= 0.0, f"invalid work-profile mean: {line}")
            # Total rounds to 0.01 s, while the mean rounds to 0.001 microseconds.
            require(abs(elapsed - calls * mean * 1.e-6) <=
                    0.005 + calls * 0.0005e-6 + 1.e-9,
                    f"work-profile mean disagrees with total/calls: {line}")
            detail_rows[name] = {"calls": calls, "elapsed": elapsed, "mean": mean}
            index += 1
        require(index > section_start, f"empty work-profile section: {path}")
    if device_stats is not None and "kick.gpu.kernel" in detail_rows:
        require(device_stats["launches"] == detail_rows["kick.gpu.kernel"]["calls"],
                f"recorded device launches differ from CUDA event calls: {path}")
    # Concurrent host intervals and device intervals are deliberately not summed.
    return rows, detail_rows


def require_phases(rows, names, label):
    missing = sorted(set(names) - rows.keys())
    require(not missing, f"{label}: missing executed child phases: {missing}")


def check_sph(rows, integrator):
    require_phases(rows, (
        "hydro.tree.create", "hydro.tree.destroy", "hydro.predictor", "hydro.corrector",
        "hydro.density", "hydro.density.reset", "hydro.density.activeSolve",
        "hydro.density.neighborSolve", "hydro.neighbors.initial", "hydro.neighbors.boxes",
        "hydro.neighbors.refresh", "hydro.force.close", "hydro.rungs.commit", "hydro.kick.open",
        "hydro.caches.clear", "hydro.caches.clearHydro", "hydro.diagnostics.energies",
        "hydro.diagnostics.timesteps"), integrator)
    calls = lambda name: rows[name]["calls"]
    for name in ("hydro.tree.destroy", "hydro.predictor", "hydro.corrector", "hydro.density",
                 "hydro.neighbors.initial", "hydro.force.close", "hydro.diagnostics.energies"):
        require(calls(name) == calls("hydro.tree.create"),
                f"{integrator}: {name} call count differs from SPH force events")
    require(calls("hydro.rungs.commit") == calls("hydro.kick.open"),
            f"{integrator}: rung commit/opening kick counts differ")
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


def run_case(executable, directory, options, deadline):
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    options = dict(options, **{"output.directory": str(directory / "simulation")})
    command = [str(executable), "--hpx:threads=2", "--hpx:bind=none"]
    command += [f"--{key}={value}" for key, value in options.items()]
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1")
    if options.get("gravity.cuda") != "true":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    remaining = deadline - time.monotonic()
    require(remaining > 0.0, "phase-profile regression exceeded its total runtime budget")
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
    rows, detail_rows = read_profile(directory / "simulation/profile.txt", output)
    require_phases(rows, ("initialize", "writeCheckpoint", "drift"), directory.name)
    if options.get("gravity.cuda") != "true":
        require(not any("cycles" in row for row in detail_rows.values()),
                f"{directory.name}: CPU fixture reported CUDA device cycles")
    return rows, detail_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--gravity", action="store_true", help="also check available FMM/PMMM modes")
    parser.add_argument("--cuda", action="store_true", help="also check one FMM CUDA fixture")
    parser.add_argument("--device-profile", action="store_true",
                        help="require per-block CUDA cycle profiling in the --cuda fixture")
    args = parser.parse_args()
    if args.device_profile and not args.cuda:
        parser.error("--device-profile requires --cuda")
    executable, base = args.executable.resolve(), args.workdir.resolve()
    deadline = time.monotonic() + (160.0 if args.cuda else 115.0)
    compiled = compiled_algorithms(str(executable))
    integrators = [name for name in ("traditional", "global") if name in compiled["integrator"]]
    require(bool(integrators), "phase-profile regression requires a hydro-capable integrator")
    common = {"gravity.cuda": "false", "run.darkMatterParticleCount": 512, "run.gasParticleCount": 512,
              "scenario.cosmology.useGlass": "false", "initial.closePack": "false",
              "gravity.enabled": "false", "gravity.meshSize": 8,
              "hydro.enabled": "true", "hydro.method": "sph",
              "chemistry.enabled": "false", "starFormation.enabled": "false",
              "run.verbose": "false", "run.scenario": "hydroWave", "run.endTime": 1.e-5,
              "run.numberOfOutputs": 2, "run.maxIterations": 1, "run.minRung": 0,
              "checkpoint.readIndex": -1, "checkpoint.intervalSeconds": 0,
              "output.silo": "false", "analysis.groups": "false", "analysis.power": "false",
              "analysis.views": "false", "analysis.slice": "false", "analysis.tracers": "false",
              "lightcone.enabled": "false"}
    count = 0
    for integrator in integrators:
        selections = application_options(executable, integrator=integrator, hydro=True)
        label = f"sph_{integrator}"
        rows, detail_rows = run_case(executable, base / label, dict(common, **selections), deadline)
        check_sph(rows, integrator)
        require(not detail_rows, f"{label}: gravity-disabled run reported gravity work")
        print(f"{label}: child phases, call counts, totals and terminal/file equality passed")
        count += 1
    if args.gravity and "pmmm" in compiled["far"]:
        for near in compiled["near"]:
            label = "pmmm_" + near
            selections = application_options(executable, near=near, integrator=integrators[0])
            options = dict(common, **selections)
            options.update({"run.scenario": "plummer", "gravity.enabled": "true",
                            "hydro.enabled": "false", "run.darkMatterParticleCount": 64, "run.gasParticleCount": 64})
            rows, detail_rows = run_case(executable, base / label, options, deadline)
            prefix = "pmmmSolve" if integrators[0] == "global" else "pmmmKick"
            phases = ["particleCounts", "prepare", "deposit", "transform", "diagnostics"]
            phases += (["evaluate"] if near == "pp" else
                       ["planGhosts", "startNearPipeline", "finishNearPipeline", "cleanup"])
            require_phases(rows, (f"{prefix}.{name}" for name in phases), label)
            print(f"{label}: gravity child phases and report accounting passed")
            count += 1
    if args.cuda and "pmmm" in compiled["far"]:
        # The near-field FMM batches run on the device; the far field stays on the host.
        label = "pmmm_fmm_cuda"
        selections = application_options(executable, near="fmm", integrator=integrators[0])
        options = dict(common, **selections)
        options.update({"run.scenario": "plummer", "gravity.enabled": "true",
                        "hydro.enabled": "false", "run.darkMatterParticleCount": 512, "run.gasParticleCount": 512,
                        "gravity.cuda": "true"})
        rows, detail_rows = run_case(executable, base / label, options, deadline)
        prefix = "pmmmSolve" if integrators[0] == "global" else "pmmmKick"
        require_phases(rows, (f"{prefix}.{name}" for name in
                              ("particleCounts", "prepare", "deposit", "transform",
                               "planGhosts", "startNearPipeline", "finishNearPipeline", "cleanup")), label)
        require_phases(detail_rows, (
            "kick.gpu.hostToDevice", "kick.gpu.kernel", "kick.gpu.deviceToHost"), label)
        device_calls = [detail_rows[name]["calls"] for name in (
            "kick.gpu.hostToDevice", "kick.gpu.kernel", "kick.gpu.deviceToHost")]
        require(device_calls[0] > 0 and len(set(device_calls)) == 1,
                f"{label}: transfer/kernel event call counts differ or are empty")
        if args.device_profile:
            require_phases(detail_rows, DEVICE_PHASES, label)
            require(all("cycles" in detail_rows[name] for name in DEVICE_PHASES),
                    f"{label}: device phase appeared outside the cycle table")
            require(sum(detail_rows[name]["cycles"] for name in DEVICE_PHASES) > 0,
                    f"{label}: device phases recorded no cycles")
        print(f"{label}: host staging, CUDA event phases and report accounting passed")
        count += 1
    device_label = "CPU/CUDA" if args.cuda else "CPU"
    print(f"Phase profiling passed for {count} sequential {device_label} fixtures")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
