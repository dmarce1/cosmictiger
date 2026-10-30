# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026  Dominic C. Marcello
#
# Distributed under the Boost Software License, Version 1.0.
# See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.

"""Run isolated, bounded evolution checks for the restored SPH path."""

import argparse
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys


def run_simulation(command, workdir, log_name):
    log_path = workdir / log_name
    with log_path.open("w") as log:
        with subprocess.Popen(command, cwd=workdir, text=True, stdout=log,
                              stderr=subprocess.STDOUT, start_new_session=True) as process:
            try:
                returncode = process.wait(timeout=240)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError(f"simulation timed out; see {log_path}")
    if returncode:
        print(log_path.read_text())
        raise RuntimeError(f"simulation exited with {returncode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("sod", "dm_cpu", "dm_cuda", "sph", "sph_mpi"))
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--mpi-exec", default="mpiexec")
    parser.add_argument("--cpu", action="store_true", help="run hydrodynamics on the SIMD CPU backend")
    parser.add_argument("--minimum-rung", type=int,
                        help="set and verify the minimum starting timestep rung")
    parser.add_argument("--extra-option", action="append", default=[],
                        help="forward an application argument, for example "
                             "--extra-option=--hydro.kernelFamily=wendland")
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    # These are dedicated CTest output files; remove previous run data so a failed
    # invocation cannot accidentally pass by examining stale results.
    for name in ("sod.txt", "progress.txt", "energy.txt", "energy.dat", "rho.dat"):
        (args.workdir / name).unlink(missing_ok=True)
    output_directory = args.workdir / "silo snapshots"
    if output_directory.exists():
        shutil.rmtree(output_directory)
    for checkpoint_dir in args.workdir.glob("checkpoint.*"):
        if checkpoint_dir.is_dir() and re.fullmatch(r"checkpoint\.[0-9]+", checkpoint_dir.name):
            shutil.rmtree(checkpoint_dir)
    command = [str(args.executable), "--hpx:threads=2", "--hpx:bind=none",
               "--initial.particlesPerDimension=8", "--analysis.views=false",
               "--lightcone.enabled=false", "--analysis.power=false", "--analysis.groups=false"]
    if args.mode == "sod":
        # Select the left half of the domain, a directory containing spaces,
        # and a cadence that does not divide the final cycle (20).
        output_config = args.workdir / "silo.ini"
        output_config.write_text("[output]\nsilo = true\nxMax = 0.5\n"
                                 "siloInterval = 7\ndirectory = silo snapshots\n")
        command += ["--run.scenario=sod", "--hydro.enabled=true", "--compute.cuda=true",
                    f"--configuration={output_config.resolve()}"]
    elif args.mode == "sph":
        command += ["--hydro.enabled=true", "--compute.cuda=true",
                    "--run.maxIterations=1", "--checkpoint.intervalSeconds=0"]
    elif args.mode == "sph_mpi":
        command += ["--hydro.enabled=true", "--compute.cuda=true",
                    "--run.maxIterations=1", "--hpx:localities=2"]
        command = [args.mpi_exec, "-n", "2"] + command
    else:
        command += ["--hydro.enabled=false", "--run.maxIterations=1",
                    "--compute.cuda=" + str(args.mode == "dm_cuda").lower()]
    if args.mode != "sod":
        # Keep the established checkpoint/restart fixture layout explicit.
        command += ["--output.directory=."]
    if args.cpu:
        command = ["--compute.cuda=false" if value == "--compute.cuda=true" else value
                   for value in command]
    # This regression inspects detailed Silo configuration messages.
    if not any(option.startswith("--run.verbose=") for option in args.extra_option):
        command += ["--run.verbose=true"]
    command += args.extra_option
    if args.minimum_rung is not None:
        command += [f"--run.minRung={args.minimum_rung}"]
    run_simulation(command, args.workdir, "run.log")
    if args.mode == "sod":
        output_log = (args.workdir / "run.log").read_text()
        if "Silo particle output: x [0, 0.5], y [0, 1], z [0, 1]" not in output_log:
            raise RuntimeError("Sod run did not select the configured Silo box")
        snapshots = sorted(output_directory.glob("view.0.*.silo"))
        cycles = {int(path.name.split(".")[2]) for path in snapshots}
        if cycles != {0, 7, 14, 20} or any(path.stat().st_size == 0 for path in snapshots):
            raise RuntimeError(f"Sod Silo cadence/final snapshot mismatch: {sorted(cycles)}")
        print("Silo: cycles 0, 7, 14, 20 in the configured directory (including final snapshot)")
        rows = [[float(value) for value in line.split()]
                for line in (output_directory / "sod.txt").read_text().splitlines()]
        if len(rows) != 1000 or any(len(row) != 5 for row in rows):
            raise RuntimeError("Sod output did not contain 1000 complete samples")
        if any(not math.isfinite(value) for row in rows for value in row):
            raise RuntimeError("Sod output contains nonfinite values")
        if any(min(row[1:]) <= 0 for row in rows):
            raise RuntimeError("Sod density or pressure is nonpositive")
        relative_l1 = sum(abs(row[1] - row[2]) for row in rows) / sum(row[2] for row in rows)
        # This deliberately coarse test catches gross integration failures;
        # it is not a convergence study or a tight accuracy target.
        if relative_l1 >= 1.0:
            raise RuntimeError(f"Sod density relative L1 error is {relative_l1:.6g}")
        print(f"Sod: 1000 finite positive samples; density relative L1={relative_l1:.6g}")
    else:
        lines = (args.workdir / "progress.txt").read_text().splitlines()
        rows = []
        for line in lines:
            fields = line.split()
            if fields and re.fullmatch(r"[+\-]?[0-9].*", fields[0]):
                rows.append([float(value) for value in fields])
        output_log = (args.workdir / "run.log").read_text()
        if not re.search(r"^\s*Step\s+z.*TFLOP/s$", output_log, re.MULTILINE):
            raise RuntimeError("terminal progress is missing the TFLOP/s column")
        terminal_rows = re.findall(
            r"^\s*\d+(?:\s+[+\-]?[\d.]+e[+\-]\d+){4}"
            r"\s+\d+\s+\d+(?:\s+\d{2,}:\d{2}::\d{2}\.\d{2}){2}"
            r"\s+([+\-]?[\d.]+e[+\-]\d+)\s*$",
            output_log, re.MULTILINE)
        if len(terminal_rows) != len(rows) or any(
                not math.isfinite(float(rate)) or float(rate) <= 0 for rate in terminal_rows):
            raise RuntimeError("terminal FLOP rates are missing, nonfinite or nonpositive")
        if any(float(rate) != row[10] for rate, row in zip(terminal_rows, rows)):
            raise RuntimeError("terminal FLOP rates disagree with progress.txt")
        expected_steps = 2
        if len(rows) != expected_steps:
            raise RuntimeError(f"{args.mode} smoke test completed {len(rows)} steps, expected {expected_steps}")
        if any(len(row) != 14 for row in rows):
            raise RuntimeError(f"{args.mode} progress contains incomplete rows")
        if any(not math.isfinite(value) for row in rows for value in row):
            raise RuntimeError(f"{args.mode} progress contains nonfinite values")
        if args.minimum_rung is not None:
            if any(row[8] < args.minimum_rung for row in rows) or any(
                    row[7] < args.minimum_rung for row in rows[1:]):
                raise RuntimeError(f"{args.mode} ignored minimum starting rung {args.minimum_rung}")
        energies = [[float(value) for value in line.split()]
                    for line in (args.workdir / "energy.txt").read_text().splitlines()
                    if not line.startswith("#")]
        if not energies or any(len(row) != 10 for row in energies) or any(not math.isfinite(value) for row in energies for value in row):
            raise RuntimeError(f"{args.mode} energy output is empty or nonfinite")
        print(f"{args.mode}: {len(rows)} completed steps with finite physical diagnostics")
        if args.mode == "sph":
            expected = rows[-1]
            checkpoints = list(args.workdir.glob("checkpoint.[0-9]*/checkpoint.0.dat"))
            if not checkpoints:
                raise RuntimeError("SPH smoke test did not save a checkpoint")
            # The periodic checkpoint is saved before this iteration. The final
            # checkpoint is saved after it, so resuming that newer checkpoint
            # would correctly advance one step beyond the reference row.
            checkpoint = int(expected[1])
            if not (args.workdir / f"checkpoint.{checkpoint}" / "checkpoint.0.dat").is_file():
                raise RuntimeError("SPH smoke test did not save the reference iteration checkpoint")
            restart_directory = args.workdir / "restart"
            if restart_directory.exists():
                shutil.rmtree(restart_directory)
            restart_directory.mkdir()
            shutil.copytree(args.workdir / f"checkpoint.{checkpoint}",
                            restart_directory / f"checkpoint.{checkpoint}")
            restart_command = [arg for arg in command
                               if not arg.startswith(("--run.maxIterations=", "--checkpoint.intervalSeconds="))]
            restart_command += [f"--checkpoint.readIndex={checkpoint}", "--run.maxIterations=0"]
            run_simulation(restart_command, restart_directory, "restart.log")
            restarted = []
            for line in (restart_directory / "progress.txt").read_text().splitlines():
                fields = line.split()
                if fields and re.fullmatch(r"[+\-]?[0-9].*", fields[0]):
                    restarted.append([float(value) for value in fields])
            if len(restarted) != 1:
                raise RuntimeError("SPH checkpoint restart did not complete exactly one step")
            if len(restarted[0]) != 14:
                raise RuntimeError("SPH checkpoint restart progress row is incomplete")
            # Compare the physical clock and scale-factor columns, excluding
            # timings and throughput. Particle payloads have a separate test.
            for index in (1, 2, 3, 4, 5, 6, 7, 8):
                if not math.isclose(restarted[0][index], expected[index], rel_tol=1e-6, abs_tol=1e-12):
                    raise RuntimeError(f"SPH restart column {index} differs: "
                                       f"{restarted[0][index]} != {expected[index]}")
            print(f"SPH checkpoint {checkpoint}: resumed physical clock matches uninterrupted evolution")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
