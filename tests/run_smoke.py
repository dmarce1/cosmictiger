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

from application_selection import application_arguments
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


def validateFlopRates(outputLog, rows):
    if any(len(row) != 16 for row in rows):
        raise RuntimeError("progress contains incomplete rows")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise RuntimeError("progress contains nonfinite values")
    if not re.search(
            r"^\s*Step\s+z.*\s+FP32 GF/s\s+FP64 GF/s\s+GFLOP/s(?:\s+\d+)+\s*$",
            outputLog, re.MULTILINE):
        raise RuntimeError("terminal progress is missing the precision FLOP rate columns")
    terminalRows = re.findall(
        r"^\s*\d+(?:\s+[+\-]?[\d.]+e[+\-]\d+){4}"
        r"\s+\d+\s+\d+(?:\s+\d{2,}:\d{2}:\d{2}\.\d{2}){2}"
        r"\s+(\S+)\s+(\S+)\s+(\S+)(?:\s+\d+)+\s*$",
        outputLog, re.MULTILINE)
    if len(terminalRows) != len(rows):
        raise RuntimeError("terminal FLOP rate rows disagree with progress.txt")
    for terminalRow, row in zip(terminalRows, rows):
        single, double, total = rates = tuple(map(float, terminalRow))
        if any(not math.isfinite(rate) or rate < 0.0 for rate in rates) or total <= 0.0:
            raise RuntimeError("terminal FLOP rates are nonfinite, negative or have no work")
        if rates != (row[14], row[15], row[10]):
            raise RuntimeError("terminal FLOP rates disagree with progress.txt")
        # Each rate is independently rounded to four significant figures.
        if not math.isclose(single + double, total, rel_tol=0.002):
            raise RuntimeError("precision FLOP rates do not sum to the total")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("sod", "dm_cpu", "dm_cuda", "sph", "sph_analysis", "sph_mpi"))
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--mpi-exec", default="mpiexec")
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2,
                        help="HPX worker threads per locality")
    parser.add_argument("--cpu", action="store_true", help="run hydrodynamics on the SIMD CPU backend")
    parser.add_argument("--gravity-cuda", action="store_true",
                        help="with sph --cpu, run CUDA gravity alongside CPU hydrodynamics and drift")
    parser.add_argument("--minimum-rung", type=int,
                        help="set and verify the minimum starting timestep rung")
    parser.add_argument("--extra-option", action="append", default=[],
                        help="forward an application argument, for example "
                             "--extra-option=--hydro.kernelFamily=wendland")
    args = parser.parse_args()
    if args.gravity_cuda and (not args.cpu or args.mode != "sph"):
        parser.error("--gravity-cuda requires mode sph and --cpu")
    args.workdir.mkdir(parents=True, exist_ok=True)
    # These are dedicated CTest output files; remove previous run data so a failed
    # invocation cannot accidentally pass by examining stale results.
    for name in ("sod.txt", "progress.txt", "energy.txt", "energy.dat", "rho.dat"):
        (args.workdir / name).unlink(missing_ok=True)
    output_directory = args.workdir / "silo snapshots"
    if output_directory.exists():
        shutil.rmtree(output_directory)
    for checkpoint_dir in args.workdir.glob("checkpoint.*"):
        if checkpoint_dir.is_symlink() and checkpoint_dir.name.startswith("checkpoint.z."):
            checkpoint_dir.unlink()
        elif checkpoint_dir.is_dir() and re.fullmatch(r"checkpoint\.[0-9]+(?:\.incomplete)?", checkpoint_dir.name):
            shutil.rmtree(checkpoint_dir)
    # sph_analysis runs the dark-matter-only halo finder and the lightcone with gas present.
    analysis = str(args.mode == "sph_analysis").lower()
    analysis_cuda = str(args.mode == "sph_analysis" and not args.cpu).lower()
    command = [str(args.executable), f"--hpx:threads={args.threads}", "--hpx:bind=none",
               "--run.darkMatterParticleCount=512", "--run.gasParticleCount=512", "--analysis.views=false",
               f"--lightcone.enabled={analysis}", "--analysis.power=false",
               f"--analysis.groups={analysis}",
               f"--analysis.groups.cuda={analysis_cuda}", f"--lightcone.cuda={analysis_cuda}"]
    hydro_mode = args.mode in ("sod", "sph", "sph_analysis", "sph_mpi")
    selector_arguments = application_arguments(args.executable.resolve(), hydro=hydro_mode)
    # SPH has no self-gravity coupling in the PMMM solver; dark-matter runs need its mesh.
    selector_arguments += (["--gravity.enabled=false"] if hydro_mode else ["--gravity.meshSize=8"])
    command += [argument for argument in selector_arguments if not any(
        option.split("=", 1)[0] == argument.split("=", 1)[0] for option in args.extra_option)]
    if args.mode == "sod":
        # Select the left half of the domain, a directory containing spaces,
        # with snapshots at every uniformly spaced phase including both endpoints.
        output_config = args.workdir / "silo.ini"
        output_config.write_text("[output]\nsilo = true\nxMax = 0.5\n"
                                 "directory = silo snapshots\n")
        command += ["--run.scenario=sod", "--hydro.enabled=true", "--gravity.cuda=true", "--hydro.cuda=true", "--drift.cuda=true",
                    f"--configuration={output_config.resolve()}"]
    elif args.mode in ("sph", "sph_analysis"):
        command += ["--hydro.enabled=true", "--gravity.cuda=true", "--hydro.cuda=true", "--drift.cuda=true",
                    "--run.maxIterations=2", "--checkpoint.intervalSeconds=0"]
    elif args.mode == "sph_mpi":
        command += ["--hydro.enabled=true", "--gravity.cuda=true", "--hydro.cuda=true", "--drift.cuda=true",
                    "--run.maxIterations=2", "--hpx:localities=2"]
        command = [args.mpi_exec, "-n", "2"] + command
    else:
        command += ["--hydro.enabled=false", "--run.maxIterations=2",
                    "--gravity.cuda=" + str(args.mode == "dm_cuda").lower(),
                    "--drift.cuda=" + str(args.mode == "dm_cuda").lower()]
    if args.mode != "sod":
        # Keep the established checkpoint/restart fixture layout explicit.
        command += ["--output.directory=."]
    if args.cpu:
        command = [value.replace("=true", "=false") if value in ("--gravity.cuda=true", "--hydro.cuda=true", "--drift.cuda=true") else value
                   for value in command]
    if args.gravity_cuda:
        command = ["--gravity.cuda=true" if value.startswith("--gravity.cuda=") else value
                   for value in command]
    # This regression inspects detailed Silo configuration messages.
    if not any(option.startswith("--run.verbose=") for option in args.extra_option):
        command += ["--run.verbose=true"]
    command += args.extra_option
    if args.minimum_rung is not None:
        command += [f"--run.minRung={args.minimum_rung}"]
    def validate_mixed_backends(output_log):
        if not args.gravity_cuda:
            return
        for key, expected in (("gravity.cuda", "T"), ("hydro.cuda", "F"), ("drift.cuda", "F")):
            if not re.search(r"^" + re.escape(key) + r"\s*:\s*" + expected + r"\s*$",
                             output_log, re.MULTILINE):
                raise RuntimeError(f"mixed SPH fixture did not retain {key}={expected}")
        if "CUDA kick arena (FMM order" not in output_log:
            raise RuntimeError("mixed SPH fixture did not execute the CUDA FMM arena path")

    run_simulation(command, args.workdir, "run.log")
    validate_mixed_backends((args.workdir / "run.log").read_text())
    if args.mode == "sod":
        output_log = (args.workdir / "run.log").read_text()
        if "Silo particle output: x [0, 0.5], y [0, 1], z [0, 1]" not in output_log:
            raise RuntimeError("Sod run did not select the configured Silo box")
        snapshots = sorted(output_directory.glob("view.0.*.silo"))
        cycles = {int(path.name.split(".")[2]) for path in snapshots}
        if cycles != set(range(21)) or any(path.stat().st_size == 0 for path in snapshots):
            raise RuntimeError(f"Sod Silo cadence/final snapshot mismatch: {sorted(cycles)}")
        print("Silo: all 21 equally spaced output phases, including both endpoints")
        rows = [[float(value) for value in line.split()]
                for line in (output_directory / "sod.txt").read_text().splitlines()]
        # The N8 fixture has 576 particles with the exact 8:1 density lattice.
        # Small domains now emit each particle once instead of repeated draws.
        expected_samples = 576
        if len(rows) != expected_samples or any(len(row) != 5 for row in rows):
            raise RuntimeError(f"Sod output did not contain {expected_samples} complete samples")
        if any(not math.isfinite(value) for row in rows for value in row):
            raise RuntimeError("Sod output contains nonfinite values")
        if any(min(row[1:]) <= 0 for row in rows):
            raise RuntimeError("Sod density or pressure is nonpositive")
        relative_l1 = sum(abs(row[1] - row[2]) for row in rows) / sum(row[2] for row in rows)
        # This deliberately coarse test catches gross integration failures;
        # it is not a convergence study or a tight accuracy target.
        if relative_l1 >= 1.0:
            raise RuntimeError(f"Sod density relative L1 error is {relative_l1:.6g}")
        print(f"Sod: {len(rows)} finite positive samples; density relative L1={relative_l1:.6g}")
    else:
        lines = (args.workdir / "progress.txt").read_text().splitlines()
        rows = []
        for line in lines:
            fields = line.split()
            if fields and re.fullmatch(r"[+\-]?[0-9].*", fields[0]):
                rows.append([float(value) for value in fields])
        output_log = (args.workdir / "run.log").read_text()
        validateFlopRates(output_log, rows)
        expected_steps = 2
        if len(rows) != expected_steps:
            raise RuntimeError(f"{args.mode} smoke test completed {len(rows)} steps, expected {expected_steps}")
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
        if args.mode == "sph_analysis":
            backend_value = "F" if args.cpu else "T"
            for key in ("analysis.groups.cuda", "lightcone.cuda"):
                if not re.search(r"^" + re.escape(key) + r"\s*:\s*" + backend_value + r"\s*$",
                                 output_log, re.MULTILINE):
                    raise RuntimeError(f"sph_analysis did not select {key}={analysis_cuda}")
            if "Doing groups" not in output_log:
                raise RuntimeError("sph_analysis did not run the halo finder")
            sidecars = sorted(args.workdir.glob("groups.*/rockstar.0.dat.format"))
            if not sidecars:
                raise RuntimeError("sph_analysis produced no ROCKSTAR catalogue sidecar")
            metadata = dict(line.split() for line in sidecars[0].read_text().splitlines() if line.strip())
            if metadata.get("halo_particles") != "dark_matter":
                raise RuntimeError("ROCKSTAR sidecar does not record dark-matter-only halo finding")
            if not re.search(r"^analysis.haloParticles\s*:\s*darkMatter\s*$", output_log, re.MULTILINE):
                raise RuntimeError("option metadata does not record dark-matter-only halo finding")
            print("sph_analysis: dark-matter-only halo finding and lightcone ran with SPH enabled")
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
            restart_command += [f"--checkpoint.readIndex={checkpoint}", "--run.maxIterations=1"]
            run_simulation(restart_command, restart_directory, "restart.log")
            validate_mixed_backends((restart_directory / "restart.log").read_text())
            restarted = []
            for line in (restart_directory / "progress.txt").read_text().splitlines():
                fields = line.split()
                if fields and re.fullmatch(r"[+\-]?[0-9].*", fields[0]):
                    restarted.append([float(value) for value in fields])
            if len(restarted) != 1:
                raise RuntimeError("SPH checkpoint restart did not complete exactly one step")
            if len(restarted[0]) != 16:
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
