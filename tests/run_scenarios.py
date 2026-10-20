# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# SPDX-License-Identifier: GPL-2.0-or-later

"""Exercise first-class scenario dispatch, solver selection, and output isolation."""

import argparse
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import struct
import sys

from run_checkpoint import checkManifest


SCENARIOS = (
    ("plummer", "plummer", True, False, False),
    ("sod", "sod", False, True, False),
    ("blast", "blast", False, True, False),
    ("rayleigh_taylor", "rayleighTaylor", True, True, False),
    ("kelvin_helmholtz", "kelvinHelmholtz", False, True, False),
    ("disc", "disc", True, True, False),
    ("polytropic_star", "polytropicStar", True, True, False),
    ("hydro_wave", "hydroWave", False, True, False),
    ("cosmology", "cosmology", True, True, True),
    ("glass", "glass", True, False, False),
)


def run_process(command, directory, success=True):
    log_path = directory / "run.log"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1")
    with log_path.open("w") as log:
        with subprocess.Popen(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT,
                              text=True, env=environment, start_new_session=True) as process:
            try:
                result = process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError(f"scenario timed out; see {log_path}")
    output = log_path.read_text()
    if (result == 0) != success:
        raise RuntimeError(f"unexpected exit {result}; see {log_path}\n{output[-6000:]}")
    return output


def setting(output, key):
    matches = re.findall(r"^" + re.escape(key) + r"\s*:\s*(\S+)\s*$", output, re.MULTILINE)
    if not matches:
        raise RuntimeError(f"missing resolved option {key}")
    return matches[-1]


def check_bool(output, key, expected):
    actual = setting(output, key).lower()
    if actual not in (("t", "true", "1") if expected else ("f", "false", "0")):
        raise RuntimeError(f"{key} is {actual}, expected {expected}")


def check_finite_rows(path):
    if not path.is_file():
        raise RuntimeError(f"missing evolution diagnostic {path}")
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"empty evolution diagnostic {path}")
    if any(not math.isfinite(float(value)) for row in rows for value in row):
        raise RuntimeError(f"nonfinite evolution diagnostic {path}")



def hydro_particle_payload(checkpoint, expected_count=None):
    """Read the all-gas physical payload, excluding native driver timing/padding."""
    data = (checkpoint / "checkpoint.0.dat").read_bytes()
    prefix = struct.pack("=2I", 0x43545350, 2)
    offset = data.rfind(prefix)
    if offset < 0:
        raise RuntimeError(f"missing versioned SPH payload in {checkpoint}")
    header = struct.unpack_from("=7I", data, offset)
    integer_bytes = header[2]
    if integer_bytes not in (4, 8) or header[5:] != (3, 4):
        raise RuntimeError(f"unsupported particle layout in {checkpoint}: {header}")
    count = int.from_bytes(data[offset + 28:offset + 28 + integer_bytes],
                           sys.byteorder, signed=True)
    if count <= 0 or (expected_count is not None and count != expected_count):
        raise RuntimeError(f"unexpected gas count in {checkpoint}: {count}")
    # Particle positions, velocities, rungs, species, and gas index precede
    # the gas header. The 38 bytes below are rates, density, flags, and six
    # predictor/gravity acceleration floats in visitSphCheckpointArrays().
    start = offset - 2 * integer_bytes - count * (26 + integer_bytes)
    end = offset + 28 + integer_bytes + count * (header[3] + header[4] + integer_bytes + 38)
    if start < 0 or end > len(data):
        raise RuntimeError(f"truncated physical particle payload in {checkpoint}")
    counts = [int.from_bytes(data[start + i * integer_bytes:start + (i + 1) * integer_bytes],
                             sys.byteorder, signed=True) for i in range(2)]
    if counts != [count, count]:
        raise RuntimeError(f"restart fixture is not all gas: {counts}")
    return data[start:end]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("power_file", type=Path)
    parser.add_argument("workdir", type=Path)
    args = parser.parse_args()
    executable = args.executable.resolve()
    recfast = executable.parent / "recfast"
    root = Path(__file__).resolve().parents[1]
    base = args.workdir.resolve()
    base.mkdir(parents=True, exist_ok=True)
    command = [str(executable), "--hpx:threads=2", "--hpx:bind=none"]
    expected_files = {stem + ".ini" for stem, *_ in SCENARIOS}
    actual_files = {path.name for path in (root / "configurations").glob("*.ini")}
    if actual_files != expected_files:
        raise RuntimeError(f"scenario configuration inventory differs: "
                           f"{sorted(actual_files ^ expected_files)}")

    def prepare(name):
        directory = base / name
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir()
        return directory

    def exercise(stem, scenario, gravity, hydro, chemistry, name=None, overrides=None,
                 relative_output=None, silo=True):
        directory = prepare(name or scenario)
        # Runtime inputs remain in the launch directory while all products go
        # into output.directory. This catches accidental chdir-based routing.
        inputs = set()
        if scenario == "cosmology":
            shutil.copyfile(args.power_file, directory / "power.init")
            inputs.add("power.init")
            if not recfast.is_file():
                raise RuntimeError(f"cosmology runtime dependency is missing: {recfast}")
            shutil.copy2(recfast, directory / "recfast")
            inputs.add("recfast")
        options = {
            "configuration": str(root / "configurations" / (stem + ".ini")),
            "compute.cuda": "false",
            "initial.particlesPerDimension": "16" if scenario == "disc" else "8",
            "run.maxIterations": "1",
            # The cold disc must support its actual configured coarse timestep.
            "run.endTime": "0" if scenario == "disc" else "0.00001",
            "checkpoint.intervalSeconds": "0",
            "chemistry.synchronizedRung": "8",
        }
        options.update(overrides or {})
        output = run_process(command + [f"--{key}={value}" for key, value in options.items()],
                             directory)
        if setting(output, "run.scenario") != scenario:
            raise RuntimeError(f"{name or scenario}: incorrect scenario dispatch")
        for key, enabled in (("gravity.enabled", gravity), ("hydro.enabled", hydro),
                             ("chemistry.enabled", chemistry)):
            check_bool(output, key, enabled)
        destination = directory / (relative_output or f"output/{scenario}")
        if not destination.is_dir():
            raise RuntimeError(f"missing scenario output directory {destination}")
        snapshots = list(destination.glob("view.*.silo"))
        if silo and (not snapshots or any(path.stat().st_size == 0 for path in snapshots)):
            raise RuntimeError(f"missing or empty particle snapshots in {destination}")
        if not silo and snapshots:
            raise RuntimeError(f"disabled particle snapshots were written in {destination}")
        energy_name = "energy.dat" if scenario not in ("plummer", "glass", "cosmology") else "energy.txt"
        check_finite_rows(destination / energy_name)
        if hydro and scenario != "cosmology":
            check_finite_rows(destination / "rho.dat")
        if not hydro and "sph_run(SPH_RUN_HYDRO)" in output:
            raise RuntimeError(f"{scenario} invoked hydrodynamics while it was disabled")
        if not gravity and "Do kick\n" in output:
            raise RuntimeError(f"{scenario} invoked the gravity kick while it was disabled")
        checkpoints = [path for path in destination.glob("checkpoint.*")
                       if path.is_dir() and re.fullmatch(r"checkpoint\.[0-9]+", path.name)]
        if not checkpoints:
            raise RuntimeError(f"{scenario} did not publish a final checkpoint")
        for checkpoint in checkpoints:
            checkManifest(checkpoint, int(checkpoint.name.split(".")[1]),
                          scenario=scenario, endTime=float(options["run.endTime"]), gravity=gravity)
        allowed = inputs | {"run.log", Path(relative_output or "output").parts[0]}
        unexpected = {path.name for path in directory.iterdir()} - allowed
        if unexpected:
            raise RuntimeError(f"outputs escaped {destination}: {sorted(unexpected)}")
        print(f"{name or scenario}: solver selection, finite diagnostics, and output containment passed")
        return directory, destination, options

    results = {case[1]: exercise(*case) for case in SCENARIOS}

    exercise("rayleigh_taylor", "rayleighTaylor", False, True, False,
             name="rayleighTaylor_no_gravity", overrides={"gravity.enabled": "false"})
    # A CLI scenario override exercises dispatch independently of the INI name;
    # disabled Silo must not prevent text diagnostics or directory creation.
    exercise("sod", "hydroWave", False, True, False, name="scenario_cli_override",
             overrides={"run.scenario": "hydroWave", "output.directory": "custom output",
                        "output.silo": "false"}, relative_output="custom output", silo=False)

    # Continue a saved fixed-time hydro state and compare its physical particle
    # payload with the same number of uninterrupted steps. Duration and coarse
    # interval settings stay identical on both sides of the restart.
    _, initial_output, restart_options = results["hydroWave"]
    source_checkpoint = initial_output / "checkpoint.1"
    if not source_checkpoint.is_dir():
        raise RuntimeError("hydroWave did not save its first completed step")
    _, reference_output, _ = exercise("hydro_wave", "hydroWave", False, True, False,
                                     name="hydroWave_uninterrupted",
                                     overrides={"run.maxIterations": "2"})
    restart_directory = prepare("hydroWave_restart")
    restart_output = restart_directory / "output/hydroWave"
    restart_output.mkdir(parents=True)
    shutil.copytree(source_checkpoint, restart_output / "checkpoint.1")
    restart_options = dict(restart_options, **{"checkpoint.readIndex": "1"})
    resumed_log = run_process(command + [f"--{key}={value}" for key, value in restart_options.items()],
                             restart_directory)
    if "Done reading checkpoint" not in resumed_log:
        raise RuntimeError("fixed-time restart did not read its checkpoint")
    checkManifest(restart_output / "checkpoint.2", 2, scenario="hydroWave", endTime=1.e-5,
                  gravity=False)
    if (hydro_particle_payload(reference_output / "checkpoint.2", 512) !=
            hydro_particle_payload(restart_output / "checkpoint.2", 512)):
        raise RuntimeError("fixed-time restart changed physical particle or SPH predictor state")
    reference_clock = (reference_output / "progress.txt").read_text().splitlines()[-1].split()
    restarted_clock = (restart_output / "progress.txt").read_text().splitlines()[-1].split()
    if reference_clock != restarted_clock:
        raise RuntimeError("fixed-time restart changed the step, clock, or timestep rung")
    print("hydroWave: restarted particle state and physical clock match uninterrupted evolution")

    rejected_directory = prepare("cross_scenario_restart")
    rejected_output = rejected_directory / "output/hydroWave"
    rejected_output.mkdir(parents=True)
    shutil.copytree(source_checkpoint, rejected_output / "checkpoint.1")
    rejected_options = dict(restart_options, **{"run.scenario": "kelvinHelmholtz"})
    rejected_log = run_process(command + [f"--{key}={value}" for key, value in rejected_options.items()],
                              rejected_directory, success=False)
    if "scenario" not in rejected_log.lower() or "Done reading checkpoint" in rejected_log:
        raise RuntimeError("cross-scenario restart was not rejected before loading particle state")

    # Two coarse intervals put one interval on each side of the stellar
    # relaxation boundary. The tiny duration keeps the N8 fixture inexpensive
    # while exercising the damping/viscosity transition and its pending kicks.
    star_duration = 1.e-5
    _, star_reference, star_options = exercise(
        "polytropic_star", "polytropicStar", True, True, False,
        name="polytropicStar_stage_uninterrupted",
        overrides={"run.numberOfSteps": "2", "run.maxIterations": "64"})
    star_clock = [line.split() for line in
                  (star_reference / "progress.txt").read_text().splitlines() if line.strip()]
    if not math.isclose(float(star_clock[-1][1]), star_duration, rel_tol=1.e-9, abs_tol=1.e-14):
        raise RuntimeError("polytropicStar did not complete both evolution stages")
    boundaries = [index for index, row in enumerate(star_clock)
                  if math.isclose(float(row[1]), 0.5 * star_duration,
                                  rel_tol=1.e-9, abs_tol=1.e-14)]
    if len(boundaries) != 1 or boundaries[0] == len(star_clock) - 1:
        raise RuntimeError("polytropicStar did not cross its synchronized relaxation boundary")
    boundary_index = boundaries[0]
    boundary_iteration = int(star_clock[boundary_index][0])
    final_iteration = int(star_clock[-1][0])
    boundary_checkpoint = star_reference / f"checkpoint.{boundary_iteration}"
    if not boundary_checkpoint.is_dir():
        raise RuntimeError("polytropicStar did not checkpoint its relaxation boundary")
    star_restart_directory = prepare("polytropicStar_stage_restart")
    star_restart_output = star_restart_directory / "output/polytropicStar"
    star_restart_output.mkdir(parents=True)
    shutil.copytree(boundary_checkpoint,
                    star_restart_output / f"checkpoint.{boundary_iteration}")
    star_restart_options = dict(star_options,
                                **{"checkpoint.readIndex": str(boundary_iteration)})
    star_resumed_log = run_process(
        command + [f"--{key}={value}" for key, value in star_restart_options.items()],
        star_restart_directory)
    if "Done reading checkpoint" not in star_resumed_log:
        raise RuntimeError("polytropicStar did not resume its relaxation-boundary checkpoint")
    star_final_checkpoint = star_restart_output / f"checkpoint.{final_iteration}"
    checkManifest(star_final_checkpoint, final_iteration, scenario="polytropicStar",
                  endTime=star_duration, gravity=True)
    if (hydro_particle_payload(star_reference / f"checkpoint.{final_iteration}") !=
            hydro_particle_payload(star_final_checkpoint)):
        raise RuntimeError("polytropicStar stage-boundary restart changed physical or predictor state")
    star_restarted_clock = [line.split() for line in
                           (star_restart_output / "progress.txt").read_text().splitlines()
                           if line.strip()]
    if star_restarted_clock != star_clock[boundary_index + 1:]:
        raise RuntimeError("polytropicStar stage-boundary restart changed its physical clock")
    print("polytropicStar: stage-boundary restart matches uninterrupted state and clock")

    for name, option in (("unknown", "--run.scenario=unknown"),
                         ("abbreviated_rayleigh_taylor", "--run.scenario=rt"),
                         ("legacy_sod_test", "--run.test=sod")):
        directory = prepare("reject_" + name)
        output = run_process(command + ["--compute.cuda=false", option], directory, success=False)
        if "scenario" not in output.lower():
            raise RuntimeError(f"{name}: rejection did not explain the scenario selection")
    print("First-class scenario regressions passed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
