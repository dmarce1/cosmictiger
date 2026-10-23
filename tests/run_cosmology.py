# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# SPDX-License-Identifier: BSL-1.0

"""Exercise cosmological turnaround, signed restart state, and stopping clocks."""

import argparse
import math
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import sys

from run_checkpoint import checkManifest


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run_simulation(command, directory, success=True):
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
                raise RuntimeError(f"cosmology run timed out; see {log_path}")
    output = log_path.read_text(errors="replace")
    require((result == 0) == success,
            f"unexpected exit {result}; see {log_path}\n{output[-6000:]}")
    return output


def checkpoint_state(path, end_time):
    """Read the fixed header and leading background state of a dark-matter checkpoint."""
    iteration = int(path.name.split(".")[1])
    checkManifest(path, iteration, scenario="cosmology", endTime=end_time, gravity=True)
    payload = (path / "checkpoint.0.dat").read_bytes()
    header_format = struct.Struct("=QII11dQiI")
    require(len(payload) >= header_format.size + 40, f"truncated driver state in {path}")
    header = header_format.unpack_from(payload)
    magic, version, flags = header[:3]
    require(magic == 0x4354535048303031 and version == 3,
            f"unsupported checkpoint header in {path}")
    require(flags & (1 | (1 << 11)) == 0,
            f"expected a dark-matter checkpoint without chemistry in {path}")
    # These are the leading fields of DriverParameters. Optional hydro and
    # chemistry blocks are absent in this explicitly dark-matter-only fixture.
    scale, derivative, _, time, end = struct.unpack_from("=5d", payload, header_format.size)
    gravitational_mass, omega_matter, omega_radiation, omega_lambda = header[7:11]
    particles = header[14]
    hubble = math.sqrt(gravitational_mass * 8.0 * math.pi * particles / (3.0 * omega_matter))
    require(all(math.isfinite(value) for value in (scale, derivative, time, end, hubble)),
            f"nonfinite cosmological state in {path}")
    require(scale > 0.0 and hubble > 0.0 and time > 0.0 and end > 0.0,
            f"nonpositive background scale or clock in {path}")
    return {"path": path, "iteration": iteration, "a": scale, "adot": derivative,
            "tau": time, "end": end, "hubble": hubble, "omega_m": omega_matter,
            "omega_r": omega_radiation, "omega_lambda": omega_lambda}


def checkpoint_states(directory, end_time):
    paths = [path for path in directory.glob("checkpoint.*")
             if path.is_dir() and re.fullmatch(r"checkpoint\.[0-9]+", path.name)]
    require(paths, f"no published checkpoints in {directory}")
    paths.sort(key=lambda path: int(path.name.split(".")[1]))
    return [checkpoint_state(path, end_time) for path in paths]


def require_close(actual, expected, name, relative=2.e-9, absolute=2.e-11):
    require(math.isclose(actual, expected, rel_tol=relative, abs_tol=absolute),
            f"{name}: {actual:.17g}, expected {expected:.17g}")


def check_completion(state, requested_duration):
    require_close(state["end"], requested_duration, "stored conformal endpoint",
                  relative=2.e-13, absolute=0.0)
    require_close(state["tau"], requested_duration, "completed conformal time",
                  relative=2.e-12, absolute=0.0)


def check_finite_progress(directory):
    path = directory / "progress.txt"
    require(path.is_file(), f"missing progress file in {directory}")
    rows = [[float(value) for value in line.split()] for line in path.read_text().splitlines()
            if line.strip() and line.split()[0] != "runtime"]
    require(rows, f"empty progress file in {directory}")
    require(all(len(row) == 14 for row in rows), f"incomplete progress output in {directory}")
    require(all(math.isfinite(value) for row in rows for value in row),
            f"nonfinite progress output in {directory}")
    return rows


def closed_matter_state(dimensionless_time):
    # Omega_m=2, Omega_k=-1 has a=1-cos(theta), H0*eta=theta.
    # The initial a=1 expanding state is theta=pi/2.
    phase = 0.5 * math.pi + dimensionless_time
    scale = 1.0 - math.cos(phase)
    return scale, math.sin(phase) / scale


def check_closed_time_table(directory, steps):
    path = directory / "time.txt"
    require(path.is_file(), f"missing background time table in {directory}")
    lines = path.read_text().splitlines()
    require(lines and "scale" in lines[0] and "redshift" in lines[0],
            f"invalid background time table header in {path}")
    rows = [[float(value) for value in line.split()] for line in lines[1:] if line.strip()]
    require(len(rows) == steps + 1 and all(len(row) == 5 for row in rows),
            f"background time table does not contain {steps + 1} complete samples")
    require(all(math.isfinite(value) for row in rows for value in row),
            "background time table contains nonfinite values")
    require(all(second[1] > first[1] and second[2] > first[2]
                for first, second in zip(rows, rows[1:])),
            "physical age or conformal time stopped increasing during contraction")
    age_span = rows[-1][1] - rows[0][1]
    conformal_span = rows[-1][2] - rows[0][2]
    # The common conversion into years cancels: the cycloid has
    # H0*Delta(t)=pi+2 and H0*Delta(eta)=pi over this complete trajectory.
    require_close(age_span / conformal_span, (math.pi + 2.0) / math.pi,
                  "physical-to-conformal elapsed-time ratio")
    initial_phase = math.pi / 2.0
    initial_age = initial_phase - math.sin(initial_phase)
    for index, (step, age, conformal, scale, redshift) in enumerate(rows):
        require(step == index, "background time table has an incorrect step index")
        fraction = index / steps
        phase = initial_phase + math.pi * fraction
        expected_scale, _ = closed_matter_state(math.pi * fraction)
        require_close(scale, expected_scale, f"time-table scale at step {index}")
        require_close(redshift, 1.0 / expected_scale - 1.0,
                      f"time-table redshift at step {index}")
        require_close((conformal - rows[0][2]) / conformal_span, fraction,
                      f"time-table conformal clock at step {index}")
        expected_fraction = (phase - math.sin(phase) - initial_age) / (math.pi + 2.0)
        require_close((age - rows[0][1]) / age_span, expected_fraction,
                      f"time-table physical age at step {index}")


def negative_lambda_interval(intervals):
    # Omega_m=2, Omega_Lambda=-1 has a=(sqrt(2)*sin(theta))^(2/3)
    # and d(H0*t)/dtheta=2/3. Integrate d(H0*eta)=d(H0*t)/a
    # from the expanding a=1 to the contracting a=1. The interval stays
    # away from either singularity and needs no turnaround root handling.
    lower = math.pi / 4.0
    width = (math.pi / 2.0) / intervals
    total = 0.0
    for index in range(intervals + 1):
        phase = lower + index * width
        scale = (math.sqrt(2.0) * math.sin(phase)) ** (2.0 / 3.0)
        weight = 1 if index in (0, intervals) else 4 if index % 2 else 2
        total += weight * (2.0 / 3.0) / scale
    return total * width / 3.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    args = parser.parse_args()
    executable = args.executable.resolve()
    base = args.workdir.resolve()
    base.mkdir(parents=True, exist_ok=True)
    command = [str(executable), "--hpx:threads=2", "--hpx:bind=none"]
    options = {
        "run.scenario": "cosmology",
        "run.verbose": "false",
        "run.numberOfSteps": "32",
        "run.maxIterations": "10000",
        "initial.particlesPerDimension": "8",
        "initial.closePack": "false",
        "initial.twoLpt": "false",
        "compute.cuda": "false",
        "gravity.enabled": "true",
        "hydro.enabled": "false",
        "chemistry.enabled": "false",
        "analysis.views": "false",
        "analysis.slice": "false",
        "analysis.tracers": "false",
        "analysis.power": "false",
        "analysis.groups": "false",
        "lightcone.enabled": "false",
        "output.silo": "false",
        "output.directory": ".",
        "checkpoint.intervalSeconds": "0",
        "cosmology.startRedshift": "0",
        "cosmology.endRedshift": "0",
        "cosmology.omegaB": "0.05",
        "cosmology.omegaC": "1.95",
        "cosmology.omegaLambda": "0",
        "cosmology.hubble": "0.7",
        # Nonzero temperature satisfies the physical option contract, while
        # the radiation correction is far below the analytic-test tolerance.
        "cosmology.thetaCmb": "0.0001",
        "cosmology.nEff": "0",
        "cosmology.sigma8": "0",
    }

    def prepare(name):
        directory = base / name
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir()
        return directory

    def exercise(name, overrides, success=True, checkpoint=None):
        directory = prepare(name)
        if checkpoint is not None:
            shutil.copytree(checkpoint, directory / checkpoint.name)
        resolved = dict(options, **overrides)
        output = run_simulation(command + [f"--{key}={value}" for key, value in resolved.items()],
                                directory, success=success)
        return directory, output

    # The saved physical mass and particle count determine H0 in code units.
    # Calibrate from a tiny actual run instead of duplicating unit constants.
    calibration, _ = exercise("calibration", {"run.endTime": "1", "run.maxIterations": "0"})
    initial = checkpoint_states(calibration, 1.0)[-1]
    require(initial["omega_m"] == 2.0 and initial["omega_r"] < 1.e-16,
            "analytic fixture has unexpected matter or radiation density")
    hubble = initial["hubble"]
    duration = math.pi / hubble

    closed, output = exercise("closed_turnaround", {"run.endTime": repr(duration)})
    require("Finished:" in output, "closed model stopped before its requested endpoint")
    states = checkpoint_states(closed, duration)
    require(any(state["adot"] > 0.0 for state in states) and
            any(state["adot"] < 0.0 for state in states),
            "closed model did not cross from expansion into contraction")
    require(max(state["a"] for state in states) > 1.99,
            "closed model did not reach its expected maximum scale factor")
    for state in states:
        scale, rate = closed_matter_state(hubble * state["tau"])
        require_close(state["a"], scale, "closed cycloid scale factor")
        require_close(state["adot"] / hubble, rate, "closed cycloid signed rate")
    check_completion(states[-1], duration)
    check_finite_progress(closed)
    check_closed_time_table(closed, 32)
    print("closed matter: expansion, turnaround, contraction, both time coordinates, and endpoint passed")

    dimensionless_interval = negative_lambda_interval(8192)
    require_close(dimensionless_interval, negative_lambda_interval(4096),
                  "independent negative-Lambda quadrature", relative=2.e-13, absolute=0.0)
    duration = dimensionless_interval / hubble
    negative, output = exercise("negative_lambda", {"run.endTime": repr(duration),
                                                    "cosmology.omegaLambda": "-1"})
    require("Finished:" in output, "negative-Lambda model stopped before its requested endpoint")
    states = checkpoint_states(negative, duration)
    final = states[-1]
    require(final["omega_lambda"] == -1.0, "numeric negative Lambda was treated as a sentinel")
    require_close(final["a"], 1.0, "negative-Lambda final scale")
    require_close(final["adot"] / hubble, -1.0, "negative-Lambda final signed rate")
    check_completion(final, duration)
    check_finite_progress(negative)

    candidates = [state for state in states[:-1] if state["adot"] < -0.1 * hubble]
    require(candidates, "no intermediate contracting checkpoint was published")
    saved = candidates[0]
    restarted, output = exercise("contracting_restart", {
        "run.endTime": repr(duration), "cosmology.omegaLambda": "-1",
        "checkpoint.readIndex": str(saved["iteration"]),
    }, checkpoint=saved["path"])
    require("Done reading checkpoint" in output, "contracting restart did not read its checkpoint")
    resumed = checkpoint_states(restarted, duration)[-1]
    require(resumed["iteration"] == final["iteration"], "restart changed the completed step count")
    for name in ("a", "adot", "tau", "end"):
        require_close(resumed[name], final[name], f"restarted {name}",
                      relative=2.e-12, absolute=2.e-14 * (hubble if name == "adot" else 1.0))
    require(resumed["adot"] < 0.0, "checkpoint restart lost the contraction branch")
    check_finite_progress(restarted)
    print("negative Lambda: literal input, recollapse, endpoint, and contracting restart passed")

    # Resuming an already completed endpoint must not execute another kick or
    # drift, advance the clock, or publish a new checkpoint iteration.
    completed, output = exercise("completed_restart", {
        "run.endTime": repr(duration), "cosmology.omegaLambda": "-1",
        "checkpoint.readIndex": str(final["iteration"]),
    }, checkpoint=final["path"])
    require("Done reading checkpoint" in output, "completed restart did not read its checkpoint")
    completed_states = checkpoint_states(completed, duration)
    require(len(completed_states) == 1 and
            completed_states[0]["iteration"] == final["iteration"],
            "completed restart advanced another timestep")
    require((completed_states[0]["path"] / "checkpoint.0.dat").read_bytes() ==
            (final["path"] / "checkpoint.0.dat").read_bytes(),
            "completed restart changed the saved endpoint state")
    progress = completed / "progress.txt"
    require(not progress.exists() or not progress.read_text().strip(),
            "completed restart produced an extra evolution row")
    print("completed restart: endpoint state and iteration remain unchanged")

    # An iteration limit can stop after the last fine step of a coarse
    # interval, before the outer driver's step counter increments. Exercise
    # that checkpoint rather than only checkpoints written during evolution.
    boundaries = [state for state in states[:-1]
                  if 1.0 <= 32.0 * state["tau"] / duration < 32.0 and
                  math.isclose(32.0 * state["tau"] / duration,
                               round(32.0 * state["tau"] / duration), abs_tol=1.e-10)]
    require(boundaries, "no intermediate coarse boundary is available for restart testing")
    boundary = boundaries[0]
    limited, output = exercise("coarse_boundary_stop", {
        "run.endTime": repr(duration), "cosmology.omegaLambda": "-1",
        "run.maxIterations": str(boundary["iteration"] - 1),
    })
    require("iteration limit" in output, "coarse-boundary run did not stop at its iteration limit")
    limited_state = checkpoint_states(limited, duration)[-1]
    require(limited_state["iteration"] == boundary["iteration"],
            "iteration limit stopped at the wrong checkpoint")
    require_close(limited_state["tau"], boundary["tau"], "coarse-boundary stop clock")
    continued, output = exercise("coarse_boundary_restart", {
        "run.endTime": repr(duration), "cosmology.omegaLambda": "-1",
        "checkpoint.readIndex": str(limited_state["iteration"]),
    }, checkpoint=limited_state["path"])
    require("Done reading checkpoint" in output, "coarse-boundary restart did not read its checkpoint")
    continued_state = checkpoint_states(continued, duration)[-1]
    require(continued_state["iteration"] == final["iteration"],
            "coarse-boundary restart changed the completed step count")
    for name in ("a", "adot", "tau", "end"):
        require_close(continued_state[name], final[name], f"coarse-boundary restarted {name}",
                      relative=2.e-12, absolute=2.e-14 * (hubble if name == "adot" else 1.0))
    check_finite_progress(continued)
    print("coarse-boundary restart: clock, branch, and final step count match uninterrupted evolution")

    _, mismatch = exercise("changed_initial_redshift", {
        "run.endTime": repr(duration), "cosmology.omegaLambda": "-1",
        "checkpoint.readIndex": str(saved["iteration"]), "cosmology.startRedshift": "1",
    }, success=False, checkpoint=saved["path"])
    require("checkpoint" in mismatch.lower() and
            re.search(r"cosmolog|background|startRedshift|initial scale", mismatch, re.IGNORECASE),
            "changed initial redshift failed without a checkpoint/background diagnostic")
    print("restart with changed background initial redshift: rejected")

    # Enabling gas selects the ordinary gravity/SPH advancement path. Its
    # optional checkpoint blocks differ from the dark-matter fixture, so use
    # the reported endpoint state and clock to verify this independent lane.
    sph, output = exercise("sph_negative_lambda", {
        "run.endTime": repr(duration), "cosmology.omegaLambda": "-1",
        "hydro.enabled": "true",
        "chemistry.enabled": "false", "cosmology.sigma8": "0",
        "checkpoint.intervalSeconds": "1000000000", "output.silo": "false",
    })
    require("Finished:" in output, "SPH negative-Lambda model stopped before its requested endpoint")
    rows = check_finite_progress(sph)
    previous_time = 0.0
    for row in rows:
        require(previous_time < row[5] <= 32.0,
                "SPH conformal clock stalled, moved backward, or passed its endpoint")
        previous_time = row[5]
    require(any(row[4] > 0.0 for row in rows) and any(row[4] < 0.0 for row in rows),
            "SPH model did not cross from expansion into contraction")
    # progress.txt prints three digits after the decimal in scientific
    # notation; keep the comparison tolerance consistent with that output.
    require_close(rows[-1][3], 1.0, "SPH final scale", relative=1.e-3, absolute=1.e-3)
    require_close(rows[-1][4] / hubble, -1.0, "SPH final signed rate",
                  relative=1.e-3, absolute=1.e-3)
    require_close(rows[-1][5], 32.0, "SPH completed conformal clock",
                  relative=0.0, absolute=1.e-3)
    print("ordinary gravity/SPH: expansion, turnaround, contraction, and endpoint passed")

    endpoint, output = exercise("redshift_endpoint", {"run.endTime": "0",
                                                      "cosmology.endRedshift": "-0.2"})
    require("Finished:" in output, "redshift endpoint did not complete")
    final = checkpoint_states(endpoint, 0.0)[-1]
    require_close(final["a"], 1.25, "requested redshift scale factor")
    check_completion(final, math.asin(0.25) / hubble)
    check_finite_progress(endpoint)
    print("default stopping: requested future redshift determines the endpoint")

    _, rejected = exercise("unreachable_endpoint", {"run.endTime": "0",
                                                    "cosmology.endRedshift": "-0.75"},
                           success=False)
    require(re.search(r"turnaround|Friedmann|scale.factor|expanding|endpoint", rejected,
                      re.IGNORECASE), "unreachable endpoint failed without a cosmology diagnostic")
    print("unreachable expanding endpoint: rejected before evolution")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
