# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# SPDX-License-Identifier: BSL-1.0

"""Measure the Hopkins (2015) one-dimensional linear sound-wave convergence.

Run, for example:
    python3 tests/run_hopkins_sound_wave.py build/hopkins/cosmictiger /tmp/sound_wave
Reprocess saved diagnostics without launching the executable:
    python3 tests/run_hopkins_sound_wave.py --reprocess /tmp/sound_wave

The particle number changes while the configured CFL criterion chooses timesteps.
These are combined space/time refinement measurements, not temporal-only tests.
No particular convergence order is asserted. All particles enter every norm.
"""

import argparse
import configparser
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time

from application_selection import cpu_backend_options

from run_scenarios import setting


AMPLITUDE = 1.0e-6
REFERENCE_DEFINITION = (
    "delta=1e-6*sin(2*pi*(x-time)); density=1+delta; velocity=delta; pressure=0.6+delta. "
    "CSV exact_* columns are checked within 32*epsilon*max(abs(reference),1e-6); "
    "norms use the independently recomputed references."
)
DEFAULT_RESOLUTIONS = (16, 32, 64, 128, 256, 512, 1024, 2048)
REQUIRED_COLUMNS = (
    "id", "x", "density", "exact_density", "velocity", "exact_velocity",
    "pressure", "exact_pressure", "support", "mass", "energy", "rung",
)
NORM_NAMES = tuple(f"{field}_{norm}" for field in ("density", "velocity", "pressure")
                   for norm in ("l1", "l2")) + (
    "density_l1_over_amplitude", "density_l2_over_amplitude",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_process(command, directory, timeout):
    """Use the scenario runner's process-group cleanup with a configurable limit."""
    log_path = directory / "run.log"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1")
    with log_path.open("w") as log:
        with subprocess.Popen(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT,
                              text=True, env=environment, start_new_session=True) as process:
            try:
                return_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError(f"sound-wave simulation timed out; see {log_path}")
    output = log_path.read_text(errors="replace")
    if return_code != 0:
        raise RuntimeError(f"exit {return_code}; see {log_path}\n{output[-6000:]}")
    return output


def analytic_reference(position, time):
    perturbation = AMPLITUDE * math.sin(2.0 * math.pi * (position - time))
    return {"exact_density": 1.0 + perturbation, "exact_velocity": perturbation,
            "exact_pressure": 0.6 + perturbation}


def resolved_settings(output):
    # Horizontal whitespace only: an empty option must not consume the next line.
    return {key: value.strip() for key, value in re.findall(
        r"^([A-Za-z]+\.[A-Za-z][A-Za-z0-9]*)[ \t]*:[ \t]*([^\r\n]*)$",
        output, re.MULTILINE)}


def read_snapshot(path, count, expected_time):
    particles = {}
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not set(REQUIRED_COLUMNS).issubset(reader.fieldnames):
            raise RuntimeError(f"missing sound-wave diagnostic columns in {path}")
        for row in reader:
            identifier = int(row["id"])
            rung = int(row["rung"])
            if identifier < 0 or identifier in particles or not 0 <= rung < 16:
                raise RuntimeError(f"invalid or repeated particle identifier/rung in {path}")
            values = {name: float(row[name]) for name in REQUIRED_COLUMNS
                      if name not in ("id", "rung")}
            if not all(math.isfinite(value) for value in values.values()):
                raise RuntimeError(f"nonfinite diagnostic in {path}, particle {identifier}")
            if not 0.0 <= values["x"] < 1.0:
                raise RuntimeError(f"unwrapped particle coordinate in {path}")
            if any(values[name] <= 0.0 for name in (
                    "density", "exact_density", "pressure", "exact_pressure", "support",
                    "mass", "energy")):
                raise RuntimeError(f"nonpositive fluid diagnostic in {path}")
            reference_time = expected_time
            if "time" in row:
                actual_time = float(row["time"])
                if not math.isfinite(actual_time) or not math.isclose(
                        actual_time, expected_time, rel_tol=2.0e-12, abs_tol=1.0e-14):
                    raise RuntimeError(f"incorrect snapshot time in {path}: {actual_time}")
                reference_time = actual_time
            for name, reference in analytic_reference(values["x"], reference_time).items():
                tolerance = 32.0 * sys.float_info.epsilon * max(abs(reference), AMPLITUDE)
                if abs(values[name] - reference) > tolerance:
                    raise RuntimeError(f"incorrect analytic {name} in {path}, particle "
                                       f"{identifier}: CSV={values[name]}, expected={reference}")
                values[name] = reference
            particles[identifier] = dict(values, rung=rung)
    if len(particles) != count:
        raise RuntimeError(f"{path} has {len(particles)} particles; expected {count}")
    return particles


def error_norms(particles):
    result = {}
    for field in ("density", "velocity", "pressure"):
        errors = [particle[field] - particle[f"exact_{field}"]
                  for particle in particles.values()]
        result[f"{field}_l1"] = math.fsum(abs(error) for error in errors) / len(errors)
        result[f"{field}_l2"] = math.sqrt(math.fsum(error * error for error in errors) /
                                        len(errors))
    for norm in ("l1", "l2"):
        result[f"density_{norm}_over_amplitude"] = result[f"density_{norm}"] / AMPLITUDE
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("nonfinite sound-wave error norm")
    return result


def conserved_totals(particles):
    return {
        "mass": math.fsum(particle["mass"] for particle in particles.values()),
        "momentum": math.fsum(particle["mass"] * particle["velocity"]
                              for particle in particles.values()),
        # The CSV energy is the conserved total energy of one particle.
        "energy": math.fsum(particle["energy"] for particle in particles.values()),
    }


def validate_run(output, destination, method, count, duration):
    expected = {"run.scenario": "soundWave", "hydro.method": method,
                "initial.particlesPerDimension": str(count)}
    for key, value in expected.items():
        if setting(output, key) != value:
            raise RuntimeError(f"incorrect resolved option {key}")
    for key in ("gravity.cuda", "gravity.enabled", "chemistry.enabled", "output.silo"):
        if setting(output, key).lower() not in ("false", "f", "0"):
            raise RuntimeError(f"{key} was enabled in the sound-wave measurement")
    reported_duration = float(setting(output, "run.endTime"))
    coarse_steps = int(setting(output, "run.numberOfSteps"))
    if (not math.isfinite(duration) or duration <= 0.0 or coarse_steps <= 0 or
            not math.isclose(duration, reported_duration, rel_tol=6.0e-7)):
        raise RuntimeError("sound-wave configuration must specify a positive end time and steps")
    initialized = re.findall(r"^Initialized particles\s*:\s*(\d+)\s*$", output, re.MULTILINE)
    if not initialized or int(initialized[-1]) != count:
        raise RuntimeError("initialized particle count differs from the requested 1D resolution")
    finished = re.findall(r"soundWave: finished at t=(\S+) after (\d+) steps", output)
    if not finished:
        raise RuntimeError("sound wave did not report completion")
    events = [dict(re.findall(r"([A-Za-z]+)=([^\s]+)", value))
              for value in re.findall(r"Hopkins step: ([^\n]+)", output)]
    outgoing_times = [float(event["time"]) for event in events if int(event["terminal"]) == 0]
    if not outgoing_times or any(value >= duration for value in outgoing_times):
        raise RuntimeError("sound wave omitted activity records or opened a step at/beyond its terminal time")
    rows = [line.split() for line in (destination / "progress.txt").read_text().splitlines()
            if line.strip()]
    if not rows:
        raise RuntimeError("sound-wave progress is empty")
    timesteps = []
    elapsed = 0.0
    for iteration, row in enumerate(rows, 1):
        if len(row) != 5 or int(row[0]) != iteration:
            raise RuntimeError("invalid or nonconsecutive sound-wave progress")
        current_time, recorded_step = map(float, row[1:3])
        minimum_rung, maximum_rung = map(int, row[3:5])
        if (not math.isfinite(current_time) or not math.isfinite(recorded_step) or
                not 0 <= minimum_rung < 16 or not 0 <= maximum_rung < 16):
            raise RuntimeError("nonfinite time or invalid rung in sound-wave progress")
        timestep = math.ldexp(duration / coarse_steps, -maximum_rung)
        timesteps.append(timestep)
        elapsed += timestep
        # progress.txt prints six significant digits; reconstruct the full clock
        # independently from its integer rungs and the resolved coarse timestep.
        if (not math.isclose(recorded_step, timestep, rel_tol=6.0e-7) or
                not math.isclose(current_time, elapsed, rel_tol=6.0e-7)):
            raise RuntimeError("sound-wave progress time differs from its rung clock")
    final_time = math.fsum(timesteps)
    if (not math.isclose(final_time, duration, rel_tol=2.0e-12) or
            not math.isclose(float(finished[-1][0]), duration, rel_tol=6.0e-7) or
            int(finished[-1][1]) != len(rows)):
        raise RuntimeError(f"incomplete sound-wave evolution: t={final_time}, target={duration}")
    return {"time": final_time, "iterations": len(rows), "minimum_timestep": min(timesteps),
            "maximum_timestep": max(timesteps), "resolved_settings": resolved_settings(output)}


def analyze_case(result, duration, output=None):
    directory = Path(result["directory"])
    destination = directory / "output"
    if output is None:
        output = (directory / "run.log").read_text(errors="replace")
    method, number = result["method"], result["resolution"]
    analysis = validate_run(output, destination, method, number, duration)
    initial = read_snapshot(destination / "sound_wave_initial.csv", number, 0.0)
    final = read_snapshot(destination / "sound_wave_final.csv", number, analysis["time"])
    if initial.keys() != final.keys():
        raise RuntimeError("particle identifiers changed during the sound-wave evolution")
    initial_totals, final_totals = conserved_totals(initial), conserved_totals(final)
    analysis.update(initial_norms=error_norms(initial), final_norms=error_norms(final),
                    initial_conserved=initial_totals, final_conserved=final_totals,
                    analytic_reference_validated=True,
                    conservation_changes={
                        "relative_mass_change": (final_totals["mass"] - initial_totals["mass"]) /
                                                initial_totals["mass"],
                        "momentum_change": final_totals["momentum"] - initial_totals["momentum"],
                        "relative_energy_change": (final_totals["energy"] - initial_totals["energy"]) /
                                                  initial_totals["energy"],
                    })
    return analysis


def update_orders(results):
    previous = {}
    for result in results:
        if result["status"] != "completed":
            continue
        method = result["method"]
        older = previous.get(method)
        result["orders"] = {}
        if older is not None:
            ratio = result["resolution"] / older["resolution"]
            result["reference_resolution"] = older["resolution"]
            result["refinement_ratio"] = ratio
            for name in NORM_NAMES:
                old_error = older["final_norms"][name]
                new_error = result["final_norms"][name]
                result["orders"][name] = ((math.log(old_error) - math.log(new_error)) / math.log(ratio)
                                           if old_error > 0.0 and new_error > 0.0 else None)
        previous[method] = result


def write_summary(base, report):
    update_orders(report["results"])
    (base / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    fields = ["method", "resolution", "status", "time", "iterations", "seconds",
              "minimum_timestep", "maximum_timestep", "reference_resolution",
              "refinement_ratio", "relative_mass_change", "momentum_change",
              "relative_energy_change", "error"]
    fields += [f"{stage}_{name}" for stage in ("initial", "final", "order")
               for name in NORM_NAMES]
    with (base / "summary.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for result in report["results"]:
            row = {name: result[name] for name in fields if name in result}
            row.update(result.get("conservation_changes", {}))
            for stage, source in (("initial", "initial_norms"), ("final", "final_norms"),
                                  ("order", "orders")):
                row.update({f"{stage}_{name}": value
                            for name, value in result.get(source, {}).items()})
            writer.writerow(row)


def reprocess_campaign(base):
    summary = base / "summary.json"
    source_hash = sha256(summary)
    report = json.loads(summary.read_text())
    for name in ("summary.json", "summary.csv"):
        original = base / name.replace(".", ".original.", 1)
        if not original.exists():
            shutil.copy2(base / name, original)
    completed = 0
    for result in report["results"]:
        directory = Path(result["directory"])
        output = (directory / "run.log").read_text(errors="replace")
        # Failed simulations keep their original status, error, and invocation.
        result["resolved_settings"] = resolved_settings(output)
        if result["status"] != "completed":
            continue
        duration = float(result.get("overrides", {}).get(
            "run.endTime", report["configuration_settings"]["run.endTime"]))
        result.update(analyze_case(result, duration, output))
        completed += 1
    report["analytic_reference_definition"] = REFERENCE_DEFINITION
    event = {"completed_utc": datetime.now(timezone.utc).isoformat(),
             "postprocessor": str(Path(__file__).resolve()),
             "postprocessor_sha256": sha256(Path(__file__)),
             "input_summary_sha256": source_hash,
             "original_runner_sha256": report["runner_sha256"],
             "reprocessed_completed_cases": completed,
             "preserved_incomplete_cases": len(report["results"]) - completed,
             "simulations_executed": 0}
    report.setdefault("postprocessing_history", []).append(event)
    write_summary(base, report)
    print(f"Reprocessed {completed} completed cases in {base}; no simulations executed.")
    return 0


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("executable", type=Path, nargs="?")
    parser.add_argument("workdir", type=Path, nargs="?", help="fresh campaign directory")
    parser.add_argument("--reprocess", type=Path,
                        help="recompute saved campaign summaries without running the executable")
    parser.add_argument("--configuration", type=Path,
                        default=root / "configurations/sound_wave.ini")
    parser.add_argument("--resolutions", nargs="+", type=int, default=DEFAULT_RESOLUTIONS)
    parser.add_argument("--methods", nargs="+", choices=("mfm", "mfv"), default=("mfm", "mfv"))
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds per simulation")
    parser.add_argument("--keep-going", action="store_true", help="measure other cases after a failure")
    args = parser.parse_args()
    if args.reprocess is not None:
        if args.executable is not None or args.workdir is not None:
            parser.error("--reprocess takes a saved campaign instead of executable/workdir")
        return reprocess_campaign(args.reprocess.resolve(strict=True))
    if args.executable is None or args.workdir is None:
        parser.error("executable and workdir are required when launching simulations")
    if (any(number < 16 for number in args.resolutions) or
            list(args.resolutions) != sorted(set(args.resolutions))):
        parser.error("resolutions must be distinct increasing integers of at least 16")
    if len(args.methods) != len(set(args.methods)):
        parser.error("methods must be distinct")
    if not math.isfinite(args.timeout) or args.timeout <= 0.0:
        parser.error("timeout must be positive and finite")
    executable = args.executable.resolve(strict=True)
    configuration = args.configuration.resolve(strict=True)
    base = args.workdir.resolve()
    cases = [(method, number, base / f"{method}_n{number}")
             for method in args.methods for number in args.resolutions]
    if (base / "summary.json").exists() or (base / "summary.csv").exists() or any(
            directory.exists() for _, _, directory in cases):
        parser.error("campaign outputs already exist; choose a fresh workdir")
    base.mkdir(parents=True, exist_ok=True)
    config = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#", ";"))
    config.optionxform = str
    config.read(configuration)
    duration = config.getfloat("run", "endTime")
    if not math.isfinite(duration) or duration <= 0.0:
        parser.error("configuration must specify a positive, finite run.endTime")
    report = {
        "schema_version": 1,
        "experiment": "Hopkins (2015) linear sound wave, combined spatial/CFL refinement",
        "reference": "https://doi.org/10.1093/mnras/stv195",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "executable": str(executable), "executable_sha256": sha256(executable),
        "configuration": str(configuration), "configuration_sha256": sha256(configuration),
        "configuration_settings": {f"{section}.{key}": value for section in config.sections()
                                   for key, value in config.items(section)},
        "runner_sha256": sha256(Path(__file__)), "python": sys.version,
        "platform": platform.platform(), "density_perturbation_amplitude": AMPLITUDE,
        "resolutions": list(args.resolutions), "methods": list(args.methods),
        "threads_per_process": args.threads, "processes_at_once": 1,
        "timeout_seconds": args.timeout,
        "norm_definition": "unweighted mean absolute error and root mean square over all N particles",
        "analytic_reference_definition": REFERENCE_DEFINITION,
        "order_definition": "log(error_coarse/error_fine)/log(N_fine/N_coarse); null for zero error",
        "environment": {key: os.environ[key] for key in ("LD_LIBRARY_PATH", "HWLOC_COMPONENTS")
                        if key in os.environ},
        "process_environment_overrides": {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                                          "OPENBLAS_NUM_THREADS": "1"},
        "status": "running", "results": [],
    }
    write_summary(base, report)
    for method, number, directory in cases:
        directory.mkdir()
        destination = directory / "output"
        settings = {"configuration": str(configuration), "hydro.method": method,
                    "initial.particlesPerDimension": number, "output.directory": str(destination),
                    **cpu_backend_options(), "compute.positionType": "fixed64",
                    "run.verbose": "true", "output.silo": "false",
                    "run.timeIntegrator": "traditional", "gravity.method.far": "fmm",
                    "gravity.expansionDegree": 6, "starFormation.enabled": "false",
                    "analysis.views": "false", "analysis.slice": "false", "analysis.power": "false",
                    "analysis.groups": "false", "analysis.tracers": "false", "lightcone.enabled": "false",
                    "checkpoint.readIndex": -1, "checkpoint.intervalSeconds": 1000000000}
        command = [str(executable), f"--hpx:threads={args.threads}", "--hpx:bind=none"]
        command += [f"--{key}={value}" for key, value in settings.items()]
        result = {"method": method, "resolution": number, "directory": str(directory),
                  "command": command, "overrides": settings, "status": "running"}
        report["results"].append(result)
        write_summary(base, report)
        print(f"Running {method} N={number}: {directory}", flush=True)
        started = time.monotonic()
        try:
            if (sha256(executable) != report["executable_sha256"] or
                    sha256(configuration) != report["configuration_sha256"]):
                raise RuntimeError("executable or configuration changed during the campaign")
            output = run_process(command, directory, args.timeout)
            result.update(analyze_case(result, duration, output), status="completed")
            print(f"  density L1={result['final_norms']['density_l1']:.9e}, "
                  f"L2={result['final_norms']['density_l2']:.9e}", flush=True)
        except (OSError, ValueError, ArithmeticError, RuntimeError) as error:
            result.update(status="failed", error=str(error))
            print(f"  FAILED: {error}", file=sys.stderr, flush=True)
        finally:
            result["seconds"] = time.monotonic() - started
            write_summary(base, report)
        if result["status"] == "failed" and not args.keep_going:
            break
    report["status"] = ("completed" if len(report["results"]) == len(cases) and
                        all(result["status"] == "completed" for result in report["results"])
                        else "failed")
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_summary(base, report)
    print(f"Results: {base / 'summary.json'} and {base / 'summary.csv'}", flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
