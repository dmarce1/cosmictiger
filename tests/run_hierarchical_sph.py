"""Run one bounded hierarchical SPH case on CPU by default, with opt-in comparisons.

The default starts one application process. Reference, CFL refinement and restart
checks are separate opt-ins, all sequential. No executable discovery/help probes
or MPI/CUDA runs are hidden in this runner. GPU execution requires --cuda.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time

from checkpoint_format import decode_checkpoint


ROOT = Path(__file__).resolve().parents[1]
CONFIGURATIONS = {"hydroWave": "hydro_wave", "sod": "sod", "blast": "blast"}
COUNTER_FIELDS = ("minimum_rung", "active", "sorted", "density", "force",
                  "source_candidates", "source_predictions")


def table(path, columns):
    rows = [[float(value) for value in line.split()] for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if not rows or any(len(row) != columns or not all(map(math.isfinite, row)) for row in rows):
        raise RuntimeError(f"{path}: empty, malformed or nonfinite diagnostics")
    return rows


def counters(output):
    result = []
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("SPH hierarchy:"):
            continue
        values = dict(re.findall(r"(\w+)=([^\s]+)", line))
        if set(values) != {"time", *COUNTER_FIELDS}:
            raise RuntimeError("hierarchical work counter fields changed")
        row = {key: int(values[key]) for key in COUNTER_FIELDS}
        row["time"] = float(values["time"])
        if not math.isfinite(row["time"]) or min(row[key] for key in COUNTER_FIELDS) < 0:
            raise RuntimeError("invalid hierarchical work counter")
        result.append(row)
    return result


def final_checkpoint(directory):
    paths = [(int(path.name.split(".")[1]), path) for path in directory.glob("checkpoint.*")
             if re.fullmatch(r"checkpoint\.[0-9]+", path.name) and path.is_dir()]
    if not paths:
        raise RuntimeError(f"{directory}: no final checkpoint")
    return max(paths)[1]


def snapshot(directory):
    checkpoint = final_checkpoint(directory)
    state = decode_checkpoint((checkpoint / "checkpoint.0.dat").read_bytes())
    if state.gas_count != state.count or not state.gas_count:
        raise RuntimeError("this fixture must contain gas only")
    # Native particle velocities are AoS. Gas row 7 is density, following
    # internal energy, support, acceleration and the two saved energy rates.
    density = sorted(row[7] for row in state.gas)
    energy = sorted(row[0] for row in state.gas)
    if min(density) <= 0 or min(energy) <= 0:
        raise RuntimeError("nonpositive final density or internal energy")
    mean_energy = sum(energy) / state.gas_count
    spatial_modes = []
    for axis in range(3):
        for mode in range(1, 5):
            for function in (math.cos, math.sin):
                mass_mode = energy_mode = momentum_mode = 0.0
                for particle in range(state.count):
                    phase = 2 * math.pi * mode * state.positions[axis][particle] / 2 ** (8 * state.position_bytes)
                    weight = function(phase) / state.count
                    gas = state.gas_indices[particle]
                    mass_mode += weight
                    energy_mode += weight * state.gas[gas][0] / mean_energy
                    momentum_mode += weight * state.velocities[3 * particle + axis] / math.sqrt(mean_energy)
                spatial_modes.extend((mass_mode, energy_mode, momentum_mode))
    anchors = {source[0]: source[1:] for source in state.hierarchical_sources}
    if state.has_hierarchical_sph and (len(anchors) != state.gas_count or 0 in anchors):
        raise RuntimeError("hierarchical checkpoint identities are missing or duplicated")
    return {"checkpoint": str(checkpoint), "time": state.driver.tau,
            "particles": state.gas_count, "rungs": sorted(set(state.rungs)),
            "density": density, "internal_energy": energy,
            "spatial_modes": spatial_modes,
            "anchors": anchors,
            "anchor_bytes_sha256": hashlib.sha256(state.hierarchical_bytes).hexdigest(),
            "velocity": [sorted(state.velocities[axis::3]) for axis in range(3)]}


def state_difference(first, second):
    if first["particles"] != second["particles"]:
        raise RuntimeError("comparison particle counts differ")
    if not math.isclose(first["time"], second["time"], rel_tol=1e-10, abs_tol=1e-12):
        raise RuntimeError("comparison physical times differ")
    result = {}
    for field in ("density", "internal_energy"):
        a, b = first[field], second[field]
        result[field + "_distribution_l1"] = sum(abs(x - y) for x, y in zip(a, b)) / sum(abs(x) for x in b)
    # The sound-speed scale avoids division by an initially zero flow speed.
    scale = math.sqrt(sum(second["internal_energy"]) / second["particles"])
    result["velocity_distribution_l1"] = sum(abs(x - y)
        for a, b in zip(first["velocity"], second["velocity"])
        for x, y in zip(a, b)) / (3 * first["particles"] * scale)
    result["spatial_mode_maximum_error"] = max(abs(x - y)
        for x, y in zip(first["spatial_modes"], second["spatial_modes"]))
    return result


def summarize(directory, args, complete=True):
    output = (directory / "run.log").read_text(errors="replace")
    state = snapshot(directory)
    if complete and not math.isclose(state["time"], args.end_time, rel_tol=1e-9, abs_tol=1e-12):
        raise RuntimeError(f"case stopped at t={state['time']}, expected {args.end_time}")
    energy = table(directory / "energy.dat", 9)
    maximum_error = max(abs(row[8]) for row in energy)
    if maximum_error > args.energy_tolerance:
        raise RuntimeError(f"energy residual {maximum_error} exceeds {args.energy_tolerance}")
    rows = counters(output)
    active_fraction = [row["active"] / state["particles"] for row in rows]
    fine = [row for row in rows if 0 < row["active"] < state["particles"]]
    full = [row for row in rows if row["active"] == state["particles"] and row["time"] > 0]
    work = {"events": len(rows), "partial_active_events": len(fine),
            "minimum_active_fraction": min(active_fraction, default=None),
            "counts": {key: sum(row[key] for row in rows) for key in COUNTER_FIELDS[2:]}}
    # These are measured total event costs. Different numbers of support-solve
    # passes can affect them; retained-cohort unit tests establish no inactive sort.
    if fine and full:
        work["fine_to_full_median_sort"] = statistics.median(row["sorted"] for row in fine) / max(
            1, statistics.median(row["sorted"] for row in full))
        work["fine_to_full_median_density"] = statistics.median(row["density"] for row in fine) / max(
            1, statistics.median(row["density"] for row in full))
    result = {"directory": str(directory), "time": state["time"],
              "particles": state["particles"], "final_rungs": state["rungs"],
              "maximum_energy_residual": maximum_error, "final_energy": energy[-1], "work": work}
    if args.scenario == "sod" and (directory / "sod.txt").exists():
        samples = table(directory / "sod.txt", 5)
        result["exact_density_l1"] = sum(abs(row[1] - row[2]) for row in samples) / sum(row[2] for row in samples)
    return result, state


def stop_process(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.wait()
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_case(args, name, integrator, cfl, *, maximum_iterations=None, restart=None):
    directory = args.workdir / name
    if args.analyze_only:
        return summarize(directory, args, complete=maximum_iterations is None)
    # Avoid deleting artifacts from an earlier run or another task.
    directory.mkdir(parents=True, exist_ok=False)
    if restart is not None:
        shutil.copytree(restart, directory / restart.name)
    options = {"run.timeIntegrator": integrator, "run.scenario": args.scenario,
               "run.endTime": args.end_time, "run.numberOfSteps": args.coarse_steps,
               "run.maxIterations": maximum_iterations or args.max_iterations,
               "run.minRung": args.minimum_rung, "run.verbose": "true",
               "initial.particlesPerDimension": args.resolution,
               "gravity.method.far": "fmm",
               "gravity.enabled": "true" if args.self_gravity else "false",
               "hydro.enabled": "true", "hydro.cfl": cfl,
               "chemistry.enabled": "false", "starFormation.enabled": "false",
               "gravity.cuda": "true" if args.cuda else "false",
               "drift.cuda": "true" if args.cuda else "false",
               "hydro.cuda": "true" if args.cuda else "false", "analysis.groups": "false",
               "analysis.views": "false", "analysis.power": "false",
               "analysis.tracers": "false", "lightcone.enabled": "false",
               "output.directory": ".", "output.silo": "false",
               "checkpoint.intervalSeconds": 1000000000}
    if restart is not None:
        options["checkpoint.readIndex"] = int(restart.name.split(".")[1])
    if args.mass_solar is not None:
        options["units.massSolar"] = args.mass_solar
    for attribute, option in (("gravity_eta", "gravity.timestepEta"),
                              ("gravity_opening_angle", "gravity.openingAngle"),
                              ("gravity_softening", "gravity.softening")):
        value = getattr(args, attribute, None)
        if value is not None:
            options[option] = value
    command = [str(args.executable), f"--hpx:threads={args.threads}", "--hpx:bind=none",
               f"--configuration={ROOT / 'configurations' / (CONFIGURATIONS[args.scenario] + '.ini')}"]
    command += [f"--{key}={value}" for key, value in options.items()]
    (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    environment = dict(os.environ, OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if not args.cuda:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    start = time.monotonic()
    with (directory / "run.log").open("w") as log:
        process = subprocess.Popen(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
        try:
            code = process.wait(timeout=args.timeout)
        finally:
            stop_process(process)
    if code:
        raise RuntimeError(f"{name} exited {code}; see {directory / 'run.log'}")
    result, state = summarize(directory, args, complete=maximum_iterations is None)
    result["wall_seconds"] = time.monotonic() - start
    return result, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--scenario", choices=CONFIGURATIONS, default="hydroWave")
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--end-time", type=float, default=0.05)
    parser.add_argument("--coarse-steps", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=256)
    parser.add_argument("--minimum-rung", type=int, default=0)
    parser.add_argument("--cfl", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--self-gravity", action="store_true", help="enable FMM self-gravity")
    parser.add_argument("--mass-solar", type=float,
                        help="override the mass unit, including the code gravity strength")
    parser.add_argument("--gravity-eta", type=float, help="gravity timestep prefactor")
    parser.add_argument("--gravity-opening-angle", type=float, help="FMM opening angle")
    parser.add_argument("--gravity-softening", type=float,
                        help="effective gravity radius in initial reference spacings; overrides the INI alias")
    parser.add_argument("--cuda", action="store_true", help="explicitly enable CUDA execution")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--reference", choices=("traditional", "global"))
    parser.add_argument("--refine", action="store_true", help="add CFL/2 and CFL/4 hierarchy runs")
    parser.add_argument("--restart", action="store_true", help="add a one-step split and resumed run")
    parser.add_argument("--analyze-only", action="store_true", help="read existing cases; start no processes")
    parser.add_argument("--fresh-run-directory", action="store_true")
    parser.add_argument("--require-mixed-rungs", action="store_true")
    parser.add_argument("--require-work-savings", action="store_true")
    parser.add_argument("--energy-tolerance", type=float, default=0.01)
    parser.add_argument("--comparison-tolerance", type=float, default=0.02)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    args.executable = args.executable.resolve()
    args.workdir = args.workdir.resolve()
    if min(args.resolution, args.coarse_steps, args.max_iterations, args.threads,
           args.end_time, args.cfl, args.timeout) <= 0 or args.minimum_rung < 0:
        parser.error("size, duration, CFL, threads and bounds must be positive")
    if args.mass_solar is not None and not (math.isfinite(args.mass_solar) and args.mass_solar > 0):
        parser.error("mass-solar must be finite and positive")
    for name in ("gravity_eta", "gravity_opening_angle", "gravity_softening"):
        value = getattr(args, name)
        if value is not None and not (math.isfinite(value) and value > 0):
            parser.error(name.replace("_", "-") + " must be finite and positive")
    if args.gravity_opening_angle is not None and args.gravity_opening_angle > 1:
        parser.error("gravity-opening-angle must not exceed one")
    args.workdir.mkdir(parents=True, exist_ok=True)
    if args.fresh_run_directory:
        if args.analyze_only:
            parser.error("--fresh-run-directory cannot analyze an existing run")
        args.workdir = Path(tempfile.mkdtemp(prefix="run_", dir=args.workdir))
    report = {"scenario": args.scenario, "runs": {}, "comparisons": {}}
    try:
        baseline, baseline_state = run_case(args, "hierarchical", "hierarchical", args.cfl)
        report["runs"]["hierarchical"] = baseline
        work = baseline["work"]
        if not work["events"]:
            raise RuntimeError("no hierarchical work counters; wrong path or missing instrumentation")
        if args.require_mixed_rungs and not work["partial_active_events"]:
            raise RuntimeError("fixture did not exercise a partial active gas population")
        if args.require_work_savings and not (
                work.get("fine_to_full_median_sort", 1) < 1 and
                work.get("fine_to_full_median_density", 1) < 1):
            raise RuntimeError("no measured fine-event sorting and density work reduction")
        if args.reference:
            reference, reference_state = run_case(args, "reference", args.reference, args.cfl / 4)
            report["runs"]["reference"] = reference
            difference = state_difference(baseline_state, reference_state)
            report["comparisons"]["ordinary_reference"] = difference
            if max(difference.values()) > args.comparison_tolerance:
                raise RuntimeError("hierarchical field distributions differ from the refined ordinary reference")
        if args.refine:
            half, half_state = run_case(args, "half_cfl", "hierarchical", args.cfl / 2)
            quarter, quarter_state = run_case(args, "quarter_cfl", "hierarchical", args.cfl / 4)
            report["runs"].update(half_cfl=half, quarter_cfl=quarter)
            coarse_error = max(state_difference(baseline_state, half_state).values())
            fine_error = max(state_difference(half_state, quarter_state).values())
            report["comparisons"]["cfl_refinement"] = {"coarse_difference": coarse_error,
                "fine_difference": fine_error,
                "observed_ratio": coarse_error / fine_error if fine_error else None}
            if coarse_error > 1e-7 and fine_error > coarse_error:
                raise RuntimeError("CFL refinement increased the inter-resolution field difference")
        if args.restart:
            split, _ = run_case(args, "split", "hierarchical", args.cfl, maximum_iterations=1)
            if split["time"] >= args.end_time:
                raise RuntimeError("split fixture finished in one step; increase the duration")
            resumed, resumed_state = run_case(args, "resumed", "hierarchical", args.cfl,
                                            restart=final_checkpoint(args.workdir / "split"))
            report["runs"].update(split=split, resumed=resumed)
            difference = state_difference(baseline_state, resumed_state)
            first_anchors, second_anchors = baseline_state["anchors"], resumed_state["anchors"]
            if first_anchors.keys() != second_anchors.keys():
                raise RuntimeError("restart changed hierarchical source identities")
            difference["anchor_maximum_relative_error"] = max(
                (abs(x - y) / max(1.0, abs(x))
                 for identity in first_anchors
                 for x, y in zip(first_anchors[identity], second_anchors[identity])), default=0)
            report["comparisons"]["anchor_raw_bytes_equal"] = (
                baseline_state["anchor_bytes_sha256"] == resumed_state["anchor_bytes_sha256"])
            report["comparisons"]["restart"] = difference
            if max(difference.values()) > 2e-6:
                raise RuntimeError("resumed physical field distributions differ from uninterrupted evolution")
        report["status"] = "passed"
    except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as error:
        report.update(status="failed", error=str(error))
    finally:
        (args.workdir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
