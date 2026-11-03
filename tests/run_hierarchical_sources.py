"""Couple GADGET-4 cooling and star formation to hierarchical SPH and MFM on mixed rungs.

A small late-starting cosmology gives the gas different rungs, and forced
star-formation thresholds make cells convert within a coarse step. For each hydro
method the runner requires events that close only part of the gas, conversions, a
consistent species count, an exact restart from the first checkpoint after a
conversion, and, at the first full synchronization, an energy residual comparable to
the traditional integrator's together with its star count and thermal energy. SPH must convert at a partial event, where each particle's source
interval is its own rung's; MFM must close cells at gravity events before their
hydro endpoints, which shortens their source intervals.
"""

import argparse
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys

from application_selection import application_options, compiled_algorithms
from checkpoint_format import read_prefix

GAS = 16 ** 3
ITERATIONS = 70


def run_process(command, directory, label):
    log_path = directory / (label + ".log")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    print(f"{directory.name}/{label}: running", flush=True)
    with log_path.open("w") as log:
        with subprocess.Popen(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT,
                              text=True, env=environment, start_new_session=True) as process:
            try:
                result = process.wait(timeout=900)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError(f"{label} timed out; see {log_path}")
    output = log_path.read_text(errors="replace")
    if result != 0:
        raise RuntimeError(f"{label} exited {result}; see {log_path}\n{output[-6000:]}")
    return output


def prepare(base, name, checkpoint=None):
    directory = base / name
    if directory.exists():
        shutil.rmtree(directory)
    (directory / "output").mkdir(parents=True)
    if checkpoint is not None:
        shutil.copytree(checkpoint, directory / "output" / checkpoint.name)
    return directory


def species(path):
    """Species counts and mean stored a^2 u; both gas layouts begin with the thermal pairs."""
    data = path.read_bytes()
    prefix = read_prefix(data)
    reader = prefix.reader
    integer_code = "q" if prefix.integer_bytes == 8 else "i"
    count = reader.take(integer_code)[0]
    gas = reader.take(integer_code)[0]
    reader.skip(3 * prefix.position_bytes * count + 13 * count)
    types = reader.take("b", count)
    reader.skip(prefix.integer_bytes * count)
    if reader.take("I", 7)[0] != 0x43545350 or reader.take(integer_code)[0] != gas:
        raise RuntimeError(f"unexpected gas layout in {path}")
    thermal = reader.take("f", 2 * gas)[::2]
    if gas != types.count(1) or any(not math.isfinite(u) or u <= 0.0 for u in thermal):
        raise RuntimeError(f"invalid gas thermal state in {path}")
    return {"gas": gas, "stars": types.count(2), "mean": math.fsum(thermal) / gas,
            "physical": data[prefix.particle_start:], "prefix": data[:prefix.driver_offset]}


def prune(directory, keep):
    """Checkpoints are written every step; keep only the ones a check compared."""
    for path in (directory / "output").glob("checkpoint.*"):
        if path.is_dir() and int(path.name.split(".")[1]) not in keep:
            shutil.rmtree(path)


def synchronization(directory, iterations):
    """The first checkpoint at the end of the first coarse step, and its index."""
    for step in range(1, iterations + 1):
        path = directory / f"output/checkpoint.{step}/checkpoint.0.dat"
        driver = read_prefix(path.read_bytes()).driver
        if driver.step == 1 and driver.integer_time == 0:
            return step, species(path)
    raise RuntimeError(f"{directory.name} did not reach a full synchronization")


def synchronized_residual(directory):
    rows = [row for row in energy_rows(directory) if float(row[0]) == 1.0]
    if len(rows) != 1:
        raise RuntimeError(f"{directory.name} has no energy row for the first coarse step")
    return float(rows[0][8])


def fields(line):
    return dict(re.findall(r"([A-Za-z_]+)=([^\s]+)", line))


def events(method, output):
    """Per hierarchical event: closed cells, the gas present, conversions and early closures."""
    result = []
    if method == "sph":
        stars = {}
        for line in output.splitlines():
            if "SPH hierarchy sources:" in line:
                values = fields(line)
                stars[values["time"]] = int(values["stars"])
            elif "SPH hierarchy:" in line:
                values = fields(line)
                result.append({"closed": int(values["active"]), "stars": stars.pop(values["time"], 0),
                               "early": 0})
        return result
    pending = None
    for line in output.splitlines():
        if "Hopkins hierarchy begin:" in line:
            values = fields(line)
            pending = {"closed": int(values["closed"]), "total": int(values["total"]),
                       "stars": int(values["stars"]),
                       "early": int(values["closed"]) - int(values["hydroDue"])}
        elif "Hopkins step:" in line and pending is not None:
            pending["stars"] += int(fields(line)["stars"])
            result.append(pending)
            pending = None
    return result


def energy_rows(directory):
    rows = [line.split() for line in (directory / "output/energy.txt").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if not rows or any(not math.isfinite(float(value)) for row in rows for value in row):
        raise RuntimeError(f"missing or nonfinite energy rows in {directory}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--methods", nargs="+", choices=("sph", "mfm"))
    args = parser.parse_args()
    executable = args.executable.resolve()
    base = args.workdir.resolve()
    base.mkdir(parents=True, exist_ok=True)
    help_text = subprocess.run([str(executable), "--help", "--hpx:threads=1", "--hpx:bind=none"],
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               timeout=60, check=True,
                               env=dict(os.environ, CUDA_VISIBLE_DEVICES="")).stdout
    capabilities = dict(re.findall(r"^\s*(Equilibrium cooling|Star formation): (enabled|disabled)",
                                   help_text, re.MULTILINE))
    if capabilities.get("Star formation") != "enabled":
        raise RuntimeError("hierarchical sources require a build with star formation")
    compiled = compiled_algorithms(str(executable))
    if not {"hierarchical", "traditional"} <= set(compiled["integrator"]) or "fmm" not in compiled["far"]:
        raise RuntimeError("hierarchical sources require FMM and both comparison integrators")
    probe = subprocess.run([str(executable), "--run.test=checkpointState", "--hydro.enabled=false",
                            "--initial.particlesPerDimension=8", "--hpx:threads=1", "--hpx:bind=none",
                            "--run.timeIntegrator=traditional", "--gravity.method.far=fmm"],
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120,
                           check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="")).stdout
    meshless = (re.search(r"^build\.hopkins\s*:\s*T\s*$", probe, re.MULTILINE) is not None and
                re.search(r"^compute\.positionType\s*:\s*fixed64\s*$", probe, re.MULTILINE) is not None)
    methods = args.methods or ["sph"] + (["mfm"] if meshless else [])

    def options(method, integrator, iterations):
        values = {"run.scenario": "cosmology", "run.maxIterations": iterations,
                  "run.numberOfSteps": 64, "run.minRung": 0, "run.verbose": "true",
                  "hpx:threads": args.threads, "hpx:bind": "none",
                  "hydro.enabled": "true", "hydro.method": method, "gravity.enabled": "true",
                  "gravity.cuda": "false", "hydro.cuda": "false", "drift.cuda": "false",
                  "chemistry.cuda": "false", "starFormation.cuda": "false",
                  "initial.particlesPerDimension": 16, "initial.closePack": "false",
                  "initial.useGlass": "false", "cosmology.startRedshift": 4,
                  "cosmology.endRedshift": 0, "chemistry.enabled": "true",
                  "chemistry.initializationTemperature": 10000, "starFormation.enabled": "true",
                  # Mean hydrogen density at z=4 is about 2e-5 per cubic centimetre:
                  # mild overdensities become eligible within a coarse step.
                  "starFormation.criticalOverDensity": 0,
                  "starFormation.criticalPhysicalDensity": 1.e-5,
                  "starFormation.starFormationTimeGyr": 0.2,
                  "analysis.groups": "false", "analysis.tracers": "false",
                  "analysis.power": "false", "analysis.views": "false",
                  "lightcone.enabled": "false", "output.silo": "false",
                  "output.directory": "output", "checkpoint.intervalSeconds": 0}
        values.update(application_options(executable, far="fmm", integrator=integrator, hydro=True))
        return [str(executable), *[f"--{key}={value}" for key, value in values.items()]]

    for method in methods:
        name = f"{method}_hierarchical"
        directory = prepare(base, name)
        output = run_process(options(method, "hierarchical", ITERATIONS), directory, "run")
        record = events(method, output)
        formed = sum(event["stars"] for event in record)
        gas = GAS
        partial_events = partial_conversions = 0
        for event in record:
            # Compare with the gas present before this event's conversions.
            partial_events += event["closed"] < gas
            partial_conversions += event["stars"] > 0 and event["closed"] < gas
            gas -= event["stars"]
        if partial_events == 0:
            raise RuntimeError(f"{name} never closed part of the gas; rungs were not mixed")
        if formed == 0 or (method == "sph" and partial_conversions == 0):
            raise RuntimeError(f"{name} formed {formed} stars, {partial_conversions} at partial events")
        if method == "mfm" and not any(event["early"] > 0 for event in record):
            raise RuntimeError(f"{name} never closed a cell before its hydro endpoint")
        checkpoints = [species(directory / f"output/checkpoint.{step}/checkpoint.0.dat")
                       for step in range(1, ITERATIONS + 1)]
        if any(state["gas"] + state["stars"] != GAS or state["stars"] > GAS // 4
               for state in checkpoints) or checkpoints[-1]["stars"] != formed:
            raise RuntimeError(f"{name} species {checkpoints[-1]} disagree with {formed} conversions")
        residual = synchronized_residual(directory)
        synchronized, final = synchronization(directory, ITERATIONS)
        split = next(step for step, state in enumerate(checkpoints, start=1) if state["stars"])
        if split >= ITERATIONS:
            raise RuntimeError(f"{name} first converted at its last step")
        resumed = prepare(base, name + "_restart", directory / f"output/checkpoint.{split}")
        output = run_process(options(method, "hierarchical", ITERATIONS - split) + [
            f"--checkpoint.readIndex={split}"], resumed, "restart")
        if "Done reading checkpoint" not in output:
            raise RuntimeError(f"{name} did not load its restart")
        restart = species(resumed / f"output/checkpoint.{ITERATIONS}/checkpoint.0.dat")
        if (restart["physical"], restart["prefix"]) != (checkpoints[-1]["physical"],
                                                         checkpoints[-1]["prefix"]):
            raise RuntimeError(f"{name} restart from checkpoint {split} changed the state")
        # The integrators take different sub-steps; compare them at the same time.
        reference = prepare(base, f"{method}_traditional")
        run_process(options(method, "traditional", ITERATIONS), reference, "run")
        _, expected = synchronization(reference, ITERATIONS)
        traditional = synchronized_residual(reference)
        if abs(residual) > 2.0 * abs(traditional) + 2.e-5:
            raise RuntimeError(f"{name} energy residual {residual:.3e}; traditional {traditional:.3e}")
        # Without sources the two MFM integrators already differ by about 7% in the
        # mean thermal energy of this structured case. The sources raise it about
        # sixfold, so either tolerance still exposes a missed or doubled source.
        tolerance = 0.15 if method == "mfm" else 0.05
        if (abs(final["stars"] - expected["stars"]) > max(3, expected["stars"] // 4) or
                abs(final["mean"] - expected["mean"]) > tolerance * expected["mean"]):
            raise RuntimeError(f"{name} synchronized with {final['stars']} stars and mean a^2 u "
                               f"{final['mean']:.6e}; traditional {expected['stars']}, "
                               f"{expected['mean']:.6e}")
        print(f"{name}: {formed} conversions ({partial_conversions} of {partial_events} partial "
              f"events), exact restart from checkpoint {split}; at the first synchronization "
              f"(checkpoint {synchronized}) energy residual {residual:.3e} versus traditional "
              f"{traditional:.3e}, {final['stars']} versus {expected['stars']} stars, mean a^2 u "
              f"{final['mean']:.6e} versus {expected['mean']:.6e}", flush=True)
        prune(directory, {split, synchronized, ITERATIONS})
        prune(resumed, {ITERATIONS})
        prune(reference, set())
    print("Hierarchical source regressions passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
