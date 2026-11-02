"""Check independent MFM/gravity clocks, real wakes, restart and refinement."""

import argparse
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from checkpoint_format import read_prefix
from run_hopkins import particle_state, check_conservation
from run_scenarios import run_process


def records(output, prefix):
    return [dict(re.findall(r"([A-Za-z]+)=([^\s]+)", line))
            for line in output.splitlines() if line.startswith(prefix)]


def checkpoints(directory):
    return sorted((path for path in directory.glob("output/checkpoint.*") if path.is_dir()),
                  key=lambda path: int(path.suffix[1:]))


def positions(state):
    return {anchor[0]: position for anchor, position in
            zip(state["integration_state"], state["positions"])}


def difference(first, second):
    if first["clock"][0] != second["clock"][0]:
        raise RuntimeError("reference clocks differ")
    left, right = positions(first), positions(second)
    if left.keys() != right.keys():
        raise RuntimeError("reference identities differ")
    errors = [abs((left[key][d] - right[key][d] + .5) % 1.0 - .5)
              for key in left for d in range(3)]
    return max(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    parser.add_argument("--fresh-run-directory", action="store_true")
    args = parser.parse_args()
    args.workdir = args.workdir.resolve()
    args.workdir.mkdir(parents=True, exist_ok=True)
    if args.fresh_run_directory:
        args.workdir = Path(tempfile.mkdtemp(prefix="run_", dir=args.workdir))
    command = [str(args.executable.resolve()), f"--hpx:threads={args.threads}", "--hpx:bind=none"]
    common = {"run.timeIntegrator": "hierarchical", "run.numberOfSteps": 2,
              "run.maxIterations": 4096, "run.verbose": "true",
              "initial.particlesPerDimension": 8, "hydro.enabled": "true",
              "hydro.method": "mfm", "hydro.cuda": "false", "gravity.cuda": "false",
              "drift.cuda": "false", "gravity.method.far": "fmm",
              "chemistry.enabled": "false", "starFormation.enabled": "false",
              "analysis.groups": "false", "analysis.tracers": "false",
              "analysis.power": "false", "analysis.views": "false",
              "lightcone.enabled": "false", "output.silo": "false",
              "output.directory": "output", "checkpoint.intervalSeconds": 0}

    def run(name, options, restart=None):
        directory = args.workdir / name
        directory.mkdir(exist_ok=False)
        values = dict(common, **options)
        if restart:
            (directory / "output").mkdir()
            shutil.copytree(restart, directory / "output" / restart.name)
            values["checkpoint.readIndex"] = int(restart.suffix[1:])
        invocation = command + [f"--{key}={value}" for key, value in values.items()]
        (directory / "command.json").write_text(json.dumps(invocation, indent=2) + "\n")
        output = run_process(invocation, directory, True)
        paths = checkpoints(directory)
        states = [particle_state(path, "mfm") for path in paths]
        if not states or not math.isclose(states[-1]["clock"][0], values["run.endTime"], abs_tol=1e-14):
            raise RuntimeError(f"{name} did not reach its endpoint")
        if any(state["mass"] != state["fixed_mass"] * state["count"] for state in states):
            raise RuntimeError(f"{name} changed MFM mass")
        return output, paths, states

    report = {}
    try:
        blast = {"run.scenario": "blast", "run.endTime": .004, "gravity.enabled": "false"}
        output, paths, states = run("blast", blast)
        steps = records(output, "Hopkins step:")
        trees = records(output, "Hopkins tree:")
        if not any(int(row["woken"]) > 0 for row in steps):
            raise RuntimeError("blast did not exercise a real hydro wake")
        if not any(int(row["active"]) < int(row["total"]) and int(row["cached"]) > 0 for row in steps):
            raise RuntimeError("blast did not retain sleeping pair intervals")
        if not any(anchor[58] != rung for state in states for anchor, rung in
                   zip(state["integration_state"], state["rungs"])):
            raise RuntimeError("hydro and gravity clocks never differed")
        if not any(int(row["sorted"]) == 0 and int(row["updated"]) > 0 for row in trees[1:]):
            raise RuntimeError("persistent tree topology was not reused")
        for state in states[1:]:
            check_conservation(states[0], state, "hierarchical MFM blast")
        _, _, resumed = run("blast_restart", blast, paths[0])
        for key in ("physical_records", "integration_state", "pair_cache", "clock"):
            if resumed[-1][key] != states[-1][key]:
                raise RuntimeError(f"restart changed {key}")
        report["blast"] = {"steps": len(states), "wakes": sum(int(row["woken"]) for row in steps),
                           "restart": "exact", "conservation": "passed"}

        wave = {"run.scenario": "hydroWave", "run.endTime": .04, "gravity.enabled": "true",
                "units.massSolar": 1e24, "gravity.softening": .25}
        final = []
        initial_energies = []
        for name, cfl, eta, integrator in (("wave", .1, .2, "hierarchical"),
                ("wave_half", .05, .1, "hierarchical"),
                ("wave_quarter", .025, .05, "hierarchical"),
                ("reference", .0125, .025, "global")):
            _, wave_paths, snapshots = run(name, dict(wave, **{"hydro.cfl": cfl,
                "gravity.timestepEta": eta, "run.timeIntegrator": integrator}))
            final.append(snapshots[-1])
            first_rank = sorted(wave_paths[0].glob("checkpoint.*.dat"))[0]
            initial_energies.append(read_prefix(first_rank.read_bytes()).driver.energy0)
        errors = [difference(state, final[-1]) for state in final[:-1]]
        if errors[-1] > .002 or errors[-1] > errors[0]:
            raise RuntimeError(f"MFM gravity position refinement failed: {errors}")
        # Initial synchronized energy is independent of the first opening kick.
        # Reading the native driver value avoids energy.dat's six-digit rounding.
        # All hierarchical runs share the same initial top gravity tree; the
        # global reference may differ at float force-reduction precision.
        energy_scale = max(abs(value) for value in initial_energies)
        if not all(math.isfinite(value) for value in initial_energies) or energy_scale == 0:
            raise RuntimeError(f"invalid initial energies: {initial_energies}")
        hierarchical_spread = max(initial_energies[:-1]) - min(initial_energies[:-1])
        reference_energy_error = max(abs(value - initial_energies[-1])
                                     for value in initial_energies[:-1])
        if hierarchical_spread > 1e-12 * energy_scale or \
                reference_energy_error > 3e-7 * energy_scale:
            raise RuntimeError(f"initial synchronized energy depends on timestep: {initial_energies}")
        report["self_gravity"] = {"position_max_errors": errors,
                                  "initial_energies": initial_energies,
                                  "initial_energy_spread": hierarchical_spread,
                                  "initial_energy_reference_error": reference_energy_error,
                                  "status": "passed"}
        report["status"] = "passed"
    except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as error:
        report.update(status="failed", error=str(error))
    (args.workdir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
