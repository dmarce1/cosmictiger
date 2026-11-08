# CosmicTiger - GADGET-4 application and restart regression; BSL-1.0.

"""Exercise source events, particle persistence, and cosmological GADGET-4 restarts."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import sys

from application_selection import application_options, compiled_algorithms
from run_checkpoint import checkManifest, fnv1a64
from checkpoint_format import decode_checkpoint


def run_process(command, directory, label, success=True, cuda=False):
    log_path = directory / (label + ".log")
    environment = dict(os.environ, OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if not cuda:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    print(f"{label}: running", flush=True)
    with log_path.open("w") as log:
        with subprocess.Popen(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT,
                              text=True, env=environment, start_new_session=True) as process:
            try:
                result = process.wait(timeout=240)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError(f"{label} timed out; see {log_path}")
    output = log_path.read_text(errors="replace")
    if (result == 0) != success:
        raise RuntimeError(f"{label} exited {result}; see {log_path}\n{output[-6000:]}")
    return output


def prepare(base, name, checkpoint=None):
    directory = base / name
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    if checkpoint is not None:
        shutil.copytree(checkpoint, directory / checkpoint.name)
    return directory


def read_particle_state(checkpoint, expected_count, ranks=1):
    """Read the versioned Ne/SFR payload and exclude driver timing/padding."""
    states = {}
    payloads = []
    for rank in range(ranks):
        data = (checkpoint / f"checkpoint.{rank}.dat").read_bytes()
        decoded = decode_checkpoint(data, ranks=ranks)
        if not decoded.has_stars or not decoded.has_cooling:
            raise RuntimeError("checkpoint did not contain cooling and star metadata")
        types = decoded.types
        payloads.append(decoded.physical)
        for index in range(decoded.count):
            identity, counter, formation, rate, elapsed = decoded.stars[index]
            if identity == 0 or identity in states or not all(math.isfinite(x) for x in (formation, rate, elapsed)):
                raise RuntimeError("duplicate IDs or nonfinite source metadata")
            if min(rate, elapsed) < 0 or (types[index] != 1 and rate != 0):
                raise RuntimeError("invalid stored SFR or elapsed time")
            if (formation >= 0) != (types[index] == 2):
                raise RuntimeError("stellar identity disagrees with birth timestamp")
            states[identity] = {"type": types[index], "counter": counter, "formation": formation,
                                "specific_sfr": rate, "elapsed": elapsed}
            if types[index] == 1:
                gas_index = decoded.gas_indices[index]
                cooling = decoded.cooling[gas_index]
                energy = decoded.internal_energies[gas_index]
                if not math.isfinite(cooling[0]) or cooling[0] < 0:
                    raise RuntimeError("invalid electron guess")
                if not math.isfinite(energy) or energy <= 0:
                    raise RuntimeError("invalid gas thermal energy")
                states[identity]["cooling"] = cooling
                states[identity]["energy"] = energy
    if len(states) != expected_count:
        raise RuntimeError("checkpoint changed the total particle count")
    return tuple(payloads), states


def check_diagnostics(path):
    rows = [line.split() for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if not rows or any(not math.isfinite(float(value)) for row in rows for value in row):
        raise RuntimeError(f"missing or nonfinite evolution diagnostics in {path}")


def force_saved_conversion_rate(checkpoint, ranks):
    """Make the next saved-SFR trial certain without altering any mechanical state."""
    manifest = checkpoint / "checkpoint.meta"
    records = manifest.read_text().splitlines()
    for rank in range(ranks):
        path = checkpoint / f"checkpoint.{rank}.dat"
        payload = bytearray(path.read_bytes())
        state = decode_checkpoint(payload, ranks=ranks)
        record_size = struct.calcsize("=QQddd")
        start = state.particle_end - state.count * record_size
        for index, kind in enumerate(state.types):
            if kind == 1:
                struct.pack_into("=d", payload, start + index * record_size + 24, 1.e100)
        # Reparse to validate the complete checkpoint contract before executing.
        decode_checkpoint(payload, ranks=ranks)
        path.write_bytes(payload)
        records[rank + 1] = f"{rank} {len(payload)} {fnv1a64(payload):016x}"
    manifest.write_text("\n".join(records) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--fixtures-only", action="store_true")
    parser.add_argument("--skip-fixtures", action="store_true")
    parser.add_argument("--time-integrator", choices=("traditional", "hierarchical", "global"))
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--mpi-exec")
    parser.add_argument("--mpi-numproc-flag", default="-n")
    parser.add_argument("--mpi-preflag", action="append", default=[])
    parser.add_argument("--mpi-postflag", action="append", default=[])
    parser.add_argument("--hpx-argument", action="append", default=[])
    args = parser.parse_args()
    executable = args.executable.resolve()
    base = args.workdir.resolve()
    base.mkdir(parents=True, exist_ok=True)
    available = compiled_algorithms(str(executable))
    integrators = [value for value in ("traditional", "hierarchical", "global")
                   if value in available["integrator"] and
                   (value != "hierarchical" or "fmm" in available["far"]) and
                   (args.time_integrator is None or value == args.time_integrator)]
    if not integrators:
        raise RuntimeError("star formation requires a compiled SPH-compatible integrator")
    device = str(args.cuda).lower()
    ranks = 2 if args.mpi_exec else 1
    common = {"gravity.cuda": device, "hydro.cuda": device, "drift.cuda": device,
              "chemistry.cuda": device, "starFormation.cuda": device, "hydro.enabled": "true",
              "starFormation.enabled": "true", "chemistry.enabled": "true",
              "run.darkMatterParticleCount": 512, "run.gasParticleCount": 512, "initial.closePack": "false",
              "scenario.cosmology.useGlass": "false", "analysis.groups": "false",
              "analysis.views": "false", "analysis.tracers": "false",
              "analysis.power": "false", "lightcone.enabled": "false",
              "output.directory": ".", "output.silo": "false",
              "hpx:bind": "none", "gravity.meshSize": 8}
    selection = application_options(executable, integrator=integrators[0], hydro=True)
    evidence = {"fixtures": [], "cosmology": []}

    def command(options):
        result = [str(executable), *args.hpx_argument,
                  *[f"--{key}={value}" for key, value in options.items()]]
        if args.mpi_exec:
            result = [args.mpi_exec, args.mpi_numproc_flag, "2", *args.mpi_preflag,
                      result[0], *args.mpi_postflag, *result[1:], "--hpx:localities=2",
                      "--hpx:ini=hpx.parcel.mpi.enable=1"]
        return result

    def execute(options, directory, label, success=True):
        return run_process(command(options), directory, label, success=success, cuda=args.cuda)

    for workers in (() if args.skip_fixtures else ((1,) if ranks == 2 else (1, 2))):
        for fixture, marker in (("star_formation", "GADGET-4 star formation passed:"),
                                ("starFormationParticles", "Star-formation particles passed:"),
                                ("checkpointState", "Checkpoint state and mode validation passed")):
            name = fixture + "_threads_" + str(workers)
            directory = prepare(base, name)
            output = execute(dict(common, **selection,
                **{"run.test": fixture, "hpx:threads": workers,
                   "chemistry.initializationTemperature": 10000}), directory, name)
            if marker not in output:
                raise RuntimeError(f"{name} did not report its assertions passed")
            evidence["fixtures"].append({"name": fixture, "workers": workers, "passed": True})

    if not args.fixtures_only:
        solvers = [("fmm", None)] if "fmm" in available["far"] else []
        if not solvers:
            solvers = [("none", None)]
        for far, near in solvers:
            for integrator in integrators:
                selections = application_options(executable, far=far, near=near,
                                                 integrator=integrator, hydro=True)
                gravity = far != "none"
                label = "_".join(value for value in (far, near, integrator) if value)
                options = dict(common, **selections, **{"run.scenario": "cosmology",
                    "gravity.enabled": str(gravity).lower(), "run.maxIterations": 2,
                    "run.numberOfSteps": 64, "run.minRung": 0,
                    "cosmology.sigma8": 0, "cosmology.startRedshift": 49,
                    "cosmology.endRedshift": 0, "checkpoint.intervalSeconds": 0,
                    "hpx:threads": 1, "chemistry.initializationTemperature": 10000})
                baseline = prepare(base, "cosmology_" + label)
                execute(options, baseline, "baseline")
                for iteration in (1, 2):
                    checkManifest(baseline / f"checkpoint.{iteration}", iteration, gravity=gravity, ranks=ranks)
                _, first = read_particle_state(baseline / "checkpoint.1", 1024, ranks=ranks)
                reference_payload, reference = read_particle_state(baseline / "checkpoint.2", 1024, ranks=ranks)
                if set(first) != set(reference):
                    raise RuntimeError("evolution changed stable identities")
                if not any(state["type"] == 1 and state["counter"] > 0 for state in reference.values()):
                    raise RuntimeError("full-step source did not advance gas event counters")
                cooling_changed = sum(
                    state["type"] == 1 and reference[identity]["type"] == 1
                    and (state["cooling"] != reference[identity]["cooling"]
                         or state["energy"] != reference[identity]["energy"])
                    for identity, state in first.items())
                if cooling_changed == 0:
                    raise RuntimeError("cosmological gas thermal state did not evolve")
                check_diagnostics(baseline / "energy.txt")
                resumed = prepare(base, "cosmology_" + label + "_restart", baseline / "checkpoint.1")
                resumed_options = dict(options, **{"run.maxIterations": 1, "checkpoint.readIndex": 1})
                output = execute(resumed_options, resumed, "restart")
                if "Done reading checkpoint" not in output:
                    raise RuntimeError("restart did not load the saved simulation")
                checkManifest(resumed / "checkpoint.2", 2, gravity=gravity, ranks=ranks)
                resumed_payload, resumed_state = read_particle_state(resumed / "checkpoint.2", 1024, ranks=ranks)
                if reference_payload != resumed_payload or reference != resumed_state:
                    raise RuntimeError("restart changed particle arrays, electron state, or saved SFR")
                if (resumed / "energy.txt").exists():
                    check_diagnostics(resumed / "energy.txt")
                for key, value in (("starFormation.seed", 2), ("starFormation.beta", 0.15),
                                   ("starFormation.criticalOverDensity", 58),
                                   ("starFormation.criticalPhysicalDensity", 0.1),
                                   ("chemistry.initializationTemperature", 20000),
                                   ("chemistry.minimumSpecificEnergy", 1e8)):
                    directory = prepare(base, "reject_" + label + "_" + key.split(".")[1],
                                        baseline / "checkpoint.1")
                    rejected = execute(dict(resumed_options, **{key: value,
                        "run.maxIterations": 0}), directory, "incompatible_restart", success=False)
                    if "checkpoint" not in rejected.lower() or "Done reading checkpoint" in rejected:
                        raise RuntimeError("incompatible source parameters were not rejected before loading")
                evidence["cosmology"].append({"gravity": far, "near": near, "integrator": integrator,
                    "particles": len(reference), "gas": sum(s["type"] == 1 for s in reference.values()),
                    "stars": sum(s["type"] == 2 for s in reference.values()),
                    "maximum_event_counter": max(s["counter"] for s in reference.values()),
                    "source_schedule": "event_then_cooling_full_step", "exact_particle_restart": True,
                    "thermal_particles_changed": cooling_changed, "exact_electron_sfr_restart": True,
                    "changed_parameters_rejected": True})
                print(f"{label}: finite cosmology and exact electron/SFR restart passed", flush=True)

                # Start two trajectories from the same controlled checkpoint.
                # The first completed interval must include conversion before
                # its final checkpoint; the subsequent restart must not replay it.
                conversion = prepare(base, "conversion_" + label, baseline / "checkpoint.1")
                force_saved_conversion_rate(conversion / "checkpoint.1", ranks)
                _, before_conversion = read_particle_state(conversion / "checkpoint.1", 1024, ranks)
                conversion_options = dict(options, **{"checkpoint.readIndex": 1,
                                                       "run.maxIterations": 2})
                execute(conversion_options, conversion, "conversion")
                _, converted = read_particle_state(conversion / "checkpoint.2", 1024, ranks)
                new_stars = [identity for identity in converted
                             if before_conversion[identity]["type"] == 1 and converted[identity]["type"] == 2]
                if not new_stars:
                    raise RuntimeError("a certain saved-SFR event produced no stars at a completed endpoint")
                for identity in new_stars:
                    if converted[identity]["counter"] != before_conversion[identity]["counter"] + 1:
                        raise RuntimeError("conversion did not consume exactly one source event")
                reference_converted, reference_stars = read_particle_state(conversion / "checkpoint.3", 1024, ranks)
                conversion_restart = prepare(base, "conversion_restart_" + label,
                                             conversion / "checkpoint.2")
                execute(dict(options, **{"checkpoint.readIndex": 2, "run.maxIterations": 1}),
                        conversion_restart, "conversion_restart")
                resumed_converted, resumed_stars = read_particle_state(conversion_restart / "checkpoint.3", 1024, ranks)
                if reference_converted != resumed_converted or reference_stars != resumed_stars:
                    raise RuntimeError("restart across gas-to-star conversion changed the physical state")
                for identity in new_stars:
                    if resumed_stars[identity] != converted[identity]:
                        raise RuntimeError("newborn star source metadata changed after conversion")
                evidence["cosmology"][-1]["forced_conversion"] = {
                    "new_stars": len(new_stars), "exact_restart": True,
                    "remaining_gas": sum(value["type"] == 1 for value in converted.values())}
                print(f"{label}: {len(new_stars)} endpoint conversions and exact subsequent restart passed", flush=True)
    (base / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print("Star formation application regression passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
