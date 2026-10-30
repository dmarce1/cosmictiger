# CosmicTiger - SH03 application and restart regression; BSL-1.0.

"""Exercise source events, particle persistence, and cosmological SH03 restarts."""

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
from run_checkpoint import checkManifest, checkpoint_position_bytes


STAR_MAGIC = 0x4354534650415254
STAR_RECORD = struct.Struct("=QQddfdddd")
CHEMISTRY_RECORD = struct.Struct("=9f2d")


def run_process(command, directory, label, success=True):
    log_path = directory / (label + ".log")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
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


def read_particle_state(checkpoint, expected_count):
    """Exclude driver timing/padding; retain physical arrays and every source scalar."""
    data = (checkpoint / "checkpoint.0.dat").read_bytes()
    position_bytes = checkpoint_position_bytes(data)
    flags = struct.unpack_from("=I", data, 12)[0]
    if not flags & (1 << 20) or not flags & (1 << 11):
        raise RuntimeError("checkpoint did not mark both star formation and chemistry")
    magic = struct.pack("=QQ", STAR_MAGIC, 1)
    star_offset = data.find(magic)
    if star_offset < 0 or data.find(magic, star_offset + 1) >= 0:
        raise RuntimeError("missing or ambiguous versioned star formation particle block")
    end = star_offset + len(magic) + expected_count * STAR_RECORD.size
    if end > len(data):
        raise RuntimeError("truncated star formation particle metadata")
    candidates = []
    for integer_bytes in (4, 8):
        for version in (40, 42, 44, 46):
            gas_magic = struct.pack("=7I", 0x43545350, version, integer_bytes, 8, 12, 3, 4)
            offset = data.find(gas_magic)
            if offset < 0:
                continue
            start = offset - 2 * integer_bytes - expected_count * (
                3 * position_bytes + 14 + integer_bytes)
            if start < 0:
                continue
            count = int.from_bytes(data[start:start + integer_bytes], sys.byteorder, signed=True)
            gas_count = int.from_bytes(data[start + integer_bytes:start + 2 * integer_bytes],
                                       sys.byteorder, signed=True)
            stored_gas_count = int.from_bytes(data[offset + 28:offset + 28 + integer_bytes],
                                              sys.byteorder, signed=True)
            if count == expected_count and 0 <= gas_count <= count and gas_count == stored_gas_count:
                candidates.append((start, integer_bytes, gas_count, offset, version))
    if len(candidates) != 1:
        raise RuntimeError(f"cannot identify a unique physical particle block: {candidates}")
    start, integer_bytes, gas_count, gas_offset, gas_version = candidates[0]
    type_offset = start + 2 * integer_bytes + expected_count * (3 * position_bytes + 13)
    types = data[type_offset:type_offset + expected_count]
    if len(types) != expected_count or any(kind not in (0, 1, 2) for kind in types):
        raise RuntimeError("invalid particle species in checkpoint")
    if types.count(1) != gas_count:
        raise RuntimeError("gas population disagrees with checkpoint particle types")
    viscosity_bytes = {40: 0, 42: 8, 44: 12, 46: 12}[gas_version]
    # Fixed SPH arrays contain 58 bytes plus one particle index per gas particle.
    chemistry_offset = gas_offset + 28 + integer_bytes + gas_count * (
        58 + integer_bytes + viscosity_bytes)
    if chemistry_offset + gas_count * CHEMISTRY_RECORD.size != star_offset:
        raise RuntimeError("combined SPH chemistry payload has an unexpected size")
    cat_offset = type_offset + expected_count
    gas_indices = set()
    states = {}
    for index in range(expected_count):
        offset = star_offset + len(magic) + index * STAR_RECORD.size
        record = STAR_RECORD.unpack_from(data, offset)
        identity, counter, formation, wind, metal, half, heating, optical, elapsed = record
        if identity == 0 or identity in states or not all(math.isfinite(value) for value in record[2:]):
            raise RuntimeError("checkpoint has duplicate IDs or nonfinite source metadata")
        if not 0 <= metal <= 1 or min(wind, half, optical, elapsed) < 0:
            raise RuntimeError("checkpoint has invalid source clocks or metallicity")
        if (formation >= 0) != (types[index] == 2):
            raise RuntimeError("stellar identity disagrees with its formation timestamp")
        states[identity] = {"type": types[index], "counter": counter, "formation": formation,
                            "wind": wind, "metal": metal, "half": half, "elapsed": elapsed}
        if types[index] == 1:
            gas_index = int.from_bytes(data[cat_offset + index * integer_bytes:
                                            cat_offset + (index + 1) * integer_bytes],
                                       sys.byteorder, signed=True)
            if gas_index < 0 or gas_index >= gas_count or gas_index in gas_indices:
                raise RuntimeError("gas particle indices do not form a unique mapping")
            gas_indices.add(gas_index)
            chemistry = CHEMISTRY_RECORD.unpack_from(
                data, chemistry_offset + gas_index * CHEMISTRY_RECORD.size)
            if (not all(math.isfinite(value) for value in chemistry)
                    or any(abs(value) > 0.5 for value in chemistry[:6])
                    or min(chemistry[6:9]) <= 0):
                raise RuntimeError("checkpoint has invalid chemical abundances, energy or EOS")
            fractions = [value if math.copysign(1.0, value) > 0 else 1.0 + value
                         for value in chemistry[:6]]
            if (fractions[1] + fractions[2] > 1.0 + 1e-6
                    or any(value != 0.0 for value in fractions[3:])):
                raise RuntimeError("checkpoint has invalid Atomic6 chemical composition")
            if chemistry[9:] != (0.0, 0.0):
                raise RuntimeError("chemistry and star formation both own a pending source step")
            states[identity]["chemistry"] = chemistry
    return data[start:end], states


def check_diagnostics(path):
    rows = [line.split() for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if not rows or any(not math.isfinite(float(value)) for row in rows for value in row):
        raise RuntimeError(f"missing or nonfinite evolution diagnostics in {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--fixtures-only", action="store_true")
    args = parser.parse_args()
    executable = args.executable.resolve()
    base = args.workdir.resolve()
    base.mkdir(parents=True, exist_ok=True)
    available = compiled_algorithms(str(executable))
    integrators = [value for value in ("traditional", "global") if value in available["integrator"]]
    if not integrators:
        raise RuntimeError("star formation requires a compiled SPH-compatible integrator")
    common = {"compute.cuda": "false", "hydro.enabled": "true",
              "starFormation.enabled": "true", "chemistry.enabled": "true",
              "initial.particlesPerDimension": 8, "initial.closePack": "false",
              "initial.useGlass": "false", "analysis.groups": "false",
              "analysis.views": "false", "analysis.tracers": "false",
              "analysis.power": "false", "lightcone.enabled": "false",
              "output.directory": ".", "output.silo": "false",
              "hpx:bind": "none", "gravity.meshSize": 8}
    selection = application_options(executable, integrator=integrators[0], hydro=True)
    evidence = {"fixtures": [], "cosmology": []}

    def command(options):
        return [str(executable), *[f"--{key}={value}" for key, value in options.items()]]

    for workers in (1, 2):
        for fixture, marker in (("star_formation", "Star formation passed:"),
                                ("starFormationParticles", "Star-formation particles passed:"),
                                ("checkpointState", "Checkpoint state and mode validation passed")):
            name = fixture + "_threads_" + str(workers)
            directory = prepare(base, name)
            output = run_process(command(dict(common, **selection,
                **{"run.test": fixture, "hpx:threads": workers, "chemistry.initialization": "manual"})), directory, name)
            if marker not in output:
                raise RuntimeError(f"{name} did not report its assertions passed")
            evidence["fixtures"].append({"name": fixture, "workers": workers, "passed": True})

    if not args.fixtures_only:
        solvers = [("fmm", None)] if "fmm" in available["far"] else []
        if "pmmm" in available["far"]:
            solvers.extend(("pmmm", near) for near in ("pp", "fmm") if near in available["near"])
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
                    "hpx:threads": 1})
                baseline = prepare(base, "cosmology_" + label)
                run_process(command(options), baseline, "baseline")
                for iteration in (1, 2):
                    checkManifest(baseline / f"checkpoint.{iteration}", iteration, gravity=gravity)
                _, first = read_particle_state(baseline / "checkpoint.1", 1024)
                reference_payload, reference = read_particle_state(baseline / "checkpoint.2", 1024)
                if set(first) != set(reference):
                    raise RuntimeError("evolution changed stable particle identities")
                gas_states = [state for state in first.values() if state["type"] == 1]
                if not gas_states or not any(state["counter"] > 0 for state in reference.values()):
                    raise RuntimeError("cosmological source closing did not advance gas event counters")
                if not any(state["half"] > 0 for state in gas_states):
                    raise RuntimeError("restart fixture did not capture pending source half steps")
                chemistry_changed = sum(
                    state["type"] == 1 and reference[identity]["type"] == 1
                    and state["chemistry"][:7] != reference[identity]["chemistry"][:7]
                    for identity, state in first.items())
                if chemistry_changed == 0:
                    raise RuntimeError("ordinary cosmological gas did not evolve its chemistry")
                check_diagnostics(baseline / "energy.txt")
                resumed = prepare(base, "cosmology_" + label + "_restart", baseline / "checkpoint.1")
                resumed_options = dict(options, **{"run.maxIterations": 1, "checkpoint.readIndex": 1})
                output = run_process(command(resumed_options), resumed, "restart")
                if "Done reading checkpoint" not in output:
                    raise RuntimeError("restart did not load its saved simulation")
                checkManifest(resumed / "checkpoint.2", 2, gravity=gravity)
                resumed_payload, resumed_state = read_particle_state(resumed / "checkpoint.2", 1024)
                if reference_payload != resumed_payload or reference != resumed_state:
                    raise RuntimeError("restart changed particle arrays or persistent star formation state")
                # A short restart may not cross the next coarse diagnostic boundary.
                if (resumed / "energy.txt").exists():
                    check_diagnostics(resumed / "energy.txt")
                for key, value in (("starFormation.seed", 2), ("starFormation.beta", 0.15),
                                   ("starFormation.windEnergyFraction", 0.5),
                                   ("cosmology.heliumFraction", 0.3),
                                   ("chemistry.relativeTolerance", 0.0002),
                                   ("chemistry.energyAbsoluteTolerance", 2e-8)):
                    directory = prepare(base, "reject_" + label + "_" + key.split(".")[1],
                                        baseline / "checkpoint.1")
                    rejected = run_process(command(dict(resumed_options, **{key: value,
                        "run.maxIterations": 0})), directory, "incompatible_restart", success=False)
                    if "checkpoint" not in rejected.lower() or "Done reading checkpoint" in rejected:
                        raise RuntimeError("incompatible source parameters were not rejected before loading")
                entry = {"gravity": far, "near": near, "integrator": integrator, "particles": len(reference),
                         "gas": sum(state["type"] == 1 for state in reference.values()),
                         "stars": sum(state["type"] == 2 for state in reference.values()),
                         "maximum_event_counter": max(state["counter"] for state in reference.values()),
                         "pending_source_restart": True, "exact_particle_restart": True,
                         "chemistry_particles_changed": chemistry_changed,
                         "exact_chemistry_restart": True,
                         "changed_chemistry_parameters_rejected": True,
                         "changed_parameters_rejected": True}
                evidence["cosmology"].append(entry)
                print(f"{label}: finite cosmology, pending source state and exact restart passed", flush=True)
    (base / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print("Star formation application regression passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
