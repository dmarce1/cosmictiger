# CosmicTiger - position backend selection and checkpoint regression; BSL-1.0.

"""Exercise configured position storage, CLI/INI checks, and native checkpoints."""

import argparse
from pathlib import Path
import re
import subprocess
import struct
import sys

from run_checkpoint import checkManifest, checkpoint_position_bytes, fnv1a64, prepareCase, runSimulation
from run_scenarios import hydro_particle_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--gravity-method", choices=("fmm", "pmmm"), required=True)
    parser.add_argument("--near-method", choices=("pp", "fmm"))
    parser.add_argument("--order", type=int)
    parser.add_argument("--time-integrator", choices=("leapfrog", "hierarchical"), required=True)
    parser.add_argument("--without-sph", action="store_true")
    parser.add_argument("--position-type", choices=("fixed32", "fixed64"), required=True)
    args = parser.parse_args()
    executable = args.executable.resolve()
    selected = args.position_type
    opposite = "fixed64" if selected == "fixed32" else "fixed32"
    bits = 32 if selected == "fixed32" else 64
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    common = ["--hpx:threads=2", "--hpx:bind=none", "--compute.cuda=false",
              "--hydro.enabled=false", "--chemistry.enabled=false",
              "--initial.particlesPerDimension=4", "--analysis.groups=false",
              "--analysis.tracers=false", "--analysis.views=false", "--analysis.power=false",
              "--lightcone.enabled=false", "--output.silo=false", "--output.directory=."]
    if args.gravity_method == "pmmm":
        common.append("--gravity.meshSize=8")
    metadata = ["--run.test=checkpointState"]

    def selection(name, options, program=executable):
        directory = prepareCase(workdir, name)
        output = runSimulation([str(program), *common, *metadata, *options], directory, "run.log")
        if not re.search(r"^compute\.positionType\s*:\s*" + selected + r"\s*$",
                         output, re.MULTILINE):
            raise RuntimeError(f"{name}: incorrect selected position representation")
        buildSettings = {"timeIntegrator": args.time_integrator,
                         "gravityMethod": args.gravity_method}
        if args.near_method is not None:
            buildSettings["pmmmNearMethod"] = args.near_method
        if args.order is not None:
            buildSettings["fmmOrder"] = str(args.order)
        for setting, expectedValue in buildSettings.items():
            if not re.search(r"^build\." + setting + r"\s*:\s*" + expectedValue + r"\s*$",
                             output, re.MULTILINE):
                raise RuntimeError(f"{name}: position backend changed compiled {setting}")
        if f"Checkpoint position payload round trip passed ({bits} bits)" not in output:
            raise RuntimeError(f"{name}: native position payload was not verified")

    def rejection(name, options):
        directory = prepareCase(workdir, name)
        output = runSimulation([str(executable), *common, *options], directory,
                               "run.log", expectSuccess=False)
        diagnostic = (f"this build uses {selected} positions; configure with "
                      f"-DCOSMICTIGER_POSITION_TYPE={opposite} to use {opposite}")
        if diagnostic not in output:
            raise RuntimeError(f"{name}: incompatible position selection lacked build guidance")

    selection("default", [])
    selection("explicit", [f"--compute.positionType={selected}"])
    selection("abbreviated", [f"--compute.positionT={selected}"])
    selection("spaced", ["--compute.positionType", selected])
    rejection("opposite", [f"--compute.positionType={opposite}"])
    rejection("opposite_abbreviated", [f"--compute.positionT={opposite}"])
    rejection("opposite_spaced", ["--compute.positionType", opposite])
    config = workdir / "position.ini"
    config.write_text(f"[compute]\npositionType = {selected}\n")
    selection("ini", [f"--configuration={config}"])
    selection("abbreviated_config", [f"--config={config}"])
    rejection("cli_opposite_precedence", ["--configuration", str(config),
                                           f"--compute.positionType={opposite}"])
    config.write_text(f"compute.positionType = {opposite}\n")
    rejection("ini_opposite", [f"--configuration={config}"])
    selection("cli_precedence", [f"--configuration={config}",
                                 f"--compute.positionType={selected}"])
    alias = workdir / "application_alias"
    alias.unlink(missing_ok=True)
    alias.symlink_to(executable)
    selection("symlink", [f"--compute.positionType={selected}"], alias)
    invalid = prepareCase(workdir, "invalid")
    output = runSimulation([str(executable), *common, "--compute.positionType=float64"],
                           invalid, "run.log", expectSuccess=False)
    if "compute.positionType must be fixed32 or fixed64" not in output:
        raise RuntimeError("invalid position selection lacked its diagnostic")

    # Exercise a complete driver checkpoint in addition to the exact raw-word
    # round trips above, then reject changing representation during restart.
    baseline = prepareCase(workdir, "evolution")
    evolution = [str(executable), *common, "--run.scenario=plummer",
                 "--run.endTime=0.00001", "--run.numberOfSteps=2",
                 "--checkpoint.intervalSeconds=1000000000"]
    runSimulation([*evolution, f"--compute.positionType={selected}", "--run.maxIterations=1"],
                  baseline, "run.log")
    checkpoints = [path for path in baseline.glob("checkpoint.*")
                   if path.is_dir() and re.fullmatch(r"checkpoint\.[0-9]+", path.name)]
    if not checkpoints:
        raise RuntimeError("configured position evolution did not publish a checkpoint")
    checkpoint = max(checkpoints, key=lambda path: int(path.name.split(".")[1]))
    iteration = int(checkpoint.name.split(".")[1])
    checkManifest(checkpoint, iteration, scenario="plummer", endTime=0.00001)
    payload = (checkpoint / "checkpoint.0.dat").read_bytes()
    if checkpoint_position_bytes(payload) != bits // 8:
        raise RuntimeError("checkpoint used the wrong position representation")
    directory = prepareCase(workdir, "restart", checkpoint)
    output = runSimulation([*evolution, f"--compute.positionType={selected}",
                            f"--checkpoint.readIndex={iteration}", "--run.maxIterations=0"],
                           directory, "run.log")
    if "Done reading checkpoint" not in output:
        raise RuntimeError("checkpoint was not loaded")
    wrong_width = prepareCase(workdir, "restart_wrong_width", checkpoint)
    rank_file = wrong_width / checkpoint.name / "checkpoint.0.dat"
    payload = bytearray(rank_file.read_bytes())
    flags = int.from_bytes(payload[12:16], sys.byteorder) ^ (1 << 15)
    payload[12:16] = flags.to_bytes(4, sys.byteorder)
    rank_file.write_bytes(payload)
    # Keep the deliberate layout mismatch internally consistent so integrity
    # validation succeeds and the reader reaches its representation check.
    manifest = rank_file.parent / "checkpoint.meta"
    lines = manifest.read_text().splitlines()
    lines[1] = f"0 {len(payload)} {fnv1a64(payload):016x}"
    manifest.write_text("\n".join(lines) + "\n")
    output = runSimulation([*evolution, f"--checkpoint.readIndex={iteration}",
                            "--run.maxIterations=0"], wrong_width, "run.log", expectSuccess=False)
    if "particle layout" not in output:
        raise RuntimeError("position-width mismatch lacked a layout diagnostic")
    # Exercise the SPH analysis reader using the configured representation.
    hydro = prepareCase(workdir, "hydro")
    hydro_options = [option for option in common if not option.startswith(
        ("--hydro.enabled=", "--initial.particlesPerDimension="))]
    output = runSimulation([str(executable), *hydro_options, f"--compute.positionType={selected}",
                            "--hydro.enabled=true",
                            "--initial.particlesPerDimension=8", "--run.scenario=sod",
                            "--run.endTime=0.00001", "--run.numberOfSteps=2", "--run.maxIterations=1",
                            "--checkpoint.intervalSeconds=1000000000"], hydro, "run.log",
                           expectSuccess=not args.without_sph and args.time_integrator == "leapfrog")
    if args.without_sph:
        if "COSMICTIGER_WITH_SPH=ON" not in output:
            raise RuntimeError("SPH rejection lacked SPH build guidance")
        print(f"{selected} selection, raw checkpoint bits, evolution/restart, width rejection, "
              "and disabled SPH rejection passed")
        return 0
    if args.time_integrator == "hierarchical":
        if "COSMICTIGER_TIME_INTEGRATOR=leapfrog" not in output:
            raise RuntimeError("SPH rejection lacked integrator build guidance")
        print(f"{selected} selection, raw checkpoint bits, evolution/restart, width rejection, "
              "and hierarchical SPH incompatibility passed")
        return 0
    hydro_checkpoint = hydro / "checkpoint.1"
    checkManifest(hydro_checkpoint, 1, scenario="sod", endTime=0.00001, gravity=False)
    if checkpoint_position_bytes((hydro_checkpoint / "checkpoint.0.dat").read_bytes()) != bits // 8:
        raise RuntimeError("hydro checkpoint used the wrong position representation")
    payload = hydro_particle_payload(hydro_checkpoint)
    candidates = [(payload.find(struct.pack("=7I", 0x43545350, version, size, 8, 12, 3, 4)), size)
                  for size in (4, 8) for version in (2, 4, 6)]
    sph_offset, integer_bytes = max(candidates)
    if sph_offset < 0:
        raise RuntimeError("hydro checkpoint reader could not locate the SPH header")
    count = int.from_bytes(payload[:integer_bytes], sys.byteorder, signed=True)
    if bits == 64:
        raw_positions = struct.unpack_from(f"={3 * count}Q", payload, 2 * integer_bytes)
        if not any(word & 0xFFFFFFFF for word in raw_positions):
            raise RuntimeError("fixed64 hydro checkpoint reader lost position precision")
    mislabeled = prepareCase(workdir, "hydro_wrong_width", hydro_checkpoint) / hydro_checkpoint.name
    rank_file = mislabeled / "checkpoint.0.dat"
    payload = bytearray(rank_file.read_bytes())
    flags = int.from_bytes(payload[12:16], sys.byteorder) ^ (1 << 15)
    payload[12:16] = flags.to_bytes(4, sys.byteorder)
    rank_file.write_bytes(payload)
    try:
        hydro_particle_payload(mislabeled)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("hydro payload was silently parsed with the wrong position width")
    print(f"{selected} selection, raw checkpoint bits, evolution/restart, hydro readers, "
          "and width rejection passed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
