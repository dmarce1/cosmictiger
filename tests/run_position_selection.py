# CosmicTiger - position backend selection and checkpoint regression; BSL-1.0.

"""Exercise CLI/INI dispatch, native low-bit payloads, and restart width checks."""

import argparse
from pathlib import Path
import re
import subprocess
import struct
import sys

from run_checkpoint import checkManifest, checkpoint_position_bytes, prepareCase, runSimulation
from run_scenarios import hydro_particle_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    args = parser.parse_args()
    executable = args.executable.resolve()
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    common = ["--hpx:threads=2", "--hpx:bind=none", "--compute.cuda=false",
              "--hydro.enabled=false", "--chemistry.enabled=false",
              "--initial.particlesPerDimension=4", "--analysis.groups=false",
              "--analysis.tracers=false", "--analysis.views=false", "--analysis.power=false",
              "--lightcone.enabled=false", "--output.silo=false", "--output.directory=."]
    metadata = ["--run.test=checkpointState"]

    def selection(name, options, expected, program=executable):
        directory = prepareCase(workdir, name)
        output = runSimulation([str(program), *common, *metadata, *options], directory, "run.log")
        if not re.search(r"^compute\.positionType\s*:\s*" + expected + r"\s*$",
                         output, re.MULTILINE):
            raise RuntimeError(f"{name}: incorrect selected position representation")
        bits = 64 if expected == "fixed64" else 32
        if f"Checkpoint position payload round trip passed ({bits} bits)" not in output:
            raise RuntimeError(f"{name}: native position payload was not verified")

    selection("default", [], "fixed32")
    selection("explicit32", ["--compute.positionType=fixed32"], "fixed32")
    selection("explicit64", ["--compute.positionType=fixed64"], "fixed64")
    selection("abbreviated64", ["--compute.positionT=fixed64"], "fixed64")
    selection("spaced64", ["--compute.positionType", "fixed64"], "fixed64")
    selection("direct_companion", [], "fixed64",
              executable.with_name("cosmictiger_fixed64"))
    selection("companion32", ["--compute.positionType=fixed32"], "fixed32",
              executable.with_name("cosmictiger_fixed64"))
    config = workdir / "position.ini"
    config.write_text("[compute]\npositionType = fixed64\n")
    selection("ini64", [f"--configuration={config}"], "fixed64")
    selection("abbreviated_config", [f"--config={config}"], "fixed64")
    selection("cli_precedence", [f"--configuration={config}",
                                 "--compute.positionType=fixed32"], "fixed32")
    config.write_text("compute.positionType = fixed32\n")
    selection("cli64_precedence", ["--configuration", str(config),
                                   "--compute.positionType=fixed64"], "fixed64")
    alias = workdir / "application_alias"
    alias.unlink(missing_ok=True)
    alias.symlink_to(executable)
    selection("symlink64", ["--compute.positionType=fixed64"], "fixed64", alias)
    invalid = prepareCase(workdir, "invalid")
    output = runSimulation([str(executable), *common, "--compute.positionType=float64"],
                           invalid, "run.log", expectSuccess=False)
    if "compute.positionType must be fixed32 or fixed64" not in output:
        raise RuntimeError("invalid position selection lacked its diagnostic")

    # Exercise a complete driver checkpoint in addition to the exact raw-word
    # round trips above, then reject changing representation during restart.
    baseline = prepareCase(workdir, "evolution64")
    evolution = [str(executable), *common, "--run.scenario=plummer",
                 "--run.endTime=0.00001", "--run.numberOfSteps=2",
                 "--checkpoint.intervalSeconds=1000000000"]
    runSimulation([*evolution, "--compute.positionType=fixed64", "--run.maxIterations=1"],
                  baseline, "run.log")
    checkpoints = [path for path in baseline.glob("checkpoint.*")
                   if path.is_dir() and re.fullmatch(r"checkpoint\.[0-9]+", path.name)]
    if not checkpoints:
        raise RuntimeError("fixed64 evolution did not publish a checkpoint")
    checkpoint = max(checkpoints, key=lambda path: int(path.name.split(".")[1]))
    iteration = int(checkpoint.name.split(".")[1])
    checkManifest(checkpoint, iteration, scenario="plummer", endTime=0.00001)
    payload = (checkpoint / "checkpoint.0.dat").read_bytes()
    if checkpoint_position_bytes(payload) != 8:
        raise RuntimeError("fixed64 checkpoint omitted its representation flag")
    for kind, success in (("fixed64", True), ("fixed32", False)):
        directory = prepareCase(workdir, "restart_" + kind, checkpoint)
        output = runSimulation([*evolution, f"--compute.positionType={kind}",
                                f"--checkpoint.readIndex={iteration}", "--run.maxIterations=0"],
                               directory, "run.log", expectSuccess=success)
        if success and "Done reading checkpoint" not in output:
            raise RuntimeError("fixed64 checkpoint was not loaded")
        if not success and "particle layout" not in output:
            raise RuntimeError("position-width mismatch lacked a layout diagnostic")
    # Exercise the SPH analysis reader on real fixed64 data, including
    # coordinates below Q0.32.
    hydro = prepareCase(workdir, "hydro64")
    hydro_options = [option for option in common if not option.startswith(
        ("--hydro.enabled=", "--initial.particlesPerDimension="))]
    runSimulation([str(executable), *hydro_options, "--compute.positionType=fixed64",
                   "--hydro.enabled=true",
                   "--initial.particlesPerDimension=8", "--run.scenario=sod",
                   "--run.endTime=0.00001", "--run.numberOfSteps=2", "--run.maxIterations=1",
                   "--checkpoint.intervalSeconds=1000000000"], hydro, "run.log")
    hydro_checkpoint = hydro / "checkpoint.1"
    checkManifest(hydro_checkpoint, 1, scenario="sod", endTime=0.00001, gravity=False)
    if checkpoint_position_bytes((hydro_checkpoint / "checkpoint.0.dat").read_bytes()) != 8:
        raise RuntimeError("fixed64 hydro checkpoint omitted its representation flag")
    payload = hydro_particle_payload(hydro_checkpoint)
    candidates = [(payload.find(struct.pack("=7I", 0x43545350, version, size, 8, 12, 3, 4)), size)
                  for size in (4, 8) for version in (2, 4, 6)]
    sph_offset, integer_bytes = max(candidates)
    if sph_offset < 0:
        raise RuntimeError("fixed64 hydro checkpoint reader could not locate the SPH header")
    count = int.from_bytes(payload[:integer_bytes], sys.byteorder, signed=True)
    raw_positions = struct.unpack_from(f"={3 * count}Q", payload, 2 * integer_bytes)
    if not any(word & 0xFFFFFFFF for word in raw_positions):
        raise RuntimeError("fixed64 hydro checkpoint reader lost position precision")
    mislabeled = prepareCase(workdir, "hydro_wrong_width", hydro_checkpoint) / hydro_checkpoint.name
    rank_file = mislabeled / "checkpoint.0.dat"
    payload = bytearray(rank_file.read_bytes())
    flags = int.from_bytes(payload[12:16], sys.byteorder) & ~(1 << 15)
    payload[12:16] = flags.to_bytes(4, sys.byteorder)
    rank_file.write_bytes(payload)
    try:
        hydro_particle_payload(mislabeled)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("fixed64 hydro payload was silently parsed as fixed32")
    print("Position selection, raw checkpoint bits, fixed64 evolution/restart, hydro readers, "
          "and width rejection passed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
