# CosmicTiger - position backend selection and checkpoint regression; BSL-1.0.

"""Exercise configured position storage, CLI/INI checks, and native checkpoints."""

import argparse
from pathlib import Path
import re
import subprocess
import struct
import sys

from checkpoint_format import decode_checkpoint
from run_checkpoint import checkManifest, checkpoint_position_bytes, fnv1a64, prepareCase, runSimulation
from run_scenarios import hydro_particle_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--near-method", choices=("pp", "fmm"), default="fmm")
    parser.add_argument("--order", type=int)
    parser.add_argument("--time-integrator", choices=("traditional", "hierarchical", "global"), required=True)
    parser.add_argument("--without-sph", action="store_true")
    parser.add_argument("--position-type", choices=("fixed32", "fixed64"), required=True)
    args = parser.parse_args()
    executable = args.executable.resolve()
    selected = args.position_type
    opposite = "fixed64" if selected == "fixed32" else "fixed32"
    bits = 32 if selected == "fixed32" else 64
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    common = ["--hpx:threads=2", "--hpx:bind=none", "--gravity.cuda=false",
              "--hydro.enabled=false", "--chemistry.enabled=false",
              "--run.darkMatterParticleCount=64", "--run.gasParticleCount=64", "--analysis.groups=false",
              "--analysis.tracers=false", "--analysis.views=false", "--analysis.power=false",
              "--lightcone.enabled=false", "--output.silo=false", "--output.directory=."]
    common += [f"--run.timeIntegrator={args.time_integrator}", "--gravity.meshSize=8",
               f"--gravity.method.near={args.near_method}"]
    if args.order is not None:
        common.append(f"--gravity.expansionDegree={args.order - 1}")
    metadata = ["--run.test=checkpointState"]

    def selection(name, options, program=executable):
        directory = prepareCase(workdir, name)
        output = runSimulation([str(program), *common, *metadata, *options], directory, "run.log")
        if not re.search(r"^compute\.positionType\s*:\s*" + selected + r"\s*$",
                         output, re.MULTILINE):
            raise RuntimeError(f"{name}: incorrect selected position representation")
        buildSettings = {"run.timeIntegrator": args.time_integrator,
                         "gravity.method.near": args.near_method}
        if args.order is not None:
            buildSettings["gravity.expansionDegree"] = str(args.order - 1)
        for setting, expectedValue in buildSettings.items():
            if not re.search(r"^" + re.escape(setting) + r"\s*:\s*" + expectedValue + r"\s*$",
                             output, re.MULTILINE):
                raise RuntimeError(f"{name}: position backend changed selected {setting}")
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
    selection("spaced", ["--compute.positionType", selected])
    rejection("opposite", [f"--compute.positionType={opposite}"])
    rejection("opposite_spaced", ["--compute.positionType", opposite])
    config = workdir / "position.ini"
    config.write_text(f"[compute]\npositionType = {selected}\n")
    selection("ini", [f"--configuration={config}"])
    # The application requires exact option names. Its early position-ABI
    # parser must use the same policy instead of accepting abbreviations.
    for name, option in (("abbreviated", f"--compute.positionT={selected}"),
                         ("opposite_abbreviated", f"--compute.positionT={opposite}"),
                         ("abbreviated_config", f"--config={config}")):
        directory = prepareCase(workdir, name)
        output = runSimulation([str(executable), *common, option], directory,
                               "run.log", expectSuccess=False)
        if "unrecognized option" not in output or option not in output:
            raise RuntimeError(f"{name}: abbreviated option lacked its rejection diagnostic")
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
    if iteration != 1:
        raise RuntimeError("run.maxIterations=1 did not stop after exactly one iteration")
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
    restarted_checkpoints = [path.name for path in directory.glob("checkpoint.*")
                            if path.is_dir() and re.fullmatch(r"checkpoint\.[0-9]+", path.name)]
    if restarted_checkpoints != [checkpoint.name]:
        raise RuntimeError("run.maxIterations=0 advanced the restarted checkpoint")
    if (directory / checkpoint.name / "checkpoint.0.dat").read_bytes() != payload:
        raise RuntimeError("zero-iteration restart changed the saved particle state")
    progress = directory / "progress.txt"
    if progress.exists() and progress.read_text().strip():
        raise RuntimeError("zero-iteration restart produced an evolution row")
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
        ("--hydro.enabled=", "--run.darkMatterParticleCount=", "--run.gasParticleCount="))]
    hierarchicalHydro = args.time_integrator == "hierarchical"
    # SPH has no self-gravity coupling in the PMMM solver: hydro runs are
    # gravity free and cannot use the gravity-only hierarchical integrator.
    hydroSupported = not args.without_sph and not hierarchicalHydro
    hydro_options.append("--gravity.enabled=false")
    output = runSimulation([str(executable), *hydro_options, f"--compute.positionType={selected}",
                            "--hydro.enabled=true",
                            "--run.darkMatterParticleCount=512", "--run.gasParticleCount=512", "--run.scenario=sod",
                            "--run.endTime=0.00001", "--run.numberOfSteps=2", "--run.maxIterations=1",
                            "--checkpoint.intervalSeconds=1000000000"], hydro, "run.log",
                           expectSuccess=hydroSupported)
    if args.without_sph:
        if "COSMICTIGER_ENABLE_SPH=ON" not in output:
            raise RuntimeError("SPH rejection lacked SPH build guidance")
        print(f"{selected} selection, raw checkpoint bits, evolution/restart, width rejection, "
              "and disabled SPH rejection passed")
        return 0
    if not hydroSupported:
        if "hierarchical SPH requires FMM gravity" not in output:
            raise RuntimeError("SPH rejection lacked integrator build guidance")
        print(f"{selected} selection, raw checkpoint bits, evolution/restart, width rejection, "
              "and non-FMM hierarchical SPH rejection passed")
        return 0
    hydro_checkpoint = hydro / "checkpoint.1"
    checkManifest(hydro_checkpoint, 1, scenario="sod", endTime=0.00001, gravity=hierarchicalHydro)
    if checkpoint_position_bytes((hydro_checkpoint / "checkpoint.0.dat").read_bytes()) != bits // 8:
        raise RuntimeError("hydro checkpoint used the wrong position representation")
    state = decode_checkpoint((hydro_checkpoint / "checkpoint.0.dat").read_bytes())
    if bits == 64 and not any(word & 0xFFFFFFFF for axis in state.positions for word in axis):
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
