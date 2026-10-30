# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026  Dominic C. Marcello
#
# Distributed under the Boost Software License, Version 1.0.
# See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.

"""Exercise published checkpoint integrity and restart compatibility."""

import argparse
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import struct
import sys


def checkpoint_position_bytes(payload):
    """Read the position width from a current versioned checkpoint header."""
    if len(payload) < 16:
        raise RuntimeError("truncated checkpoint header")
    magic = int.from_bytes(payload[:8], byteorder=sys.byteorder)
    version = int.from_bytes(payload[8:12], byteorder=sys.byteorder)
    if magic != 0x4354535048303031 or version != 3:
        raise RuntimeError(f"unexpected checkpoint header: magic={magic:x}, version={version}")
    flags = int.from_bytes(payload[12:16], byteorder=sys.byteorder)
    return 8 if flags & (1 << 15) else 4


def fnv1a64(payload):
    checksum = 0xCBF29CE484222325
    for byte in payload:
        checksum = ((checksum ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return checksum


def runSimulation(command, workdir, logName, expectSuccess=True):
    logPath = workdir / logName
    with logPath.open("w") as log:
        with subprocess.Popen(command, cwd=workdir, text=True, stdout=log,
                              stderr=subprocess.STDOUT, start_new_session=True) as process:
            try:
                returnCode = process.wait(timeout=240)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError(f"simulation timed out; see {logPath}")
    output = logPath.read_text(errors="replace")
    if (returnCode == 0) != expectSuccess:
        raise RuntimeError(f"{logName} exited with {returnCode}; see {logPath}:\n"
                           + output[-8000:])
    if not expectSuccess and "Done reading checkpoint" in output:
        raise RuntimeError(f"{logName} loaded a checkpoint that should have been rejected")
    return output


def prepareCase(baseWorkdir, name, sourceCheckpoint=None):
    workdir = baseWorkdir / name
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    if sourceCheckpoint is not None:
        shutil.copytree(sourceCheckpoint, workdir / sourceCheckpoint.name)
    return workdir


def checkFreshOutput(options, baseWorkdir, baseline):
    """Reject occupied checkpoint output before changing any prior run files."""
    def snapshot(directory):
        result = {}
        for path in directory.rglob("*"):
            name = path.relative_to(directory).as_posix()
            if path.is_symlink():
                result[name] = ("symlink", os.readlink(path))
            elif path.is_file():
                result[name] = ("file", path.read_bytes(), path.stat().st_mtime_ns)
            else:
                result[name] = ("directory",)
        return result

    def reject(workdir, outputDirectory, command):
        before = snapshot(outputDirectory)
        output = runSimulation(command + ["--run.maxIterations=0",
                                           "--checkpoint.intervalSeconds=1000000000"],
                               workdir, "fresh_start.log", expectSuccess=False)
        if ("existing checkpoint" not in output.lower() or
                "--output.directory" not in output or "--checkpoint.readIndex" not in output):
            raise RuntimeError("fresh run lacked actionable checkpoint-output guidance")
        if snapshot(outputDirectory) != before:
            raise RuntimeError("rejected fresh run changed existing output or wrote simulation files")

    # Reuse the actual completed output, including progress and diagnostics.
    # Keep invocation logs/profiles in a separate working directory.
    workdir = prepareCase(baseWorkdir, "fresh_existing_output")
    command = [option for option in options if not option.startswith("--output.directory=")]
    reject(workdir, baseline, command + [f"--output.directory={baseline}"])
    for name, entry in (("incomplete_output", "checkpoint.999.incomplete"),
                        ("numbered_file_output", "checkpoint.999")):
        workdir = prepareCase(baseWorkdir, name)
        outputDirectory = workdir / "preserved output"
        outputDirectory.mkdir()
        checkpoint = outputDirectory / entry
        if entry.endswith(".incomplete"):
            checkpoint.mkdir()
            (checkpoint / "preserve.txt").write_text("unfinished checkpoint\n")
        else:
            checkpoint.write_text("occupied checkpoint path\n")
        (outputDirectory / "progress.txt").write_text("prior run progress\n")
        command = [option for option in options if not option.startswith("--output.directory=")]
        command += ["--output.directory=preserved output"]
        reject(workdir, outputDirectory, command)
    print("fresh output: existing checkpoints and staging paths rejected before output changes")


def checkManifest(checkpointDir, iteration, scenario="cosmology", endTime=0.0, gravity=True,
                  testFixture=False):
    manifest = checkpointDir / "checkpoint.meta"
    if not manifest.is_file():
        raise RuntimeError("completed checkpoint has no manifest")
    lines = manifest.read_text().splitlines()
    if len(lines) != (2 if testFixture else 3) or lines[0].split() != [
                                             "COSMICTIGER_CHECKPOINT", "1" if testFixture else "2", "3",
                                             str(iteration), "1"]:
        raise RuntimeError(f"unexpected checkpoint manifest: {lines}")
    rank, byteCount, checksumText = lines[1].split()
    if rank != "0" or not re.fullmatch(r"[0-9a-fA-F]{1,16}", checksumText):
        raise RuntimeError(f"invalid rank record in checkpoint manifest: {lines[1]}")
    payload = (checkpointDir / "checkpoint.0.dat").read_bytes()
    if int(byteCount) != len(payload) or int(checksumText, 16) != fnv1a64(payload):
        raise RuntimeError("checkpoint manifest size or checksum differs from rank file")
    if not testFixture:
        scenarioLabel, scenarioBytes, scenarioChecksum = lines[2].split()
        if scenarioLabel != "SCENARIO" or not re.fullmatch(r"[0-9a-fA-F]{1,16}", scenarioChecksum):
            raise RuntimeError(f"invalid scenario record in checkpoint manifest: {lines[2]}")
        scenarioPayload = (checkpointDir / "scenario.meta").read_bytes()
        if int(scenarioBytes) != len(scenarioPayload) or int(scenarioChecksum, 16) != fnv1a64(scenarioPayload):
            raise RuntimeError("checkpoint manifest size or checksum differs from scenario metadata")
        scenarioLines = scenarioPayload.decode().splitlines()
        if (len(scenarioLines) != 4 or scenarioLines[0] != "COSMICTIGER_SCENARIO 1" or
                scenarioLines[1] != scenario or float(scenarioLines[2]) != endTime or
                scenarioLines[3] != str(int(gravity))):
            raise RuntimeError(f"unexpected checkpoint scenario metadata: {scenarioLines}")
    checkpoint_position_bytes(payload)


def hydro_particle_payload(checkpoint, expected_count=None):
    """Read the all-gas physical payload, excluding native driver timing/padding."""
    data = (checkpoint / "checkpoint.0.dat").read_bytes()
    position_bytes = checkpoint_position_bytes(data)
    candidates = [data.find(struct.pack("=7I", 0x43545350, version, integer_bytes, 8, 12, 3, 4))
                  for integer_bytes in (4, 8) for version in (2, 4, 6, 8)]
    offset = max(candidates)
    if offset < 0:
        raise RuntimeError(f"missing versioned SPH payload in {checkpoint}")
    header = struct.unpack_from("=7I", data, offset)
    integer_bytes = header[2]
    if integer_bytes not in (4, 8) or header[5:] != (3, 4):
        raise RuntimeError(f"unsupported particle layout in {checkpoint}: {header}")
    count = int.from_bytes(data[offset + 28:offset + 28 + integer_bytes],
                           sys.byteorder, signed=True)
    if count <= 0 or (expected_count is not None and count != expected_count):
        raise RuntimeError(f"unexpected gas count in {checkpoint}: {count}")
    # Particle positions, velocities, rungs, species, and gas index precede
    # the gas header. The 38 bytes below are rates, density, flags, and six
    # predictor/gravity acceleration floats in visitSphCheckpointArrays().
    start = offset - 2 * integer_bytes - count * (3 * position_bytes + 14 + integer_bytes)
    # Price et al. (version 8) shares the Cullen-Dehnen divergence layout.
    viscosity_bytes = {2: 0, 4: 8, 6: 12, 8: 12}[header[1]]
    end = offset + 28 + integer_bytes + count * (
        header[3] + header[4] + integer_bytes + 38 + viscosity_bytes)
    if start < 0 or end > len(data):
        raise RuntimeError(f"truncated physical particle payload in {checkpoint}")
    counts = [int.from_bytes(data[start + i * integer_bytes:start + (i + 1) * integer_bytes],
                             sys.byteorder, signed=True) for i in range(2)]
    if counts != [count, count]:
        raise RuntimeError(f"restart fixture is not all gas: {counts}")
    return data[start:end]


def checkViscosityRestarts(executable, baseWorkdir):
    """Check every viscosity switch with and without the independent shear limiter."""
    configuration = Path(__file__).resolve().parents[1] / "configurations/hydro_wave.ini"
    common = [str(executable), f"--configuration={configuration}", "--hpx:threads=2",
              "--hpx:bind=none", "--initial.particlesPerDimension=8", "--compute.cuda=false",
              "--run.endTime=0.00001", "--checkpoint.intervalSeconds=0",
              "--output.directory=.", "--output.silo=false", "--analysis.views=false",
              "--analysis.power=false", "--analysis.groups=false", "--lightcone.enabled=false"]
    for model in ("none", "morrisAndMonaghan", "cullenAndDehnen", "priceEtAl"):
        for limiter in (False, True):
            name = f"viscosity_{model}_{int(limiter)}"
            options = common + [f"--hydro.viscosityEvolution={model}",
                                f"--hydro.balsaraSwitch={str(limiter).lower()}"]
            reference = prepareCase(baseWorkdir, name)
            runSimulation(options + ["--run.maxIterations=2"], reference, "run.log")
            checkpoint = reference / "checkpoint.1"
            if not checkpoint.is_dir():
                raise RuntimeError(f"{name} did not save a completed first step")
            resumed = prepareCase(baseWorkdir, name + "_restart", checkpoint)
            output = runSimulation(options + ["--run.maxIterations=1", "--checkpoint.readIndex=1"],
                                   resumed, "restart.log")
            if "Done reading checkpoint" not in output:
                raise RuntimeError(f"{name} did not load the checkpoint")
            if hydro_particle_payload(reference / "checkpoint.2", 512) != hydro_particle_payload(
                    resumed / "checkpoint.2", 512):
                raise RuntimeError(f"{name} restart changed particles, stage fields, or viscosity")
            checkManifest(resumed / "checkpoint.2", 2, scenario="hydroWave", endTime=1.e-5,
                          gravity=False)
            if not limiter:
                mutations = {"model": "--hydro.viscosityEvolution=" + (
                    "morrisAndMonaghan" if model == "none" else "none"),
                             "beta": "--hydro.viscosityBeta=3"}
                if model == "morrisAndMonaghan":
                    mutations.update({"floor": "--hydro.viscosityAlphaMin=0.2",
                                      "decay": "--hydro.viscosityDecay=0.3"})
                if model == "cullenAndDehnen":
                    mutations.update({"maximum": "--hydro.viscosityAlphaMax=3",
                                      "decay": "--hydro.viscosityDecayCD=0.1"})
                if model == "priceEtAl":
                    mutations.update({"maximum": "--hydro.viscosityAlphaMaxPrice=0.5",
                                      "decay": "--hydro.viscosityDecayPrice=0.2"})
                for label, changed in mutations.items():
                    directory = prepareCase(baseWorkdir, name + "_reject_" + label, checkpoint)
                    changed_key = changed.split("=", 1)[0] + "="
                    changed_options = [option for option in options
                                       if not option.startswith(changed_key)]
                    output = runSimulation(changed_options + ["--run.maxIterations=0",
                                           "--checkpoint.readIndex=1", changed],
                                           directory, "restart.log", expectSuccess=False)
                    if "checkpoint" not in output.lower():
                        raise RuntimeError(f"{name}/{label} lacked a checkpoint diagnostic")
            print(f"{name}: exact restart and viscosity compatibility passed")


def checkPressureOperatorRestarts(executable, baseWorkdir):
    """Restart each SPH pressure operator exactly and reject a restart with the other.

    Grad-H corrections are not checkpointed; the first density pass after the restart
    rebuilds them, so the resumed run must reproduce the uninterrupted payload bit for bit.
    """
    configuration = Path(__file__).resolve().parents[1] / "configurations/hydro_wave.ini"
    common = [str(executable), f"--configuration={configuration}", "--hpx:threads=2",
              "--hpx:bind=none", "--initial.particlesPerDimension=8", "--compute.cuda=false",
              "--run.endTime=0.00001", "--checkpoint.intervalSeconds=0",
              "--output.directory=.", "--output.silo=false", "--analysis.views=false",
              "--analysis.power=false", "--analysis.groups=false", "--lightcone.enabled=false"]
    for operator, other in (("variational", "monaghan"), ("monaghan", "variational")):
        name = f"pressure_operator_{operator}"
        options = common + [f"--hydro.pressureOperator={operator}"]
        reference = prepareCase(baseWorkdir, name)
        runSimulation(options + ["--run.maxIterations=2"], reference, "run.log")
        checkpoint = reference / "checkpoint.1"
        if not checkpoint.is_dir():
            raise RuntimeError(f"{name} did not save a completed first step")
        resumed = prepareCase(baseWorkdir, name + "_restart", checkpoint)
        output = runSimulation(options + ["--run.maxIterations=1", "--checkpoint.readIndex=1"],
                               resumed, "restart.log")
        if "Done reading checkpoint" not in output:
            raise RuntimeError(f"{name} did not load the checkpoint")
        if hydro_particle_payload(reference / "checkpoint.2", 512) != hydro_particle_payload(
                resumed / "checkpoint.2", 512):
            raise RuntimeError(f"{name} restart changed particles or stage fields")
        directory = prepareCase(baseWorkdir, name + "_reject", checkpoint)
        output = runSimulation(common + [f"--hydro.pressureOperator={other}",
                                         "--run.maxIterations=0", "--checkpoint.readIndex=1"],
                               directory, "restart.log", expectSuccess=False)
        if "pressure operator" not in output:
            raise RuntimeError(f"{name} restart with {other} lacked the operator diagnostic")
        print(f"{name}: exact restart and operator compatibility passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    sph_mode = parser.add_mutually_exclusive_group()
    sph_mode.add_argument("--viscosity-only", action="store_true")
    sph_mode.add_argument("--without-sph", action="store_true")
    parser.add_argument("--fresh-output-only", action="store_true")
    args = parser.parse_args()
    if args.viscosity_only and args.fresh_output_only:
        parser.error("--fresh-output-only cannot be combined with --viscosity-only")
    executable = args.executable.resolve()
    baseWorkdir = args.workdir.resolve()
    baseWorkdir.mkdir(parents=True, exist_ok=True)

    if args.viscosity_only:
        checkViscosityRestarts(executable, baseWorkdir)
        checkPressureOperatorRestarts(executable, baseWorkdir)
        return 0

    options = [str(executable), "--hpx:threads=2", "--hpx:bind=none",
               "--run.timeIntegrator=traditional", "--gravity.method.far=fmm",
               "--initial.particlesPerDimension=8", "--hydro.enabled=false",
               "--compute.cuda=false", "--analysis.views=false",
               "--lightcone.enabled=false", "--analysis.power=false",
               "--analysis.groups=false", "--gravity.kernelDegree=8", "--output.directory=.",
               # Checkpoint publication and integrity need no linear-power calculation.
               "--cosmology.sigma8=0"]
    baseline = prepareCase(baseWorkdir, "baseline")
    # The first step crosses this redshift. Its checkpoint must keep a numbered
    # restart path, while the second step must produce a final checkpoint.
    (baseline / "checkpoint.txt").write_text("48.9\n")
    (baseline / "checkpoint.z.42.0").write_text("unrelated redshift alias name\n")
    baselineOutput = runSimulation(options + ["--run.maxIterations=1",
                                              "--checkpoint.intervalSeconds=1000000000"],
                                   baseline, "run.log")
    if "Checkpoint saved:" not in baselineOutput:
        raise RuntimeError("normal exit did not finish a checkpoint")
    checkpoints = [path for path in baseline.glob("checkpoint.*")
                   if path.is_dir() and re.fullmatch(r"checkpoint\.[0-9]+", path.name)]
    if not checkpoints:
        raise RuntimeError("normal exit did not publish a numbered checkpoint")
    triggeredCheckpoint = baseline / "checkpoint.1"
    if triggeredCheckpoint not in checkpoints:
        raise RuntimeError("redshift-triggered checkpoint has no numbered restart path")
    checkpoint = max(checkpoints, key=lambda path: int(path.name.split(".")[1]))
    iteration = int(checkpoint.name.split(".")[1])
    if iteration <= 1:
        raise RuntimeError("normal exit did not publish a final checkpoint")
    if list(baseline.glob("checkpoint.*.incomplete")):
        raise RuntimeError("completed checkpoint left a staging directory")
    checkManifest(triggeredCheckpoint, 1)
    checkManifest(checkpoint, iteration)
    checkFreshOutput(options, baseWorkdir, baseline)

    def restart(name, mutate=None, expectSuccess=True):
        workdir = prepareCase(baseWorkdir, name, checkpoint)
        copiedCheckpoint = workdir / checkpoint.name
        if mutate is not None:
            mutate(copiedCheckpoint)
        output = runSimulation(options + [f"--checkpoint.readIndex={iteration}",
                                          "--run.maxIterations=0",
                                          "--checkpoint.intervalSeconds=1000000000"],
                               workdir, "restart.log", expectSuccess=expectSuccess)
        if expectSuccess and "Done reading checkpoint" not in output:
            raise RuntimeError(f"{name} did not load the checkpoint")
        if not expectSuccess and not re.search(r"checkpoint|manifest", output, re.IGNORECASE):
            raise RuntimeError(f"{name} failed without a checkpoint diagnostic")
        return output

    restart("current")
    if args.fresh_output_only:
        print("fresh output: new runs succeed and explicit checkpoint restarts remain supported")
        return 0

    def leaveOnlyStagingDirectory(checkpointDir):
        checkpointDir.rename(checkpointDir.with_name(checkpointDir.name + ".incomplete"))

    restart("staging_only", leaveOnlyStagingDirectory, expectSuccess=False)

    def removeManifest(checkpointDir):
        (checkpointDir / "checkpoint.meta").unlink()

    restart("missing_manifest", removeManifest, expectSuccess=False)

    def removeScenarioMetadata(checkpointDir):
        (checkpointDir / "scenario.meta").unlink()

    restart("missing_scenario", removeScenarioMetadata, expectSuccess=False)

    def corruptScenarioMetadata(checkpointDir):
        # Whitespace is semantically acceptable, but must invalidate the hash.
        path = checkpointDir / "scenario.meta"
        path.write_bytes(path.read_bytes() + b"\n")

    restart("corrupt_scenario", corruptScenarioMetadata, expectSuccess=False)

    def changeRankCount(checkpointDir):
        manifest = checkpointDir / "checkpoint.meta"
        lines = manifest.read_text().splitlines()
        fields = lines[0].split()
        fields[-1] = "2"
        lines[0] = " ".join(fields)
        manifest.write_text("\n".join(lines) + "\n")

    restart("wrong_rank_count", changeRankCount, expectSuccess=False)

    def corruptPayload(checkpointDir):
        rankFile = checkpointDir / "checkpoint.0.dat"
        payload = bytearray(rankFile.read_bytes())
        payload[len(payload) // 2] ^= 0x01
        rankFile.write_bytes(payload)

    restart("corrupt_payload", corruptPayload, expectSuccess=False)

    def truncatePayload(checkpointDir):
        rankFile = checkpointDir / "checkpoint.0.dat"
        payload = rankFile.read_bytes()
        rankFile.write_bytes(payload[:-16])

    restart("truncated_payload", truncatePayload, expectSuccess=False)

    def unsupportedVersion(checkpointDir):
        rankFile = checkpointDir / "checkpoint.0.dat"
        payload = bytearray(rankFile.read_bytes())
        payload[8:12] = (4).to_bytes(4, byteorder=sys.byteorder)
        rankFile.write_bytes(payload)
        manifest = checkpointDir / "checkpoint.meta"
        lines = manifest.read_text().splitlines()
        lines[1] = f"0 {len(payload)} {fnv1a64(payload):016x}"
        manifest.write_text("\n".join(lines) + "\n")

    restart("unsupported_version", unsupportedVersion, expectSuccess=False)

    def retiredParticleMesh(checkpointDir):
        rankFile = checkpointDir / "checkpoint.0.dat"
        payload = bytearray(rankFile.read_bytes())
        flags = int.from_bytes(payload[12:16], byteorder=sys.byteorder)
        payload[12:16] = (flags | (1 << 16)).to_bytes(4, byteorder=sys.byteorder)
        rankFile.write_bytes(payload)
        # Keep the manifest valid so this exercises solver compatibility,
        # rather than stopping at the earlier corruption check.
        manifest = checkpointDir / "checkpoint.meta"
        lines = manifest.read_text().splitlines()
        lines[1] = f"0 {len(payload)} {fnv1a64(payload):016x}"
        manifest.write_text("\n".join(lines) + "\n")

    output = restart("retired_pm", retiredParticleMesh, expectSuccess=False)
    if "Ordinary PM gravity checkpoints are no longer supported" not in output:
        raise RuntimeError("retired PM checkpoint was not rejected before reading its payload")

    def downgradeToVersionOne(checkpointDir):
        removeManifest(checkpointDir)
        rankFile = checkpointDir / "checkpoint.0.dat"
        with rankFile.open("r+b") as stream:
            stream.seek(8)
            stream.write((1).to_bytes(4, byteorder=sys.byteorder))

    # Retain the current scenario sidecar while checking legacy payload decoding.
    # Unlabelled old checkpoints are explicitly rejected by missing_scenario.
    restart("version_one", downgradeToVersionOne)
    print(f"checkpoint {iteration}: redshift and final publication, payload version 3, "
          "scenario manifest v2, restart, corruption rejection, and labelled version 1 decoding passed")
    if not args.without_sph:
        checkViscosityRestarts(executable, baseWorkdir)
        checkPressureOperatorRestarts(executable, baseWorkdir)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
