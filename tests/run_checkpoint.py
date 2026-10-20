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
import sys


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
    magic = int.from_bytes(payload[:8], byteorder=sys.byteorder)
    version = int.from_bytes(payload[8:12], byteorder=sys.byteorder)
    if magic != 0x4354535048303031 or version != 3:
        raise RuntimeError(f"unexpected checkpoint header: magic={magic:x}, version={version}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    args = parser.parse_args()
    executable = args.executable.resolve()
    baseWorkdir = args.workdir.resolve()
    baseWorkdir.mkdir(parents=True, exist_ok=True)

    options = [str(executable), "--hpx:threads=2", "--hpx:bind=none",
               "--initial.particlesPerDimension=8", "--hydro.enabled=false",
               "--compute.cuda=false", "--analysis.views=false",
               "--lightcone.enabled=false", "--analysis.power=false",
               "--analysis.groups=false", "--gravity.kernelDegree=8", "--output.directory=."]
    baseline = prepareCase(baseWorkdir, "baseline")
    # The first step crosses this redshift. Its checkpoint must keep a numbered
    # restart path, while the second step must produce a final checkpoint.
    (baseline / "checkpoint.txt").write_text("48.9\n")
    baselineOutput = runSimulation(options + ["--run.maxIterations=1",
                                              "--checkpoint.intervalSeconds=1000000000"],
                                   baseline, "run.log")
    if "Done writing checkpoint" not in baselineOutput:
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

    restart("current")

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
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
