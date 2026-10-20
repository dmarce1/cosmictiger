#!/usr/bin/env python3
"""Run bounded fixed-support glass diagnostics, without the cosmological driver."""

import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def main():
    archive = Path(__file__).resolve().parent
    root = archive.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("initial_glass", "response_glass", "evolution_glass",
                                         "control_16", "control_half", "radius_control"))
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--executable", type=Path, default=root / "build/release/cosmictiger")
    parser.add_argument("--threads", type=int, choices=(2, 4), default=4)
    args = parser.parse_args()
    base = args.work_root.resolve()
    base.mkdir(parents=True, exist_ok=True)
    work = base / args.case
    work.mkdir()
    shutil.copy2(archive / "glass.txt", base / "glass.txt")
    support = .05 if args.case == "radius_control" else .025
    command = [str(args.executable.resolve(strict=True)), "--run.test=linearGrowth",
               "--initial.particlesPerDimension=8", "--initial.closePack=false",
               "--initial.useGlass=false", "--hydro.enabled=false", "--compute.cuda=false",
               "--gravity.softeningMode=fixed", "--gravity.kernelFamily=ferrers",
               "--gravity.kernelDegree=4", f"--gravity.supportRadius={support}",
               "--gravity.openingAngle=0.1", f"--hpx:threads={args.threads}", "--hpx:bind=none"]
    settings = {"COSMICTIGER_LINEAR_GROWTH_LOAD_FILE": str(base / "glass.txt"),
                "COSMICTIGER_LINEAR_GROWTH_PARTICLE_OUTPUT": "1",
                "COSMICTIGER_LINEAR_GROWTH_SKIP_RESPONSE": "1"}
    if args.case == "response_glass":
        settings.update(COSMICTIGER_LINEAR_GROWTH_SKIP_RESPONSE="0",
                        COSMICTIGER_LINEAR_GROWTH_MODES="1,0,0;2,0,0;3,0,0;0,3,0;0,0,3;1,2,2;2,1,2;2,2,1",
                        COSMICTIGER_LINEAR_GROWTH_AMPLITUDES="0.002,0.001")
    request = {"evolution_glass": "requests_8.csv", "control_16": "requests_16.csv",
               "control_half": "requests_half.csv"}.get(args.case)
    if request:
        shutil.copy2(archive / request, work / "requests.csv")
        settings.update(COSMICTIGER_LINEAR_GROWTH_PAIRED="1",
                        COSMICTIGER_LINEAR_GROWTH_EVOLVE_FILE=str(work / "requests.csv"))
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("COSMICTIGER_LINEAR_GROWTH_")}
    environment.update(settings, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    record = dict(command=command, environment=settings, directory=str(work),
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (work / "command.json").write_text(json.dumps(record, indent=2) + "\n")
    start = time.monotonic()
    with (work / "run.log").open("w") as log:
        with subprocess.Popen(command, cwd=work, env=environment, stdout=log,
                              stderr=subprocess.STDOUT, start_new_session=True) as process:
            try:
                code = process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError(f"Diagnostic exceeded 30 minutes: {work}")
    record.update(return_code=code, elapsed_seconds=time.monotonic() - start,
                  finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (work / "command.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(dict(case=args.case, return_code=code,
                          elapsed_seconds=record["elapsed_seconds"])), flush=True)
    if code:
        raise RuntimeError(f"Diagnostic failed: {work}")


if __name__ == "__main__":
    main()
