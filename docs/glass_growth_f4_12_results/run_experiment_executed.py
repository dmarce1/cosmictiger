#!/usr/bin/env python3
"""Repeat the glass/lattice campaign with Ferrers exponent four and count twelve."""
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess

base = Path(__file__).resolve().parent
root = base.parents[3]
executable = root / "build/release/cosmictiger"
command = [str(executable), "--run.test=linearGrowth",
           "--initial.particlesPerDimension=8", "--initial.closePack=false",
           "--initial.useGlass=false", "--hydro.enabled=false", "--compute.cuda=true",
           "--gravity.softeningMode=adaptive", "--gravity.kernelFamily=ferrers",
           "--gravity.kernelDegree=8", "--gravity.neighborNumber=12",
           "--gravity.openingAngle=0.1", "--hpx:threads=4", "--hpx:bind=none"]
modes = "1,0,0;2,0,0;3,0,0;0,1,0;0,2,0;0,3,0;0,0,1;0,0,2;0,0,3;1,1,0;1,1,1;2,2,1;2,1,2;1,2,2"

def run_load(load):
    cases = [(f"adaptive_{load}", "requests_8.csv", True),
             (f"control_16_{load}", "requests_16.csv", False),
             (f"control_half_{load}", "requests_half.csv", False)]
    for name, request, response in cases:
        work = base / name
        work.mkdir()
        settings = {"COSMICTIGER_LINEAR_GROWTH_PAIRED": "1",
                    "COSMICTIGER_LINEAR_GROWTH_EVOLVE_FILE": str(base / request),
                    "COSMICTIGER_LINEAR_GROWTH_MODES": modes,
                    "COSMICTIGER_LINEAR_GROWTH_AMPLITUDES": "0.01,0.005"}
        if not response:
            settings["COSMICTIGER_LINEAR_GROWTH_SKIP_RESPONSE"] = "1"
        if load == "glass":
            settings["COSMICTIGER_LINEAR_GROWTH_LOAD_FILE"] = str(base / "glass.txt")
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("COSMICTIGER_LINEAR_GROWTH_")}
        environment.update(settings)
        record = dict(command=command, environment=settings, directory=str(work),
                      started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (work / "command.json").write_text(json.dumps(record, indent=2) + "\n")
        with (work / "run.log").open("w") as log:
            with subprocess.Popen(command, cwd=work, env=environment, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True) as process:
                try:
                    return_code = process.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise RuntimeError(f"Timed out: {work}")
        record.update(return_code=return_code,
                      finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (work / "command.json").write_text(json.dumps(record, indent=2) + "\n")
        print(name, "exit", return_code, flush=True)
        if return_code:
            raise RuntimeError(f"Failed: {work}")

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(run_load, load) for load in ("glass", "lattice")]
    for future in concurrent.futures.as_completed(futures):
        future.result()
