#!/usr/bin/env python3
"""Run the staged F2/24 glass IC correction measurements and lattice comparison."""

import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess


def main():
    archive = Path(__file__).resolve().parent
    root = archive.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("model", "calibrated"))
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--executable", type=Path, default=root / "build/release/cosmictiger")
    arguments = parser.parse_args()
    base = arguments.work_root.resolve()
    base.mkdir(parents=True, exist_ok=True)
    executable = arguments.executable.resolve(strict=True)
    command = [str(executable), "--run.test=linearGrowth",
               "--initial.particlesPerDimension=8", "--initial.closePack=false",
               "--initial.useGlass=false", "--hydro.enabled=false", "--compute.cuda=true",
               "--gravity.softeningMode=adaptive", "--gravity.kernelFamily=ferrers",
               "--gravity.kernelDegree=4", "--gravity.neighborNumber=24",
               "--gravity.openingAngle=0.1", "--hpx:threads=4", "--hpx:bind=none"]
    cases = ({"model": [("model_glass", "requests_model.csv", True),
                         ("corrected_lattice", "requests_lattice.csv", False)],
              "calibrated": [("calibrated_glass", "requests_calibrated.csv", True),
                              ("control_16_glass", "requests_control_16.csv", True),
                              ("control_half_glass", "requests_control_half.csv", True)]}
             [arguments.phase])
    for name, request, glass in cases:
        if (base / name).exists():
            raise FileExistsError(f"Refusing to overwrite existing run: {base / name}")
        if not (archive / request).is_file():
            raise FileNotFoundError(archive / request)
    shutil.copy2(archive / "glass.txt", base / "glass.txt")

    def run(case):
        name, request, glass = case
        work = base / name
        work.mkdir()
        shutil.copy2(archive / request, work / "requests.csv")
        settings = {"COSMICTIGER_LINEAR_GROWTH_PAIRED": "1",
                    "COSMICTIGER_LINEAR_GROWTH_SKIP_RESPONSE": "1",
                    "COSMICTIGER_LINEAR_GROWTH_EVOLVE_FILE": str(work / "requests.csv")}
        if glass:
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
        futures = [executor.submit(run, case) for case in cases]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
