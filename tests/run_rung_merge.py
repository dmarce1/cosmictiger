# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# SPDX-License-Identifier: BSL-1.0

"""Check the inclusive time-bin merge threshold in the hierarchical gravity driver.

A Plummer sphere on a coarse base step occupies rungs 0-3. The first three
coarse synchronizations would merge rung 0 into 1, 1 into 2 and 2 into 3 with
finer fractions of about 0.98, 0.94 and 0.63 of the particles on or above the
lowest rung. run.rungMergeFraction must merge exactly where that fraction is
met or exceeded: all three at 0 and at the default 1/3, the first two at 0.75,
and none at 1 for this fixture. The default also promotes equal-cost ties; the
paper describes strict cost saving.
"""

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess

from application_selection import application_arguments, cpu_backend_options

from checkpoint_format import read_prefix


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run_case(executable, base, name, fraction):
    directory = base / name
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    options = dict(cpu_backend_options(), **{
        "run.scenario": "plummer",
        "run.numberOfOutputs": "3",
        "run.endTime": "0.0016",
        "run.maxIterations": "400",
        "run.rungMergeFraction": fraction,
        "run.darkMatterParticleCount": "1728", "run.gasParticleCount": "1728",
        "analysis.groups": "false",
        "analysis.power": "false",
        "analysis.views": "false",
        "analysis.tracers": "false",
        "lightcone.enabled": "false",
        "output.silo": "false",
        "output.directory": ".",
        "checkpoint.intervalSeconds": "0",
    })
    command = [str(executable), "--hpx:threads=2", "--hpx:bind=none"]
    command += application_arguments(executable, integrator="hierarchical")
    command += [f"--{key}={value}" for key, value in options.items()]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1")
    result = subprocess.run(command, cwd=directory, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=environment, timeout=300)
    (directory / "run.log").write_text(result.stdout)
    require(result.returncode == 0, f"{name} exited {result.returncode}; see {directory / 'run.log'}")
    require("Merging rung" not in result.stdout, "rung merges should be silent")
    require("Coarse step:" not in result.stdout, "coarse-step banners should be omitted")
    headers = re.findall(r"^\s*Step\s+z\s+a\s+tau\s+dt\s+MinRung", result.stdout, re.MULTILINE)
    require(headers, f"{name}: progress column headings are missing")
    require(not re.search(r"\bSILO\b", result.stdout),
            "disabled SILO output should not be announced")
    # Observe the actual saved lowest rung, rather than a user-facing notice.
    checkpoints = sorted(directory.glob("checkpoint.*/checkpoint.0.dat"),
                         key=lambda path: int(path.parent.name.split(".")[1]))
    drivers = [read_prefix(path.read_bytes()).driver for path in checkpoints]
    require(drivers, "rung merge fixture produced no checkpoints")
    floor = 0
    merges = []
    for driver in drivers:
        if driver.min_rung > floor:
            merges.extend(range(floor, driver.min_rung))
        floor = driver.min_rung
    return merges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    args = parser.parse_args()
    executable = args.executable.resolve()
    base = args.workdir.resolve()

    default = run_case(executable, base, "gadget4_third", "0.3333333333333333")
    require(default == [0, 1, 2],
            f"default fraction merged rungs {default}, expected 0, 1 and 2")
    require(run_case(executable, base, "always", "0") == default,
            "fraction 0 did not reproduce the default merges")
    three_quarters = run_case(executable, base, "three_quarters", "0.75")
    require(three_quarters == default[:2],
            f"fraction 0.75 merged {three_quarters}, expected {default[:2]}")
    require(run_case(executable, base, "no_empty_floor", "1") == [],
            "fraction 1 merged a rung with particles on its lowest level")
    print("hierarchical rung merging follows run.rungMergeFraction")


if __name__ == "__main__":
    main()
