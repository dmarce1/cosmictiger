# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# SPDX-License-Identifier: BSL-1.0

"""Check GADGET-4 time-bin merging in the hierarchical gravity driver.

A Plummer sphere on a coarse base step occupies rungs 0-3. The first three
coarse synchronizations would merge rung 0 into 1, 1 into 2 and 2 into 3 with
finer fractions of about 0.94, 0.71 and 0.39 of the particles on or above the
lowest rung. run.rungMergeFraction must merge exactly where that fraction is
exceeded: all three at 0 and at the GADGET-4 default 1/3, the first two at 0.5,
and none at 1.
"""

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess

from application_selection import application_arguments, cpu_backend_options

MERGE_PATTERN = re.compile(r"Merging rung (\d+) into rung (\d+): (\d+) of (\d+) particles on finer "
                           r"rungs exceed run\.rungMergeFraction = (\S+)")


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
        "run.numberOfSteps": "4",
        "run.endTime": "0.0016",
        "run.maxIterations": "400",
        "run.rungMergeFraction": fraction,
        "initial.particlesPerDimension": "12",
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
    command += application_arguments(executable, far="fmm", integrator="hierarchical")
    command += [f"--{key}={value}" for key, value in options.items()]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1")
    result = subprocess.run(command, cwd=directory, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=environment, timeout=300)
    (directory / "run.log").write_text(result.stdout)
    require(result.returncode == 0, f"{name} exited {result.returncode}; see {directory / 'run.log'}")
    merges = [(int(lower), int(upper), int(fine), int(active))
              for lower, upper, fine, active, _ in MERGE_PATTERN.findall(result.stdout)]
    for lower, upper, fine, active in merges:
        require(upper == lower + 1, f"{name} merged rung {lower} into {upper}")
        require(fine > float(fraction) * active,
                f"{name} merged rung {lower} with {fine} of {active} finer particles")
    return merges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    args = parser.parse_args()
    executable = args.executable.resolve()
    base = args.workdir.resolve()

    default = run_case(executable, base, "gadget4_third", "0.3333333333333333")
    require([merge[0] for merge in default] == [0, 1, 2],
            f"default fraction merged rungs {default}, expected 0, 1 and 2")
    require(run_case(executable, base, "always", "0") == default,
            "fraction 0 did not reproduce the default merges")
    half = run_case(executable, base, "half", "0.5")
    require(half == default[:2], f"fraction 0.5 merged {half}, expected {default[:2]}")
    _, _, fine, active = default[2]
    require(not fine > 0.5 * active,
            f"rung 2 held {fine} of {active} finer particles, above the 0.5 threshold")
    require(run_case(executable, base, "never", "1") == [], "fraction 1 merged a rung")
    for lower, upper, fine, active in default:
        print(f"rung {lower} -> {upper}: {fine}/{active} = {fine / active:.3f} on finer rungs")
    print("hierarchical rung merging follows run.rungMergeFraction")


if __name__ == "__main__":
    main()
