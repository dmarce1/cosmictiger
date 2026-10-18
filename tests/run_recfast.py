#!/usr/bin/env python3
"""Exercise the built RECFAST executable and the production table reader."""

import argparse
import math
from pathlib import Path
import subprocess


# Frozen RECFAST 1.5.2 reference histories, generated from the pre-fix source
# with gfortran -O2 -ffixed-line-length-none so the full H(z) equation is read.
# These are regression values, not an independent accuracy claim for RECFAST.
# The second model is the upstream README example:
# https://www.astro.ubc.ca/people/scott/recfast.readme
CASES = {
    "fiducial": (
        "0.049389 0.26503 0.685581\n67.32 2.7255 0.2454006",
        {9990: 1.1637701, 3000: 1.0817954, 1500: 0.95491475,
         1100: 0.144875, 1000: 0.048746363, 500: 0.00068398028,
         100: 0.00027258145, 50: 0.00023856637, 0: 0.00017376092},
    ),
    "documented": (
        "0.04 0.20 0.76\n70 2.725 0.25",
        {9990: 1.1678627, 3000: 1.0838507, 1500: 0.95919325,
         1100: 0.15016408, 1000: 0.050829628, 500: 0.0007244921,
         100: 0.00028891125, 50: 0.00025324353, 0: 0.00018577978},
    ),
}


def runCase(executable, readers, workdir, name, parameters, reference):
    caseDir = workdir / name
    caseDir.mkdir(parents=True, exist_ok=True)
    output = caseDir / "history.dat"
    output.unlink(missing_ok=True)
    completed = subprocess.run(
        [str(executable)], input=f"history.dat\n{parameters}\n1\n6\n",
        text=True, capture_output=True, cwd=caseDir, timeout=30,
    )
    (caseDir / "recfast.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"RECFAST failed for {name}: {completed.stderr}")
    lines = output.read_text().splitlines()
    if len(lines) != 1001 or lines[0].split() != ["z", "x_e"]:
        raise AssertionError(f"Unexpected RECFAST table shape/header for {name}")
    table = {}
    for index, line in enumerate(lines[1:]):
        redshift, electrons = map(float, line.split())
        if redshift != 9990 - 10 * index:
            raise AssertionError(f"Invalid redshift grid at row {index}")
        if not math.isfinite(electrons) or not 0.0 <= electrons <= 2.0:
            raise AssertionError(f"Invalid electron abundance at z={redshift}")
        table[int(redshift)] = electrons
    for redshift, expected in reference.items():
        actual = table[redshift]
        if not math.isclose(actual, expected, rel_tol=2.0e-5, abs_tol=1.0e-10):
            raise AssertionError(
                f"{name}: xe(z={redshift})={actual:.10g}; expected {expected:.10g}"
            )
    for reader in readers:
        subprocess.run([str(reader), str(output)], check=True, timeout=15)
    print(f"{name}: 1000 rows, {len(reference)} reference abundances, {len(readers)} production readers passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--reader", type=Path, nargs="+", required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    arguments = parser.parse_args()
    for name, (parameters, reference) in CASES.items():
        runCase(arguments.executable.resolve(),
                [reader.resolve() for reader in arguments.reader],
                arguments.workdir.resolve(), name, parameters, reference)


if __name__ == "__main__":
    main()
