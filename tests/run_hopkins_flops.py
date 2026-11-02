# CosmicTiger - Hopkins precision accounting regression; BSL-1.0.

"""Exercise MFM/MFV accounting, progress output, and normal profile output."""

import argparse
import math
import os
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--time-integrator", choices=("traditional", "global"), required=True)
    args = parser.parse_args()
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for method in ("mfm", "mfv"):
        directory = (args.workdir / method).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        profile = directory / "profile.txt"
        profile.unlink(missing_ok=True)
        command = [str(args.executable.resolve()), "--run.scenario=gresho",
                   "--hydro.enabled=true", f"--hydro.method={method}",
                   "--gravity.enabled=false", "--chemistry.enabled=false",
                   "--starFormation.enabled=false", f"--run.timeIntegrator={args.time_integrator}",
                   "--initial.particlesPerDimension=8", "--compute.cuda=false",
                   "--run.verbose=true",
                   "--run.maxIterations=2", "--run.endTime=0.001", "--run.numberOfSteps=2",
                   "--output.silo=false", f"--output.directory={directory}",
                   "--analysis.groups=false", "--analysis.power=false", "--analysis.views=false",
                   "--lightcone.enabled=false", "--checkpoint.intervalSeconds=0",
                   "--hpx:threads=2", "--hpx:bind=none"]
        result = subprocess.run(command, cwd=directory, env=environment, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
        (directory / "run.log").write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(f"{method} failed ({result.returncode}):\n{result.stdout}")
        if re.search(r"^\s*hydro\.viscosityAlpha\s*:", result.stdout, re.MULTILINE):
            raise RuntimeError(f"{method} printed the unused SPH viscosity alpha")
        if not all(column in result.stdout for column in ("FP32 TF/s", "FP64 TF/s", "Mixed TF/s")):
            raise RuntimeError(f"{method} omitted the precision rate columns")
        rows = [line.split() for line in result.stdout.splitlines()
                if re.match(r"^\s*\d+\s+[\d.eE+-]+\s+[\d.eE+-]+\s+[\d.]+\s+\d+/", line)]
        if len(rows) != 2:
            raise RuntimeError(f"{method} expected two progress rows, found {len(rows)}")
        for row in rows:
            single, double, mixed, total = map(float, row[-4:])
            if not all(math.isfinite(value) and value >= 0.0 for value in (single, double, mixed, total)):
                raise RuntimeError(f"{method} invalid precision rates: {row[-4:]}")
            if mixed != 0.0:
                raise RuntimeError(f"{method} submitted operations without an arithmetic precision")
            if double <= 0.0:
                raise RuntimeError(f"{method} did not account for its dominant FP64 hydro work")
            if not math.isclose(single + double + mixed, total, rel_tol=0.002):
                raise RuntimeError(f"{method} precision rates do not sum to the total")
        if not profile.is_file() or "hydro" not in profile.read_text():
            raise RuntimeError(f"{method} did not write its completed phase profile")
    print("MFM/MFV FP64 accounting, rate totals, and profile.txt passed")


if __name__ == "__main__":
    main()
