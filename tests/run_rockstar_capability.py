# CosmicTiger - ROCKSTAR build capability regression; BSL-1.0.

"""Check ROCKSTAR help, disabled startup metadata, and unavailable option errors."""

import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile

from application_selection import cpu_backend_options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--enabled", action="store_true")
    args = parser.parse_args()
    base = args.workdir.resolve()
    base.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run_", dir=base))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1")
    command = [str(args.executable.resolve()), "--hpx:threads=2", "--hpx:bind=none"]
    backends = cpu_backend_options()

    def run(name, options, success=True):
        workdir = directory / name
        workdir.mkdir()
        arguments = [f"--{key}" if value is None else f"--{key}={value}"
                     for key, value in options.items()]
        log = workdir / "run.log"
        try:
            result = subprocess.run(command + arguments, cwd=workdir, env=environment,
                                    text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=90)
        except subprocess.TimeoutExpired as error:
            output = error.stdout or ""
            log.write_text(output.decode(errors="replace") if isinstance(output, bytes) else output)
            raise RuntimeError(f"{name}: timed out; see {log}") from error
        log.write_text(result.stdout)
        if (result.returncode == 0) != success:
            raise RuntimeError(f"{name}: unexpected exit {result.returncode}; see {log}\n"
                               + result.stdout[-4000:])
        return result.stdout

    helpText = run("help", {**backends, "help": None})
    expected = "enabled" if args.enabled else "disabled"
    if not re.search(r"^\s*ROCKSTAR and lightcones:\s*" + expected + r"\s*$",
                     helpText, re.MULTILINE):
        raise RuntimeError(f"help did not report ROCKSTAR and lightcones {expected}")
    if args.enabled:
        print("ROCKSTAR capability help passed: enabled")
        return

    def choices(label):
        match = re.search(r"^\s*" + re.escape(label) + r":([^\n]*)", helpText, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"help omitted compiled {label}")
        return match[1].split()

    solvers = choices("Gravity solvers")
    far = next((value for value in ("pmmm", "none") if value in solvers), None)
    if far is None:
        raise RuntimeError("no compiled gravity solver supports the startup fixture")
    if "hierarchical" not in choices("Time integrators"):
        raise RuntimeError("the sole integration scheme is unavailable in the startup fixture")
    options = {
        **backends,
        "run.scenario": "hydroWave" if far == "none" else "plummer",
        "run.maxIterations": "0",
        "run.numberOfOutputs": "2",
        "run.verbose": "false",
        "gravity.enabled": "false" if far == "none" else "true",
        "hydro.enabled": "true" if far == "none" else "false",
        "chemistry.enabled": "false",
        "starFormation.enabled": "false",
        "run.darkMatterParticleCount": "512", "run.gasParticleCount": "512",
        "analysis.groups": "false",
        "analysis.views": "false",
        "analysis.tracers": "false",
        "analysis.power": "false",
        "lightcone.enabled": "false",
        "output.silo": "false",
        "output.directory": ".",
        "checkpoint.intervalSeconds": "0",
    }
    if far == "pmmm":
        near = choices("PMMM near solvers")
        options["gravity.meshSize"] = "8"

    startup = run("startup", options)
    if not re.search(r"^build\.rockstar\s*:\s*F\s*$", startup, re.MULTILINE):
        raise RuntimeError("startup did not report build.rockstar=false")
    for key in ("analysis.groups", "analysis.groups.cuda", "lightcone.enabled", "lightcone.cuda"):
        output = run(key.replace(".", "_"), {**options, key: "true"}, success=False)
        if "COSMICTIGER_ENABLE_ROCKSTAR=ON" not in output:
            raise RuntimeError(f"{key}: missing guidance to enable COSMICTIGER_ENABLE_ROCKSTAR")
    print("ROCKSTAR disabled capability passed: help, startup metadata, and four option errors")


if __name__ == "__main__":
    main()
