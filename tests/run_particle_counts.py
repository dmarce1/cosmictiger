"""Check rounded populations, independent species, and restart metadata in production."""
import argparse
import math
import os
from pathlib import Path
import re
import shutil
import subprocess

from application_selection import application_arguments, cpu_backend_options
from checkpoint_format import decode_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    args = parser.parse_args()
    executable = args.executable.resolve()
    base = args.workdir.resolve()
    base.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1")

    def run(name, scenario, dark_matter, gas, expected_dm, expected_gas, close_pack=False):
        directory = base / name
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir()
        options = dict(cpu_backend_options(), **{
            "run.scenario": scenario, "run.maxIterations": 1,
            "run.endTime": 0.000001, "run.numberOfSteps": 64,
            "run.darkMatterParticleCount": dark_matter,
            "run.gasParticleCount": gas, "initial.closePack": str(close_pack).lower(),
            "hydro.enabled": str(gas != 0).lower(), "hydro.neighborNumber": 42,
            "scenario.disc.centralAcceleration": 1 if scenario == "disc" else 0,
            "cosmology.sigma8": 0, "analysis.groups": "false", "analysis.views": "false",
            "analysis.power": "false", "analysis.tracers": "false", "lightcone.enabled": "false",
            "output.silo": "false", "output.directory": ".", "checkpoint.intervalSeconds": 0,
            "chemistry.enabled": "false", "starFormation.enabled": "false",
        })
        command = [str(executable), "--hpx:threads=2", "--hpx:bind=none"]
        command += application_arguments(executable, far="fmm", integrator="traditional", hydro=gas != 0)
        command += [f"--{key}={value}" for key, value in options.items()]
        result = subprocess.run(command, cwd=directory, env=environment, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        (directory / "run.log").write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {directory / 'run.log'}")
        files = list(directory.glob("checkpoint.[0-9]*/checkpoint.0.dat"))
        if not files:
            raise RuntimeError(f"{name} did not write a checkpoint")
        checkpoint = max(files, key=lambda path: int(path.parent.name.split(".")[1]))
        state = decode_checkpoint(checkpoint.read_bytes())
        actual = state.count - state.gas_count, state.gas_count
        if actual[0] != expected_dm or (expected_gas is not None and actual[1] != expected_gas):
            raise RuntimeError(f"{name}: populations {actual}, expected {(expected_dm, expected_gas)}")
        if actual[0] < dark_matter or actual[1] < gas:
            raise RuntimeError(f"{name}: rounded population {actual} undershot the request")
        if gas and scenario == "cosmology":
            dm_mass, gas_mass = state.header.physics[:2]
            dm_fraction = expected_dm * dm_mass / (expected_dm * dm_mass + expected_gas * gas_mass)
            def setting(key):
                return float(re.search(r"^" + re.escape(key) + r"\s*:\s*(\S+)", result.stdout, re.M)[1])
            if expected_dm and not math.isclose(dm_fraction, setting("cosmology.omegaC") /
                    (setting("cosmology.omegaC") + setting("cosmology.omegaB")), rel_tol=1e-6):
                raise RuntimeError(f"{name}: unequal counts changed the cosmic mass fractions")
            if expected_dm != expected_gas and state.initial_counts != (expected_dm, expected_gas):
                raise RuntimeError(f"{name}: checkpoint lost independent initial counts")
        print(f"{name}: dark matter={actual[0]}, gas={actual[1]}")
        return command, directory

    run("cubic_999", "cosmology", 999, 0, 1000, 0)
    run("cubic_1001", "cosmology", 1001, 0, 1331, 0)
    run("plummer_999", "plummer", 999, 0, 999, 0)
    run("bcc_999", "cosmology", 999, 0, 1024, 0, True)
    command, directory = run("independent", "cosmology", 999, 511, 1000, 512)
    # Exercise the flagged count extension through algorithm selection and restart.
    restarted = subprocess.run(command + ["--checkpoint.readIndex=1"], cwd=directory,
                               env=environment, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=120)
    (directory / "restart.log").write_text(restarted.stdout)
    if restarted.returncode:
        raise RuntimeError(f"independent restart failed; see {directory / 'restart.log'}")
    run("gas_only_cosmology", "cosmology", 0, 999, 0, 1000)
    run("fcc_999", "gresho", 0, 999, 0, 1372, True)
    run("sod_577", "sod", 0, 577, 0, 1944)
    run("disc_999", "disc", 0, 999, 0, None)


if __name__ == "__main__":
    main()
