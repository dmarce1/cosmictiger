# CosmicTiger - SPH thermal formulation and storage integration regression; BSL-1.0.

"""Check entropy/energy selection, all thermal layouts, and coupled source fixtures."""

import argparse
import os
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--with-chemistry", action="store_true")
    parser.add_argument("--with-star-formation", action="store_true")
    parser.add_argument("--with-silo", action="store_true")
    args = parser.parse_args()
    if args.with_star_formation and not args.with_chemistry:
        parser.error("--with-star-formation requires --with-chemistry")
    executable = args.executable.resolve()
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    common = {
        "hydro.enabled": "true", "gravity.enabled": "false",
        "run.darkMatterParticleCount": "512", "run.gasParticleCount": "512",
        "chemistry.enabled": "false", "starFormation.enabled": "false",
        "analysis.groups": "false", "analysis.power": "false", "analysis.views": "false",
        "analysis.tracers": "false", "lightcone.enabled": "false", "output.silo": "false",
    }
    cudaKeys = ("gravity.cuda", "gravity.error.cuda", "hydro.cuda", "drift.cuda", "sort.cuda",
                "initial.cuda", "analysis.power.cuda", "analysis.groups.cuda",
                "analysis.tracers.cuda", "analysis.slice.cuda", "analysis.views.cuda",
                "lightcone.cuda", "chemistry.cuda", "starFormation.cuda")
    common.update({key: "false" for key in cudaKeys})
    completed = 0

    def runCase(name, test, options=None, formulation="densityEntropy", hopkins=False,
                marker="Checkpoint state and mode validation passed", diagnostic=None):
        nonlocal completed
        directory = workdir / name
        directory.mkdir(parents=True, exist_ok=True)
        selected = dict(common, **{"run.test": test, "output.directory": str(directory)})
        # Physical scenario selectors and developer fixtures are mutually exclusive,
        # except for the metadata-only checkpoint test.
        if test == "checkpointState":
            selected["run.scenario"] = "hydroWave"
        selected.update(options or {})
        # The output fixture verifies that this exact two-worker CLI is preserved in Silo.
        threads = 2 if test == "unitsOutput" else 1
        command = [str(executable), f"--hpx:threads={threads}", "--hpx:bind=none",
                   *(f"--{key}={value}" for key, value in selected.items())]
        try:
            result = subprocess.run(command, cwd=directory, env=environment, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        except subprocess.TimeoutExpired as error:
            output = error.stdout or b""
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            (directory / "run.log").write_text(output)
            raise RuntimeError(f"{name} timed out; see {directory / 'run.log'}") from error
        (directory / "run.log").write_text(result.stdout)
        if diagnostic is not None:
            if result.returncode == 0 or diagnostic not in result.stdout:
                raise RuntimeError(f"{name} did not reject with {diagnostic!r}; "
                                   f"see {directory / 'run.log'}\n{result.stdout[-6000:]}")
        else:
            if result.returncode != 0 or marker not in result.stdout:
                raise RuntimeError(f"{name} did not complete its fixture; "
                                   f"see {directory / 'run.log'}\n{result.stdout[-6000:]}")
            for key, expected in (("hydro.sphFormulation", formulation),
                                  ("hydro.hopkins", "T" if hopkins else "F")):
                if not re.search(r"^" + re.escape(key) + r"\s*:\s*" + re.escape(expected) + r"\s*$",
                                 result.stdout, re.MULTILINE):
                    raise RuntimeError(f"{name} resolved incorrect {key}; see {directory / 'run.log'}")
        completed += 1

    runCase("default", "checkpointState")
    for formulation in ("densityEntropy", "densityEnergy"):
        runCase("cli_" + formulation, "checkpointState",
                {"hydro.sphFormulation": formulation}, formulation=formulation)
        configuration = workdir / f"{formulation}.ini"
        configuration.write_text(f"[hydro]\nsphFormulation = {formulation}\n")
        runCase("ini_" + formulation, "checkpointState",
                {"configuration": str(configuration)}, formulation=formulation)
    runCase("cli_overrides_ini", "checkpointState",
            {"configuration": str(workdir / "densityEnergy.ini"),
             "hydro.sphFormulation": "densityEntropy"})
    invalidConfiguration = workdir / "invalid.ini"
    invalidConfiguration.write_text("[hydro]\nsphFormulation = unknown\n")
    for label, options in (("cli", {"hydro.sphFormulation": "unknown"}),
                           ("ini", {"configuration": str(invalidConfiguration)})):
        runCase("invalid_" + label, "checkpointState", options,
                diagnostic="hydro.sphFormulation must be densityEntropy or densityEnergy")
    runCase("hopkins_requires_hydro", "checkpointState",
            {"hydro.hopkins": "true", "hydro.enabled": "false"},
            diagnostic="hydro.hopkins=true requires hydro.enabled=true")

    # sphOnlyDiagnostics exercises real allocation, a complete particle checkpoint,
    # density refresh, and the production two-stage midpoint operator.
    for formulation in ("densityEntropy", "densityEnergy"):
        for hopkins in (False, True):
            options = {"hydro.sphFormulation": formulation,
                       "hydro.hopkins": str(hopkins).lower()}
            runCase(f"layout_{formulation}_{int(hopkins)}", "sphOnlyDiagnostics", options,
                    formulation=formulation, hopkins=hopkins,
                    marker="SPH-only diagnostics and thermal lifecycle passed")

    for hopkins in (False, True):
        options = {"hydro.sphFormulation": "densityEnergy",
                   "hydro.hopkins": str(hopkins).lower()}
        suffix = "hopkins_energy" if hopkins else "energy"
        if args.with_chemistry:
            runCase("cooling_" + suffix, "sphChemistryTimesteps",
                    dict(options, **{"chemistry.enabled": "true"}),
                    formulation="densityEnergy", hopkins=hopkins,
                    marker="SPH chemistry timesteps passed")
        if args.with_star_formation:
            coupled = dict(options, **{"chemistry.enabled": "true", "starFormation.enabled": "true"})
            for test, marker in (("star_formation", "GADGET-4 star formation passed"),
                                 ("starFormationParticles", "Star-formation particles passed"),
                                 ("sphSourceTiming", "SPH source timing passed")):
                runCase(test + "_" + suffix, test, coupled, formulation="densityEnergy",
                        hopkins=hopkins, marker=marker)
    if args.with_silo:
        runCase("energy_output", "unitsOutput", {"hydro.sphFormulation": "densityEnergy"},
                formulation="densityEnergy", marker="Boost.Units Silo output checks passed")
    print(f"SPH thermal formulation validation passed ({completed} sequential cases)")


if __name__ == "__main__":
    main()
