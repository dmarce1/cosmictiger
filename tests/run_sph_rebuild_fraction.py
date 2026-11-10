"""Check SPH rebuild fraction configuration and actual octree lifecycle decisions."""

import argparse
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True

from application_selection import application_arguments, compiled_algorithms, cpu_backend_options
from checkpoint_format import require
from run_checkpoint import prepareCase, runSimulation
from run_sph_octree import EVENT, checkpoint_state, validate_events


OPTION = "compute.sphTreeRebuildFraction"
METADATA_COMPLETE = "Checkpoint state and mode validation passed"


def resolved_fraction(output, expected, label):
    values = re.findall(r"^" + re.escape(OPTION) + r"\s*:\s*(\S+)", output, re.MULTILINE)
    require(values, f"{label}: resolved options omitted {OPTION}")
    require(all(math.isclose(float(value), expected, rel_tol=2e-6, abs_tol=1e-12)
                for value in values), f"{label}: incorrect resolved rebuild fraction {values}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--time-integrator", action="append",
                        choices=("global", "traditional", "hierarchical"))
    parser.add_argument("--mpi-exec")
    parser.add_argument("--mpi-numproc-flag", default="-n")
    parser.add_argument("--mpi-preflag", action="append", default=[])
    parser.add_argument("--mpi-postflag", action="append", default=[])
    parser.add_argument("--hpx-argument", action="append", default=[])
    args = parser.parse_args()
    executable = args.executable.resolve()
    root = args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    ranks = 2 if args.mpi_exec else 1
    available = compiled_algorithms(str(executable))
    integrators = args.time_integrator or ["hierarchical"]
    require(integrators, "no supported integrator was compiled")
    require(all(name in available["integrator"] for name in integrators),
            "a requested integrator was not compiled")
    common = {"hpx:threads": args.threads, "hpx:bind": "none", **cpu_backend_options(),
              "run.darkMatterParticleCount": 512, "run.gasParticleCount": 512, "scenario.cosmology.useGlass": "false",
              "initial.closePack": "false", "hydro.enabled": "true",
              "gravity.enabled": "false", "chemistry.enabled": "false",
              "starFormation.enabled": "false", "analysis.groups": "false",
              "analysis.tracers": "false", "analysis.views": "false",
              "analysis.power": "false", "analysis.slice": "false",
              "lightcone.enabled": "false", "output.silo": "false",
              "output.directory": ".", "checkpoint.intervalSeconds": 0,
              "run.verbose": "true", "compute.sphBucketSize": 16,
              "hydro.viscosityCoefficientMin": 1, "hydro.viscosityCoefficientMax": 1}

    def run(name, options, integrator=None, *, success=True, extra=(), localities=ranks):
        directory = prepareCase(root, name)
        command = [str(executable), *args.hpx_argument, *application_arguments(
            executable, integrator=integrator or integrators[0], hydro=True)]
        command += [f"--{key}={value}" for key, value in dict(common, **options).items()]
        command += extra
        if localities == 2:
            command = [args.mpi_exec, args.mpi_numproc_flag, "2", *args.mpi_preflag,
                       command[0], *args.mpi_postflag, *command[1:],
                       "--hpx:localities=2", "--hpx:ini=hpx.parcel.mpi.enable=1"]
        return directory, runSimulation(command, directory, "run.log", expectSuccess=success)

    _, help_text = run("help", {}, extra=["--help"], localities=1)
    require(re.search(r"^\s+--" + re.escape(OPTION) + r"\b", help_text, re.MULTILINE),
            f"help omitted {OPTION}")
    require("fraction of all particles" in " ".join(help_text.split()),
            "help did not explain that the threshold includes nongas particles")
    metadata = {"run.test": "checkpointState", "hydro.enabled": "false"}
    evidence = {"ranks": ranks, "metadata": {}, "evolution": {}}

    def check_metadata(name, expected, options=None, extra=()):
        _, output = run(name, dict(metadata, **(options or {})), extra=extra)
        require(METADATA_COMPLETE in output, f"{name}: bounded metadata fixture did not finish")
        resolved_fraction(output, expected, name)
        evidence["metadata"][name] = expected

    check_metadata("default", 1.0 / 3.0)
    for value in (0, 0.1, 0.5, 1):
        check_metadata(f"cli_{value}", value, {OPTION: value})
    config = root / "fraction.ini"
    config.write_text("[compute]\nsphTreeRebuildFraction = 0.5\n")
    check_metadata("ini", 0.5, extra=[f"--configuration={config}"])
    check_metadata("cli_precedence", 0.1, {OPTION: 0.1}, [f"--configuration={config}"])
    for value in ("-0.1", "1.01", "nan", "inf"):
        name = "invalid_" + value.replace("-", "minus_")
        _, output = run(name, dict(metadata, **{OPTION: value}), success=False, localities=1)
        require(OPTION in output, f"{name}: rejection did not identify the invalid option")
        require(METADATA_COMPLETE not in output, f"{name}: invalid fraction reached the fixture")
        evidence["metadata"][name] = "rejected"
    config.write_text("[compute]\nsphTreeRebuildFraction = -0.1\n")
    _, output = run("invalid_ini", metadata, success=False,
                    extra=[f"--configuration={config}"], localities=1)
    require(OPTION in output and METADATA_COMPLETE not in output,
            "invalid INI fraction was not rejected before execution")
    # CLI precedence must also replace an invalid INI value before validation.
    check_metadata("cli_over_invalid_ini", 1, {OPTION: 1}, [f"--configuration={config}"])
    print("SPH rebuild fraction help, defaults, CLI/INI precedence and rejection passed", flush=True)

    # A localized blast produces hot fine-rung targets and cold coarse-rung
    # sources. Four events are enough to exercise sparse retained membership;
    # the global integrator remains fully active regardless of the threshold.
    blast = {"run.scenario": "blast", "run.endTime": 0.005,
             "run.numberOfOutputs": 2, "run.maxIterations": 4,
             "run.darkMatterParticleCount": 4096, "run.gasParticleCount": 3456}
    for integrator in integrators:
        for name, fraction in (("zero", 0), ("default", 1.0 / 3.0), ("one", 1)):
            label = f"{integrator}_{name}"
            options = dict(blast)
            if name != "default":
                options[OPTION] = fraction
            directory, output = run(label, options, integrator)
            resolved_fraction(output, fraction, label)
            counts = validate_events(output, label, integrator, fraction)
            events = EVENT.findall(output)
            _, _, _, species = checkpoint_state(directory, blast["run.maxIterations"], ranks)
            # The blast initializer rounds the request up to two interleaved
            # n^3 lattices; 3,456 = 2 * 12^3 is already whole.
            require(species == (0, 3456), f"{label}: blast particle population changed")
            require(all(int(total) == sum(species) for _, _, total, _ in events),
                    f"{label}: lifecycle denominator differs from the total particle count")
            if fraction == 0:
                require(counts["retain"] == 0, f"{label}: zero threshold retained the tree")
            elif integrator != "global":
                require(counts["retain"] > 0,
                        f"{label}: blast failed to exercise a below-threshold active subset")
            evidence["evolution"][label] = {**counts, "fraction": fraction,
                                             "particles": sum(species)}
            print(f"{label}: {counts['rebuild']} rebuilds, {counts['retain']} retained events",
                  flush=True)

    (root / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print("SPH rebuild fraction application regression passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
