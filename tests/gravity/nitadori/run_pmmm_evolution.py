# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# SPDX-License-Identifier: BSL-1.0

"""Check periodic PMMM hierarchical evolution, restart, and solver identity."""

import argparse
import ctypes
from pathlib import Path
import re
import struct
import subprocess
import sys

sys.dont_write_bytecode = True

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from run_checkpoint import fnv1a64, prepareCase, runSimulation
from pmmm_evolution_helpers import (check_clock, check_single_drift, compare_restart,
                                    Header, read_checkpoint, require)


def use_legacy_pp_extension(checkpoint, ranks):
    """Turn a copied v2 PP checkpoint into the historical seven-word v1 layout."""
    manifest = checkpoint / "checkpoint.meta"
    lines = manifest.read_text().splitlines()
    offset = ctypes.sizeof(Header)
    for rank in range(ranks):
        path = checkpoint / f"checkpoint.{rank}.dat"
        data = path.read_bytes()
        extension = struct.unpack_from("=8Q", data, offset)
        require(extension[0] == 2 and extension[7] == 0,
                "legacy fixture requires a version-two PP checkpoint")
        data = (data[:offset] + struct.pack("=Q", 1) + data[offset + 8:offset + 56]
                + data[offset + 64:])
        path.write_bytes(data)
        lines[rank + 1] = f"{rank} {len(data)} {fnv1a64(data):016x}"
    manifest.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--mpi-exec")
    parser.add_argument("--mpi-numproc-flag", default="-n")
    parser.add_argument("--mpi-preflag", action="append", default=[])
    parser.add_argument("--mpi-postflag", action="append", default=[])
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--position-type", choices=("fixed32", "fixed64"), default="fixed32")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    executable = args.executable.resolve()
    root = args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    ranks = args.ranks if args.mpi_exec else 1
    common = {"hpx:threads": 2, "hpx:bind": "none", "gravity.method": "pmmm",
              "gravity.meshSize": 8, "gravity.fmmOrder": 7, "gravity.pmmmNearWidth": 2,
              "gravity.enabled": "true", "compute.cuda": str(args.cuda).lower(),
              "compute.positionType": args.position_type,
              "initial.particlesPerDimension": 8, "initial.closePack": "false",
              "initial.useGlass": "false", "chemistry.enabled": "false",
              "run.endTime": 0.005, "run.numberOfSteps": 4,
              "analysis.groups": "false", "analysis.tracers": "false",
              "analysis.views": "false", "analysis.power": "false", "lightcone.enabled": "false",
              "output.silo": "false", "output.directory": ".", "checkpoint.intervalSeconds": 0}
    if ranks > 1:
        common["hpx:localities"] = ranks

    def run(directory, options, success=True):
        command = [str(executable)] + [f"--{key}={value}" for key, value in options.items()]
        if args.mpi_exec:
            command = ([args.mpi_exec, args.mpi_numproc_flag, str(ranks)] + args.mpi_preflag
                       + command[:1] + args.mpi_postflag + command[1:])
        if ranks > 1:
            command += ["--hpx:ini=hpx.parcel.mpi.enable=1",
                        "--hpx:ini=hpx.parcel.tcp.enable=0"]
        return runSimulation(command, directory, "run.log", expectSuccess=success)

    for near_method, hydro in (("pp", False), ("fmm", False), ("pp", True), ("fmm", True)):
        label = near_method + ("_mixed" if hydro else "_dark_matter")
        options = dict(common, **{"hydro.enabled": str(hydro).lower(),
                                  "gravity.pmmmNearMethod": near_method,
                                  "run.scenario": "cosmology" if hydro else "plummer"})
        if hydro:
            rejected = prepareCase(root, label + "_rejected")
            log = run(rejected, dict(options, **{"run.maxIterations": 0}), False)
            require("SPH currently requires gravity.method=fmm when gravity.enabled=true" in log,
                    "PMMM with SPH was not rejected with the expected diagnostic")
            print(f"{label}: unsupported PMMM with SPH rejected at startup")
            continue
        baseline = prepareCase(root, label + "_baseline")
        run(baseline, dict(options, **{"run.maxIterations": 3}))
        paths = sorted((path for path in baseline.glob("checkpoint.*") if path.is_dir() and
                        re.fullmatch(r"checkpoint\.[0-9]+", path.name)),
                       key=lambda path: int(path.name.split(".")[1]))
        require([int(path.name.split(".")[1]) for path in paths] == [1, 2, 3, 4],
                f"{label}: expected one PMMM checkpoint per step")
        states = [read_checkpoint(path, ranks) for path in paths]
        require(any(len({row["rung"] for row in rows}) > 1 for _, _, rows in states),
                "hierarchical evolution fixture did not exercise simultaneous particle rungs")
        check_clock(states)
        for before, after in zip(states, states[1:]):
            check_single_drift(before, after)
        source = baseline / "checkpoint.2"
        resumed = prepareCase(root, label + "_resumed", source)
        log = run(resumed, dict(options, **{"checkpoint.readIndex": 2,
                                          "run.maxIterations": 1}))
        require("Done reading checkpoint" in log, "PMMM restart did not load its checkpoint")
        compare_restart(states[-1], read_checkpoint(resumed / "checkpoint.4", ranks))
        if near_method == "pp":
            legacy = prepareCase(root, label + "_legacy", source)
            use_legacy_pp_extension(legacy / source.name, ranks)
            # The inspection decoder and application must both accept v1 as PP.
            read_checkpoint(legacy / source.name, ranks)
            run(legacy, dict(options, **{"checkpoint.readIndex": 2, "run.maxIterations": 1}))
            compare_restart(states[-1], read_checkpoint(legacy / "checkpoint.4", ranks))
            rejected = prepareCase(root, label + "_legacy_reject_fmm", source)
            use_legacy_pp_extension(rejected / source.name, ranks)
            log = run(rejected, dict(options, **{"gravity.pmmmNearMethod": "fmm",
                "checkpoint.readIndex": 2, "run.maxIterations": 0}), False)
            require("PMMM checkpoint" in log, "legacy PP checkpoint accepted FMM near solver")
        changes = (("mesh", {"gravity.meshSize": 10}),
                   ("order", {"gravity.fmmOrder": 5}),
                   ("near_width", {"gravity.pmmmNearWidth": 1}),
                   ("near_method", {"gravity.pmmmNearMethod": "fmm" if near_method == "pp" else "pp"}),
                   ("fmm", {"gravity.method": "fmm"}))
        for name, change in changes:
            rejected = prepareCase(root, label + "_reject_" + name, source)
            log = run(rejected, dict(options, **change,
                                    **{"checkpoint.readIndex": 2, "run.maxIterations": 0}), False)
            require("checkpoint" in log.lower(), f"{name} mismatch lacked a checkpoint diagnostic")
        print(f"{label}: hierarchical dyadic steps, scheduled single drifts, restart state, and PMMM mesh/order/near-width/method/solver "
              "mismatch rejection passed")
    print(f"PMMM evolution passed ({ranks} rank(s), {'CUDA' if args.cuda else 'CPU'} drift, "
          f"{args.position_type})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, struct.error, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
