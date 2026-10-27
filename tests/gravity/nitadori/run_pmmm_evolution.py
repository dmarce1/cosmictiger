# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
# SPDX-License-Identifier: BSL-1.0

"""Check periodic PMMM evolution, restart, and compiled solver/integrator identity."""

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


def rewriteCheckpoint(checkpoint, ranks, transform):
    """Update a copied rank payload and its manifest checksum together."""
    manifest = checkpoint / "checkpoint.meta"
    lines = manifest.read_text().splitlines()
    offset = ctypes.sizeof(Header)
    for rank in range(ranks):
        path = checkpoint / f"checkpoint.{rank}.dat"
        data = path.read_bytes()
        data = transform(data, offset)
        path.write_bytes(data)
        lines[rank + 1] = f"{rank} {len(data)} {fnv1a64(data):016x}"
    manifest.write_text("\n".join(lines) + "\n")


def use_legacy_pp_extension(checkpoint, ranks):
    """Give a copied checkpoint the historical seven-word PP-only v1 layout."""
    def transform(data, offset):
        extension_version = struct.unpack_from("=Q", data, offset)[0]
        require(extension_version in (1, 2), "legacy fixture has an unknown PMMM version")
        extension_size = {1: 7, 2: 8}[extension_version]
        return (data[:offset] + struct.pack("=Q", 1) + data[offset + 8:offset + 56]
                + data[offset + 8 * extension_size:])
    rewriteCheckpoint(checkpoint, ranks, transform)


def changeCompiledIdentity(checkpoint, ranks, *, order=None, near_method=None, fmm=False,
                           time_integrator=None):
    """Present the executable with a validly checksummed incompatible variant."""
    def transform(data, offset):
        payload = bytearray(data)
        require(struct.unpack_from("=Q", payload, offset)[0] == 2,
                "identity fixture requires PMMM extension version two")
        if order is not None:
            struct.pack_into("=Q", payload, offset + 8 * 2, order)
        if near_method is not None:
            struct.pack_into("=Q", payload, offset + 8 * 7, int(near_method == "fmm"))
        if fmm or time_integrator is not None:
            header = Header.from_buffer_copy(payload)
            if time_integrator == "hierarchical":
                header.flags |= 1 << 6
            elif time_integrator == "leapfrog":
                header.flags &= ~(1 << 6)
            if fmm:
                header.flags &= ~(1 << 17)
            payload[:offset] = bytes(header)
            if fmm:
                del payload[offset:offset + 64]
        return bytes(payload)
    rewriteCheckpoint(checkpoint, ranks, transform)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--mpi-exec")
    parser.add_argument("--mpi-numproc-flag", default="-n")
    parser.add_argument("--mpi-preflag", action="append", default=[])
    parser.add_argument("--mpi-postflag", action="append", default=[])
    parser.add_argument("--disable-tcp", action="store_true",
                        help="disable the TCP parcelport when it is present in the HPX build")
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--position-type", choices=("fixed32", "fixed64"), default="fixed32")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--near-method", choices=("pp", "fmm"), required=True)
    parser.add_argument("--order", type=int, choices=range(3, 10), required=True)
    parser.add_argument("--time-integrator", choices=("leapfrog", "hierarchical"), required=True)
    args = parser.parse_args()
    executable = args.executable.resolve()
    root = args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    ranks = args.ranks if args.mpi_exec else 1
    common = {"hpx:threads": 2, "hpx:bind": "none",
              "gravity.meshSize": 8, "gravity.pmmmNearWidth": 2,
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
            command.append("--hpx:ini=hpx.parcel.mpi.enable=1")
            if args.disable_tcp:
                command.append("--hpx:ini=hpx.parcel.tcp.enable=0")
        return runSimulation(command, directory, "run.log", expectSuccess=success)

    near_method = args.near_method
    for hydro in (False, True):
        label = near_method + ("_mixed" if hydro else "_dark_matter")
        options = dict(common, **{"hydro.enabled": str(hydro).lower(),
                                  "run.scenario": "cosmology" if hydro else "plummer"})
        if hydro:
            rejected = prepareCase(root, label + "_rejected")
            log = run(rejected, dict(options, **{"run.maxIterations": 0}), False)
            diagnostic = ("COSMICTIGER_TIME_INTEGRATOR=leapfrog"
                          if args.time_integrator == "hierarchical" else
                          "SPH currently requires an FMM build when gravity.enabled=true")
            require(diagnostic in log,
                    "PMMM with SPH was not rejected with the expected diagnostic")
            print(f"{label}: unsupported PMMM with SPH rejected at startup")
            continue
        baseline = prepareCase(root, label + "_baseline")
        log = run(baseline, dict(options, **{"run.maxIterations": 3}))
        for key, expected in (("build.timeIntegrator", args.time_integrator),
                              ("build.gravityMethod", "pmmm"),
                              ("build.pmmmNearMethod", args.near_method),
                              ("build.fmmOrder", str(args.order))):
            require(re.search(r"^" + re.escape(key) + r"\s*:\s*" + expected + r"\s*$",
                              log, re.MULTILINE), f"compiled variant differs from {key}={expected}")
        paths = sorted((path for path in baseline.glob("checkpoint.*") if path.is_dir() and
                        re.fullmatch(r"checkpoint\.[0-9]+", path.name)),
                       key=lambda path: int(path.name.split(".")[1]))
        require([int(path.name.split(".")[1]) for path in paths] == [1, 2, 3, 4],
                f"{label}: expected one PMMM checkpoint per step")
        states = [read_checkpoint(path, ranks, args.time_integrator) for path in paths]
        if args.time_integrator == "hierarchical":
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
        compare_restart(states[-1], read_checkpoint(
            resumed / "checkpoint.4", ranks, args.time_integrator))
        if near_method == "pp":
            legacy = prepareCase(root, label + "_legacy", source)
            use_legacy_pp_extension(legacy / source.name, ranks)
            # The inspection decoder and application must both accept v1 as PP.
            read_checkpoint(legacy / source.name, ranks, args.time_integrator)
            run(legacy, dict(options, **{"checkpoint.readIndex": 2, "run.maxIterations": 1}))
            compare_restart(states[-1], read_checkpoint(
                legacy / "checkpoint.4", ranks, args.time_integrator))
        else:
            rejected = prepareCase(root, label + "_legacy_reject_fmm", source)
            use_legacy_pp_extension(rejected / source.name, ranks)
            log = run(rejected, dict(options, **{
                "checkpoint.readIndex": 2, "run.maxIterations": 0}), False)
            require("PMMM checkpoint" in log, "legacy PP checkpoint accepted FMM near solver")
        changes = (("mesh", {"gravity.meshSize": 10}),
                   ("near_width", {"gravity.pmmmNearWidth": 1}))
        for name, change in changes:
            rejected = prepareCase(root, label + "_reject_" + name, source)
            log = run(rejected, dict(options, **change,
                                    **{"checkpoint.readIndex": 2, "run.maxIterations": 0}), False)
            require("checkpoint" in log.lower(), f"{name} mismatch lacked a checkpoint diagnostic")
        identities = (("order", {"order": 3 if args.order != 3 else 4}),
                      ("near_method", {"near_method": "fmm" if near_method == "pp" else "pp"}),
                      ("solver", {"fmm": True}),
                      ("integrator", {"time_integrator": "hierarchical"
                                      if args.time_integrator == "leapfrog" else "leapfrog"}))
        for name, identity in identities:
            rejected = prepareCase(root, label + "_reject_" + name, source)
            changeCompiledIdentity(rejected / source.name, ranks, **identity)
            log = run(rejected, dict(options, **{
                "checkpoint.readIndex": 2, "run.maxIterations": 0}), False)
            require("checkpoint" in log.lower(), f"{name} mismatch lacked a checkpoint diagnostic")
        print(f"{label}: {args.time_integrator} dyadic steps, scheduled single drifts, restart state, "
              "and PMMM mesh/order/near-width/method/solver/integrator mismatch rejection passed")
    print(f"PMMM evolution passed ({ranks} rank(s), {'CUDA' if args.cuda else 'CPU'} drift, "
          f"{args.position_type}, {args.time_integrator})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, struct.error, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
