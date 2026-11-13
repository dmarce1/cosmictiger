#!/usr/bin/env python3
#  CosmicTiger - A cosmological N-Body code
#  Copyright (C) 2026  Dominic C. Marcello
#
#  Distributed under the Boost Software License, Version 1.0.
#  See LICENSE_1_0.txt or https://www.boost.org/LICENSE_1_0.txt.

"""Validate SPH coupled to PMMM self-gravity with the linear Jeans wave.

The jeansWave scenario sets 4πGρ₀ = c²k²/2, so a standing wave oscillates at
ω_J = ck/√2 instead of the sound frequency ck. The run uses the hierarchical
integrator, the only one that couples gas to PMMM. On one locality the measured
frequency must be near ω_J and far from ck; on 2 and 4 localities (PMMM, ghost
leaves and the gas coupling distributed) it must equal the one-locality value to
float roundoff.
"""

import argparse
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True

from application_selection import cpu_backend_options
from checkpoint_format import require
from run_checkpoint import prepareCase, runSimulation

RESULT = re.compile(r"^Jeans wave: gravity=(\w) time=(\S+) velocity_mode=(\S+) density_mode=(\S+) "
                    r"omega_fit=(\S+) omega_jeans=(\S+) omega_sound=(\S+) amplitude=(\S+) "
                    r"relative_error=(\S+)$", re.MULTILINE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--mpi-exec")
    parser.add_argument("--mpi-numproc-flag", default="-n")
    parser.add_argument("--mpi-preflag", action="append", default=[])
    parser.add_argument("--mpi-postflag", action="append", default=[])
    parser.add_argument("--hpx-argument", action="append", default=[])
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    executable = args.executable.resolve()
    root = args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    preset = Path(__file__).resolve().parents[1] / "configurations" / "jeans_wave.ini"
    options = {"configuration": preset, "run.gasParticleCount": 8000, "output.silo": "false", "output.directory": ".",
               "checkpoint.intervalSeconds": 1000000, "analysis.views": "false",
               "analysis.power": "false", "hpx:bind": "none", **cpu_backend_options()}

    def run(ranks):
        directory = prepareCase(root, f"ranks_{ranks}")
        threads = max(1, args.threads // ranks)
        command = [str(executable), *args.hpx_argument,
                   *[f"--{key}={value}" for key, value in options.items()],
                   f"--hpx:threads={threads}"]
        if ranks > 1:
            command = [args.mpi_exec, args.mpi_numproc_flag, str(ranks), *args.mpi_preflag,
                       command[0], *args.mpi_postflag, *command[1:], f"--hpx:localities={ranks}"]
            if "--hpx:ini=hpx.parcel.mpi.enable=1" not in command:
                command.append("--hpx:ini=hpx.parcel.mpi.enable=1")
        output = runSimulation(command, directory, "run.log", timeout=900)
        match = RESULT.search(output)
        require(match is not None, f"ranks={ranks}: no Jeans wave result")
        require(match[1] == "T", f"ranks={ranks}: gravity was not enabled")
        if ranks > 1:
            require("SPH ghosts: syncs=" in output, f"ranks={ranks}: no ghost exchange reported")
        return {"omega": float(match[5]), "jeans": float(match[6]), "sound": float(match[7]),
                "amplitude": float(match[8]), "error": float(match[9])}

    evidence = {"ranks_1": run(1)}
    single = evidence["ranks_1"]
    # SPH sound speed at 20 particles per wavelength costs a few percent; gravity must
    # still separate the frequency clearly from the sound frequency.
    require(single["error"] < 0.08,
            f"Jeans frequency {single['omega']:.6e} misses {single['jeans']:.6e} by "
            f"{single['error']:.3f}")
    require(abs(single["omega"] - single["jeans"]) < 0.25 * abs(single["sound"] - single["jeans"]),
            "the frequency is not closer to the Jeans value than to the sound value")
    if args.mpi_exec:
        for ranks in (2, 4):
            result = run(ranks)
            evidence[f"ranks_{ranks}"] = result
            difference = abs(result["omega"] - single["omega"]) / single["omega"]
            require(difference < 1.0e-4,
                    f"ranks={ranks}: frequency {result['omega']:.9e} differs from one locality "
                    f"by {difference:.3e}")
    (root / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    for label, result in evidence.items():
        print(f"{label}: omega={result['omega']:.9e} jeans={result['jeans']:.9e} "
              f"sound={result['sound']:.9e} error={result['error']:.4f}", flush=True)
    print("SPH Jeans wave matches the PMMM dispersion relation", flush=True)


if __name__ == "__main__":
    main()
