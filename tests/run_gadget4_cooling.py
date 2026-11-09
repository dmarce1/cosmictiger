"""Exercise GADGET-4 cooling through real SPH startup, steps and restarts."""
import argparse
import math
from pathlib import Path
import struct
import subprocess
import sys

from application_selection import application_arguments
from checkpoint_format import decode_checkpoint
from run_checkpoint import checkManifest, fnv1a64, prepareCase, runSimulation


def gas_state(checkpoint, expected_count=512, ranks=1):
    payloads = []
    gas_count = 0
    for rank in range(ranks):
        data = (checkpoint / f"checkpoint.{rank}.dat").read_bytes()
        state = decode_checkpoint(data, ranks=ranks)
        if not state.has_cooling or state.has_stars:
            raise RuntimeError("Expected equilibrium-cooling SPH state without star formation")
        gas_count += state.gas_count
        for (ne,) in state.cooling:
            if not (math.isfinite(ne) and ne >= 0):
                raise RuntimeError(f"Invalid equilibrium-cooling electron fraction: {ne}")
        payloads.append(data[state.gas_start:state.gas_end])
    if gas_count != expected_count:
        raise RuntimeError(f"Unexpected total gas count: {gas_count}")
    return tuple(payloads)


def obsolete_uv_payload(data, ranks=1):
    state = decode_checkpoint(data, ranks=ranks)
    payload = bytearray(data)
    struct.pack_into('=d', payload, state.cooling_offset, 5.)
    # Version five inserted a UV table checksum after its six doubles. Preserve
    # that older shape so failure must precede driver/particle offset decoding.
    checksum_offset = state.cooling_offset + 6 * struct.calcsize('=d')
    payload[checksum_offset:checksum_offset] = struct.pack('=Q', 0xBAD0BAD0BAD0BAD0)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--time-integrator", default="traditional",
                        choices=("traditional", "hierarchical", "global"))
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--mpi-exec")
    parser.add_argument("--mpi-numproc-flag", default="-n")
    parser.add_argument("--mpi-preflag", action="append", default=[])
    parser.add_argument("--mpi-postflag", action="append", default=[])
    parser.add_argument("--hpx-argument", action="append", default=[])
    args = parser.parse_args()
    executable = args.executable.resolve()
    base = args.workdir.resolve()
    base.mkdir(parents=True, exist_ok=True)
    ranks = 2 if args.mpi_exec else 1
    # The hierarchical ladder needs gravity: it is the integrator that couples gas to PMMM.
    gravity = args.time_integrator == "hierarchical"
    options = [str(executable), *args.hpx_argument,
               f"--hpx:threads={1 if args.mpi_exec else 2}", "--hpx:bind=none",
               "--run.scenario=cosmology", "--run.darkMatterParticleCount=512", "--run.gasParticleCount=512",
               "--hydro.enabled=true", f"--gravity.enabled={str(gravity).lower()}",
               "--gravity.meshSize=8", "--chemistry.enabled=true",
               "--starFormation.enabled=false",
               "--chemistry.initializationTemperature=1000000", "--chemistry.minimumSpecificEnergy=0",
               f"--gravity.cuda={str(args.cuda).lower()}",
               f"--hydro.cuda={str(args.cuda).lower()}",
               f"--drift.cuda={str(args.cuda).lower()}",
               f"--chemistry.cuda={str(args.cuda).lower()}",
               "--cosmology.sigma8=0", "--output.directory=.",
               "--checkpoint.intervalSeconds=0", "--output.silo=false", "--analysis.views=false",
               "--analysis.power=false", "--analysis.groups=false", "--lightcone.enabled=false"]
    options += application_arguments(executable, integrator=args.time_integrator, hydro=True)
    if args.mpi_exec:
        options = [args.mpi_exec, args.mpi_numproc_flag, "2", *args.mpi_preflag,
                   options[0], *args.mpi_postflag, *options[1:], "--hpx:localities=2",
                   "--hpx:ini=hpx.parcel.mpi.enable=1"]
    reference = prepareCase(base, "reference")
    try:
        runSimulation(options + ["--run.maxIterations=2"], reference, "run.log")
    except RuntimeError:
        output = (reference / "run.log").read_text(errors="replace")
        unavailable = ("This executable was compiled without CUDA support",
                       "Selected CUDA backends require an available CUDA device")
        if args.cuda and any(message in output for message in unavailable):
            print("SKIP: requested CUDA cooling backend is unavailable")
            return 77
        raise
    first = reference / "checkpoint.1"
    final = reference / "checkpoint.2"
    checkManifest(first, 1, ranks=ranks, gravity=gravity)
    checkManifest(final, 2, ranks=ranks, gravity=gravity)
    first_state = gas_state(first, ranks=ranks)
    final_state = gas_state(final, ranks=ranks)
    if first_state == final_state:
        raise RuntimeError("Completed cooling/SPH steps did not evolve the gas payload")
    resumed = prepareCase(base, "restart", first)
    runSimulation(options + ["--run.maxIterations=1", "--checkpoint.readIndex=1"], resumed, "restart.log")
    if gas_state(resumed / "checkpoint.2", ranks=ranks) != final_state:
        raise RuntimeError("Restart changed the gas evolution or saved electron guess")
    incompatible = prepareCase(base, "changed_floor", first)
    changed = [option for option in options if not option.startswith("--chemistry.minimumSpecificEnergy=")]
    output = runSimulation(changed + ["--chemistry.minimumSpecificEnergy=1", "--run.maxIterations=0",
                           "--checkpoint.readIndex=1"], incompatible, "reject.log", expectSuccess=False)
    if "Checkpoint GADGET-4 cooling model or initialization settings differ" not in output:
        raise RuntimeError("Changed cooling floor lacked a checkpoint compatibility diagnostic")
    old_model = prepareCase(base, "retired_uv_checkpoint", first)
    old_checkpoint = old_model / first.name
    manifest = old_checkpoint / "checkpoint.meta"
    records = manifest.read_text().splitlines()
    for rank in range(ranks):
        path = old_checkpoint / f"checkpoint.{rank}.dat"
        payload = obsolete_uv_payload(path.read_bytes(), ranks)
        path.write_bytes(payload)
        records[rank + 1] = f"{rank} {len(payload)} {fnv1a64(payload):016x}"
    manifest.write_text("\n".join(records) + "\n")
    output = runSimulation(options + ["--run.maxIterations=0", "--checkpoint.readIndex=1"],
                           old_model, "reject.log", expectSuccess=False)
    if "Checkpoint particle cooling model is incompatible with GADGET-4 cooling" not in output:
        raise RuntimeError("Version-five UV checkpoint was not rejected before reading later blocks")
    retired = prepareCase(base, "retired_network")
    output = runSimulation(options + ["--chemistry.network=molecular9", "--run.maxIterations=0"],
                           retired, "reject.log", expectSuccess=False)
    if "chemistry.network" not in output:
        raise RuntimeError("Retired reaction-network option lacked an option diagnostic")
    print(f"GADGET-4 cooling {args.time_integrator}, {ranks} rank(s), "
          f"{'CUDA' if args.cuda else 'CPU'}: finite equilibrium state, exact restart, "
          "parameter compatibility, retired-UV checkpoint and retired-network rejection passed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
