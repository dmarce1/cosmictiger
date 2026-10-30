# CosmicTiger - particle-mesh failure integration regression; BSL-1.0.
"""Require prompt MPI termination after partial PMMM collective dispatch fails."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from application_selection import application_arguments


def stop_process_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_case(args, name, marker, blocker):
    workdir = args.workdir.resolve() / name
    workdir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["OMP_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    command = ([args.mpi_exec, args.mpi_numproc_flag, "2"]
               + args.mpi_preflag + [str(args.executable.resolve())]
               + args.mpi_postflag
               + [f"--run.test={name}", "--gravity.meshSize=8",
                  "--initial.particlesPerDimension=8", "--hydro.enabled=false",
                  "--compute.cuda=false", "--hpx:threads=2", "--hpx:bind=none",
                  "--hpx:localities=2", "--hpx:ini=hpx.parcel.mpi.enable=1",
                  "--hpx:ini=hpx.threadpools.io_pool_size=1"])
    command += application_arguments(args.executable.resolve(), far="pmmm")
    if args.disable_tcp:
        command.append("--hpx:ini=hpx.parcel.tcp.enable=0")
    log_path = workdir / "failure.log"
    with log_path.open("w") as log:
        with subprocess.Popen(command, cwd=workdir, env=environment, text=True,
                              stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True) as process:
            try:
                returncode = process.wait(timeout=30)
            except subprocess.TimeoutExpired as error:
                stop_process_group(process)
                raise RuntimeError(f"{name}: collective dispatch hung; see {log_path}") from error
    output = log_path.read_text()
    required = [f"PMMM dispatch test entered collective on locality {blocker}\n",
                f"PMMM dispatch test injected {marker} failure\n",
                "Particle-mesh collective dispatch failed; aborting MPI job"]
    if returncode == 0 or any(item not in output for item in required):
        raise RuntimeError(f"{name}: expected fatal dispatch failure ({returncode}); see {log_path}")
    if "PMMM dispatch test unexpectedly returned" in output:
        raise RuntimeError(f"{name}: dispatch returned after failure; see {log_path}")
    # MPI launchers may translate exit codes, so the application's diagnostic
    # and a failed launcher status are required instead of a launcher-specific
    # numeric encoding of MPI_Abort's error code.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--mpi-exec", required=True)
    parser.add_argument("--mpi-numproc-flag", default="-n")
    parser.add_argument("--mpi-preflag", action="append", default=[])
    parser.add_argument("--mpi-postflag", action="append", default=[])
    parser.add_argument("--disable-tcp", action="store_true",
                        help="disable the TCP parcelport when it is present in the HPX build")
    args = parser.parse_args()
    run_case(args, "particleMeshDispatchLaunchFailure", "launch", 0)
    run_case(args, "particleMeshDispatchAsyncFailure", "asynchronous", 1)
    print("PMMM collective failures passed: partial launch and asynchronous failure terminate MPI")


if __name__ == "__main__":
    main()
