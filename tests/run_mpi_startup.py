# CosmicTiger - MPI startup integration regression; BSL-1.0.
"""Check required MPI thread support before HPX startup and fixed64 dispatch."""

import argparse
import os
from pathlib import Path
import signal
import subprocess


def run_case(args, name, *, level=None, ranks=1, target_rank=None, fixed64=False):
    workdir = args.workdir.resolve() / name
    workdir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["OMP_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["LD_PRELOAD"] = str(args.guard.resolve())
    environment.pop("COSMICTIGER_TEST_MPI_THREAD_LEVEL", None)
    environment.pop("COSMICTIGER_TEST_MPI_THREAD_RANK", None)
    if level is not None:
        environment["COSMICTIGER_TEST_MPI_THREAD_LEVEL"] = level
    if target_rank is not None:
        environment["COSMICTIGER_TEST_MPI_THREAD_RANK"] = str(target_rank)
    command = [str(args.executable.resolve()), "--help", "--compute.cuda=false",
               "--hpx:threads=2", "--hpx:bind=none"]
    if fixed64:
        command.append("--compute.positionType=fixed64")
    if ranks > 1:
        command = ([args.mpi_exec, args.mpi_numproc_flag, str(ranks)]
                   + args.mpi_preflag + command[:1] + args.mpi_postflag + command[1:])
        command.append(f"--hpx:localities={ranks}")
    log_path = workdir / "startup.log"
    with log_path.open("w") as log:
        with subprocess.Popen(command, cwd=workdir, env=environment, text=True,
                              stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True) as process:
            try:
                returncode = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError(f"{name}: MPI startup timed out; see {log_path}")
    output = log_path.read_text()
    for rank in range(ranks):
        for stage in ("initialized", "finalized"):
            marker = f"MPI startup probe {stage} rank {rank}\n"
            if output.count(marker) != 1:
                raise RuntimeError(f"{name}: expected one {stage} on rank {rank}; see {log_path}")
    if level is None:
        if returncode != 0 or "CosmicTiger options" not in output:
            raise RuntimeError(f"{name}: MULTIPLE startup failed ({returncode}); see {log_path}")
    else:
        if (returncode == 0 or "CosmicTiger requires MPI_THREAD_MULTIPLE" not in output
                or level not in output or "CosmicTiger options" in output):
            raise RuntimeError(f"{name}: unsupported thread level was not rejected; see {log_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("guard", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--mpi-exec")
    parser.add_argument("--mpi-numproc-flag", default="-n")
    parser.add_argument("--mpi-preflag", action="append", default=[])
    parser.add_argument("--mpi-postflag", action="append", default=[])
    args = parser.parse_args()
    if args.mpi_exec:
        run_case(args, "multiple_two_ranks", ranks=2)
        run_case(args, "serialized_one_rank", level="MPI_THREAD_SERIALIZED",
                 ranks=2, target_rank=1)
    else:
        run_case(args, "multiple")
        run_case(args, "multiple_fixed64", fixed64=True)
        for level in ("MPI_THREAD_SINGLE", "MPI_THREAD_FUNNELED", "MPI_THREAD_SERIALIZED"):
            run_case(args, level.lower(), level=level)
        run_case(args, "serialized_fixed64", level="MPI_THREAD_SERIALIZED", fixed64=True)
    print("MPI startup passed: MULTIPLE accepted, lower levels rejected, MPI finalized")


if __name__ == "__main__":
    main()
