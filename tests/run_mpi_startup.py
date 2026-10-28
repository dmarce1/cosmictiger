# CosmicTiger - MPI startup integration regression; BSL-1.0.
"""Check hwloc defaults, MPI thread support, profiling shutdown, and configured position storage."""

import argparse
import os
from pathlib import Path
import signal
import subprocess


def run_case(args, name, *, level=None, ranks=1, target_rank=None, explicit_position=False,
             hwloc_components=None, threads=2, apex_reports=False):
    workdir = args.workdir.resolve() / name
    workdir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if apex_reports:
        # Exercise profiling shutdown independently of user report opt-outs.
        environment = {key: value for key, value in environment.items()
                       if not key.startswith(("APEX_", "TAU_"))}
        report_directory = workdir / "apex"
        report_directory.mkdir(exist_ok=True)
        environment["APEX_OUTPUT_FILE_PATH"] = str(report_directory)
        environment.update({key: "1" for key in
                            ("APEX_SCREEN_OUTPUT", "APEX_CSV_OUTPUT", "APEX_TASKTREE_OUTPUT")})
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["OMP_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["LD_PRELOAD"] = str(args.guard.resolve())
    environment.pop("HWLOC_COMPONENTS", None)
    # The launcher discovers topology before the application can set its default.
    if ranks > 1 and hwloc_components is None:
        hwloc_components = "-gl"
    if hwloc_components is not None:
        environment["HWLOC_COMPONENTS"] = hwloc_components
    environment["COSMICTIGER_TEST_HWLOC_COMPONENTS"] = (
        "-gl" if hwloc_components is None else hwloc_components)
    environment.pop("COSMICTIGER_TEST_MPI_THREAD_LEVEL", None)
    environment.pop("COSMICTIGER_TEST_MPI_THREAD_RANK", None)
    if level is not None:
        environment["COSMICTIGER_TEST_MPI_THREAD_LEVEL"] = level
    if target_rank is not None:
        environment["COSMICTIGER_TEST_MPI_THREAD_RANK"] = str(target_rank)
    command = [str(args.executable.resolve()), "--help", "--compute.cuda=false",
               f"--hpx:threads={threads}", "--hpx:bind=none"]
    if explicit_position:
        command.append(f"--compute.positionType={args.position_type}")
    if ranks > 1:
        command = ([args.mpi_exec, args.mpi_numproc_flag, str(ranks)]
                   + args.mpi_preflag + command[:1] + args.mpi_postflag + command[1:])
        command.append(f"--hpx:localities={ranks}")
        command.append("--hpx:ini=hpx.parcel.mpi.enable=1")
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
                raise RuntimeError(f"{name}: MPI startup or shutdown timed out; see {log_path}")
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
    parser.add_argument("--position-type", choices=("fixed32", "fixed64"), required=True)
    parser.add_argument("--mpi-exec")
    parser.add_argument("--mpi-numproc-flag", default="-n")
    parser.add_argument("--mpi-preflag", action="append", default=[])
    parser.add_argument("--mpi-postflag", action="append", default=[])
    args = parser.parse_args()
    if args.mpi_exec:
        run_case(args, "multiple_two_ranks", ranks=2)
        # A single worker exposes collective profiling shutdown deadlocks.
        run_case(args, "profiling_shutdown_three_ranks", ranks=3, threads=1,
                 apex_reports=True)
        run_case(args, "serialized_one_rank", level="MPI_THREAD_SERIALIZED",
                 ranks=2, target_rank=1)
    else:
        run_case(args, "multiple")
        run_case(args, "multiple_explicit_position", explicit_position=True)
        run_case(args, "custom_hwloc_components", hwloc_components="-gl,-opencl")
        run_case(args, "custom_hwloc_components_explicit_position", explicit_position=True,
                 hwloc_components="-gl,-opencl")
        for level in ("MPI_THREAD_SINGLE", "MPI_THREAD_FUNNELED", "MPI_THREAD_SERIALIZED"):
            run_case(args, level.lower(), level=level)
        run_case(args, "serialized_explicit_position", level="MPI_THREAD_SERIALIZED", explicit_position=True)
    print("MPI startup passed: hwloc defaults/overrides checked, MULTIPLE accepted, "
          "lower levels rejected, MPI finalized")


if __name__ == "__main__":
    main()
