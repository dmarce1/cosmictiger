"""Check two-rank CosmicTiger startup and clean APEX/TAU profiling shutdown."""

import argparse
from contextlib import nullcontext
import csv
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile

from run_apex_profiler import require


def check_reports(directory, output):
    profiles = sorted(path for path in (directory / "tau").rglob("profile.*")
                      if re.fullmatch(r"profile\.\d+\.\d+\.\d+", path.name))
    ranks = {path.name.split(".")[1] for path in profiles}
    require(ranks == {"0", "1"},
            f"TAU profiles must contain MPI ranks 0 and 1; found {sorted(ranks)}")
    csv_path = directory / "apex" / "apex_profiles.csv"
    require(csv_path.is_file(), "APEX did not produce its combined CSV profile")
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(rows and {row.get("rank") for row in rows} == {"0", "1"},
            "APEX CSV profile must contain both MPI ranks")
    tree_path = directory / "apex" / "apex_tasktree.csv"
    require(tree_path.is_file() and tree_path.stat().st_size > 0,
            "APEX did not produce its combined task tree")
    require("Runtime overlap" not in output,
            "TAU reported overlapping runtime timers during MPI startup or shutdown")
    merge_threads = re.findall(
        r"Merging common tree for all ranks\.\.\. done in [^\r\n]* with (\d+) threads?\.", output)
    require(merge_threads == ["1"],
            "APEX must merge the MPI task tree once using one thread during shutdown; "
            f"observed thread counts {merge_threads}")
    return len(profiles)


def run_regression(args, directory):
    require(not any(directory.iterdir()), "TAU MPI regression output directory must be empty")
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("APEX_", "TAU_"))}
    environment.update({"TAU_EXEC": str(args.tau_exec.resolve()),
                        "COSMICTIGER_PROFILE_DIR": str(directory),
                        "PROFILEDIR": str(directory / "tau"),
                        "HWLOC_COMPONENTS": "-gl", "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1"})
    command = ([args.mpi_exec, args.mpi_numproc_flag, "2"] + args.mpi_preflag +
               [str(args.launcher.resolve()), str(args.executable.resolve())] +
               args.mpi_postflag + ["--help", "--hpx:threads=2", "--hpx:bind=none"])
    log_path = directory / "startup.log"
    with log_path.open("w") as log:
        with subprocess.Popen(command, cwd=directory, env=environment,
                              stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True) as process:
            try:
                returncode = process.wait(timeout=90)
            except subprocess.TimeoutExpired as exception:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                output = log_path.read_text(errors="replace")
                raise RuntimeError("Two-rank APEX/TAU startup or shutdown timed out.\n" +
                                   output[-12000:]) from exception
    output = log_path.read_text(errors="replace")
    try:
        require(returncode == 0, f"Two-rank CosmicTiger exited with status {returncode}")
        require("CosmicTiger options" in output, "CosmicTiger did not reach its --help output")
        profile_count = check_reports(directory, output)
    except Exception as exception:
        raise RuntimeError(f"{exception}\nMPI application output:\n{output[-12000:]}") from exception
    print(f"TAU MPI startup regression passed: {profile_count} thread profiles across ranks 0 and 1; "
          "combined APEX CSV and task tree; one-thread tree merge; clean shutdown")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("launcher", type=Path)
    parser.add_argument("tau_exec", type=Path)
    parser.add_argument("mpi_exec")
    parser.add_argument("--mpi-numproc-flag", default="-n")
    parser.add_argument("--mpi-preflag", action="append", default=[])
    parser.add_argument("--mpi-postflag", action="append", default=[])
    parser.add_argument("--output-dir", type=Path,
                        help="retain evidence in an empty directory instead of a temporary one")
    args = parser.parse_args()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.resolve()
        args.output_dir.mkdir(parents=True, exist_ok=True)
    with (nullcontext(args.output_dir) if args.output_dir is not None else
          tempfile.TemporaryDirectory(prefix="cosmictiger_tau_mpi_startup_")) as output:
        run_regression(args, Path(output))


if __name__ == "__main__":
    main()
