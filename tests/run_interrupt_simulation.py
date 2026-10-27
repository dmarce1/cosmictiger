"""Interrupt a small running simulation and verify orderly checkpoint/profile delivery."""

import argparse
from contextlib import nullcontext
import csv
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

from run_apex_profiler import require
from run_checkpoint import checkManifest, checkpoint_position_bytes, fnv1a64
from run_tau_profiler import read_tau_profiles


END_TIME = 0.001
STEP_COUNT = 10000
RANK_RECORDER = """
import os
from pathlib import Path
import sys
rank = sys.argv[2]
if rank == 'auto':
    rank = next((os.environ[name] for name in
                 ('OMPI_COMM_WORLD_RANK', 'PMI_RANK', 'PMIX_RANK') if name in os.environ), None)
    if rank is None:
        raise RuntimeError('MPI launcher did not supply a rank for the interrupt test')
Path(sys.argv[1], f'rank.{int(rank)}.pid').write_text(str(os.getpid()) + '\\n')
os.execvpe(sys.argv[3], sys.argv[3:], os.environ)
"""


def progress_rows(path, scenario):
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines(keepends=True):
        if not line.endswith("\n"):
            continue
        try:
            values = [float(value) for value in line.split()]
        except ValueError:
            continue
        if len(values) == (5 if scenario == "hydroWave" else 14):
            require(all(math.isfinite(value) for value in values), "nonfinite simulation progress")
            rows.append(values)
    return rows


def check_checkpoint(directory, iteration, scenario, ranks):
    gravity = scenario == "plummer"
    if ranks == 1:
        checkManifest(directory, iteration, scenario=scenario, endTime=END_TIME, gravity=gravity)
        return
    lines = (directory / "checkpoint.meta").read_text().splitlines()
    require(len(lines) == ranks + 2 and lines[0].split() ==
            ["COSMICTIGER_CHECKPOINT", "2", "3", str(iteration), str(ranks)],
            "interrupted simulation did not publish a complete distributed checkpoint")
    seen = set()
    for line in lines[1:1 + ranks]:
        rank, size, checksum = line.split()
        require(rank not in seen, "checkpoint manifest repeats a rank")
        seen.add(rank)
        payload = (directory / f"checkpoint.{rank}.dat").read_bytes()
        require(len(payload) == int(size) and fnv1a64(payload) == int(checksum, 16),
                f"interrupted checkpoint rank {rank} has an invalid size or checksum")
        checkpoint_position_bytes(payload)
    require(seen == {str(rank) for rank in range(ranks)}, "checkpoint is missing an MPI rank")
    label, size, checksum = lines[-1].split()
    payload = (directory / "scenario.meta").read_bytes()
    require(label == "SCENARIO" and len(payload) == int(size) and
            fnv1a64(payload) == int(checksum, 16), "checkpoint scenario metadata is incomplete")
    metadata = payload.decode().splitlines()
    require(len(metadata) == 4 and metadata[:2] == ["COSMICTIGER_SCENARIO 1", scenario] and
            float(metadata[2]) == END_TIME and metadata[3] == str(int(gravity)),
            "checkpoint scenario metadata differs from the interrupted simulation")


def check_profiles(directory, ranks, tau, sampler_disabled):
    expected_ranks = {str(rank) for rank in range(ranks)}
    with (directory / "apex" / "apex_profiles.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(rows and {row["rank"] for row in rows} == expected_ranks,
            "APEX did not finalize its profiles for every interrupted rank")
    with (directory / "apex" / "apex_tasktree.csv").open(newline="") as stream:
        tree = list(csv.DictReader(stream))
    require(tree and {row["process rank"] for row in tree} == expected_ranks,
            "APEX did not finalize its task tree for every interrupted rank")
    cpu_reports = list(directory.glob("cosmictiger.profile.*.txt"))
    if sampler_disabled:
        require(not cpu_reports, "disabled CPU sampling produced reports")
    else:
        require(len(cpu_reports) == ranks, "missing or duplicate final CPU profiles")
        for report in cpu_reports:
            text = report.read_text()
            require("Sampling is still active" not in text, "interrupt left an active CPU sampler")
            diagnostics = re.search(
                r"Delivered samples: (\d+); dropped samples: (\d+); "
                r"timer overruns \(missing ticks\): (\d+); setup failures: (\d+)", text)
            require(diagnostics is not None and int(diagnostics[4]) == 0,
                    f"CPU sampling did not finalize successfully: {report}")
    if tau:
        profiles, rows = read_tau_profiles(directory / "tau")
        require({path.name.split(".")[1] for path in profiles} == expected_ranks,
                "TAU did not finalize its profiles for every interrupted rank")
        require(any("[SAMPLE]" in row["name"] and row["calls"] > 0 for row in rows),
                "TAU did not retain ordinary-function samples from the simulation")


def run_regression(args, directory):
    require(not any(directory.iterdir()), "simulation interrupt output directory must be empty")
    ranks = 2 if args.mpi_exec or args.mpi_wrapper else 1
    target_rank = args.signal_rank if args.signal_rank is not None else ranks - 1
    require(0 <= target_rank < ranks, "signal rank is outside this test's MPI ranks")
    require((args.launcher is None) == (args.tau_exec is None),
            "TAU simulation tests require both --launcher and --tau-exec")
    source = Path(__file__).resolve().parents[1]
    preset = "hydro_wave" if args.scenario == "hydroWave" else "plummer"
    simulation = directory / "simulation"
    (directory / "apex").mkdir()
    options = {"configuration": source / "configurations" / f"{preset}.ini",
               "compute.cuda": "false", "compute.positionType": "fixed32",
               "initial.particlesPerDimension": 8, "initial.closePack": "false",
               "initial.useGlass": "false", "chemistry.enabled": "false",
               "gravity.enabled": str(args.scenario == "plummer").lower(),
               "gravity.meshSize": 8, "gravity.pmmmNearWidth": 2,
               "run.endTime": END_TIME, "run.numberOfSteps": STEP_COUNT,
               "run.maxIterations": 1000000, "run.verbose": "true",
               "analysis.groups": "false", "analysis.tracers": "false",
               "analysis.views": "false", "analysis.power": "false",
               "lightcone.enabled": "false", "output.silo": "false",
               "output.directory": simulation, "checkpoint.intervalSeconds": 1000000000,
               "hpx:threads": 2, "hpx:bind": "none"}
    if ranks > 1:
        options["hpx:localities"] = ranks
    application = [str(args.executable.resolve())] + args.mpi_postflag + [
        f"--{key}={value}" for key, value in options.items()]
    if args.launcher:
        application.insert(0, str(args.launcher.resolve()))
    recorded = [sys.executable, "-c", RANK_RECORDER, str(directory),
                "auto" if ranks > 1 else "0"] + application
    if args.mpi_wrapper:
        require(args.mpi_numproc_flag == "-n", "outer wrapper uses the standard -n rank flag")
        command = [sys.executable, str(args.mpi_wrapper.resolve()), "-n", str(ranks),
                   "--mpiexec", args.mpi_exec or "mpiexec"]
        command += [f"--mpi-arg={flag}" for flag in args.mpi_preflag]
        command += recorded
    elif ranks > 1:
        command = [args.mpi_exec, args.mpi_numproc_flag, str(ranks)] + args.mpi_preflag + recorded
    else:
        command = recorded
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("APEX_", "TAU_"))}
    environment.update({"APEX_OUTPUT_FILE_PATH": str(directory / "apex"),
                        "COSMICTIGER_PROFILE_DIR": str(directory),
                        "PROFILEDIR": str(directory / "tau"), "APEX_THROTTLE_TIMERS": "0",
                        "HWLOC_COMPONENTS": "-gl", "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""})
    if args.tau_exec:
        environment["TAU_EXEC"] = str(args.tau_exec.resolve())
    log_path = directory / "simulation.log"
    with log_path.open("w") as log:
        with subprocess.Popen(command, cwd=directory, env=environment, stdout=log,
                              stderr=subprocess.STDOUT, start_new_session=True) as process:
            try:
                deadline = time.monotonic() + 90
                while not (progress_rows(simulation / "progress.txt", args.scenario) and
                           all((directory / f"rank.{rank}.pid").is_file() for rank in range(ranks))):
                    require(process.poll() is None,
                            f"simulation exited before a completed step ({process.returncode})")
                    require(time.monotonic() < deadline, "simulation did not finish its first step")
                    time.sleep(0.01)
                if args.mpi_wrapper:
                    # Reproduce terminal Ctrl-C on the outer process group; its
                    # MPI launcher runs in a separate session under the wrapper.
                    os.killpg(process.pid, signal.SIGINT)
                else:
                    pid = int((directory / f"rank.{target_rank}.pid").read_text())
                    require(pid > 1, "rank recorder returned an invalid process ID")
                    os.kill(pid, signal.SIGINT)
                returncode = process.wait(timeout=90)
            except Exception as exception:
                raise RuntimeError(f"Simulation interrupt failed: {exception}\n" +
                                   log_path.read_text(errors="replace")[-16000:]) from exception
            finally:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
    output = log_path.read_text(errors="replace")
    try:
        require(returncode == 0, f"interrupted simulation did not exit normally ({returncode})")
        require("Ctrl-C: clean shutdown requested" in output,
                "simulation did not acknowledge the interrupt request")
        require(not any(marker in output for marker in
                        ("Runtime overlap", "SIGABRT", "terminate called", "Aborted")),
                "runtime failure during simulation cleanup")
        rows = progress_rows(simulation / "progress.txt", args.scenario)
        require(rows, "interrupted simulation lost its completed progress rows")
        iteration = int(rows[-1][0]) if args.scenario == "hydroWave" else int(rows[-1][1]) + 1
        require(rows[-1][1] < END_TIME if args.scenario == "hydroWave" else
                rows[-1][5] < STEP_COUNT, "simulation reached its end before interruption")
        check_checkpoint(simulation / f"checkpoint.{iteration}", iteration, args.scenario, ranks)
        check_profiles(directory, ranks, args.launcher is not None, args.sampler_disabled)
    except Exception as exception:
        raise RuntimeError(f"{exception}\nSimulation output:\n{output[-16000:]}") from exception
    recipient = "outer MPI wrapper" if args.mpi_wrapper else f"application rank {target_rank}"
    print(f"Simulation interrupt passed: {args.scenario}, {ranks} ranks, signal to {recipient}; "
          f"normal exit, final checkpoint {iteration}, and completed profiling reports")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--scenario", choices=("plummer", "hydroWave"), required=True)
    parser.add_argument("--launcher", type=Path, help="TAU profiling launcher")
    parser.add_argument("--tau-exec", type=Path)
    parser.add_argument("--mpi-exec")
    parser.add_argument("--mpi-numproc-flag", default="-n")
    parser.add_argument("--mpi-preflag", action="append", default=[])
    parser.add_argument("--mpi-postflag", action="append", default=[])
    parser.add_argument("--mpi-wrapper", type=Path, help="test terminal Ctrl-C through run_mpi.py")
    parser.add_argument("--signal-rank", type=int, help="default: worker rank 1 for MPI, otherwise 0")
    parser.add_argument("--sampler-disabled", action="store_true",
                        help="expect an application built without the CPU sampler")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.resolve()
        args.output_dir.mkdir(parents=True, exist_ok=True)
    with (nullcontext(args.output_dir) if args.output_dir is not None else
          tempfile.TemporaryDirectory(prefix="cosmictiger_interrupt_simulation_")) as output:
        run_regression(args, Path(output))


if __name__ == "__main__":
    main()
