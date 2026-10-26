# CosmicTiger - HPX locality stack-trace regression; BSL-1.0.
"""Verify stack-trace startup on console and worker HPX localities."""

import argparse
import os
from pathlib import Path
import re
import signal
import subprocess


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run_case(args, name, *, ranks=1, crash_locality=None):
    workdir = args.workdir.resolve() / name
    workdir.mkdir(parents=True, exist_ok=True)
    trace_path = workdir / "stack_trace.txt"
    trace_path.unlink(missing_ok=True)
    command = [str(args.executable.resolve()), f"--expected-localities={ranks}",
               f"--hpx:localities={ranks}", "--hpx:threads=2", "--hpx:bind=none"]
    if crash_locality is not None:
        command.append(f"--crash-locality={crash_locality}")
    if ranks > 1:
        command.extend(["--hpx:ini=hpx.parcel.bootstrap=mpi",
                        "--hpx:ini=hpx.parcel.mpi.enable=1",
                        "--hpx:ini=hpx.parcel.tcp.enable=0"])
        command = ([args.mpi_exec, args.mpi_numproc_flag, str(ranks)]
                   + args.mpi_preflag + command[:1] + args.mpi_postflag + command[1:])
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    log_path = workdir / "startup.log"
    with log_path.open("w") as log:
        with subprocess.Popen(command, cwd=workdir, env=environment, text=True,
                              stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True) as process:
            try:
                returncode = process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError(f"{name}: HPX fixture timed out; see {log_path}")
    output = log_path.read_text()
    for locality in range(ranks):
        main_calls = 1 if locality == 0 else 0
        marker = (f"STACK_TRACE_LOCALITY locality={locality} "
                  f"hpx_main_calls={main_calls} installed=1")
        require(output.count(marker) == 1,
                f"{name}: missing or repeated locality startup marker; see {log_path}")
    if crash_locality is None:
        require(returncode == 0, f"{name}: startup failed ({returncode}); see {log_path}")
        require(not trace_path.exists(), f"{name}: successful startup wrote a crash report")
        return
    if ranks == 1:
        require(returncode == -signal.SIGABRT,
                f"{name}: expected SIGABRT termination, got {returncode}; see {log_path}")
    else:
        require(returncode != 0,
                f"{name}: MPI launcher accepted a crashed worker; see {log_path}")
    require(trace_path.exists(), f"{name}: crash report missing; see {log_path}")
    report = trace_path.read_text()
    identity = rf"CosmicTiger stacktrace: host=\S+ pid=\d+ locality={crash_locality}\b"
    for label, diagnostic in (("stderr", output), ("file", report)):
        require(re.search(identity, diagnostic) is not None,
                f"{name}: {label} lacks crashing locality identity; see {log_path}")
        require("Signal: SIGABRT\n" in diagnostic,
                f"{name}: {label} lacks SIGABRT label; see {log_path}")
        require("stackTraceHpxCrash" in diagnostic,
                f"{name}: {label} lacks crashing frame; see {log_path}")
        require(re.search(r"tests/stack_trace_hpx\.cpp:[1-9][0-9]*", diagnostic) is not None,
                f"{name}: {label} lacks crash source line; see {log_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--mpi-exec", "--mpiexec", dest="mpi_exec")
    parser.add_argument("--mpi-numproc-flag", default="-n")
    parser.add_argument("--mpi-preflag", action="append", default=[])
    parser.add_argument("--mpi-postflag", action="append", default=[])
    parser.add_argument("--mpi-only", action="store_true")
    args = parser.parse_args()
    if args.mpi_only and not args.mpi_exec:
        parser.error("--mpi-only requires --mpi-exec")
    if not args.mpi_only:
        run_case(args, "local_startup")
        run_case(args, "local_crash", crash_locality=0)
    if args.mpi_exec:
        run_case(args, "mpi_startup", ranks=2)
        run_case(args, "mpi_worker_crash", ranks=2, crash_locality=1)
    print("HPX stack trace passed: locality startup, fatal signal, source line, and crash file")


if __name__ == "__main__":
    main()
