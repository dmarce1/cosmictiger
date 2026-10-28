"""Check cooperative HPX shutdown and completed profiles after one interrupt."""

import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time

from run_apex_profiler import check_apex, check_samples, require


def run_interrupt_fixture(command, directory, environment, interrupt_signal=signal.SIGINT):
    log_path = directory / "fixture_output.txt"
    ready_path = directory / "interrupt_ready.pid"
    notice = ("Ctrl-C: clean shutdown requested; finishing active work. "
              "Press Ctrl-C again to terminate immediately.")
    with log_path.open("w") as log:
        with subprocess.Popen(command, cwd=directory, env=environment, stdout=log,
                              stderr=subprocess.STDOUT, start_new_session=True) as process:
            try:
                deadline = time.monotonic() + 45
                while not (ready_path.is_file() and ready_path.read_text().strip().isdigit()):
                    require(process.poll() is None,
                            f"interrupt fixture exited before readiness ({process.returncode})")
                    require(time.monotonic() < deadline, "interrupt fixture readiness timed out")
                    time.sleep(0.01)
                pid = int(ready_path.read_text().strip())
                require(pid > 1, "interrupt fixture reported an invalid process ID")
                os.kill(pid, interrupt_signal)
                returncode = process.wait(timeout=60)
            except Exception as exception:
                raise RuntimeError(f"Interrupt fixture failed: {exception}\n" +
                                   log_path.read_text(errors="replace")[-12000:]) from exception
            finally:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)
    output = log_path.read_text(errors="replace")
    require(returncode == 0, f"first interrupt did not exit cleanly ({returncode}):\n{output}")
    require(output.count(notice) == 1, f"first interrupt did not produce one shutdown notice:\n{output}")
    require(output.count("APEX_INTERRUPT_FINALIZED=1") == 1,
            f"first interrupt did not return through HPX finalization:\n{output}")
    require(not any(marker in output for marker in
                    ("Runtime overlap", "SIGABRT", "terminate called", "Aborted")),
            f"runtime failure during interrupt cleanup:\n{output}")
    return subprocess.CompletedProcess(command, returncode, stdout=output)


def run_regression(args, directory):
    require(not any(directory.iterdir()), "interrupt regression output directory must be empty")
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("APEX_", "TAU_"))}
    environment.update({"APEX_OUTPUT_FILE_PATH": str(directory), "APEX_THROTTLE_TIMERS": "0",
                        "HWLOC_COMPONENTS": "-gl", "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1"})
    command = [str(args.executable.resolve()), "--interrupt-exit", "--hpx:threads=2",
               "--hpx:bind=none"]
    if args.sampler_disabled:
        command.append("--sampler-disabled")
    result = run_interrupt_fixture(command, directory, environment,
                                   getattr(signal, args.interrupt_signal))
    if args.require_compiled_disabled:
        require(result.stdout.count("APEX_CPU_SAMPLER_COMPILED=0") == 1,
                "interrupt fixture was not compiled with PROFILE_DISABLE")
    markers = re.findall(r"^APEX_FULL_THREAD_DESCRIPTION=([01])$", result.stdout, re.MULTILINE)
    require(len(markers) == 1, "fixture did not report its FULL description setting")
    require("CPU Timers" not in result.stdout and
            "CosmicTiger statistical CPU profile (" not in result.stdout,
            "interrupt printed detailed profiler reports to the terminal")
    check_apex(directory, markers[0] == "1")
    if args.sampler_disabled:
        require(not (directory / "cpu_samples.txt").exists() and
                not list(directory.glob("cosmictiger.profile.*.txt")),
                "disabled CPU sampling unexpectedly produced reports after an interrupt")
    else:
        check_samples(directory)
        require(len(list(directory.glob("cosmictiger.profile.*.txt"))) == 1,
                "interrupt shutdown did not produce exactly one automatic CPU report")
    print("Interrupt profiler regression passed: cooperative HPX shutdown, APEX reports, " +
          ("disabled CPU sampler" if args.sampler_disabled else "completed CPU samples"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--sampler-disabled", action="store_true")
    parser.add_argument("--require-compiled-disabled", action="store_true")
    parser.add_argument("--interrupt-signal", choices=("SIGINT", "SIGUSR1"), default="SIGINT")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.require_compiled_disabled:
        args.sampler_disabled = True
    if args.output_dir is not None:
        args.output_dir = args.output_dir.resolve()
        args.output_dir.mkdir(parents=True, exist_ok=True)
    with (nullcontext(args.output_dir) if args.output_dir is not None else
          tempfile.TemporaryDirectory(prefix="cosmictiger_interrupt_profiler_")) as output:
        run_regression(args, Path(output))


if __name__ == "__main__":
    main()
