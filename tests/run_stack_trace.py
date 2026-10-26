# CosmicTiger - fatal-signal diagnostics regression; BSL-1.0.
"""Check stack-trace output, fatal exit status, and signal-handler ownership."""

import argparse
from pathlib import Path
import re
import signal
import socket
import subprocess


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run_fixture(executable, workdir, arguments):
    workdir.mkdir(parents=True, exist_ok=True)
    with subprocess.Popen([str(executable), *arguments], cwd=workdir,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True) as process:
        try:
            stdout, stderr = process.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            raise RuntimeError(f"{workdir.name}: stack trace fixture timed out\n{stderr}")
    (workdir / "stdout.log").write_text(stdout)
    (workdir / "stderr.log").write_text(stderr)
    return process.returncode, process.pid, stdout, stderr


def check_trace(output, pid, signal_name, label):
    identity = f"CosmicTiger stacktrace: host={socket.gethostname()} pid={pid} locality=17"
    require(output.count(identity) == 1, f"{label}: missing or duplicate process identity\n{output}")
    require(f"Signal: {signal_name}\n" in output, f"{label}: incorrect signal label\n{output}")
    require("stackTraceTestCrash" in output, f"{label}: missing symbolized crash frame\n{output}")
    require(re.search(r"tests/stack_trace\.cpp:[1-9][0-9]*", output) is not None,
            f"{label}: missing crash source file and line\n{output}")


def check_fatal(executable, workdir, signal_name, *, mode="fatal", output_path=None):
    arguments = [mode, signal_name]
    if output_path is not None:
        arguments.append(str(output_path))
    returncode, pid, stdout, stderr = run_fixture(executable, workdir, arguments)
    require(returncode == -getattr(signal, signal_name),
            f"{workdir.name}: expected termination by {signal_name}, got {returncode}\n{stderr}")
    check_trace(stderr, pid, signal_name, f"{workdir.name} stderr")
    require("CosmicTiger stacktrace:" not in stdout,
            f"{workdir.name}: fatal diagnostic was sent to stdout")
    return pid, stderr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    args = parser.parse_args()
    executable = args.executable.resolve()
    workdir = args.workdir.resolve()

    lifecycle_dir = workdir / "lifecycle"
    returncode, _, _, stderr = run_fixture(executable, lifecycle_dir, ["lifecycle"])
    require(returncode == 0, f"signal-handler lifecycle failed ({returncode})\n{stderr}")
    require(not (lifecycle_dir / "stack_trace.txt").exists(),
            "normal handler lifetime unexpectedly wrote a crash report")

    for signal_name in ("SIGABRT", "SIGSEGV", "SIGBUS", "SIGFPE", "SIGILL"):
        case_dir = workdir / signal_name.lower()
        case_dir.mkdir(parents=True, exist_ok=True)
        output_path = case_dir / "configured_trace.txt"
        prefix = "Existing diagnostic data\n"
        output_path.write_text(prefix)
        pid, stderr = check_fatal(executable, case_dir, signal_name, output_path=output_path)
        report = output_path.read_text()
        require(report.startswith(prefix), f"{signal_name}: existing output was overwritten")
        check_trace(report, pid, signal_name, f"{signal_name} file")
        require(report[len(prefix):] == stderr,
                f"{signal_name}: file and stderr diagnostics differ")
        require(not (case_dir / "stack_trace.txt").exists(),
                f"{signal_name}: configured output path was ignored")

    default_dir = workdir / "default_append"
    default_path = default_dir / "stack_trace.txt"
    default_dir.mkdir(parents=True, exist_ok=True)
    default_path.unlink(missing_ok=True)
    _, first_report = check_fatal(executable, default_dir, "SIGABRT")
    _, second_report = check_fatal(executable, default_dir, "SIGSEGV")
    require(default_path.read_text() == first_report + second_report,
            "default output did not append successive crash reports")

    worker_dir = workdir / "worker"
    pid, stderr = check_fatal(executable, worker_dir, "SIGABRT", mode="worker")
    report = (worker_dir / "stack_trace.txt").read_text()
    check_trace(report, pid, "SIGABRT", "worker file")
    require(report.endswith(stderr), "worker file omitted the current crash report")

    unavailable_dir = workdir / "unavailable_output"
    unavailable_path = unavailable_dir / "missing_parent" / "trace.txt"
    check_fatal(executable, unavailable_dir, "SIGSEGV", output_path=unavailable_path)
    require(not unavailable_path.exists(), "unavailable output path unexpectedly exists")

    interrupt_dir = workdir / "interrupt"
    returncode, _, _, stderr = run_fixture(executable, interrupt_dir, ["interrupt"])
    require(returncode == -signal.SIGINT, f"SIGINT exit status changed ({returncode})\n{stderr}")
    require("CosmicTiger stacktrace:" not in stderr,
            "SIGINT unexpectedly triggered a fatal diagnostic")
    require(not (interrupt_dir / "stack_trace.txt").exists(),
            "SIGINT unexpectedly wrote a crash report")
    print("Stack trace passed: fatal signals, symbols, source lines, output paths, and ownership")


if __name__ == "__main__":
    main()
