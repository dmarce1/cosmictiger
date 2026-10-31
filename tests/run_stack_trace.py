# CosmicTiger - fatal-signal diagnostics regression; BSL-1.0.
"""Check stack-trace output, fatal exit status, and signal-handler ownership."""

import argparse
import os
from pathlib import Path
import re
import select
import signal
import socket
import subprocess
import time


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


def check_blocked_reporter(executable, workdir, *, recursive=False):
    """Hold the first handler in FIFO write until the second fault is observed."""
    workdir.mkdir(parents=True, exist_ok=True)
    fifo_path = workdir / "report.fifo"
    fifo_path.unlink(missing_ok=True)
    os.mkfifo(fifo_path)
    fifo = os.open(fifo_path, os.O_RDWR | os.O_NONBLOCK)
    filler_size = 0
    stderr = bytearray()
    fifo_output = bytearray()
    stdout = b""
    waiting = b"Another thread received a fatal signal; waiting for the crash report.\n"
    try:
        # Keep a reader attached but fill the pipe completely. The handler's
        # open succeeds, and its stderr header precedes its blocked FIFO write.
        for chunk in (b"x" * 4096, b"x"):
            while True:
                try:
                    filler_size += os.write(fifo, chunk)
                except BlockingIOError:
                    break
        mode = "recursive" if recursive else "concurrent"
        with subprocess.Popen([str(executable), mode, "SIGABRT", str(fifo_path)],
                              cwd=workdir, stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              bufsize=0) as process:
            def await_stderr(marker):
                deadline = time.monotonic() + 10.0
                while marker not in stderr:
                    remaining = deadline - time.monotonic()
                    require(remaining > 0, f"{mode}: timed out waiting for {marker!r}")
                    ready, _, _ = select.select([process.stderr], [], [], remaining)
                    require(ready, f"{mode}: missing signal synchronization marker {marker!r}")
                    chunk = os.read(process.stderr.fileno(), 4096)
                    require(chunk, f"{mode}: terminated before {marker!r}: {stderr!r}")
                    stderr.extend(chunk)

            try:
                await_stderr(b"locality=17\n")
                process.stdin.write(b"2")
                process.stdin.close()
                process.stdin = None
                if not recursive:
                    # Observing this message proves the second thread entered
                    # the production wait path before the first can finish.
                    await_stderr(waiting)
                    deadline = time.monotonic() + 20.0
                    readers = [fifo, process.stderr.fileno()]
                    while process.poll() is None:
                        remaining = deadline - time.monotonic()
                        require(remaining > 0, "concurrent reporters did not terminate")
                        ready, _, _ = select.select(readers, [], [], min(remaining, 0.1))
                        for descriptor in ready:
                            chunk = os.read(descriptor, 65536)
                            if descriptor == fifo:
                                fifo_output.extend(chunk)
                            elif chunk:
                                stderr.extend(chunk)
                            else:
                                readers.remove(descriptor)
                    while True:
                        try:
                            fifo_output.extend(os.read(fifo, 65536))
                        except BlockingIOError:
                            break
                stdout, tail = process.communicate(timeout=10)
                stderr.extend(tail)
            except BaseException:
                if process.poll() is None:
                    process.kill()
                stdout, tail = process.communicate(timeout=5)
                stderr.extend(tail)
                raise
            finally:
                (workdir / "stdout.log").write_bytes(stdout)
                (workdir / "stderr.log").write_bytes(stderr)
            expected_signal = signal.SIGSEGV if recursive else signal.SIGABRT
            require(process.returncode == -expected_signal,
                    f"{mode}: wrong fatal signal {process.returncode}: {stderr!r}")
            if recursive:
                require(waiting not in stderr, "reporter recursion incorrectly waited for itself")
            else:
                require(fifo_output[:filler_size] == b"x" * filler_size,
                        "concurrent fixture lost its FIFO synchronization prefix")
                report = fifo_output[filler_size:].decode(errors="replace")
                (workdir / "report.txt").write_text(report)
                check_trace(report, process.pid, "SIGABRT", "concurrent file")
                require(stderr.replace(waiting, b"").decode(errors="replace") == report,
                        "concurrent crash truncated or duplicated the reporter's trace")
                require(not stdout, "concurrent fatal diagnostic was sent to stdout")
    finally:
        os.close(fifo)
        fifo_path.unlink(missing_ok=True)


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

    check_blocked_reporter(executable, workdir / "concurrent")
    check_blocked_reporter(executable, workdir / "recursive", recursive=True)

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
    print("Stack trace passed: fatal signals, concurrent faults, recursion, symbols, source lines, "
          "output paths, and ownership")


if __name__ == "__main__":
    main()
