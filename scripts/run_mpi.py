#!/usr/bin/env python3
"""Run CosmicTiger with Open MPI and cooperative first-Ctrl-C shutdown."""

import argparse
import ctypes
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def parent_death_signal(number, parent):
    """Have Linux terminate this owned process if its supervisor disappears."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, number, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "Unable to set parent-death signal")
    if os.getppid() != parent:
        os.kill(os.getpid(), number)


def child_setup(number):
    parent = os.getpid()
    return lambda: parent_death_signal(number, parent)


class Signals:
    def __init__(self):
        self.interrupts = 0
        self.stop = False
        self.terminate = False
        for number in (signal.SIGINT, signal.SIGUSR1, signal.SIGTERM, signal.SIGHUP):
            signal.signal(number, self.receive)

    def receive(self, number, _frame):
        if number == signal.SIGINT:
            self.interrupts += 1
        elif number == signal.SIGUSR1:
            self.stop = True
        else:
            self.terminate = True


def send(process, number, group=False):
    try:
        if group:
            os.killpg(process.pid, number)
        else:
            process.send_signal(number)
    except ProcessLookupError:
        pass


def cleanup(process):
    if process is None or process.poll() is not None:
        return
    send(process, signal.SIGTERM, group=True)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        send(process, signal.SIGKILL, group=True)
        process.wait()


def exit_code(code):
    return code if code >= 0 else 128 - code


def run_rank(arguments):
    state_directory = Path(arguments[0])
    command = arguments[2:]
    if arguments[1] != "--" or not command:
        raise ValueError("Invalid internal rank command")
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    state = Signals()
    parent_death_signal(signal.SIGTERM, os.getppid())
    # The outer supervisor sends no SIGUSR1 until every rank can receive it.
    (state_directory / f"rank-{rank}.ready").touch()
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    environment = os.environ.copy()
    environment["COSMICTIGER_INTERRUPT_READY_FD"] = str(write_fd)
    process = None
    try:
        process = subprocess.Popen(
            command, env=environment, pass_fds=(write_fd,), start_new_session=True,
            preexec_fn=child_setup(signal.SIGKILL),
        )
        os.close(write_fd)
        write_fd = -1
        ready = bytearray()
        application_pid = None
        notified = False
        while process.poll() is None:
            if state.terminate or state.interrupts:
                cleanup(process)
                return 130
            if application_pid is None:
                try:
                    ready.extend(os.read(read_fd, 128))
                except BlockingIOError:
                    pass
                if len(ready) > 128:
                    raise RuntimeError("Invalid application readiness response")
                if b"\n" in ready:
                    application_pid = int(ready.split(b"\n", 1)[0])
                    # Only signal an application inside the group we created.
                    try:
                        application_group = os.getpgid(application_pid)
                    except ProcessLookupError:
                        application_group = process.pid  # Already finished.
                    if application_group != process.pid:
                        raise RuntimeError("Application readiness PID is outside its owned group")
            if state.stop and application_pid is not None and not notified:
                try:
                    os.kill(application_pid, signal.SIGUSR1)
                except ProcessLookupError:
                    pass
                notified = True
            time.sleep(0.02)
        return exit_code(process.returncode)
    finally:
        cleanup(process)
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)


def main():
    if sys.platform != "linux":
        raise RuntimeError("This Open MPI supervisor currently requires Linux")
    if sys.argv[1:2] == ["--rank"]:
        return run_rank(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--np", type=int, required=True, help="number of MPI ranks")
    parser.add_argument("--mpiexec", default=os.environ.get("MPIEXEC", "mpiexec"))
    parser.add_argument("--mpi-arg", action="append", default=[],
                        help="extra launcher argument; use --mpi-arg=--option")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="executable or scripts/profile_tau.sh followed by its arguments")
    args = parser.parse_args()
    if args.np < 1 or not args.command:
        parser.error("a positive rank count and an application command are required")
    if args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("an application command is required")
    state = Signals()
    process = None
    # The working directory must be visible at the same path on every host,
    # just like this script and the application executable.
    with tempfile.TemporaryDirectory(prefix=".cosmictiger_mpi_", dir=Path.cwd()) as directory:
        command = [args.mpiexec, *args.mpi_arg, "-n", str(args.np), sys.executable,
                   str(Path(__file__).resolve()), "--rank", directory, "--", *args.command]
        try:
            process = subprocess.Popen(command, start_new_session=True,
                                       preexec_fn=child_setup(signal.SIGTERM))
            requested = False
            forwarded = False
            emergency_deadline = None
            while process.poll() is None:
                if (state.interrupts or state.stop) and not requested:
                    print("Ctrl-C: requesting clean shutdown; waiting for rank/application readiness. "
                          "Press Ctrl-C again to abort immediately.", file=sys.stderr, flush=True)
                    requested = True
                if (state.interrupts > 1 or state.terminate) and emergency_deadline is None:
                    send(process, signal.SIGINT if not state.terminate else signal.SIGTERM)
                    emergency_deadline = time.monotonic() + 3
                elif requested and not forwarded and emergency_deadline is None:
                    if all((Path(directory) / f"rank-{rank}.ready").exists()
                           for rank in range(args.np)):
                        send(process, signal.SIGUSR1)
                        forwarded = True
                if emergency_deadline is not None and time.monotonic() >= emergency_deadline:
                    cleanup(process)
                    return 130
                time.sleep(0.02)
            return exit_code(process.returncode)
        finally:
            cleanup(process)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"run_mpi.py: {error}", file=sys.stderr)
        sys.exit(1)
