#!/usr/bin/env python3
"""Check the MPI supervisor's signal forwarding without an MPI installation."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


FAKE_MPI = r'''#!/usr/bin/env python3
import os, signal, subprocess, sys, time
children = []
stop = False
def receive(number, frame):
    global stop
    if number != signal.SIGUSR1:
        stop = True
    for child in children:
        if child.poll() is None:
            child.send_signal(signal.SIGUSR1 if number == signal.SIGUSR1 else signal.SIGTERM)
signal.signal(signal.SIGUSR1, receive)
signal.signal(signal.SIGINT, receive)
signal.signal(signal.SIGTERM, receive)
count = int(sys.argv[2])
time.sleep(0.2)
for rank in range(count):
    env = os.environ.copy()
    env['OMPI_COMM_WORLD_RANK'] = str(rank)
    children.append(subprocess.Popen(sys.argv[3:], env=env))
while any(child.poll() is None for child in children):
    time.sleep(0.02)
sys.exit(130 if stop else max(child.returncode for child in children))
'''

FAKE_APP = r'''import os, signal, sys, time
from pathlib import Path
rank = os.environ['OMPI_COMM_WORLD_RANK']
base = Path(sys.argv[1])
mode = sys.argv[2]
(base / ('pid-' + rank)).write_text(str(os.getpid()))
stop = False
def receive(number, frame):
    global stop
    stop = True
    (base / ('notified-' + rank)).write_text('yes')
# A signal during this interval would kill the app without the readiness gate.
time.sleep(0.3)
signal.signal(signal.SIGUSR1, receive)
fd = int(os.environ['COSMICTIGER_INTERRUPT_READY_FD'])
os.write(fd, (str(os.getpid()) + '\n').encode())
os.close(fd)
if mode == 'exit':
    sys.exit(7)
while not stop or mode == 'stubborn':
    time.sleep(0.02)
(base / ('clean-' + rank)).write_text('yes')
'''


def wait_until(predicate, description, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(f"Timed out: {description}")


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


def exercise(wrapper, base, fake_mpi, app, mode):
    base.mkdir()
    log_path = base / "launcher.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(wrapper), "-n", "2", "--mpiexec", str(fake_mpi),
             sys.executable, str(app), str(base), "stubborn" if mode == "second" else mode],
            cwd=base, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            wait_until(lambda: len(list(base.glob("pid-*"))) == 2, "applications started")
            pids = [int(path.read_text()) for path in base.glob("pid-*")]
            if mode == "clean":
                # Exercise actual terminal-style process-group delivery, before
                # the applications have installed their SIGUSR1 handlers.
                os.killpg(process.pid, signal.SIGINT)
                assert process.wait(timeout=8) == 0, log_path.read_text()
                assert len(list(base.glob("clean-*"))) == 2, log_path.read_text()
            elif mode == "second":
                os.killpg(process.pid, signal.SIGINT)
                wait_until(lambda: len(list(base.glob("notified-*"))) == 2,
                           "first interrupt reached both applications")
                os.killpg(process.pid, signal.SIGINT)
                assert process.wait(timeout=8) != 0
            elif mode == "death":
                os.kill(process.pid, signal.SIGKILL)
                process.wait(timeout=8)
            else:
                assert process.wait(timeout=8) == 7, log_path.read_text()
            wait_until(lambda: not any(alive(pid) for pid in pids), "owned applications exited")
        finally:
            if process.poll() is None:
                os.kill(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.kill(process.pid, signal.SIGKILL)
                    process.wait()
            for path in base.glob("pid-*"):
                pid = int(path.read_text())
                if alive(pid):
                    os.killpg(pid, signal.SIGKILL)


def main():
    wrapper = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else (
        Path(__file__).resolve().parents[1] / "scripts/run_mpi.py")
    with tempfile.TemporaryDirectory(prefix="cosmictiger_mpi_launcher_") as directory:
        root = Path(directory)
        fake_mpi = root / "fake_mpiexec"
        fake_mpi.write_text(FAKE_MPI)
        fake_mpi.chmod(0o755)
        app = root / "fake_application.py"
        app.write_text(FAKE_APP)
        for mode in ("clean", "second", "death", "exit"):
            exercise(wrapper, root / mode, fake_mpi, app, mode)
            print(f"MPI launcher {mode}: passed")


if __name__ == "__main__":
    main()
