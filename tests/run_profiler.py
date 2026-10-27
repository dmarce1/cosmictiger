"""Check sampler attribution, diagnostics, and normal/Ctrl-C final report delivery."""

import argparse
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def summary(report):
    match = re.search(
        r"Delivered samples: (\d+); dropped samples: (\d+); "
        r"timer overruns \(missing ticks\): (\d+); setup failures: (\d+)", report)
    require(match is not None, "missing sample-loss and setup-failure diagnostics")
    return tuple(int(value) for value in match.groups())


def run(executable, directory, arguments=()):
    result = subprocess.run([str(executable), *arguments], cwd=directory,
                            text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=45)
    (directory / "run.log").write_text(result.stdout)
    require(result.returncode == 0,
            f"profiler executable failed ({result.returncode}):\n{result.stdout}")
    return result.stdout


def check_report(report):
    require("Sampling is still active" not in report, "final report is not stopped")
    samples, dropped, overruns, failures = summary(report)
    require(samples > 0, "CPU work produced no timer samples")
    require(dropped == 0 and failures == 0, "samples were dropped or timer setup failed")
    workers = re.search(r"Attached OS workers: (\d+); registered function labels: (\d+)", report)
    require(workers is not None and int(workers[1]) >= 3,
            "main thread and both HPX workers were not attached")
    require(int(workers[2]) > 0, "PROFILE did not register runtime metadata")
    rows = re.findall(r"^\s*(\d+)\s+([0-9.e+-]+)\s+([0-9.]+)\s+(.+)$",
                      report, re.MULTILINE)
    require(any("detail::unregisteredHotLoop(" in row[3] for row in rows),
            "unregistered CPU work was not attributed to its compiled function")
    require("regressionMetadataIsNotAttribution" not in report,
            "metadata replaced an interrupted function's actual symbol")
    require(sum(int(row[0]) for row in rows) == samples - dropped,
            "function rows do not conserve the delivered sample count")
    for count, seconds, percentage, function in rows:
        require(math.isclose(float(seconds), int(count) * 0.01, rel_tol=1e-6),
                f"wrong default 10 ms CPU-time estimate for {function}")
        require(math.isclose(float(percentage), 100 * int(count) / samples,
                             abs_tol=0.011), f"wrong sample percentage for {function}")


def wait_until(process, condition, message, timeout=45):
    deadline = time.monotonic() + timeout
    while not condition():
        require(process.poll() is None,
                f"{message}: fixture exited early ({process.returncode})")
        require(time.monotonic() < deadline, f"{message}: timed out")
        time.sleep(0.01)


def check_interrupt(executable, directory, block_output=False):
    directory.mkdir()
    output_path = directory / "stdout.log"
    error_path = directory / "stderr.log"
    mode = "--interrupt-block-output" if block_output else "--interrupt-exit"
    command = [str(executable), mode, "--hpx:threads=2", "--hpx:bind=none"]
    notice = ("Ctrl-C: clean shutdown requested; finishing active work. "
              "Press Ctrl-C again to terminate immediately.")
    with output_path.open("w") as output, error_path.open("w") as error:
        process = subprocess.Popen(command, cwd=directory, stdout=output, stderr=error,
                                   start_new_session=True)
        try:
            ready = directory / "interrupt_ready.txt"
            wait_until(process, lambda: ready.exists() and ready.read_text() == "ready\n",
                       "interrupt fixture did not finish sampled HPX work")
            process.send_signal(signal.SIGINT)
            if block_output:
                wait_until(process, lambda: notice in error_path.read_text(),
                           "first interrupt did not request cooperative shutdown", timeout=20)
                require(process.poll() is None,
                        "first interrupt forcibly terminated uncooperative work")
                require("CosmicTiger statistical CPU profile (" not in output_path.read_text(),
                        "terminal report escaped the fixture's stdout lock")
                process.send_signal(signal.SIGINT)
            try:
                returncode = process.wait(timeout=5 if block_output else 20)
            except subprocess.TimeoutExpired as exception:
                raise RuntimeError("interrupt fixture did not terminate promptly") from exception
        except Exception as exception:
            raise RuntimeError(f"{mode}: {exception}\nstdout:\n{output_path.read_text()}"
                               f"\nstderr:\n{error_path.read_text()}") from exception
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
    output = output_path.read_text()
    error = error_path.read_text()
    expected_returncode = -signal.SIGINT if block_output else 0
    require(returncode == expected_returncode,
            f"interrupt fixture returned {returncode}, expected {expected_returncode}:\n{error}")
    require(error.count(notice) == 1, "interrupt did not request exactly one clean shutdown")
    require("SIGABRT" not in error and "terminate called" not in error and
            "Aborted" not in error, f"HPX aborted during interrupt reporting:\n{error}")
    if not block_output:
        require(output.count("INTERRUPT_HPX_FINALIZED=1") == 1,
                "first interrupt did not return cleanly from HPX")
        reports = list(directory.glob("cosmictiger.profile.*.txt"))
        require(len(reports) == 1, "interrupt did not produce one per-process report file")
        report = reports[0].read_text()
        require(output.count("CosmicTiger statistical CPU profile (") == 1 and
                output.endswith(report),
                "interrupt did not emit exactly one matching terminal/file report")
        check_report(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    args = parser.parse_args()
    executable = args.executable.resolve()
    with tempfile.TemporaryDirectory(prefix="cosmictiger_profiler_") as temporary:
        directory = Path(temporary)
        output = run(executable, directory, ("--hpx:bind=none",))
        report = (directory / "expected.txt").read_text()
        require((directory / "explicit.txt").read_text() == report,
                "explicit report differs from the stopped snapshot")
        automatic = list(directory.glob("cosmictiger.profile.*.txt"))
        require(len(automatic) == 1, "exit did not produce one per-process report file")
        require(automatic[0].read_text() == report,
                "automatic exit file differs from the stopped snapshot")
        require(output.count(report) == 2 and output.endswith(report),
                "explicit and automatic terminal reports differ from their files")
        check_report(report)
        active_exit = directory / "automatic_exit"
        active_exit.mkdir()
        exit_output = run(executable, active_exit, ("--automatic-exit", "--hpx:bind=none"))
        exit_files = list(active_exit.glob("cosmictiger.profile.*.txt"))
        require(len(exit_files) == 1, "active exit did not produce one per-process report file")
        exit_report = exit_files[0].read_text()
        require(exit_output.count("CosmicTiger statistical CPU profile (") == 1 and
                exit_output.endswith(exit_report),
                "active exit did not emit exactly one matching terminal/file report")
        check_report(exit_report)
        invalid = directory / "invalid"
        invalid.mkdir()
        invalid_output = run(executable, invalid, ("--invalid-interval",))
        require("invalid CPU sampling interval" in invalid_output,
                "invalid interval did not emit a diagnostic")
        require(summary((invalid / "invalid_interval.txt").read_text()) == (0, 0, 0, 1),
                "invalid interval did not report exactly one setup failure")
        check_interrupt(executable, directory / "interrupt_exit")
        check_interrupt(executable, directory / "interrupt_block_output", block_output=True)
    print("Profiler regression passed: HPX waits, PC attribution, diagnostics, final reports, "
          "and Ctrl-C clean shutdown/emergency escape")


if __name__ == "__main__":
    main()
