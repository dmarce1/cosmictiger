"""Check sampler attribution, diagnostics, and automatic final report delivery."""

import argparse
import math
from pathlib import Path
import re
import subprocess
import tempfile


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
    print("Profiler regression passed: HPX waits, PC attribution, diagnostics, and final reports")


if __name__ == "__main__":
    main()
