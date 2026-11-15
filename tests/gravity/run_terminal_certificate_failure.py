#!/usr/bin/env python3
"""Require a specific production CUDA terminal failure in an isolated process."""

import argparse
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable")
    parser.add_argument("term", choices=("newtonian", "periodic"))
    args = parser.parse_args()
    command = [args.executable, f"--reject-terminal-{args.term}"]
    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", timeout=45, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"Failed to complete CUDA certificate fixture: {error}", file=sys.stderr)
        return 1
    print(result.stdout, end="")
    if result.returncode == 77:
        return 77
    term = "Newtonian" if args.term == "newtonian" else "periodic"
    diagnostic = f"CUDA terminal failed {term} certificate"
    device_assertion = re.search(
        rf'Condition "[^\n]*{re.escape(diagnostic)}[^\n]*" failed in '
        r'[^\n]+ on line \d+ in CUDA_ARCH', result.stdout,
    )
    # A poisoned CUDA context must fail synchronization, and the exact device
    # assertion must identify the intended contract violation. Generic launch,
    # memory-access, or setup failures are not accepted as negative coverage.
    if (result.returncode == 0 or device_assertion is None or
            "complete bounded production kick:" not in result.stdout):
        print(f"Missing checked device failure: {diagnostic}", file=sys.stderr)
        return 1
    print(f"Verified checked device failure: {diagnostic}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
