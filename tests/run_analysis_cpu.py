# CosmicTiger - CPU analysis integration regression; BSL-1.0.
"""Run deterministic production power/lightcone pipelines with CUDA unavailable."""

import argparse
import os
from pathlib import Path
import signal
import subprocess

from application_selection import application_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--guard", type=Path)
    parser.add_argument("--ranks", type=int, choices=(1, 2), default=1)
    parser.add_argument("--mpi-exec", default="mpiexec")
    parser.add_argument("--mpi-numproc-flag", default="-n")
    parser.add_argument("--mpi-preflag", action="append", default=[])
    parser.add_argument("--mpi-postflag", action="append", default=[])
    parser.add_argument("--hpx-option", action="append", default=[])
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--hydro", action="store_true",
                        help="run with SPH enabled: gas joins maps and spectra, not the finders")
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    output = workdir / "lightcone"
    output.mkdir(exist_ok=True)
    for epoch in range(4):
        for rank in range(args.ranks):
            for product in ("lc.groups", "lc.healpix"):
                (output / f"{product}.{epoch}.{rank}").unlink(missing_ok=True)
            (output / f"lc.groups.{epoch}.{rank}.format").unlink(missing_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["OMP_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    if args.guard is not None:
        if not args.guard.is_file():
            raise RuntimeError(f"CUDA runtime guard was not built: {args.guard}")
        environment["LD_PRELOAD"] = str(args.guard.resolve())
    command = [str(args.executable.resolve()), "--run.test=analysisBackends",
               "--gravity.cuda=false", f"--hydro.enabled={str(args.hydro).lower()}",
               "--initial.particlesPerDimension=8", "--initial.closePack=false",
               "--analysis.views=false", "--analysis.groups=false", "--analysis.power=true",
               "--lightcone.enabled=true", f"--lightcone.directory={output}",
               "--lightcone.mapSize=1", "--lightcone.linkingLength=0.28",
               f"--hpx:threads={args.threads}", "--hpx:bind=none"]
    command += application_arguments(args.executable.resolve(), hydro=args.hydro)
    if "--gravity.method.far=pmmm" in command:
        command.append("--gravity.meshSize=8")
    if args.ranks > 1:
        command = [args.mpi_exec, args.mpi_numproc_flag, str(args.ranks), *args.mpi_preflag,
                   command[0], *args.mpi_postflag, *command[1:]]
        command += [f"--hpx:localities={args.ranks}",
                    *(args.hpx_option or ["--hpx:ini=hpx.parcel.mpi.enable=1"])]
    log_path = workdir / "analysis_backends.log"
    with log_path.open("w") as log:
        with subprocess.Popen(command, cwd=workdir, env=environment, text=True,
                              stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True) as process:
            try:
                returncode = process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError(f"CPU analysis timed out; see {log_path}")
    marker = "CPU analysis backend integration regression passed"
    if returncode != 0 or marker not in log_path.read_text():
        raise RuntimeError(f"CPU analysis failed ({returncode}); see {log_path}")
    species = "SPH species" if args.hydro else "dark matter"
    print(f"CPU power/lightcone pipelines passed with {args.ranks} locality/localities ({species})")


if __name__ == "__main__":
    main()
