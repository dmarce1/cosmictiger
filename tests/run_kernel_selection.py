# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026 Dominic C. Marcello
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License, version 2 or any later version.
# This program is distributed without any warranty. See COPYING for details.

"""Exercise every production gravity kernel with an independent hydro selection."""

import argparse
from pathlib import Path
import re
import subprocess


KERNEL_SPECS = (
    *(('spline', degree) for degree in range(3, 10)),
    ('wendland', 2), ('wendland', 5), ('wendland', 8),
    ('ferrers', 2), ('ferrers', 4), ('ferrers', 6), ('ferrers', 8),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('workdir', type=Path)
    parser.add_argument('--cuda', action='store_true',
                        help='also execute the CUDA gravity and halo-potential paths')
    args = parser.parse_args()
    executable = args.executable.resolve()
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    backend = 'cuda' if args.cuda else 'cpu'
    command = [str(executable), '--run.test=sphGravity',
               '--initial.particlesPerDimension=8', '--hydro.enabled=true',
               f'--compute.cuda={str(args.cuda).lower()}',
               '--hydro.neighborNumber=200', '--hpx:threads=2', '--hpx:bind=none']
    failures = []
    for gravity, hydro in zip(KERNEL_SPECS, reversed(KERNEL_SPECS)):
        gravity_family, gravity_degree = gravity
        hydro_family, hydro_degree = hydro
        name = (f'{backend}_gravity_{gravity_family}_{gravity_degree}'
                f'_hydro_{hydro_family}_{hydro_degree}')
        case_directory = workdir / name
        case_directory.mkdir(parents=True, exist_ok=True)
        log = workdir / f'{name}.log'
        options = [f'--gravity.kernelFamily={gravity_family}',
                   f'--gravity.kernelDegree={gravity_degree}',
                   f'--hydro.kernelFamily={hydro_family}',
                   f'--hydro.kernelDegree={hydro_degree}']
        # Keep device visibility: even CPU-force regressions can allocate CUDA
        # managed particle storage when the executable was built with CUDA.
        try:
            result = subprocess.run(command + options, cwd=case_directory, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    timeout=60)
        except subprocess.TimeoutExpired as error:
            output = error.stdout or ''
            if isinstance(output, bytes):
                output = output.decode(errors='replace')
            log.write_text(output + '\nKernel-selection regression timed out after 60 seconds.\n')
            failures.append(f'{name}: timed out; see {log}')
            print(f'FAIL {name}: timed out', flush=True)
            continue
        log.write_text(result.stdout)
        problem = None
        if result.returncode != 0:
            problem = f'exited with status {result.returncode}'
        elif 'SPH gravity regression passed' not in result.stdout:
            problem = 'missing gravity regression completion marker'
        else:
            for key, expected in (
                    ('gravity.kernelFamily', gravity_family),
                    ('gravity.kernelDegree', gravity_degree),
                    ('hydro.kernelFamily', hydro_family),
                    ('hydro.kernelDegree', hydro_degree)):
                if not re.search(r'^' + re.escape(key) + r'\s*:\s*' +
                                 re.escape(str(expected)) + r'\s*$',
                                 result.stdout, re.MULTILINE):
                    problem = f'{key} did not retain its requested value'
                    break
        if problem:
            failures.append(f'{name}: {problem}; see {log}')
            print(f'FAIL {name}: {problem}', flush=True)
        else:
            print(f'PASS {name}', flush=True)
    if failures:
        raise RuntimeError('\n'.join(failures))
    print(f'All {len(KERNEL_SPECS)} {backend} kernel-selection regressions passed.')


if __name__ == '__main__':
    main()
