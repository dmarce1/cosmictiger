#!/usr/bin/env python3
"""Run the bounded step-7 CPU comparison; save exact inputs, command and provenance.

Default experiment: 96 deterministic points, 32768 points in two compact halos,
three paired repeats, two MPI ranks with two HPX threads each. Run only within
an explicitly authorized shared allocation. Does not use GPUs or remote hosts.
"""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import random
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--mpi', default='mpiexec')
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'command.json').exists():
        parser.error('use a new output directory to keep each experiment distinct')
    generator = random.Random(731924)
    small = root / 'small.txt'
    clustered = root / 'clustered.txt'
    small.write_text(''.join(' '.join(format(generator.random(), '.17g') for _ in range(3)) + '\n'
                             for _ in range(96)))
    with clustered.open('w') as output:
        for index in range(32 ** 3):
            center = .25 if index % 2 == 0 else .75
            output.write(' '.join(format((center + generator.gauss(0, .035)) % 1, '.17g')
                                  for _ in range(3)) + '\n')
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / 'tests'))
    from application_selection import cpu_backend_options
    executable = args.executable.resolve()
    options = {**cpu_backend_options(), 'run.scenario': 'benchmark',
               'benchmark.testFiles': f'{small},{clustered}',
               'benchmark.openingAngles': '.4,.5', 'benchmark.size': 1,
               'benchmark.repeats': 3, 'benchmark.compareSourceForest': 'true',
               'compute.bucketSize': 128, 'gravity.smoothingLength': .02,
               'output.silo': 'false', 'output.directory': str(root),
               'hpx:threads': 2, 'hpx:bind': 'none', 'hpx:localities': 2,
               'hpx:ini': 'hpx.parcel.mpi.enable=1'}
    command = [args.mpi, '-n', '2', str(executable)] + [f'--{k}={v}' for k, v in options.items()]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    def git(*arguments):
        return subprocess.check_output(['git', '-C', str(repo), *arguments])
    files = git('ls-files', '--cached', '--others', '--exclude-standard',
                'cosmictiger', 'src', 'cmake', 'CMakeLists.txt').decode().splitlines()
    manifest = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
                for name in files if (repo / name).is_file()}
    provenance = {'platform': platform.platform(),
                  'cpu_model': next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), 'unknown'),
                  'command': command, 'options': options, 'git_head': git('rev-parse', 'HEAD').decode().strip(),
                  'source_sha256': manifest,
                  'executable_sha256': hashlib.sha256(executable.read_bytes()).hexdigest(),
                  'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (small, clustered)},
                  'environment': {key: environment[key] for key in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS',
                                                                  'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}}
    (root / 'command.json').write_text(json.dumps(provenance, indent=2) + '\n')
    cache = executable.parent / 'CMakeCache.txt'
    if cache.exists():
        (root / 'CMakeCache.txt').write_bytes(cache.read_bytes())
    (root / 'tracked.patch').write_bytes(git('diff', '--binary'))
    with (root / 'run.log').open('w') as log:
        result = subprocess.run(command, cwd=root, env=environment, stdout=log,
                                stderr=subprocess.STDOUT, timeout=900)
    if result.returncode:
        raise SystemExit(f'comparison failed ({result.returncode}); see {root / "run.log"}')
    from check_fmm_measurement import verify
    verify(root)
    print(f'Comparison and measurement invariants passed: {root}')


if __name__ == '__main__':
    main()
