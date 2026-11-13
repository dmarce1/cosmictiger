"""Exercise static FMM benchmark input, tiling, pairing and timing/count reporting."""
import argparse
import math
import os
from pathlib import Path
import random
import shlex
import subprocess

from application_selection import cpu_backend_options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('workdir', type=Path)
    parser.add_argument('--mpi', type=Path)
    args = parser.parse_args()
    executable = args.executable.resolve()
    root = args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       CUDA_VISIBLE_DEVICES='')
    generator = random.Random(357)
    files = []
    for index, count in enumerate((64, 96)):
        path = root / f'particles_{index}.txt'
        path.write_text('# x, y, z; redshift = 2.1880095155e+00\n' + ''.join(
            ' '.join(f'{generator.random():16.10e}' for _ in range(3)) + '\n'
            for _ in range(count)))
        files.append(path)
    common = {**cpu_backend_options(), 'run.scenario': 'benchmark',
              'benchmark.testFiles': ','.join(map(str, files)),
              'benchmark.openingAngles': '0.4, 0.7', 'benchmark.size': 2,
              'benchmark.repeats': 2, 'compute.bucketSize': 8,
              'output.silo': 'false', 'hpx:threads': 1 if args.mpi else 2,
              'hpx:bind': 'none'}
    prefix = []
    if args.mpi:
        prefix = [str(args.mpi), '-n', '2']
        common['hpx:localities'] = 2
        common['hpx:ini'] = 'hpx.parcel.mpi.enable=1'

    def run(name, changes=None, expected=None):
        directory = root / name
        directory.mkdir(exist_ok=True)
        options = {**common, 'output.directory': str(directory), **(changes or {})}
        result = subprocess.run(prefix + [str(executable)] +
                                [f'--{key}={value}' for key, value in options.items()],
                                cwd=directory, env=environment, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        (directory / 'run.log').write_text(result.stdout)
        if expected is None:
            assert result.returncode == 0, result.stdout[-6000:]
        else:
            assert result.returncode != 0 and expected in result.stdout, result.stdout[-6000:]
        return result.stdout

    output = run('paired')
    def records(prefix):
        return [dict(item.split('=', 1) for item in shlex.split(line[len(prefix):]))
                for line in output.splitlines() if line.startswith(prefix)]
    cases = [dict(item.split('=', 1) for item in shlex.split(line[len('Benchmark '):]))
             for line in output.splitlines() if line.startswith('Benchmark case=')]
    repeats = records('Benchmark repeat ')
    summaries = records('Benchmark summary ')
    assert len(cases) == 2 and len(repeats) == 4 and len(summaries) == 2
    for index, case in enumerate(cases):
        assert int(case['particles']) == (64, 96)[index] * 8
        assert int(case['base_particles']) == (64, 96)[index]
        assert math.isclose(float(case['theta']), (0.4, 0.7)[index], rel_tol=1e-7)
        assert int(case['localities']) == (2 if args.mpi else 1)
        assert case['backend'] == 'cpu'
    for row in repeats:
        assert int(row['flops']) == sum(int(row[key]) for key in ('flops32', 'flops64', 'flops_other'))
        assert int(row['flops32']) > 0 and int(row['flops64']) > 0
        for key in ('tree_s', 'force_s', 'cleanup_s', 'total_s', 'gflops'):
            assert math.isfinite(float(row[key])) and float(row[key]) > 0
        assert math.isclose(float(row['gflops']), int(row['flops']) / float(row['total_s']) * 1e-9,
                            rel_tol=1e-8)
    for row in summaries:
        assert int(row['flops']) == sum(int(r['flops']) for r in repeats if r['case'] == row['case'])
        assert float(row['min_s']) <= float(row['median_s']) <= float(row['max_s'])
    if not args.mpi:
        for name, changes, expected in (
            ('zero_size', {'benchmark.size': 0}, 'must be at least 1'),
            ('zero_repeats', {'benchmark.repeats': 0}, 'must be at least 1'),
            ('pairing', {'benchmark.openingAngles': '0.4'}, 'exactly one angle'),
            ('empty_entry', {'benchmark.testFiles': str(files[0]) + ','}, 'empty list entry'),
            ('nan_angle', {'benchmark.openingAngles': 'nan,0.7'}, 'must be finite'),
            ('bad_angle', {'benchmark.openingAngles': '0.4x,0.7'}, 'must be finite'),
            ('zero_angle', {'benchmark.openingAngles': '0,0.7'}, 'must be finite'),
            ('missing', {'benchmark.testFiles': str(root/'missing_particles.txt') + ',' + str(files[1])}, 'Cannot open'),
            ('overflow', {'benchmark.size': 2147483647}, 'too many particles'),
        ):
            run(name, changes, expected)
        for name, content in (('extra_column', '0.1 0.2 0.3 0.4\n'),
                              ('outside_box', '1.0 0.2 0.3\n'),
                              ('nonfinite', 'nan 0.2 0.3\n'), ('empty', '# x, y, z\n')):
            bad = root / f'{name}.txt'
            bad.write_text(content)
            run(name, {'benchmark.testFiles': str(bad) + ',' + str(files[1])},
                'Empty or unreadable' if name == 'empty' else 'expected exactly three')
    print('Benchmark tiling counts, paired angles, repeated solves, FLOPs and validation passed.')


if __name__ == '__main__':
    main()
