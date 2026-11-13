"""Exercise static FMM benchmark input, tiling, pairing and timing/count reporting."""
import argparse
import math
import os
from pathlib import Path
import random
import re
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
    blocks = re.split(r'^Benchmark case (\d+) / (\d+)\n', output, flags=re.MULTILINE)
    assert len(blocks) == 7, output
    for offset in (1, 4):
        case, case_count, block = blocks[offset:offset + 3]
        index = int(case) - 1
        assert int(case_count) == 2
        population = re.search(r'Particles:\s+(\d+) \((\d+) base particles x (\d+)\^3 copies\)', block)
        assert population and tuple(map(int, population.groups())) == (
            (64, 96)[index] * 8, (64, 96)[index], 2)
        assert re.search(r'Solver:\s+cpu \| Localities: ' + str(2 if args.mpi else 1) +
                         r' \| Repeats: 2', block)
        assert math.isclose(float(re.search(r'Theta:\s+(\S+)', block)[1]),
                            (0.4, 0.7)[index], rel_tol=1e-7)
        timings = []
        average = None
        solve_flops = None
        tree_nodes = None
        tree_leaves = None
        particles_per_leaf = None
        summary = None
        section = None
        for line in block.splitlines():
            fields = line.split()
            if line.strip().startswith('Repeat'):
                section = 'timings'
            elif line.strip() == 'Timing summary':
                section = 'summary'
            elif line.strip() == 'Tree and work per solve':
                section = 'tree'
            elif fields and fields[0].isdigit():
                if section == 'timings':
                    timings.append(list(map(float, fields[1:])))
                elif section == 'tree' and len(fields) == 5:
                    assert int(fields[0]) == int(population[1])
                    tree_nodes, tree_leaves = map(int, fields[1:3])
                    particles_per_leaf = float(fields[3])
                    solve_flops = int(fields[4])
            elif fields and fields[0] == 'Mean':
                average = list(map(float, fields[1:]))
            elif section == 'summary' and len(fields) == 4 and fields[0][0].isdigit():
                summary = list(map(float, fields))
        assert len(timings) == 2 and solve_flops is not None and solve_flops > 0
        assert tree_nodes is not None and tree_nodes > 1 and tree_nodes % 2 == 1
        assert tree_leaves is not None and tree_nodes == 2 * tree_leaves - 1
        assert math.isclose(particles_per_leaf, int(population[1]) / tree_leaves, abs_tol=0.005)
        assert average is not None and len(average) == 5
        for timing in timings:
            assert len(timing) == 5
            assert all(math.isfinite(value) and value > 0 for value in timing)
            # Allow rounding to the displayed microseconds and two GFLOP/s decimals.
            assert math.isclose(timing[4], solve_flops / timing[3] * 1e-9,
                                rel_tol=1e-3, abs_tol=0.01)
        for column in range(4):
            assert math.isclose(average[column], sum(row[column] for row in timings) / 2,
                                rel_tol=1e-3, abs_tol=1e-6)
        assert summary is not None and summary[0] <= summary[1] <= summary[3]
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
