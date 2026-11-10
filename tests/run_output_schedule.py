"""Exercise inclusive physical/conformal output phases and an interior restart."""
import argparse
import math
import os
from pathlib import Path
import re
import shutil
import subprocess

from application_selection import cpu_backend_options
from checkpoint_format import decode_checkpoint


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('workdir', type=Path)
    args = parser.parse_args()
    executable = args.executable.resolve()
    root = args.workdir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       CUDA_VISIBLE_DEVICES='', MPLBACKEND='Agg')
    common = {**cpu_backend_options(), 'hpx:threads': 2, 'hpx:bind': 'none',
              'run.darkMatterParticleCount': 64, 'run.gasParticleCount': 512,
              'gravity.meshSize': 8, 'gravity.enabled': 'true', 'hydro.enabled': 'false',
              'chemistry.enabled': 'false', 'starFormation.enabled': 'false',
              'run.maxIterations': 2000, 'run.minRung': 1,
              'analysis.groups': 'false', 'analysis.power': 'false',
              'analysis.views': 'false', 'analysis.tracers': 'false',
              'analysis.slice': 'false', 'lightcone.enabled': 'false',
              'output.silo': 'true', 'output.units': 'code',
              'checkpoint.intervalSeconds': 0, 'scenario.cosmology.twoLpt': 'false',
              'scenario.cosmology.useGlass': 'false', 'cosmology.sigma8': 0}

    def run(name, values, *, reset=True, succeeds=True):
        directory = root / name
        if reset and directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
        options = dict(common, **{'output.directory': str(directory)}, **values)
        command = [str(executable), *[f'--{key}={value}' for key, value in options.items()]]
        result = subprocess.run(command, cwd=directory, env=environment, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, timeout=180)
        (directory / ('restart.log' if not reset else 'run.log')).write_text(result.stdout)
        require((result.returncode == 0) == succeeds,
                f'{name}: exit={result.returncode}\n{result.stdout[-6000:]}')
        return directory, result.stdout

    def check(directory, count, duration, *, cosmology=False, fixed=False):
        cycles = {int(path.name.split('.')[2]) for path in directory.glob('view.0.*.silo')}
        require(cycles == set(range(count)), f'{directory}: output cycles {sorted(cycles)}')
        if fixed:
            times = [float(line.split()[0]) for line in (directory / 'energy.dat').read_text().splitlines()
                     if line.strip() and not line.startswith('#')]
            expected = [duration * i / (count - 1) for i in range(count)]
            require(len(times) == count, f'{directory}: {len(times)} diagnostic phases, expected {count}')
            require(all(math.isclose(a, b, rel_tol=2e-6, abs_tol=1e-14) for a, b in zip(times, expected)),
                    f'{directory}: nonuniform physical output times {times}')
        else:
            rows = [line.split() for line in (directory / 'time.txt').read_text().splitlines()[1:]
                    if line.strip()]
            require(len(rows) == count, f'{directory}: time.txt did not include exactly {count} endpoints')
            column = 2 if cosmology else 1
            times = [float(row[column]) for row in rows]
            spacing = (times[-1] - times[0]) / (count - 1)
            require(spacing > 0 and all(math.isclose(b - a, spacing, rel_tol=2e-8,
                    abs_tol=4e-14 * max(abs(value) for value in times))
                    for a, b in zip(times, times[1:])), f'{directory}: nonuniform time table')
        files = sorted(directory.glob('checkpoint.*/checkpoint.0.dat'),
                       key=lambda path: int(path.parent.name.split('.')[1]))
        require(files, f'{directory}: missing checkpoint')
        state = decode_checkpoint(files[-1].read_bytes())
        require(state.header.steps == count - 1, 'checkpoint did not retain its interval-count encoding')
        require(math.isclose(state.driver.tau, duration, rel_tol=1e-12), 'checkpoint missed the endpoint')
        print(f'{directory.name}: {count} physical/conformal phases, both endpoints and checkpoint count passed')

    for count in (2, 5):
        values = {'run.scenario': 'plummer', 'run.numberOfOutputs': count, 'run.endTime': 0.0001}
        directory, _ = run(f'physical_{count}', values)
        check(directory, count, 0.0001)
    values = {'run.scenario': 'hydroWave', 'run.numberOfOutputs': 5, 'run.endTime': 0.001,
              'gravity.enabled': 'false', 'hydro.enabled': 'true', 'run.darkMatterParticleCount': 0}
    directory, _ = run('fixed_physical', values)
    check(directory, 5, 0.001, fixed=True)
    values = {'run.scenario': 'polytropicStar', 'run.numberOfOutputs': 2,
              'run.endTime': 0.00001, 'hydro.enabled': 'true'}
    directory, _ = run('two_stage_physical', values)
    check(directory, 2, 0.00001, fixed=True)
    values = {'run.scenario': 'cosmology', 'run.numberOfOutputs': 4, 'run.endTime': 0.0002,
              'units.boxLength': 20}
    directory, _ = run('conformal', values)
    check(directory, 4, 0.0002, cosmology=True)
    partial, _ = run('conformal_restart', dict(values, **{'run.maxIterations': 1}))
    state = decode_checkpoint((partial / 'checkpoint.1/checkpoint.0.dat').read_bytes())
    require(state.driver.integer_time != 0, 'restart fixture did not stop inside an output interval')
    directory, _ = run('conformal_restart', dict(values, **{'checkpoint.readIndex': 1}), reset=False)
    check(directory, 4, 0.0002, cosmology=True)

    for option, value in [('run.numberOfSteps', '4'), ('output.siloInterval', '1'),
                          ('run.timeIntegrator', 'hierarchical')]:
        _, output = run('retired_' + option, {option: value}, succeeds=False)
        require('unrecognized option' in output and option in output, f'{option} still accepted')
    for count in (0, 1):
        _, output = run('invalid_count_' + str(count), {'run.numberOfOutputs': count}, succeeds=False)
        require('run.numberOfOutputs must be at least 2' in output, 'endpoint count validation missing')
    print('Inclusive output schedule and restart checks passed.')


if __name__ == '__main__':
    main()
