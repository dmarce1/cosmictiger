"""Exercise CPU particle storage/gravity while CUDA devices and runtime calls are unavailable."""
import argparse
import os
from pathlib import Path
import subprocess

from application_selection import application_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('guard', type=Path)
    parser.add_argument('workdir', type=Path)
    parser.add_argument('--without-chemistry', action='store_true')
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES'] = ''
    environment['LD_PRELOAD'] = str(args.guard.resolve())
    command = [str(args.executable.resolve()), '--compute.cuda=false', '--hydro.enabled=true',
               '--initial.particlesPerDimension=8', '--analysis.views=false',
               '--analysis.power=false', '--analysis.groups=false', '--lightcone.enabled=false',
               '--hpx:threads=2', '--hpx:bind=none', '--output.directory=.']
    command += application_arguments(args.executable.resolve(), far="fmm", hydro=True)
    cases = [('checkpointState', 'Checkpoint state and mode validation passed', []),
             ('sphGravity', 'SPH gravity regression passed', [])]
    if not args.without_chemistry:
        cases.append(('sphGravity', 'SPH gravity regression passed',
                      ['--chemistry.enabled=true', '--chemistry.initialization=manual']))
    for case, (test, marker, options) in enumerate(cases):
        result = subprocess.run(command + [f'--run.test={test}'] + options, cwd=workdir,
                                env=environment, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=120)
        log = workdir / f'{case}_{test}.log'
        log.write_text(result.stdout)
        if result.returncode != 0 or marker not in result.stdout:
            raise RuntimeError(f'{test}: CPU-only regression failed ({result.returncode}); see {log}')
    print('CPU execution passed: SPH checkpoint/sorting and mixed-particle gravity, no CUDA calls')


if __name__ == '__main__':
    main()
