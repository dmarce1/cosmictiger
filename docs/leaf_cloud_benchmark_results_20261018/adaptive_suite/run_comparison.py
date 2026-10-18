import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

repo = Path('/home/dmarce1/workspace/cosmictiger')
work = Path(__file__).resolve().parent
output = repo / 'docs/leaf_cloud_benchmark_results_20261018/adaptive_suite'
output.mkdir(exist_ok=True)
manifest = json.loads((work / 'build_manifest.json').read_text())
assert manifest['completed']
shutil.copy2(work / 'build_manifest.json', output / 'build_manifest.json')
for source in ['build_adaptive.py', 'stf_walk.cpp', 'harness_changes.patch']:
    shutil.copy2(work / source, output / source)
samples = 15
pairs = 5
assert {1, 2}.issubset(os.sched_getaffinity(0))
assert not (output / 'runs.json').exists()
for info in manifest['binaries'].values():
    assert hashlib.sha256(Path(info['path']).read_bytes()).hexdigest() == info['sha256']
metadata = {
    'samples_per_case_per_process': samples, 'warmup_per_case_per_process': 1,
    'paired_processes_per_backend': pairs, 'cpu_affinity': [1, 2], 'hpx_workers': 2,
    'lscpu': subprocess.check_output(['lscpu', '--json'], text=True),
    'compiler': subprocess.check_output(['/usr/bin/c++', '--version'], text=True),
    'cuda': subprocess.check_output(['/usr/local/cuda/bin/nvcc', '--version'], text=True),
    'gpu_before': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,temperature.gpu,clocks.sm,clocks.mem,utilization.gpu', '--format=csv'], text=True),
    'scope': 'Complete periodic tree creation and gravity kick; fixed 32768-particle jittered lattice and eight clusters plus an adaptive dense cube with diffuse particles; all particles active; force-only and potential-enabled. Memory daemon stopped; HPX task quiescence included in kick time. Reset/export excluded from reported tree/kick times.',
}
(output / 'runtime_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
shutil.copy2(__file__, output / 'run_comparison.py')
runs = []
fingerprints = {}
common = ['--run.test=stfBenchmark', '--initial.particlesPerDimension=32', '--hydro.enabled=false',
          '--gravity.openingAngle=0.8', '--gravity.effectiveRadius=0.01876988456953534',
          '--gravity.kernelFamily=ferrers', '--gravity.kernelDegree=4',
          '--hpx:threads=2', '--hpx:bind=none', '--analysis.views=false',
          '--analysis.power=false', '--analysis.groups=false', '--lightcone.enabled=false']
for backend in ['cpu', 'cuda']:
    for pair in range(pairs):
        order = ['before', 'after'] if pair % 2 == 0 else ['after', 'before']
        for variant in order:
            name = f'{backend}_pair{pair}_{variant}'
            directory = work / 'runs' / name
            directory.mkdir(parents=True)
            command = ['taskset', '-c', '1,2', manifest['binaries'][variant]['path'],
                       *common, '--compute.cuda=' + ('true' if backend == 'cuda' else 'false')]
            env = dict(os.environ, COSMICTIGER_STF_WALK_SAMPLES=str(samples),
                       OMP_NUM_THREADS='2', OMP_THREAD_LIMIT='2')
            logfile = output / (name + '.log')
            start = time.monotonic()
            with logfile.open('w') as stream:
                status = subprocess.run(command, cwd=directory, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=120).returncode
            duration = time.monotonic() - start
            if status:
                raise RuntimeError(f'{name} exited {status}; see {logfile}')
            text = logfile.read_text()
            config = next(line for line in text.splitlines() if line.startswith('STF_WALK_CONFIG,'))
            assert f'cuda={int(backend == "cuda")},' in config
            assert ('Occupancy is ' in text) == (backend == 'cuda')
            medians = []
            measurements = []
            for line in text.splitlines():
                if line.startswith('STF_WALK_MEDIAN,'):
                    fields = line.split(',')
                    row = dict(fixture=fields[1], backend=fields[2], potential=bool(int(fields[3])),
                               samples=int(fields[4]), particles=int(fields[5]),
                               reset_seconds=float(fields[6]), tree_seconds=float(fields[7]),
                               kick_seconds=float(fields[8]), total_seconds=float(fields[9]),
                               position_hash=fields[10])
                    assert row['backend'] == backend and row['samples'] == samples and row['particles'] == 32768
                    assert fingerprints.setdefault(row['fixture'], row['position_hash']) == row['position_hash']
                    medians.append(row)
                elif line.startswith('STF_WALK_SAMPLE,'):
                    fields = line.split(',')
                    if int(fields[4]) >= 0:
                        measurements.append(dict(fixture=fields[1], backend=fields[2],
                                                 potential=bool(int(fields[3])), sample=int(fields[4]),
                                                 reset_seconds=float(fields[5]), tree_seconds=float(fields[6]),
                                                 kick_seconds=float(fields[7])))
            assert len(medians) == 6 and len(measurements) == 6 * samples
            forces = directory / 'stf_walk_forces.csv'
            runs.append(dict(variant=variant, backend=backend, pair=pair, command=command,
                             cwd=str(directory), environment={'COSMICTIGER_STF_WALK_SAMPLES': str(samples),
                             'OMP_NUM_THREADS': '2', 'OMP_THREAD_LIMIT': '2'}, exit_code=status,
                             wall_seconds=duration, config=config, medians=medians, samples=measurements,
                             force_csv=str(forces), force_csv_sha256=hashlib.sha256(forces.read_bytes()).hexdigest()))
            (output / 'runs.json').write_text(json.dumps(runs, indent=2) + '\n')
            print(name + ': ' + ', '.join(f'{row["fixture"]}/phi={int(row["potential"])} {1e3*row["kick_seconds"]:.3f} ms' for row in medians), flush=True)
metadata['gpu_after'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,temperature.gpu,clocks.sm,clocks.mem,utilization.gpu', '--format=csv'], text=True)
(output / 'runtime_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
print('All paired measurements completed.', flush=True)
