#!/usr/bin/env python3
"""Run the opt-in STF walk benchmark over binaries, kick policies and buckets.

Each process runs --run.test=stfBenchmark once for one binary, policy, backend,
bucket and FMM order (--gravity.fmmOrder), cycling through the selected fixtures
with potential off and on. --threads sets the HPX worker threads of every
process (pinned with taskset to --cpus, by default CPUs 1 through the thread
count) and --particles-per-dimension the fixture side (32: 32,768 particles).
Processes alternate between binaries and policies within each pair so slow
drifts affect every variant alike. Whole-kick times come from the harness; the
CUDA kick arena line and force exports are collected for every run, and the
harness compares a particle sample with the direct Ewald sum when
--reference is positive. Nothing here regenerates or rebuilds binaries.
Each binary receives a bounded, CPU-only --help probe so newer builds enable
--run.verbose=true for arena details while historical baselines keep their
original command line.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import statistics
import subprocess
import sys
import time


def parse_binary(text):
    label, _, rest = text.partition('=')
    path, _, flags = rest.partition(':')
    if not label or not path:
        raise argparse.ArgumentTypeError('expected LABEL=PATH[:arena]')
    return {'label': label, 'path': str(Path(path).resolve()), 'arena': flags == 'arena'}


def parse_policy(text):
    label, _, options = text.partition('=')
    if not label:
        raise argparse.ArgumentTypeError('expected LABEL=--option=value,...')
    return {'label': label, 'options': [item for item in options.split(',') if item]}


def supports_run_verbose(binary):
    if 'supports_run_verbose' not in binary:
        command = [binary['path'], '--help', '--compute.cuda=false',
                   '--hpx:threads=1', '--hpx:bind=none']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OMP_THREAD_LIMIT='1')
        try:
            result = subprocess.run(command, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f'Cannot probe options for {binary["label"]}: {error}') from error
        if result.returncode:
            raise RuntimeError(f'Option probe for {binary["label"]} exited {result.returncode}:\n'
                               + result.stdout)
        binary['supports_run_verbose'] = bool(
            re.search(r'^\s*--run\.verbose(?:\s|=|$)', result.stdout, re.MULTILINE))
    return binary['supports_run_verbose']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    rows = {}
    with Path(path).open(newline='') as handle:
        for row in csv.DictReader(handle):
            key = (row['fixture'], int(row['x_raw']), int(row['y_raw']), int(row['z_raw']))
            values = tuple(float(row[field]) for field in ('gx', 'gy', 'gz', 'phi'))
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(f'Nonfinite force export in {path}: {key}')
            rows[key] = values
    return rows


def describe(reference, candidate, vector):
    indices = range(3) if vector else (3,)
    reference_sum = 0.0
    difference_sum = 0.0
    relative = []
    for first, second in zip(reference, candidate):
        ref_squared = math.fsum(first[i] * first[i] for i in indices)
        delta_squared = math.fsum((second[i] - first[i]) ** 2 for i in indices)
        reference_sum += ref_squared
        difference_sum += delta_squared
        if ref_squared > 0:
            relative.append(math.sqrt(delta_squared / ref_squared))
    return {
        'relative_l2': math.sqrt(difference_sum / reference_sum) if reference_sum else None,
        'max_pointwise_relative': max(relative) if relative else None,
        'median_pointwise_relative': statistics.median(relative) if relative else None,
    }


def compare_forces(reference_path, candidate_path):
    reference = read_rows(reference_path)
    candidate = read_rows(candidate_path)
    if reference.keys() != candidate.keys():
        raise RuntimeError('Force exports do not cover the same particles')
    result = {}
    for fixture in sorted({key[0] for key in reference}):
        keys = sorted(key for key in reference if key[0] == fixture)
        first = [reference[key] for key in keys]
        second = [candidate[key] for key in keys]
        result[fixture] = {'acceleration': describe(first, second, True),
                           'potential': describe(first, second, False)}
    return result


def parse_log(text, samples, fixtures):
    run = {'medians': [], 'samples': [], 'references': [], 'leaf_histogram': [],
           'arena': None, 'occupancy': None}
    for line in text.splitlines():
        fields = line.split(',')
        if line.startswith('STF_WALK_MEDIAN,'):
            run['medians'].append(dict(fixture=fields[1], backend=fields[2],
                                       potential=bool(int(fields[3])), samples=int(fields[4]),
                                       particles=int(fields[5]), reset_seconds=float(fields[6]),
                                       tree_seconds=float(fields[7]), kick_seconds=float(fields[8]),
                                       total_seconds=float(fields[9]), position_hash=fields[10]))
        elif line.startswith('STF_WALK_SAMPLE,'):
            if int(fields[4]) >= 0:
                run['samples'].append(dict(fixture=fields[1], potential=bool(int(fields[3])),
                                           sample=int(fields[4]), tree_seconds=float(fields[6]),
                                           kick_seconds=float(fields[7]),
                                           total_potential=float(fields[14])))
        elif line.startswith('STF_WALK_REFERENCE,'):
            names = ['force_relative_l2', 'force_relative_p50', 'force_relative_p99',
                     'force_relative_max', 'potential_relative_l2', 'potential_relative_p50',
                     'potential_relative_p99', 'potential_relative_max']
            run['references'].append(dict(fixture=fields[1], backend=fields[2],
                                          samples=int(fields[3]),
                                          **{name: float(value) for name, value in zip(names, fields[4:])}))
        elif line.startswith('STF_WALK_LEAF_HISTOGRAM,'):
            run['leaf_histogram'].append(dict(fixture=fields[1], depth=int(fields[2]),
                                              particles=int(fields[3]), leaves=int(fields[4])))
        elif line.startswith('CUDA kick arena'):
            run['arena'] = line.strip()
        elif line.startswith('Occupancy is '):
            run['occupancy'] = line.strip()
    expected = 2 * len(fixtures)
    if len(run['medians']) != expected or len(run['samples']) != expected * samples:
        raise RuntimeError('Incomplete benchmark output')
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--binary', type=parse_binary, action='append', required=True,
                        help='LABEL=PATH[:arena]; :arena marks binaries accepting compute.cudaKick* options')
    parser.add_argument('--policy', type=parse_policy, action='append', default=[],
                        help='LABEL=--compute.option=value,... applied to arena binaries (default: auto)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixtures', default='lattice,clustered,adaptive')
    parser.add_argument('--buckets', default='64')
    parser.add_argument('--orders', default='7',
                        help='comma list of FMM orders passed as --gravity.fmmOrder (3 to 9)')
    parser.add_argument('--backends', default='cuda')
    parser.add_argument('--samples', type=int, default=15)
    parser.add_argument('--pairs', type=int, default=3)
    parser.add_argument('--reference', type=int, default=0,
                        help='particles compared with the direct Ewald sum (0 disables)')
    parser.add_argument('--theta', default='0.8')
    parser.add_argument('--cpus', default=None,
                        help='taskset CPU list for the HPX workers (default: 1-THREADS)')
    parser.add_argument('--threads', type=int, default=2, help='HPX worker threads per process')
    parser.add_argument('--particles-per-dimension', type=int, default=32,
                        help='fixture side passed as --initial.particlesPerDimension')
    parser.add_argument('--extra-option', action='append', default=[],
                        help='additional application option passed to every run')
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--idle-wait', default=None, metavar='PATTERN',
                        help='before each process, wait while a process whose name matches this '
                             'pgrep -x pattern (e.g. "ninja|ncu|nsys") is running')
    parser.add_argument('--wait-gpu-idle', action='store_true',
                        help='before each process, wait while nvidia-smi lists any compute process '
                             '(a cosmictiger run shows as main-thread#0 to pgrep, not by name)')
    args = parser.parse_args()
    if args.threads < 1:
        sys.exit('--threads must be positive')
    if args.cpus is None:
        args.cpus = f'1-{args.threads}'
    fixtures = args.fixtures.split(',')
    buckets = [int(value) for value in args.buckets.split(',')]
    orders = [int(value) for value in args.orders.split(',')]
    if any(order < 3 or order > 10 for order in orders):
        sys.exit('--orders must list FMM orders from 3 to 9')
    backends = args.backends.split(',')
    policies = args.policy or [{'label': 'auto', 'options': []}]
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'runs.json').exists():
        sys.exit('Output directory already holds runs.json; choose a fresh directory')
    capabilities = {}
    for binary in args.binary:
        if binary['path'] in capabilities:
            binary['supports_run_verbose'] = capabilities[binary['path']]
        else:
            capabilities[binary['path']] = supports_run_verbose(binary)
    metadata = {
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'platform': platform.platform(),
        'command': shlex.join(sys.argv),
        'binaries': [dict(binary, sha256=digest(binary['path'])) for binary in args.binary],
        'policies': policies, 'fixtures': fixtures, 'buckets': buckets, 'orders': orders,
        'backends': backends,
        'samples': args.samples, 'pairs': args.pairs, 'reference': args.reference,
        'threads': args.threads, 'cpus': args.cpus,
        'particles_per_dimension': args.particles_per_dimension,
        'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                   cwd=Path(__file__).resolve().parent).stdout.strip(),
        'nvidia_smi': subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,clocks.sm,'
                                      'clocks.mem,utilization.gpu,memory.used', '--format=csv'],
                                     capture_output=True, text=True).stdout,
        'load_average': os.getloadavg(),
    }
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    variants = []
    for binary in args.binary:
        for policy in (policies if binary['arena'] else [{'label': 'auto', 'options': []}]):
            variants.append((binary, policy))
    common = ['--run.test=stfBenchmark',
              f'--initial.particlesPerDimension={args.particles_per_dimension}',
              '--hydro.enabled=false', f'--gravity.openingAngle={args.theta}',
              '--gravity.effectiveRadius=0.01876988456953534', '--gravity.kernelFamily=ferrers',
              '--gravity.kernelDegree=4', f'--hpx:threads={args.threads}', '--hpx:bind=none',
              '--analysis.views=false', '--analysis.power=false', '--analysis.groups=false',
              '--lightcone.enabled=false'] + args.extra_option
    runs = []
    for pair in range(args.pairs):
        order = variants if pair % 2 == 0 else list(reversed(variants))
        for binary, policy in order:
            for backend in backends:
                for bucket in buckets:
                    for order in orders:
                        if not binary['arena'] and order != 7:
                            # A fixed-order baseline binary only measures order 7.
                            continue
                        name = (f'{binary["label"]}_{policy["label"]}_{backend}_b{bucket}'
                                f'_o{order}_pair{pair}')
                        directory = args.output / 'runs' / name
                        directory.mkdir(parents=True)
                        command = ['taskset', '-c', args.cpus, binary['path'], *common,
                                   '--compute.cuda=' + ('true' if backend == 'cuda' else 'false'),
                                   f'--compute.bucketSize={bucket}']
                        if binary['supports_run_verbose']:
                            command += ['--run.verbose=true']
                        if binary['arena']:
                            command += [f'--gravity.fmmOrder={order}'] + policy['options']
                        env = dict(os.environ, COSMICTIGER_STF_WALK_SAMPLES=str(args.samples),
                                   COSMICTIGER_STF_WALK_BUCKET=str(bucket),
                                   COSMICTIGER_STF_WALK_FIXTURE='all' if len(fixtures) == 3 else fixtures[0],
                                   COSMICTIGER_STF_WALK_REFERENCE=str(args.reference),
                                   OMP_NUM_THREADS=str(args.threads), OMP_THREAD_LIMIT=str(args.threads))
                        if len(fixtures) not in (1, 3):
                            sys.exit('The harness selects one fixture or all three')
                        logfile = directory / 'stdout.log'
                        while busy(args):
                            print(f'waiting for other work to finish before {name}', flush=True)
                            time.sleep(15)
                        begin = time.monotonic()
                        with logfile.open('w') as stream:
                            status = subprocess.run(command, cwd=directory, env=env, stdout=stream,
                                                    stderr=subprocess.STDOUT,
                                                    timeout=args.timeout).returncode
                        duration = time.monotonic() - begin
                        text = logfile.read_text()
                        if status:
                            raise RuntimeError(f'{name} exited {status}; see {logfile}')
                        run = parse_log(text, args.samples, fixtures)
                        forces = directory / 'stf_walk_forces.csv'
                        run.update(binary=binary['label'], policy=policy['label'], backend=backend,
                                   bucket=bucket, order=order, pair=pair, command=command,
                                   cwd=str(directory), wall_seconds=duration,
                                   force_csv=str(forces), force_csv_sha256=digest(forces))
                        runs.append(run)
                        (args.output / 'runs.json').write_text(json.dumps(runs, indent=2) + '\n')
                        print(name + ': ' + ', '.join(
                            f'{row["fixture"]}/phi={int(row["potential"])} {1e3 * row["kick_seconds"]:.3f} ms'
                            for row in run['medians']), flush=True)
    summary = summarize(runs, args)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.output / 'summary.md').write_text(render(summary))
    print('Completed; see ' + str(args.output / 'summary.md'), flush=True)


def busy(args):
    if args.idle_wait and subprocess.run(['pgrep', '-x', args.idle_wait],
                                         capture_output=True).returncode == 0:
        return True
    if args.wait_gpu_idle:
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                              capture_output=True, text=True).stdout.strip()
        if apps:
            return True
    return False


def summarize(runs, args):
    timing = {}
    for run in runs:
        for row in run['medians']:
            key = (run['binary'], run['policy'], run['backend'], run['bucket'], run['order'],
                   row['fixture'], row['potential'])
            timing.setdefault(key, []).append(row['kick_seconds'])
    table = []
    for key, values in sorted(timing.items()):
        table.append(dict(binary=key[0], policy=key[1], backend=key[2], bucket=key[3],
                          order=key[4], fixture=key[5], potential=key[6],
                          kick_ms_median=1e3 * statistics.median(values),
                          kick_ms_min=1e3 * min(values), kick_ms_max=1e3 * max(values),
                          pairs=len(values)))
    references = []
    for run in runs:
        for reference in run['references']:
            record = dict(reference)
            record.update(binary=run['binary'], policy=run['policy'], backend=run['backend'],
                          bucket=run['bucket'], order=run['order'], pair=run['pair'])
            references.append(record)
    arenas = sorted({(run['binary'], run['policy'], run['backend'], run['bucket'], run['order'],
                      run['arena'] or run['occupancy'] or '') for run in runs})
    # Compare every CUDA export with the matching CPU export of the same binary,
    # policy, bucket and order, and every variant with the first binary's CUDA
    # export at the same bucket and order.
    comparisons = []
    first_pair = {(run['binary'], run['policy'], run['backend'], run['bucket'], run['order']): run
                  for run in runs if run['pair'] == 0}
    reference_label = args.binary[0]['label']
    for key, run in sorted(first_pair.items()):
        cpu = first_pair.get((key[0], key[1], 'cpu', key[3], key[4]))
        if key[2] == 'cuda' and cpu is not None:
            comparisons.append(dict(kind='cuda_vs_cpu', binary=key[0], policy=key[1],
                                    bucket=key[3], order=key[4],
                                    result=compare_forces(cpu['force_csv'], run['force_csv'])))
        base = first_pair.get((reference_label, 'auto', key[2], key[3], key[4]))
        if base is not None and base is not run:
            comparisons.append(dict(kind='vs_' + reference_label, binary=key[0], policy=key[1],
                                    backend=key[2], bucket=key[3], order=key[4],
                                    result=compare_forces(base['force_csv'], run['force_csv'])))
    repeat_hashes = {}
    for run in runs:
        repeat_hashes.setdefault((run['binary'], run['policy'], run['backend'], run['bucket'],
                                  run['order']), set()).add(run['force_csv_sha256'])
    return {'timing': table, 'references': references,
            'arena_lines': [dict(binary=a[0], policy=a[1], backend=a[2], bucket=a[3], order=a[4],
                                 line=a[5]) for a in arenas],
            'comparisons': comparisons,
            'repeat_export_hashes': [dict(binary=k[0], policy=k[1], backend=k[2], bucket=k[3],
                                          order=k[4], distinct_exports=len(v))
                                     for k, v in sorted(repeat_hashes.items())]}


def render(summary):
    lines = ['| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |',
             '|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for row in summary['timing']:
        lines.append(f'| {row["binary"]} | {row["policy"]} | {row["backend"]} | {row["bucket"]} | '
                     f'{row["order"]} | {row["fixture"]} | {int(row["potential"])} | '
                     f'{row["kick_ms_median"]:.3f} | {row["kick_ms_min"]:.3f} | '
                     f'{row["kick_ms_max"]:.3f} | {row["pairs"]} |')
    if summary['references']:
        lines += ['', '| binary | policy | backend | bucket | order | fixture | force rel L2 | '
                  'force p99 | potential rel L2 | potential p99 |',
                  '|---|---|---|---:|---:|---|---:|---:|---:|---:|']
        seen = set()
        for row in summary['references']:
            key = (row['binary'], row['policy'], row['backend'], row['bucket'], row['order'],
                   row['fixture'])
            if key in seen:
                continue
            seen.add(key)
            lines.append(f'| {row["binary"]} | {row["policy"]} | {row["backend"]} | {row["bucket"]} | '
                         f'{row["order"]} | {row["fixture"]} | {row["force_relative_l2"]:.3e} | '
                         f'{row["force_relative_p99"]:.3e} | {row["potential_relative_l2"]:.3e} | '
                         f'{row["potential_relative_p99"]:.3e} |')
    lines += ['', '| binary | policy | backend | bucket | order | kernel policy line |',
              '|---|---|---|---:|---:|---|']
    for row in summary['arena_lines']:
        lines.append(f'| {row["binary"]} | {row["policy"]} | {row["backend"]} | {row["bucket"]} | '
                     f'{row["order"]} | `{row["line"]}` |')
    lines += ['', '| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | '
              'acceleration max pointwise | potential rel L2 |',
              '|---|---|---|---:|---:|---|---:|---:|---:|']
    for item in summary['comparisons']:
        for fixture, result in item['result'].items():
            acceleration = result['acceleration']
            potential = result['potential']
            lines.append(f'| {item["kind"]} | {item["binary"]} | {item["policy"]} | {item["bucket"]} | '
                         f'{item["order"]} | {fixture} | {acceleration["relative_l2"]:.3e} | '
                         f'{acceleration["max_pointwise_relative"]:.3e} | '
                         f'{potential["relative_l2"]:.3e} |')
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    main()
