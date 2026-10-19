#!/usr/bin/env python3
"""Summarize Nsight Systems kick profiles and theta sweeps of the walk benchmark.

nsys mode:   summarize_kick_profiles.py nsys DIR
  DIR holds <variant>_<round>/ subdirectories, each with run.log (the benchmark
  output) and stats_cuda_gpu_kern_sum.csv from `nsys stats`. Reports, per
  variant, the median whole-kick time from the STF_WALK_MEDIAN rows, the mean
  kick kernel duration from the kernel summary, and the difference (host side).
  When stats_cuda_gpu_trace.csv is present too, the kick kernel launches are
  paired in order with the benchmark's kicks (one launch per kick: a warmup
  and the samples with forces only, then the same with the potential) and the
  per-phase medians of the kernel duration and of the host share (whole kick
  minus its kernel) are reported instead of the mean.

theta mode:  summarize_kick_profiles.py theta DIR
  DIR holds theta_<value>/ result directories written by run_cuda_kick_arena.py
  (one order per run). Prints, per order and theta, the kick time and the direct
  reference force error, then for a few target errors the fastest (order, theta)
  whose measured error is at or below the target: a matched-accuracy comparison
  on the measured grid, without interpolation.
"""
import csv
import json
import statistics
import sys
from pathlib import Path


def nsys(directory):
    rows = {}
    for sub in sorted(Path(directory).iterdir()):
        if not sub.is_dir():
            continue
        variant = sub.name.rsplit('_', 1)[0]
        kicks = []
        samples = []
        for line in (sub / 'run.log').read_text().splitlines():
            if line.startswith('STF_WALK_MEDIAN,'):
                fields = line.split(',')
                kicks.append((int(fields[3]), float(fields[8])))
            elif line.startswith('STF_WALK_SAMPLE,'):
                fields = line.split(',')
                samples.append((int(fields[3]), int(fields[4]), float(fields[7])))
        kernel = None
        summary = sub / 'stats_cuda_gpu_kern_sum.csv'
        if summary.exists():
            with summary.open(newline='') as handle:
                for row in csv.DictReader(handle):
                    if 'cudaKickKernel' in row.get('Name', ''):
                        kernel = (int(row['Instances']), float(row['Avg (ns)']) * 1e-6)
        phases = None
        trace = sub / 'stats_cuda_gpu_trace.csv'
        if trace.exists() and samples:
            launches = []
            with trace.open(newline='') as handle:
                for row in csv.DictReader(handle):
                    if 'cudaKickKernel' in row.get('Name', ''):
                        launches.append((int(row['Start (ns)']), float(row['Duration (ns)']) * 1e-6))
            launches = [duration for _, duration in sorted(launches)]
            if len(launches) == len(samples):
                phases = {}
                for (phi, sample, kick), duration in zip(samples, launches):
                    if sample >= 0:
                        phases.setdefault(phi, []).append((1e3 * kick, duration))
        rows.setdefault(variant, []).append((kicks, kernel, phases))
    print('| variant | round | whole kick forces (ms) | whole kick forces+potential (ms) | kernel launches | mean kernel (ms) | host share forces (ms) |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for variant, records in rows.items():
        for index, (kicks, kernel, phases) in enumerate(records, 1):
            forces = next((1e3 * t for phi, t in kicks if phi == 0), float('nan'))
            potential = next((1e3 * t for phi, t in kicks if phi == 1), float('nan'))
            launches, mean = kernel if kernel else (0, float('nan'))
            print(f'| {variant} | {index} | {forces:.3f} | {potential:.3f} | {launches} | {mean:.3f} | '
                  f'{forces - mean:.3f} |')
    if any(phases for records in rows.values() for _, _, phases in records):
        print()
        print('Per-phase medians over the samples (kernel launches paired with kicks in order):')
        print()
        print('| variant | round | forces: kick (ms) | kernel (ms) | host (ms) | forces+potential: kick (ms) | kernel (ms) | host (ms) |')
        print('|---|---:|---:|---:|---:|---:|---:|---:|')
        for variant, records in rows.items():
            for index, (kicks, kernel, phases) in enumerate(records, 1):
                if not phases:
                    continue
                cells = []
                for phi in (0, 1):
                    pairs = phases.get(phi, [])
                    kick = statistics.median(k for k, _ in pairs) if pairs else float('nan')
                    duration = statistics.median(d for _, d in pairs) if pairs else float('nan')
                    host = statistics.median(k - d for k, d in pairs) if pairs else float('nan')
                    cells += [f'{kick:.3f}', f'{duration:.3f}', f'{host:.3f}']
                print(f'| {variant} | {index} | ' + ' | '.join(cells) + ' |')


def theta(directory):
    table = {}
    for sub in sorted(Path(directory).glob('theta_*')):
        value = float(sub.name.split('_', 1)[1])
        runs = json.loads((sub / 'runs.json').read_text())
        for run in runs:
            for row in run['medians']:
                if row['potential']:
                    continue
                error = next((r['force_relative_l2'] for r in run['references']
                              if r['fixture'] == row['fixture']), None)
                table[(run['order'], value, row['fixture'])] = (1e3 * row['kick_seconds'], error)
    print('| fixture | order | theta | kick forces (ms) | force rel L2 |')
    print('|---|---:|---:|---:|---:|')
    for (order, value, fixture), (ms, error) in sorted(table.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
        print(f'| {fixture} | {order} | {value} | {ms:.3f} | {error if error is None else f"{error:.3e}"} |')
    fixtures = sorted({k[2] for k in table})
    for fixture in fixtures:
        print(f'\nFastest measured configuration at or below a force error target ({fixture}):')
        print('| target force rel L2 | order | theta | kick forces (ms) | measured error |')
        print('|---:|---:|---:|---:|---:|')
        for target in (3e-2, 1e-2, 5e-3, 3e-3, 2e-3, 1e-3, 5e-4):
            candidates = [(ms, order, value, error) for (order, value, f), (ms, error) in table.items()
                          if f == fixture and error is not None and error <= target]
            if candidates:
                ms, order, value, error = min(candidates)
                print(f'| {target:.0e} | {order} | {value} | {ms:.3f} | {error:.3e} |')
            else:
                print(f'| {target:.0e} | none | | | |')


if __name__ == '__main__':
    mode, directory = sys.argv[1], sys.argv[2]
    nsys(directory) if mode == 'nsys' else theta(directory)
