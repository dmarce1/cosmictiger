#!/usr/bin/env python3
"""Split each profiled kick at its GPU phase (Nsight Systems CUDA API trace).

Usage: summarize_kick_handoff_profile.py DIR

DIR holds <variant>_<round>/ subdirectories, each with run.log (the walk
benchmark output) and the `nsys stats` exports stats_cuda_api_trace.csv and
stats_cuda_gpu_trace.csv. The kick launches its kernel once, from the thread
that hands the last workitem to the GPU, after the whole CPU walk has ended;
the launch is surrounded by a burst of CUDA API calls (device properties,
stream creation, the parameter upload, the launch, the blocking result copy,
the stream teardown). The GPU phase is that burst: API calls of any thread
chained by gaps below 1 ms around the launch. Each benchmark kick is paired
in order with a launch, and the kick time splits into

  host     kick minus GPU phase: the CPU walk down to the handoff level, the
           workitem bookkeeping and the preparation in KickWorkspace::toGpu
           before the burst, and the completion after it;
  launch   GPU phase minus the kernel: API calls, the upload and the wait
           around the kernel;
  kernel   the kick kernel's duration on the device.

It also counts the unified-memory migrations the device performs during the
kernel (page faults of the managed tree and particle arrays). The per-kick
values go to kick_split.csv in each run directory. The unified-memory rows
make the GPU trace export several megabytes per run; once they have been
removed from it, the counts are taken from the existing kick_split.csv. Medians
over the samples per round, then the median over the rounds per variant.

Usage: summarize_kick_handoff_profile.py --nvtx DIR

For runs of a binary instrumented with NVTX start/end ranges (the kick in the
walk benchmark, KickWorkspace::addWork and the stages of toGpu and
cudaExecuteKicks; see nsys_handoff_large/nvtx/instrumentation.patch), reads
the NVTX_EVENTS table of each run's kick.sqlite (written by `nsys stats`),
writes the per-kick phases to kick_phases.csv in the run directory and prints
the same medians as above for

  walk     kick start to the start of toGpu: the CPU walk down to the
           handoff level, ending when the last workitem is added;
  prepare  toGpu's bookkeeping before the launch (depth sort, tree map,
           remote nodes and list remapping, workitem sort);
  pack     cudaExecuteKicks packing the workitems and lists for the upload;
  device   upload, kernel and result copy (stream round trip);
  other    the rest of toGpu's launch step (tree and particle memory advice);
  tail     end of toGpu to the end of the kick;
  addWork  summed over threads: time in KickWorkspace::addWork, and the part
           of it spent holding the workspace mutex;
  cpu*     when the CPU walk is instrumented too (nvtx_walk/): thread time
           summed over the CPU walk's nodes ("cpu node": list processing and
           interactions of a node) and within it over the Ewald and direct
           cell-cell and the cell-particle calls, and the number of CPU nodes;

with the workitem and list counts toGpu reports. The results .gitignore keeps
kick.sqlite out of the repository; kick_phases.csv holds what this mode reads
from it.
"""
import csv
import re
import sqlite3
import statistics
import sys
from pathlib import Path

GAP_NS = 1_000_000


def samples(log):
    out = []
    for line in log.read_text().splitlines():
        if line.startswith('STF_WALK_SAMPLE,'):
            fields = line.split(',')
            out.append((int(fields[3]), int(fields[4]), 1e3 * float(fields[7])))
    return out


def phases(sub):
    api = []
    with (sub / 'stats_cuda_api_trace.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            start = int(row['Start (ns)'])
            api.append((start, start + int(row['Duration (ns)']), row['Name'], row['CorrID']))
    api.sort()
    kernels = {}
    unified = []
    with (sub / 'stats_cuda_gpu_trace.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            start = int(row['Start (ns)'])
            end = start + int(row['Duration (ns)'])
            if 'cudaKickKernel' in row['Name']:
                kernels[row['CorrId']] = (start, end)
            elif 'Unified Host-to-Device' in row['Name']:
                unified.append((start, float(row['Bytes (MB)'] or 0.0)))
    unified.sort()
    recorded = None
    if not unified and (sub / 'kick_split.csv').exists():
        with (sub / 'kick_split.csv').open(newline='') as handle:
            recorded = [(int(row['unified_h2d_transfers']), float(row['unified_h2d_mb']))
                        for row in csv.DictReader(handle)]
    result = []
    for index, (start, end, name, corr) in enumerate(api):
        if not name.startswith('cudaLaunchKernel') or corr not in kernels:
            continue
        first = index
        while first > 0 and api[first][0] - api[first - 1][1] < GAP_NS:
            first -= 1
        last = index
        reach = end
        while last + 1 < len(api) and api[last + 1][0] - reach < GAP_NS:
            last += 1
            reach = max(reach, api[last][1])
        begin = api[first][0]
        kernelBegin, kernelEnd = kernels[corr]
        if recorded is not None:
            count, megabytes = recorded[len(result)]
        else:
            faults = [mb for t, mb in unified if kernelBegin <= t <= kernelEnd]
            count, megabytes = len(faults), sum(faults)
        result.append(((reach - begin) * 1e-6, (kernelEnd - kernelBegin) * 1e-6, count, megabytes))
    return result


def main(directory):
    rows = {}
    for sub in sorted(Path(directory).iterdir()):
        if not (sub / 'stats_cuda_api_trace.csv').exists():
            continue
        variant, round_ = sub.name.rsplit('_', 1)
        kicks = samples(sub / 'run.log')
        gpu = phases(sub)
        if len(kicks) != len(gpu):
            print(f'{sub.name}: {len(kicks)} kicks but {len(gpu)} launches; skipped')
            continue
        with (sub / 'kick_split.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(('do_phi', 'sample', 'kick_ms', 'gpu_phase_ms', 'kernel_ms',
                             'unified_h2d_transfers', 'unified_h2d_mb'))
            for (phi, sample, kick), (phase, kernel, count, megabytes) in zip(kicks, gpu):
                writer.writerow((phi, sample, f'{kick:.4f}', f'{phase:.4f}', f'{kernel:.4f}', count,
                                 f'{megabytes:.3f}'))
        per = {}
        for (phi, sample, kick), (phase, kernel, count, megabytes) in zip(kicks, gpu):
            if sample >= 0:
                per.setdefault(phi, []).append((kick, kick - phase, phase - kernel, kernel, count,
                                                megabytes))
        rows.setdefault(variant, []).append((int(round_), per))
    columns = ('kick', 'host', 'launch', 'kernel', 'UM faults', 'UM MB')
    for phi, label in ((0, 'forces'), (1, 'forces and potential')):
        print(f'\nGPU-phase split, {label}: medians over the samples of each round (ms)\n')
        print('| variant | round | ' + ' | '.join(columns) + ' |')
        print('|---|---:|' + '---:|' * len(columns))
        summary = {}
        for variant, records in rows.items():
            for round_, per in sorted(records):
                values = per.get(phi, [])
                if not values:
                    continue
                medians = [statistics.median(v[i] for v in values) for i in range(len(columns))]
                summary.setdefault(variant, []).append(medians)
                print(f'| {variant} | {round_} | ' +
                      ' | '.join(f'{m:.0f}' if i == 4 else f'{m:.2f}' for i, m in enumerate(medians)) +
                      ' |')
        print(f'\nMedian over the rounds, {label} (ms)\n')
        print('| variant | ' + ' | '.join(columns) + ' |')
        print('|---|' + '---:|' * len(columns))
        for variant, medians in summary.items():
            cells = [statistics.median(m[i] for m in medians) for i in range(len(columns))]
            print(f'| {variant} | ' +
                  ' | '.join(f'{c:.0f}' if i == 4 else f'{c:.2f}' for i, c in enumerate(cells)) + ' |')


NVTX_COLUMNS = ('kick', 'walk', 'prepare', 'pack', 'device', 'other', 'tail', 'addWork', 'held',
                'cpuNode', 'cpuEwald', 'cpuDirect', 'cpuCp', 'cpuNodes', 'items', 'dchecks',
                'echecks', 'treeIds')
COUNT_COLUMNS = ('cpuNodes', 'items', 'dchecks', 'echecks', 'treeIds')


def nvtx_kicks(sub):
    connection = sqlite3.connect(sub / 'kick.sqlite')
    events = connection.execute(
        'SELECT start, end, text FROM NVTX_EVENTS WHERE text IS NOT NULL ORDER BY start').fetchall()
    connection.close()
    kicks = [(s, e) for s, e, text in events if text == 'kick' and e is not None]
    rows = []
    for begin, end in kicks:
        inside = [(s, e, text) for s, e, text in events if begin <= s <= end]

        def span(name):
            found = [(s, e) for s, e, text in inside if text == name and e is not None]
            return found[0] if found else (float('nan'), float('nan'))

        prepare = span('toGpu prepare')
        execute = span('toGpu execute')
        pack = span('execute pack')
        device = span('execute device')

        def total(name):
            return sum(e - s for s, e, text in inside if text == name and e is not None)

        add = total('addWork')
        held = total('addWork held')
        counts = {'items': 0, 'dchecks': 0, 'echecks': 0, 'treeIds': 0}
        for _, _, text in inside:
            if text.startswith('toGpu counts'):
                for key, value in re.findall(r'(\w+)=(\d+)', text):
                    if key in counts:
                        counts[key] = int(value)
        ms = 1e-6
        rows.append({
            'kick': (end - begin) * ms,
            'walk': (prepare[0] - begin) * ms,
            'prepare': (prepare[1] - prepare[0]) * ms,
            'pack': (pack[1] - pack[0]) * ms,
            'device': (device[1] - device[0]) * ms,
            'other': ((execute[1] - execute[0]) - (pack[1] - pack[0]) - (device[1] - device[0])) * ms,
            'tail': (end - execute[1]) * ms,
            'addWork': add * ms,
            'held': held * ms,
            'cpuNode': total('cpu node') * ms,
            'cpuEwald': total('cpu cc ewald') * ms,
            'cpuDirect': total('cpu cc direct') * ms,
            'cpuCp': total('cpu cp') * ms,
            'cpuNodes': sum(1 for _, _, text in inside if text == 'cpu node'),
            **counts})
    return rows


def main_nvtx(directory):
    rows = {}
    for sub in sorted(Path(directory).iterdir()):
        if not (sub / 'kick.sqlite').exists():
            continue
        variant, round_ = sub.name.rsplit('_', 1)
        kicks = samples(sub / 'run.log')
        phases_ = nvtx_kicks(sub)
        if len(kicks) != len(phases_):
            print(f'{sub.name}: {len(kicks)} kicks but {len(phases_)} kick ranges; skipped')
            continue
        with (sub / 'kick_phases.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(('do_phi', 'sample', 'benchmark_kick_ms') + NVTX_COLUMNS)
            for (phi, sample, kick), phase in zip(kicks, phases_):
                writer.writerow((phi, sample, f'{kick:.4f}') +
                                tuple(phase[c] if isinstance(phase[c], int) else f'{phase[c]:.4f}'
                                      for c in NVTX_COLUMNS))
        per = {}
        for (phi, sample, _), phase in zip(kicks, phases_):
            if sample >= 0:
                per.setdefault(phi, []).append([phase[c] for c in NVTX_COLUMNS])
        rows.setdefault(variant, []).append((int(round_), per))

    shown = [c for c in NVTX_COLUMNS
             if c not in ('cpuNode', 'cpuEwald', 'cpuDirect', 'cpuCp', 'cpuNodes') or
             any(values[NVTX_COLUMNS.index(c)] for records in rows.values() for _, per in records
                 for phase in per.values() for values in phase)]
    keep = [NVTX_COLUMNS.index(c) for c in shown]

    def cell(index, value):
        return f'{value:.0f}' if NVTX_COLUMNS[index] in COUNT_COLUMNS else f'{value:.2f}'

    for phi, label in ((0, 'forces'), (1, 'forces and potential')):
        print(f'\nNVTX phases, {label}: medians over the samples of each round (ms; counts per kick)\n')
        print('| variant | round | ' + ' | '.join(shown) + ' |')
        print('|---|---:|' + '---:|' * len(shown))
        summary = {}
        for variant, records in rows.items():
            for round_, per in sorted(records):
                values = per.get(phi, [])
                if not values:
                    continue
                medians = [statistics.median(v[i] for v in values) for i in range(len(NVTX_COLUMNS))]
                summary.setdefault(variant, []).append(medians)
                print(f'| {variant} | {round_} | ' +
                      ' | '.join(cell(i, medians[i]) for i in keep) + ' |')
        print(f'\nMedian over the rounds, {label} (ms; counts per kick)\n')
        print('| variant | ' + ' | '.join(shown) + ' |')
        print('|---|' + '---:|' * len(shown))
        for variant, medians in summary.items():
            cells = [statistics.median(m[i] for m in medians) for i in range(len(NVTX_COLUMNS))]
            print(f'| {variant} | ' + ' | '.join(cell(i, cells[i]) for i in keep) + ' |')


if __name__ == '__main__':
    if sys.argv[1] == '--nvtx':
        main_nvtx(sys.argv[2])
    else:
        main(sys.argv[1])
