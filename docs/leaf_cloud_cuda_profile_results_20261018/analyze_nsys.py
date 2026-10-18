#!/usr/bin/env python3
"""Align an Nsight Systems SQLite export with STF walk benchmark stdout.

No CUDA workload is launched. All time units in the output are milliseconds.
CUDA API durations often include time waiting for a kernel, so they must not be
added to kernel duration. Launch-period event totals also include CPU-side tree
rebuild/reset time between benchmark samples; they are diagnostic context, not a
direct partition of the reported kick timer. Only kick_ms - kernel_ms is an exact
measured non-kernel residual (assuming one cudaKickKernel per benchmark sample).
"""
import argparse
import collections
import csv
import json
import pathlib
import sqlite3
import statistics


def duration_ms(event):
    return (event['end'] - event['start']) / 1e6


def interval_union_ms(events, start, end):
    intervals = sorted((max(start, e['start']), min(end, e['end'])) for e in events
                       if e['start'] < end and e['end'] > start)
    total = 0
    cursor = start
    for lo, hi in intervals:
        total += max(0, hi - max(lo, cursor))
        cursor = max(cursor, hi)
    return total / 1e6


def summarize_events(events, start, end, key):
    if end is None:
        return None
    groups = collections.defaultdict(list)
    for event in events:
        if event['start'] < end and event['end'] > start:
            groups[key(event)].append(event)
    out = {}
    for label, group in sorted(groups.items()):
        out[label] = {
            'count': len(group),
            'sum_duration_ms': sum(max(0, min(end, e['end']) - max(start, e['start'])) / 1e6 for e in group),
            'union_duration_ms': interval_union_ms(group, start, end),
            'span_ms': (min(end, max(e['end'] for e in group)) - max(start, min(e['start'] for e in group))) / 1e6,
        }
        if any('bytes' in e for e in group):
            out[label]['bytes'] = sum(e.get('bytes', 0) for e in group)
    return out


def parse_samples(path, fixture):
    samples = []
    if path is None:
        return samples
    with path.open() as stream:
        for line in stream:
            if not line.startswith('STF_WALK_SAMPLE,'):
                continue
            row = next(csv.reader([line]))
            if row[2] != 'cuda' or (fixture and row[1] != fixture):
                continue
            samples.append(dict(fixture=row[1], do_phi=int(row[3]), sample=int(row[4]),
                                setup_ms=1000 * float(row[5]), tree_ms=1000 * float(row[6]),
                                kick_ms=1000 * float(row[7])))
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', type=pathlib.Path)
    parser.add_argument('--stdout', type=pathlib.Path)
    parser.add_argument('--fixture', help='Filter stdout samples (only if trace contains this fixture alone).')
    parser.add_argument('--samples', type=int, default=15,
                        help='Samples per phi group, excluding one warmup; used only without stdout.')
    parser.add_argument('--output', type=pathlib.Path, required=True, help='Output JSON; sibling CSV also written.')
    args = parser.parse_args()
    connection = sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    def rows(table):
        return [dict(r) for r in connection.execute('SELECT * FROM "' + table + '"')] if table in tables else []

    strings = {r['id']: r['value'] for r in rows('StringIds')}
    apis = rows('CUPTI_ACTIVITY_KIND_RUNTIME') + rows('CUPTI_ACTIVITY_KIND_DRIVER')
    for event in apis:
        event['name'] = strings.get(event.get('nameId'), str(event.get('nameId')))
    apis.sort(key=lambda e: e['start'])
    kernels = rows('CUPTI_ACTIVITY_KIND_KERNEL')
    kernels = [k for k in kernels if 'cudaKickKernel' in strings.get(k.get('demangledName'), '')]
    kernels.sort(key=lambda e: e['start'])
    if not kernels:
        raise SystemExit('No cudaKickKernel events found.')
    launches = []
    for kernel in kernels:
        matched = [a for a in apis if a.get('correlationId') == kernel['correlationId']
                   and 'LaunchKernel' in a['name']]
        if len(matched) != 1:
            raise SystemExit(f"Expected exactly one launch API for correlation {kernel['correlationId']}: {matched}")
        launches.append(matched[0])
    copies = rows('CUPTI_ACTIVITY_KIND_MEMCPY')
    memsets = rows('CUPTI_ACTIVITY_KIND_MEMSET')
    synchronizations = rows('CUPTI_ACTIVITY_KIND_SYNCHRONIZATION')
    copy_labels = {r['id']: r['label'] for r in rows('ENUM_CUDA_MEMCPY_OPER')}
    migration_labels = {r['id']: r['label'] for r in rows('ENUM_CUDA_UNIF_MEM_MIGRATION')}
    for event in copies:
        event['copy_label'] = copy_labels.get(event.get('copyKind'), str(event.get('copyKind')))
        cause = event.get('migrationCause')
        event['migration_label'] = migration_labels.get(cause, str(cause)) if cause is not None else None
    copy_key = lambda event: event['copy_label'] + (' / ' + event['migration_label'] if event['migration_label'] else '')
    samples = parse_samples(args.stdout, args.fixture)
    if args.stdout and len(samples) != len(kernels):
        raise SystemExit(f'Cannot align {len(samples)} stdout samples with {len(kernels)} kernels.')
    if not args.stdout:
        if len(kernels) != 2 * (args.samples + 1):
            raise SystemExit('No stdout provided and kernel count differs from two groups with --samples + 1 launches each.')
        samples = [dict(fixture=args.fixture or 'unknown', do_phi=i // (args.samples + 1),
                        sample=i % (args.samples + 1) - 1) for i in range(len(kernels))]
    per_launch = []
    for index, (kernel, launch, sample) in enumerate(zip(kernels, launches, samples)):
        next_start = launches[index + 1]['start'] if index + 1 < len(launches) else None
        post_limit = next_start if next_start is not None else float('inf')
        sync = next((a for a in apis if a['start'] >= launch['end'] and a['start'] < post_limit
                     and a['globalTid'] == launch['globalTid'] and 'Synchronize' in a['name']), None)
        first_return_copy = next((a for a in apis if a['start'] >= launch['end'] and a['start'] < post_limit
                                 and a['globalTid'] == launch['globalTid'] and 'Memcpy' in a['name']), None)
        record = dict(sample, index=index, warmup=sample['sample'] < 0,
                      correlation_id=kernel['correlationId'], device=kernel['deviceId'], stream=kernel['streamId'],
                      kernel_start_ns=kernel['start'], kernel_end_ns=kernel['end'],
                      launch_api_start_ns=launch['start'], launch_api_end_ns=launch['end'],
                      kernel_ms=duration_ms(kernel), launch_api_ms=duration_ms(launch),
                      launch_to_kernel_start_ms=(kernel['start'] - launch['start']) / 1e6,
                      launch_return_to_kernel_start_ms=(kernel['start'] - launch['end']) / 1e6,
                      registers_per_thread=kernel.get('registersPerThread'),
                      grid=[kernel.get('gridX'), kernel.get('gridY'), kernel.get('gridZ')],
                      block=[kernel.get('blockX'), kernel.get('blockY'), kernel.get('blockZ')],
                      static_shared_bytes=kernel.get('staticSharedMemory'),
                      dynamic_shared_bytes=kernel.get('dynamicSharedMemory'),
                      local_memory_per_thread=kernel.get('localMemoryPerThread'),
                      local_memory_total=kernel.get('localMemoryTotal'))
        preceding_copy = next((a for a in reversed(apis) if a['end'] <= launch['start']
                               and a['globalTid'] == launch['globalTid'] and 'Memcpy' in a['name']), None)
        if preceding_copy:
            payload_copies = [e for e in copies if e.get('correlationId') == preceding_copy['correlationId']
                              and e.get('copyKind') == 1]
            record.update(prelaunch_memcpy_api_ms=duration_ms(preceding_copy),
                          explicit_htod_bytes=sum(e['bytes'] for e in payload_copies),
                          explicit_htod_device_ms=sum(duration_ms(e) for e in payload_copies))
        if 'kick_ms' in sample:
            record['nonkernel_kick_ms'] = sample['kick_ms'] - record['kernel_ms']
        if sync:
            record.update(first_sync_name=sync['name'], first_sync_api_ms=duration_ms(sync),
                          kernel_end_to_sync_start_ms=(sync['start'] - kernel['end']) / 1e6,
                          kernel_end_to_sync_end_ms=(sync['end'] - kernel['end']) / 1e6,
                          launch_to_sync_end_ms=(sync['end'] - launch['start']) / 1e6)
        if first_return_copy:
            record.update(postlaunch_memcpy_api_ms=duration_ms(first_return_copy),
                          kernel_end_to_memcpy_return_ms=(first_return_copy['end'] - kernel['end']) / 1e6)
        record['copies_during_kernel'] = summarize_events(copies, kernel['start'], kernel['end'], copy_key)
        record['copies_after_kernel_until_next_launch'] = summarize_events(copies, kernel['end'], next_start, copy_key)
        record['copies_launch_until_next_launch'] = summarize_events(copies, launch['start'], next_start, copy_key)
        record['apis_launch_until_next_launch'] = summarize_events(apis, launch['start'], next_start, lambda a: a['name'])
        record['sync_launch_until_next_launch'] = summarize_events(synchronizations, launch['start'], next_start,
                                                                  lambda e: str(e.get('syncType')))
        record['memsets_launch_until_next_launch'] = summarize_events(memsets, launch['start'], next_start, lambda e: 'memset')
        if next_start is not None:
            record['launch_to_next_launch_ms'] = (next_start - launch['start']) / 1e6
            record['kernel_end_to_next_launch_ms'] = (next_start - kernel['end']) / 1e6
        per_launch.append(record)
    summary = {}
    groups = collections.defaultdict(list)
    for record in per_launch:
        if not record['warmup']:
            groups[(record['fixture'], record['do_phi'])].append(record)
    scalar_metrics = ['prelaunch_memcpy_api_ms', 'explicit_htod_bytes', 'explicit_htod_device_ms', 'kick_ms', 'kernel_ms', 'nonkernel_kick_ms', 'launch_api_ms', 'launch_to_kernel_start_ms',
                      'launch_return_to_kernel_start_ms', 'first_sync_api_ms', 'kernel_end_to_sync_start_ms',
                      'kernel_end_to_sync_end_ms', 'launch_to_sync_end_ms', 'postlaunch_memcpy_api_ms',
                      'kernel_end_to_memcpy_return_ms', 'launch_to_next_launch_ms', 'kernel_end_to_next_launch_ms']
    nested_metrics = ['copies_during_kernel', 'copies_after_kernel_until_next_launch', 'copies_launch_until_next_launch',
                      'apis_launch_until_next_launch', 'memsets_launch_until_next_launch', 'sync_launch_until_next_launch']
    for (fixture, phi), group in sorted(groups.items()):
        out = {'samples': len(group)}
        for metric in scalar_metrics:
            values = [g[metric] for g in group if metric in g]
            if values:
                out[metric] = dict(n=len(values), median=statistics.median(values), mean=statistics.mean(values),
                                   min=min(values), max=max(values))
        for metric in nested_metrics:
            out[metric] = {}
            labels = sorted({label for g in group for label in (g.get(metric) or {})})
            for label in labels:
                eligible = [g[metric] for g in group if g[metric] is not None]
                fields = sorted({field for g in eligible for field in g.get(label, {})})
                out[metric][label] = {field: dict(n=len(eligible), median=statistics.median(
                    g.get(label, {}).get(field, 0) for g in eligible), mean=statistics.mean(
                    g.get(label, {}).get(field, 0) for g in eligible)) for field in fields}
        summary[f'{fixture}:phi={phi}'] = out
    payload = dict(database=str(args.database.resolve()), stdout=str(args.stdout.resolve()) if args.stdout else None,
                   caveats=[
                       'API durations overlap kernel execution and must not be added to kernel duration.',
                       'Nonkernel kick is stdout kick minus one aligned cudaKickKernel; includes CPU work and memory fault handling.',
                       'Launch-to-launch event windows include reset/tree work between samples and are not exact kick-timer bounds.',
                       'The final launch has no following launch, so launch-to-launch and post-kernel window metrics are null.',
                       'Transfer bytes count whole overlapping events; transfer durations are clipped to each diagnostic window.',
                       'Warmups are aligned from sample=-1 and excluded from summaries; every kernel remains in launch detail.',
                   ], tables=sorted(tables), summary=summary, launches=per_launch,
                   memory_events=copies, synchronization_events=synchronizations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    flat = [{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in per_launch]
    fieldnames = list(dict.fromkeys(key for row in flat for key in row))
    with args.output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
