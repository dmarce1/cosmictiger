#!/usr/bin/env python3
"""Analyze completed NVTX ranges inside each STF benchmark kick.

Reads an existing SQLite export only. Cross-worker range durations are both
summed and unioned; unions measure wall coverage without counting overlap twice.
Nested labels are not additive. Summed range time is not CPU utilization: an
instrumented scope may include a suspended HPX task or a GPU wait.
"""
import argparse
import collections
import csv
import json
import pathlib
import sqlite3
import statistics

from analyze_nsys import parse_samples, interval_union_ms


def segments_by_labels(events, start, end):
    boundaries = collections.defaultdict(list)
    for event in events:
        lo, hi = max(start, event['start']), min(end, event['end'])
        if hi > lo:
            boundaries[lo].append((event['label'], 1))
            boundaries[hi].append((event['label'], -1))
    boundaries[start]
    boundaries[end]
    active = collections.Counter()
    previous = start
    result = collections.Counter()
    for timestamp in sorted(boundaries):
        if timestamp > previous:
            key = tuple(sorted(label for label, count in active.items() if count > 0))
            result[key] += (timestamp - previous) / 1e6
        for label, change in boundaries[timestamp]:
            active[label] += change
        previous = timestamp
    return [{'labels': list(key), 'duration_ms': value} for key, value in sorted(result.items())]


def summarize(values):
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                min=min(values), max=max(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', type=pathlib.Path)
    parser.add_argument('--stdout', type=pathlib.Path, required=True)
    parser.add_argument('--fixture', default='adaptive')
    parser.add_argument('--root-label', required=True)
    parser.add_argument('--expect-label', action='append', default=[], help='Include zero-valued absent labels.')
    parser.add_argument('--sum-payload-label', action='append', default=[],
                        help='Sum numeric payloads only for labels whose payload denotes an additive quantity.')
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    connection = sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if 'NVTX_EVENTS' not in tables:
        raise SystemExit('No NVTX_EVENTS table found.')
    strings = {r['id']: r['value'] for r in connection.execute('SELECT * FROM StringIds')}
    nvtx = [dict(r) for r in connection.execute('SELECT * FROM NVTX_EVENTS ORDER BY start')]
    valid = []
    incomplete = []
    for event in nvtx:
        event['label'] = strings.get(event.get('textId'), event.get('text') or '')
        if event.get('eventType') not in [59, 60, 70, 71]:
            continue
        if event.get('end') is None:
            incomplete.append(event)
            continue
        event['duration_ms'] = (event['end'] - event['start']) / 1e6
        valid.append(event)
    roots = [r for r in valid if r['label'] == args.root_label]
    roots.sort(key=lambda e: e['start'])
    samples = parse_samples(args.stdout, args.fixture)
    if len(roots) != len(samples):
        counts = dict(collections.Counter(e['label'] for e in valid))
        raise SystemExit(f'Expected one root range per sample: {len(roots)} roots vs {len(samples)} samples. Labels: {counts}')
    for previous, following in zip(roots, roots[1:]):
        if previous['end'] > following['start']:
            raise SystemExit('Root ranges overlap; they do not represent sequential benchmark kicks.')
    kernels = [dict(r) for r in connection.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL')]
    kernels = [k for k in kernels if 'cudaKickKernel' in strings.get(k.get('demangledName'), '')]
    ranges = [r for r in valid if r['label'] != args.root_label]
    labels = sorted(set(args.expect_label) | {r['label'] for r in ranges})
    payload_columns = [r[1] for r in connection.execute('PRAGMA table_info(NVTX_EVENTS)')
                       if 'value' in r[1].lower() or 'payload' in r[1].lower()]
    launches = []
    for index, (root, sample) in enumerate(zip(roots, samples)):
        start, end = root['start'], root['end']
        selected = [r for r in ranges if r['start'] < end and r['end'] > start]
        gpu = [k for k in kernels if k['start'] < end and k['end'] > start]
        if len(gpu) != 1:
            raise SystemExit(f'Root {index} overlaps {len(gpu)} kernels; expected one.')
        kernel = gpu[0]
        crossing = [r for r in selected if r['start'] < start or r['end'] > end]
        root_duration = (end - start) / 1e6
        record = dict(sample, index=index, warmup=sample['sample'] < 0,
                      root_start_ns=start, root_end_ns=end, root_ms=root_duration,
                      root_minus_stdout_ms=root_duration - sample['kick_ms'],
                      kernel_ms=interval_union_ms(gpu, start, end),
                      pre_kernel_ms=(kernel['start'] - start) / 1e6,
                      post_kernel_ms=(end - kernel['end']) / 1e6,
                      range_count=len(selected), crossing_range_count=len(crossing),
                      all_child_ranges_sum_ms=sum((min(end,r['end'])-max(start,r['start']))/1e6 for r in selected),
                      all_child_ranges_union_ms=interval_union_ms(selected,start,end),
                      labels={},
                      wall_by_active_labels_ms=segments_by_labels(selected,start,end))
        for label in labels:
            group = [r for r in selected if r['label'] == label]
            value = dict(count=len(group),
                         summed_range_ms=sum((min(end,r['end'])-max(start,r['start']))/1e6 for r in group),
                         union_ms=interval_union_ms(group,start,end),
                         union_pre_kernel_ms=interval_union_ms(group,start,kernel['start']),
                         union_during_kernel_ms=interval_union_ms(group,kernel['start'],kernel['end']),
                         union_post_kernel_ms=interval_union_ms(group,kernel['end'],end),
                         maximum_range_ms=max((r['duration_ms'] for r in group),default=0),
                         begin_thread_count=len({r.get('globalTid') for r in group}),
                         end_thread_count=len({r.get('endGlobalTid') for r in group if r.get('endGlobalTid') is not None}))
            payload_values = [{k:r[k] for k in payload_columns if r.get(k) is not None} for r in group]
            payload_values = [v for v in payload_values if v]
            if payload_values:
                value['payload_values'] = payload_values
                if label in args.sum_payload_label:
                    value['payload_sums'] = {k:sum(v[k] for v in payload_values if isinstance(v.get(k),(int,float)))
                                             for k in payload_columns}
            record['labels'][label] = value
        record['uncovered_wall_ms'] = root_duration - record['all_child_ranges_union_ms']
        launches.append(record)
    summaries = {}
    scalar_metrics = ['kick_ms','root_ms','root_minus_stdout_ms','kernel_ms','pre_kernel_ms','post_kernel_ms',
                      'range_count','crossing_range_count','all_child_ranges_sum_ms','all_child_ranges_union_ms','uncovered_wall_ms']
    label_metrics = ['count','summed_range_ms','union_ms','union_pre_kernel_ms','union_during_kernel_ms',
                     'union_post_kernel_ms','maximum_range_ms','begin_thread_count','end_thread_count']
    for fixture, phi in sorted({(r['fixture'],r['do_phi']) for r in launches}):
        group = [r for r in launches if not r['warmup'] and r['fixture']==fixture and r['do_phi']==phi]
        out = {metric:summarize([r[metric] for r in group]) for metric in scalar_metrics}
        out['labels'] = {label:{metric:summarize([r['labels'][label][metric] for r in group])
                               for metric in label_metrics} for label in labels}
        for label in args.sum_payload_label:
            fields = {k for r in group for k in r['labels'].get(label,{}).get('payload_sums',{})}
            if fields:
                out['labels'][label]['payload_sums'] = {k:summarize([r['labels'][label].get('payload_sums',{}).get(k,0)
                                                                    for r in group]) for k in fields}
        summaries[f'{fixture}:phi={phi}'] = out
    output = dict(database=str(args.database.resolve()),stdout=str(args.stdout.resolve()),root_label=args.root_label,
                  payload_columns=payload_columns,summary=summaries,launches=launches,
                  incomplete_ranges=incomplete,ranges=valid,
                  caveats=['Summed range duration can double-count nested scopes and concurrent workers; union_ms does not.',
                           'Different labels may overlap and must not be added. wall_by_active_labels_ms is an exact nonoverlapping partition.',
                           'Scoped elapsed time includes HPX suspension and waits, so summed range duration is not CPU utilization.',
                           'Pre/kernel/post windows use the sole kernel inside each root; async ranges are matched by timestamp, independent of worker identity.',
                           'Payloads are preserved; sums are only produced for explicitly requested additive payload labels.',
                           'Warmups are sample=-1 and excluded from summaries. Missing expected labels are represented as zeros.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    scalar_rows = [{k:r[k] for k in ['index','fixture','do_phi','sample','warmup']+scalar_metrics} for r in launches]
    with args.output.with_suffix('.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(scalar_rows[0]));writer.writeheader();writer.writerows(scalar_rows)
    label_rows = [dict(index=r['index'],fixture=r['fixture'],do_phi=r['do_phi'],sample=r['sample'],label=label,
                       **{k:v[k] for k in label_metrics}) for r in launches for label,v in r['labels'].items()]
    with args.output.with_name(args.output.stem+'_labels.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(label_rows[0]));writer.writeheader();writer.writerows(label_rows)
    print(json.dumps(summaries,indent=2))


if __name__ == '__main__':
    main()
