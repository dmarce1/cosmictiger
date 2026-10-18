#!/usr/bin/env python3
"""Aggregate six read-only Nsight trace analyses; never launches a workload."""
import collections
import json
import pathlib
import statistics

base = pathlib.Path(__file__).resolve().parent
manifest = json.loads((base / 'profile_manifest.json').read_text())
profiles = [p for p in manifest['profiles'] if p['kind'] == 'nsys']
records = []
for profile in profiles:
    data = json.loads((base / (profile['name'] + '_analysis.json')).read_text())
    for record in data['launches']:
        if record['warmup']:
            continue
        row = dict(record, pair=profile['pair'], variant=profile['variant'])
        for metric, prefix, selector in [
            ('copies_during_kernel', 'uvm_htod_kernel', 'Unified Host-to-Device'),
            ('copies_after_kernel_until_next_launch', 'uvm_dtoh_postkernel', 'Unified Device-to-Host'),
            ('copies_after_kernel_until_next_launch', 'explicit_htod_between', 'Host-to-Device'),
        ]:
            group = row[metric]
            if group is not None:
                filtered = [v for k, v in group.items() if k == selector or k.startswith(selector + ' / ')]
                for field in ['bytes', 'count', 'sum_duration_ms']:
                    row[prefix + '_' + field] = sum(v[field] for v in filtered)
                # Individual migration causes share a span, so take their maximum rather than add.
                row[prefix + '_max_cause_span_ms'] = max((v['span_ms'] for v in filtered), default=0)
        records.append(row)
metrics = ['kick_ms', 'kernel_ms', 'nonkernel_kick_ms', 'launch_api_ms', 'launch_to_kernel_start_ms',
           'kernel_end_to_sync_end_ms', 'postlaunch_memcpy_api_ms', 'kernel_end_to_memcpy_return_ms',
           'uvm_htod_kernel_bytes', 'uvm_htod_kernel_count', 'uvm_htod_kernel_sum_duration_ms',
           'uvm_htod_kernel_max_cause_span_ms', 'uvm_dtoh_postkernel_bytes', 'uvm_dtoh_postkernel_count',
           'uvm_dtoh_postkernel_sum_duration_ms', 'uvm_dtoh_postkernel_max_cause_span_ms',
           'explicit_htod_bytes', 'explicit_htod_device_ms', 'prelaunch_memcpy_api_ms']
result = {}
for phi in [0, 1]:
    group = [r for r in records if r['do_phi'] == phi]
    summaries = {}
    for metric in metrics:
        variants = {}
        for variant in ['before', 'after']:
            values = [r[metric] for r in group if r['variant'] == variant and metric in r]
            run_medians = [statistics.median(r[metric] for r in group
                if r['variant'] == variant and r['pair'] == pair and metric in r) for pair in range(3)]
            variants[variant] = dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                                     median_of_run_medians=statistics.median(run_medians))
        delta = variants['after']['mean'] - variants['before']['mean']
        summaries[metric] = dict(**variants, mean_delta=delta,
            mean_percent=100 * delta / variants['before']['mean'] if variants['before']['mean'] else None)
    pairs = []
    for pair in range(3):
        entry = {'pair': pair}
        for metric in ['kick_ms', 'kernel_ms', 'nonkernel_kick_ms']:
            values = {v: statistics.mean(r[metric] for r in group if r['variant'] == v and r['pair'] == pair)
                      for v in ['before', 'after']}
            entry[metric] = dict(**values, delta=values['after'] - values['before'])
        pairs.append(entry)
    result[str(phi)] = dict(metrics=summaries, pairs=pairs,
        nonkernel_share_of_mean_kick_delta=summaries['nonkernel_kick_ms']['mean_delta'] / summaries['kick_ms']['mean_delta'])
payload = dict(method='Arithmetic means across 3 pairs x 15 post-warmup samples per phi and variant. Means preserve additive kick=kernel+residual decomposition. Median and median-of-run-medians also included.',
               caveats=['Nsight profiling perturbs absolute timing; use original unprofiled benchmark for headline performance.',
                        'Nonkernel residual is measured wall kick minus the CUDA kernel; it includes host preparation, CPU computations, memory-fault handling, and synchronization.',
                        'Launch-window UVM metrics include inter-sample reset/tree work and cannot locate CPU fault time specifically inside the kick timer.',
                        'Final sample lacks a following launch, so post-kernel migration metrics cover 42 instead of45 samples for phi1.',
                        'Summed transfer durations are not page-fault service durations; API durations overlap kernel execution.'],
               groups=result)
(base / 'nsys_aggregate.json').write_text(json.dumps(payload, indent=2) + '\n')
lines = ['# CUDA trace comparison', '', payload['method'], '',
         'Three alternating before/after pairs; each launch aligned one-to-one with benchmark stdout. Warmups excluded.', '',
         '| Mode | Component | Before (ms) | After (ms) | Change (ms) | Change (%) |',
         '|---|---|---:|---:|---:|---:|']
for phi in [0,1]:
    for metric, label in [('kick_ms','Whole kick'),('kernel_ms','CUDA kernel'),('nonkernel_kick_ms','Outside kernel')]:
        m=result[str(phi)]['metrics'][metric]
        lines.append(f"| {'Force + potential' if phi else 'Force'} | {label} | {m['before']['mean']:.4f} | {m['after']['mean']:.4f} | {m['mean_delta']:+.4f} | {m['mean_percent']:+.2f}% |")
lines += ['', 'The nonkernel residual accounts for ' + ' and '.join(f"{100*result[str(phi)]['nonkernel_share_of_mean_kick_delta']:.1f}%" for phi in [0,1]) + ' of the mean regression, respectively.', '',
          '| Mode | Diagnostic | Before | After |', '|---|---|---:|---:|']
for phi in [0,1]:
    for metric,label in [('launch_to_kernel_start_ms','Launch to kernel start (ms)'),('kernel_end_to_sync_end_ms','Kernel end to sync return (ms)'),('uvm_htod_kernel_bytes','UVM HtoD bytes during kernel'),('uvm_htod_kernel_sum_duration_ms','UVM HtoD transfer sum (ms)'),('uvm_dtoh_postkernel_bytes','UVM DtoH bytes until next launch'),('uvm_dtoh_postkernel_sum_duration_ms','UVM DtoH transfer sum (ms)'),('uvm_dtoh_postkernel_max_cause_span_ms','UVM DtoH migration span (ms)'),('explicit_htod_bytes','Explicit HtoD payload bytes'),('prelaunch_memcpy_api_ms','Prelaunch HtoD API time (ms)')]:
        m=result[str(phi)]['metrics'][metric]
        fmt='.0f' if metric.endswith('bytes') else '.4f'
        lines.append(f"| {'Force + potential' if phi else 'Force'} | {label} | {m['before']['mean']:{fmt}} | {m['after']['mean']:{fmt}} |")
lines += ['', 'UVM migration volume, transfer time, and post-kernel span are essentially unchanged. Launch and synchronization latencies change by only microseconds. These traces localize most extra time outside the GPU kernel, but cannot alone identify a particular CPU routine.', '', 'Caveats:', '']
lines += ['- '+c for c in payload['caveats']]
(base / 'nsys_summary.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
