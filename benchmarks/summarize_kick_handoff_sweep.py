#!/usr/bin/env python3
"""Summarize a handoff-divisor sweep over thread counts and particle counts.

Usage: summarize_kick_handoff_sweep.py ROOT [--reference POLICY] [--output table.md]

ROOT holds one run_cuda_kick_arena.py output directory per stage, named
t<threads>_n<side> (metadata.json records threads and particles_per_dimension;
the name is only used for ordering). For every stage, fixture and potential
setting it tabulates the median over pairs of the per-process median kick time
of each policy, the pairwise ratio of the reference policy's time to the
policy's (above one: the policy is faster than the reference) with its range
over pairs, and names the fastest policy. The reference defaults to
handoff164, the two-blocks-per-multiprocessor default. A second table gives,
per stage and policy, the geometric mean over all cases of the policy's time
relative to the reference and the worst case (both below one: faster), and a
closing table each stage's load average at the start and its worst pair
spread (slowest over fastest per-process median of one policy across pairs).
"""
import argparse
import json
import math
from pathlib import Path
import re
from statistics import median


def load_stage(directory):
    metadata = json.loads((directory / 'metadata.json').read_text())
    runs = json.loads((directory / 'runs.json').read_text())
    cells = {}
    for run in runs:
        for row in run['medians']:
            cell = (run['bucket'], row['fixture'], row['potential'])
            cells.setdefault(cell, {}).setdefault(run['policy'], {})[run['pair']] = row['kick_seconds']
    spread = max((max(t.values()) / min(t.values()) for cell in cells.values() for t in cell.values()
                  if len(t) > 1), default=1.0)
    return dict(threads=metadata.get('threads', 2), side=metadata.get('particles_per_dimension', 32),
                particles=runs[0]['medians'][0]['particles'], cells=cells, metadata=metadata,
                spread=spread, wall=[run['wall_seconds'] for run in runs])


def policy_key(policy):
    digits = re.findall(r'\d+', policy)
    return (int(digits[0]) if digits else -1, policy)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', type=Path)
    parser.add_argument('--reference', default='handoff164')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    stages = [load_stage(d) for d in sorted(args.root.iterdir())
              if d.is_dir() and (d / 'runs.json').exists()]
    stages.sort(key=lambda s: (s['side'], s['threads']))
    lines = []
    best = []
    for stage in stages:
        policies = sorted({p for cell in stage['cells'].values() for p in cell}, key=policy_key)
        lines += ['', f'### {stage["particles"]:,} particles, {stage["threads"]} threads '
                  f'(`{args.root.name}/t{stage["threads"]}_n{stage["side"]}/`)', '',
                  '| bucket | fixture | potential | ' + ' | '.join(policies) + ' | fastest |',
                  '|---:|---|---:|' + '---:|' * len(policies) + '---|']
        for cell in sorted(stage['cells']):
            times = stage['cells'][cell]
            reference = times.get(args.reference, {})
            fields = []
            medians = {}
            for policy in policies:
                values = times.get(policy)
                if not values:
                    fields.append('')
                    continue
                med = 1e3 * median(values.values())
                medians[policy] = med
                text = f'{med:.2f} ms'
                if reference and policy != args.reference:
                    ratios = [reference[p] / values[p] for p in values if p in reference]
                    if ratios:
                        text += f', {median(ratios):.3f}x ({min(ratios):.3f}-{max(ratios):.3f})'
                fields.append(text)
            fastest = min(medians, key=medians.get)
            loss = medians.get(args.reference, float('nan')) / medians[fastest]
            best.append(dict(side=stage['side'], particles=stage['particles'], threads=stage['threads'],
                             bucket=cell[0], fixture=cell[1], potential=cell[2], fastest=fastest,
                             reference_over_fastest=loss))
            lines.append(f'| {cell[0]} | {cell[1]} | {int(cell[2])} | ' + ' | '.join(fields) +
                         f' | {fastest} |')
    lines += ['', f'### Time relative to {args.reference}, geometric mean over cases (worst case)', '']
    all_policies = sorted({p for stage in stages for cell in stage['cells'].values() for p in cell},
                          key=policy_key)
    lines += ['| particles | threads | ' + ' | '.join(all_policies) + ' |',
              '|---:|---:|' + '---:|' * len(all_policies)]
    for stage in stages:
        fields = []
        for policy in all_policies:
            ratios = []
            for cell, times in stage['cells'].items():
                reference = times.get(args.reference)
                values = times.get(policy)
                if reference and values:
                    ratios.append(median(values.values()) / median(reference.values()))
            if ratios:
                mean = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
                fields.append(f'{mean:.3f} ({max(ratios):.3f})')
            else:
                fields.append('')
        lines.append(f'| {stage["particles"]:,} | {stage["threads"]} | ' + ' | '.join(fields) + ' |')
    lines += ['', f'### Fastest policy per case (reference {args.reference} time over the fastest time)', '',
              '| particles | threads | bucket | fixture | potential | fastest | reference / fastest |',
              '|---:|---:|---:|---|---:|---|---:|']
    for row in best:
        lines.append(f'| {row["particles"]:,} | {row["threads"]} | {row["bucket"]} | {row["fixture"]} | '
                     f'{int(row["potential"])} | {row["fastest"]} | {row["reference_over_fastest"]:.3f} |')
    lines += ['', '### Stage conditions', '',
              '| particles | threads | load average at start | worst pair spread | process wall s |',
              '|---:|---:|---:|---:|---:|']
    for stage in stages:
        lines.append(f'| {stage["particles"]:,} | {stage["threads"]} | '
                     f'{stage["metadata"]["load_average"][0]:.2f} | {stage["spread"]:.3f} | '
                     f'{min(stage["wall"]):.1f}-{max(stage["wall"]):.1f} |')
    text = '\n'.join(lines).lstrip('\n') + '\n'
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
