#!/usr/bin/env python3
"""Summarize run_cuda_kick_arena.py results relative to one reference variant.

Usage: compare_kick_arena_runs.py RESULTS_DIR [--reference binary:policy]
                                  [--output table.md]

For every (backend, bucket, order, fixture, potential) cell it reports the median
over pairs of the per-process median kick time for each variant, the ratio
reference / variant (speedup above one means the variant is faster) and the
range of that ratio across pairs, plus the direct-reference force and potential
errors when the runs recorded them. Ratios are computed pairwise, so slow drifts
of the shared GPU affect both sides of each pair alike.
"""
import argparse
import json
from pathlib import Path
from statistics import median


def load_runs(directory):
    runs = json.loads((Path(directory) / 'runs.json').read_text())
    for run in runs:
        run.setdefault('order', 7)
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('results', type=Path)
    parser.add_argument('--reference', default=None,
                        help='binary:policy used as the reference (default: first run)')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    runs = load_runs(args.results)
    if args.reference:
        reference_binary, reference_policy = args.reference.split(':')
    else:
        reference_binary, reference_policy = runs[0]['binary'], runs[0]['policy']
    cells = {}
    for run in runs:
        for row in run['medians']:
            cell = (run['backend'], run['bucket'], row['fixture'], row['potential'])
            variant = (run['binary'], run['policy'], run['order'])
            cells.setdefault(cell, {}).setdefault(variant, {})[run['pair']] = row['kick_seconds']
    errors = {}
    for run in runs:
        for reference in run['references']:
            errors[(run['binary'], run['policy'], run['order'], run['backend'], run['bucket'],
                    reference['fixture'])] = reference
    lines = ['| backend | bucket | fixture | potential | binary | policy | order | kick ms | '
             'speedup vs reference | paired range | force rel L2 | potential rel L2 |',
             '|---|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|']
    for cell in sorted(cells):
        variants = cells[cell]
        reference_key = next((k for k in variants if k[0] == reference_binary and
                              k[1] == reference_policy), None)
        reference_times = variants.get(reference_key, {}) if reference_key else {}
        for variant in sorted(variants):
            times = variants[variant]
            med = 1e3 * median(times.values())
            speedup = ''
            spread = ''
            if reference_times and variant != reference_key:
                ratios = [reference_times[p] / times[p] for p in times if p in reference_times]
                if ratios:
                    speedup = f'{median(ratios):.3f}'
                    spread = f'{min(ratios):.3f}-{max(ratios):.3f}'
            error = errors.get((variant[0], variant[1], variant[2], cell[0], cell[1], cell[2]))
            force = f'{error["force_relative_l2"]:.3e}' if error else ''
            potential = f'{error["potential_relative_l2"]:.3e}' if error else ''
            lines.append(f'| {cell[0]} | {cell[1]} | {cell[2]} | {int(cell[3])} | {variant[0]} | '
                         f'{variant[1]} | {variant[2]} | {med:.3f} | {speedup} | {spread} | '
                         f'{force} | {potential} |')
    text = '\n'.join(lines) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
