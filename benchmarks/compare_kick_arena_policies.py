#!/usr/bin/env python3
"""Compare force exports of kick arena policies against one policy per bucket.

Usage: compare_kick_arena_policies.py RESULTS_DIR [--binary arena]
                                      [--reference auto] [--pair 0]

For every (backend, bucket, order, fixture) cell of a run_cuda_kick_arena.py
result directory it reports the relative L2 and maximum pointwise relative
differences of the accelerations and potentials of each policy of the chosen
binary against the reference policy of the same binary, bucket and order. This
is the numerical check that tiled sink processing (sink tiles smaller than the
bucket) and global force backing reproduce the whole-bucket forces.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_cuda_kick_arena import compare_forces  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('results', type=Path)
    parser.add_argument('--binary', default='arena')
    parser.add_argument('--reference', default='auto')
    parser.add_argument('--pair', type=int, default=0)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    runs = json.loads((args.results / 'runs.json').read_text())
    selected = {(run['backend'], run['bucket'], run.get('order', 7), run['policy']): run
                for run in runs if run['binary'] == args.binary and run['pair'] == args.pair}
    lines = ['| backend | bucket | order | policy | fixture | acceleration rel L2 | '
             'acceleration max pointwise | potential rel L2 | potential max pointwise |',
             '|---|---:|---:|---|---|---:|---:|---:|---:|']
    for key in sorted(selected):
        backend, bucket, order, policy = key
        if policy == args.reference:
            continue
        reference = selected.get((backend, bucket, order, args.reference))
        if reference is None:
            continue
        result = compare_forces(reference['force_csv'], selected[key]['force_csv'])
        for fixture, item in result.items():
            acceleration = item['acceleration']
            potential = item['potential']
            lines.append(f'| {backend} | {bucket} | {order} | {policy} | {fixture} | '
                         f'{acceleration["relative_l2"]:.3e} | '
                         f'{acceleration["max_pointwise_relative"]:.3e} | '
                         f'{potential["relative_l2"]:.3e} | '
                         f'{potential["max_pointwise_relative"]:.3e} |')
    text = '\n'.join(lines) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
