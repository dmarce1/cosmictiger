#!/usr/bin/env python3
"""Compare the force exports of two binaries of a run_cuda_kick_arena.py run.

Usage: compare_kick_arena_binaries.py RESULTS_DIR REFERENCE_LABEL CANDIDATE_LABEL
                                      [--policy auto] [--pair 0] [--output FILE]

For every (backend, bucket, order, fixture) cell it reports the relative L2
and maximum pointwise relative differences of the candidate binary's
accelerations and potentials against the reference binary's, same policy and
pair. Zero means the exports are bit-identical.
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
    parser.add_argument('reference')
    parser.add_argument('candidate')
    parser.add_argument('--policy', default='auto')
    parser.add_argument('--pair', type=int, default=0)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    runs = json.loads((args.results / 'runs.json').read_text())
    selected = {(run['binary'], run['backend'], run['bucket'], run.get('order', 7)): run
                for run in runs if run['policy'] == args.policy and run['pair'] == args.pair}
    lines = ['| reference | candidate | backend | bucket | order | fixture | acceleration rel L2 | '
             'acceleration max pointwise | potential rel L2 | potential max pointwise |',
             '|---|---|---|---:|---:|---|---:|---:|---:|---:|']
    for key in sorted(selected):
        binary, backend, bucket, order = key
        if binary != args.candidate:
            continue
        reference = selected.get((args.reference, backend, bucket, order))
        if reference is None:
            continue
        result = compare_forces(reference['force_csv'], selected[key]['force_csv'])
        for fixture, item in result.items():
            acceleration = item['acceleration']
            potential = item['potential']
            lines.append(f'| {args.reference} | {args.candidate} | {backend} | {bucket} | {order} | '
                         f'{fixture} | {acceleration["relative_l2"]:.3e} | '
                         f'{acceleration["max_pointwise_relative"]:.3e} | '
                         f'{potential["relative_l2"]:.3e} | '
                         f'{potential["max_pointwise_relative"]:.3e} |')
    text = '\n'.join(lines) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
