#!/usr/bin/env python3
"""Report how consistent the pairs of one run_cuda_kick_arena.py directory are.

Usage: check_kick_arena_pairs.py RESULTS_DIR [--max-spread RATIO]

For every (binary, policy, backend, bucket, order, fixture, potential) cell it
takes the ratio of the slowest to the fastest per-process median kick time
across pairs and prints the worst cell, the load average at the start and the
wall-clock range of the processes. With --max-spread it exits 1 when the worst
ratio exceeds RATIO, so a stage script can repeat a stage that other load on
the workstation contaminated (clean stages on this workstation stay below
1.1; contaminated ones reached 1.5-2.0).
"""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('results', type=Path)
    parser.add_argument('--max-spread', type=float, default=None)
    args = parser.parse_args()
    runs = json.loads((args.results / 'runs.json').read_text())
    metadata = json.loads((args.results / 'metadata.json').read_text())
    cells = {}
    for run in runs:
        for row in run['medians']:
            key = (run['binary'], run['policy'], run['backend'], run['bucket'], run['order'],
                   row['fixture'], row['potential'])
            cells.setdefault(key, []).append(row['kick_seconds'])
    worst, worst_key = 1.0, None
    for key, times in cells.items():
        if len(times) > 1 and max(times) / min(times) > worst:
            worst, worst_key = max(times) / min(times), key
    walls = [run['wall_seconds'] for run in runs]
    print(f'{args.results}: worst pair spread {worst:.3f} {worst_key}, load at start '
          f'{metadata["load_average"][0]:.2f}, process wall {min(walls):.1f}-{max(walls):.1f} s')
    if args.max_spread is not None and worst > args.max_spread:
        sys.exit(1)


if __name__ == '__main__':
    main()
