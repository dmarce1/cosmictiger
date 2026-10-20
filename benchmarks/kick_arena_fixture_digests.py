#!/usr/bin/env python3
"""Digest each fixture's rows of a kick arena stage's force exports.

Usage: kick_arena_fixture_digests.py STAGE [--output fixture_digests.md]

STAGE is a run_cuda_kick_arena.py output directory. The driver records one
sha256 per process over stf_walk_forces.csv, which holds all three fixtures,
so two divisions that agree on the lattice but not on the adaptive fixture get
different digests. This script hashes the rows of each fixture separately
(the first column names the fixture), writes STAGE/fixture_digests.json with
one record per run, and prints, per fixture, which policies produced
identical rows (the same letter) in every pair. Run it after the driver has
finished and before the 900 MB exports are deleted.
"""
import argparse
import hashlib
import json
from pathlib import Path


def fixture_digests(path):
    digests = {}
    with open(path, 'rb') as stream:
        stream.readline()
        for line in stream:
            fixture = line[:line.index(b',')].decode()
            digests.setdefault(fixture, hashlib.sha256()).update(line)
    return {fixture: digest.hexdigest() for fixture, digest in digests.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    runs = json.loads((args.stage / 'runs.json').read_text())
    records = []
    for run in runs:
        export = Path(run['force_csv'])
        records.append(dict(binary=run['binary'], policy=run['policy'], pair=run['pair'],
                            bucket=run['bucket'], order=run['order'],
                            export_sha256=run['force_csv_sha256'],
                            fixtures=fixture_digests(export)))
    (args.stage / 'fixture_digests.json').write_text(json.dumps(records, indent=2) + '\n')
    fixtures = sorted({fixture for record in records for fixture in record['fixtures']})
    policies = list(dict.fromkeys(record['policy'] for record in records))
    lines = ['| fixture | ' + ' | '.join(policies) + ' |',
             '|---|' + '---:|' * len(policies)]
    for fixture in fixtures:
        letters = {}
        cells = []
        for policy in policies:
            digests = {record['fixtures'][fixture] for record in records
                       if record['policy'] == policy}
            if len(digests) != 1:
                cells.append(f'{len(digests)} distinct')
                continue
            digest = digests.pop()
            letters.setdefault(digest, chr(ord('A') + len(letters)))
            cells.append(f'{letters[digest]} ({digest[:8]})')
        lines.append(f'| {fixture} | ' + ' | '.join(cells) + ' |')
    text = '\n'.join(lines) + '\n'
    print(text, end='')
    if args.output:
        args.output.write_text(text)


if __name__ == '__main__':
    main()
