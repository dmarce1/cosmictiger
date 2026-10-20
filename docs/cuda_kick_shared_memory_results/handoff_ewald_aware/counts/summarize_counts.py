#!/usr/bin/env python3
"""Tabulate the handoff counts of the instrumented binary (instrumentation.patch).

Usage: summarize_counts.py STAGE [--output counts.md]

STAGE is a run_cuda_kick_arena.py output directory of the *_counts binary.
Every kick prints KICK_HANDOFF_COUNTS from KickWorkspace::toGpu (workitems,
workitems with Ewald checks, Ewald and direct checks handed over, particles,
largest workitem, log2 size histogram) and then KICK_CPU_COUNTS from the STF
walk (fixture, potential, sample, and the CPU walk's Ewald cell-cell, direct
cell-cell, cell-particle and particle-cell counts). The counts do not depend
on the sample; the table gives the forces-only kick of the last sample.
"""
import argparse
import json
from pathlib import Path


def parse(text):
    rows = {}
    handoff = None
    for line in text.splitlines():
        if line.startswith('KICK_HANDOFF_COUNTS,'):
            fields = line.split(',')
            handoff = dict(workitems=int(fields[1]), ewald_items=int(fields[2]),
                           ewald_checks=int(fields[3]), direct_checks=int(fields[4]),
                           particles=int(fields[5]), max_particles=int(fields[6]),
                           histogram=fields[7].strip(';'))
        elif line.startswith('KICK_CPU_COUNTS,'):
            fields = line.split(',')
            row = dict(handoff or {}, cpu_ewald_cc=int(fields[4]), cpu_direct_cc=int(fields[5]),
                       cpu_cp=int(fields[6]), cpu_pc=int(fields[7]))
            rows[(fields[1], int(fields[2]))] = row
            handoff = None
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    runs = json.loads((args.stage / 'runs.json').read_text())
    records = []
    for run in runs:
        rows = parse((Path(run['cwd']) / 'stdout.log').read_text())
        for (fixture, potential), row in rows.items():
            records.append(dict(policy=run['policy'], fixture=fixture, potential=potential, **row))
    (args.stage / 'counts.json').write_text(json.dumps(records, indent=2) + '\n')
    lines = ['| fixture | policy | workitems | with Ewald checks | Ewald checks handed over | '
             'direct checks handed over | largest workitem | CPU nodes (workitems - 1) | CPU Ewald CC | '
             'CPU direct CC | workitem sizes (log2 bin: count) |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for fixture in ('lattice', 'clustered', 'adaptive'):
        for record in records:
            if record['fixture'] != fixture or record['potential']:
                continue
            lines.append(f'| {fixture} | {record["policy"]} | {record["workitems"]:,} | '
                         f'{record["ewald_items"]:,} | {record["ewald_checks"]:,} | '
                         f'{record["direct_checks"]:,} | {record["max_particles"]:,} | '
                         f'{record["workitems"] - 1:,} | {record["cpu_ewald_cc"]:,} | '
                         f'{record["cpu_direct_cc"]:,} | {record["histogram"]} |')
    text = '\n'.join(lines) + '\n'
    print(text, end='')
    if args.output:
        args.output.write_text(text)


if __name__ == '__main__':
    main()
