#!/usr/bin/env python3
"""Attribute the stall samples of an `ncu --page source --csv --print-source sass,cuda` export to code regions.

Usage: summarize_ncu_regions.py EXPORT.csv NAME=FILE:FIRST-LAST [NAME=FILE:FIRST-LAST ...]
                                [--library SUFFIX,SUFFIX,...] [--sass NAME]

The export lists every SASS instruction once per inlining level (the same
address appears under the caller's line, under simd_math.hpp's simdFma and
under cmath's fma), so per-line sums over-count the inlined arithmetic.
This script keeps one row per address, preferring the attribution to a
project file over a library frame, orders the rows by address, and assigns
each row to the region (a file suffix and an inclusive line range) of its
source line. Rows attributed to library frames (--library, default the
standard math and SIMD wrappers) inherit the region of the nearest preceding
project row, which is where their inlined code sits. It prints per region the
warp stall samples, executed warp instructions, average active threads,
static instruction count, the stall-reason mix and the opcode mix; with
--sass NAME it also lists that region's instructions in address order with
their samples and top stall reason.
"""
import argparse
import collections
import csv
from pathlib import Path

LIBRARY = ('cmath', 'simd_math.hpp', 'math.hpp', 'math_functions.hpp', 'containers.hpp',
           'tensor.hpp', 'ewald_indices.hpp', 'sm_30_intrinsics.hpp', 'sm_20_intrinsics.hpp',
           'helpers.h')


def parse_rows(path):
    """Yield (address, file, line, sass, samples, executed, thread_executed, stalls, function)."""
    current = function = header = None
    source_line = None
    with open(path, newline='') as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            if row[0] == 'File Path':
                current = row[1]
                continue
            if row[0] == 'Function Name':
                function = row[1]
                continue
            if row[0] == 'Line No':
                header = row
                continue
            if header is None:
                continue
            if row[0].isdigit():
                source_line = int(row[0])
                continue
            if row[0] == '' and len(row) > 2 and row[2].startswith('0x'):
                record = dict(zip(header, row))
                try:
                    samples = float(record['Warp Stall Sampling (All Samples)'])
                    executed = float(record['Instructions Executed'])
                    threads = float(record['Thread Instructions Executed'] or 0)
                except (KeyError, ValueError):
                    continue
                stalls = {key[6:]: float(value) for key, value in record.items()
                          if key.startswith('stall_') and not key.endswith(')')
                          and value not in ('', '-')}
                yield (int(row[2], 16), current, source_line, row[3].strip(), samples, executed,
                       threads, stalls, function)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('export', type=Path)
    parser.add_argument('regions', nargs='+', metavar='NAME=FILE:FIRST-LAST')
    parser.add_argument('--library', default=','.join(LIBRARY))
    parser.add_argument('--sass', default=None, metavar='NAME')
    args = parser.parse_args()
    regions = []
    for spec in args.regions:
        name, _, rest = spec.partition('=')
        file_suffix, _, lines = rest.rpartition(':')
        first, last = (int(value) for value in lines.split('-'))
        regions.append((name, file_suffix, first, last))
    library = tuple(item for item in args.library.split(',') if item)
    best = {}
    for row in parse_rows(args.export):
        is_library = row[1].endswith(library)
        rank = 1 if is_library else 0
        if row[0] not in best or rank < best[row[0]][0]:
            best[row[0]] = (rank, row)
    rows = sorted((value[1] for value in best.values()), key=lambda row: row[0])

    def region_of(file_path, line):
        for name, suffix, first, last in regions:
            if file_path.endswith(suffix) and first <= line <= last:
                return name
        return None

    assigned = []
    last_region = None
    totals = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for row in rows:
        region = region_of(row[1], row[2])
        if region is None:
            if last_region is not None and row[1].endswith(library):
                region = last_region
            else:
                region = 'other'
                last_region = None
        else:
            last_region = region
        entry = totals[region]
        entry[0] += row[4]
        entry[1] += row[5]
        entry[2] += row[6]
        entry[3] += 1
        assigned.append((row, region))
    total_samples = sum(entry[0] for entry in totals.values())
    total_executed = sum(entry[1] for entry in totals.values())
    print(f'{len(rows)} instructions, {total_samples:.0f} stall samples, '
          f'{total_executed:.0f} warp instructions executed')
    print('| region | samples | share | warp instructions | share | avg. threads | static instructions |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    names = [name for name, *_ in regions] + ['other']
    for name in names:
        if name not in totals:
            continue
        samples, executed, threads, static = totals[name]
        print(f'| {name} | {samples:.0f} | {100 * samples / total_samples:.1f}% | {executed:.0f} | '
              f'{100 * executed / total_executed:.1f}% | {threads / max(executed, 1):.1f} | {static} |')
    for name in names[:-1]:
        selected = [row for row, region in assigned if region == name]
        if not selected:
            continue
        stalls = collections.Counter()
        opcodes = collections.Counter()
        opcode_samples = collections.Counter()
        for row in selected:
            for key, value in row[7].items():
                stalls[key] += value
            parts = row[3].split()
            opcode = parts[1] if parts and parts[0].startswith('@') and len(parts) > 1 else (parts[0] if parts else '?')
            opcodes[opcode] += row[5]
            opcode_samples[opcode] += row[4]
        stall_total = sum(stalls.values())
        print(f'\n{name}: stalls ' + ', '.join(
            f'{key} {100 * value / max(stall_total, 1):.0f}%' for key, value in stalls.most_common(6)))
        print(f'{name}: opcodes (executed / samples) ' + ', '.join(
            f'{key} {value / 1e6:.2f}M / {opcode_samples[key]:.0f}' for key, value in opcodes.most_common(10)))
    if args.sass:
        print(f'\nSASS of {args.sass} (address, line, samples, executed, top stall, instruction):')
        for row, region in assigned:
            if region != args.sass:
                continue
            top = max(row[7].items(), key=lambda item: item[1]) if row[7] else ('', 0)
            print(f'{row[0]:#x} {Path(row[1]).name}:{row[2]} s={row[4]:.0f} e={row[5]:.0f} '
                  f'{top[0]} {top[1]:.0f}  {row[3]}')


if __name__ == '__main__':
    main()
