#!/usr/bin/env python3
"""Per-file and per-function totals of an `ncu --page source --csv --print-source sass,cuda` export.

Usage: summarize_ncu_source.py EXPORT.csv [FILE:NAME:FIRST-LAST ...]

Deduplicates SASS instructions across inlining frames, preferring project
attribution over library wrappers, then prints totals per file and,
for the optional FILE:NAME:FIRST-LAST ranges (FILE a path suffix, lines
inclusive), per named range, so the share of a routine in the kernel's time
can be read off without the full report.
"""
import sys

from summarize_ncu_regions import LIBRARY, parse_rows


def main():
    path = sys.argv[1]
    ranges = []
    for spec in sys.argv[2:]:
        file_suffix, name, lines = spec.rsplit(':', 2)
        first, last = (int(v) for v in lines.split('-'))
        ranges.append((file_suffix, name, first, last))
    per_file = {}
    per_range = {name: [0.0, 0.0] for _, name, _, _ in ranges}
    total = [0.0, 0.0]
    unique = {}
    for row in parse_rows(path):
        key = (row[8], row[0])  # Function and SASS address, independent of inline frame.
        rank = int(row[1].endswith(LIBRARY))
        if key not in unique or rank < unique[key][0]:
            unique[key] = (rank, row)
    for _, row in unique.values():
        _, current, line, _, samples, executed, *_ = row
        entry = per_file.setdefault(current, [0.0, 0.0])
        entry[0] += samples
        entry[1] += executed
        total[0] += samples
        total[1] += executed
        for file_suffix, name, first, last in ranges:
            if current.endswith(file_suffix) and first <= line <= last:
                per_range[name][0] += samples
                per_range[name][1] += executed
    print(f'total samples {total[0]:.0f}, warp instructions {total[1]:.0f}')
    print('| file | samples | share | warp instructions |')
    print('|---|---:|---:|---:|')
    for name, (samples, executed) in sorted(per_file.items(), key=lambda item: -item[1][0]):
        print(f'| {name.split("/")[-1]} | {samples:.0f} | {100 * samples / max(total[0], 1):.1f}% | {executed:.0f} |')
    if ranges:
        print()
        print('| range | samples | share | warp instructions |')
        print('|---|---:|---:|---:|')
        for name, (samples, executed) in per_range.items():
            print(f'| {name} | {samples:.0f} | {100 * samples / max(total[0], 1):.1f}% | {executed:.0f} |')


if __name__ == '__main__':
    main()
