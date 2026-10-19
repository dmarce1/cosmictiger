#!/usr/bin/env python3
"""Sum executed warp instructions per device function from Nsight Compute reports.

Usage: attribute_ncu_functions.py LABEL=REPORT.ncu-rep [LABEL=REPORT.ncu-rep ...]

Each report is exported with `ncu --import REPORT --page source --csv
--print-source sass`, whose output is one section per function ("Kernel Name"
line, header line, one row per SASS instruction with an "Instructions Executed"
column). The sums are printed as a Markdown table, functions sorted by the
first report's count, with the difference against the first report for every
other one. Functions are keyed by the identifier before the parameter list
(template arguments kept, so instantiations stay distinct but a changed
signature does not split a function).
"""
import csv
import io
import subprocess
import sys

NCU = '/usr/local/cuda/bin/ncu'


def attribute(report):
    text = subprocess.run([NCU, '--import', report, '--page', 'source', '--csv',
                           '--print-source', 'sass'], capture_output=True, text=True,
                          check=True).stdout
    counts = {}
    name = None
    header = None
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        if row[0] == 'Kernel Name':
            name = row[1]
            header = None
            continue
        if header is None:
            header = row
            column = header.index('Instructions Executed')
            continue
        if len(row) > column and row[column].strip():
            key = short(name)
            counts[key] = counts.get(key, 0) + int(row[column].replace(',', ''))
    return counts


def short(name):
    depth = 0
    for index, char in enumerate(name):
        if char == '(' and depth == 0:
            name = name[:index]
            break
        depth += {'<': 1, '>': -1}.get(char, 0)
    return name.strip()


def main():
    labelled = [arg.split('=', 1) for arg in sys.argv[1:]]
    if not labelled or any(len(item) != 2 for item in labelled):
        sys.exit(__doc__)
    results = [(label, attribute(report)) for label, report in labelled]
    names = sorted({n for _, c in results for n in c}, key=lambda n: -results[0][1].get(n, 0))
    print('| device function | ' + ' | '.join(
        [results[0][0]] + [f'{label} | difference' for label, _ in results[1:]]) + ' |')
    print('|---|' + '---:|' * (1 + 2 * (len(results) - 1)))
    for name in names + ['total']:
        cells = []
        base = results[0][1].get(name, 0) if name != 'total' else sum(results[0][1].values())
        cells.append(f'{base:,}')
        for _, counts in results[1:]:
            value = counts.get(name, 0) if name != 'total' else sum(counts.values())
            cells.append(f'{value:,}')
            cells.append(f'{value - base:+,}')
        print(f'| `{name}` | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main()
