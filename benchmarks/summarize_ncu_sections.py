#!/usr/bin/env python3
"""Per-function totals of an `ncu --page source --csv --print-source sass` export.

Usage: summarize_ncu_sections.py SOURCE.csv [...]

For every function section of the export (the kernel and each device function
it calls) it prints the SASS row count, the warp stall samples and the warp
instructions executed, so the share of each routine in the kernel's time can be
read off without the full report.
"""
import csv, sys
for path in sys.argv[1:]:
    print('==', path)
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = None; kernel = None; stats = {}
        for row in reader:
            if not row: continue
            if row[0] == 'Kernel Name':
                kernel = row[1]; header = None; stats.setdefault(kernel, [0, 0.0, 0.0]); continue
            if row[0] == 'Address':
                header = row; continue
            if header:
                d = dict(zip(header, row))
                s = stats[kernel]
                s[0] += 1
                try: s[1] += float(d['Warp Stall Sampling (All Samples)'])
                except: pass
                try: s[2] += float(d['Instructions Executed'])
                except: pass
        for k, v in stats.items():
            print(f'{v[0]:7d} rows samples={v[1]:10.0f} exec={v[2]:12.0f}  {k[:110]}')
