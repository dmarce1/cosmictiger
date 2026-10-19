#!/usr/bin/env python3
"""Print selected Nsight Compute metrics of two `ncu --page raw --csv` exports side by side.

Usage: compare_ncu_raw.py BASELINE.csv CANDIDATE.csv [substring ...]

Without substrings a fixed set of kernel-level metrics (duration, issued and
executed instructions, occupancy, local and shared memory traffic, stall
reasons) is printed; with substrings every metric whose name contains one of
them. Values are taken from the first kernel row of each export.
"""
import csv
import sys

DEFAULT = ['gpu__time_duration.sum', 'sm__cycles_elapsed.avg', 'smsp__inst_executed.sum',
           'smsp__inst_issued.sum', 'sm__warps_active.avg.pct_of_peak_sustained_active',
           'launch__registers_per_thread', 'launch__occupancy_limit', 'launch__shared_mem_per_block',
           'local_op_ld.sum', 'local_op_st.sum', 'mem_shared_op_ld.sum', 'mem_shared_op_st.sum',
           'data_bank_conflicts_pipe_lsu_mem_shared', 'lts__t_bytes.sum', 'dram__bytes.sum',
           'issue_stalled', 'sm__inst_executed_pipe_', 'smsp__thread_inst_executed_per_inst_executed.ratio',
           'smsp__sass_average_branch_targets_threads_uniform']


def load(path):
    with open(path, newline='') as handle:
        rows = list(csv.reader(handle))
    names, units = rows[0], rows[1]
    data = next(r for r in rows[2:] if r and r[0].strip().isdigit())
    return {name: (unit, value) for name, unit, value in zip(names, units, data)}


def main():
    first, second = load(sys.argv[1]), load(sys.argv[2])
    patterns = sys.argv[3:] or DEFAULT
    print('| metric | unit | baseline | candidate | ratio |')
    print('|---|---|---:|---:|---:|')
    for name in first:
        if not any(p in name for p in patterns):
            continue
        unit, a = first[name]
        b = second.get(name, ('', ''))[1]
        try:
            x, y = float(a.replace(',', '')), float(b.replace(',', ''))
            ratio = f'{y / x:.3f}' if x else ''
        except ValueError:
            ratio = ''
        if a in ('', '0', '0.00') and b in ('', '0', '0.00'):
            continue
        print(f'| {name} | {unit} | {a} | {b} | {ratio} |')


if __name__ == '__main__':
    main()
