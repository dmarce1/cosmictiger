#!/usr/bin/env python3
"""Instruction forms of selected source lines in an nvdisasm -g listing.

sass_regions.py SASS_TXT loop FILE:L1-L2[,L3-L4...]
    For the instructions attributed to those source lines, find the innermost
    loop (a backward BRA to a label) enclosing each and print every such loop
    body from its label to the back-edge branch: instruction total, the
    floating-point / memory / integer-and-control split, the opcode
    histogram, and the memory instructions attributed to the given lines.
sass_regions.py SASS_TXT site FILE:L1-L2[,L3-L4...]
    Print the memory instructions attributed to those source lines (opcode
    histogram per line, then the instructions in address order).
"""
import re
import sys
from collections import Counter, OrderedDict

LOC = re.compile(r'\s*//## File "([^"]+)", line (\d+)')
LABEL = re.compile(r'^(\.L_x_\d+):')
INS = re.compile(r'\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*;')
MEM = re.compile(r'^(LDGSTS|LDS|STS|LDL|STL|LDG|STG|LD|ST|ATOMS|ATOM|RED)(\.|$)')
FP = re.compile(r'^(F|MUFU|I2F|F2I|HFMA|HADD|HMUL)')


def parse(path):
    records = []
    cur = None
    label = None
    labels = {}
    for ln in open(path, errors="replace"):
        ln = ln.rstrip("\n")
        m = LOC.match(ln)
        if m:
            cur = (m.group(1).split("/")[-1], int(m.group(2)))
            continue
        m = LABEL.match(ln)
        if m:
            label = m.group(1)
            continue
        m = INS.match(ln)
        if m:
            addr = int(m.group(1), 16)
            text = m.group(2)
            body = re.sub(r'^@!?U?P[T0-9]\s+', '', text)
            opcode = body.split()[0] if body.split() else body
            if label:
                labels[label] = len(records)
            records.append(dict(addr=addr, label=label, loc=cur, opcode=opcode, text=text))
            label = None
    return records, labels


def parse_lines(spec):
    file, ranges = spec.split(":")
    lines = set()
    for r in ranges.split(","):
        a, _, b = r.partition("-")
        a = int(a)
        b = int(b) if b else a
        lines.update(range(a, b + 1))
    return file, lines


def classify(opcode):
    if MEM.match(opcode):
        return "memory"
    if FP.match(opcode):
        return "floating point"
    return "integer and control"


def targets(records, file, lines):
    return [i for i, r in enumerate(records)
            if r["loc"] and r["loc"][0] == file and r["loc"][1] in lines]


def loops(records, labels):
    out = []
    for i, r in enumerate(records):
        if r["opcode"] == "BRA":
            m = re.search(r'`\((\.L_x_\d+)\)', r["text"])
            if m and m.group(1) in labels and labels[m.group(1)] < i:
                out.append((labels[m.group(1)], i))
    return out


def report_loop(records, lo, hi, file, lines):
    body = records[lo:hi + 1]
    split = Counter(classify(r["opcode"]) for r in body)
    hist = Counter(r["opcode"] for r in body)
    print(f"== loop [0x{records[lo]['addr']:x},0x{records[hi]['addr']:x}] "
          f"({records[lo]['label']}): {len(body)} instructions "
          f"({split['floating point']} floating point, {split['memory']} memory, "
          f"{split['integer and control']} integer and control)")
    print("    " + ", ".join(f"{k} {v}" for k, v in hist.most_common()))
    mem = [r for r in body if classify(r["opcode"]) == "memory"]
    by_line = OrderedDict()
    for r in mem:
        key = f"{r['loc'][0]}:{r['loc'][1]}" if r["loc"] else "?"
        by_line.setdefault(key, Counter())[r["opcode"]] += 1
    print("    memory instructions by source line: " + "; ".join(
        f"{k}: " + ", ".join(f"{o} {n}" for o, n in c.items()) for k, c in by_line.items()))
    sel = [r for r in body if r["loc"] and r["loc"][0] == file and r["loc"][1] in lines]
    print(f"    instructions attributed to the given lines ({len(sel)}):")
    for r in sel:
        print(f"      {r['addr']:6x} {r['loc'][0]}:{r['loc'][1]:<5} {r['text']}")


def main():
    records, labels = parse(sys.argv[1])
    mode = sys.argv[2]
    file, lines = parse_lines(sys.argv[3])
    idx = targets(records, file, lines)
    if mode == "loop":
        all_loops = loops(records, labels)
        chosen = OrderedDict()
        for i in idx:
            enclosing = [(lo, hi) for lo, hi in all_loops if lo <= i <= hi]
            if not enclosing:
                continue
            lo, hi = min(enclosing, key=lambda p: p[1] - p[0])
            chosen.setdefault((lo, hi), []).append(i)
        print(f"{len(idx)} instructions attributed to {sys.argv[3]}; "
              f"{len(chosen)} innermost loops enclose them")
        for (lo, hi), members in chosen.items():
            print(f"-- {len(members)} of them in this loop")
            report_loop(records, lo, hi, file, lines)
    elif mode == "site":
        sel = [records[i] for i in idx]
        mem = [r for r in sel if classify(r["opcode"]) == "memory"]
        by_line = OrderedDict()
        for r in mem:
            by_line.setdefault(r["loc"][1], Counter())[r["opcode"]] += 1
        print(f"{len(sel)} instructions attributed to {sys.argv[3]}, {len(mem)} memory")
        for line, c in sorted(by_line.items()):
            print(f"    {file}:{line}: " + ", ".join(f"{o} {n}" for o, n in c.items()))
        for r in mem:
            print(f"      {r['addr']:6x} {r['loc'][0]}:{r['loc'][1]:<5} {r['text']}")
    else:
        sys.exit("mode is loop or site")


if __name__ == "__main__":
    main()
