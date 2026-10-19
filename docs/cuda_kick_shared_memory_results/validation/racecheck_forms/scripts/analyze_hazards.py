#!/usr/bin/env python3
"""Tally compute-sanitizer racecheck hazard records (--racecheck-report all) and analysis reports."""
import re, sys, collections
path = sys.argv[1]
hz = re.compile(r"^========= (\w+): (?:\((?P<tag1>[^)]*)\) )?Potential (?P<kind>\w+) hazard detected(?: \((?P<tag2>[^)]*)\))? at __shared__ (?P<addr>0x[0-9a-f]+) in block \((?P<blk>[^)]*)\)")
acc = re.compile(r"^=========\s+(?P<op>Write|Read) Thread \((?P<thr>[^)]*)\) at \S+\+(?P<pc>0x[0-9a-f]+) in (?P<loc>\S+)")
rep = re.compile(r"^========= (\w+): Race reported between (?P<op1>\w+) access at \S+\+(?P<pc1>0x[0-9a-f]+) in (?P<loc1>\S+)")
rep2 = re.compile(r"^=========\s+and (?P<op2>\w+) access at \S+\+(?P<pc2>0x[0-9a-f]+) in (?P<loc2>\S+) \[(?P<n>\d+) hazards\]")
kinds = collections.Counter(); same = collections.Counter(); byread = collections.Counter(); bywrite = collections.Counter()
pairs = collections.Counter(); blocks = set(); reports = collections.Counter(); nrep = 0
cur = None; accs = []
def flush():
    global cur, accs
    if cur and len(accs) >= 2:
        w = [a for a in accs if a["op"] == "Write"]; r = [a for a in accs if a["op"] == "Read"]
        a, b = accs[0], accs[1]
        key = "same thread" if a["thr"] == b["thr"] else "different threads"
        same[(cur["kind"], key)] += 1
        pairs[(a["loc"], b["loc"])] += 1
        if r: byread[r[0]["loc"]] += 1
        if w: bywrite[w[0]["loc"]] += 1
    cur = None; accs = []
for ln in open(path, errors="replace"):
    m = hz.match(ln)
    if m:
        flush(); cur = m.groupdict(); cur["tag"] = cur["tag1"] or cur["tag2"]; kinds[(cur["kind"], cur["tag"])] += 1; blocks.add(cur["blk"]); continue
    m = acc.match(ln)
    if m and cur is not None:
        accs.append(m.groupdict()); continue
    m = rep.match(ln)
    if m:
        last = m.groupdict(); continue
    m = rep2.match(ln)
    if m and last:
        nrep += 1; reports[(last["loc1"], m.group("loc2"), int(m.group("n")))] += 1; last = None; continue
    if ln.startswith("========= RACECHECK SUMMARY"): print(ln.strip())
flush()
print("hazard records:", sum(kinds.values()), dict(kinds))
print("same vs different thread:", dict(same))
print("blocks with hazards:", len(blocks))
print("by read location:", dict(byread.most_common()))
print("by write location:", dict(bywrite.most_common()))
print("analysis reports:", nrep)
for (l1, l2, n), c in sorted(reports.items()): print(f"   {c:3d} x  {l1} vs {l2} [{n} hazards]")
