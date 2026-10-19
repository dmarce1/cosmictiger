import re, sys
src = sys.argv[1]
interesting = re.compile(r"\b(LDGSTS|ARRIVES|ATOMS|WARPSYNC|BAR\.|LDS|STS|NANOSLEEP|MEMBAR|DEPBAR|LDGDEPBAR|EXIT|CALL|RET|BRA|VOTE|LD\.E|ST\.E|LDG|STG|ERRBAR|YIELD|CCTL|ATOM|RED)\b")
cur = None; label = None
for ln in open(src, errors="replace"):
    ln = ln.rstrip("\n")
    m = re.match(r'\s*//## File "([^"]+)", line (\d+)', ln)
    if m:
        cur = (m.group(1).split("/")[-1], int(m.group(2))); continue
    m = re.match(r'^(\.L_x_\d+):', ln)
    if m:
        label = m.group(1); continue
    m = re.match(r'\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*;', ln)
    if m and cur:
        addr, ins = m.group(1), m.group(2)
        f, l = cur
        if interesting.search(ins):
            lab = (label + ":") if label else ""
            print(f"{addr:>6} {lab:<12} {f}:{l:<5} {ins}")
        label = None
