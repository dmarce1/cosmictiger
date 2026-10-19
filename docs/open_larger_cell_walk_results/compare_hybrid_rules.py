import re, sys, pathlib
old_root, new_root = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
def parse(path):
    rows = {}; t = None
    for line in path.read_text().splitlines():
        m = re.match(r"tree_kick: ([\d.e+-]+) s", line)
        if m: t = float(m.group(1)); continue
        m = re.match(r"walk theta ([\d.]+) openLargerCell \d: tests (\d+) ccDirect (\d+) ccEwald (\d+) cp (\d+) pc (\d+) ppLeafPairs (\d+) ppPairs (\d+)", line)
        if m: rows[round(float(m.group(1)), 3)] = dict(tests=int(m.group(2)), ccD=int(m.group(3)), ccE=int(m.group(4)), cp=int(m.group(5)), pc=int(m.group(6)), ppLeaf=int(m.group(7)), pp=int(m.group(8)), time=t)
    return rows
for ic in ('plummer', 'cosmo'):
    for rule in ('true',):
        a = parse(old_root / f'{ic}_open_{rule}' / 'sweep.log'); b = parse(new_root / f'{ic}_open_{rule}' / 'sweep.log')
        print(f"\n### {ic}, open-larger: radius-rule hybrids (R) versus cost-rule hybrids (C)\n")
        print("| theta | CP R | CP C | PC R | PC C | PP pairs R | PP pairs C | M2L direct R | C | tests R | C | kick s R | C |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for t in sorted(a):
            x, y = a[t], b[t]
            print(f"| {t:.2f} | {x['cp']} | {y['cp']} | {x['pc']} | {y['pc']} | {x['pp']} | {y['pp']} | {x['ccD']} | {y['ccD']} | {x['tests']} | {y['tests']} | {x['time']:.2f} | {y['time']:.2f} |")
        bad = [t for t in b if b[t]['cp'] != b[t]['pc']]
        print(f"\n{ic} cost rule: CP == PC at every theta: {not bad}")
