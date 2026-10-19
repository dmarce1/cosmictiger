import re, sys, pathlib
root = pathlib.Path(sys.argv[1])
def parse(path):
    rows = {}
    counts = None
    for line in path.read_text().splitlines():
        m = re.match(r"walk theta ([\d.]+) openLargerCell (\d): tests (\d+) ccDirect (\d+) ccEwald (\d+) cp (\d+) pc (\d+) ppLeafPairs (\d+) ppPairs (\d+) opened (\d+) deferred (\d+)", line)
        if m:
            theta = float(m.group(1))
            counts = dict(tests=int(m.group(3)), ccD=int(m.group(4)), ccE=int(m.group(5)), cp=int(m.group(6)), pc=int(m.group(7)), ppLeaf=int(m.group(8)), pp=int(m.group(9)), opened=int(m.group(10)), deferred=int(m.group(11)))
            rows[theta] = counts
            continue
        m = re.match(r"walk theta ([\d.]+) openLargerCell \d: rank-local net force residual ([\d.e+-]+)", line)
        if m:
            rows[float(m.group(1))]['residual'] = float(m.group(2))
            continue
        m = re.match(r"^([\d.]+e[+-]\d+) ([\d.]+e[+-]\d+) ([\d.]+e[+-]\d+)$", line)
        if m:
            theta = round(float(m.group(1)), 3)
            if theta in rows:
                rows[theta]['f50'] = float(m.group(2)); rows[theta]['f99'] = float(m.group(3))
    return rows
for ic in ['plummer', 'cosmo']:
    a = parse(root / f'{ic}_open_false' / 'run.log')
    b = parse(root / f'{ic}_open_true' / 'run.log')
    print(f"\n### {ic}: lockstep (L) vs open-larger (O), 32768 particles, CPU walk\n")
    print("| theta | M2L direct L | M2L direct O | ratio | M2L Ewald L | O | tests L | O | PP pairs L | O | CP L | CP O | deferred O | residual L | residual O | force 50% L | O | force 99% L | O |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for theta in sorted(a):
        x, y = a[theta], b[theta]
        print(f"| {theta:.2f} | {x['ccD']} | {y['ccD']} | {y['ccD']/x['ccD']:.2f} | {x['ccE']} | {y['ccE']} | {x['tests']} | {y['tests']} | {x['pp']} | {y['pp']} | {x['cp']} | {y['cp']} | {y['deferred']} | {x['residual']:.1e} | {y['residual']:.1e} | {x.get('f50',float('nan')):.2e} | {y.get('f50',float('nan')):.2e} | {x.get('f99',float('nan')):.2e} | {y.get('f99',float('nan')):.2e} |")
    # symmetry check: cp == pc within each run
    for name, r in (('L', a), ('O', b)):
        bad = [t for t in r if r[t]['cp'] != r[t]['pc']]
        print(f"{ic} {name}: cp==pc at every theta: {not bad}")
