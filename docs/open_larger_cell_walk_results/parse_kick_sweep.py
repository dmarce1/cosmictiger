# Tabulate the GPU-free count sweeps (runs_kick/<ic>_open_<rule>/sweep.log) and
# merge the force errors and net-force residuals from the earlier force sweeps
# (runs/<ic>_open_<rule>/run.log), whose forces the counter change did not touch.
import re, sys, pathlib
kick_root = pathlib.Path(sys.argv[1]); force_root = pathlib.Path(sys.argv[2])
def parse_kick(path):
    rows = {}; theta = None
    for line in path.read_text().splitlines():
        m = re.match(r"tree_kick: ([\d.e+-]+) s", line)
        if m: pending_time = float(m.group(1)); continue
        m = re.match(r"walk theta ([\d.]+) openLargerCell \d: tests (\d+) ccDirect (\d+) ccEwald (\d+) cp (\d+) pc (\d+) ppLeafPairs (\d+) ppPairs (\d+) opened (\d+) deferred (\d+)", line)
        if m:
            theta = round(float(m.group(1)), 3)
            rows[theta] = dict(tests=int(m.group(2)), ccD=int(m.group(3)), ccE=int(m.group(4)), cp=int(m.group(5)), pc=int(m.group(6)), ppLeaf=int(m.group(7)), pp=int(m.group(8)), opened=int(m.group(9)), deferred=int(m.group(10)), time=pending_time)
    return rows
def parse_force(path):
    rows = {}
    for line in path.read_text().splitlines():
        m = re.match(r"walk theta ([\d.]+) openLargerCell \d: rank-local net force residual ([\d.e+-]+)", line)
        if m: rows.setdefault(round(float(m.group(1)), 3), {})['residual'] = float(m.group(2)); continue
        m = re.match(r"^([\d.]+e[+-]\d+) ([\d.]+e[+-]\d+) ([\d.]+e[+-]\d+)$", line)
        if m:
            t = round(float(m.group(1)), 3); rows.setdefault(t, {}); rows[t]['f50'] = float(m.group(2)); rows[t]['f99'] = float(m.group(3))
    return rows
for ic, title in (('plummer', 'Plummer model, radius 0.01'), ('cosmo', 'cosmological initial condition, z = 49')):
    a = parse_kick(kick_root / f'{ic}_open_false' / 'sweep.log'); b = parse_kick(kick_root / f'{ic}_open_true' / 'sweep.log')
    fa = parse_force(force_root / f'{ic}_open_false' / 'run.log'); fb = parse_force(force_root / f'{ic}_open_true' / 'run.log')
    print(f"\n### {title}: lockstep (L) versus open-larger (O)\n")
    print("| theta | M2L direct L | O | O/L | M2L Ewald L | O | tests L | O | PP pairs L | O | CP=PC L | O | deferred O | kick s L | O | net-force residual L | O | force error 50% L | O | 99% L | O |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in sorted(a):
        x, y = a[t], b[t]; p, q = fa.get(t, {}), fb.get(t, {})
        cp = f"{x['cp']}" if x['cp'] == x['pc'] else f"{x['cp']}/{x['pc']}"
        cq = f"{y['cp']}" if y['cp'] == y['pc'] else f"{y['cp']}/{y['pc']}"
        print(f"| {t:.2f} | {x['ccD']} | {y['ccD']} | {y['ccD']/x['ccD']:.2f} | {x['ccE']} | {y['ccE']} | {x['tests']} | {y['tests']} | {x['pp']} | {y['pp']} | {cp} | {cq} | {y['deferred']} | {x['time']:.2f} | {y['time']:.2f} | {p.get('residual', float('nan')):.1e} | {q.get('residual', float('nan')):.1e} | {p.get('f50', float('nan')):.2e} | {q.get('f50', float('nan')):.2e} | {p.get('f99', float('nan')):.2e} | {q.get('f99', float('nan')):.2e} |")
    for name, r in (('lockstep', a), ('open-larger', b)):
        bad = [t for t in r if r[t]['cp'] != r[t]['pc']]
        print(f"\n{ic} {name}: CP and PC counts equal at every theta: {not bad}" + (f" (unequal at {bad})" if bad else ""))
