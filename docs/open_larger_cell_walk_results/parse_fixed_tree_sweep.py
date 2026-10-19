# Tabulate a fixed-tree sweep (run_fixed_tree_sweep.sh) for both walk rules and
# read off the open-larger work at the lockstep rule's median force error.
# Usage: parse_fixed_tree_sweep.py ROOT ic tree_theta
import math, re, sys, pathlib

root, ic, tree_theta = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]

def parse(path):
    rows, time = {}, None
    for line in path.read_text().splitlines():
        m = re.match(r"tree_kick: ([\d.e+-]+) s", line)
        if m:
            time = float(m.group(1)); continue
        m = re.match(r"walk theta ([\d.]+) openLargerCell \d: tests (\d+) ccDirect (\d+) ccEwald (\d+) "
                     r"cp (\d+) pc (\d+) ppLeafPairs (\d+) ppPairs (\d+) opened (\d+) deferred (\d+)", line)
        if m:
            t = round(float(m.group(1)), 3)
            rows[t] = dict(theta=t, tests=int(m.group(2)), ccD=int(m.group(3)), ccE=int(m.group(4)),
                           cp=int(m.group(5)), pc=int(m.group(6)), pp=int(m.group(8)),
                           deferred=int(m.group(10)), time=time)
            continue
        m = re.match(r"walk theta ([\d.]+) openLargerCell \d: rank-local net force residual ([\d.e+-]+)", line)
        if m:
            rows[round(float(m.group(1)), 3)]['residual'] = float(m.group(2)); continue
        m = re.match(r"^([\d.]+e[+-]\d+) ([\d.]+e[+-]\d+) ([\d.]+e[+-]\d+)$", line)
        if m:
            t = round(float(m.group(1)), 3)
            rows[t]['f50'] = float(m.group(2)); rows[t]['f99'] = float(m.group(3))
    return [rows[t] for t in sorted(rows)]

L = parse(root / f"{ic}_tree{tree_theta}_open_false" / "run.log")
O = parse(root / f"{ic}_tree{tree_theta}_open_true" / "run.log")
assert [r['theta'] for r in L] == [r['theta'] for r in O], "the two rules swept different angles"

print(f"\n### {ic}, tree built at theta {tree_theta}: lockstep (L) versus open-larger (O), kick theta swept\n")
print("| kick theta | M2L direct L | O | M2L Ewald L | O | PP pairs L | O | CP=PC L | O | kick s L | O | residual L | O | force error 50% L | O | 99% L | O |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for x, y in zip(L, O):
    cp = f"{x['cp']}" if x['cp'] == x['pc'] else f"{x['cp']}/{x['pc']}"
    cq = f"{y['cp']}" if y['cp'] == y['pc'] else f"{y['cp']}/{y['pc']}"
    print(f"| {x['theta']:.3f} | {x['ccD']} | {y['ccD']} | {x['ccE']} | {y['ccE']} | {x['pp']} | {y['pp']} | {cp} | {cq} "
          f"| {x['time']:.2f} | {y['time']:.2f} | {x['residual']:.1e} | {y['residual']:.1e} "
          f"| {x['f50']:.2e} | {y['f50']:.2e} | {x['f99']:.2e} | {y['f99']:.2e} |")

# Open-larger work at the lockstep median error: find the open-larger sweep
# interval bracketing the lockstep error, place the angle by log-linear
# interpolation of the error, and interpolate the counts linearly in the angle.
def matched(err, rows):
    for a, b in zip(rows, rows[1:]):
        lo, hi = sorted((a['f50'], b['f50']))
        if lo <= err <= hi and lo < hi:
            w = (math.log(err) - math.log(a['f50'])) / (math.log(b['f50']) - math.log(a['f50']))
            out = {k: a[k] + w * (b[k] - a[k]) for k in ('theta', 'ccD', 'ccE', 'pp', 'time')}
            return out
    return None

print(f"\n### {ic}, tree built at theta {tree_theta}: open-larger work at the lockstep median error\n")
print("| lockstep theta | median error | open-larger theta at that error | M2L direct O/L | M2L Ewald O/L | M2L total O/L | PP pairs O/L | kick time O/L |")
print("|---|---|---|---|---|---|---|---|")
ratios = {k: [] for k in ('offset', 'ccD', 'ccE', 'total', 'pp', 'time')}
for x in L:
    m = matched(x['f50'], O)
    if m is None:
        print(f"| {x['theta']:.3f} | {x['f50']:.2e} | outside the open-larger sweep | | | | | |")
        continue
    total = (m['ccD'] + m['ccE']) / (x['ccD'] + x['ccE'])
    print(f"| {x['theta']:.3f} | {x['f50']:.2e} | {m['theta']:.3f} | {m['ccD'] / x['ccD']:.2f} | {m['ccE'] / x['ccE']:.2f} "
          f"| {total:.2f} | {m['pp'] / x['pp']:.2f} | {m['time'] / x['time']:.2f} |")
    for k, v in (('offset', x['theta'] - m['theta']), ('ccD', m['ccD'] / x['ccD']), ('ccE', m['ccE'] / x['ccE']),
                 ('total', total), ('pp', m['pp'] / x['pp']), ('time', m['time'] / x['time'])):
        ratios[k].append(v)

def summary(v):
    v = sorted(v)
    return f"{v[len(v) // 2]:.2f} ({v[0]:.2f} to {v[-1]:.2f})"

print(f"\n{ic}, tree {tree_theta}, over the {len(ratios['ccD'])} matched angles, median (range): "
      f"angle offset {summary(ratios['offset'])}, M2L direct O/L {summary(ratios['ccD'])}, "
      f"M2L Ewald O/L {summary(ratios['ccE'])}, M2L total O/L {summary(ratios['total'])}, "
      f"PP pairs O/L {summary(ratios['pp'])}, kick time O/L {summary(ratios['time'])}")
