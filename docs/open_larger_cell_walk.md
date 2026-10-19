# Open-larger cell opening in the CPU tree walk

A Dehnen-style opening rule for the FMM tree walk: when a cell pair fails the
acceptance test, open the cell with the larger radius instead of opening both.
The rule is behind `gravity.openLargerCell`, default `true`, on both the CPU
and the CUDA walk; `false` restores the lockstep walk for comparison. The kick
stamps the rule onto its parameters, so the CPU walk, the remote kick action
and the CUDA kernel apply the same rule and the accepted pair set stays
reciprocal across a CPU/CUDA handoff.

The hybrid choice for a leaf pair that fails the cell-cell test is a flop
comparison rather than a radius rule: see "Hybrids by cost" below. Source
cells without particles never enter the walk's lists: see "Empty source
cells". The two rules are compared at equal force error on one tree in
"Equal accuracy on one tree" under Results.

Branch: `order_aware_cuda_shared_memory`. Results from 2026-10-19 on the
24-core workstation, CPU walk with 8 HPX threads, FMM order 7, 32768 particles
for the force sweeps.

## The walk today

Each sink node walks its own checklist of source nodes. The acceptance test
is symmetric in the two cells: it compares the centre distance with the sum of
the two radii divided by theta, plus the support-separation terms in
`cosmictiger/fmm/fmm_acceptance.hpp`. An internal sink makes one pass over its
list, opens every rejected internal source, and hands the opened children plus
every rejected source leaf down to both of its children. Only a leaf sink keeps
iterating. Cell-cell pairs therefore always meet at equal depth, except that a
rejected source leaf is already passed down unopened.

The accepted pair set is reciprocal: from either cell's walk the same unordered
pairs are accepted, because the test is a symmetric function of the pair and the
opening decision is deterministic. Each pair is then evaluated twice, once into
each local expansion, from the same Green tensor at plus and minus the
separation. The generated M2L keeps exactly the terms with source degree plus
local degree at most p (`tools/generate_stf_m2l.py`, the `rank` loop), and that
set is closed under swapping the two roles, so the total force cell A exerts
on cell B is the exact negative of the reverse up to roundoff, and the pair
potential energy agrees from both sides. The hybrid cell-particle and
particle-cell selections mirror each other the same way.

## The rule

The lockstep descent is not what protects momentum; the symmetric, deterministic
pair decision is. Any rule that depends only on the unordered pair keeps it.
For a rejected pair whose source is not a leaf, `detail::kickOpenOrDefer` in
`cosmictiger/gravity/kick_walk.hpp` now does one of three things:

- Source larger than the sink, or the sink is a leaf: open the source and retest
  its children at the current sink level. Internal sinks now iterate their list
  like leaf sinks do.
- Sink larger than the source: hand the source down unopened to the sink's
  children. This generalises the existing leaf path.
- Equal radii, both internal: open the source and hand its children down, which
  is the lockstep behaviour.

The tree bisects each box at the midpoint of its longest axis, so same-depth
cells share a box size while their radii come from the enclosing sphere of
their particles. In clustered regions those radii differ by large factors, and
the test only cares about their sum, so opening the smaller cell doubles the
pair count for almost no gain.

The CUDA walk in `cosmictiger/gravity/kick_cuda_kernel.hpp` applies the same rule with
a device twin of the helper, `cudaOpenOrDefer`, a `bothlist` for the
equal-radius case whose children go to the sink's children, and the per-level
loop iterating for internal sinks when the rule is on.

## Hybrids by cost

A cell-particle or particle-cell interaction stands in for direct summation
of a leaf pair, and whichever side is evaluated explicitly, both walks have to
pick the same pairing. The selection in `cosmictiger/fmm/fmm_acceptance.hpp` now
compares flop counts whose only runtime inputs are the two particle counts;
the per-order costs come from the generated kernels through
`FmmHybridCost<order>` in `cosmictiger/fmm/fmm_kernels.hpp`:

- direct summation: sinkCount times sourceCount times 38 flops, the Newtonian
  pair of the direct kernel. Hybrids are only considered for support-separated
  pairs, which never take the softened branch, so that is the right cost;
- a hybrid: one M2P on its own side and one P2L on the other side for every
  explicitly evaluated particle, charged as the mean of the two per side.

Among the valid choices the cheapest wins, so the leaf with fewer particles is
the one evaluated explicitly, and small leaves fall back to direct summation.
Equal counts keep the smaller-radius cell as the cloud and equal radii fall
back to direct summation, as before. A leaf against an internal cell is not
costed: its alternative is opening the cell, not direct summation. The CUDA
walk uses the same constants, so the decision is identical on both backends.
The reciprocity test in `tests/fmm_acceptance.cpp` covers the cost rule, and
the walk fixtures whose expected hybrid changed under it were updated.

Effect of the cost rule on the open-larger walk, from the count sweep
(`hybrid_rule_comparison.md`): on the Plummer model it moves 1 to 10 percent
of the hybrids to direct summation and changes the particle pairs by less than
0.1 percent. On the cosmological model at theta 0.2 it moves 24 percent of the
hybrids to direct summation, whose pairs grow from 4.7 to 12.2 million, since
the Ewald criterion leaves that tree with many small leaves; above theta 0.6
nothing changes. On the open-larger walk kick times move by at most 8 percent
either way, within the timing noise. On the lockstep walk at theta 0.2 and
0.25 the rule sends nearly all hybrids to direct summation (2.17 million to
17 thousand at theta 0.2, with particle pairs rising from 4.7 to 36 million),
which halves the median force error there but costs 4 to 15 percent of kick
time. So the flop comparison is close to balanced on this hardware with a
slight tilt toward direct summation: the 38-flop Newtonian pair of the SIMD
kernel is cheaper per flop than the hybrid operators assume, and an effective
pair cost somewhat above 38 would move the break-even toward the hybrids.
The regression fixture's hybrid counts fell from 300 to 237 (lockstep) and
358 to 327 (open-larger) for the same reason, with the net-force residual
unchanged at roundoff. CPU and CUDA forces agree to 3e-7 of the rms force
under both rules.

## Counters

`cosmictiger/gravity/kick_interaction_counts.hpp` adds rank-local counters filled by
the CPU walk: acceptance tests, direct and Ewald M2L, cell-particle,
particle-cell, particle-particle leaf pairs and particle pairs, source openings
and deferred sources. Sources without particles never enter the lists (see
"Empty source cells"), so they are neither tested nor counted and the cell
counters are plain list sizes. `run.test=force` prints the counters and the
rank-local net force residual, the norm of the summed force divided by the sum
of force norms, after every kick; `run.test=kick` prints the counters and the
kick time at `gravity.openingAngle`.

## Empty source cells

The Ewald leaf criterion splits a sparse box down to single particles beside
empty siblings, and the tree keeps those empty cells as leaves of radius zero.
The walk used to treat them like any other source: an opened cell pushed both
children onto the retry or deferred list, so an empty child was tested by the
sink, handed down to both sink children when it failed, and evaluated as a
zero-mass M2L wherever it was accepted. The CUDA walk skips a source with no
particles at the acceptance test. The CPU walk now keeps them out of the lists
instead: `detail::kickPushOccupied` in `cosmictiger/gravity/kick_walk.hpp` drops a
child with no particles when `kickOpenOrDefer` opens its parent, in both the
Ewald and the direct pass. The lists start from the root and grow only there,
so no list ever holds an empty cell. The accepted pair set for occupied cells
is unchanged; the only effect on the forces is roundoff, because removing a
zero entry from an M2L list shifts the sources behind it into other SIMD lanes
of `cpuGravityCc` and changes the order of the final lane sum.

Walk regression fixture (`fmm_walk_cpu_ctest_empty_skip.log` against the
earlier `fmm_walk_cpu_cuda_ctest.log`): acceptance tests fell from 32116 to
31345 for the lockstep rule and from 31472 to 31207 for the open-larger rule;
every M2L, hybrid, pair, opening and deferral count is identical; the
net-force residuals moved from 3.924e-9 to 3.923e-9 and from 1.774e-9 to
1.764e-9, and the rms force difference between the rules stayed at 7.665e-4.

Plummer model, 32768 particles, 5 HPX threads, `run.test=kick`, the walk
before the change (`kick_empty_before_*.log`, the binary of the previous
commit) and after it (`kick_empty_after_*.log`), run back to back:

| theta | rule | tests before | after | change | M2L direct | M2L Ewald | PP pairs | kick s before | after |
|---|---|---|---|---|---|---|---|---|---|
| 0.2 | lockstep | 505012 | 429894 | -14.9% | 62534 | 31292 | 591029358 | 0.34 | 0.33 |
| 0.2 | open-larger | 516366 | 501650 | -2.9% | 48058 | 27694 | 590659954 | 0.28 | 0.29 |
| 0.8 | lockstep | 52358 | 51614 | -1.4% | 15974 | 2398 | 67435294 | 0.056 | 0.053 |
| 0.8 | open-larger | 57960 | 57598 | -0.6% | 11608 | 1846 | 66473574 | 0.044 | 0.047 |

The M2L, hybrid and pair counts are the same before and after because the
counters already excluded empty sources; what the old walk spent on them shows
only in the acceptance tests, 75118 fewer at theta 0.2 with the lockstep rule,
and in the earlier count under Observations of 103892 direct M2L list entries
with empty cells against 62534 without, that is 41358 zero-mass M2L
evaluations at that angle. The lockstep walk tested many more empty cells than
the open-larger walk because it hands every opened child down to both sink
children, and an empty leaf that failed there was handed down again by every
sink below until some ancestor accepted it, whereas the open-larger walk
retests an opened child once, at the sink's own level. The kick time is within
the timing noise at both angles: on this model the kick is dominated by the
particle pairs, 591 million at theta 0.2 and 67 million at theta 0.8, and the
M2L evaluations the walk no longer does were cheap zero-mass ones.

## Reproduce

The walk regression (`ctest -R fmm_walk_cpu`) builds a Plummer clump beside a
uniform background, runs both rules on the same tree and checks that both net
force residuals are at roundoff, that the forces agree to the expansion error,
and that the new rule actually defers sources. The force sweeps:

```bash
docs/open_larger_cell_walk_results/run_force_compare.sh plummer cosmo
```

runs `run.test=force` with `--compute.cuda=false` for both rules on the Plummer
model and on the cosmological initial condition; its analytic comparison uses
the GPU for the direct reference sums. The count sweep

```bash
docs/open_larger_cell_walk_results/run_count_sweep.sh plummer cosmo
```

runs `run.test=kick` per opening angle without touching the GPU, and
`parse_kick_sweep.py` merges its counters and kick times with the force errors
and residuals of the force sweep. The counter change between the two sweeps
only affects what is counted, not the walk, so the forces are the same. Both
scripts take `ROOT`, `BIN`, `THREADS` and (the count sweep) `THETAS` from the
environment. The equal-accuracy sweep

```bash
TREE_THETA=0.4 docs/open_larger_cell_walk_results/run_fixed_tree_sweep.sh cosmo
```

runs `run.test=force` with `--run.treeOpeningAngle=0.4` for both rules, and
`parse_fixed_tree_sweep.py ROOT cosmo 0.4` tabulates it and reads off the
open-larger work at the lockstep median error.

## Results

Walk regression fixture (`fmm_walk_cpu`, 1536-particle Plummer clump of
radius 0.02 plus 512 uniform particles, theta 0.8, same tree for both rules):

| rule | tests | M2L direct | M2L Ewald | CP | PC | PP pairs | deferred | net-force residual |
|---|---|---|---|---|---|---|---|---|
| lockstep | 32116 | 11664 | 9310 | 300 | 300 | 1000622 | 0 | 2.4e-9 |
| open-larger | 31472 | 7892 | 6324 | 358 | 358 | 996222 | 6667 | 6.6e-10 |

Forces of the two rules differ by 7.7e-4 of the rms force; normalised per
particle by the larger of its own force and the rms force, the median
difference is 1.0e-4, the 99th percentile 2.3e-3 and the maximum 4.3e-3.

Sweeps over the opening angle, 32768 particles, same tree per angle for both
rules. Counts and kick times (5 HPX threads, wall seconds for the kick alone)
come from the `run.test=kick` sweep; residuals and force errors (median and
99th percentile relative force error against the direct periodic sum of 1000
sampled particles) from the `run.test=force` sweep. Both sweeps use the
cost-based hybrid selection; the earlier radius-rule sweeps are kept as
`kick_sweep_*.log`, `force_*.log` and `tables_radiusrule.md`; the force logs
are trimmed to their summary lines. Cell counts
exclude empty source cells; CP and PC are equal at every angle for both rules.

### Plummer model, radius 0.01: lockstep (L) versus open-larger (O)

| theta | M2L direct L | O | O/L | M2L Ewald L | O | tests L | O | PP pairs L | O | CP=PC L | O | deferred O | kick s L | O | net-force residual L | O | force error 50% L | O | 99% L | O |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.20 | 62534 | 48058 | 0.77 | 31292 | 27694 | 505012 | 516366 | 591029358 | 590659954 | 26944 | 34344 | 69036 | 0.29 | 0.27 | 8.9e-10 | 1.3e-10 | 4.31e-03 | 4.31e-03 | 3.14e-01 | 3.14e-01 |
| 0.25 | 33940 | 26224 | 0.77 | 20126 | 17604 | 329930 | 354224 | 415165884 | 414944752 | 27968 | 30842 | 45401 | 0.21 | 0.19 | 6.8e-10 | 2.9e-10 | 4.31e-03 | 4.31e-03 | 3.14e-01 | 3.14e-01 |
| 0.30 | 19196 | 15628 | 0.81 | 15180 | 12744 | 236498 | 260908 | 300907748 | 300522364 | 26596 | 27250 | 31925 | 0.16 | 0.14 | 6.5e-10 | 4.1e-10 | 4.31e-03 | 4.31e-03 | 3.14e-01 | 3.14e-01 |
| 0.35 | 18878 | 14038 | 0.74 | 11508 | 9596 | 192862 | 215472 | 232767716 | 232385178 | 23808 | 23953 | 28226 | 0.13 | 0.13 | 2.0e-09 | 6.1e-10 | 4.31e-03 | 4.31e-03 | 3.15e-01 | 3.15e-01 |
| 0.40 | 20378 | 14116 | 0.69 | 8694 | 7622 | 162966 | 184164 | 183721034 | 183568400 | 20954 | 21088 | 25631 | 0.10 | 0.10 | 2.2e-09 | 1.1e-09 | 4.31e-03 | 4.31e-03 | 3.14e-01 | 3.14e-01 |
| 0.45 | 18046 | 11958 | 0.66 | 6930 | 6010 | 132136 | 150414 | 140766706 | 140651226 | 18098 | 18193 | 21167 | 0.08 | 0.08 | 2.2e-10 | 2.6e-09 | 4.31e-03 | 4.31e-03 | 3.13e-01 | 3.14e-01 |
| 0.50 | 19732 | 13074 | 0.66 | 5630 | 4766 | 111288 | 126668 | 112638292 | 112383124 | 14358 | 14396 | 19159 | 0.07 | 0.07 | 1.7e-09 | 9.6e-10 | 4.26e-03 | 4.31e-03 | 3.14e-01 | 3.14e-01 |
| 0.55 | 20316 | 13586 | 0.67 | 4688 | 3840 | 96180 | 109428 | 100092652 | 100054374 | 11180 | 11315 | 17141 | 0.06 | 0.06 | 8.8e-10 | 5.0e-10 | 4.37e-03 | 4.32e-03 | 3.13e-01 | 3.15e-01 |
| 0.60 | 20848 | 13756 | 0.66 | 3908 | 3164 | 87688 | 98476 | 90778798 | 90576344 | 9440 | 9660 | 15255 | 0.06 | 0.06 | 2.4e-09 | 2.1e-09 | 4.36e-03 | 4.35e-03 | 3.10e-01 | 3.14e-01 |
| 0.65 | 17642 | 11712 | 0.66 | 3506 | 2762 | 73964 | 83196 | 81574044 | 81142056 | 7668 | 7847 | 12715 | 0.05 | 0.05 | 1.5e-09 | 1.4e-09 | 4.39e-03 | 4.36e-03 | 3.08e-01 | 3.10e-01 |
| 0.70 | 17082 | 11652 | 0.68 | 3118 | 2394 | 67044 | 74260 | 75254798 | 74184626 | 6352 | 6398 | 11153 | 0.04 | 0.05 | 1.7e-09 | 3.0e-09 | 4.46e-03 | 4.39e-03 | 3.09e-01 | 3.08e-01 |
| 0.75 | 16454 | 11578 | 0.70 | 2724 | 2104 | 59898 | 66422 | 69955406 | 69093974 | 4902 | 5003 | 9850 | 0.05 | 0.04 | 2.8e-09 | 3.4e-09 | 4.72e-03 | 4.80e-03 | 3.08e-01 | 3.07e-01 |
| 0.80 | 15974 | 11608 | 0.73 | 2398 | 1846 | 52358 | 57960 | 67435294 | 66473574 | 3132 | 3288 | 8358 | 0.04 | 0.04 | 4.6e-09 | 2.6e-09 | 5.64e-03 | 5.78e-03 | 3.02e-01 | 3.01e-01 |
| 0.85 | 15064 | 11370 | 0.75 | 2180 | 1620 | 47798 | 53230 | 66259578 | 65261962 | 2302 | 2478 | 7468 | 0.04 | 0.04 | 3.2e-09 | 3.6e-09 | 6.06e-03 | 6.08e-03 | 2.99e-01 | 2.99e-01 |
| 0.90 | 14996 | 11098 | 0.74 | 2082 | 1462 | 46306 | 50594 | 65278450 | 64261582 | 2080 | 2152 | 7056 | 0.04 | 0.03 | 3.5e-09 | 3.8e-09 | 6.54e-03 | 6.54e-03 | 2.99e-01 | 2.99e-01 |
| 0.95 | 14644 | 10858 | 0.74 | 1808 | 1320 | 44290 | 48272 | 64496694 | 63526466 | 1836 | 1846 | 6674 | 0.04 | 0.03 | 4.4e-09 | 6.5e-09 | 6.66e-03 | 6.70e-03 | 2.99e-01 | 2.99e-01 |



### cosmological initial condition, z = 49: lockstep (L) versus open-larger (O)

| theta | M2L direct L | O | O/L | M2L Ewald L | O | tests L | O | PP pairs L | O | CP=PC L | O | deferred O | kick s L | O | net-force residual L | O | force error 50% L | O | 99% L | O |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.20 | 30430550 | 15371616 | 0.51 | 14982627 | 10454879 | 68355515 | 62993247 | 36189146 | 12215846 | 16826 | 1879642 | 15065602 | 8.98 | 8.52 | 7.5e-10 | 5.9e-10 | 9.01e-06 | 1.39e-05 | 1.34e-04 | 1.51e-04 |
| 0.25 | 10795478 | 6203460 | 0.57 | 8809795 | 5389931 | 29603152 | 27763922 | 45039786 | 15658782 | 133156 | 768831 | 6729441 | 5.51 | 4.27 | 6.7e-10 | 4.4e-10 | 1.64e-05 | 3.04e-05 | 1.26e-04 | 2.54e-04 |
| 0.30 | 4966932 | 2233770 | 0.45 | 4728259 | 2852989 | 14064590 | 12758846 | 17770034 | 17262102 | 287100 | 518371 | 2977284 | 3.39 | 2.05 | 5.3e-10 | 5.4e-10 | 5.80e-05 | 7.42e-05 | 4.85e-04 | 5.60e-04 |
| 0.35 | 3338054 | 1560856 | 0.47 | 2201263 | 1488021 | 8103062 | 7724870 | 11974666 | 11654806 | 175382 | 321460 | 1803543 | 1.50 | 1.22 | 5.3e-10 | 4.9e-10 | 1.61e-04 | 1.87e-04 | 1.75e-03 | 1.91e-03 |
| 0.40 | 2325314 | 1122892 | 0.48 | 1259506 | 817302 | 5260514 | 4963762 | 8916660 | 8702068 | 111850 | 208640 | 1150139 | 1.05 | 0.73 | 4.8e-10 | 4.3e-10 | 3.43e-04 | 4.63e-04 | 3.83e-03 | 4.27e-03 |
| 0.45 | 678068 | 311386 | 0.46 | 738640 | 472844 | 2303290 | 2189474 | 25820622 | 25795016 | 108655 | 110477 | 493818 | 0.56 | 0.40 | 6.2e-10 | 5.8e-10 | 4.11e-04 | 4.20e-04 | 4.40e-03 | 4.27e-03 |
| 0.50 | 530982 | 249996 | 0.47 | 418690 | 285612 | 1565626 | 1532230 | 20371852 | 20338856 | 75812 | 80560 | 346147 | 0.36 | 0.26 | 4.7e-10 | 4.2e-10 | 8.48e-04 | 9.62e-04 | 8.86e-03 | 1.07e-02 |
| 0.55 | 404002 | 201256 | 0.50 | 264949 | 182045 | 1123038 | 1116926 | 16799288 | 16764444 | 56783 | 59330 | 248896 | 0.23 | 0.17 | 4.6e-10 | 5.9e-10 | 1.45e-03 | 1.57e-03 | 1.78e-02 | 1.91e-02 |
| 0.60 | 312242 | 158662 | 0.51 | 186346 | 125148 | 848090 | 843906 | 13627912 | 13583404 | 45037 | 46863 | 185137 | 0.17 | 0.11 | 5.7e-10 | 5.1e-10 | 2.36e-03 | 2.51e-03 | 2.97e-02 | 2.99e-02 |
| 0.65 | 128242 | 73480 | 0.57 | 122584 | 89812 | 408946 | 435370 | 29650002 | 29275442 | 14549 | 14791 | 97902 | 0.10 | 0.11 | 5.7e-10 | 5.6e-10 | 2.48e-03 | 2.61e-03 | 3.92e-02 | 4.65e-02 |
| 0.70 | 101720 | 60586 | 0.60 | 93646 | 70544 | 322762 | 350966 | 25063760 | 24591314 | 12069 | 11880 | 77827 | 0.08 | 0.08 | 5.5e-10 | 5.3e-10 | 3.74e-03 | 4.33e-03 | 4.80e-02 | 5.65e-02 |
| 0.75 | 81072 | 49628 | 0.61 | 75193 | 57319 | 261982 | 289630 | 21937550 | 21404106 | 10268 | 10067 | 63593 | 0.07 | 0.06 | 5.3e-10 | 6.5e-10 | 6.04e-03 | 6.83e-03 | 6.94e-02 | 8.26e-02 |
| 0.80 | 47422 | 24812 | 0.52 | 60880 | 45892 | 162346 | 175382 | 42438054 | 42408192 | 1900 | 4180 | 33304 | 0.07 | 0.06 | 6.0e-10 | 4.1e-10 | 8.28e-03 | 9.59e-03 | 1.35e-01 | 1.37e-01 |
| 0.85 | 41708 | 22968 | 0.55 | 53464 | 37420 | 142570 | 148214 | 35115268 | 34996394 | 1865 | 3373 | 29228 | 0.06 | 0.05 | 6.6e-10 | 5.9e-10 | 1.31e-02 | 1.62e-02 | 2.09e-01 | 2.17e-01 |
| 0.90 | 38998 | 22078 | 0.57 | 49750 | 31350 | 132666 | 129094 | 31398108 | 30581950 | 1834 | 2629 | 26650 | 0.05 | 0.04 | 7.1e-10 | 6.6e-10 | 1.87e-02 | 2.51e-02 | 2.46e-01 | 2.98e-01 |
| 0.95 | 33506 | 19256 | 0.57 | 42712 | 24750 | 113914 | 105218 | 25280288 | 23720640 | 1800 | 2167 | 22040 | 0.05 | 0.03 | 7.2e-10 | 6.2e-10 | 3.13e-02 | 3.75e-02 | 4.24e-01 | 4.40e-01 |

### Equal accuracy on one tree, cosmological initial condition

`run.treeOpeningAngle` makes `run.test=force` build the tree once at that
angle and sweep only the kick's opening angle, from the tree's angle to 0.95
in steps of 0.025. The tree's Ewald leaf criterion in `src/gravity/tree.cpp` bounds a
leaf's enclosing radius by 0.25 theta_tree / (1 + theta_tree), which is
exactly the bound the Ewald pass needs for a leaf pair at the nearest image
distance to be accepted at any kick angle of at least theta_tree, so larger
kick angles never meet an unacceptable leaf pair. Both rules then walk the
same tree at the same angles, and the comparison is read off at equal median
force error: for each lockstep angle, the open-larger angle with the same
median error is found by log-linear interpolation between adjacent sweep
points, its counts and kick time are interpolated linearly in the angle, and
the ratio to the lockstep value is tabulated (`parse_fixed_tree_sweep.py`).
The median error comes from `analyticCompare` on 1000 randomly sampled
particles against the direct periodic sum, and its sampling noise is about
10 percent, which moves a matched angle by up to one sweep step; the ratios
are a band, not a curve. 32768 particles, 8 HPX threads, CPU walk, trees
built at theta 0.2, 0.4 and 0.6 (`force_fixed_tree*_cosmo_open_*.log`,
trimmed to their summary lines); `tables_fixed_tree.md` holds all six tables.
The first row of each sweep reproduces the per-angle sweep's counts at that
angle, since it is the same tree.

Tree built at theta 0.4, kick angle swept, lockstep (L) versus open-larger (O):

| kick theta | M2L direct L | O | M2L Ewald L | O | PP pairs L | O | CP=PC L | O | kick s L | O | residual L | O | force error 50% L | O | 99% L | O |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.400 | 2325314 | 1122892 | 1259506 | 817302 | 8916660 | 8702068 | 111850 | 208640 | 0.74 | 0.43 | 4.8e-10 | 4.3e-10 | 3.10e-04 | 4.13e-04 | 3.05e-03 | 3.78e-03 |
| 0.425 | 1957014 | 966320 | 971443 | 620029 | 7680190 | 7536430 | 91185 | 170406 | 0.59 | 0.40 | 5.6e-10 | 5.7e-10 | 5.57e-04 | 6.90e-04 | 5.89e-03 | 5.96e-03 |
| 0.450 | 1669484 | 833954 | 738640 | 472844 | 6530054 | 6416868 | 76851 | 143080 | 0.48 | 0.30 | 6.2e-10 | 5.8e-10 | 9.29e-04 | 1.15e-03 | 1.03e-02 | 1.12e-02 |
| 0.475 | 1435100 | 722694 | 557803 | 366379 | 5867592 | 5774948 | 63376 | 119251 | 0.37 | 0.26 | 5.2e-10 | 5.2e-10 | 1.21e-03 | 1.46e-03 | 1.31e-02 | 1.24e-02 |
| 0.500 | 1245746 | 633302 | 418690 | 285612 | 5406900 | 5332144 | 52151 | 99088 | 0.30 | 0.20 | 4.6e-10 | 4.2e-10 | 1.64e-03 | 1.89e-03 | 1.49e-02 | 1.65e-02 |
| 0.525 | 1087916 | 561056 | 325618 | 227506 | 4908584 | 4844384 | 43637 | 83282 | 0.21 | 0.16 | 3.5e-10 | 5.1e-10 | 2.02e-03 | 2.41e-03 | 2.08e-02 | 2.25e-02 |
| 0.550 | 959154 | 499010 | 264949 | 182045 | 4504802 | 4449892 | 36720 | 70592 | 0.18 | 0.15 | 4.6e-10 | 5.9e-10 | 2.61e-03 | 2.96e-03 | 2.62e-02 | 2.80e-02 |
| 0.575 | 848834 | 443456 | 221815 | 148707 | 4149324 | 4097456 | 31567 | 60835 | 0.15 | 0.11 | 3.8e-10 | 5.3e-10 | 3.35e-03 | 3.86e-03 | 2.95e-02 | 3.31e-02 |
| 0.600 | 754794 | 396308 | 186346 | 125148 | 3730692 | 3673094 | 27853 | 53090 | 0.13 | 0.11 | 5.7e-10 | 5.1e-10 | 4.27e-03 | 4.67e-03 | 3.45e-02 | 4.26e-02 |
| 0.625 | 666972 | 352382 | 150082 | 104898 | 3263556 | 3206120 | 26076 | 47579 | 0.12 | 0.09 | 5.1e-10 | 4.8e-10 | 5.49e-03 | 7.13e-03 | 5.28e-02 | 6.01e-02 |
| 0.650 | 591046 | 315752 | 122584 | 89812 | 2866624 | 2817778 | 23403 | 41947 | 0.10 | 0.08 | 5.9e-10 | 5.6e-10 | 7.96e-03 | 8.60e-03 | 7.18e-02 | 7.47e-02 |
| 0.675 | 531780 | 286852 | 106594 | 79230 | 2514784 | 2460134 | 20101 | 36747 | 0.08 | 0.06 | 5.2e-10 | 5.0e-10 | 9.91e-03 | 1.20e-02 | 7.97e-02 | 1.07e-01 |
| 0.700 | 481692 | 263272 | 93646 | 70544 | 2227724 | 2163712 | 17987 | 32228 | 0.07 | 0.06 | 5.3e-10 | 5.2e-10 | 9.95e-03 | 1.38e-02 | 1.03e-01 | 1.22e-01 |
| 0.725 | 439496 | 243800 | 84904 | 63928 | 2087688 | 2014948 | 16091 | 28006 | 0.07 | 0.05 | 4.8e-10 | 4.2e-10 | 1.27e-02 | 1.60e-02 | 1.24e-01 | 1.34e-01 |
| 0.750 | 398556 | 225726 | 75193 | 57319 | 1981642 | 1883694 | 14379 | 24249 | 0.06 | 0.05 | 5.3e-10 | 6.2e-10 | 1.58e-02 | 1.84e-02 | 1.50e-01 | 1.72e-01 |
| 0.775 | 356582 | 204642 | 67084 | 51128 | 1808182 | 1682504 | 13562 | 21910 | 0.05 | 0.05 | 5.8e-10 | 5.1e-10 | 1.88e-02 | 2.08e-02 | 1.75e-01 | 2.33e-01 |
| 0.800 | 318584 | 185026 | 60880 | 45892 | 1589946 | 1470680 | 13355 | 20533 | 0.05 | 0.04 | 6.1e-10 | 4.4e-10 | 2.56e-02 | 2.76e-02 | 2.77e-01 | 2.94e-01 |
| 0.825 | 287424 | 169578 | 56374 | 41300 | 1386862 | 1299066 | 12803 | 18885 | 0.05 | 0.04 | 5.9e-10 | 7.2e-10 | 3.52e-02 | 3.73e-02 | 3.21e-01 | 3.05e-01 |
| 0.850 | 258078 | 156334 | 53464 | 37420 | 1241224 | 1178568 | 10940 | 16025 | 0.05 | 0.04 | 6.4e-10 | 5.6e-10 | 4.35e-02 | 4.83e-02 | 4.06e-01 | 4.28e-01 |
| 0.875 | 238024 | 148238 | 51874 | 34502 | 1141212 | 1099462 | 8828 | 12548 | 0.05 | 0.03 | 7.5e-10 | 6.6e-10 | 5.58e-02 | 5.97e-02 | 4.83e-01 | 4.88e-01 |
| 0.900 | 219394 | 140514 | 49750 | 31350 | 1076338 | 1045108 | 7227 | 9689 | 0.04 | 0.03 | 7.1e-10 | 7.0e-10 | 6.72e-02 | 7.35e-02 | 5.20e-01 | 5.80e-01 |
| 0.925 | 201460 | 131286 | 46789 | 28285 | 1019250 | 995982 | 5875 | 7679 | 0.04 | 0.03 | 7.0e-10 | 7.6e-10 | 8.12e-02 | 8.11e-02 | 5.92e-01 | 6.40e-01 |
| 0.950 | 184650 | 120736 | 42712 | 24750 | 975428 | 956856 | 4959 | 6538 | 0.03 | 0.03 | 7.1e-10 | 6.3e-10 | 8.43e-02 | 9.08e-02 | 7.85e-01 | 7.33e-01 |

Open-larger work at the lockstep median error, tree built at theta 0.4:

| lockstep theta | median error | open-larger theta at that error | M2L direct O/L | M2L Ewald O/L | M2L total O/L | PP pairs O/L | kick time O/L |
|---|---|---|---|---|---|---|---|
| 0.400 | 3.10e-04 | outside the open-larger sweep | | | | | |
| 0.425 | 5.57e-04 | 0.415 | 0.53 | 0.72 | 0.59 | 1.04 | 0.69 |
| 0.450 | 9.29e-04 | 0.440 | 0.53 | 0.72 | 0.59 | 1.05 | 0.71 |
| 0.475 | 1.21e-03 | 0.456 | 0.56 | 0.80 | 0.63 | 1.07 | 0.79 |
| 0.500 | 1.64e-03 | 0.486 | 0.55 | 0.79 | 0.61 | 1.03 | 0.78 |
| 0.525 | 2.02e-03 | 0.507 | 0.56 | 0.83 | 0.63 | 1.06 | 0.94 |
| 0.550 | 2.61e-03 | 0.535 | 0.56 | 0.79 | 0.61 | 1.04 | 0.85 |
| 0.575 | 3.35e-03 | 0.562 | 0.56 | 0.75 | 0.60 | 1.03 | 0.88 |
| 0.600 | 4.27e-03 | 0.588 | 0.55 | 0.73 | 0.59 | 1.04 | 0.86 |
| 0.625 | 5.49e-03 | 0.610 | 0.57 | 0.78 | 0.61 | 1.07 | 0.89 |
| 0.650 | 7.96e-03 | 0.640 | 0.56 | 0.78 | 0.60 | 1.04 | 0.84 |
| 0.675 | 9.91e-03 | 0.661 | 0.57 | 0.80 | 0.61 | 1.06 | 0.94 |
| 0.700 | 9.95e-03 | 0.661 | 0.63 | 0.91 | 0.67 | 1.19 | 1.03 |
| 0.725 | 1.27e-02 | 0.686 | 0.63 | 0.89 | 0.67 | 1.12 | 0.88 |
| 0.750 | 1.58e-02 | 0.723 | 0.62 | 0.86 | 0.65 | 1.02 | 0.91 |
| 0.775 | 1.88e-02 | 0.755 | 0.62 | 0.84 | 0.66 | 1.02 | 0.99 |
| 0.800 | 2.56e-02 | 0.793 | 0.60 | 0.78 | 0.63 | 0.96 | 0.79 |
| 0.825 | 3.52e-02 | 0.820 | 0.60 | 0.75 | 0.62 | 0.96 | 0.73 |
| 0.850 | 4.35e-02 | 0.840 | 0.63 | 0.73 | 0.64 | 0.99 | 0.69 |
| 0.875 | 5.58e-02 | 0.867 | 0.63 | 0.68 | 0.64 | 0.99 | 0.69 |
| 0.900 | 6.72e-02 | 0.889 | 0.66 | 0.66 | 0.66 | 0.99 | 0.77 |
| 0.925 | 8.12e-02 | 0.925 | 0.65 | 0.60 | 0.64 | 0.98 | 0.72 |
| 0.950 | 8.43e-02 | 0.934 | 0.69 | 0.63 | 0.68 | 1.01 | 0.82 |

Over the matched angles of each tree, median (range) of the open-larger to
lockstep ratios, and of the angle offset lockstep minus open-larger:

| tree theta | matched angles | angle offset | M2L direct O/L | M2L Ewald O/L | M2L total O/L | PP pairs O/L | kick time O/L |
|---|---|---|---|---|---|---|---|
| 0.2 | 30 | 0.01 (-0.01 to 0.03) | 0.59 (0.54 to 0.69) | 0.74 (0.59 to 0.95) | 0.61 (0.55 to 0.75) | 0.53 (0.40 to 0.74) | 0.71 (0.60 to 0.90) |
| 0.4 | 22 | 0.01 (-0.00 to 0.04) | 0.60 (0.53 to 0.69) | 0.78 (0.60 to 0.91) | 0.63 (0.59 to 0.68) | 1.04 (0.96 to 1.19) | 0.84 (0.69 to 1.03) |
| 0.6 | 14 | 0.01 (-0.00 to 0.03) | 0.55 (0.52 to 0.58) | 0.77 (0.61 to 0.83) | 0.62 (0.58 to 0.64) | 1.02 (0.96 to 1.10) | 0.89 (0.78 to 0.96) |

## Observations

- Momentum: the net-force residual stays at 1e-10 to 6e-9 for both rules at
  every angle and on both models. The open-larger rule is as reciprocal as the
  lockstep rule, as the symmetry argument predicts.
- Work: at fixed theta the open-larger rule does 45 to 61 percent of the
  direct M2L work on the cosmological model and 66 to 81 percent on the
  Plummer model, and 70 to 90 percent of the Ewald M2L work. Particle pairs
  are unchanged to within one percent, since leaf sinks always open. Acceptance
  tests move both ways: up to 8 percent fewer at small theta on the
  cosmological model, up to 14 percent more at large theta, because an
  internal sink now retests opened children at its own level. Kick wall time
  follows the M2L count where M2L dominates: on the cosmological model at
  theta 0.25 to 0.6 the open-larger kick takes 60 to 80 percent of the
  lockstep time; the Plummer kick is dominated by its 590 million particle
  pairs at theta 0.2 and does not change. The timings were taken while other
  sessions shared the machine and are noisy at the 20 percent level.
- Accuracy at fixed theta: on the Plummer model the force errors are identical
  for both rules; they are dominated by the dense core and do not depend on
  theta. On the near-uniform cosmological model the open-larger rule's median
  force error is larger at the same theta, by 2 to 35 percent from theta 0.3
  upwards and by 50 to 85 percent at theta 0.2 and 0.25, because the pairs it
  accepts sit closer to the acceptance limit on average, while the lockstep
  rule over-opens and gains accuracy it was not asked for; at the smallest
  angles the lockstep walk also sums most of its leaf pairs directly under
  the cost rule, which is exact. A fair cost
  comparison needs equal accuracy, so the open-larger run should use a
  slightly smaller theta: see the next point.
- Equal accuracy: on one tree the open-larger rule reaches the lockstep
  rule's median force error at an angle 0.01 to 0.03 smaller, and there it
  does 55 to 65 percent of the lockstep M2L work, 53 to 69 percent of the
  direct M2L and 60 to 95 percent of the Ewald M2L, on all three trees. The
  fixed-angle comparison overstated the direct M2L saving (45 to 61 percent)
  by about the one sweep step the error penalty costs. Particle pairs are
  unchanged on the trees built at theta 0.4 and 0.6, but halved on the tree
  built at theta 0.2, whose leaves hold a few particles: under the cost rule
  the lockstep walk sums its leaf pairs directly while the open-larger walk
  keeps the larger cell closed and uses a hybrid, 16826 against 1879642
  hybrids at theta 0.2. The kick time at equal error is 0.6 to 1.0 of the
  lockstep time, with medians of 0.71, 0.84 and 0.89 on the three trees,
  inside the 20 percent timing noise. The Ewald M2L count at a given kick
  angle is the same on all three trees, 1259506 lockstep and 817302
  open-larger at theta 0.4, because the Ewald pass is decided above the
  leaves.
- Hybrids: with empty cells counted, the cell-particle and particle-cell
  counts differed by up to 12 percent at theta 0.2. An empty sibling cell is
  tested as a source by every neighbour, but its own walk returns before
  counting anything, so it inflates one side. Counting only occupied sources
  (`hybrid_split_*_theta0.2.log`) makes the two counts equal exactly, and
  equal separately for leaf and internal partners: Plummer at theta 0.2 has
  25646 pairs with a leaf partner and 9680 with an internal partner on both
  sides, and the cosmological model has 2167894 and 0. The hybrid selection is
  reciprocal, for both rules.
- Empty cells cost real work: excluding them removed 40 percent of the direct
  M2L entries of the Plummer model at theta 0.2 (103892 to 62534) and 2 percent
  on the cosmological model. The Ewald criterion splits large sparse boxes down
  to single particles with empty siblings, and the CPU walk tested and
  evaluated those zero-mass cells like any other source; the CUDA Ewald pass
  already skipped them. The walk now keeps them out of its lists: see "Empty
  source cells" for the measured effect.

## Follow-ups

- Repeat the equal-accuracy comparison on the CUDA walk, by kick time, since
  the CUDA walk reports no counts.
- Pin the analytic sample, or enlarge it, so the matched angle is not blurred
  by the 10 percent sampling noise of the median error.
- The kick-time ratio at equal error lags the M2L ratio (0.7 to 0.9 against
  0.6); profile where the open-larger kick spends the difference, in the
  acceptance tests it repeats at the sink's level or in the extra hybrids.

