# Open-larger cell opening in the CPU tree walk

A Dehnen-style opening rule for the FMM tree walk: when a cell pair fails the
acceptance test, open the cell with the larger radius instead of opening both.
The rule is behind `gravity.openLargerCell`, default `true`, on both the CPU
and the CUDA walk; `false` restores the lockstep walk for comparison. The kick
stamps the rule onto its parameters, so the CPU walk, the remote kick action
and the CUDA kernel apply the same rule and the accepted pair set stays
reciprocal across a CPU/CUDA handoff.

The hybrid choice for a leaf pair that fails the cell-cell test is a flop
comparison rather than a radius rule: see "Hybrids by cost" below.

Branch: `order_aware_cuda_shared_memory`. Results from 2026-10-19 on the
24-core workstation, CPU walk with 8 HPX threads, FMM order 7, 32768 particles
for the force sweeps.

## The walk today

Each sink node walks its own checklist of source nodes. The acceptance test
is symmetric in the two cells: it compares the centre distance with the sum of
the two radii divided by theta, plus the support-separation terms in
`cosmictiger/fmm_acceptance.hpp`. An internal sink makes one pass over its
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
`cosmictiger/kick_walk.hpp` now does one of three things:

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

The CUDA walk in `cosmictiger/kick_cuda_impl.hpp` applies the same rule with
a device twin of the helper, `cudaOpenOrDefer`, a `bothlist` for the
equal-radius case whose children go to the sink's children, and the per-level
loop iterating for internal sinks when the rule is on.

## Hybrids by cost

A cell-particle or particle-cell interaction stands in for direct summation
of a leaf pair, and whichever side is evaluated explicitly, both walks have to
pick the same pairing. The selection in `cosmictiger/fmm_acceptance.hpp` now
compares flop counts whose only runtime inputs are the two particle counts;
the per-order costs come from the generated kernels through
`FmmHybridCost<order>` in `cosmictiger/fmm_kernels.hpp`:

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

`cosmictiger/kick_interaction_counts.hpp` adds rank-local counters filled by
the CPU walk: acceptance tests, direct and Ewald M2L, cell-particle,
particle-cell, particle-particle leaf pairs and particle pairs, source openings
and deferred sources. Sources without particles are not counted, because an
empty sibling cell reaches every neighbour's lists while its own walk returns
before counting anything. `run.test=force` prints the counters and the
rank-local net force residual, the norm of the summed force divided by the sum
of force norms, after every kick; `run.test=kick` prints the counters and the
kick time at `gravity.openingAngle`.

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
only affects what is counted, not the walk, so the forces are the same.

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
  slightly smaller theta; the sweep rebuilds the tree per theta, so that
  comparison is not available from these runs.
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
  to single particles with empty siblings, and the CPU walk tests and
  evaluates those zero-mass cells like any other source; the CUDA Ewald pass
  already skips them. Dropping empty sources from the checklist is an easy
  saving on clustered trees, independent of the opening rule.

## Follow-ups

- Compare the rules at equal force error on one tree: build the tree once and
  sweep the kick's theta only.
- Skip zero-mass source cells in the CPU walk, as the CUDA Ewald pass does.

