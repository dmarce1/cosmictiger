# Hierarchical gravity and time-bin merging

the fixed hierarchical scheme advances gravity with the Hamiltonian split of
GADGET-4 ([Springel et al. 2021, MNRAS 506, 2871, section 4.2](https://wwwmpa.mpa-garching.mpg.de/gadget4/gadget4-code-paper.pdf)).
`kickStepHierarchical` in `src/core/driver.cpp` runs one synchronization point.
Call its lowest synchronized rung n and its finest occupied rung m. The set S_L
holds every particle on rung L or finer. Each solve builds a tree of one S_L and
computes forces on S_L from S_L only:

- The closing ascent visits L = m, ..., n + 1 with the old rungs. It applies
  -F(S_L) over half a rung-L step.
- The top solve at L = n applies +F(S_n) over half a rung-n step, both closing
  and opening. It assigns the new rungs.
- The opening descent visits L = n + 1, ... with the new rungs. It applies
  -F(S_L) over half a rung-L step, then drifts.

Added over the nested levels, a synchronization at rung n kicks with
+F(S_n) dt_n - F(S_n+1) dt_n+1 - F(S_n+2) dt_n+2 - ..., where dt_L is half a
rung-L step. The active rung has weight +1. Each finer rung has weight -1/2
relative to the rung above it.

## Cost of a rung and the merge rule

The work of a solve is proportional to the size of its set. Over one rung-L
step, S_L+1 is solved three times: twice at the synchronization where rung L
is active (ascent and descent) and once at the middle synchronization, where
rung L+1 is the top. Counting every level gives GADGET-4 eq. (52):

    K_hierarchical = N_0 + 3 * sum_{k >= 1} 2^(k-1) N_k,

with N_k = |S_n+k|. N_0 is therefore every particle on or above the lowest
rung, not only the particles on that rung.

Merging rung n into rung n + 1 replaces one rung-n step with two rung-(n+1)
steps. The deeper sets cost the same either way, so the hierarchy costs
N_0 + 3 N_1 against 2 N_0 for the merged steps. The paper describes strict
cost saving, N_1 > N_0 / 3. The selected policy also promotes at equal cost:

    2 N_0 <= N_0 + 3 N_1
    N_0 <= 3 N_1
    N_1 >= ceil(N_0 / 3).

The last expression uses integer particle counts. The default comparison uses
integer division and remainder, so neither multiplication overflow nor rounding
large counts through floating point can move the equality boundary. A finer
rung must be occupied; an isolated lowest rung is not promoted.

The result does not depend on whether n + 1 is the finest rung. N_1 counts every
finer rung, and the solves of deeper sets cancel between the two choices. In
terms of the lowest rung alone, with n_n particles, the same rule reads
N_1 >= n_n / 2, with equality included.

## The option

`run.rungMergeFraction` (default 1/3) is the threshold on N_1 / N_0. It is
checked at the top solve when the lowest rung equals the driver's floor
`minrung0`. When N_1 >= `run.rungMergeFraction` * N_0 and N_1 > 0, the top solve
only closes rung n. Every particle is raised to rung n + 1, which opens the new step, and
the floor moves to n + 1. The run prints one line per merge:

    Merging rung 0 into rung 1: 412 of 1000 particles on finer rungs meet run.rungMergeFraction = 0.333333

The value must lie in [0, 1]. Zero merges whenever a finer rung is occupied. One
merges only when the lowest rung is empty and every active particle is finer.
Nondefault fractions use the same inclusive comparison. The floor relaxes by
one rung every eighth coarse step of a full evaluation, and no merge is considered at that step. That lets the particles
regain the longer step once the finer rungs empty.

The sole integration scheme couples PMMM gravity to midpoint SPH. Its long
kicks occur at outer-interval boundaries; short gravity uses the ladder above.
Each short solve rebuilds the selected frozen cells with bounds containing
their current particles. Hydro advances positive physical intervals once per
event; signed ladder corrections update gravity alone, with closing gravity
before hydro and opening gravity afterward. See the
[SPH midpoint contract](sph_rk2_time_integration.md) and
[frozen PMMM membership](pmmm.md#outer-interval-with-frozen-near-membership).
Merging is disabled for the
terminal closing ladder and in the fixed-time scenario driver (`src/core/scenario_driver.cpp`). Those
callers pass `nocrop`.

Before November 2026 the rule was hard-coded as three times the finer count
exceeding the lowest rung's own count. That is N_1 > N_0 / 4, which merged
too early. For 76 finer particles above 300 on the lowest rung, it merged
although the hierarchy costs 376 + 3 * 76 = 604 particle-solves per coarse step
against 2 * 376 = 752 merged.

`tests/rung_merge.cpp` (`ctest -R rung_merge`) checks the boundary cases. It
also models the ladder's solve counts by brute force over four rungs and checks
that the decision promotes exactly when the merged schedule is cheaper or ties,
provided a finer rung is occupied. It also covers maximum-size particle counts,
empty rungs, and configurable-threshold endpoints.
