# Hierarchical SPH and persistent hydro source geometry

Hierarchical SPH advances gas on individual timestep rungs while gravity uses
its existing hierarchical kicks. Fine hydro events query a retained gas source
tree, predict the sources they touch to the event time, and run the ordinary
SPH kernels on packed active sinks. They do not require a complete gas sort or
a density solve for every inactive particle.

The spatial design follows GADGET-4's separate hydro neighbor tree and on-demand
drifts ([code paper, section 5.7.1](https://wwwmpa.mpa-garching.mpg.de/gadget4/gadget4-code-paper.pdf#page=39)).
The gas tree contains every hydro source even when the current gravity tree
contains only an active rung suffix. Its bounding boxes move with the source
trajectories; particles need not remain inside the original kd splitting planes.

## Selection

Build with `COSMICTIGER_ENABLE_SPH=ON`,
`COSMICTIGER_ENABLE_INTEGRATOR_HIERARCHICAL=ON`, and a gravity solver capability.
The runtime may disable self-gravity:

```ini
[run]
timeIntegrator = hierarchical
[hydro]
enabled = true
method = sph
[gravity]
enabled = false
```

With self-gravity enabled, FMM and PMMM use the same hydro scheduling interface.
Viscosity evolution must be `none` or `morrisAndMonaghan`: the
`cullenAndDehnen` and `priceEtAl` switches require the full material acceleration
gradient, while a fine hierarchical gravity walk supplies only its active
subset field. Those switches remain available with self-gravity disabled.
Cooling and star formation are supported; see
[Cooling and star formation](#cooling-and-star-formation).

## Tree and particle lifetime

`cosmictiger/hydro/source_tree.hpp` owns a persistent geometry tree indexed by
stable source IDs. Its trajectory envelopes, support bounds, signal-speed
bounds, and rung bounds are independent of the hydro method's fluid state.
Updating an existing source refits its leaf and ancestors without sorting the
inactive population. Structural membership changes rebuild the topology.
The position envelope is evaluated at the requested event time, so coherent
bulk motion translates a clump's boxes instead of enlarging every box by the
clump's complete displacement since the last sort.

SPH keeps immutable fluid anchors grouped by rung. An anchor records its epoch,
valid endpoint, position, drift velocity, synchronized velocity, hydro
acceleration, thermal state, support, grad-H correction, and prediction rates.
Source predictions are cached at the event time. Replacing active anchors leaves
slower anchors intact. The spatial tree returns candidate IDs; the SPH adapter
predicts the corresponding fluid state and the pair kernels apply their exact
support-distance tests.

Inactive positions follow the saved drift segment. Velocity and internal energy
use their saved rates. Density uses its logarithmic rate; free support follows
the corresponding density scaling within the configured bounds. Grad-H follows
its retained rate and vanishes when a support limit holds the smoothing radius
fixed. These are source predictions, not fresh inactive density solves.

The geometry API also supports quadratic generator trajectories and separate
fluid velocities, allowing fixed-mass MFM to share spatial indexing without
reusing SPH's thermodynamic predictor. MFM pair-flux closure remains a separate
integrator operation.

Gas-array slots may move during sorting or domain migration. The retained
sources own their snapshots and do not retain mutable gas-slot indices. Main
particle swaps repair the inverse gas mapping. Hierarchical migration moves only
the selected main-particle suffix, preserves the inactive prefix and its rung
stack, and transfers the source anchors with their gas particles. The next
active refresh updates ownership before any hydro source query.

Gravity tree cleanup does not destroy the hydro source tree. Gas teardown and
checkpoint loading reset the source cache. A hierarchical restart reconstructs
it from the saved anchors and retains the checkpoint's domain ownership and rung
stack. Repartitioning a staggered checkpoint before closing the pending ladder
would invalidate that state.

## Hydro and gravity intervals

At each top event, gravity closes its pending gas kick and defers the next gas
opening. SPH closes the completed hydro interval, solves active supports and
densities, evaluates the new force, and chooses the next rungs. Gravity then
opens its kick; each exact rung opens hydro and retains its drift trajectory
before advancing. Hydro acceleration excludes the self-gravity kick, while the
timestep proposal still includes the gravity force and its completed timestep
bound. This avoids applying gravity twice.

CPU and CUDA use the same staged source/sink layout. Source rows remain immutable
while kernels write the packed active sinks. This lifetime separation permits
source prediction and a future CPU/CUDA division without sharing mutable
particle permutations with an in-flight kernel.

The SPH update is one-sided on active sinks. It does not apply immediate
reciprocal kicks to sleeping gas, so exact pairwise momentum or energy
conservation across different rungs is not guaranteed. Accuracy and conservation
must be measured at the selected timestep resolution.

## Cooling and star formation

`chemistry.enabled` (GADGET-4 equilibrium cooling, `docs/primordial_chemistry.md`) and
`starFormation.enabled` (Springel & Hernquist multiphase star formation,
`docs/star_formation.md`) act as in the traditional SPH step: the source runs once
on each gas particle when its interval closes, after the closing corrector and
before the opening force selects the next rung. Each particle integrates the
source over the anchor interval its hydro closure has just completed
(`validUntil - epoch`), so particles on different rungs cool and form stars over
their own durations. Sleeping particles receive no source between their events;
neighbours see their predicted, unsourced state, as under traditional individual
timesteps. Every locality stages its results before any locality commits them.

A conversion removes the gas particle at that closure. The step rebuilds its
active work list, because conversions move gas slots, and re-solves the active
supports and densities without the new star before the opening force; the
refresh preceding each pass drops the star from the retained sources. The top
gravity kick deferred the particle's opening while it was gas, so the star takes
the collisionless top opening instead: its rung follows the kick's top selection
from the gravity timestep the kick stored for it, and it receives the top half
kick together with the gas. The terminal energy evaluation applies sources and
conversions over the outstanding intervals but opens nothing.

The heating diagnostic and the `SPH hierarchy sources` log row report the energy
the sources removed and the conversions at each event. Events between full
synchronizations change only their active gas, so the cosmological expansion
work uses the current thermal energy of all gas, summed after each step.

Hierarchical SPH has no wake-up path. A source that heats a particle does not
shorten an interval a sleeping neighbour has already selected; the signal bound
applies only to rungs selected afterwards.

## Incoming signals and work counters

The source tree supplies the nonlocal signal-travel timestep bound of GADGET-4
equation (79), in addition to the ordinary CFL, acceleration, thermal, and local
neighbor-rung limits. It considers approaching sources beyond the current
support and prunes nodes using conservative distance and signal-speed bounds.

The geometry bounds enclose the implemented predictor within each saved
interval. They do not predict every nonlinear future flow or unannounced heating
impulse. Querying an expired source is an error; the tree does not provide a
universal replacement for physical wake-ups.

`SPH hierarchy` log rows report active sinks, sorted records, density work, force
work, source candidates, and source predictions. Fine events still sort their
active sink batches. Refit work and source prediction are distinct from sorting;
the initial tree build and structural rebuilds should be distinguished from
routine fine-event updates when assessing performance.

## Validation status

Local CPU checks on `g4_hydro_tree` cover periodic source queries, moving bounds,
retained state, migration planning, gravity-kick separation, and the peculiar
motion density derivative with grad-H. The 8³ hydro-wave case passes comparison
with refined global integration and an exact restart. With self-gravity
(`units.massSolar=1e24`, `gravity.softening=0.25`), successive CFL refinements
reduce the field difference from 0.003743 to 0.001905; the baseline maximum
energy residual is 8.20e-6. These are small regression cases, not production
accuracy or speed measurements.

The short self-gravitating blast case also passes the refined-global comparison
and exact restart. Four of its nine hydro events advance one of 432 gas sinks;
the measured fine/full sorting and density-work ratios are both 1/432. Its
maximum energy residual is 1.22e-4. This demonstrates reduced work in those
operations, not an overall speedup.
A two-rank cosmological SPH plus hierarchical-gravity smoke run completes two
steps with finite physical diagnostics. This checks distributed execution, not
cosmological convergence or multi-rank restart equivalence.

The retained close-pair stress runner `tests/run_hierarchical_sph_gravity.py`
(512 gas particles, `units.massSolar=1e26`, `gravity.softening=0.018769884569535343`,
end time 0.05) failed before gravity gained the direct-pair timestep bound of
`docs/kernels.md`: the hierarchy energy residual was 0.09586 against the 0.002
gate, and the three global reference refinements gave 0.5009, 0.6536, and 0.1177
without converging. Full-force analysis of that failure (recorded in the earlier
`sph_hierarchical` work) showed the acceleration criterion selecting a step of
about 29% of a softened close pair's oscillation period, because the pair's net
force is small near its crossing while its spatial derivative is not. With the
bound, the same runner, parameters, and tolerances pass on CPU (2026-11-02,
`build/validation/sph_gravity_paircap`): hierarchy peak residual 4.87e-4 with
256 partial-active events and fine/full median sorting and density ratios 0.5;
global residuals 6.82e-4, 4.98e-4, and 4.85e-4 at eta, eta/2, and eta/4;
successive global refinements reduce the largest field difference from 0.003419
to 0.001042 (velocity distribution); the hierarchy differs from the quarter-eta
global result by at most 0.001576. The eta/4 global reference needed 151 s
single-threaded, beyond the runner's 120 s per-process cap, so that one run was
produced by the same `run_case` with a longer timeout and the committed runner
compared all four with `--analyze-only`. This is a timestep-resolution result at
fixed force settings; it is not a performance or production claim.

Cooling and star formation (CPU, Release, Fixed64, FMM, 2026-11-03): on the
homogeneous 8³ lattice the hierarchical and traditional integrators give the same
conversions and mean thermal energy for SPH and for MFM (`hopkins_thermal_sources`,
`star_formation_application`, and `gadget4_cooling_integration_hierarchical_cpu` and
its two-locality `_mpi` twin, all with exact restarts).
`hierarchical_sources_application` runs a 16³ cosmology from redshift 4 with
forced star-formation thresholds. Its SPH case closes part of the gas at 65 events,
forms 5 stars, restarts byte-exactly from a mid-step checkpoint taken after the first
conversion, and at the first full synchronization matches the traditional
integrator's 4 stars and mean stored thermal energy to 6e-5, with energy residuals of
-6.5e-5 and -7.1e-5. Its MFM case forms 6 stars, closes cells at gravity events before
their hydro endpoints, restarts exactly, and matches the traditional star count;
the mean thermal energy differs by 5.8%, within the 6.8% by which the two MFM
integrators already differ without sources. In a similar 16³ run that reaches two
synchronizations, the hierarchical SPH residual with sources was -2.3e-4 and -4.0e-4
while the expansion work used the synchronized thermal energy, and -4.4e-6 and 5.5e-7
with the all-gas sum (traditional: -9.7e-6 and -1.0e-5). A 32³ run of `configurations/star_formation/lcdm.ini` completed 18 hierarchical steps
with sources in a CPU build that disables the traditional and global integrators.
These cases exercise the coupling and its bookkeeping; they do not establish
converged star-formation histories.

CUDA execution, long production runs, and a performance advantage remain
unvalidated. Compilation, CPU execution, CUDA execution, numerical convergence,
and performance are separate validation claims.
