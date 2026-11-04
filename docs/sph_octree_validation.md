# Midpoint implementation validation

November 3, 2026, branch `codex/sph_octree`. The midpoint changes follow commit
`665da4279`; the older shared-leaf results below are historical only.

## Builds and kernel checks

Release CPU and CUDA FMM/SPH applications compile and link, as does the CPU
SPH-only application with gravity disabled. All 41 CPU and 23 CUDA SPH unit
tests pass, including exact-rung targets with inactive neighbors, absent gas
gravity buffers, hydro-only timestep constraints, and Morris–Monaghan evolution
over the actual midpoint interval. The CUDA tests executed on the GPU.

The CPU gravity/SPH kick fixture, checkpoint-state fixture, retained-tree fixture,
and Sod smoke test pass. The CUDA gravity/SPH kick fixture passes. Seventeen
Python regression tests pass. A separate SPH-only traditional Sod run completes,
and its global hydro regression passes raw-step, restart, endpoint, and corrupted
clock checks.

The CPU build with only the hierarchical integrator enabled also compiles,
links, and passes the hierarchical application and restart regression.

## Evolution, integration order, and restart

Global and traditional retained-octree wave and mixed gas/DM cases pass on CPU
and CUDA, including their canonical checkpoint restart comparisons. The CPU
global regression additionally verifies that `run.minRung=20` does not quantize
global timesteps: hydro-only and mixed FMM/hydro select 0.0122166788206, while
FMM-only selects 0.000162290700246 in the fixture. Restart and endpoint clipping
pass. Zero/NaN gravity and hydro intervals and a nonzero global integer phase
are rejected after recalculating checkpoint checksums.

The CPU hierarchical M&M blast passes exact midpoint-lattice checks, nonuniform
rungs, positive entropy, uniform-rung agreement with traditional integration,
and restart with bounded accumulation roundoff. The same application passes on
two MPI ranks with one HPX thread per rank. The separate distributed tree fixture
passes with two ranks and two threads per rank, exercising both domain-boundary
crossings, inactive prefixes, and threshold rebuilds with inactive particles
outside their owning domains.

The CUDA hierarchical application passes with explicitly bounded numerical
restart comparisons. Seven of 3,456 particles differ by at most two fixed32
units per coordinate; their measured velocity differences reproduce those
position differences exactly through the two rounded half-drifts. The largest
accumulated-field difference is 4.082 float epsilons of its field scale. The
GPU-only regression permits two position units and eight float epsilons; CPU
positions remain exact and CPU accumulation tolerance remains four epsilons.
Entropy uses a local two-epsilon bound. Clocks, rungs, hydro proposals, flags,
and M&M coefficients remain exact. Position matching is unique and periodic;
mutation probes reject a three-unit displacement or altered discrete state.

The runtime rebuild-fraction regression passes on CPU with all three
integrators, including the default, zero/one endpoints, CLI/INI precedence,
invalid values, retained/rebuilt event decisions, and the all-particle
denominator in mixed gas/DM cases.

The undamped, inviscid 512-particle wave uses fixed rungs and a common final time
0.05. Successive position differences under timestep halving are:

| Comparison | RMS position difference |
| --- | ---: |
| First refinement | 3.36936e-6 |
| Second refinement | 8.41589e-7 |
| Third refinement | 2.12387e-7 |

The ratios are 4.00357 and 3.96253, giving orders 2.0013 and 1.9864. Particle
correspondence uses unchanged transverse lattice columns. This demonstrates
second-order inviscid pressure evolution; viscosity and relaxation damping are
explicit first-order updates and are not used to claim second-order convergence.

The self-gravitating star has maximum gravity/hydro force ratio 0.898807 and
natural rungs 11–12. Its damped fixed-rung cases keep relative energy change
below 7.14e-4 and net momentum fraction below 2.27e-8. Uniform hierarchical and
traditional results agree exactly in this fixture. The scenario's fixed
viscosity overrides are canonicalized before checkpoint allocation/serialization;
starting with the default M&M option now passes checkpoint and exact restart.

The mixed cosmology case exercises an actual promotion from rung 9 to 10 with
64 of 128 particles on the finer rung, expansion, and exact checkpoint 7→8
restart. Separate gamma=4/3 cases in both hierarchical and traditional modes
check the boundary entropy-to-energy conversion: measured relative error is
5.8331e-8, versus 1.66677e-4 with the obsolete midpoint scale factor. Complete
self-gravity evidence is `/tmp/sph_midpoint_gravity_validated/evidence.json`.

## Storage and remaining performance limits

The old 64-byte hierarchy predictor state and future-position restoration buffer
are removed. Midpoint gas uses no persistent gravitational acceleration. Its
existing proposed-rung byte persists the hydro constraint, and global integration
adds one driver double. The retained octree still adds one 24-byte `Fixed32Range`
per node. Gas payload version 60000 intentionally rejects older gas checkpoints.

This implementation drifts all particles chronologically at each half-drift.
It trades hierarchical drift work for removal of predictor clocks and restoration
storage; no large-run speedup or scaling claim is made. Chemistry, stars, PM/P3M,
and higher-order viscosity integration were outside this implementation's scope.

Evidence logs and output directories use the `/tmp/sph_midpoint_*` prefix, with
GPU application evidence also under `build/cuda/tests/`.

# Historical shared-leaf SPH octree validation

November 3, 2026. Branch `codex/sph_octree`, based on `dc0d3f0d9`.
See [the design and implementation notes](sph_octree_plan.md).

## Builds and focused checks

Local Release builds use FMM potential degree 4, fixed32 positions, and disabled
chemistry/stars. CPU and CUDA applications and the SPH kernel test executables
compile and link. A CPU SPH-only build with all gravity disabled compiles, links,
and completes a short Sod run. An FMM/SPH build with only the hierarchical
integrator enabled compiles, links, and passes the hierarchical application test.

The 41 CPU SPH kernel tests pass. The two new CUDA checks run on the GPU and pass:
arbitrary gravity-leaf target mappings agree with the existing CPU/CUDA operator,
and the separately returned hydro force survives a gravitational component of
`1e20` without subtractive cancellation. The 82 build-graph regression checks
and the configured CPU source-graph check pass.
The full grouped-option regression also passes with hierarchical FMM/SPH:
CLI/INI precedence, checkpoint metadata, output units/frames, parameter bounds,
and rejection of unsupported source combinations.

Focused application checks pass for the tree, SPH gravity, gas checkpoint state,
and Sod smoke test. Tree fixtures exercise the inclusive one-third threshold,
signed and periodic drifts, support growth, retained membership, selective-rung
drifts, invalidation, and node serialization.

## Integration and distributed checks

`tests/run_sph_octree.py` exercises global and traditional wave and mixed gas/DM
evolution, lifecycle decisions, and exact canonical checkpoint restart. These
cases pass with CPU gravity/SPH and CUDA gravity/SPH. Global wave and mixed cases
also pass with each mixed CPU/CUDA backend combination.

The two-rank fixtures use one worker per rank. They verify retained neighbors
against direct distances after gas crosses both rank boundaries, migration of
hierarchical clocks without changing inactive rung prefixes, and a threshold
rebuild with 6/16 active particles while ten inactive particles lie outside their
owners' domains. The mixed traditional MPI evolution and restart pass.

`tests/run_sph_hierarchical.py` passes on CPU, CUDA, and two MPI ranks. Its blast
uses rungs 1–5, retains the tree on eleven ladder events after one initial rebuild,
and exercises up to 3,455 sources stored at future drift endpoints. It checks
positive entropy, source clocks, and uniform-rung agreement with traditional
integration. Restart keeps positions, entropy, entropy rates, rungs, flags, and
clocks exact; rebuilding the tree can reorder floating-point sums. Only velocity,
support, density, force, and velocity-predictor fields permit bounded roundoff,
at four float epsilons times the corresponding field scale. The scripts write
the measured differences to `evidence.json`.

## Self-gravitating gas

`tests/run_sph_hierarchical_gravity.py` uses a gas sphere with comparable gravity
and pressure forces: maximum gravity/hydro ratio 0.904, natural rungs 11–12.
At a common physical time, forced rungs 4, 5, and 6 converge toward rung 7:

| Rung | RMS position error | RMS velocity error |
| --- | ---: | ---: |
| 4 | 2.2841e-8 | 1.4271e-11 |
| 5 | 4.2183e-9 | 3.0268e-12 |
| 6 | 1.4274e-9 | 5.8629e-13 |

In this damped stage, the relative energy change remains below 0.072%, and the
net-momentum fraction below 3.24e-8. At uniform rung 5, the traditional comparison
has RMS position/velocity differences 4.76e-11 and 1.83e-15. A complete run with
minimum rung 4 checks the terminal closing ladder and final diagnostic pass.

The same regression advances 64 gas and 64 dark-matter particles through eight
cosmological events. It exercises a gravity rung merge from 9 to 10 on event
three, expansion from scale factor 0.02 to 0.02002668, and the corresponding
changing inverse drift factors. Gas clocks, positive entropy, and nonzero
gravitational forces pass their checks. Restart from event seven reproduces
event eight's complete physical and stage state exactly.

## Runtime threshold follow-up

After adding `compute.sphTreeRebuildFraction` and including equal-cost gravity
rung promotions, the CPU application was rebuilt and linked successfully.
Ten focused CTests pass: six rung-merge unit cases (including 4,095 exhaustive
four-rung cost comparisons), the rung-merge application, SPH tree/serialization,
hierarchical hydro/restart, and hierarchical self-gravity/cosmology/restart.

`tests/run_sph_rebuild_fraction.py` passes on one rank with all three integrators,
and on two MPI ranks with hierarchical integration. It checks help and the
default, CLI/INI precedence, accepted fractions, rejection of nonfinite and
out-of-range values, and the actual tree lifecycle. The four-event blast has
3,456 gas particles; the mixed case verifies the denominator is all 128
particles (64 gas plus 64 dark matter), including across ranks.

| Integrator | Fraction 0: rebuild/retain | Default 1/3 | Fraction 1 |
| --- | ---: | ---: | ---: |
| Hierarchical, one or two ranks | 5/0 | 1/4 | 1/4 |
| Traditional, one rank | 4/0 | 1/3 | 1/3 |
| Global, one rank | 4/0 | 4/0 | 4/0 |

Evidence: `build/sph_rebuild_fraction_cpu/evidence.json`,
`/tmp/sph_octree_fraction_mpi/evidence.json`, and
`/tmp/sph_octree_fraction_{build,core_tests,rung_ctest,cpu,mpi}.log`.
CUDA and the full grouped-option suite were validated before this host-side
follow-up, not rerun for it; the new focused script covers its CLI/INI behavior.

## Limits

These are focused local checks, not the entire repository suite or hosted CI.
Performance, peak transient memory, long cosmological evolution, and fixed64
runtime have not been benchmarked or established by these checks. CUDA uses
temporary launch-range gravity storage followed by grouped hydro kernels.
Hierarchy-only predictor records add 64 bytes per gas particle in these builds;
removing the persistent gravity vector is not a net-memory saving claim for
hierarchical integration. The existing active-only neighbor limiter does not
wake inactive gas midinterval. Chemistry, stars, PM, and P3M are outside this
implementation's scope.

Gas checkpoint payloads use base version 50000, plus 1000 for hierarchical state
and the existing physics suffixes. Earlier gas payloads are rejected.
