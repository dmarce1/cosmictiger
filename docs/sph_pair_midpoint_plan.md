# Conservative midpoint SPH and viscosity history

Authorized November 4, 2026. Baseline: c6c37211a on main.

The common SPH operator remains a midpoint hydro kick between chronological
half-drifts, independently of the gravity integrator. The original pair-kick
implementation deferred cooling and star formation; the subsequent source
integration applies them at completed particle intervals, as described in
[star formation](star_formation.md). Mesh-gravity/SPH combinations remain deferred.
Source integration does not restore persistent gas force rates.

## Implementation

1. Restore Cullen–Dehnen, Price, and Hu viscosity using the divergence-history
   alternative described in Cullen & Dehnen (2010), Appendix B3. Sample canonical
   peculiar divergence at each particle's completed interval endpoint for all three models,
   with the cosmological derivative conversion used by GADGET-4.
   Commit histories only after density convergence and before gas rung changes.
   Midpoint force evaluations, semiactive measurements, output preparation and
   post-source bound estimates do not commit gradient history or evolve its alpha.
   All evolving-alpha models, including Morris–Monaghan, advance alpha only at
   completed own interval endpoints; subsequent midpoint forces use that alpha.
   Morris–Monaghan uses endpoint compression without retaining divergence history.
   Startup seeds gradient history where required and never advances alpha.
2. Remove persistent gas hydro acceleration, gravity acceleration, and entropy-rate
   arrays, including checkpoint and migration payloads. Kernels may retain nullable
   diagnostic outputs for focused tests. Production updates use event-local scratch.
3. Evaluate each hydrodynamic pair on the finer endpoint's midpoint cadence. One
   endpoint owns the pair; equal-rung ownership is ordered by rank and gas index.
   Accumulate one equal-and-opposite impulse and both entropy increments. Stage
   immutable sources on every rank before evaluating; route remote impulses before
   committing the event. No global momentum correction or persistent pair cache.
4. Enforce a one-rung difference across the union of both supports. Iterate staged
   candidate rungs through all affected particles, including inactive neighbors,
   until the global fixed point is reached in one enforcement phase at a common
   all-particle synchronization boundary. This is the user's chosen policy: no
   rollback and no changes to gas rungs between these boundaries. New kernel
   contacts between boundaries are not guaranteed to satisfy the cap immediately.
5. Traditional gravity defers its gas opening half-kick until closure, retaining
   the current acceleration only in event-local scratch. Hierarchical gravity
   selects gas rungs from the full top-level acceleration, the hydro bound, and
   the usual one-rung coarsening limit; it then closes the neighbor constraints
   before selecting the next correction subset. Gas rungs remain fixed through
   that descending ladder. This deliberately replaces repeated partial-force
   timestep selection for gas with the full-force criterion. Dark matter keeps
   its existing hierarchical selection. A common *all-particle* boundary is
   required so the top force includes every dark-matter source.

The three deleted persistent arrays are absent from checkpoints and migration.
Gradient viscosity retains one dimensionless divergence sample in the scalar
`Float16Storage* sphScaledDivergence` array. Its decoded endpoint difference
replaces the acceleration-gradient estimator. Sampling before rung changes makes
the completed interval available from the live rung, so no wrapper, previous-rung
byte, or per-particle timestamp is needed. The separate latest hydro rung bound
still serves deferred timestep selection at common synchronization boundaries.

Evolving alpha retains two bytes of binary16 storage. The common history is

\[
 G_n=\nabla_{\mathbf x}\cdot\mathbf w_n=a_n^2\theta_{{\rm pec},n},
 \qquad q_n=\delta\tau_n G_n,
\]

where \(\mathbf w=a\mathbf v_{\rm pec}\), \(\mathbf x\) is comoving, and
\(\delta\tau_n\) is the particle's own conformal interval duration. Before
each dyadic rung mutation, the live old rung identifies the stored normalization;
changing to the new rung rescales the scalar without taking another sample:

\[
 q_{\rm new}=q_{\rm old}\,2^{r_{\rm old}-r_{\rm new}}.
\]

The rescaled scalar is then decoded using the live new rung. Global steps instead
use the previous completed duration in shared driver state, with the base duration
as the startup fallback before a completed step exists. No per-particle clock,
exponent, derivative, or sampled rung is stored. The magnitude of \(q_n\) is not
assumed bounded by one and is not clamped to that range.

All viscosity sensors, including Morris–Monaghan compression, Balsara, and the
gradient-model limiters, use peculiar gradients. Actual pair approach, viscous
force/heating, and the compressive signal speed retain physical relative
velocities including Hubble flow. Following GADGET-4, the shock sensor differences
decoded canonical gradients before the current scale-factor conversion:

\[
 \mathcal D_n\simeq a_n^{-3}\frac{G_n-G_{n-1}}{\Delta\tau_n},
 \qquad a_n^4\mathcal D_n\simeq
 a_n\frac{G_n-G_{n-1}}{\Delta\tau_n}.
\]

Here \(\Delta\tau_n\) is the completed interval between endpoints; the second expression is the
scaled slope passed to the model. Its continuum limit is
\(D\theta_{\rm pec}/Dt+2H\theta_{\rm pec}\), as derived in
[the viscosity documentation](monaghan_sph.md#cosmological-viscosity-conventions).
The `velocityDivergence` output key reports decoded canonical \(G\), with explicit
metadata, because no previous-sample scale factor is retained.

The endpoint difference estimates the slope centered on the completed interval's
midpoint, but becomes available only at its end. Its updated alpha is used by the
following hydro kick: this accepted viscosity timing is first order. Pressure and
viscous pair forces are still evaluated at their scheduled midpoints. All evolving
alpha, including Morris–Monaghan, follows this endpoint cadence. With cooling or
star formation, the existing endpoint density/gradient pass supplies alpha and
any required history before sources. Otherwise, every evolving-alpha model
requires an additional endpoint pass.

The history occupies exactly two bytes per gas particle in memory, checkpoint
and migration payloads: one binary16 bit pattern (`H`). The
Balsara factor occupies one `uint8_t`, decoded by dividing by 255; rounding to
nearest bounds its absolute error by \(1/510\). Its evaluation and force/source
staging remain float. With constant α or Morris–Monaghan, enabling Balsara adds
the direct SPH divergence/curl sums to the current midpoint density pass. CD, Hu
and Price instead derive Balsara from the trace and antisymmetric part of their
existing matrix-corrected velocity gradient at startup and completed own endpoints,
then retain it until the next gradient reconstruction. Their midpoint passes
perform neither a separate direct sum nor a new gradient reconstruction. This
changes the estimator from the former uncorrected kernel-weighted divergence/curl
to the same corrected matrix used by their viscosity model.
The retained Balsara byte follows reordering, checkpoints and migration only
for CD, Hu and Price when enabled. For constant α and Morris–Monaghan it remains
transient and is recomputed after restart or migration.
Balsara defaults on for constant α and MM, and off for CD, Hu and Price, which
already include their own shear limiters. Explicitly enabling the additional
factor for a gradient model prints a warning and preserves the user's setting.
The extra factor changes forces, including Price's independent beta contribution;
see the [source-backed comparison](monaghan_sph.md) for the distinction.
With all three arrays enabled, history, alpha and Balsara
therefore use \(2+2+1=5\) bytes per gas particle, down from \(8+2+4=14\).
Constant viscosity allocates no history or evolving-alpha array.

Target alpha, inverse decay time, limiter and pending divergence are four float
diagnostics, sixteen temporary bytes per gas particle during a history-updating
density event. The redundant pending model-sampled divergence is removed. These
buffers are allocated after sorting, released before source conversion, and do
not migrate or enter checkpoints.

Convergence, semiactivity and the proposed rung share one `SphParticleFlags`
byte instead of three separate byte arrays. Bits 0 and 1 hold the two flags;
bits 2–7 hold the proposed rung, with 0–15 currently valid and no unset sentinel.
Other codes are rejected. Explicit setters preserve the other bits when a field
changes; individual bits cannot supply mutable references. The byte is stored
after density and before viscosity data, and is also migrated directly. This
saves two bytes per gas particle in memory, checkpoints and migration.
The SPH checkpoint block is version 130000
plus its existing model/physics suffix. Earlier gas payloads, including base
90000 and the intermediate bases 100000, 110000 and 120000, are intentionally incompatible.

## Validation

The subsequent sentinel cleanup keeps this same one-byte layout. Proposals are
always nonnegative, with zero as the initial bound; the former code 63 is rejected.
Fresh CPU/CUDA builds, 45 CPU and 27 CUDA kernel checks, 21 Python checks both
normally and with `-O`, two CPU checkpoint/gravity tests, the explicit CD+Balsara
CUDA checkpoint/reorder fixture, and three gradient-model two-rank migrations
plus the empty-gas-rank fixture passed after this cleanup. Evidence is in
`/tmp/sph_no_sentinel_*.log`.

The final packed-flag/base-130000 implementation completed CPU and CUDA builds
with exit status zero on November 4, 2026. All 45 selected CPU checks, 27 CUDA
kernel checks and five Balsara checks passed. The Python suite passed all 21
tests under both normal execution and `python3 -O`, including all 64 valid
packed flag/rung combinations and all 192 combinations containing an invalid
rung code. Fifteen focused Balsara option cases passed; the sixteen preset
edits were statically audited, not individually run as simulations.

All ten selected CPU application/lifecycle CTests passed, covering CPU and MPI
source completion, three unit selections, gravity, checkpointing, hierarchical
stepping, retained octrees and trees. The octree fixture's shared-global-duration
normalization was corrected before the successful final run, recorded in
`/tmp/sph_packed_flags_cpu_applications.log`. Both selected CUDA application
CTests passed, as did an explicit CD-plus-Balsara CUDA fixture covering
checkpointing, reordering and mixed rungs.

Sixteen CPU restart combinations and rejection of an old base-90000 checkpoint
passed. Each of the three gradient models passed history migration on two ranks,
and the tree fixture passed with an empty gas rank. These checks validate the
focused storage, option, kernel and integration paths exercised. They are not a
full repository test suite, hosted-CI result, performance benchmark or
long-duration scientific convergence study.

Earlier base-120000 checks exercised the endpoint-history/Balsara implementation
before packing; the final results above supersede that intermediate validation
for the paths rerun. The historical records below describe older representations.

The intermediate base-100000 Python checkpoint/runner suite passed 19 tests under both normal
Python and `python3 -O` on November 4, 2026. Fixtures cover the base-100000 layout
across particle widths, integrators, source models and viscosity selections,
truncation, corrupt half coefficients/history, invalid sampled rungs, and exact
preservation of finite half values including signed zero and subnormals. These
are historical parser and runner checks for the superseded sampled-rung layout;
they do not validate the scalar endpoint layout, C++ application or scientific accuracy.

The following signed-significand/base-90000 results are historical validation of
commit `1bc12b5f6`. That follow-up passed fresh Release CPU and CUDA
application and test builds on November 4, 2026. All 64 selected CPU CTests
passed (45 SPH units, 15 viscosity/storage units, and four lifecycle fixtures,
including a two-rank source fixture). All 28 selected CUDA checks passed,
including the production GPU kernels and source-history fixture.

Storage units cover all 254 normal binary32 exponents, both signs, canonical
zero/subnormal handling, nonfinite and malformed encodings, and unchanged
own-midpoint sample intervals. Ten viscosity/Balsara restart combinations and
six additional global/hierarchical gradient-model restarts passed with exact
committed histories and the existing velocity-roundoff allowance. The actual
checkpoint reader rejected ten corrupt history encodings and a base-80000
checkpoint. Two-rank signed-history migration passed separately for CD, Price,
and Hu; own-midpoint history also passed with an empty gas rank.

Historical evidence is in `/tmp/sph_int16_{cpu_build,cuda_build,cpu_tests,cuda_tests,
extra,mpi}.log` and `/tmp/sph_int16_extra/evidence.json`. This is focused local
validation: the full options suite, complete source-application matrix, full
repository suite, hosted CI, long-duration accuracy, and performance were not
rerun for this storage follow-up.

The following compact-history/source results are historical validation of
commit `9ccd34a85`, with scaled binary16 divergences and checkpoint base 80000.
They do not validate either subsequent compact-history representation.

That November 4 update passed fresh Release CPU and CUDA application builds.
Focused units passed 75 CPU cases (one additional
GPU-only cooling case skipped) and 30 CUDA cases. They include exhaustive finite
binary16 round trips, scaled divergence range checks, changing-rung/global
sample intervals, and production CPU/CUDA viscosity comparisons.

Seven CPU lifecycle fixtures, ten viscosity/Balsara restart combinations, and
six additional gradient-model restarts across global and hierarchical stepping
passed. Committed history remains exact across those restarts; only velocity
accumulation has the existing bounded float allowance. Ten corrupt history
encodings were rejected by the actual checkpoint reader. The own-midpoint and
scratch-lifetime fixture also passed with two MPI ranks, one without gas.
Retained-octree evolution/restarts passed on CPU, CUDA and two MPI ranks, and MPI
migration passed. The complete grouped option suite passed, including eighteen
binary16 storage-bound cases. See `star_formation.md` for the source tests.

Evidence logs are `/tmp/sph_compact_cpu_units.log`,
`/tmp/sph_compact_cuda_units.log`, `/tmp/sph_compact_cpu_lifecycle.log`,
`/tmp/sph_compact_extra.log`, `/tmp/sph_compact_tree_mpi.log`,
`/tmp/sph_compact_tree_cpu_mpi.log`, `/tmp/sph_compact_tree_cuda.log`, and
`/tmp/sph_compact_options.log`. These are focused local regressions, with no
full-suite, hosted-CI, long-duration accuracy, or performance claim.

The following validation was completed locally for the original pair-kick
implementation on November 4, 2026, before the source and compact-history
follow-ups described above. It does not establish validation of those follow-ups:

- CPU application and unit targets compile/link; all 45 SPH kernel tests and
  seven viscosity tests pass. Coverage includes unequal-rung reciprocal impulses,
  inactive coarse updates, unique ownership, directed Price heating, and history
  time/scale conversions. A GCC fast-math issue that evaluated unused SIMD stack
  lanes was fixed narrowly in the 3x3 velocity-gradient reconstruction. All 134
  stack-padding layouts of the reproducer pass with floating-point traps enabled.
- CUDA application and unit targets compile/link; all 27 SPH GPU tests pass on
  the RTX 3090, including CPU/GPU pair parity. Six production CUDA/MPI checks
  pass: gravity coupling, hierarchical evolution on GPU and two ranks, octree
  evolution/restart on GPU and two ranks, and migration.
- CPU hierarchical and octree application checks pass. Gravity-free compilation
  and both application/tree checks pass. Tree fixtures exercise chains needing
  six changing passes followed by one unchanged pass, periodic contacts,
  cross-rank propagation, and a locality without gas.
- All ten viscosity/Balsara combinations pass evolution, restart, committed-history
  comparison, and compatibility rejection checks. Wave restarts require exact
  nonvelocity payloads and clocks; only tiny transverse velocity cancellation
  residuals receive a componentwise float-roundoff allowance.
- Retained-tree versus rebuilt-tree blast restarts pass bounded-roundoff checks.
  Very cold gas can differ by 52–54 local entropy ULPs while the global normalized
  entropy error remains below two float epsilon units. The blast-only comparison
  allows 64 local float epsilon units for entropy and two alpha ULPs, alongside the tighter global
  bound; discrete fields and viscosity history still compare exactly.
- Self-gravity, cosmological restart, and thermal boundary-scale regressions pass.
  Smooth inviscid pressure temporal refinement gives position-error orders
  2.0012 and 1.9867. These are time-convergence results at fixed particle sampling,
  not energy-error orders or a spatial SPH convergence claim. Viscosity is not
  required to retain second-order time accuracy.
- Startup and grouped CLI/INI tests pass, including explicit rejection of the
  deferred combinations. All 16 complete presets include the default
  `compute.sphTreeRebuildFraction = 0.3333333333333333`; the inclusive one-third
  threshold is checked against all particles, not only gas.

Evidence logs are `/tmp/sph_pair_cpu_unit_validated.log`,
`/tmp/sph_pair_viscosity_validated.log`,
`/tmp/sph_pair_cpu_stack_padding_validated.log`,
`/tmp/sph_pair_cuda_unit_validated.log`,
`/tmp/sph_pair_cuda_mpi_evolution_validated.log`,
`/tmp/sph_pair_viscosity_restart_validated.log`,
`/tmp/sph_pair_cpu_evolution_final.log`, `/tmp/sph_pair_nogravity_ctest.log`,
and `/tmp/sph_pair_options_{startup,grouped}.log`. Self-gravity evidence is in
`build/molecular_hot_cuda/tests/sph_hierarchical_gravity/evidence.json`.

Momentum is bounded by floating-point accumulation and float velocity storage;
there is no bitwise global-conservation claim. That baseline validation included
no full repository suite, hosted CI, long-duration scientific validation,
performance benchmark, or chemistry/star/mesh-SPH execution.

## Sources

- Cullen & Dehnen (2010), Appendix B3: https://arxiv.org/abs/1006.1524
- Saitoh & Makino (2009), timestep limiter: https://arxiv.org/abs/0808.0773
- Pelupessy, Jänes & Portegies Zwart (2012), hierarchical splitting:
  https://arxiv.org/abs/1205.5668

Pair scheduling and implementation details above are this code's design, not a
claim that those papers implement this particular SPH scheme.
