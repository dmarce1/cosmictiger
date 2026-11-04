# Conservative midpoint SPH and viscosity history

Authorized November 4, 2026. Baseline: c6c37211a on main.

The common SPH operator remains a midpoint hydro kick between chronological
half-drifts, independently of the gravity integrator. Chemistry, star formation,
and mesh-gravity/SPH combinations are explicitly deferred at the user's request.
Their obsolete predictor/corrector must not keep persistent gas force rates alive.

## Implementation

1. Restore Cullen–Dehnen, Price, and Hu viscosity using the divergence-history
   alternative described in Cullen & Dehnen (2010), Appendix B3. Sample physical
   divergence at the actual event time; preserve Price's background subtraction.
   Commit histories only after density convergence. Repeated same-time evaluations
   and output-only density preparation must not advance viscosity twice.
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
Gradient viscosity retains a committed divergence sample and its conformal time;
their finite difference replaces the previous acceleration-gradient estimator.
The SPH checkpoint block is version 70000 plus its existing model/physics suffix,
and old force-rate checkpoints are intentionally incompatible.

## Validation

Completed locally on November 4, 2026:

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
there is no bitwise global-conservation claim. No full repository suite, hosted
CI, long-duration scientific validation, performance benchmark, or deferred
chemistry/star/mesh-SPH execution was run.

## Sources

- Cullen & Dehnen (2010), Appendix B3: https://arxiv.org/abs/1006.1524
- Saitoh & Makino (2009), timestep limiter: https://arxiv.org/abs/0808.0773
- Pelupessy, Jänes & Portegies Zwart (2012), hierarchical splitting:
  https://arxiv.org/abs/1205.5668

Pair scheduling and implementation details above are this code's design, not a
claim that those papers implement this particular SPH scheme.
