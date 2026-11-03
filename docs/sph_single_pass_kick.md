# Single force pass per SPH step: scope

Status, November 3, 2026 (implemented without profiling, at the user's request):

- **Option A.** Gravity is joined before the closing pass. The pass runs with
  `phase = 1` and writes its rungs to `sphParticlesProposedRungs` (the
  `SphRunData::sinkProposedRungs` sink). `sphParticlesCommitProposedRungs` moves them
  into `particlesRung` before the limiter. Steps whose cooling or star formation
  changed the gas repeat the pass after the sources (`hydro.force.sources`), because
  review showed the pre-source rates can drive `u` negative in the opening kick when
  cooling removes most of the thermal energy. Adiabatic runs keep one pass. `deferAccelerationSources` and `sphParticlesCompleteForces` remain, with
  their unit test, but have no production caller.
- **Option B, storage part.** The predictor no longer kicks. Staging predicts active
  velocities and energies from the stored rates, and the closing kick applies the
  fresh rates once. `sphParticlesPredictedAccelerations` and
  `sphParticlesPredictedInternalEnergyRates` are gone from storage, migration and
  checkpoints (gas payload versions from 20000). Because staging reads rates that the
  force kernels overwrite, the multi-locality force pass stages on every locality
  before any locality computes (`sphRunStage`/`sphRunCompute`). The closing and
  opening kicks stay separate: cooling, star formation and the rung choice need the synchronized state
  between them. Inactive neighbours are still staged unpredicted.
- **Energy sums.** The gravity walks no longer sum gas kinetic energy or momenta. The
  opening kick (or a sum-only phase on closing-only steps) sums the gas at the
  synchronized state.
- **Close-pair timestep bound removed** (user decision, same day): `gravity::pairTimestep`,
  the per-sink pair curvature in every direct-pair kernel, and its SPH carrier
  `sphParticlesGravityTimesteps` are gone; gravity and hydro use the acceleration
  criterion only.
- **Hierarchical SPH removed.** SPH uses the traditional or global integrator;
  hierarchical integration is collisionless only.

The sections below are the original scope.

## The current step

One active SPH step in `sphStep2` (`src/core/driver.cpp`), closing the interval
that opened at \(t_n\) and opening the next one at \(t_{n+1}\):

| Order | Operation | Location | Rates used |
|---|---|---|---|
| 1 | Predictor: close the half-kick with the stored rates; save them | `sphParticlesApplyUpdates(…, 0, …)`, driver.cpp:295 (or :998 on the FMM overlap path) | \(a_n, \dot u_n\) |
| 2 | Density and support iterations | `prepareDensities` | |
| 3 | **Hydro force pass 1** (`hydro.force.close`), concurrent with FMM gravity | `sphRun`, phase 0 | gives \(a^\ast_{n+1}\) at the predicted state |
| 4 | Join gravity; add gravity, central acceleration, damping | `sphParticlesCompleteForces` | |
| 5 | Corrector: \(v \mathrel{+}= (a^\ast_{n+1}-a_n)\,\Delta t/2\), same for \(u\) | `sphParticlesApplyUpdates(…, 1, …)` | |
| 6 | Chemistry close, star-formation close (may convert gas and redo density) | driver.cpp:514–557 | |
| 7 | Energy diagnostics at the synchronized state | `particlesSumEnergies` | |
| 8 | **Hydro force pass 2** (`hydro.force.open`): full pair forces again, then CFL/acceleration/gravity timesteps; writes new rungs | `sphRun`, phase 1 | gives \(a_{n+1}\) at the corrected state |
| 9 | Rung limiter loop | `kSphRunRungs` | |
| 10 | Opening half-kick with the new rung's \(\Delta t'\) | `sphParticlesApplyUpdates(…, 2, …)` | \(a_{n+1}\) |

Both passes are full pair-force evaluations. Each writes `sinkAccelerations` and
`sinkInternalEnergyRate` (`sph_hydro_cpu.cpp:238`, `:252`). Only pass 2 computes
timesteps (`if (parameters.phase == 1)`, `sph_hydro_cpu.cpp:253`). The hierarchical
path does the same: `kSphRunHydro` runs at `hierarchical.cpp` lines ~914 (phase 0)
and ~953 (phase 1), with the source commit between them.

Git history doesn't say why pass 2 exists; it dates from the 2022 code (`ec03cf8a2`,
`1f26c55ab`, "Fixed bug"). `docs/monaghan_sph.md` states only that it "uses the
corrected state". The reference integrator in `tests/sph/cpu.cpp:2086`
(`step`) does **not** have it: kick, drift, predicted kick, one `evaluate()`,
correct, and the next opening kick uses the corrector's rates.

## What it buys and what it costs

Pass 2 gives three things:

1. **Opening rates from the corrected state.** The opening kick uses
   \(a(v_{n+1},u_{n+1})\) instead of \(a(v^\ast_{n+1},u^\ast_{n+1})\). The two
   differ by \(O(\Delta t)\) in the rates, so the opening kick differs at
   \(O(\Delta t^2)\) in velocity, the same order as the leapfrog's local error.
2. **Timesteps from the post-source state.** Cooling and star formation (step 6)
   change \(u\) and therefore \(c_a\) before the CFL condition is evaluated.
3. **Correct forces after star formation removes gas.** When stars convert, the
   driver redoes density (driver.cpp:530–555) and pass 2 then sees the new
   neighbour sets.

The cost is one extra full pair-force pass per active step: neighbor staging,
remote cache-line fetches, the pair loop, and on CUDA a second launch and
staging round. The density iterations and pass 1 are unchanged. So the hydro
force cost per step roughly doubles.

**First task: measure it.** `tests/run_phase_profile.py` already reports
`hydro.force.close` and `hydro.force.open` as separate rows. Run it on a
traditional SPH scenario (Sod, Sedov, Gresho) and on the cosmology gas case,
CPU and CUDA, within the 12-core budget. The saving is about the
`hydro.force.open` row, minus whatever pass 1 loses by no longer overlapping
gravity. If that row is a small share of the step, stop here.

## Option A: drop pass 2 (recommended)

Keep the predictor and corrector and their stored arrays. Remove the second
force pass. The opening kick uses the corrector's rates, as in
`tests/sph/cpu.cpp`.

### Where the timestep goes

Overlapping SPH with gravity is not a requirement for now (user decision,
November 3, 2026). So join gravity **before** pass 1 instead of after it:

- On the FMM path (driver.cpp:998–1021), call `joinGravity()` in `sphStep2`
  just before `hydro.force.close` and run pass 1 with
  `deferAccelerationSources = false`. Density and support iterations can still
  overlap gravity, since they never read it; that is free to keep and free to
  drop.
- Pass 1 then has the gravitational acceleration. It completes the acceleration inside the
  kernel (`sphCompleteAcceleration`, as pass 2 does today), and
  `sphParticlesCompleteForces` drops out of this path.
- The phase-1 timestep block (`sph_hydro_cpu.cpp:253–345`, and the matching
  block in `sph_hydro.cu`) runs in pass 1 unchanged and writes the new rungs.
  Its `SphRunReturn` diagnostics and `minimumTimestep` come from pass 1.

The one input that is now stale is \(u_a\) in the CFL and \(u/|\dot u|\) terms:
pass 1 sees the predicted \(u\), and chemistry and star formation (step 6)
change it afterwards. Two choices:

1. **Accept it (default).** Cooling lowers \(c_a\), so the pass-1 timestep is
   conservative for cooling gas. The case that matters is heating that raises
   \(c_a\) a lot within one step.
2. **Re-check after sources, only when chemistry or star formation is on.**
   Pass 1 stores \(\max_b(-\mu_{ab})\) per sink in a scratch array (not
   migrated or checkpointed). A per-particle loop after step 6 recomputes
   \(c_a\) from the current \(u_a\), recomputes \(1/\delta t_{cv}\) and the
   \(u/|\dot u|\) bound, and can only *raise* the rung.

Start with choice 1 and add choice 2 if the chemistry and star-formation timestep
tests show a difference. The rung limiter (step 9) is unchanged. Pass 1 now
writes rungs before the corrector and the sources run. Check that nothing in
steps 5–7 reads `particlesRung` expecting the old rung. Something does: the
corrector forms its half-step from `sphParticlesRung(gasIndex)`
(`sph_particles.cpp:133–136`), and the chemistry and star-formation closes
integrate over the closing interval. So pass 1 writes its proposals to a scratch
`char` array instead of `sinkParticleRungs`, and they are committed to
`particlesRung` just before the rung limiter. The limiter reads neighbours' rungs
and must see the new ones.

### Star formation

Keep a force pass only when `update.converted != 0`. In that case the density
is redone anyway, and the opening rates must not include pressure from gas that
is now a star. Conversions are rare per step, so the common path gets the saving.

### Files

| Area | Change |
|---|---|
| `src/core/driver.cpp` FMM path (~998–1021) | Join gravity before pass 1; overlap density only, or nothing |
| `src/core/driver.cpp` `sphStep2` | Pass 1 runs with `deferAccelerationSources = false` and computes timesteps; drop `sphParticlesCompleteForces` from this path; commit staged rungs before the limiter; remove the phase-1 `sphRun` except after conversions |
| `src/hydro/sph/hierarchical.cpp` (~lines 900–960) | Same change around `runEverywhere(phase 0/1)`; check the source commit and refresh order |
| `src/hydro/sph/sph_hydro_cpu.cpp`, `sph_hydro.cu` | Run the timestep block in the force pass that closes the step; write rungs to the scratch array |
| `cosmictiger/hydro/sph/sph_kernel.hpp` | `SphRunData` gains a proposed-rung sink pointer (plus \(\mu_{\max}\) if the post-source re-check is added) |
| `src/hydro/sph/staged_run.cpp` | Stage the new sink array(s) for CUDA |
| `cosmictiger/hydro/sph/sph.hpp`, kernels, `tests/sph/source_completion_cases.hpp` | `SphRunParameters::deferAccelerationSources` and `sphParticlesCompleteForces` lose their only production caller (driver.cpp:478–501). Remove them with their test, or keep them for a later re-overlap |
| `docs/monaghan_sph.md`, `docs/hierarchical_sph.md` | Remove "the opening force selects the next rung" wording |
| `docs/flop_accounting.md` | Recount the moved timestep flops |

Migration (`domain.cpp:579–716`) and the checkpoint layout
(`visitSphCheckpointArrays`) **do not change**, because the scratch array is
rebuilt every step.

### What to watch

- **Results change at truncation order.** Every scenario regression baseline
  will move slightly. Compare convergence (Sod L1 error versus resolution) and
  energy conservation drift, not bitwise output.
- **Timesteps get slightly less conservative.** The signal velocity comes from
  pass 1's predicted velocities. The `previousRung - 1` limit and the rung
  limiter still apply.
- **Lost overlap.** Pass 1 no longer runs alongside FMM gravity. Measure
  `hydro.gravity.overlap` before and after. The net saving is pass 2's cost
  minus pass 1's lost overlap with gravity.
- **Viscosity evolution.** Morris–Monaghan updates α in phases 0, 1 and 2, and
  that is unchanged. Cullen–Dehnen reads accelerations in its density-time
  exchange (`cullen_and_dehnen/particle_exchange.cpp`). After this change those
  are pass 1's completed accelerations instead of pass 2's. Check the A15
  indicator.
- **Global-timestep and PM outer-interval modes.** `selectedTimestep` currently
  takes `result.minimumTimestep` from pass 2. It must come from the new loop, and
  `closingScale` versus `t0` (driver.cpp:284) must stay as it is.

## Option B: GADGET-style predicted copy (later, optional)

Stop changing `v` and `u` before the force pass. Fill scratch arrays
`vPred = v + a·(t - t_kick)` and `uPred` for every staged source, read only
those in the force pass, then apply one kick per particle:
\(v \mathrel{+}= a_{n+1}(\Delta t + \Delta t')/2\), as gravity does.

Beyond Option A, this:

- removes the corrector arithmetic and the "undo" step;
- replaces `sphParticlesPredictedAccelerations` and
  `sphParticlesPredictedInternalEnergyRates` with per-step scratch, so they leave
  migration and checkpoints (a checkpoint layout change);
- **makes inactive neighbors time-consistent.** In traditional mode,
  `sphParticlesGlobalReadVelocities` (`sph_particles.cpp:718`) and the
  internal-energy reader stage an inactive neighbor's stored values. Those are
  its *post-opening-kick* half-step values, not values predicted to the current
  time. The hierarchical path already predicts sources (`liveSource`,
  `predictSource`). Confirm this before claiming it as an accuracy gain.

Costs: the energy diagnostics need the synchronized \(v_{n+1}\). Either compute
it on the fly as `v + a·Δt/2` or keep a split kick at that point. Every reader of
`particlesVel` for gas during the step (diagnostics, star formation, Silo output,
lightcones) must agree on which velocity it means. Option B is a larger change;
only consider it after Option A has been measured.

## Tests

Existing tests that cover this:

- `tests/sph/cpu.cpp` reference integrator: already single-pass, and should
  match more closely afterwards.
- `tests/sph/no_gravity.cpp`, `tests/sph/gravity.cpp`, `tests/particle_mesh_sort.cpp`:
  predicted-rate migration and checkpoint round trips. Unchanged under Option A;
  rewritten under Option B.
- `tests/sph/cuda.cu` `SelectedKernelsGravityTimestepsAndCpuParity`: needs to
  move to the new timestep loop.
- `tests/run_phase_profile.py` line 209 requires
  `calls("hydro.force.open") == calls("hydro.kick.open")`. Change it to allow
  `hydro.force.open` only on conversion steps.
- `sph_hierarchical_sources`, `sph_hierarchical_application`,
  `sph_hierarchical_blast`, `sph_migration(_mpi)`, `smoke_sph_*`,
  `sph_chemistry_timesteps_*`, `hydro_methods`: run all of them.

New tests:

- A test that the closing force pass gives the same rung proposals as the old
  pass 2 when both see the same state (CPU and CUDA). Also check that the
  corrector's half-step still uses the closing rung.
- A conversion step: verify that the conditional force pass runs and the opening
  kick ignores converted particles.
- A Sod or Sedov convergence comparison against the current code, using
  documented L1 errors rather than a bitwise baseline.
