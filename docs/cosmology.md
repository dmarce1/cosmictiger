# Cosmological background evolution

The background evolves through maximum expansion with a signed physical
derivative, `adot = da/dt`. Conformal time obeys `dt = a dτ`; it increases
during both expansion and contraction. Particle momenta retain the existing
convention `p = a v_peculiar`, so a drift uses `∫dτ/a = ∫dt/a²`.

## Equations and adaptive integration

The solver is adapted from `dmarce1/experimental` commit
`7ba8f1829fdede7348a1418b7cb36c0e7a955302`, with its original attribution and
license retained. Define

```
P(a) = Ωr + Ωm a + Ωk a² + ΩΛ a⁴
u = a adot / H0
```

The constraint is `u² = P(a)`. With dimensionless conformal time `q = H0 τ`,
the equations are `da/dq = u` and `du/dq = P'(a)/2`. With dimensionless
physical time `s = H0 t`, they are `da/ds = u/a` and
`du/ds = P'(a)/(2a)`. Evolving `u` rather than resetting it to `sqrt(P)`
preserves the sign at turnaround. Equivalently, the physical acceleration is

```
d²a/dt² = H0² [ΩΛ a - Ωm/(2a²) - Ωr/a³].
```

Curvature contributes to the constraint, but not directly to this physical
acceleration. The numerical integration uses Dormand–Prince 5(4), with default
relative tolerance `1e-12` and absolute tolerance `1e-14` in normalized
variables. Physical time, conformal time, and the drift integral are advanced
with the same stages. These are local error controls, not a promise of that
relative accuracy in every output. Nonfinite values, inaccessible states,
step exhaustion, and attempts to reach the `a=0` singularity produce errors.
Accepted steps may project the signed rate back onto the Friedmann constraint
when the correction fits within the existing local error budget. This removes
accumulated invariant drift near the crunch without changing the branch or
relaxing the constraint check. The dense interpolant receives the same endpoint
correction. Ill-conditioned corrections near turnaround are left unapplied.

Scale-to-time helpers describe the expanding branch. Reversed limits return
signed intervals; a simple turning-point endpoint has an integrable square-root
singularity and is handled explicitly. A target beyond the turning point is
rejected. Evolution and lightcone queries use signed time histories instead of
inverting a scale factor that occurs on two branches.

## Inputs and stopping

`cosmology.omegaLambda=auto` derives `ΩΛ = 1 - Ωm - Ωr` and sets `Ωk=0`.
Every numeric value, including `-1`, is now a literal signed dark-energy
density. When a numeric value is supplied, curvature is derived from the density
budget. Curvature residuals within eight machine epsilons times the largest
absolute density (or one, whichever is larger) are set to zero. This removes
floating-point subtraction roundoff, so the critical preset with literal
`ΩΛ=0` and `Ωm=1−Ωr` has exactly `Ωk=0` and retains its intended asymptotic
behavior. Existing INIs that used negative values as the old automatic sentinel
must replace those values with `auto`; the shipped files have been migrated.

For the cosmology scenario:

- `run.endTime=0` selects the first expanding occurrence of
  `cosmology.endRedshift`. The target must follow the starting epoch on that
  branch and be reachable.
- A positive `run.endTime` is the elapsed **conformal time in code units**
  from the initial epoch. It overrides `cosmology.endRedshift` and permits
  turnaround and contraction. It is not a physical time in years or a time
  measured from the present. Code seconds are available in the resolved unit
  settings; `time.txt` gives the corresponding physical and conformal years.
- End the run at a positive scale factor, before a Big Crunch. An endpoint
  beyond a singularity is rejected during background construction.
- `run.maxIterations` remains a separate work limit and can end a run before
  its physical target.

Gravity and SPH timestep bounds use `abs(adot)` and the acceleration. Thus the
bound remains positive during contraction and finite at maximum expansion.
The hierarchy's integer clock determines elapsed conformal time, avoiding an
extra coarse step caused by accumulated floating-point additions.

## Initial conditions and outputs

`cosmology.sigma8=0` generates an unperturbed particle lattice and bypasses
the power-spectrum solver. This provides homogeneous initial conditions for
curved and recollapsing background tests. The native **perturbed** initial
condition solver still requires a spatially flat cosmology. Native recombination
initialization also requires flatness; use `chemistry.enabled=false` or manual
chemistry initialization for a curved model. Background support alone does not
extend either of those initial-condition solvers.

The immutable background table stores adaptive dense output over the complete
run. Lightcone particles and Rockstar metadata sample the emission epoch
`τ_emit = τ_observer - radius` from that history, including on the contracting
branch. Each locality reuses its table. Restarts retain the signed checkpoint
state and reconstruct the history using the original cosmology, starting
redshift, and duration; use the original configuration when restarting. A
restart whose saved background does not match that history is rejected.
Group ages, Silo timestamps, and `time.txt` use integrated physical time, which
keeps increasing while the scale factor decreases.

Halo density conventions also remain defined at turnaround. Closed models,
negative Λ, and curved backgrounds with Λ use explicitly labelled mean-matter
thresholds: 200 for the main halo and 200/500/2500 for the secondary mass fields.
Their catalogue metadata and viewer labels identify that reference. Standard
flat nonnegative-Λ and open zero-Λ models retain their existing Bryan–Norman
and critical-density conventions. See [halo catalogue conventions](rockstar.md#catalogue-conventions).
The mean-matter thresholds are measurement definitions, not a calibrated
spherical-collapse prediction for recollapsing cosmologies.

The historical flat, expanding, nonnegative-Λ growth approximation is used
only in that regime when rescaling a power spectrum. Otherwise the file
contains measured power at the output epoch. A comment header records the
epoch and whether growth rescaling was applied; it does not claim to solve
perturbation growth in a contracting universe.

## Energy diagnostic

For collisionless particles with fixed comoving gravitational softening, `K`
is peculiar kinetic energy and `W` is gravitational potential energy. The
Layzer-Irvine equation gives

```
R = a(K + W) - a_initial(K_initial + W_initial) + integral(K da) = 0.
```

`energy.txt` reports `a*W`, `a*K`, and `cosmic`, where `cosmic` includes the
negative initial energy and the accumulated expansion work. The signed
`energy_error` is `R / (|a*K| + |a*W| + |cosmic|)` for collisionless runs.
Gas runs also include scaled thermal energy and the signed heating ledger in
the residual, and their absolute values in the normalization. This percentage
uses the sum of absolute budget terms; it is not normalized to the initial
energy or to the expansion-work integral alone.

During each drift the stored canonical momenta are constant. Writing
`Q = sum(m p^2 / 2)`, the kinetic expansion work is integrated as
`Q * (a_new - a_old) / (a_old * a_new)`. This handles both expansion and
contraction and is independent of how a drift is subdivided. `Q` uses the
momenta after the opening kicks; the reported `K` uses synchronized momenta.
Thermal and chemistry work retain their existing quadrature. Neither this
accounting identity nor a small residual establishes force or timestep
convergence.

`glass_error` measures the fractional change in `a*W` since the previous
energy output. It is a glass-relaxation diagnostic, not a cosmological energy
error, and can become large when `W` crosses zero. Energy rows are written at
coarse force evaluations and at the completed endpoint (normally `a=1`). The
terminal force evaluation synchronizes the measured momenta without another
drift or expansion-work increment. It runs after the final snapshot and
checkpoint, which retain their normal restart state. Restarting an already
completed checkpoint does not append another row. An interruption or iteration
limit before the endpoint does not produce this terminal diagnostic.

Restarts retain accumulated expansion work. Restarting a checkpoint produced
before the expansion-work correction does not repair its earlier integration
bias; use a fresh run when comparing the corrected diagnostic over the full
evolution.

Terminal wall and step durations use `hh:mm::ss.ss`. The final evolution total
starts after initialization and checkpoint loading, and includes evolution,
its outputs, and the terminal energy evaluation. A restart also shows the
recorded total including prior runs; old checkpoints may include initialization
in that prior runtime. Numeric durations in `progress.txt` remain seconds.

## Tests

`friedmann_test` checks analytic matter, radiation, curvature, de Sitter,
closed-dust and negative-Λ solutions, both clocks, signed turnaround,
integrals, dense interpolation, and invalid inputs. It also checks both
curvature signs with positive Λ, critical radiation/matter expansion through
`a=1e6` against the exact quadratic conformal-time solution, and the nonzero
limiting expansion speed of subcritical models without Λ. `cosmology_flops_test`
checks the application wrappers and adaptive work counting.
`cosmology_evolution_test` checks timestep bounds, stopping roundoff, exact
free-streaming energy accounting through expansion and contraction, drift
subdivision independence, and consistent energy normalization;
`lightcone_epoch_test` checks emission epochs across turnaround and static
background behavior.

`tests/run_cosmology.py` exercises the CPU application with homogeneous
particles: closed-dust and negative-Λ recollapse, ordinary gravity/SPH through
turnaround, checkpoint restarts during contraction and at coarse boundaries,
completed-run restarts, a future expanding redshift endpoint, and rejection of
an unreachable endpoint. Collisionless builds use `--collisionless-only` and
include hierarchical integration. The checks require one final energy row and
no duplicate on a completed restart. The script compares saved background state
with independent analytic solutions and checks monotonic clocks. The
hierarchical `fmmWalk` fixture also checks closing kicks and final energies
against independent periodic forces with nonzero velocities and mixed particle
rungs. The `rockstar_cpu` regression
checks actual lightcone halo metadata during contraction; invalid observer
times beyond a future conformal boundary are also covered by the core tests.
The options regression loads all six presets and checks their resolved
curvature, including exact zero curvature and Λ for the critical model.
These checks do not substitute for CUDA, distributed MPI, or perturbed
nonlinear recollapse validation.
