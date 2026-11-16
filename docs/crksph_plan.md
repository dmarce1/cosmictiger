# CRKSPH implementation plan

Target: Frontiere, Raskin & Owen (2017), *CRKSPH – A Conservative Reproducing
Kernel Smoothed Particle Hydrodynamics Scheme*, JCP 332, 160–209,
[arXiv:1605.00725](https://arxiv.org/abs/1605.00725). Equation numbers below
refer to the arXiv version (sections 2–3.4).

Status: plan only, written 2026-10-29. Nothing implemented. Untracked, like the
other `docs/*_plan.md` files.

Historical plan: the proposed Monaghan pressure baseline below was removed on
2026-10-30. Production SPH now always uses variational pressure with grad-H, so
the selection and baseline assumptions in this plan require revision before use.

## 1. Scope and selection

- New runtime option `hydro.formulation = monaghan | crksph` (default
  `monaghan`). Checkpoint metadata records it and a restart with a different
  value is rejected, as `balsaraSwitch` is today.
- The Monaghan path must stay bitwise identical. CRK pair loops are selected by
  a template parameter on the CPU and CUDA pair loops, not by a runtime branch
  inside the loops, so the existing kernels keep their register budget.
- Directory `cosmictiger/hydro/sph/frontiere_etal/` and `src/hydro/sph/frontiere_etal/`,
  following the by-author layout. Paper-symbol locals inside
  `NOLINTBEGIN/NOLINTEND(readability-identifier-naming)` fences, as the
  Monaghan, Cullen–Dehnen and Price et al. code does. Add the symbol table to a
  new `docs/crksph.md` and list the exception in `AGENTS.md`.

## 2. Equations to implement

Per-particle quantities (comoving positions `x`, stored support radius `H`):

| Paper | Equation | Notes |
| --- | --- | --- |
| Volume `V_i = 1/Σ_j W(r_ij, H_i)` | (75) | gather with the sink's own `H_i`; matches the existing `ΣW = 3N/(4π)` support solve, so `V_i = 4π H_i³/(3N)` whenever the solve is not bound-limited |
| Moments `m0, m1, m2` and their gradients | (16)–(21) | use the symmetric pair kernel `W_ij = ½[W(r,H_i)+W(r,H_j)]` the paper defines under (9); the repo's `gradW̃_ab` already is its gradient |
| Corrections `A_i, B_i` | (12)–(13) | needs `m2⁻¹`; reuse the symmetric 3×3 pseudo-inverse in `cullen_and_dehnen::detail::inverseMoment` (move it to `cosmictiger/math/`) |
| `∂A_i, ∂B_i` | (14)–(15) | 3 + 9 floats; derived from the moment gradients |
| Reproducing kernel and gradient | (8)–(9) | `W^R_ij` and `∂W^R_ij` depend only on *i*'s corrections |
| Density `ρ_i = Σ m_j V_j W^R_ij / Σ V_j² W^R_ij` | (76) | single material, equal gas mass |
| Velocity gradient `∂_β v^α_i = −Σ V_j v^α_ij ∂_β W^R_ij` | (74) | 9 floats, input to the limiter |

Pair quantities in the force pass:

| Paper | Equation | Notes |
| --- | --- | --- |
| Momentum `dv_i/dt = −(1/2m_i) Σ V_i V_j (P_i+P_j+Q_i+Q_j)(∂W^R_ij − ∂W^R_ji)` | (64) | `∂W^R_ji` uses *j*'s corrections with `x_ji = −x_ij`; forces are antisymmetric but not radial |
| Energy rate `du_i/dt = (1/2m_i) Σ V_i V_j (P_j+Q_j) v_ij·(∂W^R_ij − ∂W^R_ji)` | (78) | interim rate; the predictor/corrector uses it as it uses Monaghan 6.5 today |
| Viscous pressure `Q_i = ρ_i(−C_l c_i μ_i + C_q μ_i²)` | (69) | per particle, not a pair average; `Q_j` likewise with `h_j` |
| `μ_i = min(0, v̂_ij·η_i /(η_i·η_i + ε²))`, `η_i = x_ij/h_i` | (70) | `h_i = H_i/η_max` with `η_max` the kernel's extent in the paper's units (4 for W7, 2 for the cubic, 1 for Wendland) |
| Limited jump `v̂_ij = v_ij − ½ φ_ij (∂v_i + ∂v_j)·x_ij` | (71) | |
| Van Leer limiter `φ_ij` with the `η_crit, η_fold` roll-off | (72)–(73) | `r_ij` is the ratio of the two projected gradients; `φ_ij = φ_ji` |
| Compatible energy `Δu_ij = f_ij ½(v_j+v_j' − v_i − v_i')·Dv_ij/Dt`, `f_ij` from entropy `s = P/ρ^γ` | (65)–(68) | stage 2, see §6 |

Paper parameters: `C_l = 2, C_q = 1, ε² = 0.01, η_crit = 1/n_h, η_fold = 0.2`
for W7 with `n_h = 1`; Table D.1 gives `(C_l, C_q) = (1, 0.5)` for the cubic and
`(0.5, 0.25)` for Wendland kernels. `n_h` is nodes per smoothing scale `h` in
one dimension; the paper's 3D runs sample 4 radial neighbours per support
radius, about 268 particles inside `H`.

## 3. Pipeline and data flow

Existing phases: `Prehydro1` (support solve) → `Prehydro2` (density, switch
sums) → `Hydro` (phase 0) → corrector → `Hydro` (phase 1, time steps) →
`Rungs`. Under `crksph`:

1. `Prehydro1`: unchanged.
2. `Prehydro2`: computes `V_i` (75) instead of `ρ_i`; semiactive logic unchanged.
   No velocity, energy or switch sums are staged.
3. New `kSphRunCorrections` (`Prehydro3`): stages `V_j`, `H_j`, velocities.
   Loop 1 accumulates the moments (16)–(21); solve (12)–(15). Loop 2 over the
   same neighbour list computes `ρ_i` (76) and `∂v_i` (74) with the sink's own
   corrections. Writes density and a new `SphCrkState` per sink. Semiactive
   particles are processed like `Prehydro2` so inactive neighbours of active
   sinks have fresh corrections. The driver loop in `sphStep2` gains one more
   `sphRun` + `sphTreeNeighbor` round.
4. `Hydro`: stages `SphCrkState` of every neighbour in addition to the current
   fields, evaluates (64), (69)–(74), (78) and the time step.
5. `Rungs`: unchanged.

`SphCrkState` (derived every step, not checkpointed, rebuilt after migration
like the Balsara factor): `volume, A, B[3], dA[3], dB[3][3], velocityGradient[3][3]`
= 26 floats. New `sphParticlesGlobalRead…` functions and remote cache lines for
`volume` (Prehydro3) and the full state (Hydro), following
`cullen_and_dehnen/particle_exchange.hpp`. Silo output: `volume`, optionally
`crkA`.

Time step: keep the section 10.3 structure of the current code, with the
viscous term built from `max_j(−μ_i)` and `C_l, C_q` in place of `α, β`.

## 4. Cosmology

Positions, `V_i`, `ρ_i` comoving; stored `w = a v_pec`, `u_stored = a² u`.
Derive the `1/a` factors for (64) and (78) in `docs/crksph.md` the way
`docs/monaghan_sph.md` does for 6.4–6.5 (the pair terms are the Monaghan ones
with `m_j/ρ_j → V_j`, so the same `m_b/a` scaling is expected, to be checked).
The Hubble flow `a (da/dt) x_ij` enters `v̂_ij` and the velocity gradient (74),
as it enters the Balsara and Cullen–Dehnen sums today. The homogeneous
expansion term `(5−3γ)(ȧ/a) u` stays. The corrections `A, B` are purely
geometric and need no scaling.

## 5. Files

New:

- `cosmictiger/hydro/sph/frontiere_etal/reproducing_kernel.hpp` — moment structs,
  (8)–(21); templated on `Real` for scalar, SIMD pack and device use.
- `cosmictiger/hydro/sph/frontiere_etal/equations.hpp` — (64), (75), (76), (78).
- `cosmictiger/hydro/sph/frontiere_etal/viscosity.hpp` — (69)–(73).
- `cosmictiger/hydro/sph/frontiere_etal/compatible_energy.hpp` — (66)–(68), stage 2.
- `cosmictiger/hydro/sph/frontiere_etal/state.hpp`, `particle_exchange.hpp`,
  `src/hydro/sph/frontiere_etal/particle_exchange.cpp`.
- `src/hydro/sph/sph_corrections_cpu.cpp`, `src/hydro/sph/sph_corrections.cu`.
- `cosmictiger/math/symmetric_inverse.hpp` — moved from Cullen–Dehnen.
- `tests/sph/frontiere_etal/reproducing_kernel.cpp`, `viscosity.cpp`,
  `equations.cpp`; CMake targets next to `cullen_and_dehnen_test`.
- `docs/crksph.md`.

Modified: `sph.hpp` (run type, parameters), `sph.cpp` (staging), `sph_cpu.hpp`,
`sph_hydro_cpu.cpp`, `sph_hydro.cu`, `sph_prehydro_cpu.cpp`, `sph_prehydro.cu`,
`sph_cuda.hpp`, `sph_particles.{hpp,cpp}`, `driver.cpp`, `options.{hpp,cpp}`,
`sph_flops.hpp`, checkpoint metadata, Silo writer, `CMakeLists.txt`,
`AGENTS.md`, `docs/monaghan_sph.md` (cross-reference).

## 6. Stages

1. **Kernel algebra + unit tests.** Random particle clouds (cubic, W7,
   Wendland; uniform and perturbed lattices): check `Σ V_j W^R_ij = 1`,
   `Σ V_j x_ij W^R_ij = 0` to round-off, exact gradient of a random linear field
   via (23), and `∂A, ∂B` against central finite differences of `A, B`.
   Limiter: `φ = 1` for a linear velocity field, `φ = 0` for opposite-sign
   gradients, symmetry `φ_ij = φ_ji`, roll-off below `η_crit`.
2. **CPU pipeline.** Option, `Prehydro3`, CRK hydro loop with the (78) rate,
   staging, Silo. Extend `sph_cpu_test` with a CRK fixture: pair-force
   antisymmetry (momentum conservation to round-off), zero acceleration for
   uniform pressure on a glass, and the (78) work balance.
3. **CUDA pipeline.** Mirror the CPU loops; `monaghan_sph_cuda_test` style
   CPU-vs-CUDA comparison.
4. **Scenario validation.** `sod`, `blast`, `hydroWave`, `kelvinHelmholtz`,
   `disc` with `hydro.formulation=crksph`, `kernelDegree=7`, ~268 neighbours.
   Compare against Monaghan runs and the paper's figures (Sod §4.3.1, Sedov
   §4.3.2, KH §4.4.3, Gresho-like behaviour on `disc`). Add angular momentum
   to `energy.txt` since CRK forces are not radial.
5. **Compatible energy (65)–(68).** Accumulate the pair work in the phase-1
   hydro pass, which already re-evaluates pairwise accelerations after the
   corrector: closing half-kick work from `½(v^{n+½}+v^{n+1})` and `a^{n+1}_ij`,
   opening half-kick work from the new rung's half-step. Exact total-energy
   conservation holds only when both partners take the same step, i.e. in the
   uniform-rung mode (`uniformMinimumRung ≥ 0`); with individual rungs the
   `f_ij` partition is applied to each partner's own half-step and the residual
   is reported in `energy.txt`. Keep the (78) rate for predictions, as the paper
   does.
6. **Optional later:** `α_i`-scaled `C_l` to compose the existing switches with
   the φ limiter; zeroth-order mode (`B_i = 0`) that restores exact angular
   momentum, mentioned in §3.1 of the paper.

## 7. Anticipated complications

1. **Neighbour count and cost.** The paper's resolution is ~268 particles per
   support radius with W7, against the current default `hydro.neighborNumber =
   42` (kernel-weighted). CRK needs well-conditioned `m2`, so the count cannot
   stay small. Each hydro pair also costs two corrected gradients (16 extra
   floats per neighbour, 26 with the velocity gradient) and the moments pass is
   a second neighbour sweep. Expect several times the Monaghan cost per step.
   `neighborNumber` must be converted with the `supportRadiusPerEffectiveRadius`
   conventions of `docs/kernels.md`; the paper's `n_h` and `η` are in the
   kernel's own `h`, not the repo's effective radius.
2. **Staging volume.** Neighbour structs in the CUDA hydro workspace grow from
   13 to ~39 floats; `DeviceVector` capacity, block size and occupancy need
   re-tuning. Remote cache lines carry the same growth.
3. **Extra phase.** One more global staging round per step (`Prehydro3`), and
   the semiactive machinery must run for it so inactive neighbours of active
   sinks have corrections from the current positions. Shortcut to evaluate:
   `V_j = 4π H_j³/(3N)` when the solve is not bound-limited would let the
   moments be formed in `Prehydro2` from the staged `H_j`, saving the round at
   the cost of a deviation from (75) for bound-limited particles.
4. **Compatible energy under KDK with rungs.** Section 3.3 assumes one
   acceleration per step for both partners. Our KDK has two half-kicks per
   particle on individual rungs, so exact pairwise work cancellation only holds
   for synchronized partners. Stage 5 handles this explicitly; the first
   deliverable uses the (78) rate, which is the paper's own interim form.
5. **Ill-conditioned `m2`.** Planar or collinear neighbourhoods (surfaces,
   early-time lattices, the `disc` and `polytropicStar` edges) make `m2`
   singular. The Cullen–Dehnen pseudo-inverse handles rank deficiency; CRK must
   then fall back to `B_i = 0` on the unresolved subspace and that choice has to
   be deterministic across CPU, SIMD and CUDA.
6. **Angular momentum.** Not conserved exactly (§3.1 of the paper, sub-percent
   in their tests). `energy.txt` currently records only linear momentum; add
   angular momentum before the `disc` comparison.
7. **Kernel convention.** The repo measures every kernel by effective radius
   `2σ`; the paper's `η = x/h` and its viscous coefficients are per kernel
   (Table D.1). Define `η_max` per kernel family in `math/kernel.hpp` and
   express `η_i` through `H_i` so `C_l, C_q, ε, η_crit` keep the paper's meaning.
8. **Equal masses only.** (76) is written for a single material with
   `m_ij = m_j`; the repo has one gas mass, so no extra work, but the
   multi-material branch is out of scope.
9. **Flop accounting.** Every new loop needs `sph_flops` entries and the
   `tests/sph/flops.cpp` check updated.
10. **Shared checkout.** OpenAI's handoff reports pending ROCKSTAR/NFW work and
    uncommitted `AGENTS.md`; CRK touches `sph_particles.cpp`, `driver.cpp` and
    `options.cpp`, so coordinate through `handoff/claude2openai.md` before
    editing those and keep builds within the 12-core budget.

## 8. Intersections with the existing SPH variants

| Existing piece | Relation to CRKSPH |
| --- | --- |
| Monaghan (1992) pair loops, symmetric kernel `gradW̃_ab` | Base kernel and pair-average `W_ij` reused unchanged; (43)–(44) of the paper show the Monaghan momentum equation is the `W^R → W` limit of (64), which is a useful regression check (`A = 1, B = 0` must reproduce the Monaghan force up to the `V` vs `m/ρ` volume choice) |
| Support-radius solve (`ΣW = 3N/4π`) | Already the number-density volume of (75); unchanged |
| Density gather `ρ = Σ m W` | Replaced by (76) under `crksph` |
| Monaghan `Π_ab` viscosity (`viscosityMu`, `artificialViscosity`) | Replaced by per-particle `Q_i` (69)–(70); `viscosityAlpha/Beta/Epsilon` map to `C_l, C_q, ε²` with CRK defaults 2, 1, 0.01 |
| Balsara shear limiter | Superseded by the φ limiter, which uses the full RK velocity gradient; disabled under `crksph` |
| Morris–Monaghan `α` evolution | Not part of CRKSPH; rejected under `crksph` in the first cut, possible later as an `α_i C_l` multiplier |
| Cullen–Dehnen and Price et al. switches | Their `GradientSums`, `reconstructGradients` and the 3×3 pseudo-inverse are reused for `m2⁻¹`; their velocity gradient is the Monaghan matrix estimate (B7), while CRK uses (74); rejected under `crksph` in the first cut |
| Predictor/corrector half-kicks in `sphParticlesApplyUpdates` | Unchanged for stage 2 (rate-based); stage 5 adds the compatible `Δu` accumulation in the phase-1 hydro pass |
| Semiactive particles, staging, remote cache | Pattern reused for the new volume and correction fields |
| Chemistry EOS (`pressureEnergyRatio`, per-particle γ) | Compatible: `P_j`, `c_j` and `s_j = P/ρ^γ` take the per-particle values |
| Kernel families (`spline 2..9`, Wendland, Ferrers) | Any kernel works in RK form; W7 (`kernelDegree=7`) is the paper's choice and should be the CRK default |
