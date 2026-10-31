# Anninos chemistry validation — 2026-10-30

This record covers the replacement of the particle chemistry Newton/FloatFloat
solver by one scalar implementation templated on `Real`. It supersedes the
integrator and storage claims in the earlier [validation history](primordial_validation.md).
The adopted reaction catalogue and molecular EOS remain the current physical
model; this is not a reproduction of the paper's older rate fits or its complete
Eulerian hydrodynamics code.

## Implemented method

The source stages follow [Anninos et al., sections 3.1–3.2](https://arxiv.org/pdf/astro-ph/9608041):
explicit thermal subcycling, sequential scalar BDF updates, algebraic equilibrium
H−/H2+, then dominant-species hydrogen/helium and charge conservation after the
full chemical stage. Chemical subcycles retain independently integrated electrons.
Rates use the post-thermal temperature throughout chemistry, one of the paper's
supported temperature time levels. All nine species are stored directly.

The same `advance<Real>` implementation was exercised with float and double on
CPU and CUDA. There is no Newton/Jacobian, step-doubling, FloatFloat, signed-complement
encoding, or precision fallback in this particle solver. All 21 active INIs now
use the fractional-change and electron-timestep-scale controls. The minimum
electron scale affects timestep selection only; it does not modify abundances.

## Completed checks

- Release CUDA-enabled application and CPU test application built successfully.
- 21 focused CPU checks passed: ten rate/EOS tests, nine solver tests, typed units,
  and an independent Python analytic reference. The rate tests retain their
  scientific tolerances and check creation/destruction against all 37 reaction
  sources, including absent species and quadratic destruction.
- The solver tests check float/double sequential formulas, independent electrons
  between subcycles, equilibrium intermediates, direct hydrogen/helium traces,
  first-order photoionization and Compton convergence, conservation, invalid
  input, bounded work, transactional failure, and CPU worker determinism.
- All three chemistry CUDA tests passed: float and double CPU agreement, plus
  the production batch path with mixed networks, an incomplete final block,
  invalid lanes, and work-limit failures. Maximum launch: two blocks.
- All six standalone star-formation CUDA tests passed. Maximum launch: one block.
- Nine SH03 model checks and the application's star formation, particle
  persistence, and chemistry-timestep fixtures passed. The CIE flux-balance
  reference now uses double coefficients, matching the double CIE calculation;
  its existing `2e-12` tolerance was retained.
- Traditional CPU production stepping passed for atomic6, cold molecular9,
  molecule-rich molecular9 and hot molecular9. Each case passed exact gas
  checkpoint restart, Silo field readback and incompatible network/radiation
  rejection. Unsupported temperature-domain inputs were rejected.
- The full CPU star-formation application regression passed its nine fixtures
  and three cosmological combinations: traditional/global atomic6 and traditional
  molecular9. Each combination reproduced particle and chemistry state exactly
  across restart and rejected changed source settings. These cosmological cases
  retain 512 gas particles; conversion is exercised by the dedicated fixtures.
- Native atomic6 and molecular9 starts each saved 512 states matching the
  independent recombination history (48.8550491 K and H+/H = 0.000237880083).
  Global atomic6 and hot molecular9 stepping passed exact gas restart, changed
  network/radiation rejection and unsupported temperature-domain rejection.
- Two-rank MPI particle lifecycle testing passed, including combined source and
  chemistry metadata, checkpoint persistence, empty gas, and migration.

## Execution limits and evidence

Final builds and tests were pinned to at most three CPU cores in aggregate.
GPU tests ran sequentially on the RTX 3090 (82 SMs), occupying at most two SMs
at once, below the user's quarter-GPU limit while War Thunder was on. No full
GPU cosmological simulation was run for this replacement. The production sm_86
chemistry kernel reports 168 registers and 1808 bytes of stack per thread.

Primary logs are in this chat's `work/anninos/`: `release_completion_build.log`,
`production.log`, `star_cuda_tests.log`, `star_application.log`, and the
`star_application/evidence.json` restart record. Focused checks are recorded in
`/tmp/anninos_cpu_final_tests.log`, `/tmp/anninos_cuda_final_tests.log`, and
`/tmp/anninos_cpu_application_tests.log`; the latter records an initial outdated
float-reference failure, corrected by the focused double-reference rerun in
`/tmp/anninos_sh03_reference_test.log`. The additional all-reaction C/D test is
recorded in `/tmp/anninos_rates_cd_tests.log`. MPI and native/global integration
logs are `/tmp/anninos_star_particles_mpi_20261030_stage6/run.log` and
`/tmp/anninos_global_native_20261030_stage6/run.log`.

## Accuracy and compatibility limits

The ten-percent control is a substep fractional-change bound, not an adaptive
local truncation-error tolerance or a global error guarantee. Thermal/chemistry
splitting error requires refinement of the supplied outer interval. The electron
criterion does not independently limit pure H2 photodissociation; its analytic
regression explicitly checks outer-step refinement. H−/H2+ use the paper's
fast-intermediate equilibrium approximation instead of full nonequilibrium
transients. The physical rate-fit/domain limitations remain documented in
[the chemistry guide](primordial_chemistry.md).

Physics version 3 stores nine species and rejects older chemistry checkpoints.
Chemistry payloads contain twelve floats (nine species, energy, two EOS values)
plus two pending-source doubles; SPH layout versions start at 51 for chemistry
and 60 for star formation, with the existing viscosity offsets. Chemistry-disabled
checkpoint formats are unchanged. No production-scale speedup or full GPU
application accuracy claim is established by these bounded tests.
