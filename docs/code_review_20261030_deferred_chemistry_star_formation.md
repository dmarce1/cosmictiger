# Deferred chemistry and star-formation review findings

Recorded October 31, 2026 after the user resumed the bug-fix pass with **all
chemistry and star-formation bugs excluded**. The original inventory is
[code_review_20261030.md](code_review_20261030.md). This record preserves those
findings for a later, explicitly authorized pass.

**Status: OPEN — DEFERRED.** No entry below is marked resolved by an algorithm
replacement, a moved source line, an existing patch, or a commit from another
conversation. The original review was read-only; its severity and verdict
describe that snapshot, not current runtime verification. Original locations
are deliberately retained. Duplicate IDs remain searchable. Low entries
without an explicit verdict inherit the original report's confirmed category,
except entries explicitly marked plausible.

The checkout is dirty. At this inventory snapshot HEAD was `408272e8`;
`ac936ef7` introduced Anninos chemistry and CUDA stars in another conversation.
Existing chemistry/SF edits must be preserved: no revert, staging, commit,
formatting or further implementation in the deferred scope. Prior-pass
ownership snapshots are `/tmp/cosmictiger_review_fixes_20261031/baseline.diff`,
`/tmp/cosmictiger_review_fixes_20261031/baseline/`, and
`/tmp/cosmictiger_review_driver_sph_baseline/`. Scratch paths are not durable
validation artifacts.

## Validation and ownership limits

- Pre-pause CPU build attempts did not establish a complete passing
  application, CUDA, MPI, restart or long cosmological chemistry/SF regression
  for these edits. Earlier builds do not validate later edits or other commits.
- The resumed parent CPU build at `build/molecular_hot_cpu` failed in
  `src/sph/sph_chemistry.cpp:173`: `cudaSetDevice` is undeclared in the CPU
  build. The existing edit is preserved under the exclusion. The diagnostic
  log is `/tmp/cosmictiger_review_fixes_20261031/resumed_build_cpu.log`;
  the parent switched to a chemistry-disabled build for independent fixes.
  No chemistry/SF test was run for this inventory.
  The later chemistry-disabled all-target build also failed in
  `tests/native_recombination.cpp:272`: `electrons.toDouble()` refers to the
  former abundance type. This excluded chemistry test remains unchanged;
  explicit nonchemistry build targets were used for validation.
- Preserved changes include the sub-10 K continuation and physics version 4;
  per-gas SH03 caps and wind-clock-cap removal; source staging; CUDA device and
  dispatch work; supernova-temperature validation; and related docs/tests.
  They remain pending review, not declared correct.
- Dense-wind behavior remains unchanged by explicit user choice. No density
  cap or alternative closure is authorized.
- Source intervals continue to follow the selected SPH step. Do not restore
  post-chemistry outer-step retries without explicit authorization.
- Independent nonchemistry fixes may proceed in shared files. Their tests do
  not close deferred source coupling, serialization, migration or parity cases.

## Deferred bugs and integration defects

### `chemistry_star_formation#2`

Shipped cosmology configs cannot reach z=0: unheated gas falls below the hard 10 K EOS floor near z~21 and the run aborts

- Original classification: **critical**, bug, **confirmed**.
- Original location(s): `cosmictiger/chemistry/primordial_eos.hpp:171` (also `src/sph/sph_chemistry.cpp:250`; `default.ini:169`; `configurations/star_formation/lcdm.ini:91`; `configurations/lcdm_present.ini:171`)
- Status: **OPEN — DEFERRED**.
- Original concern: The primordial EOS and rates have a hard 10 K lower bound that returns UnsupportedState, and the SPH/SH03 coupling never handles gas that cools adiabatically below it. default.ini and every shipped cosmological config run chemistry plus star formation from z=49 to z=0 with no radiation table. Unheated gas (10.2 K at z=21 per the docs, voids earlier) therefore makes sphChemistryEos, the SH03 CUDA EOS kernel or the chemistry batch throw, which aborts the run near z~21-25.
- Existing changes and validation limit: Preserved uncommitted EOS continuation below 10 K, rate-fit clamps, chemistry physics-version change 3 to 4, cold tests and documentation. These are not accepted as a completed fix; physical policy, solver behavior and checkpoint compatibility remain unvalidated.

### `recent_commits_b#1`, `chemistry_star_formation#1`, `chemistry_star_formation#6`

Per-particle SH03 hazard and wind-decoupling limits are applied as a global SPH timestep cap (wind case drives rungs past kMaxRung)

- Original classification: **high**, bug, **confirmed**.
- Original location(s): `src/core/driver.cpp:516` (also `src/sph/star_formation.cpp:305-314` (host limit incl. windTimeRemaining); `cosmictiger/sph/star_formation_kernel.hpp:199-204` (CUDA twin); `src/sph/sph_hydro_cpu.cpp:298`; `src/sph/sph_hydro.cu:349`)
- Status: **OPEN — DEFERRED**.
- Original concern: The SH03 timestep limit (0.05 hazard for star-forming gas and the remaining wind decoupling time for winds) is reduced over all active gas into the scalar runParameters.maximumTimestep. That scalar caps the hydro and gravity timestep of every co-active gas particle and of newborn stars, so all gas is pinned to the most demanding star-former's rung. With power-of-two rungs, the wind-clock cap also makes a wind's remaining decoupling time at least halve each step without reaching zero, so a wind that stays above 0.1 rho_th for about its decoupling time drives the rung to kMaxRung and trips ALWAYS_ASSERT.
- Existing changes and validation limit: Preserved per-gas source-timestep storage and CPU/CUDA kernel bounds, wind-clock-cap removal, driver global-reduction changes and associated docs/tests. Source-state lifecycle, rung behavior, restart and CUDA parity are not certified.

### `sph_infrastructure#1`

Hydro reruns after chemistry/star-formation open reuse the RUNGS neighbour lists, which leave out inactive particles whose support reaches an active one

- Original classification: **high**, bug, **confirmed**.
- Original location(s): `src/core/driver.cpp:566` (also `src/core/driver.cpp:531-543` (RUNGS-phase walk); `src/core/driver.cpp:580-581` (star-formation rerun); `src/sph/sph.cpp:253-261` (outer box built from active particles only); `src/sph/sph_hydro_cpu.cpp:111-113`)
- Status: **OPEN — DEFERRED**.
- Original concern: With chemistry or star formation on, the post-source hydro reruns (driver.cpp:565-567 and 579-581) reuse the RUNGS-phase neighbour lists. Those lists are built from active-only outer boxes (driver.cpp:535). They drop leaves whose inactive particles reach an active sink only through their own support (r_ab between H_a and H_b), so the opening half-kick misses those pressure and viscosity terms on individual-rung substeps, and nothing later corrects that kick.
- Existing changes and validation limit: Preserved driver changes reusing the all-particle post-density IJ lists. The chemistry/SF post-source regression is deferred; the source change does not establish closure.

### `chemistry_star_formation#17`, `recent_commits_b#4`

supernovaTemperature has no upper bound, so dense gas later exceeds the 1e8 K CIE ceiling and aborts

- Original classification: **medium**, bug, **confirmed**.
- Original location(s): `src/core/options.cpp:157` (also `src/core/options.cpp:152`; `cosmictiger/sph/springel_and_hernquist/cooling_impl.hpp:270`; `cosmictiger/sph/springel_and_hernquist/equations.hpp:73`)
- Status: **OPEN — DEFERRED**.
- Original concern: starFormation.supernovaTemperature has no upper bound. Any T_SN above 1e8 K passes validation and SH03 calibration, but the hot-phase energy u_h = u_c + u_SN/(1+A0(rho/rho_th)^-0.8) later passes the 1e8 K CIE ceiling, and both the CPU and CUDA backends abort mid-run. The crossing is at about 5.6e3 rho_th for T_SN=2e8 K and about 1e5 rho_th for 1.1e8 K.
- Existing changes and validation limit: Preserved options validation limiting supernovaTemperature to 1e8 K. Dense-closure and host/device boundary tests are deferred.

### `scenarios_configs#7`

Cosmology INIs and default.ini enable star formation: default builds reject them, and the documented collisionless recipe fails

- Original classification: **medium**, bug, **confirmed**.
- Original location(s): `configurations/README.md:39` (also `configurations/README.md:16-17`; `configurations/README.md:52`; `configurations/README.md:137-138`; `default.ini:187`)
- Status: **OPEN — DEFERRED**.
- Original concern: default.ini, cosmology.ini, the five curvature presets and lcdm_ten_ages.ini all enable star formation. validateStarFormation rejects that in the default build (COSMICTIGER_ENABLE_STARS defaults to OFF in CMake and build.sh) and whenever hydro or chemistry is disabled. As a result, the collisionless recipe in configurations/README.md always fails on these INIs, the cosmology.ini table row omits star formation and the STARS requirement, and the descriptions of default.ini as the application defaults contradict the parser defaults (chemistry.enabled=false, starFormation.enabled=false).
- Existing changes and validation limit: Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

### `chemistry_star_formation#11`

CPU chemistry batch spawns one std::thread per HPX worker from inside an HPX task (oversubscription, blocks the worker)

- Original classification: **medium**, bug, **confirmed**.
- Original location(s): `src/chemistry/primordial_chemistry.cpp:53` (also `src/chemistry/primordial_chemistry.cpp:31`; `src/chemistry/primordial_chemistry.cpp:52`)
- Status: **OPEN — DEFERRED**.
- Original concern: The CPU chemistry batch spawns hpx::get_num_worker_threads()-1 raw std::threads from inside an HPX worker and blocks that worker in join(). Under HPX's default 'balanced' binding the new threads inherit the worker's single-PU affinity, so the batch is effectively serialized on one core. In unpinned runs it oversubscribes the --hpx:threads budget instead.
- Existing changes and validation limit: Current committed Anninos/CUDA-stars work includes HPX CPU batching changes. Presence of new dispatch code is evidence to inspect later, not resolution of this finding.

### `chemistry_star_formation#3`

chemistry.maxSubsteps above 8192 is silently ineffective: hard-coded maxAttempts caps the solver

- Original classification: **medium**, bug, **confirmed**.
- Original location(s): `src/sph/sph_chemistry.cpp:170` (also `cosmictiger/chemistry/primordial_solver.hpp:19`; `cosmictiger/chemistry/primordial_solver.hpp:435`; `cosmictiger/chemistry/primordial_solver.hpp:509`)
- Status: **OPEN — DEFERRED**.
- Original concern: integrateChemistryBatch forwards chemistry.maxSubsteps but leaves SolverOptions::maxAttempts at its compiled default of 8192. advanceWithModel returns StepLimit once the attempt loop is exhausted. Any chemistry.maxSubsteps above 8192 (effectively above 8192 minus rejected trials) therefore has no effect, and nothing validates or reports this.
- Existing changes and validation limit: The current Anninos solver has maxSubsteps and no former maxAttempts field. Retired implementation details do not close this deferred finding; configured-budget behavior needs a replacement-solver regression.

### `recent_commits_b#5`, `chemistry_star_formation#5`

Decoupled winds in gas denser than n_H=1e4 cm^-3 are sent to the non-equilibrium solver, which rejects them and aborts the run

- Original classification: **medium**, bug, **confirmed**.
- Original location(s): `src/sph/star_formation.cpp:463` (also `cosmictiger/chemistry/primordial_types.hpp:189`; `src/sph/sph_chemistry.cpp:182-186`; `cosmictiger/sph/star_formation_kernel.hpp:232-236` (CUDA twin); `src/core/options.cpp:797` (help text))
- Status: **OPEN — DEFERRED**.
- Original concern: A decoupled wind's chemistry is advanced at its own SPH density, starting with the opening half right after launch. A wind launched from SH03 gas above n_H = 1e4 cm^-3 (~3e4 rho_th), which the dense closure deliberately allows, makes validateComposition return UnsupportedState, and integrateChemistryBatch aborts the whole run. Commit 84228ed2 also dropped the density limit from the chemistry.network help text.
- Existing changes and validation limit: No density cap, recoupling policy, skipped launch or rate-domain extension is authorized. The user explicitly chose to preserve the dense-wind model and leave this unresolved.

### `tests_sph_chem_cpp#1`, `chemistry_star_formation#4`

App builds the SH03 CUDA kernel with inherited --use_fast_math, but its unit test builds it strict

- Original classification: **medium**, bug, **confirmed**.
- Original location(s): `CMakeLists.txt:397` (also `cmake/variants.cmake:24`; `CMakeLists.txt:917-923`; `cosmictiger/chemistry/primordial_eos.hpp:56`; `cosmictiger/chemistry/primordial_eos.hpp:68`)
- Status: **OPEN — DEFERRED**.
- Original concern: In the application, src/sph/star_formation_cuda.cu is compiled in cosmictiger_objects, which inherits `--use_fast_math` from cosmictiger_platform. The per-source flags `--ftz=false --prec-div=true --prec-sqrt=true --fmad=false` restore IEEE division, sqrt, denormals and no-contraction, but not the function substitution. The app's device code therefore computes expf, logf, log2f and powf in the inlined primordial EOS, rate and cooling code, and in the rung log2f, with approximate intrinsics: its PTX shows `mul.rn.f32 x, 0fBFB8AA3B; ex2.approx.f32` at primordial_eos.hpp:56 and 83 `lg2.approx.f32`.
- Existing changes and validation limit: Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

### `chemistry_star_formation#10`

New CUDA entry points never call cudaSetDevice(); batches run on whatever device the current OS thread has

- Original classification: **low**, bug, **plausible**.
- Original location(s): `src/chemistry/primordial_chemistry.cu:62`
- Status: **OPEN — DEFERRED**.
- Original concern: The primordial chemistry CUDA batch and the SH03 CUDA batch run on whatever device is current for the calling OS thread, with no rank%count device selection. On multi-GPU nodes where every rank sees every GPU, a rank&gt;=1 batch can run on GPU 0 (performance and memory headroom only).
- Existing changes and validation limit: Existing device-selection calls and separately committed backend changes are preserved. Multi-GPU locality/device behavior remains unvalidated.

## Ancillary chemistry/SF smells, tests and documentation

The original review classified these as smells. They are retained separately
for completeness, including cases with correctness implications.

### `chemistry_star_formation#9`, `recent_commits_b#2`

Serial per-particle EOS/CIE loops repeat expensive work (duplicate EOS pass, 3 bisections per dense particle, constant endpoints recomputed)

- Original classification: **medium**, smell, **confirmed**.
- Original location(s): `src/sph/sph_chemistry.cpp:463` (also `src/core/driver.cpp:506`; `src/sph/star_formation.cpp:229`; `cosmictiger/sph/springel_and_hernquist/cooling_impl.hpp:116`; `src/sph/star_formation.cpp:86`)
- Status: **OPEN — DEFERRED**.
- Original concern: Several serial per-particle thermodynamics passes repeat expensive work. The chemistry driver inverts the full EOS twice in a row (pressure integral, then cache refresh), and the SF driver repeats starFormationRefreshEos at driver.cpp:472/514. The CPU SH03 refresh runs three ~32-evaluation CIE bisections per dense particle on one thread, recomputing constant endpoints every time. The projection repeats the hot-phase inversion that evaluateEquilibrium already did.
- Existing changes and validation limit: Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

### `sph_infrastructure#4`

sphParticlesRefreshChemistryEos runs serially over every gas particle and abandons child futures on error

- Original classification: **medium**, smell, **confirmed**.
- Original location(s): `src/sph/sph_particles.cpp:847` (also `src/sph/sph_particles.cpp:1195`; `src/core/driver.cpp:247`; `src/core/driver.cpp:507`; `src/core/driver.cpp:557`)
- Status: **OPEN — DEFERRED**.
- Original concern: sphParticlesRefreshChemistryEos sweeps every local gas particle on one HPX thread, doing a strict-FP-scoped EOS root-find per particle, and sphStep2 calls it three times per (sub)step in chemistry-without-star-formation runs. sphChemistryPressureIntegral adds a fourth serial sweep and UpdateTargetAlpha is also serial. On failure the child futures are dropped without wait_all, which is harmless but inconsistent.
- Existing changes and validation limit: Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

### `core_options_driver#17`, `recent_commits_b#10`

default.ini is not the 'active source defaults' it claims to be

- Original classification: **low**, smell, **confirmed**.
- Original location(s): `default.ini:187`
- Status: **OPEN — DEFERRED**.
- Original concern: default.ini does not match the compiled defaults. It enables chemistry and star formation (so it fails on a default STARS=OFF build), its softening and centralSoftening are 1 ulp off, and it pins cosmology-specific scenario-derived keys. Its spline degree range comment is stale.
- Existing changes and validation limit: Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

### `core_options_driver#19`

docs/pmmm.md claims global-mode chemistry retries that the driver does not implement

- Original classification: **low**, smell, **confirmed**.
- Original location(s): `docs/pmmm.md:402`
- Status: **OPEN — DEFERRED**.
- Original concern: docs/pmmm.md:402-405 says global-mode chemistry opening half-steps are retried at a smaller interval when the updated gas state tightens the hydro bound. In fact sphStep2 opens chemistry once with the pre-source selected interval and commits immediately.
- Existing changes and validation limit: The standing user decision rejects post-chemistry outer-timestep retries. Preserve selected-SPH-interval coupling; a future documentation fix must not reintroduce retries.

### `recent_commits_b#3`

starFormationRefreshEos runs twice back-to-back after closing, and its pre-density call is always superseded

- Original classification: **low**, smell, **confirmed**.
- Original location(s): `src/core/driver.cpp:472`
- Status: **OPEN — DEFERRED**.
- Original concern: Two of the five SH03 EOS refreshes per SPH step do nothing. The call at driver.cpp:472 is recomputed unchanged by the call at 514, or is never read on the closingOnly terminal path.
- Existing changes and validation limit: Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

### `chemistry_star_formation#15`

SH03 adapter uses the proton mass while the chemistry boundary uses m_H; physical constants duplicated across headers

- Original classification: **low**, smell, **confirmed**.
- Original location(s): `cosmictiger/sph/springel_and_hernquist/cooling.hpp:9`
- Status: **OPEN — DEFERRED**.
- Original concern: The SH03 CIE adapter and the startup nH_th print use the proton mass (1.67262e-24 g) as the hydrogen mass, while the chemistry/SPH boundary and the SH03 CUDA kernel use constants::kMh (1.6733e-24 g).
- Existing changes and validation limit: Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

### `recent_commits_b#7`

AtomicCooling::cool and maximumEnergy are dead in production; the comment claims cool() is the single radiative-loss site

- Original classification: **low**, smell, **confirmed**.
- Original location(s): `cosmictiger/sph/springel_and_hernquist/cooling_impl.hpp:169`
- Status: **OPEN — DEFERRED**.
- Original concern: AtomicCooling::cool(), maximumEnergy() and minimumEnergy() (the last used only by cool()) have no production callers; only tests/springel_and_hernquist.cpp uses them. The comment inside cool() claims it is the single per-source-solve radiative-loss site, which misleads readers about where cooling actually happens.
- Existing changes and validation limit: Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

### `chemistry_star_formation#13`

Test-only SH03 API: cool(), starFormationProbability(), windProbability(), relaxationTime are unused in production, which re-implements the probabilities inline

- Original classification: **low**, smell, **confirmed**.
- Original location(s): `cosmictiger/sph/springel_and_hernquist/equations.hpp:168`
- Status: **OPEN — DEFERRED**.
- Original concern: AtomicCooling::cool/maximumEnergy and the SH03 starFormationProbability/windProbability helpers are exercised only by unit tests. Production computes the conversion and wind probabilities inline from the accumulated optical depth in both the CPU and CUDA paths, so tests of the helpers do not cover the formulas that …
- Existing changes and validation limit: Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

### `chemistry_star_formation#18`

Stale doc: molecular9_hot.md says the SH03 fixtures refuse compute.cuda, contradicting the new CUDA SH03 backend

- Original classification: **low**, smell, **confirmed**.
- Original location(s): `docs/molecular9_hot.md:251`
- Status: **OPEN — DEFERRED**.
- Original concern: In the working tree, docs/molecular9_hot.md:250-251 says in the present tense that the SH03 fixtures refuse compute.cuda, but the in-process star_formation fixture now accepts CUDA and runs a CPU/CUDA parity check through the new star_formation_production_cuda\_{atomic6,molecular9} tests.
- Existing changes and validation limit: Backend coverage and documentation changed in other conversations. Evidence-based documentation reconciliation remains deferred.

### `chemistry_star_formation#16`, `recent_commits_b#9`

Fatal chemistry/SH03 errors report raw enum ints and batch indices, not the offending particle state

- Original classification: **low**, smell, **confirmed**.
- Original location(s): `src/sph/sph_chemistry.cpp:184`
- Status: **OPEN — DEFERRED**.
- Original concern: Run-ending chemistry and SH03 errors report raw numeric Status values and indices into filtered batches, with no particle identity or thermodynamic state. Status 2 does not distinguish the 10 K/1e8 K temperature limits from the 1e4 cm^-3 density ceiling, and every SH03 equilibrium failure is labelled as an …
- Existing changes and validation limit: Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

### `chemistry_star_formation#12`

SH03 CIE evaluations use FloatFloat without the StrictFloatEnvironment guard every chemistry entry point installs

- Original classification: **low**, smell, **confirmed**.
- Original location(s): `src/sph/star_formation.cpp:75`
- Status: **OPEN — DEFERRED**.
- Original concern: The SH03 host CIE evaluations (model calibration, per-particle equilibria, view diagnostics) run FloatFloat chemistry primitives without the StrictFloatEnvironment that every chemistry entry point installs.
- Existing changes and validation limit: The redesign changed arithmetic and cooling. Recheck applicable floating-point preconditions later; disappearance of FloatFloat alone does not close the deferred item.

### `chemistry_star_formation#8`

CPU star-formation path duplicates the shared CUDA_EXPORT kernel logic (and the RNG) instead of calling it

- Original classification: **low**, smell, **confirmed**.
- Original location(s): `src/sph/star_formation.cpp:412`
- Status: **OPEN — DEFERRED**.
- Original concern: star_formation.cpp keeps a hand-maintained CPU copy (about 240 lines) of the CUDA_EXPORT prepare/finish/evaluateEos/maximumTimestep/openNewStar operations and the SplitMix RNG. It is an intentional oracle guarded by the CUDA production parity tests, but every change must be made twice, and CPU-only builds cannot …
- Existing changes and validation limit: Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

### `recent_commits_b#6`

Chemistry for a wind that expires mid-half-step in dense gas is integrated and then discarded, and its failure can still abort the run

- Original classification: **low**, smell, **confirmed**.
- Original location(s): `src/sph/star_formation.cpp:461`
- Status: **OPEN — DEFERRED**.
- Original concern: When a wind's decoupling clock expires inside a half-step in gas at or above the SH03 threshold, both the CPU and CUDA paths integrate chemistry for the decoupled portion and then discard the result. The projection overwrites energy and composition, and the heating ledger uses only the final energy.
- Existing changes and validation limit: Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

### `chemistry_star_formation#7`, `recent_commits_b#8`

Hydro-coupling flags are fetched with one blocking, uncached remote action per tree node

- Original classification: **low**, smell, **confirmed**.
- Original location(s): `src/sph/star_formation.cpp:764`
- Status: **OPEN — DEFERRED**.
- Original concern: In multi-locality SF runs the hydro-coupling flag is the only staged SPH source field that skips the cache-line mechanism. Each unique remote neighbour node costs its own synchronous StarFormationFetchCouplingAction round trip per kSphRunHydro pass.
- Existing changes and validation limit: Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

### `chemistry_star_formation#14`

Stochastic realization depends on the number of localities: identities come from the initial rank decomposition

- Original classification: **low**, smell, **confirmed**.
- Original location(s): `src/particles/particles.cpp:1074`
- Status: **OPEN — DEFERRED**.
- Original concern: Star-formation identities, which key every conversion and wind draw, are assigned as rank offset plus local index from the post-IC per-locality layout. The same initial conditions and seed therefore give different stochastic SF realizations for different initial locality counts.
- Existing changes and validation limit: Identity initialization, migration, checkpoint and stochastic replay are deferred together. Generic particle-storage repairs do not establish decomposition-independent star-formation draws.

### `tests_sph_chem_cpp#12`

Radau chemistry reference test silently disappears when numpy/scipy are missing

- Original classification: **low**, smell, **confirmed**.
- Original location(s): `CMakeLists.txt:964`
- Status: **OPEN — DEFERRED**.
- Original concern: The SciPy Radau reference test (primordial_radau_reference, built from tests/primordial_reference.cpp/.py) is registered only when `import numpy, scipy` succeeds at configure time. Otherwise it disappears from CTest with no configure message and no reported skip.
- Existing changes and validation limit: Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

### `tests_sph_chem_cpp#11`

SPH chemistry timestep regression runs on one locality only and only with the traditional integrator

- Original classification: **low**, smell, **confirmed**.
- Original location(s): `CMakeLists.txt:999`
- Status: **OPEN — DEFERRED**.
- Original concern: The standalone SPH chemistry timestep regression is registered only as a single-locality run, and only when the traditional integrator is compiled. Its cross-rank prepare/drain/rethrow and failed-batch atomicity paths never run with two localities, and global-only SPH+chemistry builds run no chemistry timestep …
- Existing changes and validation limit: Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

### `build_cmake_scripts#14`

Capability macros come from raw cache values, so SPH-off builds still see COSMICTIGER_ENABLE_CHEMISTRY=1 (and STARS)

- Original classification: **low**, smell, **confirmed**.
- Original location(s): `cmake/variants.cmake:45`
- Status: **OPEN — DEFERRED**.
- Original concern: The chemistry and stars macros mirror the raw cache options (header fallback 1) rather than the effective SPH-and-feature capability. Every use site must add `COSMICTIGER_WITH_SPH &&` by hand; current code does, and the build checkers encode this convention, but nothing prevents a future unguarded use.
- Existing changes and validation limit: Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

### `tests_python#10`

zip() truncation silently skips chemistry.molecularFraction in run_options manual and override loops

- Original classification: **low**, smell, **confirmed**.
- Original location(s): `tests/run_options.py:816`
- Status: **OPEN — DEFERRED**.
- Original concern: Both zip() loops pair the five-entry chemistry_fields with four-value tuples, so molecularFraction is silently skipped in the manual-default and native-override checks. The impact is small: atomic6 forbids a nonzero molecularFraction, and the hot molecular9 case already verifies a molecularFraction override.
- Existing changes and validation limit: Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

### `tests_sph_chem_cpp#7`

Snapshot record layouts are hard-coded magic offsets with no static_assert; one label is misleading

- Original classification: **low**, smell, **confirmed**.
- Original location(s): `tests/sph/chemistry.cpp:156`
- Status: **OPEN — DEFERRED**.
- Original concern: The snapshot record sizes and field offsets in tests/sph/chemistry.cpp (13 words) and tests/sph/star_formation.cpp (25 words) are literals that depend on primordial::kAbundanceCount, with no static_assert, so a change in species count would silently misalign every per-particle check.
- Existing changes and validation limit: Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

### `tests_sph_chem_cpp#4`

Host/device parity compares signed abundances via fraction(), losing minority precision

- Original classification: **low**, smell, **confirmed**.
- Original location(s): `tests/sph/star_formation_cuda.cu:84`
- Status: **OPEN — DEFERRED**.
- Original concern: The host/device parity helper compares decoded fraction() values. For upper-branch (x &gt; 0.5) abundances this discards the stored minority's precision and the branch sign bit, which the production CPU/CUDA parity check verifies. All current test inputs are lower-branch, so the gap is latent.
- Existing changes and validation limit: Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

### `tests_sph_chem_cpp#3`

star_formation_cuda tests treat any CUDA error as 'no device' and silently skip

- Original classification: **low**, smell, **confirmed**.
- Original location(s): `tests/sph/star_formation_cuda.cu:110`
- Status: **OPEN — DEFERRED**.
- Original concern: star_formation_cuda.cu has its own haveDevice() check, which skips on any cudaGetDeviceCount failure. A misconfigured CUDA driver/runtime therefore shows up as CTest 'Skipped' for these six tests, while every other CUDA suite using detail::CudaTest fails.
- Existing changes and validation limit: Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

### `tests_sph_chem_cpp#2`

RejectsInvalidSourceBeforeCommit asserts on const inputs that cannot change

- Original classification: **low**, smell, **confirmed**.
- Original location(s): `tests/sph/star_formation_cuda.cu:221`
- Status: **OPEN — DEFERRED**.
- Original concern: Two expectations in RejectsInvalidSourceBeforeCommit check fields of the caller's `inputs` vector, which runCuda receives as `Input const*` and only copies to the device. They are tautologies. Nothing checks the failing index or the staged outputs.
- Existing changes and validation limit: Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

## Mixed-scope dependencies

These findings span general infrastructure and chemistry/SF or homogeneous
recombination. Deferral applies to the relevant portion, not unrelated fixes.

### `tests_python#5`

At least eight hand-rolled, drifting decoders of the same binary checkpoint format across test scripts

- Original classification: **medium**, smell, **confirmed**.
- Original location(s): `tests/gravity/nitadori/pmmm_evolution_helpers.py:52` (also `tests/run_gravity_error.py:27`; `tests/run_global_integrator.py:18`; `tests/run_primordial_integration.py:18`; `tests/run_primordial_integration.py:46`)
- Status: **OPEN — DEFERRED (chemistry/SF portion)**.
- Original concern: The binary checkpoint layout is decoded by about ten hand-written parsers in tests/. Each handles a different subset of the header flags, optional parameter blocks and gas payload versions. Two writer changes, the always-on bit-18 algorithm block (844a55c0) and the reordered viscosity parameter block (430edba2), have already left decoders stale.
- Existing changes and validation limit: Only the chemistry/SF or recombination portion is deferred; independent generic behavior may be fixed in the active pass. Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

### `particles_domain_drift#10`

SPH migration leaves received particles in nondeterministic arrival order

- Original classification: **low**, bug, **confirmed**.
- Original location(s): `src/particles/domain.cpp:557`
- Status: **OPEN — DEFERRED (chemistry/SF portion)**.
- Original concern: The SPH migration path installs received particles in whatever order the receiveSphParticles actions took the mutex, with no sort, unlike the position-sorted non-SPH path. Because tree partitioning is unstable, the order of particles within leaves, and so the floating-point sums, can differ between identical runs with …
- Existing changes and validation limit: The active generic-SPH fix tags parcels with source rank and sequence, then sorts received parcels before installation. Particle/gas payload packing, chemistry/SF fields and stochastic mechanisms are unchanged. Chemistry/SF migration and reproducibility validation remain deferred; the generic ordering fix does not close those cases.

### `recent_commits_a#9`

Capability headers now silently default missing build macros instead of #error

- Original classification: **low**, smell, **confirmed**.
- Original location(s): `cosmictiger/core/build_config.hpp:11`
- Status: **OPEN — DEFERRED (chemistry/SF portion)**.
- Original concern: The capability headers now silently default every missing COSMICTIGER_ENABLE\_\* macro (chemistry on, FMM on, traditional-only integration) instead of raising #error. No current target relies on these defaults and the legacy macros they translate are defined nowhere, so the code is dead and could only hide a target that …
- Existing changes and validation limit: Only the chemistry/SF or recombination portion is deferred; independent generic behavior may be fixed in the active pass. Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

### `particles_domain_drift#15`

Parcel limit counts particles, not bytes, giving ~1-2 GB SPH migration messages

- Original classification: **low**, smell, **plausible**.
- Original location(s): `src/particles/domain.cpp:535`
- Status: **OPEN — DEFERRED (chemistry/SF portion)**.
- Original concern: SPH migration parcels are capped at 8M records rather than bytes, which gives roughly 0.8-2.0 GB serialized messages (2.1-2.2 GB in memory) per parcel depending on options. That is below the MPI int limit, but SF/chemistry parcels exceed HPX's default 1e9-byte max_message_size, whose enforcement on the MPI parcelport …
- Existing changes and validation limit: Only the chemistry/SF or recombination portion is deferred; independent generic behavior may be fixed in the active pass. Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

### `io_checkpoint_output_tools#10`

Silo view output copies every particle twice with a 72-byte StarFormationState each, and recomputes frame-independent diagnostics per frame

- Original classification: **low**, smell, **confirmed**.
- Original location(s): `src/io/view.cpp:69`
- Status: **OPEN — DEFERRED (chemistry/SF portion)**.
- Original concern: Silo particle output materialises two full copies of all local particles (about 104-112 B each, including an unused 72-byte StarFormationState) plus per-field double buffers, and recomputes frame-independent chemistry and SH03 diagnostics for each unit/frame directory.
- Existing changes and validation limit: Only the chemistry/SF or recombination portion is deferred; independent generic behavior may be fixed in the active pass. Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

### `cosmology#19`

recombination::advanceGasTemperature is test-only; production duplicates a modified version of the SDIRK step inline

- Original classification: **low**, smell, **confirmed**.
- Original location(s): `cosmictiger/cosmology/recombination_thermal.hpp:26`
- Status: **OPEN — DEFERRED (chemistry/SF portion)**.
- Original concern: recombination::advanceGasTemperature is a frozen-coefficient SDIRK2 step whose comment calls it the 'existing' background step, but only tests/recombination_thermal.cpp calls it. zero_order.cpp:295-317 carries a diverged inline copy that evaluates the Compton coupling and T_rad at stage times.
- Existing changes and validation limit: Only the chemistry/SF or recombination portion is deferred; independent generic behavior may be fixed in the active pass. Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

### `tests_python#14`, `recent_commits_b#11`

variational_sph_application registration needlessly requires CHEMISTRY and has a stale comment

- Original classification: **low**, smell, **confirmed**.
- Original location(s): `CMakeLists.txt:2413`
- Status: **OPEN — DEFERRED (chemistry/SF portion)**.
- Original concern: The variational_sph_application CTest is gated on COSMICTIGER_ENABLE_CHEMISTRY, and its comment still describes two pressure operators and a chemistry rejection. The driver needs neither since 0720c881, so SPH builds without chemistry silently lose the Sod/blast/hydroWave/CFL regression.
- Existing changes and validation limit: Only the chemistry/SF or recombination portion is deferred; independent generic behavior may be fixed in the active pass. Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

### `tests_python#19`

compare_native_recombination 'compare' silently drops unmatched reference rows and passes on any overlap

- Original classification: **low**, smell, **confirmed**.
- Original location(s): `tests/compare_native_recombination.py:128`
- Status: **OPEN — DEFERRED (chemistry/SF portion)**.
- Original concern: The manual `compare` subcommand (and camb_reference.compare) skips reference rows that have no candidate match and passes if any single sample matches within tolerance. A truncated or regridded candidate can therefore report passed=true. The O(N·M) matching costs about 2 s on the documented grids, not minutes.
- Existing changes and validation limit: Only the chemistry/SF or recombination portion is deferred; independent generic behavior may be fixed in the active pass. Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

### `tests_python#11`

Assert-only checks, one with a side effect, would pass vacuously under python -O / PYTHONOPTIMIZE

- Original classification: **low**, smell, **confirmed**.
- Original location(s): `tests/primordial_reference.py:75`
- Status: **OPEN — DEFERRED (chemistry/SF portion)**.
- Original concern: Several drivers implement their checks with bare assert, and some put calls with side effects inside the assert: primordialReferenceInitial and the process.wait calls in run_mpi_launcher. Under PYTHONOPTIMIZE, global_integrator_cpu would lose all its checks and mpi_interrupt_launcher its exit-code checks.
- Existing changes and validation limit: Only the chemistry/SF or recombination portion is deferred; independent generic behavior may be fixed in the active pass. Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

### `tests_python#16`

MPI, CUDA and cross-build comparison paths exist in drivers but CTest never enables them

- Original classification: **low**, smell, **confirmed**.
- Original location(s): `tests/run_global_integrator.py:155`
- Status: **OPEN — DEFERRED (chemistry/SF portion)**.
- Original concern: Four drivers have code paths that CTest never runs: the two-locality global-timestep agreement, the SPH-only versus gravity-build reference comparison, the variational SPH MPI/CUDA comparisons and primordial integration on CUDA. They are run only by hand, partly documented as such, and can rot unnoticed.
- Existing changes and validation limit: Only the chemistry/SF or recombination portion is deferred; independent generic behavior may be fixed in the active pass. Existing shared-checkout changes are preserved. Re-evaluate the original behavior and applicable regression only after this scope is reopened; this record supplies no new validation.

## Originally refuted checkpoint report

`recent_commits_a#7` — Old fixed-rung chemistry checkpoints rejected with a
misleading settings-differ message. Original location:
`src/io/checkpoint.cpp:820`; original verdict: **refuted**, severity not assigned
in the appendix. The verifier said the older checkpoint was rejected earlier
with the Monaghan-support diagnostic. Preserve that historical verdict; any
new chemistry checkpoint compatibility assessment is **OPEN — DEFERRED**.
No new conclusion follows from the Anninos layout/version change.

## Scope boundary

The nonchemistry shock thermal-work bound (`sph_physics_kernels#1`), Price
viscosity background term (`sph_physics_kernels#2`), constant-gamma energy
bookkeeping (`core_options_driver#7`), generic SPH staging lifetime
(`sph_infrastructure#7`), output cadence and gravity/drift bugs remain eligible
for the active pass. A mention of chemistry in a failure example does not by
itself turn those independent fixes into chemistry work. The chemistry-specific
portions and retained hunks listed above remain frozen.


## 2026-10-31 GADGET-4 replacement disposition

The user subsequently authorized replacing chemistry and star formation with the standard GADGET-4 model. The original findings above are preserved as historical evidence. Current per-finding applicability and validation are recorded in `gadget4_bug_disposition.txt`; implementation and numerical checks are recorded in `gadget4_validation.txt`. A superseded finding describes removed behavior, not a passed regression of the old model.

## 2026-10-31: Zero-UV follow-up and remaining deferred fixes

The original findings above are historical evidence. Current per-ID dispositions and validation limits are in `gadget4_bug_disposition.txt` and `gadget4_validation.txt`. All applicable remaining findings were addressed. GPU validation resumed only after the user said War Thunder was off.

- `chemistry_star_formation#2`: **superseded**.
- `recent_commits_b#1`, `chemistry_star_formation#1`, `chemistry_star_formation#6`: **superseded**.
- `sph_infrastructure#1`: **superseded**.
- `chemistry_star_formation#17`, `recent_commits_b#4`: **superseded**.
- `scenarios_configs#7`: **resolved**.
- `chemistry_star_formation#11`: **resolved**.
- `chemistry_star_formation#3`: **superseded**.
- `recent_commits_b#5`, `chemistry_star_formation#5`: **superseded**.
- `tests_sph_chem_cpp#1`, `chemistry_star_formation#4`: **resolved**.
- `chemistry_star_formation#10`: **resolved**.
- `chemistry_star_formation#9`, `recent_commits_b#2`: **superseded; remaining repeated sweeps resolved**.
- `sph_infrastructure#4`: **resolved**.
- `core_options_driver#17`, `recent_commits_b#10`: **resolved**.
- `core_options_driver#19`: **resolved**.
- `recent_commits_b#3`: **resolved**.
- `chemistry_star_formation#15`: **resolved**.
- `recent_commits_b#7`: **superseded**.
- `chemistry_star_formation#13`: **resolved**.
- `chemistry_star_formation#18`: **resolved**.
- `chemistry_star_formation#16`, `recent_commits_b#9`: **resolved**.
- `chemistry_star_formation#12`: **superseded**.
- `chemistry_star_formation#8`: **resolved**.
- `recent_commits_b#6`: **superseded**.
- `chemistry_star_formation#7`, `recent_commits_b#8`: **superseded**.
- `chemistry_star_formation#14`: **resolved**.
- `tests_sph_chem_cpp#12`: **superseded**.
- `tests_sph_chem_cpp#11`: **resolved**.
- `build_cmake_scripts#14`: **resolved**.
- `tests_python#10`: **superseded**.
- `tests_sph_chem_cpp#7`: **resolved**.
- `tests_sph_chem_cpp#4`: **superseded**.
- `tests_sph_chem_cpp#3`: **resolved**.
- `tests_sph_chem_cpp#2`: **resolved**.
- `tests_python#5`: **resolved**.
- `particles_domain_drift#10`: **resolved**.
- `recent_commits_a#9`: **resolved**.
- `particles_domain_drift#15`: **resolved**.
- `io_checkpoint_output_tools#10`: **resolved**.
- `cosmology#19`: **resolved**.
- `tests_python#14`, `recent_commits_b#11`: **resolved**.
- `tests_python#19`: **resolved**.
- `tests_python#11`: **resolved**.
- `tests_python#16`: **resolved**.
