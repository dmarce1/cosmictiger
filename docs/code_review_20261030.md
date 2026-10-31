# CosmicTiger code review, 2026-10-30

A read-only sweep of the whole code base for bugs and "stupid stuff", written
down for fixing later. Nothing was changed, built or run to produce it.

- **State reviewed:** `main` at `0720c881` plus the uncommitted working tree at
  about 19:20 CDT on 2026-10-30. Another session was editing star formation,
  `section_timing`, the options, the driver, CMake and the INI files during the
  review. The 68 entries located in those files are tagged *file was being
  edited during the review*: their line numbers may have moved, and some may
  already be fixed.
- **Method:** 25 reviewers, one per slice: 23 subsystem slices plus two that
  read the production diffs of the commits since `844a55c0`. A separate
  verifier then tried to refute each finding against the code. That gave 428
  raw findings: 393 confirmed, 12 plausible and 23 refuted. Folding 32
  duplicates leaves the 373 entries below. I re-checked the critical entry and
  three of the high ones against the code myself.
- **Severity:** critical = wrong physics, a crash or corruption in normal use;
  high = wrong results or a crash in a plausible configuration; medium = an
  edge case, a latent bug or a real maintainability trap; low = a minor defect
  or smell.
- **Verdicts:** *confirmed* means the verifier proved the defect from the code.
  *Plausible* means it is probably real but depends on runtime conditions; the
  entry says what would settle it.
- **IDs** such as `cosmology#2` are stable handles (`slice#n`). The slices are
  listed under [Coverage and limits](#coverage-and-limits).

| Severity | Bugs | Stupid stuff | Total |
|---|---:|---:|---:|
| Critical | 1 | 0 | 1 |
| High | 16 | 0 | 16 |
| Medium | 59 | 15 | 74 |
| Low | 66 | 216 | 282 |
| **Total** | **142** | **231** | **373** |

Sections: [Fix first](#fix-first) · [Patterns](#patterns-worth-fixing-once) ·
[Repository hygiene](#repository-hygiene) · [Coverage and limits](#coverage-and-limits) ·
[Critical](#critical-1) · [High](#high-16) · [Medium](#medium-74) ·
[Low: bugs](#low-bugs-66) · [Low: stupid stuff](#low-stupid-stuff-216) ·
[Refuted](#appendix-refuted-findings)

## Fix first

These 17 entries abort runs, break a build, or silently produce wrong results.
Details are under [Critical](#critical-1) and [High](#high-16).

**Runs that abort, builds and tests that fail**

1. `chemistry_star_formation#2` (critical): the shipped cosmology configs
   cannot reach z=0. Unheated gas falls below the hard 10 K chemistry/EOS
   domain around z≈21 (the docs give 10.20 K at z=21 and 9.31 K at z=20),
   `minimumTemperature()` returns `UnsupportedState`, and the chemistry batch
   throws. `default.ini` enables chemistry and star formation with no UV
   table. `cosmictiger/chemistry/primordial_eos.hpp:171`
2. `sph_physics_kernels#1`: the thermal-energy timestep bound also throttles
   viscous heating. A cold particle entering a strong shock gets
   dt ≈ 0.6 dt_CFL/M², and the run dies on `ALWAYS_ASSERT(nextRung < kMaxRung)`
   on both CPU and CUDA. `src/sph/sph_hydro_cpu.cpp:302`
3. `recent_commits_b#1`: the SH03 hazard and wind-decoupling limits are applied
   as one global SPH timestep cap. All gas is pinned to the most demanding
   star-former's rung, and the wind clock halves every step until `kMaxRung`
   aborts the run. `src/core/driver.cpp:516`
4. `build_cmake_scripts#1`: the default configuration with `BUILD_TESTING=ON`
   does not compile. `fmm_walk.cpp` calls `kickStepHierarchical`, which exists
   only with the hierarchical integrator, and that is off by default.
   `tests/gravity/dehnen/fmm_walk.cpp:1249`
5. `tests_python#1`: every `gravity_error_*` CTest fails, because its
   checkpoint decoder predates the runtime-algorithm block (flag bit 18).
   `tests/run_gravity_error.py:45`

**Wrong results without an error**

6. `core_options_driver#2`: the hierarchical integrator advances `scale` and
   `adot` in place, so every later rung level drifts and kicks with an
   already-advanced scale factor. `src/core/driver.cpp:1114`
7. `core_options_driver#3`: hierarchical FMM drifts move particles on
   locality 0 only. `src/core/driver.cpp:1128`
8. `particles_domain_drift#2`: hierarchical FMM never frees the remote particle
   cache, so multi-locality CPU walks keep reading positions from the first
   kick. `src/core/driver.cpp:1138`
9. `sph_infrastructure#1`: hydro reruns after the chemistry/star-formation
   source step reuse RUNGS neighbour lists that leave out inactive particles
   whose support reaches an active one. `src/core/driver.cpp:566`
10. `cosmology#1`: the kz=0 plane of the initial random field reuses the kz=1
    random numbers and comes out mirror-symmetric in y.
    `src/cosmology/initialize.cpp:528`
11. `cosmology#2`: dark-matter-only cosmology runs on several localities build
    only rank 0's part of the lattice, then carry on with N/P particles.
    `src/cosmology/initialize.cpp:698`
12. `cosmology#3`: `initial.closePack` without hydro offsets one sublattice by
    0.25 box lengths instead of 0.25 cells. `src/cosmology/initialize.cpp:726`
13. `cosmology#9`: truncated Fixed32 lattice coordinates push the M6 stencil out
    of the ghost region at locality boundaries, giving a segfault or garbage
    ICs for hydro runs on several localities. `src/cosmology/initialize.cpp:950`
14. `core_options_driver#1`: `numberOfParticles` is computed in 32-bit `int`,
    so `particlesPerDimension` above 1290 overflows. 2048 wraps to zero and
    fails with a misleading softening error; 1536 silently corrupts the units,
    softening and linking lengths. `src/core/options.cpp:1434`
15. `analysis_lightcone_bh_power#3`: every drift batch truncates and rewrites
    `lc.healpix`, so the file holds only the last batch, and the writer drops
    the last pixel. `src/analysis/lightcone.cpp:183`
16. `scenarios_configs#2`: the `glass.bin` writer and reader disagree on
    position width (fixed64 builds), particle count (`closePack`) and path.
    `src/scenarios/glass.cpp:44`

**Leaks**

17. `cuda_infra#1`: `DeviceVector` never destroys its elements, so nested
    vectors leak from the never-reset CUDA pool; the Rockstar CUDA path leaks
    every leaf's neighbour block at every output.
    `cosmictiger/cuda/device_vector.hpp:181`

## Patterns worth fixing once

The counts are approximate keyword matches over all entries.

- **Code that only runs on locality 0** (about 11 entries): hierarchical FMM
  drift, the dark-matter IC lattice, GPU HEALPix tables, scenario error norms,
  `rho.dat`, the close/direct interaction counters and more. A two-locality
  test per integrator and scenario, compared against one locality, would catch
  this whole class.
- **Hard aborts at the edge of a supported domain** (about 18): the 10 K
  chemistry floor, the 10⁴ cm⁻³ density ceiling, the 10⁸ K ceiling, `kMaxRung`
  assertions, and `THROW_ERROR` raising SIGABRT even for user configuration
  errors. Decide case by case between clamping, flagging the particle, and
  rejecting the configuration at startup.
- **Options accepted but never checked, or silently overridden** (about 12):
  `gravity.openingAngle`, `run.minRung`, `lightcone.mapSize`,
  `starFormation.supernovaTemperature`, `chemistry.maxSubsteps` above 8192,
  `compute.bucketSize` (replaced every step by a hard-coded schedule), the
  Plummer radii, and others.
- **Fast math** (about 10): the application-wide `-ffast-math` and the
  inherited `--use_fast_math` compile away every `std::isfinite` guard and make
  device results differ from the strict-math unit tests
  (`build_cmake_scripts#4`, `tests_sph_chem_cpp#1`).
- **CPU and CUDA twins that drifted apart** (about 23), and **checkpoint or
  restart gaps** (about 29), including at least eight hand-rolled Python
  checkpoint decoders that lag the C++ layout (`tests_python#1`, `#2`, `#3`,
  `#5`).
- **Dead code** (about 45) and **docs or comments that contradict the code**
  (about 30).

## Repository hygiene

These come from my own pass, not from the reviewers.

- **Leading `const`.** `.clang-format` sets `QualifierAlignment: Right`, but 17
  live declarations still write `const T`: `cosmictiger/cosmology/fft.hpp`
  (11), `src/particles/domain.cpp` (4), `src/cosmology/fft.cpp` (1) and
  `src/core/driver.cpp` (1). Running clang-format on those files fixes them.
  `src/analysis/lightcone.cpp` has 17 more, all inside the commented-out body
  of `lcAddParticle` (lines 946–996), which should be deleted
  (`analysis_lightcone_bh_power#19`).
- **`.gitignore` covers only `/build/`.** The checkout root holds 61 untracked
  entries: 30 `cosmictiger.profile.<pid>.txt` reports (`core_runtime_infra#6`),
  `apex_*.csv`, `run_*.txt`, `stack_trace.txt`, `Debug/`, `Testing/`,
  `release/`, `relwithdebinfo/`, `output/`, `__pycache__/` and Eclipse project
  files. That makes `git status` noisy and invites an accidental `git add -A`.
  It also makes a `grep -r .` from the root crawl through build and output
  trees; during this review one such grep ran for seven minutes.
- There are no `TODO`/`FIXME` markers and no `#if 0` blocks in project code.

## Coverage and limits

- Nothing was compiled, tested or executed. Every verdict comes from reading
  code, so a *confirmed* entry can still be wrong in ways only a run would
  show, and *plausible* entries need a run to settle.
- The bodies of the generated FMM kernel headers were out of scope; their
  generators and hand-written wrappers were reviewed instead.
- The four test slices (about 66,000 lines) were triaged with grep and partial
  reads, not read line by line.
- Folded duplicates appear under *Also reported as*. A few low-severity
  overlaps may remain.
- Beyond the generated kernels, reviewers reported these gaps:
  `chemistry_star_formation` did not read the star-formation, primordial and
  SH03 tests or their Python drivers; `analysis_groups_rockstar` read the group
  and Rockstar tests only partially; `gravity_cuda_kick` skipped the
  measurement sections of `docs/cuda_kick_shared_memory.md`;
  `build_cmake_scripts` checked `benchmarks/gravity/dehnen/gravity.cpp` by
  symbol name only and the HPX/APEX patches for intent only; the two
  recent-commit reviewers skipped most tests and the long docs those commits
  touched.

| Slice (ID prefix) | Scope |
|---|---|
| `core_options_driver` | options parsing, the driver loop, the scenario driver, `main.cpp` |
| `core_runtime_infra` | profiler, stack traces, HPX setup, FLOP counters, core containers and I/O helpers |
| `cuda_infra` | `DeviceVector`, the CUDA memory pool, streams, reductions, the unordered map |
| `particles_domain_drift` | particle storage, domain migration, mesh sort, drift (CPU and CUDA) |
| `io_checkpoint_output_tools` | checkpoints, views/Silo output, compression, standalone tools |
| `gravity_tree_walk_cpu` | tree build, walk, acceptance, kick, sources, force-error estimates |
| `gravity_cuda_kick` | CUDA kick kernels and the FMM operator wrappers |
| `gravity_dehnen_global_fmm` | distributed tree, remote cache, GPU staging, Ewald corrections |
| `gravity_nitadori_pmmm` | PMMM moments, FFTs, ghosts, cell schedule, near field |
| `sph_physics_kernels` | density, hydro and viscosity equations (CPU and CUDA) |
| `sph_infrastructure` | SPH particles, tree, driver, rungs, particle cache |
| `chemistry_star_formation` | primordial networks, solver, EOS, SH03 star formation |
| `cosmology` | Friedmann background, Boltzmann solver, initial conditions, FFTs |
| `analysis_groups_rockstar` | FoF groups and Rockstar halo properties |
| `analysis_lightcone_bh_power` | lightcone, HEALPix, Barnes–Hut, power spectra |
| `math_numerics` | fixed point, double-float, special functions, SIMD, kernels, units |
| `scenarios_configs` | test scenarios, `default.ini`, `configurations/` |
| `build_cmake_scripts` | CMake, `build.sh`, scripts, benchmarks |
| `fmm_generators_tools` | FMM kernel generators and the halo viewer |
| `tests_gravity_cpp`, `tests_sph_chem_cpp`, `tests_other_cpp`, `tests_python` | test-suite quality |
| `recent_commits_a` | production diffs of `844a55c0` to `819b870f` |
| `recent_commits_b` | production diffs of `2c9aa7eb` to `0720c881` |

## Critical (1)

### Chemistry and star formation

#### Shipped cosmology configs cannot reach z=0: unheated gas falls below the hard 10 K EOS floor near z~21 and the run aborts

`chemistry_star_formation#2` · **critical** · bug · confirmed

- **Where:** `cosmictiger/chemistry/primordial_eos.hpp:171` (also `src/sph/sph_chemistry.cpp:250`; `default.ini:169`; `configurations/star_formation/lcdm.ini:91`; `configurations/lcdm_present.ini:171`)
- **Problem:** The primordial EOS and rates have a hard 10 K lower bound that returns UnsupportedState, and the SPH/SH03 coupling never handles gas that cools adiabatically below it. default.ini and every shipped cosmological config run chemistry plus star formation from z=49 to z=0 with no radiation table. Unheated gas (10.2 K at z=21 per the docs, voids earlier) therefore makes sphChemistryEos, the SH03 CUDA EOS kernel or the chemistry batch throw, which aborts the run near z~21-25.
- **Failure scenario:** Run default.ini (or configurations/star_formation/lcdm.ini) unchanged: around z~25-21 the first void particle's thermal energy drops below E(10 K) and the run dies with 'Chemistry EOS rejected gas state (status 2)'.
- **Fix:** Choose and enforce a policy. Option (a): validate at startup that a chemistry cosmology run either stops above the documented cold limit or supplies a radiation table whose heating keeps the IGM above 10 K, and fail immediately with a clear message. Option (b): allow EOS inversion below 10 K (translational and rotational energy are well defined there) and hold rate and cooling fits at their 10 K values as a documented continuation. Either way, fix default.ini and the cosmological configs to match. Make the error report the particle, its temperature and the 10 K bound.


## High (16)

### Options, driver and startup

#### Per-particle SH03 hazard and wind-decoupling limits are applied as a global SPH timestep cap (wind case drives rungs past kMaxRung)

`recent_commits_b#1` · **high** · bug · confirmed · file was being edited during the review

- **Where:** `src/core/driver.cpp:516` (also `src/sph/star_formation.cpp:305-314` (host limit incl. windTimeRemaining); `cosmictiger/sph/star_formation_kernel.hpp:199-204` (CUDA twin); `src/sph/sph_hydro_cpu.cpp:298`; `src/sph/sph_hydro.cu:349`)
- **Problem:** The SH03 timestep limit (0.05 hazard for star-forming gas and the remaining wind decoupling time for winds) is reduced over all active gas into the scalar runParameters.maximumTimestep. That scalar caps the hydro and gravity timestep of every co-active gas particle and of newborn stars, so all gas is pinned to the most demanding star-former's rung. With power-of-two rungs, the wind-clock cap also makes a wind's remaining decoupling time at least halve each step without reaching zero, so a wind that stays above 0.1 rho_th for about its decoupling time drives the rung to kMaxRung and trips ALWAYS_ASSERT.
- **Failure scenario:** Traditional integrator with starFormation.winds=true. A wind with remaining decoupling time R sets cap = R/(a·T) for all active gas. With a rung-quantized dt in (cap/2, cap], each active step uses more than half of R but almost never all of it: the min() clamp fires only if dt is within about ȧdt/a of the cap. R therefore roughly halves every step and never reaches 0. Unless the particle first falls below 0.1 ρ_th, within about 15 of its steps the required rung reaches kMaxRung=16. The run then hits "SPH timestep exceeds rung range" and ALWAYS_ASSERT(nextRung &lt; kMaxRung), or "New stellar gravity timestep exceeds the rung range" when a conversion occurs that step. Before the abort, all active gas in the box runs on the wind's tiny rung. Without winds, the densest star former (about 1-4 Myr at 10^3 ρ_th) pins all IGM gas several rungs deeper than needed.
- **Fix:** Remove the windTimeRemaining term, since prepare() already splits chemistry and SH03 at expiry. If the recoupling time must be bounded, use a soft per-particle limit that may overshoot R. Do not use a hard cap. Make the 0.05 hazard limit per particle for the traditional integrator, for example by staging a per-gas maximum timestep beside sphParticlesChemistryEos and using it in rung selection in both the CPU and CUDA kernels. Keep a global reduction only when globalTimestep is true. Do not feed the gas source cap into the newborn-star timestepCap. Document the behaviour in docs/star_formation.md.
- **Also reported as** `chemistry_star_formation#1` (Wind decoupling clock used as a timestep cap shrinks geometrically (Zeno-like) until the rung limit aborts the run) at `src/sph/star_formation.cpp:312`: starFormationMaximumTimestep caps the global SPH timestep at an active wind's remaining decoupling time. Rung steps are power-of-two fractions of t0 that almost never equal the remainder exactly, so every activation leaves 0 &lt; R' &lt; R/2 and pushes the rung up by at least one. A wind still above 0.1 rho_th as its clock nears expiry therefore drives nextRung to kMaxRung=16 and trips ALWAYS_ASSERT, on both CPU and CUDA.
- **Also reported as** `chemistry_star_formation#6` (Star-formation timestep limit is a global minimum applied to every gas particle and to newborn stars) at `src/sph/star_formation.cpp:314`: starFormationMaximumTimestep reduces the per-particle SH03 hazard bound 0.05/(s(1+η)) and the remaining wind clocks to one minimum over all active gas on all localities. sphStep2 (driver.cpp:516-517) stores it in runParameters.maximumTimestep, which the CPU and CUDA hydro kernels use as the hydro and gravity dt cap for every active gas particle; the same value is also passed to starFormationOpenNewStars as the cap for newborn stars.

#### Hydro reruns after chemistry/star-formation open reuse the RUNGS neighbour lists, which leave out inactive particles whose support reaches an active one

`sph_infrastructure#1` · **high** · bug · confirmed · file was being edited during the review

- **Where:** `src/core/driver.cpp:566` (also `src/core/driver.cpp:531-543` (RUNGS-phase walk); `src/core/driver.cpp:580-581` (star-formation rerun); `src/sph/sph.cpp:253-261` (outer box built from active particles only); `src/sph/sph_hydro_cpu.cpp:111-113`)
- **Problem:** With chemistry or star formation on, the post-source hydro reruns (driver.cpp:565-567 and 579-581) reuse the RUNGS-phase neighbour lists. Those lists are built from active-only outer boxes (driver.cpp:535). They drop leaves whose inactive particles reach an active sink only through their own support (r_ab between H_a and H_b), so the opening half-kick misses those pressure and viscosity terms on individual-rung substeps, and nothing later corrects that kick.
- **Failure scenario:** Traditional (individual-rung) integrator with chemistry.enabled=true, or with starFormation.enabled=true, on a minRung&gt;0 substep. A dense active clump (small H_a, high rung) sits next to diffuse inactive gas (H_b of about 3 H_a, lower rung). For a gas particle b at r_ab of about 2 H_a whose whole leaf lies outside the active outer box of a's leaf, the rerun drops b's pressure, viscosity and work terms from dv_a/dt and du_a/dt. The particle then gets a wrong opening kick: momentum and energy errors that are never corrected and are not detected (no NaN, no assert).
- **Fix:** Delete the RUNGS-phase box and neighbour re-walk at driver.cpp:532-545 and reuse the post-density IJ lists. Those were built with kSphSetAll and x1.01, and positions and supports have not changed since. They are a superset for both the RUNGS kernel, which applies the active filter and the exact max(H_a,H_b) test itself, and the hydro reruns. This also saves two tree walks per step. Alternatively, set outerSelection = kSphSetAll at line 535, or redo a kSphSetAll box walk plus IJ walk before the reruns when chemistry or star formation is enabled.

#### Hierarchical ladder mutates scale/adot: later rung drifts and kicks use an already-advanced scale factor

`core_options_driver#2` · **high** · bug · confirmed · file was being edited during the review

- **Where:** `src/core/driver.cpp:1114` (also `src/core/driver.cpp:1020`; `src/core/driver.cpp:1025-1028`; `src/cosmology/cosmology.cpp:46`)
- **Problem:** In kickStepHierarchical, cosmosAinv(adot, scale, dt) advances the by-value scale/adot in place, so every descending rung level after the top one is wrong. Its drift factor is integrated from an already-advanced scale factor, though its drift interval is still [tau, tau+dt_L]. Its kick timestep criterion (kparams.a) and expansion limit (kparams.maxDt) also use the advanced state.
- **Failure scenario:** Cosmology with --run.timeIntegrator=hierarchical, z=49 start, 64 coarse steps. a changes by ~20% over the first t0. Rung-1 particles then drift with &lt;1/a&gt; averaged over [tau+t0, tau+1.5t0] instead of [tau, tau+0.5t0], about 16% too little displacement. Deeper rungs are affected the same way at every ladder with more than one level (every coarse and sub-boundary), and their rung selection uses the wrong a.
- **Fix:** Integrate on copies: `double aWork = scale, adotWork = adot; a = 1.0 / cosmosAinv(adotWork, aWork, dt);`. Better, add a non-mutating cosmosAinv(double adot, double a, double dt) overload or return the drift integral from a const-input helper, so scale/adot stay at a(tau) for every level.

#### FMM hierarchical rung drift only runs on the root locality

`core_options_driver#3` · **high** · bug · confirmed · file was being edited during the review

- **Where:** `src/core/driver.cpp:1128` (also `src/particles/drift.cpp:84`; `src/core/driver.cpp:859-882`; `src/core/driver.cpp:1774-1777`)
- **Problem:** With FMM gravity and run.timeIntegrator=hierarchical on more than one locality, rung drifts run only on locality 0. drift() with rung &gt;= 0 never dispatches to children, and unlike the PMMM branch the FMM branch has no particleMeshDriftRung-style broadcast. Particles owned by other localities keep being kicked but never move, and they never cross the lightcone.
- **Failure scenario:** Two or more localities, FMM + run.timeIntegrator=hierarchical: particles owned by ranks &gt;= 1 never move. Their velocities keep being kicked, so positions and velocities diverge from the root's particles: wrong physics, and wrong lightcone crossings for remote particles.
- **Fix:** Move particleMeshDriftRung out of the PMMM-only #if guard and rename it (e.g. driftRungEverywhere). Call it from the FMM branch as well, so both solvers share one broadcasting rung-drift entry. Add a 2-locality hierarchical FMM regression that checks remote-owned positions change.
- **Also reported as** `particles_domain_drift#1` (Hierarchical FMM drift moves particles on the root locality only) at `src/core/driver.cpp:1128`: With --run.timeIntegrator=hierarchical and FMM gravity on 2 or more localities, every rung's drift moves only locality 0's particles. kickStepHierarchical runs on the root and calls the locality-local, rung-restricted drift() directly. Only the PMMM branch wraps it in the fan-out helper particleMeshDriftRung. Particles on the other ranks are kicked but never moved. The scenario driver's hierarchical path is affected the same way.

#### Remote particle position cache is never invalidated in hierarchical FMM runs

`particles_domain_drift#2` · **high** · bug · confirmed

- **Where:** `src/core/driver.cpp:1138` (also `src/core/driver.cpp:886-1157`; `src/core/driver.cpp:722-723`; `src/particles/particles.cpp:1340`; `src/particles/domain.cpp:656`)
- **Problem:** kickStepHierarchical never calls particlesCacheFree(); treeReset() clears only the remote tree cache. On 2 or more localities, the index-keyed remote particle cache survives every migration, rung sort, tree partition and drift. CPU-walk sinks (all sinks in CPU builds, and CUDA sinks with multi-rank checklists) keep reading remote source positions fetched during the first kick, which gives wrong forces for the rest of the run. The traditional kickStep frees the cache after every kick, and PMMM frees it in its cleanup.
- **Failure scenario:** Hierarchical integrator with FMM on 2 or more localities. This covers CPU walks, and CUDA runs whose multi-rank checklists fall back to the CPU walk. The first kick caches remote lines. Later substeps and steps migrate, rung-sort, tree-partition and drift the remote particles, and the tree is rebuilt with new particle ranges. particlesGlobalReadPos keeps returning the lines fetched the first time, so particle-particle forces use positions of other or outdated particles for the rest of the run.
- **Fix:** Call particlesCacheFree() after each substep's kick in kickStepHierarchical, next to treeReset() at driver.cpp:1138 and in the closingOnly branch at 1089, mirroring kickStep. More robust alternatives: invalidate the particle cache from treeReset/treeCreate, or add an epoch counter that domainsEnd, particlesSortByRung, particlesPartition and drift bump, with stale lines refetched.

#### numberOfParticles is computed in 32-bit int and overflows for particlesPerDimension &gt;= 1292

`core_options_driver#1` · **high** · bug · confirmed · file was being edited during the review

- **Where:** `src/core/options.cpp:1434` (also `cosmictiger/math/math.hpp:20`; `src/particles/particles.cpp:425`)
- **Problem:** processOptions computes numberOfParticles = sqr(int N) \* int N in 32-bit signed arithmetic (UB), so every even particlesPerDimension &gt;= 1292 overflows. At 2048 the count wraps to 0 and startup aborts with a misleading softening error. At 1536 it wraps to about 1.8e19 and silently corrupts the units, gm, softening, FoF linking length and analysis normalizations.
- **Failure scenario:** --initial.particlesPerDimension=2048: 2^33 wraps to 0, so lengthCm=0, gm=NaN, and supportRadius\*N^-1/3 becomes inf. Startup then fails with a misleading 'gravity.effectiveRadius ... must remain finite' error. With 1536: the result wraps to -671088640, which becomes size_t ~1.8e19. It passes every check, then the mass/length units, gm, normalized softening, FoF linking length, lightcone nside and power normalization are all wrong. particles.cpp:425 sizes buffers from this count.
- **Fix:** Compute in 64-bit: `size_t const n = size_t(opts.initial.particlesPerDimension); opts.initial.numberOfParticles = n * n * n;` (before the closePack doubling). Optionally reject sizes whose per-locality count cannot fit ParticleInteger when LONG_LONG_PART_INT is off. Grep other N^3 products done in int (e.g. FFT sizes derived from fourierSize).


### Scenarios and INI files

#### glass.bin writer/reader mismatch: fixed64 builds write 8-byte positions, closePack writes 2x entries under a side^3 header

`scenarios_configs#2` · **high** · bug · confirmed

- **Where:** `src/scenarios/glass.cpp:44` (also `src/scenarios/glass.cpp:112`; `src/cosmology/initialize.cpp:742`; `src/cosmology/initialize.cpp:759-764`; `README.md:89-90`)
- **Problem:** saveGlass writes each coordinate with sizeof(Position), which is 8 bytes in COSMICTIGER_POSITION_BITS=64 builds. The only loader (initX0) always reads side^3 Fixed32 values per axis and never validates the file, so a fixed64-generated glass loads as silent garbage, despite the README promising Q0.32 in both modes. In addition, the header stores particlesPerDimension while the body holds numberOfParticles (2\*side^3 when closePack=true, which nothing rejects for scenario=glass). The writer puts the file in output.directory but the reader hard-codes ./glass.bin, and saveGlass runs twice on convergence.
- **Failure scenario:** fixed64 build: run scenario=glass, copy output/glass/glass.bin to the CWD, then run cosmology with initial.useGlass=true. The x array is filled with the low and high halves of 64-bit words, and y/z come from the wrong blocks. The garbage positions are still in [0,1), so nothing reports an error. Glass generated with closePack=true: y is read from the second half of the x block.
- **Fix:** Write explicit Q0.32 values, e.g. uint32_t(position.raw() &gt;&gt; 32) in fixed64 builds. Reject initial.closePack for scenario=glass, or write the true particle count. Add a magic/version/element-size header, and make the loader check it and the exact file size. Add an input-path option for initial.useGlass, or document copying output/glass/glass.bin. Remove the second saveGlass call when the in-loop save has already run.


### CUDA infrastructure

#### DeviceVector never destroys its elements, so nested vectors leak (ROCKSTAR leaks every leaf's neighbor block)

`cuda_infra#1` · **high** · bug · confirmed

- **Where:** `cosmictiger/cuda/device_vector.hpp:181` (also `src/analysis/rockstar.cpp:567`; `src/analysis/rockstar.cpp:101`; `cosmictiger/cuda/device_vector.hpp:91-100` (destroy); `cosmictiger/cuda/device_vector.hpp:158-163` (move assignment))
- **Problem:** DeviceVector's destructor, move assignment and popBack release or drop storage without destroying elements. As a result, every RockstarTree::neighbors block is leaked from the never-reset 8 GiB pool when the local `trees` in rockstarSeeds goes out of scope. These blocks are allocated (about 1 KiB per leaf) on the CUDA path for every (sub)group with more than 4096 particles, at every level and every output.
- **Failure scenario:** CUDA-enabled halo finding. rockstarSeeds (src/analysis/rockstar.cpp:567) builds `DeviceVector<RockstarTree> trees`. For every (sub)group larger than 4096 particles, the GPU path calls rockstarFindNeighbors, which does `self.neighbors.resize(leaflist.size())` for every leaf (rockstar.cpp:101). Each leaf gets a pool block of at least 1 KiB (64-particle buckets give about N/32 to N/64 leaves). When `trees` goes out of scope, only the RockstarTree array is released. Every neighbors block leaks, at every recursion level, for every halo, at every output. Example: 40M particles in large halos is about 0.8M leaves, or about 0.8 GB leaked per level per output. After a few outputs grow() gets nullptr and aborts with "OOM in device_vector", and every other pool user fails too (SPH workspaces, drift lightcone lists). The same mechanism leaks every old bucket in CudaUnorderedMap::rehash, and the elements of any nested vector that is move-assigned over or popped.
- **Fix:** Immediate fix: call `trees.resize(0)` before rockstarSeeds returns, or better, store leaf neighbor lists in one flat DeviceVector&lt;int&gt; plus offsets, which also removes the 1 KiB-per-leaf minimum. Generic fix: on the host, when !std::is_trivially_destructible_v&lt;T&gt;, run destruct(0, size\_) in ~DeviceVector and in move assignment, and destroy the popped element in popBack. Do this only together with the shallow-copy fix (cuda_infra#5) and the Pair move fix (cuda_infra#3). Otherwise rehash and erase would release aliased neighbor blocks twice. Alternatively, static_assert trivially destructible T and restructure the nested users.


### SPH

#### Thermal-energy timestep bound also limits heating, giving dt ~ dt_CFL/M^2 in shocks into cold gas and aborting at kMaxRung

`sph_physics_kernels#1` · **high** · bug · confirmed

- **Where:** `src/sph/sph_hydro_cpu.cpp:302` (also `src/sph/sph_hydro.cu:353-357`; `docs/variational_sph.md:366-369`; `docs/monaghan_sph.md:391-392`)
- **Problem:** The fractional thermal timestep bound uses |du/dt| of the total rate, which includes viscous heating. That heating does not depend on u_a, so for a cold particle entering a strong shock δt falls to about 0.6 δt_CFL/M². hydroRung then rises by about log2(M²) and the run aborts on ALWAYS_ASSERT(nextRung &lt; kMaxRung) on both CPU and CUDA. docs/variational_sph.md already records this as the 16³ traditional Sedov failure. The shipped 64³ blast.ini and cosmological accretion shocks into cold gas are plausibly affected.
- **Failure scenario:** Sedov blast with the documented u=1e-20 background, or a Mach-100 accretion shock in a molecular9/atomic6 cosmology run. On the first steps of a pre-shock particle, hydroRung jumps by about log2(M^2) above the CFL rung, nextRung reaches 16, and the run stops with "SPH timestep exceeds rung range". Otherwise the shocked layer runs 1e3-1e4 times more steps than its signal-speed limit.
- **Fix:** Accumulate the viscous heating (the m_b/a·½Π_ab F_ab v·r_phys term) in its own lane sum and leave it out of the fractional bound. Keep the bound, for either sign, on the reversible pressure work plus the expansion term. That part scales with u_a, acts as a divergence limiter, and is what testDefaultThermalTimestep checks. Let the β μ_max term of the CFL limit viscous heating. One option is to make the CFL signal speed pairwise, max_b(c_a + c_b − 3 min(w_ab·r̂\_ab, 0)), so a cold sink also sees the neighbour's sound speed and the approach speed. A blunter alternative is to soften the numerator to u_a + ½v_sig,a². Do not apply the bound only when du/dt&lt;0 unless the CFL gains the approach-speed term. Change both backends and update docs/variational_sph.md and docs/monaghan_sph.md. Then re-run blast 16³ and 64³ with the traditional integrator.


### Cosmology and initial conditions

#### kz=0 plane random seeds collide with the kz=1 draws and fold each axis separately (correlated, mirror-symmetric ICs)

`cosmology#1` · **high** · bug · confirmed

- **Where:** `src/cosmology/initialize.cpp:528` (also `src/cosmology/initialize.cpp:481` (row seed reused); `src/cosmology/initialize.cpp:491` (kz=0 skipped, so kz=1 consumes the first draw); `src/cosmology/initialize.cpp:557-563` (conj rule))
- **Problem:** In twolptGenerate the kz=0 mode of row (i0,i1) is drawn from a generator seeded and burned exactly like row (fold(i0),fold(i1)), whose first draw already went to its kz=1 mode. So the whole kz=0 plane copies (or conjugates) kz=1 random numbers. Because x and y are folded separately, (kx,ky,0) and (kx,-ky,0) also share one random number for kx!=0, which makes the z-projected IC field mirror-symmetric in y apart from its x-averaged profile. The kx=n/2 line also breaks Hermitian symmetry.
- **Failure scenario:** Any cosmology run with sigma8&gt;0: project the IC density along z -&gt; exact mirror symmetry about y=0 and y=L/2; correlate delta(kx,ky,0) with delta(kx,ky,1) -&gt; correlation coefficient ~1 instead of 0.
- **Fix:** Give the kz=0 plane its own stream, or better, use a counter-based or hashed RNG keyed by the full (i0,i1,i2) index. On the kz=0 and kz=n/2 planes, draw only one canonical member of each Hermitian pair and set ((n-i0)%n,(n-i1)%n) to its conjugate (with the sgn factor). Make self-conjugate points real or zero the Nyquist modes. Add a unit test for Hermitian symmetry and for independence of delta(kx,ky,0) from delta(kx,-ky,0) and delta(kx,ky,1). Note in the docs that realizations change.

#### DM-only multi-locality ICs: initX0 never runs on non-root localities, so only rank 0's lattice block exists

`cosmology#2` · **high** · bug · confirmed

- **Where:** `src/cosmology/initialize.cpp:698` (also `src/cosmology/initialize.cpp:683` (clearX0 uses the same condition); `src/cosmology/initialize.cpp:916` (unused initializeParticles); `src/core/driver.cpp:1364`; `src/main.cpp:362`)
- **Problem:** initialize() runs only on locality 0, and initX0/clearX0 forward to child localities only when hydro is enabled or sigma8==0. In a perturbed dark-matter-only cosmology run on more than one locality, only rank 0 creates its findMyBox() sub-block of the lattice and the other ranks keep zero particles. The run then proceeds silently with N/P particles and total mass M/P. This is the default for gravity-only builds.
- **Failure scenario:** hydro.enabled=false, sigma8=0.8, 2+ localities: rank 0 creates only its half (or quarter, ...) of the lattice; ranks&gt;0 have particlesSize()==0; the total mass is 1/P and part of the box is empty.
- **Fix:** Fan out InitX0Action and ClearX0Action unconditionally, since each locality builds only its own block, and remove the dead initializeParticles parameter from zeldovichEnd. Add a 2-locality DM-only cosmology test that checks the global particle count equals initial.numberOfParticles (twice that with hydro).

#### closePack without hydro offsets one sublattice by 0.25 box lengths instead of 0.25 cells

`cosmology#3` · **high** · bug · confirmed

- **Where:** `src/cosmology/initialize.cpp:726` (also `src/cosmology/initialize.cpp:950-974` (zeldovichEnd stencil read))
- **Problem:** With initial.closePack=true and hydro disabled, the first particle of each cell is offset by 0.25 box lengths instead of 0.25 cells. After Fixed32 wrapping, particles form pairs about 0.43 cells apart instead of a BCC lattice. On multi-locality runs zeldovichEnd, which wraps grid indices only for hydro, then reads the displacement grid far outside the locality's ghosted box.
- **Failure scenario:** initial.closePack=true, hydro.enabled=false: the lattice is distorted, not BCC (strong discreteness artifacts). With &gt;1 locality (rank 0 box [0,n/2)), the stencil reads rows near n/2..3n/4 from a vector covering [-2,n/2+3): garbage displacements or a segfault (an assert with CHECK_BOUNDS).
- **Fix:** Use 0.25 \* ninv unconditionally. Defensively, also apply the periodic wrap of i1/j1/k1 in zeldovichEnd for every particle, not only for hydro.

#### Truncated Fixed32 lattice coordinates shift the M6 stencil one cell below the ghost region at locality boundaries

`cosmology#9` · **high** · bug · confirmed

- **Where:** `src/cosmology/initialize.cpp:950` (also `cosmictiger/math/fixed.hpp` (floatingBits truncation); `src/cosmology/initialize.cpp:926-927` (ghost extents))
- **Problem:** Fixed32 truncation stores a non-dyadic lattice coordinate b/n slightly low. As a result, `int i1 = x0 * n` gives b-1 (t≈1) for the first DM plane of any locality whose box starts at such a row. Remote localities only own particles when hydro is on, and in that case the gas-wrap fix-up (lines 956-964, `i1 < box.begin - kCloudMin`) adds n. Every stencil read for that plane then lands hundreds of planes past the end of the ghosted yValue, with full M6 weights, causing a segfault or garbage displacements and velocities. Example: hydro, particlesPerDimension=768, 3 localities.
- **Failure scenario:** hydro.enabled=true, particlesPerDimension=768, 3 localities: locality 1's box begins at x=256, its first-row particles access row 253, and yValue starts at row 254, so the read is out of bounds (CHECK_BOUNDS builds assert).
- **Fix:** Compute the base cell with a tolerance larger than the Fixed32 error (≤ n/2^32): `int i1 = int(std::floor(x0 * n + 1.0e-6))`, and likewise for j1 and k1, letting t go slightly negative (cloudWeight uses |x|). Rounding the stored positions to nearest does not help, because 2^32/3 still rounds down. An alternative is to extend the ghost region one more cell on the low side (`box.begin[dim] += kCloudMin - 1`). The wrap test, which is written relative to box.begin, then becomes i1 &lt; b-1 and no longer fires for these particles. Either way, make the wrap test apply only to genuinely wrapped gas, and add an ASSERT that each box.index lies in [0, yValue.size()).


### Analysis: lightcone, BH and power

#### lc.healpix map is rewritten by every drift batch, keeping only the last one

`analysis_lightcone_bh_power#3` · **high** · bug · confirmed

- **Where:** `src/analysis/lightcone.cpp:183` (also `src/analysis/lightcone.cpp:124`; `src/analysis/lightcone.cpp:345`; `src/particles/drift.cpp:76`; `src/particles/drift.cu:224`)
- **Problem:** Each drift batch rebuilds the lightcone HEALPix map from scratch and truncates lc.healpix.&lt;epoch&gt;.&lt;rank&gt; with "wb", while epoch advances only at flushes. The file keeps only the last nonempty batch's crossings, not the epoch's shell. The writer also drops the last pixel (it writes maxPix - minPix values for the inclusive range [minPix, maxPix]).
- **Failure scenario:** In a normal lightcone run there are dozens to thousands of drift calls between flushes. The surviving lc.healpix.E.R holds only the crossings of the last drift on that rank, a thin sliver of the shell, and different ranks' files can come from different steps.
- **Fix:** Keep a persistent module-scope accumulation map (double precision) and add each batch into it in lcAddParticleEntries. Write it once per epoch in lcParts2groups before epoch++, which also covers the final forced flush, then reset it, and save/restore it in lcSave/lcLoad. Write maxPix - minPix + 1 values, or the whole map, and update docs/output_files.txt.
- **Also reported as** `analysis_lightcone_bh_power#6` (lc.healpix writes maxPix-minPix values, dropping the last occupied pixel) at `src/analysis/lightcone.cpp:191`: lcAddParticleEntries records minPix/maxPix as inclusive bounds of the occupied pixels but writes only maxPix-minPix floats. Every lc.healpix.E.R file drops the highest occupied pixel, and a batch confined to one pixel writes no values at all.


### Tests

#### Default configuration with BUILD_TESTING=ON does not compile: fmm_walk test calls kickStepHierarchical, which only exists when hierarchical integration is enabled

`build_cmake_scripts#1` · **high** · bug · confirmed · file was being edited during the review

- **Where:** `tests/gravity/dehnen/fmm_walk.cpp:1249` (also `tests/gravity/dehnen/fmm_walk.cpp:1249`; `tests/gravity/dehnen/fmm_walk.cpp:1695`; `cosmictiger/core/driver.hpp:78`; `cmake/features.cmake:47`)
- **Problem:** tests/gravity/dehnen/fmm_walk.cpp calls kickStepHierarchical at line 1249 without any guard, but driver.hpp declares that function only when COSMICTIGER_ENABLE_INTEGRATOR_HIERARCHICAL is on. That option defaults OFF, so any FMM build with BUILD_TESTING=ON (CMake's default) and the default integrators fails to compile cosmictiger. This includes the README manual and CUDA 13 builds and build.sh with tests enabled.
- **Failure scenario:** Run the README manual build (README.md:671-673, `cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release ...; cmake --build build/release`), the CUDA 13 example at README.md:700 (-DBUILD_TESTING=ON), or build.sh with "Build and run test suite: ON" and the default integrators. The compile fails with 'kickStepHierarchical' was not declared in this scope at tests/gravity/dehnen/fmm_walk.cpp:1249, so cosmictiger never links.
- **Fix:** Wrap the fmmWalkHierarchicalTerminalEnergy definition and its call (fmm_walk.cpp:1693-1697) in `#if COSMICTIGER_ENABLE_INTEGRATOR_HIERARCHICAL`; otherwise always run fmmWalkOrdinaryTimesteps. Add a default-options BUILD_TESTING=ON configure-and-build case to tests/check_build_variants.py so feature-guard mismatches in test sources are caught.

#### run_gravity_error.py checkpoint decoder ignores the runtime-algorithm block (flag bit 18): every gravity_error\_\* CTest fails

`tests_python#1` · **high** · bug · confirmed

- **Where:** `tests/run_gravity_error.py:45` (also `src/io/checkpoint.cpp:447`; `src/io/checkpoint.cpp:656`; `tests/gravity/nitadori/pmmm_evolution_helpers.py:68`; `tests/gravity/nitadori/pmmm_evolution_helpers.py:110`)
- **Problem:** run_gravity_error.readState() reads DriverParameters straight after the 120-byte header, but every checkpoint written since 844a55c0 puts a 5 x uint64 algorithm block (flag bit 18) there. The decode therefore fails with 'checkpoint driver layout mismatch' before any accuracy check, in every gravity_error\_\* test. The decoder also ignores the 16-byte global-timestep extension (bit 19) after the Driver.
- **Failure scenario:** Any build runs gravity_error_cpu, gravity_error_cuda or gravity_error_mpi. The first readState(checkpoints(baseline)[-1]) reads DriverParameters at byte 120. I checked this against a checkpoint written today (build/molecular_hot_cpu/tests/checkpoint_integrity/baseline/checkpoint.1/checkpoint.0.dat, flags=0x40000): the algorithm words (1,1,0,7,0) sit at byte 120 and the real Driver starts at byte 160 (a=0.0201, iteration=1). Reading at byte 120 gives a=5e-324 and iteration=0, so the test raises 'checkpoint driver layout mismatch' before it checks anything about gravity accuracy. This sampled-gravity-accuracy regression is now dead in every build.
- **Fix:** In readState, validate and skip the 5 x uint64 block when flags &amp; (1&lt;&lt;18): require words[0]==1 and the expected far-solver/integrator codes. Then read the optional PMMM extension. After the Driver, skip (and validate version==1 of) the 16-byte block when flags &amp; (1&lt;&lt;19). Better: move one shared checkpoint-prefix decoder into a tests module and use it from all five hand-rolled decoders, so the next layout change cannot leave one stale.


## Medium (74)

Edge-case and latent bugs, plus maintainability traps worth scheduling.

### Options, driver and startup

#### Runs started from Gadget4 ICs cannot restart: units are checked against startup options and the Gadget loader re-runs after readCheckpoint

`io_checkpoint_output_tools#4` · **medium** · bug · confirmed · file was being edited during the review

- **Where:** `src/core/driver.cpp:1391` (also `src/io/checkpoint.cpp:535-536`; `src/core/driver.cpp:1296-1298`; `src/core/driver.cpp:1336-1352`; `src/core/driver.cpp:1362-1365`)
- **Problem:** Runs started from initial.gadget4Restart write checkpoints whose headers carry the Gadget-derived units and G. A restart validates those headers bit-exactly against the massSolar-derived startup units before the Gadget override runs, so it is always rejected. The Gadget loader is also not skipped on restart, so if validation ever passed it would overwrite the restored particles, scale factor and duration.
- **Failure scenario:** Start from initial.gadget4Restart=ics and write checkpoint.N. Restarting with --checkpoint.readIndex=N aborts with "Checkpoint physics, particle layout or integrator differs": the header units came from the Gadget file, the options units from units.massSolar defaults. The background check at driver.cpp:1336-1352 would also fail because startRedshift has reverted.
- **Fix:** Derive the Gadget units, G and startRedshift in processOptions by reading the Gadget header on rank 0 before any readCheckpoint, so restart validation compares like with like. Skip particlesReadGadget4 and the params overrides when checkpoint.readIndex &gt;= 0. In fresh Gadget runs, skip initialize()'s IC generation and set only a, adot and tauMax.
- **Also reported as** `core_options_driver#15` (Gadget4 path re-reads input after a checkpoint restore, wastes a full IC generation, and leaves N-derived options stale) at `src/core/driver.cpp:1391`: Gadget4 input is read after a complete (wasted) cosmological IC generation and also after a checkpoint restore. It re-derives units and gm from the Gadget particle count but leaves numberOfParticles, softening, FoF linking length, fourierSize and mapSize on particlesPerDimension^3, and it ignores …

#### Global integrator runs every 'coarse-step' analysis/output on every global step

`core_options_driver#5` · **medium** · bug · confirmed · file was being edited during the review

- **Where:** `src/core/driver.cpp:1601` (also `src/core/driver.cpp:1667-1678`; `src/core/options.cpp:592-594`; `src/core/scenario_driver.cpp:171-176`)
- **Problem:** With run.timeIntegrator=global, fullEval is true on every raw global step and step counts global steps. So tracers, slices, HEALPix and Silo views, power spectra and FoF/Rockstar catalogues run at every global step, not once per coarse interval as the siloInterval and output docs describe. This can multiply analysis cost and disk output by t0/dt_min.
- **Failure scenario:** --run.timeIntegrator=global --analysis.groups=true --analysis.power=true: writes a groups.N directory and power.N.txt every step. Runtime becomes dominated by analysis and disk use explodes, versus 64 snapshots with the traditional integrator.
- **Fix:** Keep step as the iteration/restart counter, but track a separate coarse output index in global mode (e.g. floor((tau+eps)/t0)). Run analyses and snapshots only when that index advances, and use it for file numbering and siloInterval. Optionally clip the global dt to the next coarse boundary when outputs are enabled, so snapshots land exactly on coarse times.
- **Also reported as** `recent_commits_a#1` (Global integrator makes every step an output/analysis step (Silo cadence, power/groups, errorInterval, numbering)) at `src/core/driver.cpp:1602`: With run.timeIntegrator=global, both drivers force minrung=0, so fullEval is true and `step` increments on every adaptive global step. As a result, Silo snapshots (gated only by step % siloInterval), tracers, slices, views, power spectra, FoF groups, gravity error checks and the progress.txt header …

#### Driver silently overrides compute.bucketSize every step with a hard-coded 112-&gt;184 schedule (beyond MAX_BUCKET_SIZE)

`core_options_driver#4` · **medium** · bug · confirmed · file was being edited during the review

- **Where:** `src/core/driver.cpp:1654` (also `src/core/options.cpp:1229-1233`; `cosmictiger/gravity/kick.hpp:22-26`; `cosmictiger/gravity/kick_cuda_launch.hpp:129-132`; `docs/pmmm.md:110`)
- **Problem:** The cosmology/glass/Plummer driver overwrites compute.bucketSize every substep with a hard-coded 112-&gt;184 schedule tied to run fraction. This silently ignores the configured value (default 128, or the documented 32 for PMMM near FMM) and exceeds the MAX_BUCKET_SIZE cap that option validation enforces. It also forces a distributed setOptions on every substep.
- **Failure scenario:** --compute.bucketSize=32 (PMMM near FMM) or a GPU-tuned 64 is ignored in every cosmology run: trees use 112-184 particles per leaf, and benchmarks and tuning studies measure the hidden schedule. Silo metadata reports the overridden value.
- **Fix:** Delete the schedule (and the unused buckets[] array) and use the configured compute.bucketSize. If adaptive buckets are wanted, add an explicit opt-in option clamped to MAX_BUCKET_SIZE. Call setOptions only when a value actually changes.

#### Cosmological energy expansion term hard-codes gamma=5/3 for non-chemistry SPH

`core_options_driver#7` · **medium** · bug · confirmed · file was being edited during the review

- **Where:** `src/core/driver.cpp:1744` (also `src/core/options.cpp:1290`)
- **Problem:** For SPH cosmology without chemistry or star formation, the Layzer-Irvine expansion-work term uses +U da. The correct term is (3γ-4)U da, so for γ≠5/3 (e.g. the tested γ=7/5) energy.txt's energy_error drifts by (5-3γ)∫U da even though the SPH dynamics apply the correct (5-3γ) expansion term.
- **Failure scenario:** Cosmological SPH with --hydro.gamma=1.4 and chemistry off: each step adds U\*da instead of 0.2\*U\*da. energy_error in energy.txt drifts by ~0.8\*integral(U da), falsely reporting non-conservation or masking a real error.
- **Fix:** In the ideal-gas branch use `(3.0 * (getOptions().hydro.gamma - 1.0) - 1.0) * energies.therm`. Alternatively, compute ΣPV with sphParticlesPressure-based summation for every SPH run, so a single expression `-energies.therm + 3.0 * pressureIntegral` covers all cases. Consider adding a bound on max|dE| for the γ=7/5 cosmology case in run_variational_sph.py.

#### run.maxIterations is off by one in the cosmology driver and inconsistent with the scenario driver

`core_options_driver#6` · **medium** · bug · confirmed · file was being edited during the review

- **Where:** `src/core/driver.cpp:1860` (also `src/core/driver.cpp:1876-1877`; `src/core/scenario_driver.cpp:163-164`; `tests/run_cosmology.py:353`; `tests/run_cosmology.py:501`)
- **Problem:** The cosmology/glass/Plummer driver tests `thisIter > maxIterations` only after incrementing, so it runs maxIterations+1 substeps (one even for maxIterations=0). runFixedScenario runs exactly maxIterations. This contradicts the help text, and the test suite encodes both meanings.
- **Failure scenario:** A user asking for --run.maxIterations=10 on a cosmology run gets 11 steps and an extra checkpoint index. maxIterations=0 cannot be used to produce initial conditions without evolution.
- **Fix:** Use one meaning in both drivers: in driver() check `thisIter >= maxIterations` before starting a substep (next to the interrupt check), matching runFixedScenario. Then update the dependent tests (run_cosmology.py boundary-1, run_smoke.py, run_primordial_integration.py, run_position_selection.py, run_gravity_error.py, run_star_formation.py, run_global_integrator.py, run_pmmm_evolution.py, run_sph_only.py) and docs/glass_growth.md.

#### supernovaTemperature has no upper bound, so dense gas later exceeds the 1e8 K CIE ceiling and aborts

`chemistry_star_formation#17` · **medium** · bug · confirmed · file was being edited during the review

- **Where:** `src/core/options.cpp:157` (also `src/core/options.cpp:152`; `cosmictiger/sph/springel_and_hernquist/cooling_impl.hpp:270`; `cosmictiger/sph/springel_and_hernquist/equations.hpp:73`)
- **Problem:** starFormation.supernovaTemperature has no upper bound. Any T_SN above 1e8 K passes validation and SH03 calibration, but the hot-phase energy u_h = u_c + u_SN/(1+A0(rho/rho_th)^-0.8) later passes the 1e8 K CIE ceiling, and both the CPU and CUDA backends abort mid-run. The crossing is at about 5.6e3 rho_th for T_SN=2e8 K and about 1e5 rho_th for 1.1e8 K.
- **Failure scenario:** starFormation.supernovaTemperature=2e8: the run starts fine, then the first particle above ~5.6e3 rho_th makes evaluateAtEnergy fail and detail::equilibrium throws 'SH03 hot phase is outside the atomic cooling domain'.
- **Fix:** Reject supernovaTemperature &gt; springel_and_hernquist::kMaximumTemperature (1e8 K) in validateStarFormation and update the error message. If T_SN &gt; 1e8 K must be supported, either extend the CIE adapter above 1e8 K (fully ionized, bremsstrahlung extrapolation) or evaluate the cooling coefficient and composition at min(u_h, maxE). Optionally, have starFormationInitialize print the maximum supported density rho_th\*(A0/A_crit)^1.25 with A_crit = u_SN/(maxE-u_c)-1.
- **Also reported as** `recent_commits_b#4` (starFormation.supernovaTemperature above 1e8 K passes validation but aborts mid-run once dense gas pushes u_h past the CIE domain) at `src/core/options.cpp:156`: supernovaTemperature is validated only as &gt; 1e4 K, and makeModel checks only the threshold hot energy. For T_SN &gt; 1e8 K, the hot energy u_h = u_c + u_SN/(1+A) leaves the 10-1e8 K CIE table once A drops low enough (rho &gt; ~5.6e3 rho_th at 2e8 K, ~360 rho_th at 1e9 K), so runs that start cleanly abort …

#### gravity.openingAngle is never validated; theta &lt;= 0 crashes the tree build with 'Maximum depth exceeded'

`gravity_tree_walk_cpu#2` · **medium** · bug · confirmed · file was being edited during the review

- **Where:** `src/core/options.cpp:675` (also `src/gravity/dehnen/distributed_tree.cpp:82`; `src/gravity/tree_build.cpp:48`; `cosmictiger/gravity/fmm_acceptance.hpp:46`; `cosmictiger/gravity/opening_schedule.hpp:84`)
- **Problem:** gravity.openingAngle is never validated. For the global FMM path, theta &lt;= 0 (or NaN) makes the dehnen leaf criterion boxR &lt; 0.25\*theta/(1+theta) impossible to satisfy, so every non-empty cell is split until treeBuild aborts with 'Maximum depth exceeded' and nothing points at the option. Nonsensical values such as theta &lt; -1 or theta &gt; 1 are accepted silently.
- **Failure scenario:** A user asks for 'exact' forces with --gravity.openingAngle=0, or types a negative value. The run dies in treeBuild with THROW 'Maximum depth exceeded' after building 128 levels of empty siblings, and nothing points at the bad option.
- **Fix:** Validate the final value after the plummer default and schedule substitution, whenever the global FMM path uses it. Require it to be finite, within the normal float range (theta &gt;= FLT_MIN, matching the schedule check), and at most 1, and throw an error that names gravity.openingAngle. Consider the same upper bound for schedule thetas. Point users who want exact forces to gravity.errorInterval or run.test=force rather than theta = 0.

#### Global fixed scenarios never report completion when the accumulated stage end rounds below run.endTime

`recent_commits_a#2` · **medium** · bug · confirmed · file was being edited during the review

- **Where:** `src/core/scenario_driver.cpp:292` (also `src/core/scenario_driver.cpp:93`; `src/core/scenario_driver.cpp:150-152`; `src/core/scenario_driver.cpp:163`; `src/core/scenario_driver.cpp:280`)
- **Problem:** In global mode, runFixedScenario snaps tau to an accumulated stage end built from rounded products (duration\*steps/N summed over stages) and then requires `state.tau >= duration` exactly. When that sum rounds one ulp low (e.g. endTime=0.7 with numberOfSteps=3 gives 0.6999999999999998), a fully evolved run skips the closing diagnostics and plan.finish, reports 'paused at the iteration limit', and every restart reports incomplete again.
- **Failure scenario:** --run.scenario=sod --run.endTime=0.7 --run.numberOfSteps=3 --run.timeIntegrator=global: 0.7\*3/3 = 0.6999999999999998 &lt; 0.7. After evolving to the end, the run skips the closing-only synchronized diagnostics and plan.finish (finishSod's analytic comparison) and prints 'paused at the iteration limit'. A restart skips the already-finished stage and again reports incomplete, so the run can never finish.
- **Fix:** Make the final stage end exactly at `duration`, e.g. stageEnd = (last stage) ? duration : duration \* cumulativeSteps / numberOfSteps. Computing ends from cumulative steps also avoids drift across intermediate stages. Use that value consistently for the skip test, the loop bound and the snap. Also make the global `completed` test (and ideally the global skip and loop tests) use cosmology::endTimeReached(state.tau, duration) or the existing 1e-12 tolerance.

#### GPU HEALPix lookup tables are initialized only on locality 0

`analysis_lightcone_bh_power#2` · **medium** · bug · confirmed

- **Where:** `src/main.cpp:109` (also `src/main.cpp:109`; `src/particles/drift.cu:127`; `src/analysis/lightcone.cpp:176`)
- **Problem:** healpixInit(), the only initializer of the per-process \__managed_\_ x2pix/y2pix tables used by the device vec2pixNest, runs only inside hpx_main on locality 0. In multi-locality CUDA lightcone runs, every other rank gets ipix = faceNum\*nside^2 for all crossings, so its lc.healpix weights pile into 12 pixels.
- **Failure scenario:** Take a multi-locality run with compute.cuda=true and lightcone.enabled=true. On ranks &gt;= 1 the GPU drift (drift.cu:127) gets ipf=0 for every crossing, so entry.pix = faceNum\*nside^2. lcAddParticleEntries then deposits all of that rank's lightcone weights into the first pixel of each of the 12 base faces of lc.healpix.E.R. The CPU drift (chealpix) on the same rank would produce correct pixels.
- **Fix:** Call healpixInit() inside hpxInitCuda() right after cudaInit(), which already runs on every locality, so cudaInit's cudaDeviceReset cannot wipe the tables. Better, replace the runtime tables with compile-time constexpr/\__constant_\_ bit-interleave tables so no initialization exists. Remove the root-only call from main.cpp.


### Scenarios and INI files

#### Cosmology INIs and default.ini enable star formation: default builds reject them, and the documented collisionless recipe fails

`scenarios_configs#7` · **medium** · bug · confirmed · file was being edited during the review

- **Where:** `configurations/README.md:39` (also `configurations/README.md:16-17`; `configurations/README.md:52`; `configurations/README.md:137-138`; `default.ini:187`)
- **Problem:** default.ini, cosmology.ini, the five curvature presets and lcdm_ten_ages.ini all enable star formation. validateStarFormation rejects that in the default build (COSMICTIGER_ENABLE_STARS defaults to OFF in CMake and build.sh) and whenever hydro or chemistry is disabled. As a result, the collisionless recipe in configurations/README.md always fails on these INIs, the cosmology.ini table row omits star formation and the STARS requirement, and the descriptions of default.ini as the application defaults contradict the parser defaults (chemistry.enabled=false, starFormation.enabled=false).
- **Failure scenario:** Fresh build (STARS=OFF): `./cosmictiger --configuration=configurations/cosmology.ini` fails with "Star formation requires COSMICTIGER_ENABLE_STARS=ON". Any build: `--configuration=cosmology.ini --hydro.enabled=false --chemistry.enabled=false` fails at startup.
- **Fix:** Pick one policy. (a) Set starFormation.enabled=false in default.ini, cosmology.ini and the presets, and keep star formation in star_formation/lcdm.ini. (b) Keep it enabled, but add `--starFormation.enabled=false` to the collisionless recipe at configurations/README.md:39-40 and 137-138, and add star formation plus "requires COSMICTIGER_ENABLE_STARS=ON" to the cosmology.ini row.

#### Disc velocities hard-code GM=1 and RT relies on an INI-only acceleration; both scenarios run wrong physics without their INI

`scenarios_configs#8` · **medium** · bug · confirmed

- **Where:** `src/scenarios/disc.cpp:56` (also `src/scenarios/rayleigh_taylor.cpp:37`; `src/core/options.cpp:983-990`; `configurations/disc.ini:8`)
- **Problem:** initializeDisc sets v = sqrt(1/r), which assumes hydro.centralAcceleration (the GM of the imposed central force, |a| = centralAcceleration/r^2) equals 1, but it never reads or validates that option. The parser default is 0 and no disc-specific default is applied, so `--run.scenario=disc` without disc.ini launches an unbound ring, and any value other than 1 gives non-circular orbits. Rayleigh-Taylor reads hydro.zPlaneAcceleration consistently, but its default of 0 silently gives an undriven run with no instability.
- **Failure scenario:** `./cosmictiger --run.scenario=disc`: the ring disperses ballistically. `--run.scenario=rayleighTaylor` without the INI: nothing grows. disc.ini with centralAcceleration=2: velocities are a factor sqrt(2) sub-Keplerian.
- **Fix:** In initializeDisc use v = sqrt(opts.hydro.centralAcceleration / r), keeping the existing (x,y)/r direction, and throw if centralAcceleration &lt;= 0. In initializeRayleighTaylor, throw if zPlaneAcceleration &lt;= 0. Alternatively, set scenario defaults in processOptions when the keys are not explicit (centralAcceleration=1 for disc, zPlaneAcceleration=0.5 for rayleighTaylor), following the plummer openingAngle default at options.cpp:1748-1753.

#### Plummer radius sampler can loop forever: the numerical enclosed mass plateaus below the largest uniform deviate

`scenarios_configs#3` · **medium** · bug · confirmed · file was being edited during the review

- **Where:** `src/scenarios/plummer.cpp:30` (also `src/scenarios/plummer.cpp:50-52`; `src/scenarios/plummer.cpp:59-61`)
- **Problem:** sampleDensityDistribution integrates Plummer mass with a dr=0.001 trapezoid in double. menc stops growing at 1-1.338e-9 (last increment near r=30,008), but sampleUniform can return up to 1-2^-33, so 6 of the 2^32 possible draws make while(menc &lt; target) loop forever. With the fixed seed 42, a replay shows no hang for N &lt;= 134,217,728 (ppd &lt;= 512). The hang is latent for larger N or any other seed, and every draw costs about 2000 std::function+pow iterations on average.
- **Failure scenario:** `--run.scenario=plummer --initial.particlesPerDimension=1024`: with roughly 50% probability (deterministic for seed 42) initialization on locality 0 spins forever in sampleDensityDistribution.
- **Fix:** Replace the numerical integration with the closed-form inverse r = 1/sqrt(X^(-2/3) - 1). Truncate or resample draws whose r\*plummerRadius exceeds about 0.5, so tail particles do not wrap through the periodic box. If the integrator is kept, clamp target below the attainable plateau and cap r.

#### Polytropic-star stage viscosity (alpha 0 then 0.05) is silently ignored under the shipped viscosityEvolution=morrisAndMonaghan

`scenarios_configs#1` · **medium** · bug · confirmed

- **Where:** `src/scenarios/polytropic_star.cpp:234` (also `src/core/scenario_driver.cpp:157-160`; `src/sph/sph_hydro_cpu.cpp:82-83`; `src/sph/sph_hydro.cu:187-189`; `configurations/polytropic_star.ini:9-10`)
- **Problem:** runFixedScenario applies EvolutionStage::viscosityAlpha only to hydro.viscosityAlpha, and the CPU and CUDA hydro kernels read that value only when viscosityEvolution == none. The option default and every shipped INI use morrisAndMonaghan. As a result, the polytropic star's documented alpha=0 relaxation and alpha=0.05 evolution actually run with per-particle Morris-Monaghan alpha &gt;= viscosityAlphaMin = 0.1 and beta = 2\*alpha.
- **Failure scenario:** Run configurations/polytropic_star.ini. Stage 1 starts every particle at alpha=0.1, and the source term raises it further, instead of zero viscosity. Stage 2 runs with alpha &gt;= 0.1 instead of 0.05. The relaxed star and the final L2 error come from a different viscosity setup than the documented one.
- **Fix:** Throw in runFixedScenario when stage.viscosityAlpha &gt;= 0 is combined with a variable-viscosity model. Then either set viscosityEvolution=none in polytropic_star.ini, or extend EvolutionStage to carry a viscosity model and a beta. Under none, also override beta: setting alpha=0 alone leaves beta=2. Update the INI header and configurations/README.md to match what actually runs.

#### Scenario error norms (Sod, blast, polytrope) sample only locality-0 particles after domain decomposition

`scenarios_configs#6` · **medium** · bug · confirmed

- **Where:** `src/scenarios/sod.cpp:40` (also `src/scenarios/blast.cpp:38`; `src/scenarios/polytropic_star.cpp:135`; `src/scenarios/sod.cpp:36`; `src/scenarios/blast.cpp:33`)
- **Problem:** finishSod, finishBlast and finishPolytropicStar run only on locality 0 and sample only that locality's local sphParticles arrays. Every force step migrates gas particles between localities (domainsBegin/End), so in multi-locality runs the reported error norms and sod.txt/blast.txt describe only locality 0's subdomain. Separately, with the traditional integrator the compared densities lag the post-drift positions and end time by one active step, because only the global branch re-evaluates forces before finish.
- **Failure scenario:** Sod on 2 localities: locality 0 holds only one x-slab of the tube, so the reported errors describe that slab. An unwritable or missing output directory makes fopen return NULL, and fprintf then segfaults.
- **Fix:** Have each locality compute partial sums (l1, l2, lmax, norm1, norm2, and optionally sample rows) in an HPX plain action, and reduce them on locality 0, following the domainsGetLoads/countDomainSplits pattern. Alternatively, sample deterministically by global particle id. Use an owned, seeded std::mt19937, and THROW_ERROR if fopen fails. For the non-global integrators, run a final density/force evaluation at the end positions before calling finish, as the global branch does.

#### Sod (and KH) density ratio depends on resolution; the shipped sod.ini runs 1:0.152, not the documented 1:0.125

`scenarios_configs#4` · **medium** · bug · confirmed

- **Where:** `src/scenarios/sod.cpp:83` (also `src/scenarios/sod.cpp:100-101`; `configurations/sod.ini:8`; `src/scenarios/kelvin_helmholtz.cpp:30-33`; `configurations/README.md:109`)
- **Problem:** initializeSod rounds the two lattice dimensions independently and then recomputes rho_R = (rightDim/leftDim)^3. The shipped sod.ini (ppd=32: leftDim 15, rightDim 8) therefore runs rho_R = 0.1517, while its header documents 0.125; ppd 48 and 64 give 0.1420 and 0.1133. finishSod compares against the realized states, so each L1 is self-consistent, but resolution studies compare different Riemann problems. The KH analogue (ratio 2.06 at ppd=32) is cosmetic.
- **Failure scenario:** configurations/sod.ini at ppd=32 gives a density contrast of 6.6 instead of 8. Going from 32 to 64 changes the right-state density from 0.152 to 0.113, so the convergence comparison mixes two different problems.
- **Fix:** Derive one dimension from the other: rightDim = max(1, lround(cbrt(N/72))), leftDim = 2\*rightDim. The ratio is then exactly 1/8 at every resolution. For KH, an exact ratio of 2 is impossible with equal-mass cubic lattices, so print or document the realized ratio. Print the realized physical densities in the detailPrint instead of the neighbour-normalized values, and mention gamma=5/3 in the sod.ini header.


### Chemistry and star formation

#### CPU chemistry batch spawns one std::thread per HPX worker from inside an HPX task (oversubscription, blocks the worker)

`chemistry_star_formation#11` · **medium** · bug · confirmed · file was being edited during the review

- **Where:** `src/chemistry/primordial_chemistry.cpp:53` (also `src/chemistry/primordial_chemistry.cpp:31`; `src/chemistry/primordial_chemistry.cpp:52`)
- **Problem:** The CPU chemistry batch spawns hpx::get_num_worker_threads()-1 raw std::threads from inside an HPX worker and blocks that worker in join(). Under HPX's default 'balanced' binding the new threads inherit the worker's single-PU affinity, so the batch is effectively serialized on one core. In unpinned runs it oversubscribes the --hpx:threads budget instead.
- **Failure scenario:** --hpx:threads=12 on a 24-core node shared with other jobs: chemistry phases use ~23 runnable threads, exceeding the intended core budget and slowing both HPX and chemistry threads.
- **Fix:** In the SPH bridge, run the batch on the HPX pool (hpx::experimental::for_loop or chunked hpx::async, each chunk with its own StrictFloatEnvironment). Keep the std::thread path only for standalone library use. If raw threads must stay, reset their affinity to the process mask before running.

#### chemistry.maxSubsteps above 8192 is silently ineffective: hard-coded maxAttempts caps the solver

`chemistry_star_formation#3` · **medium** · bug · confirmed · file was being edited during the review

- **Where:** `src/sph/sph_chemistry.cpp:170` (also `cosmictiger/chemistry/primordial_solver.hpp:19`; `cosmictiger/chemistry/primordial_solver.hpp:435`; `cosmictiger/chemistry/primordial_solver.hpp:509`)
- **Problem:** integrateChemistryBatch forwards chemistry.maxSubsteps but leaves SolverOptions::maxAttempts at its compiled default of 8192. advanceWithModel returns StepLimit once the attempt loop is exhausted. Any chemistry.maxSubsteps above 8192 (effectively above 8192 minus rejected trials) therefore has no effect, and nothing validates or reports this.
- **Failure scenario:** User sets chemistry.maxSubsteps=50000 to get through hot dense molecular gas; the solver still returns StepLimit after 8192 attempts and integrateChemistryBatch throws 'Chemistry failed ... status 4', aborting the run despite the option.
- **Fix:** Derive maxAttempts from maxSubsteps in integrateChemistryBatch, for example 2\*maxSubsteps with saturation, or expose chemistry.maxAttempts with validation and add it to the checkpoint parameters. Document that rejected trials count against the attempt budget.

#### Serial per-particle EOS/CIE loops repeat expensive work (duplicate EOS pass, 3 bisections per dense particle, constant endpoints recomputed)

`chemistry_star_formation#9` · **medium** · smell · confirmed · file was being edited during the review

- **Where:** `src/sph/sph_chemistry.cpp:463` (also `src/core/driver.cpp:506`; `src/sph/star_formation.cpp:229`; `cosmictiger/sph/springel_and_hernquist/cooling_impl.hpp:116`; `src/sph/star_formation.cpp:86`)
- **Problem:** Several serial per-particle thermodynamics passes repeat expensive work. The chemistry driver inverts the full EOS twice in a row (pressure integral, then cache refresh), and the SF driver repeats starFormationRefreshEos at driver.cpp:472/514. The CPU SH03 refresh runs three ~32-evaluation CIE bisections per dense particle on one thread, recomputing constant endpoints every time. The projection repeats the hot-phase inversion that evaluateEquilibrium already did.
- **Failure scenario:** 1e6 gas particles on one locality with molecular9: each serial EOS pass costs tens of seconds, and it is done twice back to back; 1e5 dense SF particles cost tens of seconds per step in starFormationRefreshEos on the CPU path while other HPX workers idle.
- **Fix:** Refresh first at driver.cpp:506-507 and sum the pressure integral from sphParticlesChemistryEos (or have the refresh return the sum). Skip the starFormationRefreshEos at driver.cpp:514 when the refresh at 472 already ran. Parallelize the per-particle EOS/SH03 loops with hpx::for_loop, one StrictFloatEnvironment per chunk. Cache the 10 K/1e8 K endpoint states in AtomicCooling (or pass them in). Let evaluateEquilibrium return the hot AtomicEquilibrium.
- **Also reported as** `recent_commits_b#2` (CPU SH03 per-particle work is single-threaded and each CIE inversion re-evaluates both domain endpoints) at `src/sph/star_formation.cpp:229`: On the CPU branch, the SH03 EOS refresh, timestep limit and source prepare loops run serially on one thread per locality. Each star-forming particle costs three evaluateEquilibrium calls per refresh, and the refresh runs over all gas regardless of rung.

#### Decoupled winds in gas denser than n_H=1e4 cm^-3 are sent to the non-equilibrium solver, which rejects them and aborts the run

`recent_commits_b#5` · **medium** · bug · confirmed · file was being edited during the review

- **Where:** `src/sph/star_formation.cpp:463` (also `cosmictiger/chemistry/primordial_types.hpp:189`; `src/sph/sph_chemistry.cpp:182-186`; `cosmictiger/sph/star_formation_kernel.hpp:232-236` (CUDA twin); `src/core/options.cpp:797` (help text))
- **Problem:** A decoupled wind's chemistry is advanced at its own SPH density, starting with the opening half right after launch. A wind launched from SH03 gas above n_H = 1e4 cm^-3 (~3e4 rho_th), which the dense closure deliberately allows, makes validateComposition return UnsupportedState, and integrateChemistryBatch aborts the whole run. Commit 84228ed2 also dropped the density limit from the chemistry.network help text.
- **Failure scenario:** A high-resolution run where star-forming gas exceeds n_H=1e4 (about 3e4 ρ_th) launches a wind. On its next opening half the wind is not projected, so chemistry runs at n_H&gt;1e4 and the run aborts with "Chemistry failed on rank R batch index J status ...".
- **Fix:** Make the behaviour explicit. Options: evolve decoupled winds above the chemistry density limit at a clamped n_H (the wind is hydro-decoupled, so its ISM density is not physically its own); keep the SH03 closure until the density falls below the limit; or skip wind launches above the limit and count them in a diagnostic. Apply the same change to the CUDA path, which pushes inputs[index].density.
- **Also reported as** `chemistry_star_formation#5` (Hard 1e4 cm^-3 density ceiling aborts the whole run for dense winds (and any chemistry-only collapse)) at `cosmictiger/chemistry/primordial_types.hpp:189`: validateComposition rejects n_H &gt; 1e4 cm^-3 with UnsupportedState, and integrateChemistryBatch turns any single particle's failure into a run-level exception. One SH03 wind, which is integrated with ordinary chemistry at its current launch-like density, or any chemistry-only collapse above the …


### Core runtime

#### Fatal-signal handler kills the process when a second thread faults, truncating the first thread's stack trace

`core_runtime_infra#3` · **medium** · bug · confirmed

- **Where:** `src/core/stack_trace.cpp:86` (also `src/core/stack_trace.cpp:80`; `src/core/stack_trace.cpp:125`; `src/core/interrupt_profile.cpp:93`)
- **Problem:** When a second thread enters the fatal-signal handler while the first is still symbolizing, the process-wide `reporting` flag makes it reset the signal to SIG_DFL and re-raise at once. That kills the process, so the first thread's trace never reaches stderr or stack_trace.txt. The window is widened by Boost's per-thread (thread_local) libbacktrace state being cold on worker threads.
- **Failure scenario:** A bad index in an HPX parallel loop makes several workers SIGSEGV within microseconds. Thread A starts writing the header and the trace. Thread B immediately calls terminateWithSignal and the process dies, so stderr and stack_trace.txt get at most the header and 'Signal: SIGSEGV' with no trace. This is exactly when the trace is most needed.
- **Fix:** Store the reporting thread's tid (syscall(SYS_gettid)) in an atomic. Terminate immediately only when that same thread re-enters. Other faulting threads should wait in an async-signal-safe loop (pause/sigsuspend/nanosleep) until the reporter calls terminateWithSignal. Add a bounded watchdog, for example terminate after N seconds, so a deadlocked symbolizer cannot hang a batch job. Use pthread_sigmask instead of sigprocmask for portability.


### CUDA infrastructure

#### CUDA_CHECK evaluates its argument twice, only prints, and is an unbraced if

`cuda_infra#2` · **medium** · bug · confirmed

- **Where:** `cosmictiger/cuda/cuda.hpp:44` (also `cosmictiger/gravity/kick_cuda_launch.hpp:300-310`; `cosmictiger/gravity/kick_cuda_launch.hpp:315-318`; `src/gravity/dehnen/ewald_indices.cu:75`; `src/sph/sph.cu:17`)
- **Problem:** CUDA_CHECK evaluates the wrapped call a second time on failure, reports only through printf on stdout, keeps executing, and is an unbraced `if` that an else can bind to. Real failures, such as cudaMalloc in the kick launcher after dataSize has already been updated, therefore turn into later unrelated crashes or silent continuation.
- **Failure scenario:** (1) kick_cuda_launch.hpp:300-310 frees the old devDataPtr, sets `dataSize = allocSize`, then calls `CUDA_CHECK(cudaMalloc(&devDataPtr, allocSize))`. On out-of-memory the macro calls cudaMalloc again, prints and continues. devDataPtr is null or the already-freed pointer. dataSize already claims the new size, so the allocation is never retried, and this kick and every later one copy parameters into that pointer.
- **Fix:** Use `#define CUDA_CHECK(expr) do { cudaError_t const cudaCheckStatus_ = (expr); if (cudaCheckStatus_ != cudaSuccess) { THROW_ERROR("CUDA error %s at %s:%i\n", cudaGetErrorString(cudaCheckStatus_), __FILE__, __LINE__); } } while (0)`, or hpx::cuda::experimental::check_cuda_error. Add a separate non-fatal CUDA_WARN that evaluates once and writes to stderr, for advisory calls (cudaAdviseMemory, prefetch) that may legitimately fail.
- **Also reported as** `tests_other_cpp#3` (Production CUDA_CHECK double-evaluates its argument; cuda_memory stub claims to match but does not) at `cosmictiger/cuda/cuda.hpp:44`: Production CUDA_CHECK evaluates its argument twice on failure, is not statement-safe (bare if, unparenthesized argument) and only prints. No current call site triggers the dangling-else case, and the re-evaluated calls are benign.

#### 2 GiB limits: int byte counts in copyCudaMemory and DeviceVector, and nullptr pool returns that callers do not check

`cuda_infra#7` · **medium** · bug · confirmed

- **Where:** `cosmictiger/cuda/cuda_memory.hpp:67` (also `src/cuda/cuda_memory.cu:103-118`; `src/cuda/cuda_memory.cu:194`; `cosmictiger/cuda/device_vector.hpp:154`; `cosmictiger/cuda/device_vector.hpp:223-239`)
- **Problem:** Byte counts are truncated to int: copyCudaMemory takes an int, and DeviceVector keeps int size and capacity and doubles an int capacity. In CPU mode, copying a DeviceVector&lt;RockstarParticle&gt; larger than about 29.8M elements (72 B each) passes a negative int to memcpy and segfaults. In GPU mode, the pool refuses any request over 2 GiB with nullptr. Only DeviceVector::grow checks that nullptr. power.cu and the small-allocation sites write through it unchecked, while the CPU backend throws instead.
- **Failure scenario:** CPU-mode ROCKSTAR on a halo of more than about 27M particles (RockstarParticle is 80 bytes): the by-value copy into rockstarSeeds, or growth, turns the byte count into a negative int, and host memcpy receives size_t(negative) and segfaults. In GPU mode the same halo hits the 2 GiB pool cap and aborts with "OOM in device_vector".
- **Fix:** Make copyCudaMemory take size_t (with a size_t loop on the device). Use size_t, or checked int arithmetic with an explicit ALWAYS_ASSERT limit, for DeviceVector byte counts and capacity doubling. Give cudaAllocate one failure behavior on every backend: throw std::bad_alloc on the host and assert with size and pool state on the device. Check the result at power.cu, lightcone.cu, bh.cu and drift.cu, or route them through a checked helper.

#### CudaUnorderedMap::rehash leaks the old table and copies every entry instead of moving it

`cuda_infra#3` · **medium** · bug · confirmed

- **Where:** `cosmictiger/cuda/cuda_unordered_map.hpp:35` (also `cosmictiger/core/containers.hpp:154-155`; `cosmictiger/cuda/cuda_unordered_map.hpp:136`; `src/analysis/lightcone.cpp:919-937`)
- **Problem:** Pair's user-declared copy operations suppress its move operations, so CudaUnorderedMap::rehash deep-copies LightconeParticles entries and shallow-copies tree-node arrays, which then alias the originals' neighbor lists. It also leaks the entire previous bucket table together with the entries' buffers, because DeviceVector's destructor does not destroy elements. This happens rarely, when lcInit adds enough new pixel keys after an nside change.
- **Failure scenario:** lcInit runs after every lightcone flush (driver.cpp:1547) and inserts pixel keys with operator[]. nside shrinks with tau, so a rank's pixel range shifts onto keys it did not hold before. When those insertions push size()/buckets above 2, rehash copies every entry, including deep copies of particle vectors not yet reset in that loop. It also leaks the whole previous table and all previously reserved per-pixel capacity from the 8 GiB pool.
- **Fix:** Add `Pair(Pair&&) = default; Pair& operator=(Pair&&) = default;` (or drop the user-declared copy operations). In rehash, move the entries, then after the swap destroy the old inner buckets (for example `newBuckets.resize(0)` once the moved-from entries are empty). The robust fix is to fix cuda_infra#1 and cuda_infra#5 so that DeviceVector destroys and copies elements correctly. Also consider erasing stale pixel keys in lcInit instead of keeping them forever.

#### DeviceVector's device-side collective contract breaks for swap/move, nested element types and popBack

`cuda_infra#6` · **medium** · bug · confirmed

- **Where:** `cosmictiger/cuda/device_vector.hpp:108` (also `cosmictiger/cuda/device_vector.hpp:158-168`; `cosmictiger/cuda/device_vector.hpp:57-76`; `cosmictiger/cuda/device_vector.hpp:282-288`; `cosmictiger/cuda/device_vector.hpp:14-42`)
- **Problem:** DeviceVector's device-side contract is broken for several operations. swap() and the move operations built on it are run by every thread with no barrier. construct()/destruct() build elements one per thread even when the element's own constructor or destructor is block-collective (nested DeviceVector, RockstarTree, LightconeTreeNode). popBack() lacks a leading barrier. In multi-warp blocks each of these is a race or undefined behavior, but no current kernel uses them that way, so it is a latent trap.
- **Failure scenario:** (A) In a 128-thread SPH-style kernel, `workspace.list = std::move(tmp)` on shared vectors: if warp 0 finishes its swap before warp 1 starts, warp 1 swaps back. The net effect is no swap, or a mix of ptr\_ from one vector and size\_ from the other. (B) A device-side resize of DeviceVector&lt;DeviceVector&lt;X&gt;&gt;, or of a vector of RockstarTree or LightconeTreeNode, leaves every element not constructed by thread 0 with garbage ptr\_, size\_ and capacity\_.
- **Fix:** Make swap collective (syncthreads; thread 0 swaps; syncthreads), or forbid swap and move on the device. Add a leading syncthreads() to popBack. In construct()/destruct(), static_assert that T is trivially default-constructible and trivially destructible (or give nested types a collective per-element path), and assert 1-D blocks. Also have ~DeviceVector (and move assignment) call destruct(0, size\_) before destroy();

#### DeviceVector copy is a raw memcpy (shallow for element types that own storage), and the pool has no double-free detection

`cuda_infra#5` · **medium** · bug · confirmed

- **Where:** `cosmictiger/cuda/device_vector.hpp:154` (also `cosmictiger/cuda/device_vector.hpp:147-150`; `src/cuda/cuda_memory.cu:120-127`; `cosmictiger/cuda/cuda_unordered_map.hpp:35`; `cosmictiger/cuda/cuda_unordered_map.hpp:151`)
- **Problem:** DeviceVector's copy operations memcpy elements, so element types that own pool storage (LightconeTreeNode, RockstarTree) end up sharing buffers, and the pool allocator has no double-free detection. Today this is reachable only through treeMap rehash, where leaking the originals masks it. It will turn into use-after-free and double free as soon as element destruction is added.
- **Failure scenario:** `DeviceVector<RockstarTree> copy = trees;` followed by `copy.resize(0); trees.resize(0);` destroys both copies of each element's neighbors, so each pool block is released twice. The next two cudaAllocate calls of that size hand the same block to two owners, and data is silently corrupted. This path is reachable today when CudaUnorderedMap rehashes treeMap entries (a Pair copy becomes a DeviceVector&lt;LightconeTreeNode&gt; copy).
- **Fix:** On the host, when !std::is_trivially_copyable_v&lt;T&gt;, destroy the existing elements and copy element by element (assignment or std::uninitialized_copy); keep memcpy for trivially copyable T. Alternatively, constrain or delete the copy operations for non-trivially-copyable T. In CudaMemory, stamp a freed marker in the block header on free, clear it on allocate, and ALWAYS_ASSERT on a second free, at least in debug builds.

#### GPU selection uses global rank % device count, and the HPX event-pool warm-up opens a context on every visible GPU

`cuda_infra#10` · **medium** · bug · confirmed

- **Where:** `src/cuda/cuda.cpp:159` (also `src/cuda/cuda.cpp:163-168`; `src/core/hpx.cpp:137-151`; `src/cuda/cuda_stream.cpp:64`)
- **Problem:** Each locality picks its GPU as global HPX rank % visible device count. That oversubscribes one GPU and idles another unless ranks are placed contiguously per node. In addition, HPX's cuda_event_pool, warmed during cudaInit, calls cudaSetDevice and creates 128 events on every visible device, so every locality opens a primary context on every visible GPU. That wastes device memory and can fail in exclusive-process mode.
- **Failure scenario:** Two nodes with two GPUs each run four localities with cyclic placement (mpirun --map-by node, or srun -m cyclic). Node A runs ranks 0 and 2, and both select device 0: GPU 1 idles and GPU 0 is shared, roughly halving GPU throughput. With four localities on a four-GPU node and all GPUs visible, the event pool gives each GPU four contexts (hundreds of MiB each), even though each rank uses only one device.
- **Fix:** Derive a node-local rank (keep the per-locality hostnames from detectProcessesPerNode, or read SLURM_LOCALID / OMPI_COMM_WORLD_LOCAL_RANK) and use localRank % count in both cudaSetDevice() and cudaGetDevice(). Document, or warn at startup, that each locality should see exactly one GPU (CUDA_VISIBLE_DEVICES or --gpu-bind), so the HPX event pool touches only that device.

#### Every CudaStream builds a new HPX target: cudaGetDeviceProperties (1.4 ms measured) plus a stream create/destroy per object

`cuda_infra#9` · **medium** · smell · confirmed

- **Where:** `src/cuda/cuda_stream.cpp:99` (also `src/cuda/cuda_stream.cpp:152-153`; `src/analysis/rockstar.cu:73`; `cosmictiger/gravity/kick_cuda_launch.hpp:429`; `src/particles/drift.cu:215`)
- **Problem:** Every CudaStream builds a new hpx::cuda::experimental::target. Its constructor unconditionally calls cudaGetDeviceProperties (about 1.4 ms per call, measured in docs/cuda_kick_shared_memory.md), and each object also creates and destroys a CUDA stream. All GPU launch sites (kick, drift, SPH, BH, power, lightcone, rockstar) construct one per call, and rockstar does so on every label-propagation iteration, with no stream reuse.
- **Failure scenario:** rockstarFindSubgroupsCuda (rockstar.cu:70-80) creates a CudaStream per label-propagation iteration. A large group that needs about 10 iterations spends about 14 ms just constructing targets, and this repeats for every group above 4096 particles at every recursion level. Each kick, each drift (drift.cu:215), each SPH pass (sph.cpp:668) and each BH evaluation pays the same cost.
- **Fix:** Lease streams from a per-device pool that is filled lazily after cudaInit's cudaDeviceReset. A thread-safe free list of Impl objects, each owning one target or stream, keeps today's exclusive-use semantics. Alternatively, own a cudaStream_t directly and wait with hpx::cuda::experimental::detail::get_future_with_event(stream, device), which is exported at cuda_future.hpp:236; this avoids constructing a target at all.


### Particles, domains and drift

#### SPH migration repacks every particle serially each step and keeps a full-size copy afterwards

`particles_domain_drift#5` · **medium** · smell · confirmed

- **Where:** `src/particles/domain.cpp:528` (also `src/particles/domain.cpp:570-574`; `src/particles/domain.cpp:395-408`; `src/particles/domain.cpp:446-453`)
- **Problem:** Every SPH force evaluation, including every substep, serially packs all local particles into 248-264 B SphMigrationParticle records and serially unpacks them, even when none leave the domain (always the case on one locality). The static retainedSphParticles buffer keeps its N-record capacity afterwards because clear() does not free memory, and each particle repeats std::string option comparisons.
- **Failure scenario:** Single-locality SPH run with 2x256^3 particles: every step copies about 8.6 GB twice on one thread and keeps about 8.6 GB of extra memory permanently. Multi-node runs see the same doubling of peak memory, which can exhaust a node's memory at the first migration.
- **Fix:** Scan in parallel for departing particles only and send just those. Compact the particle and gas arrays in place; a stable parallel compaction preserves gas order if catIndex/particleIndex are fixed up for moved entries. Append received records at the end. If the full repack is kept: parallelize pack and unpack, release the buffer with Vector&lt;SphMigrationParticle&gt;().swap(retainedSphParticles), and compute the option checks once per call as bools.

#### Cache-line fetch actions read particle arrays without the lock that guards reallocation

`particles_domain_drift#3` · **medium** · bug · confirmed

- **Where:** `src/particles/particles.cpp:831` (also `src/particles/particles.cpp:853`; `src/particles/particles.cpp:878`; `src/particles/particles.cpp:1405`; `src/particles/particles.cpp:1735`)
- **Problem:** particlesFetchCacheLine serves remote ranks by reading particlesX/Ty/Sph and size without the shared lock that particlesGetLocal takes. Dehnen GPU staging reallocates and frees those arrays mid-kick under the exclusive lock whenever ghost appends exceed capacity. In multi-locality CUDA FMM runs, a concurrent position/type line fetch can therefore read freed managed memory. The fetch can come from another rank's CPU-walk sinks, or from another rank's hydro staging, which reads types via the cache. The result is intermittent SIGSEGV or garbage positions.
- **Failure scenario:** Multi-locality CUDA FMM run. Rank A finishes its walk and stages GPU work, appending ghosts beyond capacity, which reallocates and frees the old managed arrays. Meanwhile rank B's CPU-path walk (multi-rank checklist) or its hydro type read runs ParticlesFetchCacheLineAction on A, which reads the freed managed memory: SIGSEGV or garbage positions. This is intermittent and happens whenever the ghost count exceeds the 5% capacity headroom.
- **Fix:** Take std::shared_lock&lt;SharedMutexType&gt;(sharedMutex) inside particlesFetchCacheLine, and for consistency in the group fetchers; the cost is negligible next to the RPC. Alternatively, reserve the ghost capacity before the walk starts so staging never reallocates while remote readers are active. Remove the unused Rungs/Vels cache paths.
- **Also reported as** `gravity_dehnen_global_fmm#2` (GPU staging reallocates particle arrays while peers read them through the unlocked particle cache-line server) at `src/gravity/dehnen/gpu_staging.cpp:155`: Between remote roots, dehnenStageGpu reallocates the local particle arrays (memcpy plus cudaFree) under the exclusive particles mutex. particlesFetchCacheLine, the server behind remote particlesGlobalReadPos, reads those arrays with no lock.

#### Checkpoint writes the ParticleInteger rung boundaries with sizeof(int)

`particles_domain_drift#7` · **medium** · bug · confirmed

- **Where:** `src/particles/particles.cpp:1582` (also `src/particles/particles.cpp:1622-1625`; `cosmictiger/particles/particles.hpp:46`; `src/particles/particles.cpp:1032`)
- **Problem:** particlesSave/particlesLoad handle the ParticleInteger rung stack with sizeof(int). In a LONG_LONG_PART_INT build, fwrite(rungBegins.data(), sizeof(int), n) stores only the first n/2 8-byte entries, so after a restart the upper half of the stack is zero. In the next ascending ladder of a hierarchical run, those stages then use [0,N) as the source/sink range, slow particles enter fast-stage forces, and exact restart is lost.
- **Failure scenario:** LONG_LONG_PART_INT build, hierarchical run with rungBegins = {b0,b1,b2,b3}. The file holds only b0 and b1. After restart rungBegins = {b0,b1,0,0}, so the next ascent pops the wrong current ranges and exact restart is lost. Related problems in the same build: kNoParticleIndex becomes +4294967295 instead of -1, and int locals truncate above 2^31 local particles ('int oldsz = size;' at 1032, and lines 1048, 786, 614, 1780, 1897).
- **Fix:** Make the rung-stack payload width explicit. Write the count as a fixed-width integer read into its own local variable (not the reused `size`), and write rungBegin and the elements with sizeof(ParticleInteger), or convert them to int64_t. For the default int build, sizeof(ParticleInteger) is 4, so its format does not change. In tests/gravity/nitadori/pmmm_evolution_helpers.py, parse rung_begin and rung_begins with integer_code.


### I/O, checkpoints and tools

#### Rank 0 reads and FNV-hashes every locality's checkpoint file serially on every write and every restart

`io_checkpoint_output_tools#3` · **medium** · smell · confirmed

- **Where:** `src/io/checkpoint.cpp:289` (also `src/io/checkpoint.cpp:199-219`; `src/io/checkpoint.cpp:349-367`; `src/io/checkpoint.cpp:1410-1415`)
- **Problem:** Rank 0 alone byte-wise FNV-hashes every locality's checkpoint file after each write (manifest) and again before each restart (validation), so checkpoint and restart cost grows with total checkpoint size at about 1 GB/s on one core while the other localities sit idle.
- **Failure scenario:** A 2048^3 dark-matter run on 64 nodes has a checkpoint of roughly 250 GB. Rank 0 must stream all of it at about 1 GB/s, so each checkpoint write takes several extra minutes, and the same again before a restart, with 63 nodes idle. Checkpoint cost grows with total size instead of per-rank size.
- **Fix:** Compute size and FNV while writing each rank file, or after its fsync, and return a CheckpointFileInfo from WriteCheckpointAction so rank 0 only formats the manifest. On restart, rank 0 parses the manifest and sends each locality its expected (size, checksum). Each locality verifies its own file in parallel in a first phase, and loading starts only after every verification succeeds, which keeps the documented validate-before-load guarantee.

#### Algorithm block writes a nonzero expansion order for gravity.method.far=none, making the checkpoint unreadable

`io_checkpoint_output_tools#1` · **medium** · bug · confirmed

- **Where:** `src/io/checkpoint.cpp:627` (also `src/io/checkpoint.cpp:634-637`; `src/core/options.cpp:1061`; `src/core/options.cpp:1031-1035`)
- **Problem:** In gravity-enabled builds, a run with gravity.method.far=none (accepted when gravity.enabled=false) writes checkpoints whose algorithm block has far code 0 and an expansion order of at least 3. readCheckpointAlgorithms() requires order 0 for code 0, so every checkpoint such a run writes is unreadable. Restarting an SPH-only-build checkpoint in a gravity build is also rejected, at the order comparison.
- **Failure scenario:** Gravity build (e.g. degrees 4-6). A hydro-only scenario runs with --gravity.enabled=false --gravity.method.far=none. writeCheckpoint succeeds and the manifest validates. A restart with --checkpoint.readIndex=N aborts in readCheckpointSelection()/readDriverState() with "Invalid checkpoint algorithm selection extension".
- **Fix:** In checkpointAlgorithms() write `opts.gravity.method == "none" ? 0 : opts.gravity.fmmOrder` for the order slot. The writer and the readDriverState comparison both call this function, so they stay consistent. Do not zero fmmOrder in processOptions, because setFmmOrder(0) aborts in gravity builds. Add a checkpointState case with method none in a gravity build. Alternatively, reject method.far=none in gravity builds to match the help text.
- **Also reported as** `recent_commits_a#3` (Checkpoint algorithm extension records FMM method/order for gravity-disabled runs, breaking hydro-only restart portability) at `src/io/checkpoint.cpp:624`: For gravity.enabled=false runs in gravity-capable builds, checkpointAlgorithms still records the defaulted far method and fmmOrder (expansionDegree+1). Restart restores and exact-matches these, so hydro-only checkpoints no longer move between SPH-only builds, builds with a different or smaller …

#### Restart from an older checkpoint aborts mid-run when the original run already wrote a checkpoint at the same iteration

`io_checkpoint_output_tools#2` · **medium** · bug · confirmed

- **Where:** `src/io/checkpoint.cpp:1383` (also `src/io/checkpoint.cpp:52`; `src/io/checkpoint.cpp:1386-1390`; `src/io/checkpoint.cpp:1417-1419`; `src/core/driver.cpp:1900-1905`)
- **Problem:** Restarts skip the existing-checkpoint scan. A restart from checkpoint K in the original output.directory therefore aborts, through THROW_ERROR/abort, when it reaches an iteration the original run already wrote (always the final checkpoint, and redshift-list checkpoints) or a leftover .incomplete staging directory. This loses all work since K and skips the final diagnostics.
- **Failure scenario:** A run completes and writes final checkpoint.F. The user restarts from checkpoint.K (K&lt;F), for example to regenerate Silo output with a different output.units, which docs/units.md says is outside the restart contract. The restarted run reaches iteration F and aborts with a stack trace in the final writeCheckpoint, skipping the final diagnostics.
- **Fix:** During restart startup on rank 0, next to validateCheckpointDirectory, scan output.directory for checkpoint.&lt;i&gt; and checkpoint.&lt;i&gt;.incomplete with i &gt; readIndex. Fail fast with actionable guidance, such as moving them aside or restarting in a copied directory, or accept an explicit opt-in to move them. Alternatively, during a restart, warn and skip a write whose numbered target already exists instead of aborting.

#### groups2halofunc counts subhalos as halos and normalizes by a hard-coded 684 Mpc box

`io_checkpoint_output_tools#6` · **medium** · bug · confirmed

- **Where:** `src/tools/groups2halofunc.cpp:106` (also `src/tools/groups2halofunc.cpp:83-84`; `src/tools/groups2halofunc.cpp:20-34`; `src/analysis/groups.cpp:404-423`; `src/analysis/rockstar_catalogue.cpp:825`)
- **Problem:** groups2halofunc builds its mass function from groups.N.R.dat, which now holds one entry per ROCKSTAR halo, subhalos included, with inclusive SO masses and no host flag. Subhalos are therefore counted as separate halos. The volume comes from a hard-coded 684 Mpc default instead of the box length stored in groups.dat, and the ranks are discovered by probing files.
- **Failure scenario:** For any snapshot with substructure, the printed dn/dlog10M is inflated at low mass. Running on a 100 Mpc box without --analysis.boxSize divides by 684^3 Mpc^3, which is off by a factor of about 300. A missing middle rank file silently truncates the catalogue.
- **Fix:** Compute the function from rockstar.R.dat records with pid == -1, or add a host flag to the catalogue the tool reads. Take boxLength from groups.dat (GroupHeader) and the rank count from rockstar_hosts.complete, and remove the 684 default. Guard against empty catalogues and dlog == 0. Drop the hard-coded particle mass in the progress print.


### Gravity: PMMM (nitadori)

#### Collective dispatch throws away the exception text and aborts the MPI job on any PMMM prepare/transform/destroy error, even with one locality

`gravity_nitadori_pmmm#1` · **medium** · bug · confirmed

- **Where:** `cosmictiger/gravity/nitadori/collective_dispatch.hpp:31` (also `src/gravity/nitadori/collective_dispatch.cpp:24-27`; `src/gravity/nitadori/solver.cpp:113-116`; `src/gravity/nitadori/fft.cpp:274-276`; `src/gravity/nitadori/fft.cpp:400-403`)
- **Problem:** dispatchCollectively handles every PMMM prepare/transform/destroy failure with catch(...) and calls abortCollectiveDispatch() straight away. The abort worker prints only a fixed line before MPI_Abort(..., 86), so the exception text (FFT allocation or plan failure, bad_alloc, non-representable moments, MPI errors) and the failing rank are lost, even when only one locality runs, and main.cpp's error and cleanup reporting for these phases is unreachable.
- **Failure scenario:** A user picks a mesh or order whose FFT buffers don't fit. fftwf_alloc returns null, collectiveCheck throws the 'Unable to allocate…' error on all ranks, the continuation catches it, and the process exits with code 86 after printing only the generic message. An out-of-memory condition looks the same as an internal bug. A NaN particle position (non-representable moment) behaves the same way.
- **Fix:** In the continuation, add `catch (std::exception const& e)` that writes a fixed prefix and e.what() to stderr with fputs (no allocation, no HPX I/O), then calls abortCollectiveDispatch(). Keep catch(...) for everything else. When count == 1, skip the abort and let the exception propagate through the outer future.get(). Optionally have collectiveCheck/prepareCollectively throw a dedicated agreed-failure type;

#### Ghost-cell receive starts one blocking HPX task and one remote action per ghost fragment, with no limit on concurrency

`gravity_nitadori_pmmm#2` · **medium** · smell · confirmed

- **Where:** `src/gravity/nitadori/cell_schedule.cpp:184` (also `src/gravity/nitadori/ghost_exchange.cpp:135-137`; `src/gravity/nitadori/ghost_exchange.cpp:18`)
- **Problem:** startPipeline launches one HPX task per ghost fragment, all at once. Each task makes a synchronous remote PmmmReadGhostParticlesAction call, and each request suspends an owner-side thread on source-&gt;ready until that cell's tree is built. On each locality, live suspended threads (512 KiB stack reservation each) and in-flight parcels therefore scale with the whole ghost shell.
- **Failure scenario:** In a large distributed near-FMM run, each locality ends up with tens of thousands of suspended HPX threads on both requester and owner, plus a burst of tens of thousands of parcels. Stack memory spikes and the parcel layer congests. HPX stack or thread limits can make the solve fail or crawl.
- **Fix:** Batch ghost requests per owner into one action carrying many keys, issued in key order within a bounded window, or use a fixed pool of receiver tasks that pull indices from an atomic counter. On the owner, reply through a continuation (source-&gt;ready.then(...)) or a per-owner reply aggregator instead of suspending one HPX thread per request.

#### Cell-forest tree arena reserves 4 node slots per source particle (about 1 KB per particle) and caps local particle count near 5.4e8

`gravity_nitadori_pmmm#4` · **medium** · bug · confirmed

- **Where:** `src/gravity/nitadori/cell_tree.cpp:311` (also `src/gravity/tree_arena.cpp:226`; `src/gravity/tree_arena.cpp:238-240`; `src/gravity/nitadori/ghost_exchange.cpp:123-125`)
- **Problem:** treePrepareCellForestLocal reserves treeNodeCapacity(N) + 4N + roots\*(levels+2) + slack arena slots, although bounded-geometry cell trees need at most 2n_c-1 nodes per cell (at most 2N in total). The extra 4N term reserves about 1 KB of node-plus-multipole memory per owned+ghost particle (64x the base estimate at the default bucket), and through the int TreeId range check it caps the PMMM near-FMM path at about 5.3e8 local source particles.
- **Failure scenario:** 1024^3 particles on two localities (5.4e8 each) with gravity.method.near=fmm: planGhosts calls treeAllocateNodesLocal, which throws "Tree arena exceeds TreeId index range". Global FMM handles the same run. At 1e8 particles per locality the reservation is about 100 GB, which can fail under heuristic or strict overcommit or managed-memory limits.
- **Fix:** Replace 4N with the proven 2N bound (or a depth-aware bound) and drop the now-redundant treeNodeCapacity term. Better still, size from a typical estimate (8N/B + C\*(levels+2) + line slack) and make the arena chunked or growable, or retry with a larger arena after "Tree arena full". Check the int range up front with a PMMM-specific message. Document the per-slot cost (sizeof(TreeNode) + multipole stride) in the memory section of docs/pmmm.md. Do not use 2\*ceil(N/(B/2+1)) as a hard bound.

#### Hierarchical stage meshes can thrash the 8-entry / 2x-byte FFT cache and rebuild the Ewald Green filters at every stage

`gravity_nitadori_pmmm#5` · **medium** · smell · plausible

- **Where:** `src/gravity/nitadori/fft.cpp:27` (also `cosmictiger/gravity/nitadori/mesh_size.hpp:154`; `src/gravity/nitadori/solver.cpp:164-166`; `src/gravity/nitadori/fft.cpp:243-258`; `src/gravity/nitadori/fft.cpp:347-385`)
- **Problem:** In hierarchical PMMM runs, the LRU FFT/Green cache (8 entries, 2x base bytes) can evict the base mesh. This happens when there are 9 or more distinct stage mesh sizes, or when stage bytes exceed the budget. The base Ewald Green function, about N0^3 x 8e5 flops with no symmetry reuse, would then be rebuilt roughly once per big step. That is not every stage, but it is very expensive.
- **Failure scenario:** A clustered cosmological hierarchical run with about 9 or more populated rungs, or with large second and third rung fractions, rebuilds a 100^3–200^3 Green filter at each ladder stage. That costs seconds per stage.
- **Fix:** Never evict the base mesh, since it is the costliest and LRU's worst case. Size the count limit to the rung span, or round stage meshes to a small fixed set of sizes. Log cache misses. Exploit the cubic symmetry of the periodic Green function in initializeGreen.
- **Still open:** The mechanism is real. initialize() keeps the active cache plus 7 inactive entries (kMaxMeshCaches = 8) and evicts least-recently-used entries whenever `retainedBytes > meshCachePool.byteBudget` (= 2 x the base FFT bytes, fft.cpp:243-258). A miss re-runs initializeGreen, which calls green() for every mesh cell;

#### kickCellBatch allocates and frees CUDA managed memory for every near-FMM batch, and cudaFree synchronizes the whole device

`gravity_nitadori_pmmm#3` · **medium** · smell · confirmed

- **Where:** `src/gravity/nitadori/fmm_batch.cpp:120` (also `src/cuda/cuda.cpp:56-58`; `src/cuda/cuda.cpp:97`; `src/gravity/nitadori/cell_schedule.cpp:102`)
- **Problem:** With CUDA enabled (the default in CUDA builds), every near-FMM ready batch allocates its output with cudaMallocManaged and releases it with cudaFree. Both are blocking, device-synchronizing driver calls, so each batch stalls a worker OS thread behind whatever kick kernel is running, on top of allocation and page-migration overhead.
- **Failure scenario:** A CUDA near-FMM solve with many small batches: GPU kernels from concurrent batches are serialized by cudaFree's implicit sync, and allocation overhead dominates the small batches.
- **Fix:** Reuse grow-only output buffers from a per-worker or global pool, like NearWorkspacePool. Alternatively, allocate one solve-lifetime buffer, or use stream-ordered cudaMallocAsync/cudaFreeAsync on the launch stream. Managed memory is a poor fit for this short-lived output.


### Gravity: tree, walk and kick

#### Remote TreeNode deserialization carves multipole records out of the local tree arena via a second, unsynchronized counter

`gravity_tree_walk_cpu#1` · **medium** · bug · plausible

- **Where:** `src/gravity/tree_arena.cpp:103` (also `cosmictiger/gravity/tree.hpp:76`; `src/gravity/tree_arena.cpp:121`; `src/gravity/tree_arena.cpp:171`; `src/gravity/tree_arena.cpp:302`)
- **Problem:** Every remote cache line deserialized in the dehnen global-FMM walk takes about 682 multipole records (one per slot, placeholders included) from the local tree arena via nextMultId. That arena is sized only for local nodes (8N/bucket, 2M floor), with no allowance for remote records or allocator slack, so large multi-locality kicks may hit 'Tree multipole arena full'. Separately, the two counters stay disjoint only because every deserialization currently happens after the local build; a load before a later TreeAllocator::reset in the same epoch would have its record overwritten by treeStoreNode.
- **Failure scenario:** (a) Large multi-locality global-FMM run (for example N_local around 16-32M with bucketSize 128, so nodesSize sits at the 2M floor): local nodes plus per-thread allocation-line holes (treeAllocLineSize = 16 cache lines per worker) leave a few hundred thousand free slots. A kick that fetches more than about 300-1000 remote lines then throws 'Tree multipole arena full' inside HPX parcel deserialization and aborts the step.
- **Fix:** Budget remote records explicitly. Either add slack and a remote-cache allowance to the dehnen capacity (as nitadori's treePrepareCellForestLocal does), or allocate remote moments from a separate per-epoch, chunked, managed-memory pool. It must stay device-accessible because CUDA kicks dereference node.mpos, and it should be reset in treeAllocateNodesLocal. Point placeholder slots at one shared zero record instead of allocating for them.
- **Still open:** I checked each claim. (1) TreeNode::load (tree.hpp:76) runs `mpos = treeAddMultipole(multipolePosition, multipole);`. HPX 1.11 does call load once per element of Vector&lt;TreeNode&gt;.


### SPH

#### Price et al. switch counts the Hubble-flow deceleration as compression, so alpha saturates at alpha_max in essentially all cosmological gas

`sph_physics_kernels#2` · **medium** · bug · confirmed

- **Where:** `cosmictiger/sph/price_etal/viscosity.hpp:83` (also `src/sph/sph_prehydro_cpu.cpp:281-288`; `src/sph/sph_prehydro.cu:320-325`; `src/core/driver.cpp:270-272`; `cosmictiger/sph/cullen_and_dehnen/viscosity.hpp:242-249`)
- **Problem:** In cosmological runs the Price et al. shock indicator uses the full physical d(∇·v)/dt, which includes the homogeneous Hubble deceleration 3a⁴Ḣ &lt; 0, and it has no convergence or R limiter. In Hubble-like gas, A = 3ξa⁴|Ḣ| and α_loc = min(30ξ(1+q)(h_M H/c)², α_max), which saturates at α_max whenever h_M H/c ≳ 0.15. That is true of typical IGM, so every physically approaching pair there feels α_max.
- **Failure scenario:** hydro.viscosityEvolution=priceEtAl with run.scenario=cosmology: targetAlpha equals viscosityAlphaMaxPrice for nearly every particle outside hydrostatic cores at every step. Converging subsonic flows and sound waves in the IGM are damped as if alpha were 1 everywhere.
- **Fix:** Remove the homogeneous background from the indicator. In price_etal::diagnostics, replace `Real(3) * expansionAcceleration` with `Real(3) * sqr(a·da/dt)`, i.e. 3a⁴H². This needs passing expansionRate instead of, or alongside, expansionAcceleration. The indicator becomes a·tr(G) − tr(V'²) + 3a⁴H², which is a⁴[D(∇·v_phys)/Dt − 3Ḣ] and vanishes for pure Hubble flow. Alternatively or additionally, set A only where the physical divergence is negative (converging flow), mirroring CD's R=1 protection.

#### sphParticlesRefreshChemistryEos runs serially over every gas particle and abandons child futures on error

`sph_infrastructure#4` · **medium** · smell · confirmed

- **Where:** `src/sph/sph_particles.cpp:847` (also `src/sph/sph_particles.cpp:1195`; `src/core/driver.cpp:247`; `src/core/driver.cpp:507`; `src/core/driver.cpp:557`)
- **Problem:** sphParticlesRefreshChemistryEos sweeps every local gas particle on one HPX thread, doing a strict-FP-scoped EOS root-find per particle, and sphStep2 calls it three times per (sub)step in chemistry-without-star-formation runs. sphChemistryPressureIntegral adds a fourth serial sweep and UpdateTargetAlpha is also serial. On failure the child futures are dropped without wait_all, which is harmless but inconsistent.
- **Failure scenario:** Chemistry run with about 1e7 gas particles per locality: three serial EOS sweeps per step take seconds of wall time while all other cores sit idle. If one particle is rejected, the exception escapes while the remote refreshes are still running.
- **Fix:** Partition the loop over hpxHardwareConcurrency() tasks as sphParticlesResetConverged does. Each task gets its own StrictFloatEnvironment through sphChemistryEos, and writes are per-index, so results are bit-identical. Apply the same to sphChemistryPressureIntegral, combining per-task partial sums in a fixed order, and to starFormationRefreshEos's CPU branch. Wrap the body in try/catch with hpx::wait_all on the child and worker futures before rethrowing.


### Cosmology and initial conditions

#### Fixed Boltzmann start (T=0.07 MeV) caps k at ~32 Mpc^-1, but ICs request 1.05\*sqrt(3)\*k_Ny, so common boxes abort with a misleading error

`cosmology#5` · **medium** · smell · confirmed

- **Where:** `src/cosmology/boltzmann.cpp:343` (also `src/cosmology/zero_order.cpp:170`; `src/cosmology/initialize.cpp:127-129`; `docs/boltzmann_solver.md:220`)
- **Problem:** The background start is fixed at T=0.07 MeV, which caps the Boltzmann solver at k of about 32 Mpc^-1. The IC generator requests up to 1.05\*sqrt(3)\*k_Nyquist without checking first. Any setup with mean interparticle spacing below about 0.178 Mpc (about 0.12 Mpc/h, not the reported 0.26), such as 100 Mpc/h at 1024^3 or 25 Mpc/h at 256^3, aborts only after the full mode solve on every locality. The error message names neither k nor box size nor resolution. The cap is documented, and 25.271h is an unexplained magic constant.
- **Failure scenario:** units.boxLength=100 (Mpc/h), particlesPerDimension=1024, sigma8&gt;0: initialization throws after spawning ~1000 Boltzmann tasks on every locality.
- **Fix:** In computePowerSpectrum, before spawning tasks, compute the supported k_max = 0.05/tau(amin). If the requested maximum exceeds it, throw a message giving the required k, the supported k, and the minimum box length or maximum particlesPerDimension. Longer term, derive amin from the requested k_max (extending the thermal table), with convergence validation. Name or derive the 25.271h floor.

#### 1/sinc^6 deconvolution is wrong for particles on mesh nodes: near-Nyquist IC modes are over- or under-amplified, and Nyquist is kept in odd derivatives

`cosmology#6` · **medium** · bug · confirmed

- **Where:** `src/cosmology/initialize.cpp:514` (also `src/cosmology/initialize.cpp:567-573`; `src/cosmology/initialize.cpp:438-440`; `cosmictiger/cosmology/lpt.hpp:162`; `cosmictiger/cosmology/fft.hpp:115`)
- **Problem:** fourierSize == particlesPerDimension, so DM lattice particles sit on FFT nodes and (non-closePack) gas sits at half-cell offsets. The M6 interpolation's true transfer is then the aliased sum Σ_m Ŵ(k+2πm) on nodes, or Σ_m (−1)^m Ŵ(k+2πm) at half offset, not the sinc^6(kh/2) that cloudFilter divides out. Per axis, DM displacements and velocities are scaled ×1.05/1.30/2.00 at 0.75/0.9/1.0 k_Ny and gas ×0.95/0.70/0. Odd-derivative Nyquist modes are kept (×15 per axis before interpolation), and the kz=n/2 plane is never Hermitian-paired, so the c2r pass discards half its variance.
- **Failure scenario:** Default cosmology run: the DM IC power along the axes is 1.7x too high at 0.9 k_Ny and 4x at k_Ny, while the gas IC power is 0.5x at 0.9 k_Ny, creating a spurious small-scale DM-gas relative displacement.
- **Fix:** For lattice ICs, drop the 1/sinc^6 filter and the B-spline interpolation. Read DM displacements and velocities directly at their nodes. For gas (and closePack offsets), multiply the Fourier field by exp(i k·δ) before the inverse FFT and then read nodes directly. This needs one extra inverse FFT per species and field. Keep 1/sinc^6 plus M6 interpolation only for glass or irregular particle loads. Zero the Nyquist index along each differentiated axis in odd-derivative fields.

#### twolpt() allocates a zero complex vector the size of the REAL box and accumulates it into the complex grid every 2LPT phase

`cosmology#7` · **medium** · smell · confirmed

- **Where:** `src/cosmology/initialize.cpp:599`
- **Problem:** Every 2LPT phase, twolpt() allocates an all-zero complex vector the size of the local real box (n^3/P) and adds it into the complex grid. Because the real box is never inside the complex box, the zeros go through the chunked self-send path. The cost is about 8 bytes per cell of extra peak IC memory (8.6 GB at 1024^3 on one node), plus up to about 4 bytes per cell of in-flight copies and about 39 GB of useless copying and locking per IC.
- **Failure scenario:** 1024^3 2LPT ICs on one node: +8.6 GB peak memory and ~39 GB of pointless copying and locking in total, which can push a run near its memory limit into OOM.
- **Fix:** Delete the local yValue allocation and the extra fft3dAccumulateComplex(box, yValue) call in twolpt(); twolptGenerate already accumulates and joins its modes.

#### Glass tile range uses (int)box.end \* nf (the cast binds first), dropping tiles on multi-locality runs

`cosmology#4` · **medium** · bug · confirmed

- **Where:** `src/cosmology/initialize.cpp:772` (also `src/cosmology/initialize.cpp:773`; `src/cosmology/initialize.cpp:774`)
- **Problem:** In initX0's glass branch the cast binds before the multiply, so each locality's end tile index is 1 unless its box ends at exactly 1.0. Multi-locality glass runs then generate only glass tile 0 for those boxes: rank 0 of 2 loses x in [1/nf,0.5) when nf&gt;=3. Interior boxes get no particles or a negative-size vector exception.
- **Failure scenario:** initial.useGlass=true with hydro (so initX0 fans out) on 2 localities with nf&gt;=2: rank 0 (x in [0,0.5)) generates only x in [0,1/nf); particles are missing.
- **Fix:** Use exclusive tile bounds jb = int(std::floor(box.begin[d] \* nf)) and je = std::min(int(std::ceil(box.end[d] \* nf)), nf) for all three dimensions. Add a multi-locality useGlass test that checks the global particle count.

#### zeldovichEnd keeps the whole ghosted displacement grid in a function-static vector for the rest of the run

`cosmology#8` · **medium** · smell · confirmed

- **Where:** `src/cosmology/initialize.cpp:934`
- **Problem:** zeldovichEnd keeps its ghosted displacement grid in a function-static vector that is never released. Every locality therefore holds about 4 bytes per local cell (4.4 GB at 1024^3 on one node) for the whole simulation after initialization.
- **Failure scenario:** 1024^3 single-node run: 4.4 GB of dead memory is held throughout evolution, reducing headroom for trees and GPU staging.
- **Fix:** Make yValue a local and capture it by reference in the task lambda (all tasks are joined before return), or swap it with an empty Vector at the end of zeldovichEnd.


### Analysis: FoF and Rockstar

#### Fetch/refresh of the last group-tree cache line reads past nodes.size() when the arena's final chunk is partial

`analysis_groups_rockstar#12` · **medium** · bug · confirmed

- **Where:** `src/analysis/group_tree.cpp:369` (also `src/analysis/group_tree.cpp:381`; `src/analysis/group_tree.cpp:241`; `src/analysis/group_tree.cpp:143`)
- **Problem:** The group-tree cache-line fetch and refresh always read lineSize nodes. Because the arena size is not a multiple of lineSize and the allocator hands out a final partial chunk, a remote read of a node in that chunk makes the owning rank read past the end of `nodes`. That is an out-of-bounds read, a possible segfault, and a throw under CHECK_BOUNDS. The gravity-tree equivalent already clamps.
- **Failure scenario:** In a clustered snapshot where the group-tree arena (8\*N/256 nodes) is nearly exhausted, a remote rank reads a node in the final partial chunk. The owner reads up to lineSize-1 elements past the allocation, which can crash, and CHECK_BOUNDS builds throw.
- **Fix:** Use `int const end = std::min(begin + lineSize, (int) nodes.size());` in both groupTreeFetchCacheLine and groupTreeRefreshCacheLine, mirroring remote_cache.cpp, and add ALWAYS_ASSERT(index &lt; nodes.size()). Alternatively, round the arena size up to a multiple of treeCacheLineSize.

#### More than 32767 halos in one FoF group aborts the whole run (15-bit haloId, short pid)

`analysis_groups_rockstar#4` · **medium** · bug · plausible

- **Where:** `src/analysis/groups.cpp:396` (also `src/analysis/rockstar_catalogue.cpp:746`; `src/analysis/rockstar.cpp:793`; `cosmictiger/analysis/rockstar_record.hpp:219`)
- **Problem:** A FoF group with more than 32767 ROCKSTAR halos makes groupsReduce throw (snapshot), or makes rockstarMakeRecord's ALWAYS_ASSERT abort (lightcone, which has no pre-check). Either way the whole run terminates instead of degrading. Whether production runs reach that halo count per FoF group depends on run scale.
- **Failure scenario:** In a very large run (about 1e10-1e11 particles), a massive b=0.28 FoF group (a cluster plus bridged filament halos with 1e7-1e8 particles) can contain over 32767 halos of at least 10 particles. The groups analysis throws or asserts and the run ends after hours of compute.
- **Fix:** Share one capacity policy between both paths: widen haloId/pid in a format_version 3 record, or split oversized results into extra synthetic fofId work units, or drop or merge the excess halos with a logged warning instead of aborting. Allow exactly 32768 halos under the current ABI, add the same pre-check before the lightcone loop, and document the limit in docs/rockstar.md.
- **Still open:** Verified from code: the snapshot path throws `std::runtime_error("ROCKSTAR record halo ID capacity exceeded")` inside the groupsReduce hpx::async task. future.get() rethrows it at lines 447-450 and through remote GroupsReduceAction futures.

#### Spatial FoF never pulls labels from single-particle leaves: zero-volume bounding boxes fail the volume()&gt;0 overlap test

`analysis_groups_rockstar#1` · **medium** · bug · confirmed

- **Where:** `src/analysis/groups_find.cpp:116` (also `src/analysis/group_tree.cpp:205`; `cosmictiger/math/range.hpp:313`; `cosmictiger/analysis/rockstar_core.hpp:145`)
- **Problem:** groupsFind admits neighbour nodes only when `periodicIntersection(...).volume() > 0`. That rejects every leaf whose tight box has zero width in some dimension, and single-particle leaves always do. Because linking is pull-only, nobody reads such a particle's label: it falls out of its FoF group when it holds the component's minimum label, and if it is the only bridge between two clumps the group is split in two.
- **Failure scenario:** A node of 257 particles splits at the spatial midpoint with 1 particle on the left and a 256-particle clump on the right, the lone particle within b of the clump. After particlesSort the lone particle has the lower global index, so it keeps its own label and the clump never adopts it: it drops out of the FoF group. If the lone particle is the only link between two clumps, the FoF group is split in two.
- **Fix:** Add a closed-interval periodic overlap predicate and use it here. Per dimension, accept if any shift s in {0,+1,-1} satisfies a.end+s &gt;= b.begin &amp;&amp; b.end &gt;= a.begin+s, i.e. a periodic version of rockstar::detail::boxesOverlap. Do NOT just test the spans of the returned periodicIntersection range for &gt;= 0: its `<=` fall-through has already replaced the zero-width overlap with a negative shifted span. Add a spatial-FoF regression, e.g.

#### CPU phase-space linking writes labels into neighbour leaves without marking them active, so the lastActive filter can lose links

`analysis_groups_rockstar#2` · **medium** · bug · confirmed

- **Where:** `src/analysis/rockstar.cpp:372` (also `src/analysis/rockstar.cpp:364`; `src/analysis/rockstar.cpp:419`; `src/analysis/rockstar.cpp:456`; `src/analysis/rockstar.cu:88`)
- **Problem:** The CPU phase-space FoF (rockstarFindSubgroups) lowers and initializes labels of particles in neighbouring leaves but marks only itself active. The next pass skips leaves that are not lastActive, so links between two inactive leaves are never rechecked and the iteration can stop with linked particles still split. This usually strips a few particles from a subgroup, but it can occasionally destroy a small subgroup. The CUDA path, which only pulls labels, has no such error.
- **Failure scenario:** A Python replica of the CPU traversal (scratch rockstar_cpu_fof_sim2.py / sim3.py) on 2D filament configurations left a true connected component split into two labels in 13 of 400 trials (buckets 2-6) and 2 of 150 trials at bucket 8. It was not reproduced at bucket 16, or in 6D at bucket 64 (8 trials), so production frequency looks low but nonzero.
- **Fix:** Make the CPU leaf update pull-only like the CUDA kernel: modify only self particles, and either pre-assign unique labels or initialize only self's particle. That keeps the lastActive optimization valid, because every change then marks its owning leaf. Alternatively, OR a dirty flag into any leaf that is written, and into its ancestors, before copying active to lastActive. Add a regression that compares labels with union-find components for small buckets and for bucket 64.

#### Seed assignment indexes subgroups[-1] when no seed distance compares below FLT_MAX

`analysis_groups_rockstar#8` · **medium** · bug · confirmed

- **Where:** `src/analysis/rockstar.cpp:680` (also `src/analysis/rockstar.cpp:665`; `src/analysis/rockstar.cpp:710`; `src/analysis/rockstar_catalogue.cpp:443`)
- **Problem:** The seed-assignment loop never checks for minIndex == -1. When every seed at a level has a sigma2V shrunk to about 1e-40 (a cold-seed floor rescaled by a tiny sigmaV^2), the float distance metric overflows to inf (or NaN under FTZ), no dist compares below FLT_MAX, and the code indexes and pushBacks into subgroups[-1]. That corrupts memory. Only engineered degenerate inputs trigger it.
- **Failure scenario:** Synthetic or degenerate cold inputs (identical velocities inside nested seeds with small parent dispersion) under a -ffast-math production build: out-of-bounds pushBack into subgroups[-1], memory corruption or a crash, where the docs promise degenerate inputs terminate safely.
- **Fix:** Re-apply a normal-range floor after rescaling, e.g. sg.sigma2V = std::max(sg.sigma2V \* sigmaV \* sigmaV, 1e-30f), and likewise for sigma2X, or keep these statistics in double. Compute the assignment metric in double, or clamp sigmaVInv and rdynInv to finite values. Add an explicit `if (minIndex < 0)` fallback, such as the nearest seed by position or the largest seed, and use ALWAYS_ASSERT for real invariants.

#### Snapshot RockstarRecord positions are not wrapped into the periodic box

`analysis_groups_rockstar#3` · **medium** · bug · confirmed

- **Where:** `src/analysis/rockstar_catalogue.cpp:750` (also `src/analysis/groups.cpp:400`; `src/analysis/rockstar_catalogue.cpp:839`)
- **Problem:** Snapshot RockstarRecord positions are written as particles.front().x plus the halo's periodic offset, with no wrap. Halos in FoF groups that straddle the periodic boundary can therefore get x,y,z outside [0, L) Mpc/h. This contradicts the documented 'absolute position' and the wrapped GroupEntry.com, and breaks box-range filters such as the halo viewer's.
- **Failure scenario:** A FoF group straddles x=0, particles.front() sits at x=0.999 and the halo peak at 0.001 in box units. The record stores x = 1.001 L h Mpc/h. Viewer spatial filters over [0,L] miss the halo, and cross-matching or plotting puts it outside the box. Roughly half of the halos that straddle a boundary are affected.
- **Fix:** In groupsReduce, after rockstarMakeRecord, wrap each coordinate into the box: v -= L\*floor(v/L), where L = box length in Mpc/h (positionConversion). Alternatively pass a periodic flag into rockstarMakeRecord and wrap origin+group.x in box units before scaling. Leave lightcone records unwrapped. Add a groupsCatalogue test with a FoF group straddling x=0.


### Analysis: lightcone, BH and power

#### lightcone.mapSize is never validated; bad values index past the HEALPix map

`analysis_lightcone_bh_power#7` · **medium** · bug · confirmed

- **Where:** `src/analysis/lightcone.cpp:176` (also `src/core/options.cpp:1536`; `src/analysis/healpix.cu:97`; `src/analysis/healpix.cu:171`; `src/particles/drift_cpu.cpp:56`)
- **Problem:** lightcone.mapSize is used as the HEALPix NEST nside without validation. A user-supplied 0, a non-power-of-two value or a value above 8192 makes chealpix and the device vec2pixNest return pixel indices at or beyond 12\*mapSize^2 (or crash). lcAddParticleEntries then writes pix[entry.pix] out of bounds, which is heap corruption in NDEBUG builds.
- **Failure scenario:** With lightcone.mapSize=3 or 100, the bit interleaving assumes a power of two, so face-11 pixels can exceed 12\*nside^2-1 and pix[entry.pix] writes past the heap buffer. mapSize=0 aborts in chealpix or trips the device ALWAYS_ASSERT. mapSize &gt; 8192 exceeds both the traditional chealpix limit and the device assert.
- **Fix:** In processOptions, after the automatic substitution and when lightcone.enabled, THROW_ERROR unless mapSize is a power of two in [1, 8192]. That cap matches the traditional chealpix API, the device kNsMax and the int LightconeEntry::pix. Also add ALWAYS_ASSERT(entry.pix &gt;= 0 &amp;&amp; size_t(entry.pix) &lt; npix) before the deposit. Moving to vec2pix_nest64 and a 64-bit pix is only needed if larger maps are wanted.

#### lcTimeToFlush changes rank 0's nsideValue between lcInit calls

`analysis_lightcone_bh_power#4` · **medium** · bug · confirmed

- **Where:** `src/analysis/lightcone.cpp:245` (also `src/analysis/lightcone.cpp:878`; `src/analysis/lightcone.cpp:685`; `src/analysis/lightcone.cpp:859`; `src/core/driver.cpp:1487`)
- **Problem:** lcTimeToFlush calls computeNside(tau) on rank 0 only, and that call overwrites the global nsideValue between lcInit decompositions. When late-time nside drops to 0, rank 0's vec2pix and pixNeighbors switch to the single-pixel path while its healpix/npixValue/myPixRange and every other rank keep the old decomposition. In multi-locality runs the next flush, usually the final forced one, then routes all of rank 0's particles into pixel 0 and splits boundary groups.
- **Failure scenario:** Late in a run (for example N=64^3 with b=0.28, once (tauMax-tau)/tauMax &lt; ~5e-3), a non-flushing lcTimeToFlush leaves rank 0 with nsideValue=0, and the final checkLc(true) flush then runs with it. Rank 0's vec2pix returns pixel 0 for every particle: it keeps all its buffered particles, including those that belong to other ranks' pixels, and lumps everything it receives into pixel 0.
- **Fix:** Make computeNside pure (return the value only) and assign nsideValue solely in lcInit. Delete the dead nside/npix/ranks computation from lcTimeToFlush. Optionally make vec2pix/pixNeighbors key off the decomposition object (healpix == nullptr or npixValue == 1) rather than nsideValue, so the two cannot diverge.

#### HEALPix decomposition depends on each locality's own hardware concurrency

`analysis_lightcone_bh_power#5` · **medium** · bug · plausible

- **Where:** `src/analysis/lightcone.cpp:873` (also `src/analysis/lightcone.cpp:910`; `src/analysis/lightcone.cpp:362`; `src/analysis/lightcone.cpp:697`; `src/core/hpx.cpp:35`)
- **Problem:** Each locality derives the lightcone HEALPix decomposition (nsideValue, npixValue, myPixRange) from its own hwloc PU count and SMT flag, with no global agreement. On localities with different effective PU counts, pixel ownership disagrees and lcSendParticles dereferences a null mutexMap entry. Differing SMT state alone does not trigger it, because the SMT halving compensates.
- **Failure scenario:** Take two nodes with different hardware-thread counts (or SMT active on only one) and get different nside values. The sender's pix2rank routes a particle to a rank whose myPixRange does not contain the pixel the receiver computes. lcSendParticles then evaluates \*mutexMap[pix] on a missing key: many threads concurrently insert a null shared_ptr and dereference it (segfault), and particleMap[pix] inserts and rehashes the CudaUnorderedMap under …
- **Fix:** Compute nside once on rank 0 (or reduce hardware concurrency to a global minimum) and pass it in the LcInitAction arguments, so every locality builds the same decomposition. In lcSendParticles use mutexMap.find(pix) plus ALWAYS_ASSERT (or mutexMap.at) instead of operator[] so a routing mismatch fails loudly instead of racing and dereferencing null.
- **Still open:** lcInit runs on every locality (LcInitAction broadcast), and each one computes `double nside = std::sqrt(8.0 * hpxHardwareConcurrency() * hpxSize() / 12.0);`, then npixValue, myPixRange and the healpix object, locally. Nothing reduces or checks the result.

#### Power-spectrum error pass filters the already deconvolved FFT field a second time

`analysis_lightcone_bh_power#1` · **medium** · bug · confirmed

- **Where:** `src/analysis/power.cpp:211` (also `src/analysis/power.cpp:143`; `src/analysis/power.cpp:79`; `src/core/driver.cpp:1224`)
- **Problem:** powerSpectrumBinDeviation multiplies the FFT field, already deconvolved in place by powerSpectrumBin's 1/sinc^6 quintic window, by a second leftover 1/sinc^3-per-axis factor before computing (|y|^2 - mean)^2. perr, and with it the lower/upper columns of power.N.txt, is inflated from ~2x at k=n/4 to ~19x near Nyquist. The central P(k) is correct.
- **Failure scenario:** This happens in every run with analysis.power=true. For a mode next to the axis Nyquist (i=n/2-1, j=k=0), sinc(pi/2)=0.637, so pwr is about 15x the bin mean and (pwr-mean)^2 is about 200\*mean^2 instead of about mean^2. At k=n/4 the power is already inflated 1.9x per axis. perr, and with it the lower/upper columns that doPowerSpectrum writes to power.N.txt, is grossly overestimated over the upper half of the k range.
- **Fix:** Delete the c0 computation and the in-place multiply in powerSpectrumBinDeviation and use pwr = std::norm(yValue[idx]) on the field as left by the first pass. Better, accumulate sum(p) and sum(p^2) in double in powerSpectrumBin and derive the variance there, which drops the second distributed sweep. Also switch the per-bin power/ktot accumulators from float to double.

#### Folded density deposit does O(N) work and sends a full grid from every rank

`analysis_lightcone_bh_power#9` · **medium** · smell · confirmed

- **Where:** `src/analysis/power.cpp:284` (also `src/analysis/power.cpp:253`; `src/analysis/power_cpu.cpp:69`; `src/analysis/power.cu:30`)
- **Problem:** Every locality loops over all R..8R grid tiles. For each tile it rescans all its particles (single-threaded on CPU) and sends the padded tile through synchronous per-piece remote accumulates, even when the tile is empty. That costs O(N) work and about N floats of traffic per rank, O(R\*N) in total. The CUDA path also reallocates and refills its buffers for every tile.
- **Failure scenario:** With 64 ranks and 2048^3 particles, each rank scans its particles about 64 times and pushes about 8.6e9 floats (34 GB) through fft3dAccumulateReal, which issues synchronous remote actions per piece. That is about 2.2 TB per power spectrum, although only the tiles overlapping each rank's domain carry nonzero data.
- **Fix:** Bin local particles by tile once, or compute the folded bounding box of the local particles and visit only intersecting tiles. Skip tiles with no deposit. Hoist the mass array and device buffers out of the tile loop. Consider making fft3dAccumulateReal pieces asynchronous.


### Math and numerics

#### CUDA radial-kernel evaluation likely spills to local memory and serializes constant-cache reads in SPH pair loops

`math_numerics#14` · **medium** · smell · confirmed

- **Where:** `cosmictiger/math/radial_kernel.hpp:127` (also `cosmictiger/math/radial_kernel.hpp:315-351` (value/derivative piece selection); `src/math/kernel.cu:17-18` (\__constant_\_ tables); `cosmictiger/sph/sph_cuda.hpp:172-173`; `src/sph/sph_hydro.cu:217-218`)
- **Problem:** On the sm_86 Release build, radial_kernel evaluation inside the CUDA SPH pair loops (kernelW/dkernelWDq in sphCudaPrehydro1 and sphCudaHydro) has three costs. The runtime-degree de Casteljau `work` array lives in local memory (LDL/STL). Coefficient rows are read with per-lane-indexed LDC after reconvergence, which replays when a warp spans both spline pieces. Every double table entry is also converted to float per evaluation (F2F.F32.F64 + DSETP + DADD) on a GPU with 1:64 FP64 throughput.
- **Failure scenario:** If confirmed, each kernelW or dkernelWDq call in sph_cuda.hpp:172-173 and sph_hydro.cu:217-218 costs LDL/STL traffic plus serialized constant loads. Confirm with cuobjdump -sass (look for LDL/STL in the SPH kernels) or Nsight Compute local-memory and constant-cache stall metrics.
- **Fix:** Upload a float mirror of the hydro table (coefficients, derivative coefficients, change points and a precomputed 1/(b-a) per span) for device use. Dispatch on the KernelSpec once per launch into templates with a compile-time degree and pieceCount. Evaluate each span with a fully unrolled register de Casteljau, keeping the two-weight Bernstein form that the code comments say is needed for accuracy near a span's right endpoint, rather than switching to Horner.

#### boxDistanceSquared decides inside/outside without periodic wrap but measures distance with periodic wrap

`math_numerics#1` · **medium** · bug · confirmed

- **Where:** `cosmictiger/math/range.hpp:494` (also `cosmictiger/math/range.hpp:500-502` (Q0.32 scalar path); `cosmictiger/math/range.hpp:549` (SIMD path); `tests/numerical_utilities.cpp:168-169` (tests only use boxes inside [0,1)); `cosmictiger/gravity/fmm_geometry.hpp:80,93` (callers via fmmSink/SourceBoxDistanceSquared))
- **Problem:** boxDistanceSquared decides 'outside' by comparing the point with the unwrapped box endpoints, but measures the distance with minimum-image wrapping, in all three paths (Q0.64 scalar, Q0.32 scalar, Q0.32 SIMD). For a box that extends past 0 or 1, a point inside the wrapped part gets a nonzero distance: box x in [0.95,1.05] with x=0.02 gives d2=9e-4, and box [-0.0625,0.125] with x=0.95 gives 1.5625e-4. Today's gravity tree boxes always lie within [0,1], and the function is exact there, including end==1.0, so the bug is latent. The comment and the CUDA test cases suggest out-of-range boxes are supported.
- **Failure scenario:** Box x-range [0.95, 1.05], point x = 0.02, which is 1.02 and inside the box. Q0.64 path: below = true (0.02 &lt; 0.95), above = false (upper &gt;= 1), so outside = 1. delta = min(|dist(0.95,0.02)| = 0.07, |dist(1.05,0.02)| = 0.03) = 0.03, giving d2 = 9e-4 instead of 0. boxIntersectsSphere(box, x, 0.02) returns false. The 32-bit and SIMD paths give the same result. Box [1.2, 1.3] with point 0.25 returns 0.05^2 instead of 0.
- **Fix:** Compute the periodic interval distance per dimension in exact integer arithmetic. Let u = Position(x) - Position(box.begin) as an unsigned wrapped subtraction, and len = the exact (end - begin) shifted to Position scale (valid while len &lt; 1). The point is inside iff u &lt;= len; otherwise delta = min(u - len, 1 - u). Use this in the scalar path and do the same per lane in the SIMD Q0.32 path.

#### SymmetricTracelessTensor: operator- returns other - this, and the non-const operator() aliases or overruns for n&gt;=2 in NDEBUG builds

`math_numerics#2` · **medium** · bug · confirmed

- **Where:** `cosmictiger/math/tensor.hpp:72` (also `cosmictiger/math/tensor.hpp:77-88` (non-const operator()); `cosmictiger/math/tensor.hpp:90-108` (const operator() with recursion); `cosmictiger/core/assert.hpp:11-12` (ASSERT compiled out under NDEBUG))
- **Problem:** SymmetricTracelessTensor::operator- returns other - \*this, so a - b yields b - a. The non-const operator()(l,m,n), which overload resolution picks for any non-const tensor even for reads, has only NDEBUG-stripped ASSERTs. In release builds, (l,m,2) with l+m&gt;0 goes past the kN\_-element storage and n&gt;=3 silently aliases the (l,m,0) slot, while the const overload correctly rebuilds those components from tracelessness. Neither path is reached today: operator- has no callers, and the non-const reads use nz&lt;=1 or (0,0,2).
- **Failure scenario:** SymmetricTracelessTensor&lt;float,4&gt; a, b; auto c = a - b; gives c == b - a, a silent sign flip in any future derivative or multipole difference. On a non-const tensor t, float v = t(0,1,2); reads index 2 + p\*p = 18, past the end of the 17-element storage. t(1,0,3) returns the (1,0,0) component.
- **Fix:** Change line 72 to `result[i] = (*this)[i] - other[i];` and add CUDA_EXPORT, or delete operator- since nothing uses it. For the non-const operator(), either make the n guard an ALWAYS_ASSERT(n &lt;= 1 || (n == 2 &amp;&amp; l == 0 &amp;&amp; m == 0)), or rename it to a stored-component accessor (for example `stored(l,m,n)`) so plain reads resolve to the const reconstructing overload, via std::as_const where needed.


### Build system and scripts

#### COSMICTIGER_WITH_MAX_BUCKET=32/64 is accepted, but the default compute.bucketSize=128 is then rejected on every run

`build_cmake_scripts#3` · **medium** · bug · confirmed · file was being edited during the review

- **Where:** `CMakeLists.txt:42` (also `src/core/options.cpp:534`; `src/core/options.cpp:1229`; `default.ini:28`; `CMAKE_OPTIONS.md:37`)
- **Problem:** CMake and CMAKE_OPTIONS.md accept COSMICTIGER_WITH_MAX_BUCKET=32 or 64. However, the runtime default compute.bucketSize is hard-coded to 128 (options.cpp:533), as are default.ini and all shipped INIs. processOptions rejects bucketSize &gt; MAX_BUCKET_SIZE, so every run and application test in such a build aborts at startup unless --compute.bucketSize is lowered by hand.
- **Failure scenario:** Configure with -DCOSMICTIGER_WITH_MAX_BUCKET=64. The build succeeds, but every run (any scenario and essentially every application CTest) aborts with "compute.bucketSize must not exceed the compile-time maximum 64 (COSMICTIGER_WITH_MAX_BUCKET)".
- **Fix:** Default compute.bucketSize to MAX_BUCKET_SIZE, matching the stated tuning assumption that the runtime bucket equals the cap, or to std::min(128, MAX_BUCKET_SIZE). Comment out or drop the hard-coded `bucketSize = 128` in default.ini and the configurations so they inherit the compiled default. Update run_options.py:949 to expect the compiled value.

#### App builds the SH03 CUDA kernel with inherited --use_fast_math, but its unit test builds it strict

`tests_sph_chem_cpp#1` · **medium** · bug · confirmed · file was being edited during the review

- **Where:** `CMakeLists.txt:397` (also `cmake/variants.cmake:24`; `CMakeLists.txt:917-923`; `cosmictiger/chemistry/primordial_eos.hpp:56`; `cosmictiger/chemistry/primordial_eos.hpp:68`)
- **Problem:** In the application, src/sph/star_formation_cuda.cu is compiled in cosmictiger_objects, which inherits `--use_fast_math` from cosmictiger_platform. The per-source flags `--ftz=false --prec-div=true --prec-sqrt=true --fmad=false` restore IEEE division, sqrt, denormals and no-contraction, but not the function substitution. The app's device code therefore computes expf, logf, log2f and powf in the inlined primordial EOS, rate and cooling code, and in the rung log2f, with approximate intrinsics: its PTX shows `mul.rn.f32 x, 0fBFB8AA3B; ex2.approx.f32` at primordial_eos.hpp:56 and 83 `lg2.approx.f32`.
- **Failure scenario:** molecular9 network, compute.cuda=true, star formation on, ordinary gas containing H2. On the device the EOS temperature, heat capacity and adiabatic index of that gas use approximate expf: errors of tens of ulp, about 1e-5 relative. The CPU and the strict chemistry library compute these accurately. star_formation_cuda.MolecularOrdinaryGasEos still passes because that test builds the kernel strictly.
- **Fix:** Remove `--use_fast_math` from this file's nvcc command line. No later flag undoes its function substitution. Build src/sph/star_formation_cuda.cu, with the strict CUDA flags, in its own OBJECT or STATIC library that does not link cosmictiger_platform, as cosmictiger_sph_chemistry already does. Give it only the include directories, definitions, CUDA::cudart and Boost it needs, and set CUDA_SEPARABLE_COMPILATION to match.
- **Also reported as** `chemistry_star_formation#4` (Production SH03 CUDA backend inherits --use_fast_math, so device expf/powf/logf differ from the host and from the strict chemistry library) at `CMakeLists.txt:397`: star_formation_cuda.cu is compiled in cosmictiger_objects and inherits cosmictiger_platform's CUDA --use_fast_math. The per-source --ftz/--prec-div/--prec-sqrt/--fmad flags do not cancel it (nvcc still passes -fast-math to cicc), so production device SH03 rate, cooling and molecular9 EOS …

#### CPU-only builds register six CTests that pass --compute.cuda=true, so they always fail

`build_cmake_scripts#2` · **medium** · bug · confirmed · file was being edited during the review

- **Where:** `CMakeLists.txt:2179` (also `CMakeLists.txt:2424`; `CMakeLists.txt:2385`; `tests/run_smoke.py:79`; `src/core/options.cpp:1869`)
- **Problem:** In CPU-only builds with SPH+FMM+traditional, six CTests are registered that request CUDA: sph_gravity_cuda, checkpoint_state_sph, smoke_dm_cuda, smoke_sod, smoke_sph and smoke_sph_mpi. Each passes --compute.cuda=true, directly or through run_smoke.py without --cpu. processOptions rejects that in a non-CUDA executable, so they fail on every run.
- **Failure scenario:** Run ctest in any CPU-only tree with SPH+FMM+traditional, e.g. build/molecular_hot_cpu. sph_gravity_cuda, checkpoint_state_sph, smoke_dm_cuda, smoke_sod, smoke_sph and smoke_sph_mpi fail on every run. This permanent noise hides real failures and teaches people to ignore red results.
- **Fix:** Skip sph_gravity_cuda and smoke_dm_cuda when NOT COSMICTIGER_WITH_CUDA. Derive checkpoint_state_sph's --compute.cuda from COSMICTIGER_WITH_CUDA. Register smoke_sod/smoke_sph/smoke_sph_mpi only with CUDA, since \*\_cpu twins already cover the CPU path. Alternatively, have the C++ tests return 77 with SKIP_RETURN_CODE.
- **Also reported as** `tests_python#4` (CUDA smoke modes are registered and run in CPU-only builds (hard-coded --compute.cuda=true)) at `tests/run_smoke.py:79`: The smoke_sod, smoke_sph, smoke_sph_mpi and smoke_dm_cuda CTests pass --compute.cuda=true and are registered regardless of COSMICTIGER_WITH_CUDA. In every CPU-only build they abort with 'This executable was compiled without CUDA support', so a full CPU ctest can never be green.

#### FMM walk and PMMM application regressions run only at the default expansion degree; the distributed walk is never registered in CPU-only builds

`tests_gravity_cpp#5` · **medium** · smell · confirmed · file was being edited during the review

- **Where:** `CMakeLists.txt:2238` (also `CMakeLists.txt:2218`; `CMakeLists.txt:2355-2358`; `CMakeLists.txt:1651`; `tests/gravity/nitadori/pmmm.cpp:807-810`)
- **Problem:** End-to-end walk and PMMM application tests (fmm_walk_cpu/cuda, pmmm\_\*, pmmm_evolution\_\*) run only at the build's maximum expansion degree. The per-degree CTests run only fmmTree, so the per-order kick instances and PMMM solves at lower compiled degrees are never checked against a reference. fmm_walk_mpi, the only multi-locality walk force regression, is registered only in CUDA builds, so CPU-only MPI builds never test the distributed walk handoff or the periodic root-background reduction.
- **Failure scenario:** Take a build with degrees 4-6 (orders 5..7) and a walk or periodic bug that only affects the order-5 CPU instance. ctest passes, because only order 7 runs through fmmWalk and pmmm. In a CPU-only MPI build, a distributed walk handoff regression is never exercised.
- **Fix:** Inside the fmm_degree\_${degree} loop, also register `--run.test=fmmWalk` (CPU, plus CUDA when enabled) and, for PMMM builds, `--run.test=pmmm` with `--gravity.expansionDegree=${degree}`. Move fmm_walk_mpi out of the CUDA guard and pass `--compute.cuda=$<BOOL:${COSMICTIGER_WITH_CUDA}>`, or add a CPU-only twin.

#### Application-wide -ffast-math silently removes the std::isfinite safety checks still left in several sources

`build_cmake_scripts#4` · **medium** · bug · confirmed

- **Where:** `cmake/variants.cmake:13` (also `src/gravity/kick.cpp:106`; `cosmictiger/gravity/kick_walk.hpp:437`; `src/sph/sph_hydro_cpu.cpp:307`; `cosmictiger/analysis/lightcone_entry.hpp:33`)
- **Problem:** Every application C++ TU is compiled with -ffast-math, under which GCC 15 folds std::isfinite to true. The finiteness halves of the ALWAYS_ASSERTs at kick.cpp:106, kick_walk.hpp:437 and sph_hydro_cpu.cpp:307, and the guard at lightcone_entry.hpp:33, are therefore compiled away. The NaN guard `!(timestep > 0.0)` at scenario_driver.cpp:221 becomes codegen-dependent. Non-finite timesteps or radii can slip past these checks silently.
- **Failure scenario:** A +inf or NaN timestep reaches gravityFinishGlobalKick, or the SPH δt becomes inf. The assertion still passes, kicks are applied with a non-finite dt, and velocities silently become inf/NaN instead of aborting at the intended check. lightconeEmissionScale likewise accepts a NaN radius without throwing.
- **Fix:** Move the existing exponent-bit test into a shared header (for example math::isFiniteBits(double/float)) and use it at these sites. For scenario_driver.cpp:221, test the bits explicitly instead of relying on NaN comparison semantics. Add a grep or clang-tidy CI check banning std::isfinite/isnan/isinf in sources not on the -fno-fast-math list. Alternatively compile the application with `-ffast-math -fno-finite-math-only`.
- **Also reported as** `sph_physics_kernels#5` (std::isfinite in the timestep assertion is folded away under -ffast-math) at `src/sph/sph_hydro_cpu.cpp:307`: Under the project's -ffast-math host flags, the std::isfinite term in the CPU hydro timestep assertion is folded to true. This contradicts the codebase's own isFiniteValue convention.
- **Also reported as** `tests_other_cpp#5` (linear_growth.cpp std::isfinite guards are folded away by the application's -ffast-math) at `tests/linear_growth.cpp:329`: linear_growth.cpp is compiled with the application's -ffast-math, so GCC folds the std::isfinite guards at lines 249 (displacement) and 329 (acceleration) to true. A NaN from the solver is not reported where it first appears.
- **Also reported as** `gravity_tree_walk_cpu#7` (std::isfinite guards in the kick are compiled away by the application-wide -ffast-math) at `cosmictiger/gravity/kick_walk.hpp:437`: The std::isfinite halves of the timestep guards in kickWalk's global branch and in gravityFinishGlobalKick are folded to true, because both TUs compile with -ffast-math. They therefore check only positivity.
- **Also reported as** `tests_gravity_cpp#1` (FMM walk/tree checks cannot detect NaN: std::isfinite is compiled out by the application's -ffast-math, defeating the NaN 'unwritten output' sentinels) at `tests/gravity/dehnen/fmm_walk.cpp:42`: fmm_walk.cpp and fmm_tree.cpp are compiled with the application's -ffast-math. In the shipped object, std::isfinite is folded to true and float comparisons lose their unordered (NaN) handling, so fmmWalkRequireClose and the requireFmmTree geometry and multipole checks accept NaN.


### FMM generators and tools

#### Generator docs prescribe retired CMake options, wrong defaults, wrong test names and wrong spherical template argument

`fmm_generators_tools#1` · **medium** · bug · confirmed

- **Where:** `docs/generated_fmm.md:17` (also `docs/generated_fmm.md:23`; `docs/generated_fmm.md:38`; `docs/generated_fmm.md:76`; `docs/generated_spherical.md:7`)
- **Problem:** docs/generated_fmm.md:17-24, generated_spherical.md:7-8 and generated_spherical_ewald.md:5-7 tell users to set COSMICTIGER_FMM_MIN_ORDER/MAX_ORDER and say the default is orders 4-8. Both cache options were retired in 844a55c0 and now abort configure with FATAL_ERROR; the real knob is COSMICTIGER_EXPANSION_DEGREE, default 4-6 (orders 5-7). Two more errors: generated_fmm.md:76 and generated_spherical.md:140 name the tests `generated_spherical_order_<order>` (actually `generated_spherical_degree_<degree>`, and only built when a PMMM variant is enabled), and generated_fmm.md:38-39 says spherical Multipole/Expansion take degrees when they take the FMM order.
- **Failure scenario:** A user follows generated_fmm.md and runs `cmake -DCOSMICTIGER_FMM_MIN_ORDER=3 -DCOSMICTIGER_FMM_MAX_ORDER=9 ...`. Configure aborts with 'Retired CosmicTiger cache options'. `ctest -R generated_spherical_order_` matches nothing. A reader writes `spherical::Multipole<Real, 7>` meaning degree 7 and gets degree 5 storage.
- **Fix:** generated_fmm.md:17-24: say the build compiles potential degrees from `COSMICTIGER_EXPANSION_DEGREE` (default `4-6`, i.e. orders 5-7), and that orders = degree+1 are derived internally into the FMM_MIN_ORDER/FMM_MAX_ORDER guards. To compile every order, use `-DCOSMICTIGER_EXPANSION_DEGREE=2-8`. Old caches must drop the retired entries with `-U COSMICTIGER_FMM_MIN_ORDER -U COSMICTIGER_FMM_MAX_ORDER`. Make the same change in generated_spherical.md:7-8 and generated_spherical_ewald.md:5-7.


### Tests

#### analyticCompare statistics are wrong or fragile, and its result is never asserted

`tests_gravity_cpp#7` · **medium** · bug · confirmed

- **Where:** `tests/gravity/analytic.cpp:91` (also `tests/gravity/analytic.cpp:79`; `tests/gravity/analytic.cpp:83`; `tests/gravity/analytic.cpp:86`; `tests/test.cpp:497`)
- **Problem:** analyticCompare's force metric compares only magnitudes, and its 'median' uses the wrong order statistics; these feed the documented walk-accuracy tables. It prints every sample, asserts nothing, and in CPU-only builds falls off the end of a non-void function (UB). The claimed zero-sample/100%-error scenario cannot occur, because processOptions forces analysis.saveForce=true for run.test=force.
- **Failure scenario:** Run run.test=force with analysis.saveForce=false: every reported 'Force 50/99 error' is exactly 1.0 and nothing complains. If a kick bug rotates forces but keeps their magnitudes, the reported error is still about 0.
- **Fix:** Use the vector error |g_fmm - g_ref|/|g_ref| (and |phi_fmm - phi_ref|/|phi_ref|). Compute the median as (a[N/2-1]+a[N/2])/2 for even N, and require N&gt;=2. Drop the per-sample printf or put it behind a verbose option. In the non-CUDA branch, THROW_ERROR (or compile run.test=force only in CUDA builds) instead of falling off the end. Optionally add thresholds or a small CTest wrapper.

#### analyticCompare falls off the end of a non-void function in non-CUDA builds (UB; GCC -O2 crashes)

`tests_gravity_cpp#4` · **medium** · bug · confirmed

- **Where:** `tests/gravity/analytic.cpp:103` (also `tests/test.cpp:497`)
- **Problem:** Without USE_CUDA, analyticCompare's #else branch has no return statement, and GCC -O3 emits no ret instruction for it. So run.test=force in a CPU-only FMM build, after the first kick, runs off the end of analyticCompare into an unrelated HPX function, crashing or corrupting state instead of skipping the comparison.
- **Failure scenario:** A CPU-only FMM build with tests runs --run.test=force. After the first kick, forceTest calls analyticCompare(1000), execution runs off the end of the function into unrelated code, and the process crashes or returns garbage that is then written to force_accuracy.txt.
- **Fix:** In the #else branch, return a sentinel such as Pair&lt;double&gt;{NaN, NaN} or throw, and have forceTest skip the analytic step and the force_accuracy.txt write when CUDA is unavailable. Alternatively, guard the call in forceTest with USE_CUDA. Longer term, compute forceTest's error with the CPU-capable production diagnostic in src/gravity/error.cpp (gravityErrorDirect).

#### fmm_api.cpp hard-codes FMM order 7 and fails to compile when the configured degree range excludes degree 6

`tests_gravity_cpp#3` · **medium** · bug · confirmed

- **Where:** `tests/gravity/fmm_api.cpp:19` (also `tests/gravity/fmm_api.cpp:20-22`; `tests/gravity/fmm_api.cpp:31`; `cosmictiger/gravity/fmm_kernels/cartesian_kernels.hpp:6415`; `cmake/features.cmake:65-83`)
- **Problem:** fmm_api.cpp hard-codes FMM order 7 (potential degree 6). Its generated Cartesian and spherical kernel specializations are compiled only when FMM_MIN_ORDER &lt;= 7 &lt;= FMM_MAX_ORDER. Any supported COSMICTIGER_EXPANSION_DEGREE that excludes degree 6, such as 2, 4, 2-5, 7-8 or 8, therefore stops fmm_api_test from compiling, and with it the BUILD_TESTING build.
- **Failure scenario:** Configure with -DCOSMICTIGER_EXPANSION_DEGREE=8 -DBUILD_TESTING=ON (orders 9..9). fmm_api.cpp then fails with incomplete-type errors for cartesian::detail::P2M&lt;5&gt; and spherical order-7 kernels.
- **Fix:** Instantiate checkPointFields at kFmmDefaultOrder, or loop over kFmmMinOrder..kFmmMaxOrder with the forEachFmmOrder helper that cartesian_kernels.cpp uses. The mismatched-order static_asserts only evaluate concepts and never instantiate kernels, so they can use any pair of different orders, for example kFmmMaxOrder and kFmmMaxOrder - 1. Consider also building, not just configuring, one single-degree variant such as '8' in check_build_variants.

#### At least eight hand-rolled, drifting decoders of the same binary checkpoint format across test scripts

`tests_python#5` · **medium** · smell · confirmed

- **Where:** `tests/gravity/nitadori/pmmm_evolution_helpers.py:52` (also `tests/run_gravity_error.py:27`; `tests/run_global_integrator.py:18`; `tests/run_primordial_integration.py:18`; `tests/run_primordial_integration.py:46`)
- **Problem:** The binary checkpoint layout is decoded by about ten hand-written parsers in tests/. Each handles a different subset of the header flags, optional parameter blocks and gas payload versions. Two writer changes, the always-on bit-18 algorithm block (844a55c0) and the reordered viscosity parameter block (430edba2), have already left decoders stale.
- **Failure scenario:** The next checkpoint layout change (new flag, new parameter block, new gas version) again leaves some decoders stale. The affected tests then fail with misleading 'layout mismatch' errors, or are quietly left unregistered or ignored.
- **Fix:** Add tests/checkpoint_format.py. It should decode the header, then each flagged block in writeDriverState order (algorithms, PMMM, SPH, z-plane, viscosity, chemistry, star formation, driver, global extension), then the particle, gas, domain and lightcone payloads. Drive the gas tail from one table mirroring sphCheckpointVersion(). Port every driver to it. Add a self-test that decodes each checkpoint produced by checkpoint_integrity, so writer changes fail in one place.

#### PMMM checkpoint helper reads the stale viscosity layout (index 4 as CD flag): mixed gas PMMM evolution always fails

`tests_python#2` · **medium** · bug · confirmed

- **Where:** `tests/gravity/nitadori/pmmm_evolution_helpers.py:99` (also `src/io/checkpoint.cpp:550`; `src/sph/sph_particles.cpp:889`; `tests/gravity/nitadori/run_pmmm_evolution.py:98`; `tests/gravity/nitadori/run_pmmm_evolution.py:144`)
- **Problem:** read_rank() still decodes the variable-viscosity block with the pre-430edba2 layout. It treats index 4 (now viscosityDecayCD, default 0.05 and required positive) as a 0/1 Cullen-Dehnen flag, so every mixed gas+DM PMMM evolution with morrisAndMonaghan (the default) or cullenAndDehnen raises 'invalid variable-viscosity parameters'. The gas-payload version mapping is also wrong: MM=4, CD=6 and Price=8 are keyed on index 2, and Price is not representable.
- **Failure scenario:** Any SPH+PMMM build registers pmmm_evolution_cpu/cuda/mpi with --with-sph. The mixed case passes hydro.viscosityEvolution=morrisAndMonaghan (the default), so the flag bit 14 block is written. A real MM checkpoint block decodes as (0.1, 0.2, 0.0, 2.0, 0.05, 1.0, 0.1), which I verified on build/molecular_hot_cpu/tests/checkpoint_integrity/viscosity_morrisAndMonaghan_0.
- **Fix:** Read model = viscosity[2] and require it in {0,1,2}, with the other six entries finite and positive. Set gas_version = 2 + {0: 2, 1: 4, 2: 6}[int(model)]. Decode the per-particle viscosity payload as interleaved records: take('f', (3 if model else 2) \* gas_count), sliced per particle. Correct the 'two separate arrays' comment. Add priceEtAl to --viscosity-evolution choices so the version-8 layout is exercised.

#### Power-spectrum test reference hard-codes 2^-32 position scaling; wrong in fixed64 builds

`tests_other_cpp#1` · **medium** · bug · confirmed

- **Where:** `tests/power_test_helpers.hpp:93` (also `tests/power_test_helpers.hpp:61-62` (fixture shifts by kPositionBits-32); `src/analysis/power_cpu.cpp:21` (production uses toFloat()); `cosmictiger/math/position.hpp:24` (kPositionToFloat = 0x1p-64f for fixed64); `tests/power_cpu.cpp:206-225`)
- **Problem:** referenceDensity converts Position::raw() with a hard-coded 2^-32 factor. In a COSMICTIGER_POSITION_TYPE=fixed64 build (the tests inherit COSMICTIGER_POSITION_BITS=64 through cosmictiger_default_config), every reference coordinate becomes an integer and folds to 0. PowerCpu.FoldedDensity and the two non-empty PowerCuda density tests would then fail falsely against a correct production kernel.
- **Failure scenario:** In a COSMICTIGER_POSITION_TYPE=fixed64 build (for example build/named_variants_cpu_fixed64, where only build\_\* tests have been run), PowerCpu.FoldedDensity and PowerCuda.PartialBlockDensity/MultipleBlockDensity fail on the first random particle with "density agrees with independent periodic quintic reference". These are false failures in a supported configuration, and someone could 'fix' the production kernel to match the broken oracle.
- **Fix:** Use the configured conversion: `float const coordinate = fixture.positions[dimension][index].toFloat();` (equivalently `float(raw) * kPositionToFloat`). This keeps the single-precision rounding-to-1.0f behaviour the comment describes. Then build and run power_cpu_test and power_cuda_test in a fixed64 tree, for example by adding them to the variant validation.

#### checkpoint_integrity 'version_one' restart expects success but C++ only accepts v1 headers in fixed32 builds

`tests_python#3` · **medium** · bug · confirmed

- **Where:** `tests/run_checkpoint.py:450` (also `src/io/checkpoint.cpp:520`; `src/io/checkpoint.cpp:523`; `CMakeLists.txt:2445`)
- **Problem:** checkpoint_integrity relabels a freshly written checkpoint as version 1 and requires the restart to succeed. The C++ deliberately accepts v1 only for fixed32 FMM dark-matter runs, so in any COSMICTIGER_POSITION_TYPE=fixed64 build (FMM + traditional) the version_one case is rejected and the whole test fails.
- **Failure scenario:** Configure with COSMICTIGER_POSITION_TYPE=fixed64 (as in build/named_variants_cpu_fixed64). checkpoint_integrity runs, the version-1 relabelled checkpoint reaches readDriverState, and checkpointCompatible returns false. The run throws 'Checkpoint physics, particle layout or integrator differs', so restart() with expectSuccess=True raises 'restart.log exited with 1'. The test fails in a supported build configuration.
- **Fix:** Before the version_one case, read checkpoint_position_bytes() from the baseline payload. If it is 8, run restart('version_one', downgradeToVersionOne, expectSuccess=False) and assert the 'particle layout' diagnostic, or skip with a printed note. Keep expectSuccess=True only for fixed32.


## Low: bugs (66)

Minor or unlikely-to-trigger defects. One entry each: title, location, ID, the verified problem, and the suggested fix.

### Options, driver and startup

- **timestep.txt fopen unchecked; NaN inverse timesteps hit ALWAYS_ASSERT(false)** — `src/core/driver.cpp:613` · `core_options_driver#14` · being edited  
  The timestep.txt fopen result is dereferenced unchecked. In the limiter classification, a NaN sound inverse timestep aborts on a bare ALWAYS_ASSERT(false), while a NaN viscosity or acceleration inverse timestep is silently misreported. *Fix:* Check fp and THROW_ERROR like the other writers. Test each inverse timestep with std::isfinite and throw (or log NONFINITE) naming the bad criterion before classifying.
- **Restart silently ignores a changed cosmology.endRedshift** — `src/core/driver.cpp:1297` · `core_options_driver#23` · being edited  
  For a cosmology restart with run.endTime=0, nothing compares cosmology.endRedshift with the checkpoint's tauMax. scenario.meta records only run.endTime. A changed endRedshift is silently ignored: the run stops immediately or at the old endpoint with no diagnostic, even though README says scenario.meta records the … *Fix:* In driver(), after readCheckpoint for the cosmology scenario with run.endTime==0, compute cosmology::configuredDuration(a0) and compare it with params.tauMax within a relative tolerance.
- **Tracer selection uses a particle total taken from options** — `src/core/driver.cpp:1371` · `particles_domain_drift#16`  
  Gadget restarts get a wrong tracer set because particlesSetTracers runs on the IC particles (driver.cpp:1371) before particlesReadGadget4 replaces them (driver.cpp:1393). Overwritten slots keep the old flags and grown slots have uninitialized ones. *Fix:* Select tracers after the Gadget read, or re-run particlesSetTracers there. Compute totalCount from a global sum of particlesSize(), as the gravity-error sampler does, instead of from initial.numberOfParticles.
- **int(tau / t0) % 8 misidentifies the coarse step about 4.5% of the time** — `src/core/driver.cpp:1659` · `core_options_driver#10` · being edited  
  The every-8-coarse-steps minrung0 relaxation (and nocrop) uses int(tau/t0) without a tolerance, so about 5% of coarse boundaries are misnumbered. In hierarchical runs the relaxation is then skipped or fires on the wrong step. *Fix:* Use `step % 8 == 0`, since step is the exact coarse index when itime == 0 in non-global mode, or `std::lround(tau / t0) % 8`.
- **Final view index is off by one after an interrupt at a coarse boundary and in global mode** — `src/core/driver.cpp:1891` · `core_options_driver#9` · being edited  
  The final Silo view is written as params.step+1 in two cases where step already holds the right index: an interrupt caught at a coarse boundary (state at k is labelled k+1), and every global-integrator ending (the last view skips one cycle number). *Fix:* Use `global ? step : (itime != 0 ? step + 1 : int(std::lround(tau / t0)))` with t0 = tauMax/numberOfSteps, or keep an explicit coarse-index variable that both the regular and final view calls use.
- **run.minRung (and several other options) are never range-checked; out-of-range minRung indexes rungDt out of bounds** — `src/core/options.cpp:516` · `core_options_driver#8` · being edited  
  run.minRung is never validated. With the traditional or hierarchical integrator any value &gt;= kMaxRung (16) forces rungs &gt;= 16 and aborts on ALWAYS_ASSERT(rung &lt; kMaxRung) in the first kick; the hierarchical path first reads rungDt[16] past the array. *Fix:* In processOptions: reject run.minRung &lt; 0; for the traditional and hierarchical integrators reject run.minRung &gt; kMaxRung - 1 (or kMaxRung - 2 for headroom) with a clear message, and leave the global integrator …
- **Plummer scenario silently discards explicit gravity radius options (magic constant)** — `src/core/options.cpp:1645` · `core_options_driver#16` · being edited  
  For run.scenario=plummer, any explicit gravity radius option (including the one in the shipped plummer.ini) is silently replaced by an undocumented-at-use-site formula. A verbose diagnostic also uses 1.98e33 instead of constants::kM0. *Fix:* Apply the Plummer-derived softening only when no radius key is explicit (as openingAngle does), or print a warning. Name the constant (e.g.
- **rho.dat maximum density only covers the root locality's particles** — `src/core/scenario_driver.cpp:51` · `core_options_driver#11`  
  writeScenarioDiagnostics computes the rho.dat maximum density over locality 0's SPH particles only. Multi-locality fixed-time runs therefore report a partial maximum next to globally reduced energies. *Fix:* Add a distributed max-density action over hpxChildren() (like particlesSumEnergies), or carry the maximum density in the reduced energy/hydro step result.

### Scenarios and INI files

- **Blast energy deposit is not normalized on the particle lattice, while the Sedov reference assumes E=1** — `src/scenarios/blast.cpp:91` · `scenarios_configs#5`  
  initializeBlast deposits a continuum-normalized Gaussian (sigma=0.015) with no lattice renormalization. The injected energy m\*sum f_i matches sedov::solution's hard-coded eblast=1 only when the BCC spacing is below about 1.8\*sigma. *Fix:* After filling the lattice, compute S = m\*sum f(r_i) and divide every eint by S. Alternatively, pass the realized energy (and hydro.gamma) into sedov::solution instead of hard-coding eblast=1 and gamma=5/3.
- **Lane-Emden derivative at the origin returns -3 instead of -1/3** — `src/scenarios/polytropic_star.cpp:44` · `scenarios_configs#10`  
  The Lane-Emden right-hand side returns -3 at xi=0 where the limit is theta''(0) = -1/3. The corrupted first RK4 step leaves density errors that grow to about 1.5e-4 at xi=3 and 3e-3 at xi=3.6 (below 4e-7 with the correct limit). *Fix:* Return -1.0 / 3.0, or start the integration from the series theta = 1 - xi^2/6, theta' = -xi/3 at a small xi.
- **RT lattice size truncates a cube root that sits just below an integer, losing 37% of the nominal particles** — `src/scenarios/rayleigh_taylor.cpp:25` · `scenarios_configs#14`  
  Truncating cbrt(ppd^3/96), which lies just below an integer for multiples of 16 and 32, makes rayleigh_taylor.ini (ppd 32) create 20736 particles instead of about 32768. That is 63% of the nominal count and 0.86 of the intended linear resolution. *Fix:* Use std::lround for quarterDim and keep the qd &gt;= 1 check. Note that particleCount may then slightly exceed ppd^3; I found no code that requires it to stay at or below that.
- **Rayleigh-Taylor perturbation test copies the production formula; bracket placement looks wrong** — `src/scenarios/rayleigh_taylor.cpp:67` · `tests_other_cpp#9` · plausible  
  The RT seed amplitude/4\*(1+cos(8πt)\*(1+cos(6πt))) has a range of [-0.77, 3]\*amp/4, which is inconsistent with the /4 normalization of the usual (1+cos)(1+cos)/4 product seed and suggests a misplaced parenthesis. *Fix:* Decide on the intended seed and document it with a citation in rayleigh_taylor.cpp and the configuration notes. Move the parenthesis if the product form was meant.
- **polytropicStar silently runs without self-gravity when gravity is unavailable or disabled** — `src/scenarios/scenario.cpp:29` · `scenarios_configs#13`  
  In SPH-only builds polytropicStar runs with no warning as a non-self-gravitating sphere, because only plummer and glass are rejected. It disperses while its plan is still expressed in gm-based dynamical times. tests/run_sph_only.py currently relies on this as a one-step smoke test. *Fix:* Warn, or require an explicit opt-in, when polytropicStar runs with gravity disabled. If adding a requiresGravity flag that rejects it, also remove polytropicStar from tests/run_sph_only.py or change that test to expect …
- **Scenario reference solutions and initial conditions hard-code gamma=5/3 instead of using or validating hydro.gamma** — `src/scenarios/sedov.cpp:1107` · `scenarios_configs#11`  
  The blast reference (sedov::solution) hard-codes gamma = 5/3, and the polytrope sets u = n\*P/rho (valid only for gamma = 5/3), while SPH and the other scenarios use hydro.gamma. Any other gamma silently produces wrong blast error norms and an out-of-equilibrium star. The assumption is documented but not validated. *Fix:* Pass getOptions().hydro.gamma to sed1d (and include it in the cache key). Set eint = P/((gamma-1)\*rho) in the polytrope. Alternatively, reject or warn when hydro.gamma != 5/3 for blast and polytropicStar.
- **sedov::solution interpolates on a grid spacing that differs from the computed table; cache ignores rmax and ndim** — `src/scenarios/sedov.cpp:1168` · `scenarios_configs#12`  
  The interpolation lambda divides by the post-increment nstep (10002) instead of the table spacing rmax/10000, so the profile is evaluated at 1.0002\*r (about a 2.8e-5 shock shift, negligible). The time-only cache key and the assert-only bounds checks are latent hazards that today's single fixed caller never triggers. *Fix:* Compute dr from the table's nstep before the increment (or store it explicitly). Key the cache on (time, rmax, ndim, gamma). Replace the asserts with clamping or a throw.

### Chemistry and star formation

- **New CUDA entry points never call cudaSetDevice(); batches run on whatever device the current OS thread has** — `src/chemistry/primordial_chemistry.cu:62` · `chemistry_star_formation#10` · plausible · being edited  
  The primordial chemistry CUDA batch and the SH03 CUDA batch run on whatever device is current for the calling OS thread, with no rank%count device selection. On multi-GPU nodes where every rank sees every GPU, a rank&gt;=1 batch can run on GPU 0 (performance and memory headroom only). *Fix:* Call the app's cudaSetDevice() (rank % count) in the SH03/chemistry SPH bridges before runCuda or integrateBatch(Backend::Cuda), or select the device inside runCuda and pass a device index into the standalone chemistry …

### Core runtime

- **Bounds-checked Vector::operator[] casts to std::vector&lt;T&gt;\*, ignoring Alloc (UB for Vector&lt;T, PinnedAllocator&lt;T&gt;&gt;); size_t printed with %i** — `cosmictiger/core/containers.hpp:41` · `core_runtime_infra#15`  
  In Debug builds, the bounds-checked Vector::operator[] reinterprets `this` as std::vector&lt;T, std::allocator&lt;T&gt;&gt;\*. That is undefined behaviour for the PinnedAllocator vectors used in sph.cpp and works only because the allocator is empty. *Fix:* Use `return BaseType::operator[](i);` in both overloads. In DO_BOUNDS_CHECK, cast both values to long long and print them with %lld, since the macro serves both the size_t and the int callers.
- **Assertion text goes to unflushed std::cout and is lost on abort() when stdout is redirected; the SIGABRT handler then writes a second trace** — `src/core/assert.cpp:20` · `core_runtime_infra#4`  
  cosmictigerAssert sends its 'Condition ... failed ... on rank N' report and trace to unflushed std::cout and then calls abort(), which in glibc 2.43 does not flush stdio. With stdout redirected or piped, the report never reaches the log (it survives only in stack_trace.txt), and the SIGABRT handler appends a second … *Fix:* Write the assertion report to stderr (std::cerr, fputs(stderr) or write(STDERR_FILENO)), and call std::cout.flush() and fflush(stdout) before abort().

### CUDA infrastructure

- **CudaUnorderedMap iterator, erase() and clear() are wrong (unused today)** — `cosmictiger/cuda/cuda_unordered_map.hpp:68` · `cuda_infra#4`  
  CudaUnorderedMap's unused members are broken: operator++ (both forms), begin(), operator\* (ill-formed once instantiated), erase() (removes the wrong entry, decrements size even when the key is absent, divides by zero on an empty map) and clear() (keeps a stale size\_). Only operator[] is used today. *Fix:* Delete the unused members (operator\*, both operator++, begin, erase, clear) and keep Iterator as a find-result handle. If they are needed later, fix them: use bucket\_, skip empty buckets, return `old` from the postfix …
- **Pool blocks are only 8-byte aligned (every other block is 8 mod 16)** — `src/cuda/cuda_memory.cu:97` · `cuda_infra#12`  
  Pool allocations are only 8-byte aligned: the 8-byte header plus block sizes of 2^bin + 8 make the first block, and about every other block after it, 8 mod 16. Any future pooled type with alignof &gt;= 16, or a vectorized 16-byte load from DeviceVector storage, would fault with a misaligned-address error. *Fix:* Use a 16-byte (or 32-byte) header and round each carved block size up to that alignment, so returned pointers carry the header's alignment. Add static_assert(alignof(T) &lt;= kCudaPoolAlignment) in DeviceVector.
- **Pool or malloc is chosen on every call from the mutable compute.cuda option, not recorded per allocation** — `src/cuda/cuda_memory.cu:170` · `cuda_infra#8`  
  cudaAllocate and cudaRelease pick the pool or malloc from the mutable compute.cuda option at each call, and pool allocations carry no provenance. An allocation that outlives a compute.cuda toggle is therefore released by the wrong allocator (glibc or "Corrupt free!" abort), and nothing asserts that the pool exists. *Fix:* In cudaRelease, decide by address: send pointers inside [heapBegin\_, heapBegin\_ + heapSize\_) to memory-&gt;free and everything else to freeManagedMemory.

### Particles, domains and drift

- **SPH migration leaves received particles in nondeterministic arrival order** — `src/particles/domain.cpp:557` · `particles_domain_drift#10`  
  The SPH migration path installs received particles in whatever order the receiveSphParticles actions took the mutex, with no sort, unlike the position-sorted non-SPH path. Because tree partitioning is unstable, the order of particles within leaves, and so the floating-point sums, can differ between identical runs with … *Fix:* Tag each parcel with its source rank and a sequence number and concatenate in (rank, sequence) order, or sort the received records (by position or star-formation id) before installing them after the retained ones.
- **drift() silently narrows the scale factor, dt and conformal times to float** — `src/particles/drift.cpp:22` · `particles_domain_drift#4` · also `tests_other_cpp#6`  
  cpuDrift and cudaDrift take a, dt, tau0, tau1 and tauMax as float, so the double-precision drift() API silently rounds them to 24 bits even though DriftParameters and the CUDA kernel take double. The numerical impact is small. *Fix:* Change the cpuDrift and cudaDrift prototypes (drift.hpp:18-20, drift.cpp:22, drift.cu:175) and the test stub in tests/drift_dispatch.cpp to take double.
- **CUDA drift enables the lightcone for tauMax &lt;= 0, unlike the CPU path** — `src/particles/drift.cu:192` · `particles_domain_drift#8`  
  cudaDrift and cudaDriftKernel enable lightcone processing whenever lightcone.enabled &amp;&amp; tauMax - tau1 &lt; 1, then divide by tauMax. They lack the tauMax &gt; 0 guard that cpuDrift and cpuDriftKernel apply and the CPU test asserts. No production caller passes tauMax &lt;= 0, so this is a latent divergence between the twins. *Fix:* Add `&& tauMax > 0.0f` to doLc at drift.cu:192 and in the kernel, or share one predicate helper between drift.cpp and drift.cu.
- **32-bit drift rounds every position down at each step** — `src/particles/drift_cpu.cpp:117` · `particles_domain_drift#17`  
  In the default fixed32 build, the CPU drift (SIMD and scalar) and the CUDA drift compute the new absolute position in double and store it through Fixed32's truncating double conversion. *Fix:* In the 32-bit branches of drift_cpu.cpp (SIMD and scalar tail) and drift.cu, round to nearest instead of truncating. One option is to add the displacement in raw units, `raw += PositionInteger(llrint(dx * 0x1p32))` …
- **Lightcone crossing uses post-drag velocity while displacement uses pre-drag; test enshrines it** — `src/particles/drift_cpu.cpp:139` · `tests_other_cpp#7`  
  With drag != 0, both CPU and CUDA drift damp the velocity before the lightcone crossing solve. The crossing is therefore solved with the post-drag velocity while the particle moved with the pre-drag velocity, and the emitted position is not the true crossing point. *Fix:* Compute the crossing with the velocity used for the displacement: pass pre-damping velocities to appendDriftLightconeEntries and use them in the CUDA solve, while still storing the post-drag velocity in the entry if …
- **Grown particle slots keep stale or uninitialized lastgroup and tracer values** — `src/particles/particles.cpp:991` · `particles_domain_drift#11`  
  particlesResizeStorage leaves new lastgroup slots uninitialized: on reallocation it fills Lgrp from the old capacity instead of the old size, it never resets [oldsz, sz), and it never initializes particlesTr. *Fix:* On reallocation, fill Lgrp from size (not capacity) up to newCapacity. After every resize, reset particlesLgrp[oldsz, sz) to kNoParticleGroup and particlesTr[oldsz, sz) to 0 when those arrays exist, as is already done …
- **Misleading messages and leftover debug output in particle I/O and drift** — `src/particles/particles.cpp:1859` · `particles_domain_drift#19`  
  particlesReadGadget4 reports a read failure as 'for writing' and never validates record markers, so headers from builds with NTYPES other than 2, and files containing gas, are silently misparsed instead of rejected; it also reads npart[1] entries from the start of blocks that begin with the type-0 particles. *Fix:* Change the message to 'for reading'. Check each marker against sizeof(GadgetIoHeader) and against 3\*sizeof(float)\*sum(npart) for the position and velocity blocks, and either reject npart[0] != 0 (this is a DM-only …

### I/O, checkpoints and tools

- **readCheckpointSelection resolves the checkpoint path differently from readCheckpoint in run.test mode** — `src/io/checkpoint.cpp:893` · `io_checkpoint_output_tools#15`  
  In run.test mode the startup algorithm-selection probe looks for checkpoint.N under output.directory, while readCheckpoint/writeCheckpoint use the working directory through outputPath(). A test-mode restart without explicit selectors therefore fails or probes the wrong checkpoint. *Fix:* Resolve both paths with one helper that applies outputPath's run.test and empty-directory rule. For example, pass opts.run.test into readCheckpointSelection, or build the probe path with the same rule as …
- **Tracer/slice Silo writers and view.N.dat ignore I/O errors** — `src/io/output.cpp:49` · `io_checkpoint_output_tools#14`  
  The tracer and slice Silo writers ignore DBCreateReal/DBPut\*/DBClose failures (Silo only prints its default stderr message), and outputView ignores fwrite/fclose failures, so truncated or missing tracers/slice/view outputs do not stop or flag the run. *Fix:* Throw if DBCreateReal returns NULL, wrap every DBPut\* and DBClose call in detail::checkViewWrite (or an equivalent), and check the fwrite counts and fclose in outputView, calling THROW_ERROR on failure.
- **Slice image is transposed: pixels stored y-fastest but written to Silo as x-fastest** — `src/io/output.cpp:134` · `io_checkpoint_output_tools#7`  
  The projected slice is transposed: pixels are accumulated y-fastest (xi\*res+yi), but Silo's default row-major quadvar layout is x-fastest, so x and y are swapped in slice.N.silo. The unit string is also passed as the axis label. *Fix:* Use `int const j = yi * res + xi;` (or add DBOPT_MAJORORDER=DB_COLMAJOR). Keep the axis labels as "x" and "y" and pass "Mpc/h" through DBOPT_XUNITS and DBOPT_YUNITS.
- **groups2halofunc crashes on degenerate catalogues, and debug prints corrupt its stdout output** — `src/tools/groups2halofunc.cpp:118` · `io_checkpoint_output_tools#9`  
  groups2halofunc indexes bins[INT_MIN] (undefined behaviour, likely a crash) when all halo masses are equal because dlog==0. It prints -inf rows for an empty catalogue, and a leftover debug PRINT with a hard-coded particle mass writes integer rows into the same stdout stream as the mass-function table. *Fix:* Exit with a message on empty input. Handle massMin==massMax explicitly, for example with a single bin or a widened range. Remove the debug PRINT or send it to stderr.
- **healpix2silo flips the map north/south and trusts the input header** — `src/tools/healpix2silo.cpp:138` · `io_checkpoint_output_tools#8`  
  healpix2silo passes latitude+pi/2 as the HEALPix colatitude, so the Mollweide image has the south pole at the top and +z structure from view.N.dat appears at the bottom. The tool also does not validate the header count against 12\*nside^2 and ignores Silo errors. *Fix:* Call `ang2pix_ring(nside, std::numbers::pi / 2.0 - psi, lambda, &ipring)`. Require 0 &lt; nside &lt;= 8192 and size == 12\*nside\*nside before filling healpixData. Check DBCreate, DBPut\* and DBClose and exit nonzero on failure.

### Gravity: global FMM (dehnen)

- **CPU and CUDA Ewald acceptance clamp the near-distance floor differently** — `cosmictiger/gravity/dehnen/periodic_cpu.hpp:165` · `gravity_dehnen_global_fmm#7`  
  The CPU Ewald acceptance clamps (0.5-(Rs+Ro))^2 at zero after squaring, a no-op, while CUDA clamps before squaring. The CPU therefore uses a positive floor (Rs+Ro-0.5)^2 for large pairs. Accept/open decisions agree for every theta &lt;= 1 and diverge only for unvalidated opening angles above about 1.4. *Fix:* Use `r2 = simdMax(r2, sqr(simdMax(simdBroadcast<SimdFloat>(0.5) - (selfRadius + otherRadius), simdBroadcast<SimdFloat>(0))));` to match CUDA, delete otherBox, and optionally reject gravity.openingAngle &gt; 1 (or document …
- **Global-FMM Ewald leaf criterion can be unsatisfiable, causing infinite subdivision and a misleading 'Maximum depth exceeded'** — `src/gravity/dehnen/distributed_tree.cpp:85` · `gravity_dehnen_global_fmm#8`  
  The global-FMM tree forces non-empty cells to split until boxR &lt; 0.125 - 0.25\*h, but nothing validates that this bound is positive. For a normalized support radius h &gt;= 0.5, tree construction recurses to depth 128 and fails with an unhelpful 'Maximum depth exceeded'. *Fix:* When the global FMM is selected, reject a normalized gravity.supportRadius that leaves 0.125 - 0.25\*h non-positive (or below a small margin), with a message naming the option.
- **Staging exception paths leave tasks that capture stack locals still running** — `src/gravity/dehnen/gpu_staging.cpp:334` · `gravity_dehnen_global_fmm#15`  
  dehnenStageGpu uses throwing hpx::wait_all (HPX 1.11) on one task group at a time with no catch or scope guard. If a launch or an earlier group throws, the fetch continuations, the futs3 tasks or the async sort keep running against freed stack locals (particleMap, particleRequests iterators, treeMap, workitems), and … *Fix:* Wrap the body after line 120 in try/catch. On error, join every outstanding group with hpx::wait_all_nothrow (futs1, futs2, futs3, and sfut if valid()), restore particlesResize(originalParticleCount), and rethrow.

### Gravity: tree, walk and kick

- **PC/PP sink base index truncated to int while doKick uses ParticleInteger** — `cosmictiger/gravity/gravity_cuda.hpp:457` · `gravity_cuda_kick#2`  
  cudaGravityPcDirect (gravity_cuda.hpp:457) and cudaGravityPpDirect (530) truncate the sink base index self.particleRange.first + sinkBegin to int, while doKick, the kernel and the CPU twin use ParticleInteger. *Fix:* Declare `ParticleInteger const sinkFirst` in both functions, matching doKick (kick_cuda_kernel.hpp:52).
- **Dead and uninitialized fields in KickParameters and TreeCreateParameters** — `cosmictiger/gravity/kick.hpp:103` · `gravity_tree_walk_cpu#12` · also `tests_other_cpp#2`  
  KickParameters carries dead fields (minLevel, velInit, nodeLoad) that every caller must set and every remote KickAction serializes. Its constructor leaves 15 fields indeterminate, and value-initialization does not fix that. *Fix:* Remove the dead fields from the structs, serialization and all callers (driver, fmm_batch, tests). Drop the hmax parameter.
- **All 32 lanes read-modify-write the same walk-phase word without lane election** — `cosmictiger/gravity/kick_cuda_kernel.hpp:716` · `gravity_cuda_kick#3` · plausible  
  `sparams.back().phase += 1` (lines 716 and 738) is a non-atomic read-modify-write of one global word by all 32 lanes with no lane election, which is formally a data race. It is correct today only because the warp is converged there: after \__syncthreads() at 738, and after \__syncthreads() or a uniform mbarrier wait at … *Fix:* At both sites use `if (tid == 0) { sparams.back().phase += 1; } __syncwarp();`, matching the elected updates in DeviceVector and StackVector. The following pushBack's syncthreads then orders the write for every lane.
- **Register residency limit ignores the four register sub-partitions, so it overestimates (9 vs 8 at order 4)** — `cosmictiger/gravity/kick_cuda_launch.hpp:80` · `gravity_cuda_kick#7`  
  kickRegisterResidencyLimit divides all SM registers by the per-warp allocation, ignoring CUDA's four register sub-partitions. For 171-227 registers per thread it overestimates residency (224 registers gives 9 vs 8 achieved). *Fix:* Compute the limit as 4\*floor((regsPerSM/4)/regsPerWarp), capped by blocks and warps per SM, or take it from cudaOccupancyMaxActiveBlocksPerMultiprocessor(&amp;n, kernel, 32, 0).
- **Static launch buffers are left dangling if a reallocation fails** — `cosmictiger/gravity/kick_cuda_launch.hpp:302` · `gravity_cuda_kick#8`  
  The launch-packet reallocation raises dataSize and frees the old buffers before the new allocations are known to succeed, and malloc is unchecked. Because CUDA_CHECK (cuda.hpp:44-46) only prints and re-invokes the failed call, a cudaMalloc failure is not raised at that line. *Fix:* Allocate into temporaries with a throwing check (hpx::cuda::experimental::check_cuda_error, or THROW_ERROR on non-success, plus a malloc null check).
- **KickWorkspaceTreeIdHash multiplies in int and can overflow** — `cosmictiger/gravity/kick_workspace.hpp:25` · `gravity_tree_walk_cpu#9`  
  KickWorkspaceTreeIdHash computes index\*hpxSize() in int, which is signed-overflow UB once the product passes INT_MAX. That needs on the order of 1e11 particles across many localities. Effects are limited to hash quality. *Fix:* return size_t(id.index) \* size_t(hpxSize()) + size_t(id.proc); and make the same widening in TreeIdHash.
- **close/direct interaction counters are rank-0, GPU-only, but printed as run-wide statistics** — `src/gravity/kick.cu:114` · `gravity_cuda_kick#6` · being edited · also `core_options_driver#22`  
  The close/direct PP interaction counters are per-process \__managed_\_ variables. They are incremented only in cudaGravityPpDirect, enabled and reset only on locality 0 via plain calls from driver(), and read without a cross-locality reduction. *Fix:* Broadcast enable/reset to all localities with an HPX action (or carry the flag in the kick parameters), and reduce iclose, idirect and numberOfProcessedSinkParticles across localities in getGravityCounters.
- **Global-FMM midpoint split has no coincident-particle fallback; more than bucketSize identical positions recurse to kMaxDepth and throw** — `src/gravity/tree_build.cpp:106` · `gravity_tree_walk_cpu#5`  
  Global-FMM tree builds split only at the geometric midpoint and have no tie fallback. More than compute.bucketSize particles at the same fixed-point position put all particles on one side at every level until treeBuild throws 'Maximum depth exceeded'. The PMMM (bounded) path handles the same input. *Fix:* When a midpoint split leaves one side empty, reuse the bounded-geometry fallback for every geometry: split at the occupied raw extent, or by index when the extent is zero.

### SPH

- **Early return on an empty checklist at a global node skips the neighbour-list reset of descendant localities** — `src/sph/sph.cpp:174` · `sph_infrastructure#5`  
  A Neighbors walk that hands an empty checklist to a global node above the local roots returns before reaching those roots. Those localities skip sphTreeFreeNeighborList/sphTreeClearNeighborRanges and keep the previous walk's leaf ranges. This needs at least 5 localities, or at least 3 when no gas is active. *Fix:* Reset per-locality neighbour state independently of traversal: at level 0, broadcast free plus clear to all localities, as sphTreeCacheClear does.
- **Bitwise OR instead of AND makes every interaction-selection test true** — `src/sph/sph.cpp:203` · `sph_infrastructure#2`  
  sph.cpp uses `innerSelection | kSphInteractionsI/J` where it means `&` (lines 181, 192, 203, 206), so the intended gather-only walks for prehydro1 build gather-plus-scatter lists, and the box sanity checks run on every walk, including Boxes walks. Results stay correct; the cost is wasted walk, staging and scan work. *Fix:* Change `|` to `&` at lines 181, 192, 203 and 206. Prehydro1 is the only I-only consumer and it only gathers, so the change is safe.
- **Staging futures are not drained on exception, leaving tasks with dangling references to stack atomics** — `src/sph/sph.cpp:548` · `sph_infrastructure#7`  
  SphRunWorkspace::run drains its staging futures with get() and no wait_all. A throwing staging task unwinds run()/sphRun and frees the index/particleIndex atomics and the workspace's host vectors while sibling tasks still write them. *Fix:* Call hpx::wait_all(futs.begin(), futs.end()) before the get() loop, or wrap the loop in try/catch with wait_all before rethrowing, mirroring the CPU-worker section.

### Cosmology and initial conditions

- **twolptPhase loops with int over an int64 local volume** — `src/cosmology/initialize.cpp:615` · `cosmology#10`  
  twolptPhase walks the int64 local FFT volume with an int index. In LONG_LONG_PART_INT builds with at least 2^31 cells per locality, this is signed overflow (UB) in the 2LPT source accumulation. The glass `counts` loop at line 810 cannot overflow. *Fix:* Declare the loop index as int64_t (or size_t) in twolptPhase. Leave line 810 alone or change it only for consistency.

### Analysis: lightcone, BH and power

- **Lightcone nside cap uses a radius normalized by tauMax, taken at the outer edge of the interval** — `src/analysis/lightcone.cpp:874` · `analysis_lightcone_bh_power#13`  
  The HEALPix resolution is fixed once per accumulation interval from its outer radius (scaled by R/tauMax). Flushes, especially the final forced one, then process particles down to r of about 0. Near the observer, pixels become narrower than the linking length, and the 8-neighbour pixel search misses FoF links. *Fix:* Recompute nside at flush time from the innermost radius the flush will process, and use nside=0 when that radius is below about nside\*l, for example always in the final flush. Rebuild the map before lcBuffer2homes.
- **CUDA lightcone kernel reads a label that lane 0 writes in the same step without warp synchronization** — `src/analysis/lightcone.cu:120` · `analysis_lightcone_bh_power#17`  
  In cudaLightconeKernel, lane 0 writes the shared group label `a` while other lanes read it, with no warp synchronization. Correctness of the warp-uniform loop exit and the full-mask shuffles relies on implicit lockstep execution. The block size is also assumed to equal the warp size without a static_assert. *Fix:* Load `a` once and broadcast lane 0's value with \__shfl_sync, or evaluate the predicate on lane 0 and broadcast it. Add \__syncwarp() after lane 0's update. Add static_assert(kLightconeBlockSize == kWarpSize).
- **Power error estimate counts conjugate-duplicate modes of the kz=0 plane as independent** — `src/analysis/power.cpp:149` · `analysis_lightcone_bh_power#14`  
  Conjugate-duplicate modes in the kz=0 plane are counted as independent samples in count and pdev, which underestimates the error bars at low k (by about 1.29x in the |k|=1 bin). *Fix:* Give kz=0 modes weight 1/2 in count, power and pdev, or count only one member of each conjugate pair, for example kx&gt;0 or (kx==0 and ky&gt;0) in the kz=0 plane.

### Math and numerics

- **Physical constants disagree: kH0 uses Mpc = 3.086e24 cm while lengths use kMpcToCm = 3.085678e24, and several modules keep their own constant sets** — `cosmictiger/math/constants.hpp:18` · `math_numerics#5` · being edited · also `core_options_driver#21`, `recent_commits_a#6`  
  constants::kH0 implies 1 Mpc = 3.086e24 cm, while kMpcToCm = 3.085678e24 cm. The units.boxLength path in options.cpp combines the two, giving a 1.04e-4 H0/length mismatch and particle masses 2.1e-4 low. *Fix:* Define kH0 = 1e7 / kMpcToCm, or at least derive the boxLength path from one Mpc value, and rename cosmic_constants::kH0 to kH0OverC.
- **distance(double, double) never terminates for infinite or very large separations** — `cosmictiger/math/fixed.hpp:376` · `math_numerics#10`  
  distance(double, double) wraps by repeatedly adding or subtracting 1.0, so it never terminates for infinite input or |difference| &gt;= 2^53. No current caller passes doubles (rockstar.cpp:746 uses Position values), so the bug is latent. *Fix:* Replace the loops with `difference -= std::nearbyint(difference)` (or std::remainder(difference, 1.0)), or delete distance(double, double) and the non-Q0.64 double overloads, since nothing in production uses them.
- **rand1() can return exactly 1.0f and has only float (24-bit) resolution** — `cosmictiger/math/math.hpp:30` · `math_numerics#7`  
  rand1() can return exactly 1.0f: the top 64 rand() values round to (2^31+0.5)/2^31, which narrows to 1.0f. Its output is also quantized to float resolution, and it relies on the unseeded, libc-dependent rand(). Current callers (glass positions wrap 1.0 to 0; Sod jitter) are unaffected. *Fix:* Use a seeded std::mt19937_64 with std::uniform_real_distribution&lt;double&gt; (or std::generate_canonical). At minimum, return double((double(rand()) + 0.5) / (double(RAND_MAX) + 1.0)).
- **Range::shortestDim returns an uninitialized index when every span exceeds 1; split() and longestDim have the same pattern for degenerate boxes** — `cosmictiger/math/range.hpp:244` · `math_numerics#4`  
  Range::shortestDim seeds minSpan with T(1), so it returns an uninitialized index for any box whose spans all exceed 1 (for example integer grid boxes). It has no callers. split() and longestDim use the same uninitialized-index pattern, but their current callers can never reach the degenerate case. *Fix:* Delete shortestDim. Otherwise initialize minDim/maxDim to 0 and seed minSpan/maxSpan from dimension 0 in shortestDim, longestDim and split, so degenerate boxes return a valid index.

### Build system and scripts

- **The project's `native` CUDA-architecture default overrides the standard CUDAARCHS environment variable** — `CMakeLists.txt:47` · `build_cmake_scripts#8` · being edited  
  The project's normal-variable `native` default is set before enable_language(CUDA), so it shadows the cache entry CMake seeds from $ENV{CUDAARCHS}. The first configure ignores CUDAARCHS (failing on GPU-less hosts), and later configures switch to it. *Fix:* Use `if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")`, or seed a cache entry instead of a normal variable.
- **Multi-label LABELS passed to cosmictiger_add_gtest are split by GoogleTest discovery, so every label after the first is lost** — `CMakeLists.txt:708` · `build_cmake_scripts#6` · being edited  
  GoogleTest discovery flattens list-valued PROPERTIES, so `LABELS "sph;cpu"` style arguments to cosmictiger_add_gtest keep only their first label. The rest become a bogus or dangling property, and `ctest -L cpu` silently skips 11 discovered test families. Their add_test twins do carry the cpu label. *Fix:* Add a LABELS keyword to cosmictiger_add_gtest. Apply it after discovery through the generated include file, as already done for ENVIRONMENT: 'set_tests_properties(${&lt;target&gt;\_TESTS} PROPERTIES LABELS …
- **Some multi-locality tests are not gated on the HPX MPI parcelport, and none check the available MPI slots** — `CMakeLists.txt:1276` · `build_cmake_scripts#7` · being edited  
  analysis_cpu_mpi, groupsCatalogue_mpi and gravity_error_mpi are gated only on MPIEXEC_EXECUTABLE, unlike every other multi-locality test, and run_analysis_cpu.py hard-codes `-n`. No test checks MPIEXEC_MAX_NUMPROCS, so TCP-only HPX builds and small Open MPI hosts register tests that cannot pass. *Fix:* Gate the three tests on `HPX_WITH_PARCELPORT_MPI AND MPIEXEC_EXECUTABLE`. Add ${cosmictiger_mpi_transport_args} to groupsCatalogue_mpi.
- **SIMD benchmark pipeline (prepare_simd.py and run_simd.py) cannot work against the current tree** — `benchmarks/prepare_simd.py:268` · `build_cmake_scripts#12`  
  The SIMD comparison scripts are stale and fail at several independent points. The default cache is missing, the cache must name mpi_cxx (absent with Open MPI 5), the dispatcher anchor no longer matches, COSMICTIGER_WITH_ORDER is retired, and BUILD_TESTING=OFF leaves --run.test=simdBenchmark rejected. *Fix:* Either retire the scripts or move them to a clearly historical location, since they target a one-off pre-migration baseline.
- **summarize_ncu_source.py double-counts inlined code; NCU helper hard-codes the ncu path** — `benchmarks/summarize_ncu_source.py:38` · `build_cmake_scripts#20`  
  summarize_ncu_source.py sums per-line aggregates across every file section of a mixed SASS+CUDA export. That export repeats each inlined instruction once per inlining level, so totals and shares are wrong: on a real report, 608,727 samples against the true 282,589. A project doc claim was derived from it. *Fix:* Rebuild the script on summarize_ncu_regions.parse_rows: deduplicate by SASS address (prefer project frames) before summing, and use the deduplicated kernel total as the denominator.
- **build.sh takes the default CUDA architecture from CPU-only build caches** — `build.sh:64` · `build_cmake_scripts#9`  
  build.sh takes its default CUDA architecture from the first build cache with a non-native value, including CPU-only trees whose value is meaningless (75/52/70 here). Without build/release_cuda, a CUDA build would default to compute_75 instead of the local sm_86. *Fix:* Accept a cache only when it contains `COSMICTIGER_WITH_CUDA:BOOL=ON`, and prefer the most recently modified match. Alternatively, default to native, or query `nvidia-smi --query-gpu=compute_cap`, before falling back to …

### FMM generators and tools

- **Nearest-image Ewald radial terms may underflow in float with FTZ for tiny non-zero separations (order 9)** — `tools/cartesian/ewald.py:277` · `fmm_generators_tools#16` · plausible  
  In float with FTZ, the order-9 nearest-image Ewald term d4\*rinv4 is formed as an underflowing r^4 factor times a growing r^-4 factor. It flushes to 0 for nonzero separations below ~3.3e-7 box units, although the true value (~1e-24 scaled) is a normal float. *Fix:* Form d_j\*rinv^k as c_k \* n8r^(j-k) \* s_j, with c_k = (2a/sqrt(pi))(-2a^2)^k precomputed (c_4 ~ 9.2e-24, a normal float), so r^j and r^-k never cancel numerically.
- **CSV export error after headers are sent appends a second HTTP response into the download** — `tools/halo_viewer/halo_viewer.py:452` · `fmm_generators_tools#3`  
  If the SQLite cursor fails after export() has sent the 200 headers, do_GET writes a second complete HTTP 500 response into the CSV body. Because the body ends at connection close, the client gets a truncated download with appended HTTP headers and JSON, and cannot tell that it failed. *Fix:* Record that the response has started (for example set self.\_responseStarted = True after end_headers in export). Once started, do_GET's except paths must not call send_json.

### Tests

- **cuda_memory stub maps ALWAYS_ASSERT to assert(), which is compiled out in Release (NDEBUG)** — `tests/cuda_memory_stubs/cosmictiger/cuda/cuda.hpp:12` · `tests_other_cpp#4`  
  The allocator host-test stub defines ALWAYS_ASSERT as assert(), which is compiled out in the Release test trees (NDEBUG). The allocator's queue-full and corrupt-free guards, which abort unconditionally in production, therefore become a silent return or an out-of-bounds queue index in the host regression. *Fix:* Mirror production's unconditional behaviour in the stub, for example '#define ALWAYS_ASSERT(condition) do { if (!(condition)) { std::fprintf(stderr, "Condition \"%s\" failed in %s on line %d\n", #condition, \__FILE_\_ …
- **PMMM helper decodes the Cullen-Dehnen viscosity tail as two separate arrays; writer interleaves per particle** — `tests/gravity/nitadori/pmmm_evolution_helpers.py:149` · `tests_python#6`  
  For Cullen-Dehnen checkpoints (gas version 6), the PMMM helper reads the viscosity tail as a 2G-float block followed by a G-float block. Production writes it interleaved per particle as alpha, rate, divergence. *Fix:* Decode the tail per particle: take 2 floats for each gas index, plus 1 more when the gradient switch is active (gas versions 6 and 8). Fix the comment, and fix the model index (viscosity[2]) together with this.
- **run_smoke does not clear checkpoint.N.incomplete staging directories, so one killed run blocks all later runs** — `tests/run_smoke.py:63` · `tests_python#13`  
  run_smoke deletes stale checkpoint.&lt;N&gt; directories but not checkpoint.&lt;N&gt;.incomplete staging directories. A run killed or crashed mid-checkpoint therefore makes every later dm/sph smoke run in the same workdir fail validateCheckpointOutput at startup. *Fix:* Also remove entries matching checkpoint\.[0-9]+(\.incomplete)? (and dangling checkpoint.z.\* aliases). Alternatively give each invocation a fresh output subdirectory, rmtree'd first, as prepareCase does in the other …

## Low: stupid stuff (216)

Dead code, drift, misleading comments and docs, needless work, fragile assumptions. None of these is known to change results today.

### Options, driver and startup

- **Capability headers now silently default missing build macros instead of #error** — `cosmictiger/core/build_config.hpp:11` · `recent_commits_a#9`  
  The capability headers now silently default every missing COSMICTIGER_ENABLE\_\* macro (chemistry on, FMM on, traditional-only integration) instead of raising #error. No current target relies on these defaults and the legacy macros they translate are defined nowhere, so the code is dead and could only hide a target that … *Fix:* Restore #error for missing COSMICTIGER_ENABLE\_\* and COSMICTIGER_WITH_GRAVITY/SPH macros, delete the legacy single-solver translation shims, and update the comments.
- **Three disagreeing sources of option defaults (struct initializers, po::default_value/--help, scenario overrides)** — `cosmictiger/core/options.hpp:60` · `core_options_driver#18` · being edited  
  Defaults live in three places that disagree: zero struct initializers for many fields, po::default_value (shown by --help), and scenario-derived overrides in processOptions. Directly built Options and the --help text therefore do not show the effective defaults. *Fix:* Define each default once, for example a constexpr table used both as the struct initializer and in po::value. Have --help say 'scenario default' or 'derived' for output.directory, lightcone.directory, numberOfSteps …
- **INI help comments disagree with the parser (spline degree range, PMMM batch option)** — `default.ini:100` · `scenarios_configs#22` · being edited  
  All tracked INIs describe the gravity and hydro kernelDegree as 'spline 3..9', but the parser, the validation error, the help text, README.md and the tests accept spline degrees 2..9. The pmmmBatchParticles INI comment calls it a 'ready FMM batch' size. *Fix:* Copy the parser help strings into the INI comments ('spline 2..9'; 'target sink particles per ready PMMM batch; oversized cells run alone; ignored by fmm').
- **default.ini is not the 'active source defaults' it claims to be** — `default.ini:187` · `core_options_driver#17` · being edited · also `recent_commits_b#10`  
  default.ini does not match the compiled defaults. It enables chemistry and star formation (so it fails on a default STARS=OFF build), its softening and centralSoftening are 1 ulp off, and it pins cosmology-specific scenario-derived keys. Its spline degree range comment is stale. *Fix:* Generate default.ini from the option table using the exact repr of the compiled defaults. Keep physics toggles at their compiled values, or reword the README to call the file a cosmology SPH+chemistry+stars example.
- **docs/pmmm.md claims global-mode chemistry retries that the driver does not implement** — `docs/pmmm.md:402` · `core_options_driver#19` · being edited  
  docs/pmmm.md:402-405 says global-mode chemistry opening half-steps are retried at a smaller interval when the updated gas state tightens the hydro bound. In fact sphStep2 opens chemistry once with the pre-source selected interval and commits immediately. *Fix:* Rewrite the pmmm.md sentence to match the code, docs/primordial_chemistry_plan.md:292-295 and docs/chemistry_cpu_support.md: the opening chemistry half-step uses the interval chosen from the pre-source bound, and EOS …
- **Docs still show retired option/integrator names** — `docs/terminal_output_example.txt:42` · `recent_commits_a#13`  
  The orphaned docs/terminal_output_example.txt still shows the retired `gravity.fmmOrder : 7`. Current startup output prints `gravity.expansionDegree` (6 for the default build) plus fields the example lacks, such as run.timeIntegrator, gravity.method.far, units.boxLength and compute.positionType, and … *Fix:* Regenerate docs/terminal_output_example.txt from a current run of configurations/rayleigh_taylor.ini, or delete it since nothing links to it.
- **Driver dead code and ignored parameters** — `src/core/driver.cpp:103` · `core_options_driver#13` · being edited  
  driver.cpp carries a lot of dead state: an uncalled flushTimings and its accumulation, write-only timing globals, unused locals and empty loops and branches, an always-ready lcFut, and a never-read velInit and minLevel (always 9). *Fix:* Delete the dead globals, locals, loops, branches and the flushTimings/timings pair. Drop or honour the verbose, phase and drag parameters, and remove KickParameters::velInit and minLevel, or actually use them.
- **sphStep1/sphStep2 overwrite their verbose parameter with true (leftover debug hack)** — `src/core/driver.cpp:199` · `tests_sph_chem_cpp#13` · being edited  
  sphStep1 and sphStep2 overwrite their `verbose` parameter with `true` (src/core/driver.cpp:199 and :235), so the parameter is dead and callers passing false (tests/sph/star_formation.cpp:601-604) cannot suppress diagnostics. *Fix:* Delete both `verbose = true;` lines, so the per-call flag is honoured on top of the global run.verbose gate. Alternatively, drop the `verbose` parameter from sphStep1/sphStep2 and their declarations in …
- **starFormationRefreshEos runs twice back-to-back after closing, and its pre-density call is always superseded** — `src/core/driver.cpp:472` · `recent_commits_b#3` · being edited  
  Two of the five SH03 EOS refreshes per SPH step do nothing. The call at driver.cpp:472 is recomputed unchanged by the call at 514, or is never read on the closingOnly terminal path. *Fix:* Delete the refresh at line 472. Delete the refresh at line 338: when variableViscosity is on, the refresh pass always runs, so the 'keep it if the refresh pass is skipped' case never happens.
- **Restart re-runs full scenario initialization, and finishSod silently depends on that side effect** — `src/core/scenario_driver.cpp:68` · `scenarios_configs#19`  
  Fixed-time scenarios always run the full initializer before the restart check (scenario_driver.cpp:68-69 vs 99-102). Every particle, including the polytrope's O(sum n^2 x iterations) sphere relaxation, is generated and then overwritten by readCheckpoint. *Fix:* Split ScenarioDefinition::initialize into (a) a cheap plan/options step that returns the stages, sets gasMass and other options, and carries the finish data, and (b) createParticles(). Call only (a) on restart.
- **Scenario stage durations are silently rescaled to step-count proportions** — `src/core/scenario_driver.cpp:93` · `core_options_driver#20`  
  runFixedScenario throws away each stage's requested duration and rebuilds it from the stage's share of coarse steps. Requested stage durations are therefore kept only when all stages have equal duration per step. *Fix:* Assign coarse steps from cumulative duration fractions: cumulativeSteps_i = clamp(lround(cumulativeDuration_i/defaultDuration\*numberOfSteps), ...), then stage.duration = duration\*steps_i/numberOfSteps.
- **Hierarchical branch in runFixedScenario is unreachable** — `src/core/scenario_driver.cpp:185` · `core_options_driver#12` · being edited · also `recent_commits_a#12`  
  Fixed-time scenarios always have hydro enabled and the hierarchical integrator is rejected with hydro, so runFixedScenario's hierarchical branch and the misnamed hierarchicalMesh flag are dead code. *Fix:* Delete the branch and the flag, or replace them with ALWAYS_ASSERT(!baseOptions.run.hierarchicalGravity) so the invariant is explicit.
- **Leftover Ewald Green's-function debug dump runs in every FMM simulation** — `src/main.cpp:116` · `recent_commits_a#11`  
  Every FMM gravity run writes ewald.txt, a self-described debug dump of 500 Ewald Green-function samples, into its output directory (the CWD for run.test). soft.txt is written by every gravity build. Nothing uses either file except an SPH-only test that checks they are absent. *Fix:* Write ewald.txt and soft.txt only under a debug/verbose option or a dedicated run.test. Update run_sph_only.py to use a different marker that gravity was not initialized, such as a log line, and update …

### Scenarios and INI files

- **Disc setup: z-layers asymmetric about the midplane; initial support radius uses the wrong density** — `src/scenarios/disc.cpp:42` · `scenarios_configs#20`  
  initializeDisc places its six z-layers at -3..+2 lattice spacings from z=0.5, so the disc's midplane sits 0.5 dx (more for odd dim) below the centre of the imposed central force. *Fix:* Use z = 0.5 + (k + 0.5)\*dx for k in [-nz, nz), or 2nz+1 layers centred on 0.5. Compute the initial h from the actual lattice density rho = m\*dim^3, optionally widened for the thin slab.
- **Plummer default duration is about 0.1 crossing time, does not scale with plummerRadius, and the profile is not truncated** — `src/scenarios/plummer.cpp:111` · `scenarios_configs#18` · being edited  
  The Plummer default duration is a hard-coded 1e-4, about 0.1\*sqrt(a^3/GM) at a = 0.01, and does not scale with plummerRadius. The untruncated profile puts about 0.06% of particles beyond half the box, where they wrap periodically. *Fix:* Set tauMax = k\*sqrt(a^3/(G\*M)) with a documented k. Truncate or resample radii beyond about 0.5 (or a chosen r_max). Optionally draw speeds from the Plummer distribution function (Aarseth-Henon-Wielen rejection …
- **Polytrope shells use the thin-shell volume (central mass deficit) and a magic mass constant** — `src/scenarios/polytropic_star.cpp:187` · `scenarios_configs#15`  
  Shell occupancy uses the midpoint volume 4\*pi\*r^2\*dr, which misses (pi/3)\*dr^3 per shell, so the innermost shells are under-filled: 3 vs 4.19 particles in the first shell, -7% cumulative within two shells, and a 2-3% low SPH central density. *Fix:* Use nValue = lround((4\*pi/3)\*((r+dr2/2)^3 - (r-dr2/2)^3)\*d/m). Derive gasMass from named constants (xi1, rho_c/rho_mean) or from the integrated Lane-Emden mass.
- **RT and disc 'self-gravity' and the polytrope's G come from cosmological units (G\*m about 1e-11), so self-gravity is 1e-7 to 1e-11 of the imposed forces** — `src/scenarios/scenario.cpp:26` · `scenarios_configs#9`  
  Fixed scenarios inherit the cosmological G (gravity.gm ~ 1e-11 at ppd 32-64), so the documented self-gravity in rayleighTaylor and disc is about 1e-11 (RT) and 1e-7 (disc) of the imposed fields. Each step still pays for a full gravity solve, contrary to the README/INI wording. *Fix:* Set gravity=false for rayleighTaylor and disc in the scenario table, remove 'self-gravity' from README/INI headers, and update the gravity flags in tests/run_scenarios.py.
- **Leftover dead code, debug output, abort() paths and unseeded global RNG in scenario helpers** — `src/scenarios/sod.cpp:107` · `scenarios_configs#21`  
  The scenario helpers carry leftover cruft: dead jitter code in Sod that still consumes the global RNG; a float duration constant (0.0900000036); mislabelled 1/h^3 values printed as densities; unused locals and printf+abort() in exact_sod's rtbis (latent, unreachable for the fixed Sod states); *Fix:* Delete kEta jitter, the -0.0 literals, xmax/numcells and the OCTOTIGER branch. Make kDuration a double 0.09. Fix the print labels or print rho.
- **sphere.cu lacks the CPU path's n&lt;2 guard; convergence ratio is not K/W** — `src/scenarios/sphere.cu:133` · `scenarios_configs#16`  
  The relaxation stop test divides sum(0.5\*m\*v^2) by sum_i sum_j m/r_ij, which equals (K/W)/(2n), so the documented K/W &lt; 1e-10 criterion loosens to K/W &lt; 2n\*1e-10 as shells grow. The CUDA kernel lacks the CPU's n&lt;2 guard, but with nvcc fast-math NaN comparisons are false, so a 0/1-point shell would only run 100 wasted … *Fix:* Use error = kin/(0.5\*m\*phi), or correct the sphere.hpp comment and tolerance. Mirror the n&lt;2 early return in the kernel for CPU/GPU parity. Demote the per-shell prints to detailPrint.

### Chemistry and star formation

- **SH03 adapter uses the proton mass while the chemistry boundary uses m_H; physical constants duplicated across headers** — `cosmictiger/sph/springel_and_hernquist/cooling.hpp:9` · `chemistry_star_formation#15` · being edited  
  The SH03 CIE adapter and the startup nH_th print use the proton mass (1.67262e-24 g) as the hydrogen mass, while the chemistry/SPH boundary and the SH03 CUDA kernel use constants::kMh (1.6733e-24 g). *Fix:* Use units::kHydrogenMass (or one shared primordial constant) in the SH03 adapter and the nH_th print. Replace the literal 1.602176634e-12 in star_formation_kernel.hpp with primordial::kElectronVolt, and take the …
- **AtomicCooling::cool and maximumEnergy are dead in production; the comment claims cool() is the single radiative-loss site** — `cosmictiger/sph/springel_and_hernquist/cooling_impl.hpp:169` · `recent_commits_b#7` · being edited  
  AtomicCooling::cool(), maximumEnergy() and minimumEnergy() (the last used only by cool()) have no production callers; only tests/springel_and_hernquist.cpp uses them. The comment inside cool() claims it is the single per-source-solve radiative-loss site, which misleads readers about where cooling actually happens. *Fix:* Either remove cool()/maximumEnergy()/minimumEnergy() and their test cases, or label them clearly as test-only reference integrators and rewrite the 205-206 comment to say production cooling is done by the chemistry …
- **Test-only SH03 API: cool(), starFormationProbability(), windProbability(), relaxationTime are unused in production, which re-implements the probabilities inline** — `cosmictiger/sph/springel_and_hernquist/equations.hpp:168` · `chemistry_star_formation#13` · being edited  
  AtomicCooling::cool/maximumEnergy and the SH03 starFormationProbability/windProbability helpers are exercised only by unit tests. Production computes the conversion and wind probabilities inline from the accumulated optical depth in both the CPU and CUDA paths, so tests of the helpers do not cover the formulas that … *Fix:* Add one shared CUDA_EXPORT helper that takes an accumulated optical depth (and mass loading), use it in both production paths and test it.
- **Stale doc: molecular9_hot.md says the SH03 fixtures refuse compute.cuda, contradicting the new CUDA SH03 backend** — `docs/molecular9_hot.md:251` · `chemistry_star_formation#18`  
  In the working tree, docs/molecular9_hot.md:250-251 says in the present tense that the SH03 fixtures refuse compute.cuda, but the in-process star_formation fixture now accepts CUDA and runs a CPU/CUDA parity check through the new star_formation_production_cuda\_{atomic6,molecular9} tests. *Fix:* When the CUDA SH03 backend is committed, change molecular9_hot.md:250-251 to say the star_formation fixture now has GPU coverage through star_formation_production_cuda\_\*, while starFormationParticles …
- **Fatal chemistry/SH03 errors report raw enum ints and batch indices, not the offending particle state** — `src/sph/sph_chemistry.cpp:184` · `chemistry_star_formation#16` · being edited · also `recent_commits_b#9`  
  Run-ending chemistry and SH03 errors report raw numeric Status values and indices into filtered batches, with no particle identity or thermodynamic state. Status 2 does not distinguish the 10 K/1e8 K temperature limits from the 1e4 cm^-3 density ceiling, and every SH03 equilibrium failure is labelled as an … *Fix:* Add a statusName(Status) helper. Report the particle (StarFormationState::id, or the gas index via the index map), nH, stored/thermal energy and temperature, interval, and the StepResult error and substep counts.
- **SH03 CIE evaluations use FloatFloat without the StrictFloatEnvironment guard every chemistry entry point installs** — `src/sph/star_formation.cpp:75` · `chemistry_star_formation#12` · being edited  
  The SH03 host CIE evaluations (model calibration, per-particle equilibria, view diagnostics) run FloatFloat chemistry primitives without the StrictFloatEnvironment that every chemistry entry point installs. *Fix:* Construct and check a StrictFloatEnvironment in the SH03 host entry points (model(), equilibrium() and the CPU SH03 loops, writeStarFormationGasFields), or inside AtomicCooling's out-of-line host definitions in …
- **CPU star-formation path duplicates the shared CUDA_EXPORT kernel logic (and the RNG) instead of calling it** — `src/sph/star_formation.cpp:412` · `chemistry_star_formation#8` · being edited  
  star_formation.cpp keeps a hand-maintained CPU copy (about 240 lines) of the CUDA_EXPORT prepare/finish/evaluateEos/maximumTimestep/openNewStar operations and the SplitMix RNG. It is an intentional oracle guarded by the CUDA production parity tests, but every change must be made twice, and CPU-only builds cannot … *Fix:* Preferred: have the CPU path build Input/Parameters and call the star_formation_backend functions (optionally in hpx::for_loop), and keep an independent reference only in tests.
- **Chemistry for a wind that expires mid-half-step in dense gas is integrated and then discarded, and its failure can still abort the run** — `src/sph/star_formation.cpp:461` · `recent_commits_b#6` · being edited  
  When a wind's decoupling clock expires inside a half-step in gas at or above the SH03 threshold, both the CPU and CUDA paths integrate chemistry for the decoupled portion and then discard the result. The projection overwrites energy and composition, and the heating ledger uses only the final energy. *Fix:* When project is true, do not submit chemistry: use chemistryInterval = project ? 0.0 : physicalInterval on CPU (star_formation.cpp:461) and in the CUDA kernel (star_formation_kernel.hpp:236).
- **Hydro-coupling flags are fetched with one blocking, uncached remote action per tree node** — `src/sph/star_formation.cpp:764` · `chemistry_star_formation#7` · being edited · also `recent_commits_b#8`  
  In multi-locality SF runs the hydro-coupling flag is the only staged SPH source field that skips the cache-line mechanism. Each unique remote neighbour node costs its own synchronous StarFormationFetchCouplingAction round trip per kSphRunHydro pass. *Fix:* Serve the flag through the existing cache-line machinery, e.g. a coupling cache like chemistryEosParticleCache, or pack it into the chemistry-EOS or density line entry. At minimum, batch the requests per remote rank.

### Core runtime

- **Pair declares defaulted copy ops, which suppresses its move constructor and move assignment, so every move deep-copies** — `cosmictiger/core/containers.hpp:154` · `core_runtime_infra#1`  
  Pair's explicitly defaulted copy constructor and copy assignment suppress the implicit move operations, so every std::move, vector growth, std::sort swap and future::get() of a Pair makes a deep copy. *Fix:* Delete the two defaulted copy members (lines 154-155) so the compiler generates all four copy and move operations. Keep `Pair() = default;`: because Pair(T, V) is user-declared, removing it would delete the default …
- **FixedCapacityVector (unused) traps when it holds n elements and lacks a leading barrier in pushBack/popBack** — `cosmictiger/core/fixed_capacity_vector.hpp:72` · `core_runtime_infra#13`  
  FixedCapacityVector is never instantiated, and some of its members would not even compile (begin(), end() and data() return an Array as T\*). Its capacity check traps at exactly n elements, so only n-1 are usable, and pushBack/popBack change the shared size without a leading block barrier. *Fix:* Delete the header, its four includes and its CMakeLists.txt entry. If it is ever revived: check before writing, allow size == n, add a leading \__syncthreads() to pushBack/popBack, return data\_.data(), and drop constexpr …
- **THROW_ERROR never throws: it aborts with SIGABRT, a core dump and a misleading 'Condition "false" failed in safe_io.hpp line 60', even for user configuration errors** — `cosmictiger/core/safe_io.hpp:21` · `core_runtime_infra#5`  
  THROW_ERROR prints the message and the caller's file and line, then calls ALWAYS_ASSERT(false) and abort(), so it never throws. The 93 user-input checks in options.cpp therefore end in SIGABRT with two symbolized stack traces, a stack_trace.txt entry that names only safe_io.hpp:60, and possibly a core file. *Fix:* Convert the options.cpp user-input validation sites to `throw std::invalid_argument(...)`, which main already reports cleanly.
- **ASPRINTF checks the wrong return value and expands to an unbraced if; safeFread/safeFwrite are static functions in a header** — `cosmictiger/core/safe_io.hpp:101` · `core_runtime_infra#16`  
  The ASPRINTF macro in safe_io.hpp is unused dead code with a latent bug. It tests `asprintf(...) == 0`, but asprintf returns -1 on failure, and it expands to an unbraced `if` that would capture a following `else`. *Fix:* Delete ASPRINTF, and the uncalled flushTimings in src/core/driver.cpp, whose raw asprintf ignores the -1 return. If an asprintf helper is ever needed, write it as 'do { if (asprintf(\__VA_ARGS_\_) &lt; 0) …
- **Semaphore (unused) lets transiently negative counts make tryWait fail spuriously, and waiters spin** — `cosmictiger/core/semaphore.hpp:24` · `core_runtime_infra#14`  
  The unused Semaphore class decrements before checking, so a pending large wait() makes concurrent tryWait() and wait() calls fail even when enough units are free, and its waiters busy-yield. *Fix:* Delete semaphore.hpp, its include in kick_workspace.hpp and its CMakeLists.txt entry. If a semaphore is ever needed, use hpx::counting_semaphore or std::counting_semaphore, or a CAS loop that decrements only when …
- **Timer uses non-steady high_resolution_clock, leaves startTime\_ uninitialized, and stop() is not idempotent** — `cosmictiger/core/timer.hpp:14` · `core_runtime_infra#17`  
  Timer measures intervals with high_resolution_clock, which libstdc++ defines as the non-steady system_clock, while section_timing and flops use steady_clock. A wall-clock step can therefore skew step timings, the stored runtime and the checkpoint-interval test, though shipped configs disable time-based checkpoints. *Fix:* Switch Timer to std::chrono::steady_clock. Add a `bool running_{false}` member: start() sets it, stop() accumulates only when it is set and then clears it, and `double read() const` returns time\_ plus the live interval …
- **UnorderedSetTs: every stripe shares one spinlock, insert() prints 'Inserting' per element, toVector() is unlocked (class is unused)** — `cosmictiger/core/unordered_set_ts.hpp:38` · `core_runtime_infra#12`  
  UnorderedSetTs is never instantiated. As written, its 1024 stripes share one spinlock, insert() prints and flushes "Inserting" for every element, and toVector() reads the buckets without locking. *Fix:* Delete cosmictiger/core/unordered_set_ts.hpp, its includes in src/main.cpp and cosmictiger/gravity/kick_workspace.hpp, and its CMakeLists.txt entry.
- **hpxHardwareConcurrency() is a hardware heuristic (PUs / SMT2 / ppnode), not the HPX worker count; it is used to size work everywhere** — `src/core/hpx.cpp:37` · `core_runtime_infra#8`  
  hpxHardwareConcurrency() returns hwloc's PU count divided by 2 (whenever SMT is active) and by an integer-truncated global ranks-per-node average. It ignores the real HPX worker count, yet about 50 loops size their work with it. *Fix:* Once the runtime is up, return int(hpx::get_os_thread_count()) (or get_num_worker_threads()) and drop the SMT and ppnode heuristics.
- **detectProcessesPerNode() is an O(N^2) all-to-all hostname gather at startup** — `src/core/hpx.cpp:140` · `core_runtime_infra#9`  
  detectProcessesPerNode() runs on every locality inside the init tree, after that locality's children finish, and asks all N localities for their hostnames. That makes O(N²) startup actions and O(N) per locality, and it only produces a truncated global ranks-per-node average. *Fix:* Compute the node-local rank count once. One option is MPI_Comm_split_type(MPI_COMM_TYPE_SHARED) before hpx::init. Another is to gather hostnames on the root over hpxChildren() and pass each locality its node's count …
- **interruptRequested() fans out from the root to every locality on every substep instead of using the reduction tree** — `src/core/interrupt_profile.cpp:262` · `core_runtime_infra#10`  
  With no local interrupt pending, interruptRequested() sends a query action from the root directly to every other locality on every substep. That is O(N) parcels at the root per substep, while the other global reductions use the hpxChildren() binary tree. *Fix:* Make CosmictigerInterruptQueryAction recurse over hpxChildren() and OR the results of each subtree, as readFlops does. Alternatively, fold the interrupt flag into a tree reduction that already runs every substep.
- **Memory daemon busy-waits on an HPX worker for the whole run (hundreds of millions of yields) and uses racy plain-bool flags** — `src/core/memused.cpp:66` · `core_runtime_infra#2`  
  memuseDaemon checks a 1 s timer inside a hpx::this_thread::yield() loop, so it stays permanently runnable. APEX recorded 206M yields and 105 s of daemon active time in a 352 s run, costing about 1-2% of sampled CPU and cluttering APEX task profiles. *Fix:* Sleep cooperatively with hpx::this_thread::sleep_for(std::chrono::seconds(1)), or use an hpx::condition_variable wait_for so stopping wakes it at once. Make stopDaemon a std::atomic&lt;bool&gt;.
- **Profile report lists every unresolved PC as its own line and re-walks dl_iterate_phdr per address** — `src/core/profiler.cpp:205` · `core_runtime_infra#7`  
  Unresolved samples are keyed by the exact PC, so each one gets its own report line. In the inspected report, 9% of samples, including all 3% of libucm, are scattered over about 1764 lines. *Fix:* Aggregate unresolved samples per module (for example `<unresolved> [libucm.so.0]`), optionally listing the top raw PCs separately. Take one module-table snapshot per report and binary-search it.
- **PROFILE()/profileRegisterFunction collects metadata that is never used and would silently auto-start the sampler** — `src/core/profiler.cpp:747` · `core_runtime_infra#11`  
  PROFILE() and profileRegisterFunction store function-signature metadata that nothing reads except a footer count, and no production code uses PROFILE(). Their only effects are attaching the calling thread and, implicitly, starting the sampler, with its signal handler, timers and atexit report file, when the sampler is … *Fix:* Delete PROFILE(), registerFunctionOnce, profileRegisterFunction and registrations\_, and adjust the profiler tests. If attaching extra threads is wanted, expose an explicit profileAttachCurrentThread() that never …
- **Default CPU-profile report goes to the launch directory as cosmictiger.profile.&lt;pid&gt;.txt, littering the tree and colliding across nodes and PID namespaces** — `src/core/profiler.cpp:1018` · `core_runtime_infra#6`  
  The default CPU-profile report name uses only the PID, relative to the launch directory, and is opened with "wb". Runs in separate PID namespaces, or ranks on different hosts sharing a working directory, can therefore silently overwrite each other's reports, contradicting the README's 'each process has its own file'. *Fix:* Make the default name unique, for example hostname, pid and start time, or open with O_EXCL and add a suffix on collision. Add a profile report-path setter, mirroring setStackTraceOutputPath, and call it after option …

### CUDA infrastructure

- **cuda.hpp carries dead and unsafe cuFFT helpers, duplicate declarations, and helpers that shadow CUDA runtime names** — `cosmictiger/cuda/cuda.hpp:49` · `cuda_infra#15`  
  cuda.hpp carries an unused cuFFT error path: &lt;cufft.h&gt; is pulled into every CUDA-enabled TU, the static enum formatter asserts on unhandled codes before printing, and cudaFftCheck resets the device and then continues under NDEBUG. *Fix:* Delete the cuFFT include, cudaGetErrorEnum, cudaFftCheck and CUDA_FFT_CHECK, the duplicate declarations, and the commented-out block in cuda.cpp.
- **Pool bookkeeping takes 512 MiB of managed memory per locality, most of it unreachable, yet small bins can still overflow** — `cosmictiger/cuda/cuda_memory.hpp:40` · `cuda_infra#11`  
  The pool's free-list rings are a fixed 32 x 2^20 x 16 B = 512 MiB of managed memory per CUDA locality, all written by the host at startup. About 320 MiB of it is capacity that can never be needed: bins 0-2 are unused, and with the fixed 8 GiB heap bins 13-31 can never hold 2^20 blocks. *Fix:* Give each bin a runtime-sized ring of min(2^20, nextPow2(heapSize/(2^b+8))) slots, carved from one allocation and reached through per-bin pointers, and drop bins 0-2.
- **Warp scans use threadIdx.x as the lane, and block reductions assume blockDim.x == blockSize without asserting it** — `cosmictiger/cuda/cuda_reduce.hpp:18` · `cuda_infra#13`  
  The warp scans use threadIdx.x as the lane index, and the block reductions and scan index a shared array of blockSize with threadIdx.x, without asserting blockDim.x == 32 or == blockSize respectively. All current launches satisfy those assumptions, so results are correct today. *Fix:* Use `int const lane = threadIdx.x % kWarpSize;` in the warp scans. Add a debug assert that blockDim.x == blockSize to the block variants.
- **freeHostAccessibleMemory holds the global allocator mutex across device-synchronizing cudaFree/cudaFreeHost** — `src/cuda/cuda.cpp:91` · `cuda_infra#14`  
  freeHostAccessibleMemory calls cudaFree/cudaFreeHost, which drain the device, while holding the single global std::mutex that every host-accessible allocation and free also takes. *Fix:* Find and erase the provenance entry under the lock, copy the pinned flag, release the lock, and then call cudaFreeHost, cudaFree or std::free. Consider an HPX-aware mutex (hpx::mutex or a spinlock) for the map.

### Particles, domains and drift

- **mesh_domain.hpp is used only by tests; the production rebound can create sliver domains** — `cosmictiger/particles/mesh_domain.hpp:73` · `particles_domain_drift#14`  
  mesh_domain.hpp, including its betterSplit tie-break for empty regions, is dead production code kept alive only by its own unit test since fc0d3e38. domainsRebound's own bisection collapses a zero-population domain into a box of width about 2^-21 plus a near-full sibling, though such domains essentially only occur … *Fix:* Delete mesh_domain.hpp, tests/mesh_domain.cpp and the CMake target. Optionally, in domainsRebound, fall back to the rank-proportional geometric plane when totalCount is 0, or when the count cannot move toward the target.
- **particlesOwnedSize() contract is false under Dehnen GPU staging** — `cosmictiger/particles/particles.hpp:202` · `particles_domain_drift#20`  
  particlesOwnedSize() simply returns particlesSize(), yet its header comment promises that ghosts never change the owned population. Dehnen GPU staging temporarily appends remote source particles to owned storage (gpu_staging.cpp:155, restored at :348), so during the GPU phase of a CUDA FMM kick the 'owned' count … *Fix:* Track an owned count that ghost appends do not change and return it from particlesOwnedSize(), which also makes the PMMM assertion meaningful.
- **Non-SPH migration copies whole parcels and compacts one particle at a time** — `src/particles/domain.cpp:368` · `particles_domain_drift#13`  
  The non-SPH receive action copies each parcel of up to about 1 GB instead of moving it. The sort comparator takes 120-136 B Particles by value. Net-outflow compaction in domainsEnd removes surplus slots one at a time on a single thread, calling particlesGetParticle, particlesSetParticle and particlesResize per … *Fix:* Use transParticles.push_back(std::move(parts)) and make the comparator take Particle const&amp;. Replace the surplus loop with one parallel pass that fills free slots below the new size from non-free tail particles …
- **Parcel limit counts particles, not bytes, giving ~1-2 GB SPH migration messages** — `src/particles/domain.cpp:535` · `particles_domain_drift#15` · plausible  
  SPH migration parcels are capped at 8M records rather than bytes, which gives roughly 0.8-2.0 GB serialized messages (2.1-2.2 GB in memory) per parcel depending on options. That is below the MPI int limit, but SF/chemistry parcels exceed HPX's default 1e9-byte max_message_size, whose enforcement on the MPI parcelport … *Fix:* Cap parcels by bytes, e.g. at about 256 MB divided by an estimated serialized record size computed once from the enabled options, separately for the SPH and non-SPH paths.
- **Domain lookups use unordered_map::operator[], and domain checkpoints dump raw structs** — `src/particles/domain.cpp:861` · `particles_domain_drift#18`  
  domainsRange and findParticleDomain look up boxesByKey with operator[]. A missing key would silently insert a zeroed DomainT (rank 0, empty box) and modify the map while other tasks are reading it. domainsSave and domainsLoad write std::pair&lt;size_t, DomainT&gt; and DomainLocal as raw memory without checking sizes. *Fix:* Replace operator[] with find() or at() plus ALWAYS_ASSERT in domainsRange and findParticleDomain, so both are genuinely read-only and fail loudly on a bad key.
- **drift's drag argument is always discarded** — `src/particles/drift.cpp:37` · `particles_domain_drift#9`  
  drift, cpuDrift and cudaDrift always overwrite their drag argument with a scenario-based constant (pi for glass, otherwise 0). The sqrt(glassError) value the driver passes through kickStepHierarchical and particleMeshDriftRung is therefore ignored, so the parameter is dead and the glass policy lives in two files. *Fix:* Remove drag from drift, cpuDrift, cudaDrift, particleMeshDriftRung and kickStepHierarchical, and compute the glass damping in one shared helper (e.g. in drift.hpp) used by both twins.
- **particlesFree and particlesDestroy leave stale size/capacity and dangling or leaked arrays** — `src/particles/particles.cpp:889` · `particles_domain_drift#6`  
  The teardown API is inconsistent and leaves no reusable state. particlesFree is local-only and leaves most array pointers dangling plus stale size/capacity. particlesDestroy runs on all localities but nulls X/V/R/P/accelerations without freeing and ignores Sph/Ty/Lgrp/Tr/size/capacity/rungBegin. *Fix:* Replace both with one idempotent teardown action that runs on every locality: run the release hooks, free and null every array, reset size, capacity, rungBegin and rungBegins, and clear the particle caches.
- **Stochastic realization depends on the number of localities: identities come from the initial rank decomposition** — `src/particles/particles.cpp:1074` · `chemistry_star_formation#14` · being edited  
  Star-formation identities, which key every conversion and wind draw, are assigned as rank offset plus local index from the post-IC per-locality layout. The same initial conditions and seed therefore give different stochastic SF realizations for different initial locality counts. *Fix:* Assign ids from a decomposition-independent key at IC generation (global lattice index plus species, or a hash of the Lagrangian position).
- **Unused particle helpers with latent bugs, and declarations with no definition** — `src/particles/particles.cpp:1246` · `particles_domain_drift#12`  
  particles.cpp and domain.hpp carry several uncalled helpers (particlesEnclosingSphere, particlesActivePct, particlesDisplace, particlesGetSample, particlesSaveGlass, particlesGlobalReadVels/Rungs with their caches), three declared-but-undefined functions (particlesSortBySph, particlesGroupCacheFree … *Fix:* Delete the unused functions and their actions and caches, the declared-but-undefined prototypes, and the commented-out iterator and Ewald code.

### I/O, checkpoints and tools

- **Lightcone compression module is unused in production, and compressionRatio() miscounts bytes** — `src/analysis/compress.cpp:6` · `io_checkpoint_output_tools#11`  
  The lightcone compression module is compiled into production but unused there (only tests call it, and lightcone.cpp's groupArcs is dead). Its uncalled compressionRatio() undercounts the serialized bytes (6n instead of 12n per block, and a 36- instead of 44-byte header), overstating compression by about 2x. *Fix:* Delete compress.\*, the include, groupArcs and the test, or start using it for lightcone output. If it stays, compute bytes from sizeof expressions: 44 + sum(33 + 4\*kDimensions\*n), and test the result against the actual …
- **checkpoint.txt is read from the working directory as floats, stops at the first bad line, and aliases use %.1f** — `src/io/checkpoint.cpp:83` · `io_checkpoint_output_tools#13`  
  checkpoint.txt is read from the working directory as floats, parsing stops silently at the first non-numeric line, and redshift aliases use a lossy %.1f name, so nearby requests collide or round unexpectedly (0.15 becomes checkpoint.z.0.2). *Fix:* Take the list path from an option, or resolve it relative to the configuration or output.directory. Parse each line with strtod, skipping blank and comment lines and warning on bad ones.
- **Checkpoint read never checks that the payload was fully consumed** — `src/io/checkpoint.cpp:1452` · `io_checkpoint_output_tools#12`  
  readCheckpoint never verifies that it consumed the whole rank file, so a future writer/reader layout asymmetry that leaves trailing bytes would go undetected even though the manifest checksum passes. *Fix:* After the last load, require `fgetc(file) == EOF`, or compare ftell with the file size, and THROW_ERROR otherwise. Add the same assertion to the checkpointStateTest particle round trip.
- **Silo view output copies every particle twice with a 72-byte StarFormationState each, and recomputes frame-independent diagnostics per frame** — `src/io/view.cpp:69` · `io_checkpoint_output_tools#10`  
  Silo particle output materialises two full copies of all local particles (about 104-112 B each, including an unused 72-byte StarFormationState) plus per-field double buffers, and recomputes frame-independent chemistry and SH03 diagnostics for each unit/frame directory. *Fix:* Select particle indices per box and read fields through them instead of copying records. Store StarFormationState separately and only when star formation is enabled, and reserve the vectors.
- **Rewriting a Silo particle cycle leaves the previous generation directory behind (disk leak on every restart)** — `src/io/view.cpp:901` · `io_checkpoint_output_tools#5`  
  Republishing a Silo particle cycle, which happens on restarts from coarse-boundary or final checkpoints, leaves the previous generation\_\* leaf directory on disk with nothing referencing it. This is a deliberate, tested retention policy with no cleanup, so orphaned snapshots accumulate silently, one per republish. *Fix:* Either document the retention and provide a pruning utility, or add a GC policy. For example, after the master rename, rank 0 removes generation\_\* siblings other than the one just published (optionally after a grace …
- **views2silo.sh is a one-off script with a hard-coded index range and paths** — `views2silo.sh:1` · `io_checkpoint_output_tools#16`  
  views2silo.sh is a tracked, unreferenced one-off with no shebang. It loops over a hard-coded `{100..142}` (set in 2021 for one run) and runs `./healpix2silo` on `view.$i.dat` in the current directory. *Fix:* Delete the script; the README already shows how to run healpix2silo directly. If a batch helper is wanted, replace it with a `#!/usr/bin/env bash` script using `set -euo pipefail`.

### Gravity: global FMM (dehnen)

- **Ewald table order is recorded but never checked by the accessors** — `cosmictiger/gravity/dehnen/ewald_indices.hpp:63` · `gravity_dehnen_global_fmm#9`  
  The Ewald tables record the FMM order they were built for, and a comment says the accessors are checked against it. No production accessor or caller compares constants().order with the runtime order; only a unit test does. A future runtime order change would silently read mismatched rows. *Fix:* Add a host-side ALWAYS_ASSERT(EwaldConst::constants().order == fmmOrder()) at kick entry (in kick() or dehnenPeriodic), or refill the tables in setFmmOrder once they exist. Otherwise correct the comment.
- **Ewald splitting parameter alpha = 2 is scattered as unrelated magic numbers** — `cosmictiger/gravity/dehnen/ewald_tables.hpp:68` · `gravity_dehnen_global_fmm#10`  
  The Ewald splitting parameter alpha = 2 is encoded independently in the table code (a local kAlpha plus a literal 4.0 for alpha^2) and as unrelated literals in the generator and its output. The image cutoffs are implicitly tuned to it as well. *Fix:* Define a single kEwaldAlpha (for example in defs.hpp, or emitted by tools/generate_fmm.py into the generated header). Have the generator emit expressions in it, and use it in ewaldFillTables for both c0 and d0.
- **Ewald image lists are duplicated between CPU and CUDA builds, with hard-coded unchecked counts and a debug print** — `src/gravity/dehnen/ewald_indices.cu:44` · `gravity_dehnen_global_fmm#6`  
  The Ewald real and Fourier image lists are generated by two verbatim copies: EwaldConst::init and EwaldConst::initGpu. Both write into arrays sized by the literals 178 and 92 with no runtime count check, and the CUDA copy keeps a leftover verbose debug print. Only the CPU-only copy is unit-tested. *Fix:* Generate both lists in one shared host function called by both init paths. Add ALWAYS_ASSERT(count == kEwaldRealIndexCount) and ALWAYS_ASSERT(count == kEwaldFourierIndexCount), or compute the sizes constexpr from the …
- **kickFork reads a remote node only to get processRange.first, which always equals self.proc; one thread-cap clause is unreachable** — `src/gravity/dehnen/execution.cpp:151` · `gravity_dehnen_global_fmm#12`  
  For a remote child, kickFork reads its tree node only to obtain processRange.first, which always equals self.proc; the round trip is actually paid by kickWalk's prefetch just before the fork. The `|| !selfPtr->isLocal()` thread-cap clause can never affect the outcome. *Fix:* Post remote forks to hpxLocalities()[self.proc], call treeGetNode(self) only in the local branch, and delete the dead clause (or implement the intended multi-rank threading policy).
- **kick() stores to a shared seq_cst atomic and rebuilds the context on every node visit** — `src/gravity/dehnen/execution.cpp:181` · `gravity_dehnen_global_fmm#13`  
  kick() performs a redundant sequentially consistent store of the same function pointer to a global atomic on every visited tree node, although treeAllocateNodes already registers that reader once per tree build on every locality. It also rebuilds an identical KickContext each time. *Fix:* Remove the gravitySetRemoteSourceReader call from kick() and rely on the registration in treeAllocateNodes (or keep it only at the root or KickAction entry). Build the KickContext once as a function-local static const.
- **Vestigial completion and cleanup hooks: kickClearBuffers clears nothing, kickCoverParticles affects nothing** — `src/gravity/dehnen/execution.cpp:207` · `gravity_dehnen_global_fmm#14`  
  kickClearBuffers only fans a no-op action out to the child localities and clears nothing. It is a leftover of an already-empty KickWorkspace::clearBuffers, yet driver.cpp, scenario_driver.cpp and tests/test.cpp still call it. The kickCoverParticles/KickContext::coverParticles hook runs only in non-batch GPU mode. *Fix:* Delete kickClearBuffers, ClearBuffersAction, its header declaration and the four call sites. If an end-of-run locality barrier is actually wanted, replace it with an explicitly named barrier.
- **Dead code and debug leftovers in gpu_staging.cpp** — `src/gravity/dehnen/gpu_staging.cpp:30` · `gravity_dehnen_global_fmm#16`  
  gpu_staging.cpp has several unused leftovers: - an externally linked mortonCompare with no callers; - four Timers whose readings appear only in commented-out PRINTs, one of them after the return; - unused locals `count` and `index` in the request-chunk split; *Fix:* Delete mortonCompare, the four timers (and the timer.hpp include if nothing else needs it), the commented PRINTs, the unused `count`/`index` locals and the unused captures.
- **dehnenStageGpu ignores the FmmForceOutput argument and always publishes the non-batch local return** — `src/gravity/dehnen/gpu_staging.cpp:54` · `gravity_dehnen_global_fmm#11`  
  dehnenStageGpu has the shared KickGpuExecutor signature but ignores its FmmForceOutput argument and always publishes the global local-return promise. It is correct only for the non-batch, non-forceOnly workspaces that the dehnen walk creates today, and nothing asserts that. *Fix:* Add ALWAYS_ASSERT(!params.forceOnly), plus an assert that the output buffers are null, at entry. Alternatively, forward the output to cudaExecuteKicks and skip kickSetRc when params.forceOnly is set.
- **GPU staging walks every local source subtree and builds a hash map that is never used for local nodes** — `src/gravity/dehnen/gpu_staging.cpp:139` · `gravity_dehnen_global_fmm#5`  
  dehnenStageGpu recursively collects and hash-maps the whole subtree of every local checklist id, but only remote entries are ever consulted afterwards. On a single locality, where staging runs on one thread and nothing is remote, this whole pre-launch phase is overhead paid on every GPU kick. *Fix:* In collectSources, skip ids with `ids[i].proc == hpxRank()`: no addTreeNode, no insert. When no remote id exists (always true on one locality), go straight to the sort, tree2Gpu and cudaExecuteKicks.
- **Concurrent std::unordered_map::operator[] on treeMap/particleMap from parallel staging tasks** — `src/gravity/dehnen/gpu_staging.cpp:223` · `gravity_dehnen_global_fmm#4`  
  Concurrent HPX staging tasks look up the completed treeMap and particleMap with unordered_map::operator[] (lines 223, 263-264, 291, 297). The standard does not count that call as const for data-race purposes, so this is formally outside its guarantees. *Fix:* Use find() or at() in these read-only task bodies, e.g. `auto const it = treeMap.find(left); ALWAYS_ASSERT(it != treeMap.end()); left.index = it->second;`.

### Gravity: PMMM (nitadori)

- **Far-field evaluation makes one blocking remote round trip per sink cell** — `src/gravity/nitadori/cell_schedule.cpp:116` · `gravity_nitadori_pmmm#8`  
  Far-field local expansions are fetched with one synchronous remote round trip per sink cell, in sequence. This holds in both the FMM batch loop and, per cell, in the PP path, so batch latency grows linearly with the number of sink cells whose slabs live on other localities. *Fix:* Let readCells take a list of expansion keys. In each batch, group the sink keys by slab owner, issue one asynchronous request per owner, and wait once before evaluation.
- **Frozen stage rungs are snapshotted, shipped, stored and swapped for ghosts, but nothing ever reads them** — `src/gravity/nitadori/cell_schedule.cpp:172` · `gravity_nitadori_pmmm#9`  
  The stage-rung snapshot is computed, serialized in every ghost record, stored in the ghost arena's rung plane and swapped during partitioning, but no near-field code (CPU or CUDA) ever reads ghost rungs. *Fix:* Remove stageRungs, ParticleGhost::rung, the ghost rung plane and the rung swap in fmmSourcePartition (or leave ghost view.rungs null). Fix the misleading comment in readGhostParticles.
- **The periodic-background trace (4π/3)·h² is spelled out separately in three places** — `src/gravity/nitadori/evaluation.cpp:38` · `gravity_nitadori_pmmm#14`  
  The periodic-background trace factor (4π/3)·h² is written out separately in evaluateParticles, computeReadyCells and readCells, even though the generator already outputs it as EwaldGreen's greenTrace, which green() discards. *Fix:* Add one helper in the pmmm detail namespace, e.g. `double backgroundTraceFactor(int n)` returning kGreenTrace/(n\*n), where kGreenTrace comes from the generator.
- **transform() copies all local moments into the FFT buffer in a single-threaded loop** — `src/gravity/nitadori/fft.cpp:392` · `gravity_nitadori_pmmm#7`  
  transform() zero-fills the FFT input and copies, checks and sums all local moments in one serial loop on a single HPX thread before the forward FFT. That loop is on the far-field critical path, and in PP mode the other workers sit idle during it. *Fix:* Use parallelFor over (x_local, y) rows for both the fill and the copy, keep per-task partial background sums, and reduce them in a fixed order for reproducibility.
- **Dead or test-only code in production: kickCellForest, kVerbose, the compatibility forest builder** — `src/gravity/nitadori/fmm_batch.cpp:177` · `gravity_nitadori_pmmm#10`  
  kickCellForest and cell_tree.cpp's kVerbose are dead code. Production treeCreateCell also inserts every root into cellRootReturns under a global mutex, which only the test-only treeCombineCellRoots path reads. *Fix:* Delete kickCellForest, its declaration, and the unused kVerbose. Move the cellRootReturns insert into the compatibility path (for example, record roots in treeCreateCellForest), or gate it behind a test-only switch.
- **An invalid ghost-owner count is reported as an index-range overflow** — `src/gravity/nitadori/ghost_exchange.cpp:90` · `gravity_nitadori_pmmm#13`  
  planGhosts throws the same "PMMM ghost count exceeds its index range" message for a non-positive owner count, which by construction means a corrupt directory entry, and for real ParticleInteger overflow. Neither message names the key or the rank. *Fix:* Split the condition. For `owner.count <= 0`, throw something like "PMMM owner directory has a non-positive count for cell &lt;key&gt; from rank &lt;rank&gt;".
- **Avoidable per-particle and per-cell overhead in hot loops** — `src/gravity/nitadori/mesh_exchange.cpp:28` · `gravity_nitadori_pmmm#15`  
  The PMMM hot loops carry small avoidable per-item overheads. These are a per-cell `pmmmNearMethod == "fmm"` string compare in deposit(), an out-of-line getOptions() call per particle in consumeField and particleMeshApplyField, and out-of-line bounds-checked pmmmAcceleration/pmmmPotential calls per particle in the … *Fix:* - Cache the near method (as a bool) and gm once per solve, in SolveState at prepare() or as locals hoisted out of the loops.
- **PP near field re-fetches and copies every neighbor cell's particles three times per sink cell, and keeps the source copies until the next solve** — `src/gravity/nitadori/mesh_exchange.cpp:92` · `gravity_nitadori_pmmm#6`  
  The PMMM-PP near field fetches and copies each neighbor cell's source records three times for every sink cell that needs them, and remote cells cross the network once per requesting sink cell. Deposit appends sources under one global mutex, and state.sources is never released after a PP solve. *Fix:* Cache neighbor cells by key within a batch, or fetch each remote cell once per locality (as the FMM ghost plan does). Let local reads reference state.sources without copying, and append directly into batchSources with …

### Gravity: tree, walk and kick

- **Spherical operator wrappers return the old weighted flop counts instead of the conventional ones** — `cosmictiger/gravity/fmm_kernels/spherical_operators.hpp:19` · `gravity_cuda_kick#12`  
  The spherical operator wrappers return the generator's weighted count (`kFlops_`: FMA=1, sqrt=7). The Cartesian wrappers return a `kFlops_` that is the conventional FMA=2 count, so the same field name means different conventions in the two families. *Fix:* Make every spherical wrapper, including the Ewald and trace overloads, return kConventionalFlops\_, and recount its hand-added increments in the same FMA=2 convention.
- **kick.hpp misstates the leaf occupancy that the CUDA arena is tuned for** — `cosmictiger/gravity/kick.hpp:23` · `gravity_tree_walk_cpu#15`  
  kick.hpp:22-23 and docs/cuda_kick_shared_memory.md:72-73 say leaves hold between bucketSize/2+1 and bucketSize particles. In fact treeBuild splits at the geometric midpoint and the dehnen Ewald criterion forces further splits, so leaves hold anywhere from 0 (empty, zero-radius leaves) to bucketSize particles. *Fix:* In both places, say that leaves hold 0..bucketSize particles, that the Ewald criterion splits sparse cells down to single particles with empty siblings, and that the sink tile is sized from the maximum, bucketSize.
- **Hierarchical (htime) branch of doKick has drifted from its CPU twin** — `cosmictiger/gravity/kick_cuda_kernel.hpp:187` · `gravity_cuda_kick#5` · also `gravity_tree_walk_cpu#8`  
  In the hierarchical (htime) branch, doKick accumulates kinetic energy and momenta without mass (187-191) but potential and dkin with mass (130, 285, 289), while the CPU twin is massless throughout. Hierarchical integration is rejected whenever hydro is enabled, so mass is always 1 and the diagnostics agree today. *Fix:* Pick one convention for the htime diagnostics (mass-weighted, like the ordinary branch) and apply it in both twins. Write 0.5f at kick_cuda_kernel.hpp:285 and at kick_walk.hpp:473/509.
- **Empty leaves still pay a full warp L2L translation on the GPU** — `cosmictiger/gravity/kick_cuda_kernel.hpp:486` · `gravity_cuda_kick#10`  
  In case 0 the CUDA walk does the cooperative L2LWarpInto, republishes the expansion and charges L2L flops before reading numberOfSinks. Empty leaves, produced by periodic midpoint splits and the Ewald extra subdivision, therefore pay a full warp L2L (and, for a left child, the parent's checklist pushTop) before being … *Fix:* Compute numberOfSinks first, and when it is zero, `sparams.popBack(); depth--;` immediately, skipping the L2L and its flop charge.
- **Non-default kick layouts and multi-tile sinks have no automated test; the doc's validation scripts were deleted** — `cosmictiger/gravity/kick_cuda_kernel.hpp:629` · `gravity_cuda_kick#13`  
  No automated test reaches the CUDA kick's global force backing (sinkTile=-1), global active expansion (sharedExpansion=false), pool-only lists (listReserve=0), explicit residency, or multi-tile sink loop. *Fix:* kickArenaLayoutImpl caches the layout once per process, so changing options inside a running test will not switch layouts. Add ctest variants of the existing fmm_walk_cuda command instead …
- **Dead state, unused API and a stale comment in the CUDA kick** — `cosmictiger/gravity/kick_cuda_launch.hpp:232` · `gravity_cuda_kick#11`  
  The comment at kick_cuda_launch.hpp:232-235 says the CPU/GPU handoff divides by the launch residency. That stopped being true in commit 227090b0, the same commit that added the comment: kickBlockCount now uses a fixed 2\*SMs\*cbrt rule. *Fix:* Change the comment to say the launch and the per-launch list slab are sized by residency times the SM count, and that the handoff divisor is independent of it (see kickBlockCount). Delete 'Timer tm;
- **Arena layout is cached for the process lifetime and ignores later option changes** — `cosmictiger/gravity/kick_cuda_launch.hpp:272` · `gravity_cuda_kick#9`  
  The CUDA kick arena layout is a process-lifetime static built from the options at the first launch, so later changes are ignored, and a later hydro toggle would abort on ALWAYS_ASSERT(layout.massPlane == data.sph). Hydro never changes mid-process in-tree. *Fix:* Key the cached layout and list slab on (massPlane, min(bucketSize, kMaxBucketSize), the cudaKick\* options), and re-select when a key changes.
- **CUDA kick layout selection runs device-specific CUDA calls before the launch code selects its device** — `cosmictiger/gravity/kick_cuda_launch.hpp:283` · `gravity_cuda_kick#1`  
  cudaExecuteKicksImpl builds the per-process arena layout (line 283) before selecting the device (cudaSmpCount at 284, cudaSetDevice at 428). kickSelectLayout's function-attribute, MaxDynamicSharedMemorySize and occupancy calls therefore use the calling thread's current device, while kickDeviceResources queries the … *Fix:* Make cudaSetDevice() the first statement of cudaExecuteKicksImpl, or put it at the top of kickSelectLayout. After selecting, query every attribute against the same device: either the thread's current device …
- **kickWalk: 'threadLeft' controls the right child; always-true checks and per-particle invariant recomputation** — `cosmictiger/gravity/kick_walk.hpp:185` · `gravity_tree_walk_cpu#17`  
  In kickWalk, `threadLeft` actually controls asynchronous forking of the RIGHT child; the left child always runs inline. It is also effectively always true wherever it can matter. The numberOfParticles() checks at lines 171 and 371 can never be false. *Fix:* - Rename the flag to forkRightChild or threadSibling, or remove it and pass true, since it differs only for multi-rank nodes whose remote right child kickFork already runs asynchronously.
- **Global FMM always evaluates accepted leaf-leaf pairs with M2L, even 1x1-particle leaves** — `cosmictiger/gravity/kick_walk.hpp:327` · `gravity_tree_walk_cpu#11` · plausible  
  The global FMM never uses the flop-based PP preference for accepted leaf-leaf pairs; the PMMM path does. Small accepted leaf pairs therefore always cost a full M2L, even when the code's own cost model says direct PP is cheaper. The behavior is documented as intentionally unchanged, and the payoff is unmeasured. *Fix:* Measure first with the walk's interaction counters. If worthwhile, pass kCellCell\_ for meshSize==0 in both kick_walk.hpp and kick_cuda_kernel.hpp together, update the flop accounting and walk fixtures, and update the …
- **Dead fixedRung override left in FMM CPU and CUDA kicks** — `cosmictiger/gravity/kick_walk.hpp:551` · `recent_commits_a#8`  
  No code sets KickParameters::fixedRung for FMM kicks since 844a55c0. Only the PMMM regression uses it, through particleMeshApplyField. So the FMM CPU and CUDA fixed-rung branches are dead, and the CPU comment cites a driver safety check that was removed. *Fix:* Remove the fixedRung branches from kick_walk.hpp and kick_cuda_kernel.hpp along with the stale comment. Keep fixedRung documented as PMMM-test-only, or move it into a PMMM-specific parameter.
- **CUDA and CPU kick flop constants disagree with each other and with flop_accounting.md** — `cosmictiger/gravity/kick_walk.hpp:575` · `gravity_cuda_kick#4`  
  The CPU kick charges L2P+25 per active particle (kick_walk.hpp:575), though its loop body counts only about 19 operations; the CUDA twin's L2P+18 (kick_cuda_kernel.hpp:294) matches its source. Identical CPU and CUDA kicks therefore report different operation counts. *Fix:* Set the CPU per-particle charge to L2P+18. Either hoist the per-particle gravityEffectiveRadius division out of the loop, as the GPU does, or charge +1 for it. Update flop_accounting.md:83 to 24.
- **Stale docs and comments around the tree build and flop counts** — `docs/open_larger_cell_walk.md:280` · `gravity_tree_walk_cpu#16`  
  Several references are stale. docs/open_larger_cell_walk.md:280 puts the Ewald leaf criterion in tree_build.cpp, but it lives in dehnen/distributed_tree.cpp:83-86, and the doc omits its `boxR < 0.125 - 0.25*supportRadius` term. *Fix:* - Point the walk doc at dehnen/distributed_tree.cpp and state both conditions on the cell box half-diagonal, noting that this bounds the leaf radius.
- **Particle-mass rule copied in at least five places** — `src/gravity/gravity.cpp:72` · `gravity_tree_walk_cpu#14`  
  The particle-mass rule (1 without hydro; otherwise DM mass for dark matter and gas mass for every other type, stars included) is duplicated across seven sites on host and device, plus the implicit equalMass shortcut. *Fix:* Add one CUDA_EXPORT helper keyed on the particle type, e.g. `gravityMassOfType(char type, bool hydro, float darkMatterMass, float gasMass)`.
- **Direct PP kernel re-gathers source SIMD lanes for every sink particle and heap-allocates per chunk** — `src/gravity/gravity.cpp:97` · `gravity_tree_walk_cpu#10`  
  The CPU direct particle-particle kernel re-gathers every source SIMD packet for each sink particle. It also heap-allocates five vectors per 32-leaf chunk in the hottest CPU kick loop, and carries an unused Timer and commented-out feenableexcept lines. *Fix:* Pack each chunk once into thread_local SimdAlignedVector&lt;Array&lt;SimdPosition,3&gt;&gt; and SimdFloat weight buffers, as gatherSourceMultipoles does, then loop sinks over the packed packets.
- **Global FMM kick buffers: managed realloc whenever local count changes, serial opening loop, discarded reduction, never freed** — `src/gravity/kick.cpp:75` · `recent_commits_a#4`  
  The global-FMM kick frees and reallocates three managed acceleration arrays whenever the local particle count changes, which is common after domain exchange. Under CUDA that means cudaFree plus cudaMallocManaged with an implicit device sync. *Fix:* Keep a grow-only capacity, split the finish loop across workers like particleMeshFinishKick, and either use the returned kinetic energy in place of particlesSumEnergies().kin or make both finish functions void.
- **TreeCreateParameters silently ignores its support-radius argument; walk tests pass supports that have no effect** — `src/gravity/tree_arena.cpp:162` · `tests_gravity_cpp#9`  
  TreeCreateParameters silently ignores its third (hmax) argument; leaf support always comes from each locality's getOptions().gravity.supportRadius. Callers pass contradictory values (0.f in the driver, 0.015625 in fmm_walk fixtures whose options say 0.005), so call sites imply a support that is never applied. *Fix:* Remove the third constructor parameter and update all callers: driver.cpp, cell_tree.cpp, and the tests in fmm_walk.cpp, fmm_tree.cpp, sph/gravity.cpp, linear_growth.cpp and test.cpp.
- **tree2Gpu advises only the node array; the multipole arena the kernel reads through mpos never gets ReadMostly/prefetch** — `src/gravity/tree_arena.cpp:211` · `gravity_tree_walk_cpu#6` · plausible  
  Only the node array gets cudaMemAdviseSetReadMostly. The positioned-multipole arena the CUDA kick stages through node.mpos gets no advice or prefetch, and the PMMM batch GPU path calls neither tree2Gpu nor tree2Cpu. Whether this costs measurable page-fault or thrashing time is unverified. *Fix:* Profile first. If faults or thrashing show up, apply the same advice (and optionally a prefetch of the used prefix) to multis over nodesSize\*multipoleStride.
- **nodesSize doubles as arena capacity, so the managed tree arena is needlessly reallocated whenever N_local grows back** — `src/gravity/tree_arena.cpp:243` · `gravity_tree_walk_cpu#3`  
  treeAllocateNodesLocal uses nodesSize as both the logical limit and the allocation size. Once a build requests fewer nodes, the next slightly larger request frees and reallocates both managed arenas (nodes and multis) even though they are big enough. *Fix:* Add a static nodesCapacity (in elements) next to multipoleStride. Reallocate only when sz &gt; nodesCapacity or the stride changes, optionally growing about 1.25x.
- **Dead tree API: nine undefined declarations, unused statics, and a divide-by-zero in treeGetLeafSizes** — `src/gravity/tree_arena.cpp:336` · `gravity_tree_walk_cpu#13`  
  tree.hpp declares nine functions that are defined nowhere. tree_arena.cpp keeps four unused statics plus an exported but uncalled treeGetLeafSizes/treeAvgLeafSize pair, with a registered HPX action. If it were called, it would divide 0.0 by 0 on the never-filled leafSizes and convert the NaN to int, which is UB. *Fix:* Delete the nine declarations, treeGetLeafSizes, TreeGetLeafSizesAction, treeAvgLeafSize, leafSizes, leafSizesMutex, leafParticleRanges, leafParticleRangeMutex, treeThreadCount, both kVerbose constants, the `global` …
- **treeDestroy(true) ignores freeTree and never releases the arena; treeReset/treeDestroy swallow remote errors** — `src/gravity/tree_arena.cpp:360` · `gravity_tree_walk_cpu#4`  
  treeDestroy(bool freeTree) ignores freeTree on every locality and does exactly what treeReset() does, so treeDestroy(true) never releases the tree arena, despite the name and the sphTreeDestroy(true) precedent. Child-locality errors are not swallowed, because HPX 1.11 wait_all rethrows. *Fix:* Either implement freeTree (free nodes/multis, zero nodesSize/multipoleStride/capacity, clear populatedNodes, reset setGpu/setCpu) or remove the parameter. Merge treeReset and treeDestroy into one function.

### SPH

- **viscosityMu default arguments silently encode the cubic-only h = H/2 convention** — `cosmictiger/sph/monaghan/viscosity.hpp:47` · `sph_physics_kernels#6`  
  viscosityMu's default arguments reproduce the cubic-spline h = H/2 convention whatever kernel is selected. A future production caller that omits the last two arguments compiles cleanly and gets a viscosity length and η² regularisation about 6% (WC2) to 21% (WC4) too large. *Fix:* Remove the defaults and have equations.cpp pass `Real(2), Real(1)` explicitly, or add a separately named cubic reference overload.
- **Dead and misleading code in the SPH infrastructure** — `cosmictiger/sph/sph_particles.hpp:182` · `sph_infrastructure#15`  
  The SPH infrastructure has dead code: - sphParticlesPressure has no callers, and in cosmological runs it would return a^2(gamma-1)rho_c u in stored units. - SphTreeCreateParameters(int, float) is declared but never defined. *Fix:* Delete sphParticlesPressure, or convert it to physical units and document that. Delete the undefined constructor declaration, the unused statics and locals, and the commented-out blocks.
- **The neighbour walk fetches a remote tree cache line only to find the owner, which it already knows** — `src/sph/sph.cpp:89` · `sph_infrastructure#12`  
  For remote children, sphTreeNeighborFork blocks on a tree-cache fetch just to read processRange.first, which always equals self.proc. That adds a round trip per global level to every walk, and the fetch is wasted outright in Boxes walks. *Fix:* In the remote branch dispatch to hpxLocalities()[self.proc], and call sphTreeGetNode(self) only on the local path.
- **Every sphRun re-allocates about 20 pinned staging buffers** — `src/sph/sph.cpp:332` · `sph_infrastructure#9`  
  Every sphRun call (per locality, roughly 5-15 per step) creates a new workspace whose pinned staging vectors are allocated with cudaHostAlloc and freed with cudaFreeHost. hostTrees and neighbour indices also regrow through pinned reallocations. The reserve(15) for up to 23 device buffers is stale but harmless. *Fix:* Keep a persistent per-locality staging workspace whose pinned capacity only grows: clear() but don't free between calls, and reserve hostTrees and hostNeighborNodeIndices from the previous call's sizes.
- **isConverged requires every particle in a leaf to be converged, but the kernels only mark the particles they solve** — `src/sph/sph.cpp:357` · `sph_infrastructure#11`  
  The leaf-level convergence test demands every particle's flag, but prehydro1 flags only active sinks and prehydro2 only semiactive ones. Mixed leaves are therefore re-staged and re-gathered on every extra iteration, and in prehydro2 their converged semiactive sinks are fully recomputed. *Fix:* Base the leaf predicate on the sinks that pass actually solves (active and unconverged for prehydro1; semiactive and unconverged, or not yet classified, for prehydro2), or mark untouched sinks converged in the kernels.
- **Redundant PMMM effective-radius override in SPH run setup** — `src/sph/sph.cpp:596` · `recent_commits_a#10`  
  The PMMM-gated override of gravityEffectiveRadius in SPH setup assigns the same expression that was just computed, because pmmmEffectiveRadius() is that exact formula. The branch and its conditional include are redundant. *Fix:* Drop the branch and the conditional solver.hpp include, or have SPH, the PMMM kick and the FMM path share one gravityEffectiveRadius() helper.
- **Per-particle std::string comparisons in hot loops, and a full Options copy in SphRunParameters** — `src/sph/sph_particles.cpp:148` · `sph_infrastructure#16`  
  sphParticlesApplyUpdates evaluates `getOptions().hydro.viscosityEvolution == "morrisAndMonaghan"` for every gas particle in each kick phase, before the cheap rung test. sphParticlesFetchDensityCacheLine evaluates `!= "none"` for each of about 11k entries per served line. *Fix:* Hoist the option tests out of the loops (`bool const morrisAndMonaghan = ...; bool const hasAlpha = ...;`), or store an enum or flags in HydroOptions once at option parsing.
- **1% capacity growth combined with one-particle-at-a-time resizes causes very many full reallocations** — `src/sph/sph_particles.cpp:333` · `sph_infrastructure#10` · also `scenarios_configs#17`  
  With 1% capacity growth (+1 element per growth below 200), the scenarios that append gas one particle at a time trigger roughly 100\*ln(N/100) full reallocations of every SPH array. Each one is a managed alloc plus a parallel copy, costing about 100 copies per element at startup. Cosmology ICs are unaffected. *Fix:* Have the incremental scenarios size the gas arrays once from their analytic particle count, or add a sphParticlesReserve API.
- **Kernel value and derivative evaluated twice per neighbour in the density pass** — `src/sph/sph_prehydro.cu:310` · `sph_physics_kernels#8`  
  The density pass evaluates the kernel derivative twice per in-support neighbour (Σqdw and the Balsara/MM velocity-gradient weight), and with CD also the kernel value twice (density and the R sign sum), on CPU SIMD and CUDA. The flop counters count the duplicate evaluations. *Fix:* Compute `dw_ab = dkernelWDq(q_ab)` once per neighbour (masked to q&lt;1), reuse it and w_ab for Σqdw, m_b_F_a and the ρ_a R_a sum, and drop the matching densityKernelFlops terms.
- **CUDA density pass runs up to 36 separate block reductions per sink, each with ~15 barriers** — `src/sph/sph_prehydro.cu:383` · `sph_physics_kernels#7`  
  The CUDA prehydro2 pass reduces each per-sink quantity with its own 15-barrier shared-memory all-reduce: 7 per sink by default, 34-36 with Price/Cullen-Dehnen, about 105-540 \__syncthreads per semiactive sink. Only thread 0 consumes the results. *Fix:* Gather the partial sums into one register array. Warp-reduce each entry with \__shfl_xor_sync, have lane 0 of each warp write to shared memory, then one \__syncthreads, and let thread 0 (or warp 0) finish, since nothing …
- **CPU support solve heap-allocates a vector for every particle solve** — `src/sph/sph_prehydro_cpu.cpp:48` · `sph_physics_kernels#9`  
  The CPU support-radius solve heap-allocates its per-particle distance cache on every call, i.e. once per active or unconverged particle per density pass. A buffer reused per leaf would do. This is a minor inefficiency. *Fix:* Allocate one `std::vector<SimdFloat>` scratch per leaf in sphCpuPrehydro1/2 (or thread_local), pass it by reference, and clear() and refill it per particle.
- **Support-solve target duplicated inline although sphTargetKernelSum exists** — `src/sph/sph_prehydro_cpu.cpp:58` · `sph_physics_kernels#4`  
  Both support-radius Newton solvers inline the 3N/(4π) target instead of calling sphTargetKernelSum. The free/held classification and zeta in sphGradHEvaluation use that helper, so they stay consistent with the solvers only while three copies stay identical. *Fix:* Replace the inline expressions in solveSphCpuSupportRadius and solveSphSupportRadius with `sphTargetKernelSum(targetNeighborCount)`.
- **sqrt(5/6) viscosity-length calibration repeated as an unnamed literal and duplicated formula in 6+ places** — `src/sph/sph_prehydro_cpu.cpp:385` · `sph_physics_kernels#3`  
  The sqrt(5/6) viscosity-length factor is a bare literal in four prehydro sites (CPU and CUDA, MM and Price branches). The hydro CPU and CUDA files each define it, and the 0.7116 acceleration calibration, as function-local constexprs. *Fix:* In sph_kernel.hpp (included by both sph_cpu.hpp and sph_cuda.hpp), define 'inline constexpr float kSphViscosityWidthScale = 0.9128709291752769f;
- **Minimum SPH tree arena of 2M nodes means about 370 MB per locality even for tiny runs** — `src/sph/sph_tree.cpp:265` · `sph_infrastructure#14`  
  Every hydro run allocates, touches and keeps at least kMinSphTrees = 2,097,152 SPH tree nodes per locality, whatever the particle count: about 336 MB at the default 160-byte SphTreeNode, or 369 MB with LONG_LONG_PART_INT. *Fix:* Replace the fixed floor with an explicit bound such as max(c\*N_local/bucket, workerThreads\*treeCacheLineSize + margin). Raise c (for example to 8) to cover midpoint-split imbalance and the per-thread chunk waste, or …
- **Domain boxes are computed but discarded, so every local SPH root uses the whole unit box** — `src/sph/sph_tree.cpp:341` · `sph_infrastructure#6`  
  The domain boxes for global splits are looked up but discarded, so every local SPH tree starts its k-d splits from the unit box instead of the locality's domain. This adds levels with empty leaves and leaves node.box uninformative. There is no correctness impact. *Fix:* Assign leftBox/rightBox from domainsRange as group_tree.cpp does; this is valid because gas has just been migrated into its domains. Otherwise delete the dead lookups.
- **SPH tree cache is sized with kSphTreeCacheSize but its bin index is reduced modulo kTreeCacheSize** — `src/sph/sph_tree.cpp:566` · `sph_infrastructure#13`  
  The SPH tree cache arrays are sized by kSphTreeCacheSize, but their bin index comes from TreeIdHashLo, which reduces modulo the gravity constant kTreeCacheSize. The two are equal only by coincidence, and lowering kSphTreeCacheSize alone would cause out-of-bounds access in release builds. *Fix:* Add static_assert(kSphTreeCacheSize == kTreeCacheSize) next to the declarations, or give the SPH cache its own Lo/Hi hashers parameterized on kSphTreeCacheSize.

### Cosmology and initial conditions

- **cloudWeight (CUDA_EXPORT) silently computes in double precision** — `cosmictiger/cosmology/fft.hpp:107` · `cosmology#16`  
  cloudWeight, a host/device function, evaluates all three branches in FP64 because of unsuffixed double literals. That slows the CUDA power-spectrum deposit on consumer GPUs and gains no accuracy, since the powers are computed in float. *Fix:* Evaluate in float with f-suffixed literals, preferably in the well-conditioned B-spline form w = (p(3-x) - 6p(2-x) + 15p(1-x))/120 with p(t) = max(t,0)^5.
- **recombination::advanceGasTemperature is test-only; production duplicates a modified version of the SDIRK step inline** — `cosmictiger/cosmology/recombination_thermal.hpp:26` · `cosmology#19`  
  recombination::advanceGasTemperature is a frozen-coefficient SDIRK2 step whose comment calls it the 'existing' background step, but only tests/recombination_thermal.cpp calls it. zero_order.cpp:295-317 carries a diverged inline copy that evaluates the Compton coupling and T_rad at stage times. *Fix:* Simplest: make BackgroundThermalHistory::temperature mandatory, delete the else branch, advanceGasTemperature and its TemperatureEvolution test, and remove the dead sigmaC/t/dt.
- **docs/linear_growth.md describes an initializer that no longer exists; the 2LPT rate ignores radiation** — `docs/linear_growth.md:247` · `cosmology#18`  
  docs/linear_growth.md:247-258 (plus line 62, and the comment and existing_ic\_\* columns in tests/analyze_linear_growth.py:175-246) describes initialize.cpp as using a matter+Lambda growth factor, an Omega_m^gamma rate and a radiation-free H. *Fix:* Rewrite or delete the 'Existing cosmological background approximations' section and the line-62 reference, or mark the whole doc as a dated historical record.
- **fft3dForceReal, finishForceReal and fft3d2silo are unused; fft3d2silo is also wrong for &gt;1 locality** — `src/cosmology/fft.cpp:67` · `cosmology#15`  
  fft3d2silo is dead debug code that would read out of bounds on more than one locality and mislabels the real-mesh axes. fft.hpp also carries four commented-out cloud kernels. fft3dForceReal and finishForceReal are not dead; the fft1/fft2 ctests use them. *Fix:* Delete fft3d2silo (and the silo include in fft.cpp if nothing else needs it) along with the commented-out kernels in fft.hpp. Keep fft3dForceReal and finishForceReal, or move them into test-only code.
- **fft3dAccumulateReal sends remote chunks one at a time (fut.get() inside the loop)** — `src/cosmology/fft.cpp:398` · `cosmology#12`  
  fft3dAccumulateReal sends each intersection chunk and waits for it before building the next. This serializes the power-spectrum density deposit, including self-sends on a single locality. The 2LPT path is not affected, because it accumulates locally. *Fix:* Collect the chunk futures and wait_all once, as fft3dAccumulateComplex does, with a small bounded number of chunks in flight to cap the extra copy memory.
- **fft3dInit allocates n^2 heap spinlocks per locality, and fft3dDestroy never frees them** — `src/cosmology/fft.cpp:494` · `cosmology#11`  
  fft3dInit allocates n^2 individually heap-allocated hpx::spinlocks on every locality, although only about n^2/P are ever indexed. It repeats this on every init, and fft3dDestroy never releases them, leaving about 48 B x n^2 per locality resident (about 0.8 GB at n=4096, independent of P). *Fix:* Size the locks to the local row count (local nx\*ny), store them in one contiguous std::unique_ptr&lt;SpinlockType[]&gt;, allocate once per grid size, and release them in fft3dDestroy.
- **Per-task FFTW plan creation through run_as_os_thread plus a global lock, with VLA buffers on HPX stacks** — `src/cosmology/fft.cpp:537` · `cosmology#13`  
  Each FFT phase task creates and destroys its own 1D FFTW plan through two OS-thread hops and a global mutex, about 5n/P serialized planner calls per 3D transform. The work buffers are VLAs on 512 KiB HPX stacks, which would overflow only at n of about 32768. *Fix:* Create each phase's plan once (or cache plans by n and kind) under the planner lock, before spawning tasks. Have the tasks call fftwf_execute_dft, fftwf_execute_dft_r2c or fftwf_execute_dft_c2r on per-task heap buffers …
- **Unused variables and flags, and a per-particle pow, in the IC code** — `src/cosmology/initialize.cpp:414` · `cosmology#20`  
  The IC path carries dead and noisy code that changes no results: - initialize.cpp:414: twolptCorrection2's `factor` is captured and never used. - zeldovichEnd: `initializeParticles` is only forwarded, and `mutex`, `glass`, `iValue` and `ninv` are unused. *Fix:* - Delete `factor` from twolptCorrection2. - Drop the `initializeParticles` parameter from zeldovichEnd and ZeldovichEndAction, and update its callers.
- **Every locality solves the full Einstein-Boltzmann spectrum (function-static inside twolptGenerate)** — `src/cosmology/initialize.cpp:467` · `cosmology#17`  
  Every locality independently repeats the full Einstein-Boltzmann solve of 1024 or more modes for identical tables, multiplying aggregate CPU time by P. The function-static also freezes the spectrum for the process, which is a latent stale-amplitude hazard if initialize() is ever called again with new options. *Fix:* Compute the two amplitude tables and the normalization once, either on rank 0 or with modes split across localities and gathered, then broadcast them.
- **Dead, inaccurate zero-order time conversions, an unused Cosmos class, and dead locals** — `src/cosmology/zero_order.cpp:45` · `cosmology#14`  
  ZeroOrderUniverse's conformal-time and time conversions and the Cosmos class are dead code, and they would be inaccurate if revived (float RK3 weights give about 5e-4 drift, logf endpoints, flat Lambda only). The live createZeroOrderUniverse also computes unused t/dt, sigmaC and a1, and has an empty if block. *Fix:* Delete conformalTimeToScaleFactor, scaleFactorToConformalTime, redshiftToTime and Cosmos, but keep redshiftToDensity, which a test uses.

### Analysis: FoF and Rockstar

- **groups.dat and per-rank group files contain uninitialised padding bytes** — `src/analysis/group_entry.cpp:48` · `analysis_groups_rockstar#13`  
  groups.dat (the GroupHeader's 4 int-to-double padding bytes, from an uninitialized stack object) and the per-rank group files (4 heap padding bytes per raw-written parents pair) contain uninitialized bytes. That causes sanitizer/valgrind noise and a minor information leak. *Fix:* Keep the existing on-disk layouts and zero the padding. memset the GroupHeader (or use static/zero-initialized storage) before filling it, and write each parent through a zero-initialized local struct {GroupInteger id;
- **Every locality holds a full copy of the previous snapshot's halo-to-host map** — `src/analysis/groups.cpp:76` · `analysis_groups_rockstar#18`  
  groupsInstallHistory broadcasts every previous halo-host pair to every locality, and each locality keeps the full map in a static unordered_map between analyses. Per-locality memory and broadcast volume therefore scale with the global halo count, although each rank needs only entries for halos its own particles came … *Fix:* Partition the history by halo id (for example owner = halo % hpxSize()) instead of broadcasting it, and have groupsAddParticles batch lookups of its distinct lastGroup values to the owning ranks.
- **groupsCull's distributed group-existence query has no effect: pending assignments overwrite every FoF-tagged particle afterwards** — `src/analysis/groups.cpp:159` · `analysis_groups_rockstar#6`  
  groupsReduce queues an assignment for every particle of every FoF candidate across all waves, so the final pendingAssignments loop in groupsCull overwrites everything the distributed existence query and removeStale pass could clear. *Fix:* Replace the existence query and removeStale with a purely local pass: reset every particlesGroup entry to kNoParticleGroup, then install pendingAssignments.
- **Snapshot and lightcone use different minimum FoF sizes; the lightcone record filter is a no-op that would break pid if it ever filtered** — `src/analysis/groups.cpp:394` · `analysis_groups_rockstar#19`  
  Snapshot analysis skips FoF groups under a hard-coded 20 particles, while lightcone analysis uses kRockstarMinGroup = 10, so isolated 10-19-particle hosts appear only in lightcone catalogues. *Fix:* Use one named threshold in both paths (analysis.minGroupSize, or kRockstarMinGroup, at groups.cpp:394 and lightcone.cpp:290) and document it in docs/rockstar.md.
- **groupsFind fetches both child nodes and never uses them, which can block on a remote cache read** — `src/analysis/groups_find.cpp:307` · `analysis_groups_rockstar#11`  
  groupsFind fetches both child nodes at lines 307-308 and never uses them. There is no performance effect: groupsFindFork immediately does the same synchronous groupTreeGetNode on the child and needs its processRange to dispatch, so these are simply two dead lines. *Fix:* Delete lines 307-308.
- **Dead code and dead state across the ROCKSTAR/FoF slice** — `src/analysis/rockstar.cpp:290` · `analysis_groups_rockstar#9`  
  The slice carries dead functions (rockstarFindAllLinkLens, rockstarCudaFree with its blocksUsed lease accounting), unused structs (RockstarParticles, RockstarSeed), and Subgroup fields that production never reads (hostParticleCount, depth, vcircMax, radius500, and mass200/500/2500 outside tests). *Fix:* Delete the dead functions, structs, lease counter and unused locals, plus Subgroup::hostParticleCount, depth, vcircMax and radius500. Replace Subgroup::t/w with a returned {T, W} from detail::energies.
- **The `gpu` argument of the ROCKSTAR helpers is silently overwritten** — `src/analysis/rockstar.cpp:446` · `analysis_groups_rockstar#10`  
  rockstarFindLinkLen and rockstarFindSubgroups unconditionally overwrite their `bool gpu` parameter with compute.cuda &amp;&amp; N &gt; 4096, and rockstarSeeds threads that ignored flag through every level. A caller that passes false to force the CPU path silently gets CUDA whenever CUDA is enabled and N &gt; 4096. *Fix:* Respect the argument with `gpu = gpu && parts.size() > 4 * 1024;`, or remove the parameter from rockstarFindLinkLen, rockstarFindSubgroups and rockstarSeeds and document the option and size rule in one place.
- **Avoidable superlinear and copy-heavy work in rockstarSeeds: O(G\*N) extraction, re-sort on every merge-loop step, copies instead of moves** — `src/analysis/rockstar.cpp:582` · `analysis_groups_rockstar#7`  
  rockstarSeeds does avoidable superlinear and copy-heavy bookkeeping: - it rescans the remaining particles once per subgroup, O(G\*N) per level; - it fully re-sorts the flattened seed list on every merge-loop step, O(S^2 log S) even with no merges, restarting at k=0 after each merge; *Fix:* Bucket particles by subgroup label in one pass. std::move theseParticles into the recursive call, use make_move_iterator for the insert, and push_back(std::move(sg)) / subgroups[k] = std::move(subgroups.back()).
- **rockstarHaloBound is a test-only unbinding path that has drifted from production unbinding** — `src/analysis/rockstar_catalogue.cpp:462` · `analysis_groups_rockstar#14`  
  rockstarHaloBound is a test-only duplicate of single-halo unbinding that lacks production's binding-system potentials, inclusive bulk velocity and child reconsideration. The 'unbinding order' regression exercises this duplicate instead of the production lambda in rockstarFinalizeSubgroups, leaving the production … *Fix:* Extract the production per-halo test, taking frozen potentials keyed by ordinal and a bulk velocity, into one function called by rockstarFinalizeSubgroups and by the test helper.
- **A Barnes-Hut energy evaluation per halo is thrown away, and another is done twice** — `src/analysis/rockstar_catalogue.cpp:622` · `analysis_groups_rockstar#5`  
  rockstarPruneCatalogueGroups ends by running a BH-tree energy evaluation on every surviving halo, but nothing ever reads the resulting Subgroup::t/w; consumers use a local `measured` Subgroup instead. *Fix:* Delete lines 622-624 and the energies call in rockstarHaloBound. Make detail::energies return {T, W} instead of writing Subgroup fields.

### Analysis: lightcone, BH and power

- **bhEvaluatePotential/bhEvaluatePoints reorder the caller's array but return results in the original order** — `src/analysis/bh.cpp:341` · `analysis_lightcone_bh_power#21`  
  bhEvaluatePotential takes positions by non-const reference and reorders them into tree order while building the tree, but returns potentials in the original order. After the call, x[i] and result[i] describe different particles. bhEvaluatePoints also reorders its source array. *Fix:* Take the positions by value so callers can std::move them, and reorder the internal copy. Delete the dead box computation, rpot, numberOfThreads and futs.
- **BH GPU launch-configuration and parameter-passing smells** — `src/analysis/bh.cu:350` · `analysis_lightcone_bh_power#22`  
  The self-potential occupancy query passes the static BhShmem as dynamic shared memory, so it does not match the launch (probably no practical effect on sm_86). The points launcher sizes its grid as devY.size()/64 although each block processes one point at a time. This path is test-only today. *Fix:* Pass 0 as the dynamic shared memory to match the launch. Size the points grid from devY.size(), and add an optional maximumBlockCount (as bhEvaluatePotentialGpu has) so the workspace-reuse test can still bound it.
- **lcAddParticles always returns 0, so the drift's flush report never fires** — `src/analysis/lightcone.cpp:194` · `analysis_lightcone_bh_power#20`  
  Both size_t lcAddParticles overloads always return 0, because lcAddParticleEntries ends in `return 0;`. The GPU drift's "flushed" report in drift.cu is therefore unreachable, and the return type promises a count that is never produced. *Fix:* Either return counts.back() and fix the PRINT format (%lld/%zu, or use terminal::detailPrint), or make the overloads void and remove cnt, totalCnt and the PRINT from cudaDrift.
- **lcParts2groups inserts into the unordered_map it is iterating; the merge is pointless anyway** — `src/analysis/lightcone.cpp:294` · `analysis_lightcone_bh_power#25`  
  lcParts2groups inserts the kLightconeNoGroup key into the unordered_map it is iterating. A rehash at that moment makes the following erase(j) formally UB (benign in libstdc++, apart from reordering). *Fix:* Delete the merge loop, or erase small groups with `j = groupsMap.erase(j)` without inserting a new key.
- **32-bit overflow in the lcParticleBoundaries2 work split for large ghost pixels** — `src/analysis/lightcone.cpp:753` · `analysis_lightcone_bh_power#8`  
  The work-split bounds in lcParticleBoundaries2 (lines 753-754 and 766-767) multiply int by int (DeviceVector::size() returns int) without the size_t widening used at line 399. Overflow would need more than about 4e9 lightcone particles per rank in one flush, a scale at which the module's int-indexed particleBuffer and … *Fix:* For consistency, compute the bounds as size_t(thread) \* size / numberOfThreads at 753-754 and 766-767, or replace the hand-rolled split with hpx::for_loop.
- **Ghost label exchange relies on identical trees on owner and ghost, with no size check** — `src/analysis/lightcone.cpp:756` · `analysis_lightcone_bh_power#26`  
  The ghost label and activity exchange copies the owner's arrays by index. That is correct only if the owner's and ghost's particle order and tree numbering match exactly. They do today, because both sides sort identical pre-sort copies deterministically, but sizes are never checked, so a future change would silently … *Fix:* Before copying, add `ALWAYS_ASSERT(p.first.size() == parts.group.size() && p.second.size() == nodes.size())`, and document the invariant at lcParticleBoundaries1 and lcFormTrees.
- **Dead lightcone code, including lcSiloOut writing positions as velocities** — `src/analysis/lightcone.cpp:1044` · `analysis_lightcone_bh_power#19`  
  lightcone.cpp and lightcone.hpp contain a cluster of dead code. The worst item is lcSiloOut: it has no callers, writes the x/y/z coordinates as its vx/vy/vz point variables, and does not check DBCreateReal. *Fix:* Delete the dead items, along with the silo.h include in lightcone.cpp and the cuda_fp16.h include in lightcone.hpp, which only they need. Make lcFindNeighbors return void and rename the header parameter to tau.
- **Leftover debug output in lightcone hot paths** — `src/analysis/lightcone.cu:191` · `analysis_lightcone_bh_power#18`  
  cudaLightcone prints a debug line on every group-finding iteration on every rank. lcFormTrees keeps a per-particle debug print in place of a disabled assertion. *Fix:* Remove the cudaLightcone print or route it through terminal::detailPrint. Replace the per-particle low-R print with a counter reported once per flush, or restore it as an assertion with a tolerance.
- **Empty low-k bins are reported as P=0, and the mode-averaged k is computed then discarded** — `src/analysis/power.cpp:70` · `analysis_lightcone_bh_power#12`  
  Low-k log bins narrower than the spacing of discrete |k| values are empty and are written to power.N.txt as zero-power rows. The bin k is always the geometric centre because the computed mean |k| is overwritten. A non-integer ns (fourierSize &gt; 512, not divisible by 8) truncates the top bin edge. *Fix:* Skip empty bins or flag them with NaN, report ktot/count as the bin k, and use an integer bin count, or force the last edge to kmax. Remove the dead count==1 branch, or mark those bins' errors as undefined.
- **searchKbins copies the bin-edge vector at every recursion level for every mode** — `src/analysis/power.cpp:90` · `analysis_lightcone_bh_power#10`  
  searchKbins takes the bin-edge vector by value and recurses about log2(bins) times. Every mode lookup in both binning passes therefore does about 8 heap allocations and copies of 65-257 floats, billions in total for n=1024. *Fix:* Pass `Vector<float> const&`, or compute the bin directly as floor((log k - log kmin)/dlogk) with a one-step adjustment against the float edges.
- **Power binning uses particlesPerDimension as the FFT size** — `src/analysis/power.cpp:115` · `analysis_lightcone_bh_power#15`  
  The power-spectrum binning passes use particlesPerDimension as the FFT size for frequency wrapping and window deconvolution, while the grid uses fourierSize. They match only because options.cpp forces them equal. Both passes also compute an unused boxSize. *Fix:* Pass the FFT size from powerSpectrumCompute, or read fourierSize, in both passes. Delete the unused boxSize locals.
- **accumulateDensityCuda zeroes the tile with an int loop over a 64-bit volume and repeats setup per tile** — `src/analysis/power.cu:39` · `analysis_lightcone_bh_power#24`  
  accumulateDensityCuda zeroes its managed tile element by element on the host with an int index over an int64 volume. That is UB beyond INT_MAX, which needs about 2.1e9 particles per rank and is practically unreachable. The pattern also forces page migrations in both directions. *Fix:* Use device memory with cudaMemsetAsync(devRho, 0, n3\*sizeof(float), stream) and cudaMemcpyAsync back to rho, or at least use an int64_t index.
- **CUDA density kernel recomputes stencil weights 258 times per particle, in FP64** — `src/analysis/power_cuda.cu:37` · `analysis_lightcone_bh_power#23`  
  accumulateDensityKernel evaluates cloudWeight 258 times per particle instead of 18. The compiled sm_86 code confirms that wtY and wtZ are not hoisted, and cloudWeight's double literals make each evaluation FP64 arithmetic with a full double division. masses[i] is reloaded before every atomic. *Fix:* Compute float wx[6], wy[6], wz[6] and the mass once per particle, as power_cpu.cpp does, and mark the pointers \__restrict_\_. Make cloudWeight's literals float.

### Math and numerics

- **Misleading or dead position helpers: kFixed2float is 2^-32 in 64-bit builds, rngdbl2rngfixed32 maps an end of 1.0 to 0, Fixed32Range is not a Fixed32 range, Fixed::min() is one ulp** — `cosmictiger/math/fixed.hpp:28` · `math_numerics#13`  
  kFixed2float (always 2^-32, wrong for Q0.64 positions), fixed32UnitBox() and rngdbl2rngfixed32() are dead code. rngdbl2rngfixed32 would also wrap an end of 1.0 to 0, which gives an empty or inverted box. Separately, Fixed32Range is a range of signed Q3.29 RangeFixed values, not of Fixed32, so the name is misleading. *Fix:* Delete kFixed2float, fixed32UnitBox and rngdbl2rngfixed32. If a double-to-Position box conversion is ever needed, map an end of 1.0 to Position::max().
- **InterpolationFunctor: no n&gt;=4 guard, out-of-range calls only printf and extrapolate, and the builder makes needless copies** — `cosmictiger/math/interpolation.hpp:29` · `math_numerics#11`  
  InterpolationFunctor never checks that n &gt;= 4, so smaller tables index out of bounds. Out-of-range lookups print on every call and silently extrapolate, and the builder copies the table twice. Current callers use 4097-entry, validated tables, so none of this triggers today. *Fix:* In buildInterpolationFunction, validate n &gt;= 4 and finite 0 &lt; amin &lt; amax. Throw, or clamp and warn once, on out-of-range lookups.
- **Unused kernel.hpp helpers: kernelFqinv/kernelPot apply the hydro kernel to gravity force and potential; smoothX helpers are dead and divide by zero at h == hmin** — `cosmictiger/math/kernel.hpp:61` · `math_numerics#3`  
  Five kernel.hpp helpers have no callers: kernelFqinv, kernelPot, dsmoothXDh, smoothX and dlogsmoothXDlogh. kernelFqinv and kernelPot evaluate the hydro kernel's gravity force and potential, while every production softening site uses selectedGravityKernel(). *Fix:* Delete kernelFqinv, kernelPot, dsmoothXDh, dlogsmoothXDlogh and smoothX together with the stale comment. Keep kernelW and dkernelWDq.
- **Dead helpers in math.hpp and tensor.hpp, one with a [0,1) wrap bug** — `cosmictiger/math/math.hpp:40` · `math_numerics#6`  
  These helpers have no callers: math.hpp's constrainRange, tsc, erfnearzero, expnearzero, roundDown, math::max and randNormal(), and tensor.hpp's indicesBegin, indicesInc, indicesEnd, indicesToSym, symToIndices and trlessIndex. constrainRange also leaves x == 1 unwrapped. *Fix:* Delete the unused helpers listed in the summary. Keep atomicAdd(std::atomic&lt;float&gt;&amp;, float), or replace its use in view.cpp with C++20 std::atomic&lt;float&gt;::fetch_add.
- **Fixed32Range::contains tests all 27 image combinations although the test separates by dimension (SPH neighbor-filter hot loop)** — `cosmictiger/math/range.hpp:402` · `math_numerics#9`  
  Fixed32Range::contains tests all 27 periodic image offsets jointly, although the containment test separates by dimension. A rejected particle costs up to 27 combinations of fixed-point compares (plus possible double-to-fixed conversions) where 3 per dimension would do, in every SPH neighbor filter. *Fix:* For each dimension, accept if any shift s in {-1, 0, 1} satisfies `pt + ulp >= begin + s` and `pt <= end + s + ulp`, using precomputed RangeFixed(±1.0). Return false at the first dimension with no valid shift.
- **Legacy SIMD transcendentals in simd.hpp: erf/erfc wrong for x&lt;0, exp maps NaN to 1.65e38, log2(0) gives NaN; production Ewald still uses them** — `cosmictiger/math/simd.hpp:106` · `math_numerics#8`  
  The legacy SIMD transcendentals in simd.hpp have edge-case defects. erf, erfc and erff are wrong for x &lt; 0 and have a pole near -3.0526, though they have no production callers. exp turns NaN into about 1.65e38. log2 returns NaN for 0 (and +inf or NaN for subnormals). The constants are static locals. *Fix:* Delete the unused SimdFloat erf, erfc, erff, log, log2, pow, sinf and expf overloads, and make the remaining constants constexpr locals or literals.
- **docs/kernels.md says the SPH solver still requires CUDA, but a SIMD CPU SPH backend exists** — `docs/kernels.md:147` · `math_numerics#12` · also `sph_physics_kernels#10`  
  docs/kernels.md:147-148 still says the SPH solver requires CUDA. A committed SIMD CPU SPH backend implements every SPH phase and runs when compute.cuda=false, as docs/monaghan_sph.md describes. *Fix:* Rewrite that paragraph to say the shared radial-kernel evaluators back both the CUDA and the SIMD CPU SPH solvers. Link docs/monaghan_sph.md for the remaining CUDA-only initialization and analysis paths.
- **rockstar_eigenpair test exercises findEigenpair, which production never calls** — `src/math/math.cpp:6` · `tests_other_cpp#13`  
  findEigenpair (the only function in src/math/math.cpp) is a wrapper over rockstar::symmetricEigenpairs with no production caller. Its dedicated CTest uses only a diagonal tensor, so it validates dead code that rockstar_properties already covers better. *Fix:* Delete findEigenpair, src/math/math.cpp (from variant_sources.cmake as well), the math.hpp declaration, and the rockstar_eigenpair test and target.

### Build system and scripts

- **Documented build-job cap contradicts build.sh, and the test-job limit misreads ctest --parallel** — `CMAKE_OPTIONS.md:186` · `build_cmake_scripts#10`  
  CMAKE_OPTIONS.md still documents a half-core build cap that build.sh removed, so the README and build.sh now disagree with it. build.sh's coreCount/8 test-job limit double-counts PROCESSORS, giving about 1/8 of the CPUs instead of the intended half. *Fix:* Update CMAKE_OPTIONS.md to describe build.sh's actual uncapped default; the cap was removed deliberately in 46759ca8. Set maxTestJobs to about coreCount/2 slots, let PROCESSORS do the accounting, and reword the "four …
- **Hand-maintained header list is stale and contains a duplicate** — `CMakeLists.txt:301` · `build_cmake_scripts#15` · being edited  
  The hand-maintained IDE header list contains a duplicate entry and omits 37 of 208 tracked project headers. It has no build effect but keeps drifting. *Fix:* Delete the list, or generate it with `file(GLOB_RECURSE cosmictiger_header_files CONFIGURE_DEPENDS cosmictiger/*.hpp)`.
- **GoogleTest fallback download has no integrity hash** — `CMakeLists.txt:525` · `build_cmake_scripts#16` · being edited  
  The GoogleTest FetchContent download, which every local build tree uses because GTest is not installed, has a pinned commit but no URL_HASH. Downloaded content is never verified, and configuring with tests on needs network access. *Fix:* Add `URL_HASH SHA256=<digest>`. The locally cached archive hashes to 7ff5db23de232a39cbb5c9f5143c355885e30ac596161a6b9fc50c4538bfbf01; confirm that from an independent download before pinning it.
- **Radau chemistry reference test silently disappears when numpy/scipy are missing** — `CMakeLists.txt:964` · `tests_sph_chem_cpp#12` · being edited  
  The SciPy Radau reference test (primordial_radau_reference, built from tests/primordial_reference.cpp/.py) is registered only when `import numpy, scipy` succeeds at configure time. Otherwise it disappears from CTest with no configure message and no reported skip. *Fix:* Always register primordial_radau_reference. Have primordial_reference.py exit 77 when the numpy/scipy import fails, and set SKIP_RETURN_CODE 77, which matches the project's other optional tests.
- **SPH chemistry timestep regression runs on one locality only and only with the traditional integrator** — `CMakeLists.txt:999` · `tests_sph_chem_cpp#11` · being edited  
  The standalone SPH chemistry timestep regression is registered only as a single-locality run, and only when the traditional integrator is compiled. Its cross-rank prepare/drain/rethrow and failed-batch atomicity paths never run with two localities, and global-only SPH+chemistry builds run no chemistry timestep … *Fix:* Remove the INTEGRATOR_TRADITIONAL gate, since the test already passes ${cosmictiger_test_integrator_args}. Add an 'sph_chemistry_timesteps_mpi' variant with 2 ranks under 'HPX_WITH_PARCELPORT_MPI AND …
- **Application checks compiled into more variants than they are registered for; MPI catalogue test unguarded** — `CMakeLists.txt:1280` · `tests_other_cpp#14` · being edited  
  Several MPI application tests, groupsCatalogue_mpi among them, are registered without the HPX_WITH_PARCELPORT_MPI guard or the MPI transport args, and would break against an HPX without the MPI parcelport. *Fix:* Guard every mpiexec-launched HPX test (groupsCatalogue_mpi, analysis_cpu_mpi, gravity_error_mpi) with HPX_WITH_PARCELPORT_MPI and pass ${cosmictiger_mpi_transport_args} consistently.
- **generated_fmm_current test depends on whichever clang-format is on PATH; its --min/max-order flags actually mean degrees** — `CMakeLists.txt:1548` · `build_cmake_scripts#17` · being edited · also `fmm_generators_tools#14`  
  The generated_fmm_current test runs whatever `clang-format` is on PATH, with no version check and no skip when it is missing. The generator's `--min-order`/`--max-order` flags actually take local degrees (FMM order minus 1), while the test next to it uses 'order' to mean FMM order. *Fix:* In CMake, `find_program(COSMICTIGER_CLANG_FORMAT NAMES clang-format-21 clang-format)` and check its version. Pass the path to the generator (e.g.
- **rejected_options test passes on any nonzero exit** — `CMakeLists.txt:2332` · `build_cmake_scripts#18` · being edited  
  rejected_options relies only on WILL_FAIL, so any normal nonzero exit counts as success: startup exceptions caught in main, MPI/FFTW initialization failures, or a loader error. It never checks that the option parser actually rejected `--invalid_option_for_test`. *Fix:* Remove WILL_FAIL and set `PASS_REGULAR_EXPRESSION "unrecognised option.*invalid_option_for_test"` (Boost/HPX program_options wording). WILL_FAIL must go because CTest inverts a regex match when it is set.
- **variational_sph_application registration needlessly requires CHEMISTRY and has a stale comment** — `CMakeLists.txt:2413` · `tests_python#14` · being edited · also `recent_commits_b#11`  
  The variational_sph_application CTest is gated on COSMICTIGER_ENABLE_CHEMISTRY, and its comment still describes two pressure operators and a chemistry rejection. The driver needs neither since 0720c881, so SPH builds without chemistry silently lose the Sod/blast/hydroWave/CFL regression. *Fix:* Drop COSMICTIGER_ENABLE_CHEMISTRY from the condition. Rewrite the comment to say it runs variational SPH Sod, blast and hydroWave accuracy, a CFL refinement, and an SPH cosmology case that needs gravity.
- **Stale or false statements in README, source_layout.md and the patch notes** — `README.md:743` · `build_cmake_scripts#11`  
  Four doc statements are stale. The README says numerical_utilities needs CUDA headers and that CPU-gravity runs initialize the GPU, which the code no longer does. It also claims every patch records its revision, but hpx_apex_tau.patch has none. source_layout.md omits the priceEtAl viscosity model. *Fix:* Limit the CUDA-headers sentence to compression_test. Replace the GPU-initialization sentence with: CPU runs (compute.cuda=false) do not touch CUDA; the CUDA-backend tests need a device.
- **libquadmath is linked into every build, though only the Sedov scenario uses it** — `cmake/variants.cmake:11` · `build_cmake_scripts#19`  
  libquadmath is linked unconditionally through cosmictiger_platform, but only the SPH-only src/scenarios/sedov.cpp uses \__float128. Gravity-only builds carry a needless dependency, and builds on targets without \__float128/libquadmath (e.g. GCC on AArch64) would probably fail. *Fix:* Link quadmath only alongside the SPH sources, e.g. `if(COSMICTIGER_WITH_SPH) target_link_libraries(cosmictiger_objects PUBLIC quadmath)`, behind a check_cxx_source_compiles probe for \__float128.
- **-march=native is forced on the application, tools and tests, and CMAKE_CXX_FLAGS cannot override it** — `cmake/variants.cmake:15` · `build_cmake_scripts#5`  
  -march=native is added unconditionally (whenever the compiler accepts it) to the application, tools and many tests. Because target options follow CMAKE_CXX_FLAGS, a user's -march in CMAKE_CXX_FLAGS is silently overridden. *Fix:* Add a documented cache option, e.g. COSMICTIGER_TARGET_ARCH (default `native`; empty disables it; otherwise passed as -march=&lt;value&gt;).
- **COSMICTIGER_WITH_LONG_LONG_PART_INT does not reach the standalone tests that compile application sources** — `cmake/variants.cmake:18` · `build_cmake_scripts#13`  
  COSMICTIGER_WITH_LONG_LONG_PART_INT reaches only cosmictiger_platform, not cosmictiger_default_config. Standalone tests that compile application sources using ParticleInteger (drift_dispatch_test, rayleigh_taylor_test) always build with 32-bit indices. *Fix:* Add `$<$<BOOL:${COSMICTIGER_WITH_LONG_LONG_PART_INT}>:LONG_LONG_PART_INT>` to cosmictiger_default_config, and keep or derive the platform definition from the same place.
- **Capability macros come from raw cache values, so SPH-off builds still see COSMICTIGER_ENABLE_CHEMISTRY=1 (and STARS)** — `cmake/variants.cmake:45` · `build_cmake_scripts#14`  
  The chemistry and stars macros mirror the raw cache options (header fallback 1) rather than the effective SPH-and-feature capability. Every use site must add `COSMICTIGER_WITH_SPH &&` by hand; current code does, and the build checkers encode this convention, but nothing prevents a future unguarded use. *Fix:* Emit the effective values (CHEMISTRY = SPH AND CHEMISTRY, STARS/STAR_FORMATION = SPH AND STARS) and make the header fallback 0.

### FMM generators and tools

- **cartesian_kernels.md still says the application is GPL and gives the wrong header directory** — `docs/cartesian_kernels.md:15` · `fmm_generators_tools#17`  
  docs/cartesian_kernels.md:15 still says the surrounding application is GPL, although the original code has been Boost-licensed since d5c0fe86. Line 108 gives the pre-move header directory, cosmictiger/gravity/ instead of cosmictiger/gravity/fmm_kernels/. *Fix:* Replace the clause on line 15 with a statement that CosmicTiger's original code is Boost-licensed, pointing to THIRD_PARTY_NOTICES.md for GPL components. Change line 108 to `cosmictiger/gravity/fmm_kernels/`.
- **Ewald detrace emits unused trace temporaries and counts them in the returned flop estimate** — `tools/cartesian/ewald.py:146` · `fmm_generators_tools#4`  
  computeDetrace emits and counts trace temporaries that the projection never reads: all those whose reduced index has nz&gt;=2, plus every degree-2 trace. That is 25 of 80 at order 8 and 47 of 130 at order 9. *Fix:* First collect the (degree, traceCount, reduced) keys the second loop references. Then emit and count only those temporaries in the first loop. Regenerate, and re-run fmm_flops (CountingReal) and generated_fmm_current.
- **Cartesian Ewald derivatives are emitted into the global namespace 'detail'** — `tools/cartesian/ewald.py:395` · `fmm_generators_tools#13`  
  generateEwald emits ewaldCorrectionDerivatives into a global ::detail namespace. It is the only generated header outside cosmictiger::gravity::{cartesian,spherical}. It works, because callers write ::detail::, but it shares that namespace with unrelated hand-written and test helpers. *Fix:* Emit into cosmictiger::gravity::cartesian::detail. Change the wrapper to detail::ewaldCorrectionDerivatives, update tests/gravity/dehnen/cartesian_ewald_cpu.cpp and fmm_flops.cpp, and regenerate.
- **Generated M2M/L2L multiply by powers of a runtime scaleRatio that production always passes as 1** — `tools/cartesian/translations.py:112` · `fmm_generators_tools#12`  
  The generated Cartesian M2M/L2L kernels take a runtime scaleRatio that every production caller passes as simdBroadcast&lt;Real&gt;(1). In M2M&lt;7&gt; it adds 6 power multiplies and 63 source scalings, and it turns every L2L output accumulation into an FMA. *Fix:* Emit the production M2M/L2L specializations without scaleRatio, or generate a unit-ratio variant selected by a template flag.
- **Spherical Ewald lattice radius 6 does about 3.6x more Green work than double precision needs** — `tools/fmm/spherical_ewald.py:22` · `fmm_generators_tools#7`  
  The fixed lattice radius of 6 (462 pairs) is about 2.4x more than double precision needs for degrees up to 8: R²≈20 (194 pairs) reproduces the R²=49 result to round-off. The cost falls in the cached, one-time PMMM Green setup. *Fix:* If setup time matters, lower the cutoff to R²=20, or derive separate real and reciprocal cutoffs per degree from a target tolerance.
- **spherical_ewald_kernels.hpp gets a duplicated license banner and no 'Generated ... Do not edit' marker** — `tools/fmm/spherical_ewald.py:213` · `fmm_generators_tools#8`  
  spherical_ewald_kernels.hpp carries a duplicated license banner (NOTICE plus generated_notice.txt). It is also the only generated FMM header without the 'Generated by tools/generate_fmm.py. Do not edit.' marker. *Fix:* Remove NOTICE from spherical_ewald.py, emit '// Generated by tools/generate_fmm.py. Do not edit.' as spherical.py's HEADER does, and regenerate.
- **Spherical Ewald compensated (Kahan) accumulation is not protected against the application's default -ffast-math** — `tools/fmm/spherical_ewald.py:230` · `fmm_generators_tools#6`  
  EwaldGreen's generated Kahan accumulation needs -fno-fast-math. Every current instantiation gets that flag from per-file CMake settings, but the header does not enforce it, so a future fast-math translation unit would silently lose the compensation. *Fix:* Emit a dependent guard inside EwaldGreen::evaluateWorking, for example `#ifdef __FAST_MATH__` / `static_assert(sizeof(Real) == 0, "EwaldGreen requires -fno-fast-math");` / `#endif`.
- **Generated M2L template parameter 'sourceOrder' is really a source degree and is never validated** — `tools/generate_fmm.py:503` · `fmm_generators_tools#11`  
  The template parameter `sourceOrder` of the generated Cartesian M2L::add is actually a source degree, compared with the local degree E. Nothing restricts it to E-1 or E, so a wrong value silently drops the top-rank term or reads past the end of a degree-(E-1) source. *Fix:* In the generator, rename sourceOrder to sourceDegree and emit `static_assert(sourceDegree == E || sourceDegree == E - 1)`. Ideally also check Source's static size when it has one.
- **Non-default --min-order/--max-order or --no-format silently rewrite the checked-in headers** — `tools/generate_fmm.py:735` · `fmm_generators_tools#5`  
  If generate_fmm.py is run with a narrower --min-order/--max-order, or with --no-format, and no --output-dir, it silently overwrites the checked-in kernel headers and operation_counts.json with a reduced or unformatted set. The default 4-6 build still compiles afterwards. *Fix:* When outputDir resolves to the source fmm_kernels directory, call ap.error unless the range is exactly 2..8 and formatting is on; otherwise require an explicit --output-dir.
- **Dead struct-to-class regex post-processing in generate_fmm main()** — `tools/generate_fmm.py:750` · `fmm_generators_tools#15`  
  The two struct-to-class regex rewrites in generate_fmm.py main() are dead code. No emitter produces `struct` any more, so they only add a latent risk of rewriting text that happens to match. *Fix:* Delete lines 750-751. The emitters already produce `class ... public:` directly.
- **Halo detail endpoint returns every direct subhalo; JS truncation branch is dead and doc's bounded-memory claim is false** — `tools/halo_viewer/halo_viewer.py:499` · `fmm_generators_tools#2`  
  /api/halos/&lt;id&gt; returns every direct subhalo with all 50 columns and no LIMIT, and sets subhalo_count to the list length. The client's 'Showing X of Y' truncation branch is therefore dead, and response size and DOM work grow without bound for very rich hosts. *Fix:* Add `LIMIT ?` (for example 500) to the children query and a separate `SELECT count(*) FROM halos WHERE catalog_id=? AND host_id=?` (covered by the halo_hosts index) for the true count.
- **'serve' opens the database read-write, creates missing files and directories, and migrates the schema** — `tools/halo_viewer/halo_viewer.py:592` · `fmm_generators_tools#9`  
  serve opens the database read-write: it creates missing files and parent directories and migrates v1 schemas, which is partly documented. Fresh-schema creation is not atomic, so a failed or raced first initialization leaves a database that later runs reject as containing unrelated tables. *Fix:* In serve, require an existing file and give a clear error otherwise (or add an explicit --create flag). Fail on schema version != SCHEMA_VERSION with a hint to run import, instead of migrating at startup.

### Tests

- **Non-test analysis and plotting utilities live in tests/ without CTest registration; shared /tmp MPLCONFIGDIR default** — `tests/analyze_glass_stability.py:26` · `tests_python#20`  
  Several analysis and plotting utilities in tests/ have no CTest coverage of their main paths. Their self-checks are silently left unregistered when numpy, scipy or matplotlib is missing at configure time. *Fix:* Stop defaulting MPLCONFIGDIR to a fixed /tmp path: leave it unset, or use a per-user directory such as ~/.cache/cosmictiger/matplotlib or `tempfile.mkdtemp()`.
- **check_build_variants runs whatever 'cmake' is on PATH, not the configuring CMake** — `tests/check_build_variants.py:321` · `tests_python#18` · being edited  
  build_variants_checker runs whichever `cmake` is first on PATH instead of the CMake that configured the build, so it can error out or test against a different CMake version on module-based systems. *Fix:* unittest.main() consumes argv, so pass the CMake path through the environment: `set_tests_properties(build_variants_checker PROPERTIES ENVIRONMENT "COSMICTIGER_CMAKE=${CMAKE_COMMAND}")`.
- **compare_native_recombination 'compare' silently drops unmatched reference rows and passes on any overlap** — `tests/compare_native_recombination.py:128` · `tests_python#19`  
  The manual `compare` subcommand (and camb_reference.compare) skips reference rows that have no candidate match and passes if any single sample matches within tolerance. A truncated or regridded candidate can therefore report passed=true. The O(N·M) matching costs about 2 s on the documented grids, not minutes. *Fix:* After the loop, fail unless `len(differences) == len(reference)`, listing the unmatched z values, or require an explicit `--allow-partial` (and the same reference-coverage check in camb_reference.compare).
- **analytic.cu: mistyped Madelung constant, int printed with %li, occupancy query counts static shared memory as dynamic; duplicates the production error diagnostic** — `tests/gravity/analytic.cu:221` · `tests_gravity_cpp#8`  
  The analytic CUDA reference adds a mistyped Ewald self-image constant (2.837291 instead of 2.8372974794806), prints an int with %li (UB through the printf wrapper), and passes the kernel's static shared memory as dynamic shared memory to the occupancy query. *Fix:* Use 2.8372974794806196f, print with %i (or cast to long), and pass 0 as dynamicSMemSize. Longer term, add potentials to the production direct reference (GravityReferenceSum in double) and have forceTest use it, then …
- **Cartesian Ewald CPU test has no independent oracle for derivative ranks &gt;= 2** — `tests/gravity/dehnen/cartesian_ewald_cpu.cpp:208` · `tests_gravity_cpp#10`  
  cartesian_ewald_cpu checks the Ewald correction tensor against an independent sum only for the potential and gradient (ranks 0-1), and the Fourier table only through rank 2. Higher ranks used by the periodic M2L and L2P are checked only by symmetry and float/double self-consistency, which cannot detect a symmetric … *Fix:* Extend checkPeriodicField to every rank. Either differentiate the independent sum analytically (plane-wave derivatives are exact (2\*pi\*k)^n products;
- **fmmWalkDeferredSourceHandoff verifies nothing in CPU runs and has no independent reference with CUDA** — `tests/gravity/dehnen/fmm_walk.cpp:330` · `tests_gravity_cpp#2`  
  fmmWalkDeferredSourceHandoff stores the CPU backend's outputs as the 'expected' values and compares only the CUDA backend against them. In CPU runs (fmm_walk_cpu and every CPU-only build) it verifies nothing beyond kick() not failing. *Fix:* Before the backend loop, compute an independent reference that matches the interaction used. For Ewald mode, use cpuGravityEwald over the two source leaves plus L2P, as fmmWalkVerifyEwald does.
- **GravityDirectReference.MixedMassPeriodicImageReciprocity: the masses cancel algebraically** — `tests/gravity/direct_reference.cpp:259` · `tests_gravity_cpp#14`  
  GravityDirectReference.MixedMassPeriodicImageReciprocity multiplies a mass-free reference by constant masses that factor out algebraically. Its first assertion is only an oddness check, F(s) = -F(-s), and the test name claims mixed-mass coverage it does not provide. *Fix:* Rename it, e.g. PeriodicImageForceIsOdd, and drop the masses. For mass coverage, test an API that takes source masses, e.g.
- **fmm_acceptance_cuda support-boundary assertions depend on hard-coded positions in the generated input vector** — `tests/gravity/fmm_acceptance_cuda.cu:258` · `tests_gravity_cpp#13`  
  The strict support-boundary assertions in runAcceptance use hard-coded indices (2, 6, 8, 10) that are correct only because exactly three add() calls, each pushing a case and its reverse, precede the boundary loop. Any earlier insertion retargets them. *Fix:* Save `size_t const overlapIndex = inputs.size();` before the overlapping-support add and `size_t const boundaryBegin = inputs.size();` before the boundary loop.
- **PmmmFftFlops.CachedPlansCountEachExecutionWithFmaAsTwo is a tautology and its name misleads** — `tests/gravity/nitadori/fft_flops.cpp:44` · `tests_gravity_cpp#12`  
  PmmmFftFlops.CachedPlansCountEachExecutionWithFmaAsTwo re-implements fftPlanFlops' one-line formula on the same fftwf_flops output, and checks that FFTW's static plan count is unchanged after execution. It tests no caching or per-execution accounting, despite its name; it only pins the FMA=2 convention. *Fix:* Rename it, e.g. PlanFlopsCountFmaAsTwo, and drop the post-execute check. Or replace it with a solver-level check that cache.forwardFlops/inverseFlops are charged exactly once per FFT execution and on one rank only.
- **particleMeshTimestepTest forces gravity.method to the build default, which is 'fmm' in mixed FMM+PMMM builds** — `tests/gravity/nitadori/particle_mesh_timestep.cpp:250` · `tests_gravity_cpp#15`  
  In builds that compile both the global FMM and a PMMM solver (FMM is ON by default, and the combined-solver build is explicitly tested), gravity::kBuildGravityMethod is "fmm". particleMeshTimestepTest therefore overwrites the CTest's --gravity.method.far=pmmm with "fmm", so the PMMM kick regression runs under options … *Fix:* Replace line 250 with `options.gravity.method = "pmmm";` (the file is compiled only when PMMM is enabled), or drop the override and rely on --gravity.method.far=pmmm like the sibling PMMM tests.
- **PMMM test silently skips near-FMM checks in builds with both near solvers** — `tests/gravity/nitadori/pmmm.cpp:1143` · `tests_gravity_cpp#6`  
  In builds with both PMMM near solvers, checkPmmmStreamedStages always forces near=pp, even under the pmmm_near_fmm CTest. Also, every MPI PMMM CTest uses the default near method (pp), so checkPmmmCellOverlap silently returns. *Fix:* Loop checkPmmmStreamedStages over every compiled near method using kBuildHasPmmmNearPp and kBuildHasPmmmNearFmm. Alternatively, honour the requested getOptions().gravity.pmmmNearMethod and rely on the …
- **No-op option assignments left in fixtures** — `tests/mesh_domains.cpp:359` · `tests_other_cpp#16`  
  tests/mesh_domains.cpp:359, tests/linear_growth.cpp:867 and tests/sph/gravity.cpp:653 assign run.hierarchicalGravity from getOptions() to a fresh copy of getOptions(), which does nothing. *Fix:* Delete all three lines. If a fixture really must state the mode, write `options.run.hierarchicalGravity = usesHierarchicalIntegration(options.run) && options.gravity.enabled;` so it also matches the setOptions …
- **Assert-only checks, one with a side effect, would pass vacuously under python -O / PYTHONOPTIMIZE** — `tests/primordial_reference.py:75` · `tests_python#11`  
  Several drivers implement their checks with bare assert, and some put calls with side effects inside the assert: primordialReferenceInitial and the process.wait calls in run_mpi_launcher. Under PYTHONOPTIMIZE, global_integrator_cpu would lose all its checks and mpi_interrupt_launcher its exit-code checks. *Fix:* Move side effects out of asserts, for example `status = library.primordialReferenceInitial(...)` followed by a check, and `code = process.wait(timeout=8)`.
- **Rejection 'diagnostic' checks match text that every log contains, so any failure passes** — `tests/run_cosmology.py:529` · `tests_python#7`  
  The rejection-diagnostic substring checks match text that every restart log already contains: the 'Reading checkpoint' line and the 'run.scenario : ...' banner. They never confirm that the intended rejection fired. run_cosmology's changed-redshift check is entirely vacuous. *Fix:* Assert the specific error text. Examples: 'Checkpoint cosmology/background does not match', 'Checkpoint scenario, run.endTime or gravity.enabled differs', 'PMMM checkpoint version, mesh size', 'Checkpoint gravity …
- **MPI, CUDA and cross-build comparison paths exist in drivers but CTest never enables them** — `tests/run_global_integrator.py:155` · `tests_python#16`  
  Four drivers have code paths that CTest never runs: the two-locality global-timestep agreement, the SPH-only versus gravity-build reference comparison, the variational SPH MPI/CUDA comparisons and primordial integration on CUDA. They are run only by hand, partly documented as such, and can rot unnoticed. *Fix:* Register global_integrator_mpi and variational_sph_mpi under `HPX_WITH_PARCELPORT_MPI AND MPIEXEC_EXECUTABLE`, reusing CMake's existing launcher argument lists.
- **zip() truncation silently skips chemistry.molecularFraction in run_options manual and override loops** — `tests/run_options.py:816` · `tests_python#10` · being edited  
  Both zip() loops pair the five-entry chemistry_fields with four-value tuples, so molecularFraction is silently skipped in the manual-default and native-override checks. The impact is small: atomic6 forbids a nonzero molecularFraction, and the hot molecular9 case already verifies a molecularFraction override. *Fix:* Add 0.0 as the fifth expected manual default. In the override loop, iterate an explicit list of the four atomic fields, or test the molecularFraction override with --chemistry.network=molecular9.
- **CTest outcomes depend on untracked developer files (lcdm.ini, configurations/lcdm_present.ini)** — `tests/run_options.py:1152` · `tests_python#15` · being edited  
  The working-tree grouped_options driver lints and runs the untracked root lcdm.ini and configurations/lcdm_present.ini, plus any other INI anywhere under configurations/. run_scenarios excludes lcdm_present.ini by name and fails on any other untracked top-level INI. *Fix:* Build the preset inventory from tracked files (`git ls-files default.ini 'configurations/**.ini'`), falling back to an explicit list when there is no git checkout, and share that helper between run_options.py and …
- **Some MPI drivers hard-code '-n', ignore MPIEXEC pre/post flags and parcelport settings; inconsistent CMake guard** — `tests/run_smoke.py:87` · `tests_python#17`  
  run_smoke.py (sph_mpi) and run_analysis_cpu.py hard-code `-n` and cannot receive MPIEXEC_PREFLAGS or POSTFLAGS, unlike their sibling drivers and the direct MPI CTests. analysis_cpu_mpi, groupsCatalogue_mpi and gravity_error_mpi are registered without the HPX_WITH_PARCELPORT_MPI guard that the other multi-locality … *Fix:* Give run_smoke.py and run_analysis_cpu.py (and the manual MPI paths of run_global_integrator and run_variational_sph) the same `--mpi-numproc-flag/--mpi-preflag/--mpi-postflag` options, and pass CMake's existing …
- **Sod smoke accuracy bound (relative L1 &lt; 1.0) cannot detect a non-evolving or badly wrong solution** — `tests/run_smoke.py:124` · `tests_python#8`  
  The Sod smoke test accepts any relative L1 below 1.0. Since positivity is checked separately, this fails only on density overestimates beyond about 2x. Badly wrong solutions pass, for example uniform density 0.5 (L1 about 0.43) or density near zero (about 0.999), although the measured value at N=8 is about 0.10. *Fix:* Tighten the bound from the measured result. N=8 gives 0.099-0.104, so a bound around 0.15 fits. Because L1 at N=8 may not separate an evolved solution from an unevolved one, also check a feature of the evolved solution …
- **CFL-refinement check accepts the energy error growing up to 50% when the timestep is halved** — `tests/run_variational_sph.py:587` · `tests_python#9`  
  The CFL-halving requirement passes whenever |dE| at half the CFL is at most 1.5x the coarse error. First-order or dt-independent energy errors (expected ratio about 0.6-1.0 with the observed 6-&gt;10 steps) are accepted, while the measured ratio is 0.23 at 8^3 and about 0.15 at 16^3. *Fix:* Estimate the order from the actual step counts: p = log(ratio)/log(steps_coarse/steps_fine), and require p &gt;= about 1.5. Alternatively require ratio &lt;= about 0.45, which keeps about 2x margin over the observed 0.23 and …
- **Sphere-surface test comment states a different spacing than the code checks** — `tests/scenarios/sphere_surface.cpp:72` · `tests_other_cpp#19`  
  The comment above checkEvenSpacing names the hexagonal spacing sqrt(8π/(√3 n)), but the adjacent variable `spacing` and the 0.8/1.25 bounds use sqrt(4π/n), which is about 0.931 times that. The real band is [0.744, 1.163] times the hexagonal spacing, a documentation nit that could mislead anyone retuning the bounds. *Fix:* Reword the comment to name the reference actually used, for example: 'within [0.8, 1.25] x sqrt(4 pi / n); the hexagonal spacing sqrt(8 pi / (sqrt(3) n)) is 1.075 times that'.
- **simd MathApproximations indexes 8-element arrays by kSimdFloatSize without an assert** — `tests/simd.cpp:216` · `tests_other_cpp#18`  
  testMathApproximations uses hard-coded 8-element argument arrays with SimdFloat's lane count in the generator and kSimdFloatSize in the check loop. Nothing asserts that these widths agree, so a future width change would cause out-of-bounds UB or silently check fewer lanes. *Fix:* Add `static_assert(SimdFloat::size() == kSimdFloatSize && exponentArguments.size() == SimdFloat::size());`, or size the arrays from SimdFloat::size() and loop to SimdFloat::size().
- **Snapshot record layouts are hard-coded magic offsets with no static_assert; one label is misleading** — `tests/sph/chemistry.cpp:156` · `tests_sph_chem_cpp#7`  
  The snapshot record sizes and field offsets in tests/sph/chemistry.cpp (13 words) and tests/sph/star_formation.cpp (25 words) are literals that depend on primordial::kAbundanceCount, with no static_assert, so a change in species count would silently misalign every per-particle check. *Fix:* Define named offset constants derived from primordial::kAbundanceCount, or a small enum, and add 'static_assert(kChemistryRecordWords == primordial::kAbundanceCount + 7)' and 'static_assert(kSfRecordWords == 19 + …
- **CUDA Fixture::density()/hydro(1) validate source arrays that alias sinks only when separateSources=false** — `tests/sph/cuda.cu:300` · `tests_sph_chem_cpp#6`  
  The CUDA Fixture's density() and hydro(1) validators read source arrays, which hold the kernel outputs only when sinks alias sources (separateSources=false). With separate sources the checks would be vacuous, or would fail spuriously. No current test hits this, so it is a latent trap. *Fix:* Validate data\_.sinkDensity[i] and data\_.sinkParticleRungs[data\_.sinkParticleIndices[i]] (or particleIndex(i)) in these helpers.
- **CUDA SPH test copy-pastes ~900 lines of CPU test bodies that already differ** — `tests/sph/cuda.cu:1259` · `tests_sph_chem_cpp#5`  
  Fifteen SPH test bodies (about 900 lines) are duplicated almost verbatim between cpu.cpp and cuda.cu, and the copies have drifted. For example, CUDA testSelectedKernel uses a non-wrapping pair (0.40/0.58) where the CPU copy wraps (0.98/0.16), and CUDA testZPlaneAcceleration drops the index offsets. *Fix:* Move the shared bodies into a header templated on Fixture, as global_timestep_cases.hpp does. Give both fixtures the same construction and index API (sourceIndex/sinkIndex/particleIndex plus publish helpers) and …
- **'Unchanged' checks use requireClose with an absolute floor that is loose for small values** — `tests/sph/gravity.cpp:430` · `tests_sph_chem_cpp#9`  
  The 'gravity leaves hydro smoothing unchanged' check uses requireClose with a max(1, |expected|) floor, so with hydroSupport = 0.001 it tolerates a 3% change in a quantity that should be bit-identical. *Fix:* Use exact equality for unchanged and round-trip checks, for example `requireClose(..., 0.0)` or a direct `!=` throw. Use a purely relative tolerance, or an explicit absolute scale, when comparing quantities much smaller …
- **Remote tree-cache check accepts any root mass in (0, hpxSize()]** — `tests/sph/gravity.cpp:630` · `tests_sph_chem_cpp#8`  
  The remote tree-cache check only bounds the remote index-0 monopole to (0, hpxSize()]. That accepts any leaf (0.25/0.75) or the wrong subtree root, although the expected value is exact and deterministic (2.0 on proc 0 and 1.0 on proc 1 for 2 ranks). *Fix:* Have the owning locality report nodeMass(treeLocalNode({rank, 0})) through an action, and require the cached remote value to match it exactly.
- **No-op self-assignment of hierarchicalGravity in sphGravityTest** — `tests/sph/gravity.cpp:653` · `tests_sph_chem_cpp#10`  
  sphGravityTest copies the options and then assigns run.hierarchicalGravity its own value, which is a dead self-assignment left over from when the test forced a build constant. *Fix:* Delete the line. If the test must pin a mode, set it to `usesHierarchicalIntegration(opts.run) && opts.gravity.enabled` after changing the other fields.
- **Host/device parity compares signed abundances via fraction(), losing minority precision** — `tests/sph/star_formation_cuda.cu:84` · `tests_sph_chem_cpp#4` · being edited  
  The host/device parity helper compares decoded fraction() values. For upper-branch (x &gt; 0.5) abundances this discards the stored minority's precision and the branch sign bit, which the production CPU/CUDA parity check verifies. All current test inputs are lower-branch, so the gap is latent. *Fix:* Compare `upper()` exactly and `raw()` with the relative tolerance, matching requireSfBackendParity. Add at least one upper-branch abundance (for example x = 0.99999) to an eos/timestep input so the encoding is actually …
- **star_formation_cuda tests treat any CUDA error as 'no device' and silently skip** — `tests/sph/star_formation_cuda.cu:110` · `tests_sph_chem_cpp#3` · being edited  
  star_formation_cuda.cu has its own haveDevice() check, which skips on any cudaGetDeviceCount failure. A misconfigured CUDA driver/runtime therefore shows up as CTest 'Skipped' for these six tests, while every other CUDA suite using detail::CudaTest fails. *Fix:* Include &lt;tests/cuda_test.hpp&gt;, declare `using StarFormationCuda = detail::CudaTest;`, convert the six TESTs to TEST_F, and delete haveDevice().
- **RejectsInvalidSourceBeforeCommit asserts on const inputs that cannot change** — `tests/sph/star_formation_cuda.cu:221` · `tests_sph_chem_cpp#2` · being edited  
  Two expectations in RejectsInvalidSourceBeforeCommit check fields of the caller's `inputs` vector, which runCuda receives as `Input const*` and only copies to the device. They are tautologies. Nothing checks the failing index or the staged outputs. *Fix:* Remove the two input expectations. Catch the std::runtime_error and check that its message names particle 1, that outputs[1].status == 4 (the prepare id==0 rejection) and that outputs[0].status == 0.
- **test.cpp benchmark modes: dead/no-op checks, misleading output, unchecked fopen** — `tests/test.cpp:667` · `tests_other_cpp#8`  
  test.cpp's manual run.test modes contain no-op or misleading code. bh has an empty body yet reports a GTest pass, kickTest always reports 0 FLOPS, kickTest and forceTest write through unchecked fopen results, forceTest labels a total time as an average and discards its first timer start, and there are unused locals … *Fix:* Remove bhTest and its table entry (bhPotential covers BH). Either accumulate totalFlops (for example from the flop counters) or drop the FLOPS output. Check the fopen results as bucketTest does.
- **units_output check requires the literal CLI argument --hpx:threads=2** — `tests/units_output.cpp:137` · `tests_other_cpp#15`  
  checkUnitsOutputCommandLine passes only when the recorded argv holds the literal tokens "--hpx:threads=2" and "--run.test=unitsOutput", which are what CMake happens to pass. A manual run with a different thread count or token spelling fails even though argv recording is correct. *Fix:* Check the property rather than the value: require some argument that starts with "--hpx:" (an HPX-consumed argument survived) and some argument that names unitsOutput.

## Appendix: refuted findings

Reported by a finder, then refuted by its verifier. They are listed so they are not re-reported; a few have true facts but intentional or unreachable behaviour.

- `analysis_groups_rockstar#15` The 50% bound-fraction denominator counts particles donated by rejected children (`src/analysis/rockstar_catalogue.cpp:543`). Intentional and documented: docs/rockstar.md lets a host reconsider particles rejected by its children, and the inclusive host convention counts them.
- `analysis_groups_rockstar#16` Subgroup label counter (int nextId) is consumed about N per hierarchy level and can overflow or hit kRockstarNoGroup (`src/analysis/rockstar.cu:66`). Each level recurses only into linked groups (about 70% of its particles), so labels total about 2-3 N rather than N times the depth; overflow would need about 6e8 particles in one FoF group.
- `analysis_groups_rockstar#17` group_tree broadcasts use hpx::wait_all without get(), so exceptions from child ranks are swallowed (`src/analysis/group_tree.cpp:400`). In HPX 1.11, which this project builds against, `hpx::wait_all` rethrows exceptions captured by the futures, so nothing is swallowed.
- `analysis_lightcone_bh_power#11` Collectives wait on futures without get(), silently dropping remote and worker exceptions (`src/analysis/lightcone.cpp:420`). In HPX 1.11 `hpx::wait_all` rethrows exceptions captured by the futures; the extra `get()` loops are merely redundant.
- `analysis_lightcone_bh_power#16` Power-spectrum bin sums are accumulated in float (`src/analysis/power.cpp:133`). The float sums are two-level (per slab, then across slabs). A numpy replay at n=2048 gave about 1e-6 relative error, below each bin's statistical error.
- `analysis_lightcone_bh_power#27` GPL-licensed HEALPix source vendored into a BSL-1.0 project (`src/analysis/healpix.cu:11`). Intentional and documented in README.md and THIRD_PARTY_NOTICES.md; GSL and FFTW are GPL dependencies as well.
- `fmm_generators_tools#10` Import validation gaps: host_id type, sys, catalog metadata, unused manifest count; some exceptions escape as tracebacks (`tools/halo_viewer/halo_viewer.py:164`). The only producer, `halo_catalog_export`, already guarantees every property the importer was said not to check.
- `gravity_dehnen_global_fmm#1` Thread-local last-line memo in treeCacheRead is used across an HPX suspension point (`src/gravity/dehnen/remote_cache.cpp:133`). The compiled code in five build trees re-derives the TLS address after the suspension, so a migrated task writes the memo of the worker it now runs on.
- `gravity_dehnen_global_fmm#3` Remote particle request chunk (64 Mi particles) exceeds HPX's default 1 GB parcel limit with fixed64 positions (`src/gravity/dehnen/gpu_staging.cpp:157`). The installed HPX 1.11 MPI parcelport does not enforce `max_message_size` and handles messages over 2 GiB.
- `gravity_nitadori_pmmm#11` A FLOP-accounting check can abort the run (`cosmictiger/gravity/nitadori/fft_flops.hpp:23`). `fftw_flops` returns exact integer counts stored as doubles, so the check that could abort is unreachable.
- `gravity_nitadori_pmmm#12` PMMM stores glass-signed gas gravity, while FMM stores it unsigned (`src/gravity/nitadori/kick.cpp:217`). Unreachable: option validation rejects hydro in the glass scenario, so the two sign conventions never meet.
- `recent_commits_a#5` run.minRung silently ignored by the global integrator (`src/core/driver.cpp:759`). Intentional and documented in docs/pmmm.md and the README: the global integrator does not apply the minimum-rung floor, and `run_global_integrator.py` asserts it.
- `recent_commits_a#7` Old fixed-rung chemistry checkpoints rejected with a misleading 'settings differ' message (`src/io/checkpoint.cpp:820`). Old fixed-rung chemistry checkpoints are rejected earlier, with an accurate 'Monaghan is no longer supported' message.
- `sph_infrastructure#3` thread_local tree-cache entry is updated after a suspending future.get(), so a migrated HPX thread can corrupt another OS thread's entry (`src/sph/sph_tree.cpp:608`). The compiled code reloads the TLS base after `future.get()`, so the update goes to the resuming thread's own cache entry as a matched pair.
- `sph_infrastructure#8` compute.cuda=true in a CPU-only build crashes SPH with a bare ALWAYS_ASSERT(false) (`src/sph/sph.cpp:683`). `processOptions` rejects `compute.cuda=true` in CPU-only builds at startup with a clear message, so the assert is unreachable.
- `sph_physics_kernels#11` Test-only duplicate implementations: tests validate helpers the production kernels do not call (`cosmictiger/sph/volume.hpp:65`). The helpers are an independent test oracle: `variationalReference()` checks the production CPU kernels against them, and CUDA is checked against CPU.
- `tests_gravity_cpp#11` PMMM near-kernel warp/block-boundary loop never checks the computed fields (`tests/gravity/nitadori/pmmm.cpp:315`). `checkPmmmNearBatches` covers the same batch kernel against references at 2e-11 on CPU and CUDA; only the loop's 'warp/block boundary' comment is stale.
- `tests_other_cpp#10` halo_catalog_fixture writes only the legacy 5-key .format; current 7-key format untested (`tests/halo_catalog_fixture.cpp:124`). `run_halo_viewer.py` derives 7-key catalogues from the fixture and tests both the parse and the failure cases.
- `tests_other_cpp#11` compression_test only built and registered in CUDA builds (`CMakeLists.txt:2133`). The compression module has no production caller, so CPU-build coverage protects nothing; the dead module itself is `io_checkpoint_output_tools#11`.
- `tests_other_cpp#12` Compression round-trip tolerance floors at 4e-5 box; no small-extent fixture (`tests/compression.cpp:118`). The tolerance is loose, but the compression module is dead code (`io_checkpoint_output_tools#11`).
- `tests_other_cpp#17` Host allocator test mmaps 2 GiB without MAP_NORESERVE (`tests/cuda_memory.cpp:412`). The mapping commits no physical memory, and `MAP_NORESERVE` would not help under strict overcommit anyway.
- `tests_other_cpp#20` GoogleTest bridge for run.test only runs on locality 0 and turns interrupts into failures (`tests/application_test.cpp:39`). Running on locality 0 is the design, and turning an interrupt into a failure prevents a false pass.
- `tests_python#12` run_mpi_launcher cleanup calls killpg with an application PID that is not a process group (`tests/run_mpi_launcher.py:120`). The rank supervisor starts each application in its own session, so the PID passed to `killpg` is a valid process group.
