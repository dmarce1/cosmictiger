# Bug-fix disposition for the October 30, 2026 review

Updated October 31, 2026. Scope: implement the bugs in
[the original review](code_review_20261030.md), excluding chemistry and star
formation. Existing work in the shared checkout was preserved. Nothing was
staged or committed.

Of the **142 canonical bug findings**, **131 have implementations**, **10 are
open and deferred**, and **one suspected Rayleigh–Taylor seed typo was not
established**. The latter retains the historical multiharmonic seed, now documented
and checked with an independent Fourier expression. “Implemented” describes the
source change; it does not claim every failure scenario has runtime coverage.
The [deferred inventory](code_review_20261030_deferred_chemistry_star_formation.md)
retains aliases, mixed integration concerns and related smells as well as the ten
canonical deferred bugs. Cleanup-only smells are outside this repair pass.

## Material behavior and compatibility changes

- Iteration limits now mean exactly the requested number of additional steps;
  zero performs no evolution. Global-step output cadence follows physical coarse
  boundaries. Distributed drift and final scenario diagnostics include all ranks.
- Lattice initial conditions preserve Fourier modes and use independent Hermitian
  random streams. Corrected initial conditions differ from historical realizations.
- Gas-only checkpoints no longer depend on an unused gravity solver/order. Old
  gas-only metadata remains readable. A new index-width flag deliberately rejects
  historical 64-bit-index checkpoints with truncated rung boundaries.
- Native halo catalogues warn and omit records beyond their 32,768-record limit,
  preserving valid parent references. Such catalogues are explicitly incomplete.
- Lightcone maps now accumulate all batches and persist across new checkpoints.
  Old checkpoints warn because missing historical map contributions cannot be
  reconstructed.
- CUDA device selection uses node-local ranks. HPX still creates contexts on all
  visible devices; per-rank device visibility remains a deployment requirement.

## Validation

- Selected nonchemistry fixed32 CPU application and regression targets build.
  Fixed64/64-bit-index, order9 focused targets and a gravity-free SPH application
  with maximum bucket size 32 also build.
- 204 selected CPU unit cases passed, including the corrected cold-shock fixture
  and its focused rerun. The deferred source-timestep test was excluded.
- All 16 focused fixed64/order9 tests passed: FMM API, periodic geometry/tensors,
  power reference and tiny-separation Ewald derivatives with flush-to-zero enabled.
- Build-feature checking and all 54 build-variant checker tests passed. Generated
  CTest metadata confirms separate cpu/sph/price_etal labels. Both generated FMM
  freshness and FLOP-count tests passed.
- CPU integration passed for cosmological expansion/contraction and restarts,
  fixed scenarios, global timesteps, gravity diagnostics, position selection,
  dark-matter/SPH/Sod smoke runs, Rockstar, checkpoint state and corruption
  handling, fatal-signal reporting (including concurrent faults), and halo viewing.
- MPI checks passed for sampled gravity, PMMM evolution and dispatch failures,
  SPH migration, analysis, and a three-rank FMM tree. A separate two-rank
  hierarchical FMM evolution/restart check passed; checkpoint inspection confirms
  remote-rank positions advance consistently with their velocities. That fixture
  uses a fixed scale factor and rung 1, so it does not validate cosmological
  scaling across a deeper mixed-rung ladder. CPU analysis passed both
  locally and under MPI with the no-CUDA-runtime guard actually loaded.
- Gravity-free fixed64/64-bit-index SPH and global-integrator runs passed,
  including checkpoint/restart and disabled-gravity metadata.
- A synthetic Gadget file imported ten particles despite a configured 8-cubed
  lattice, used its header-derived units/redshift, and selected seven tracers.
  Restart with the Gadget file truncated to its header reproduced the
  uninterrupted particle state and clocks exactly. Inconsistent single-file
  counts were rejected. Remote reader exception propagation was source-reviewed;
  multi-file Gadget metadata agreement was not tested.
- Host allocator, analysis-tool, Rockstar and lightcone-map regressions passed;
  affected CUDA translation units compile, including both SPH hydro stages.
  No GPU kernels or sanitizers were run.
- The full all-target build is not certified: excluded chemistry code fails on an
  undeclared cudaSetDevice, and native recombination tests use an obsolete
  electrons.toDouble() call. Those files remain untouched by the resumed scope.
- No QB logins were attempted. Local builds/tests respected the shared CPU budget.

Logs and the independent coverage audit are retained in
[/tmp/cosmictiger_review_fixes_20261031](/tmp/cosmictiger_review_fixes_20261031).
The original runs exposed test-fixture issues (cold-shock activity selection,
iteration-limit expectations, exact option spelling, profiler output, checkpoint
diagnostic text, and MPI transport flags); corrected cases were rerun successfully.
The generated Ewald freshness check initially found a missing license notice;
restoring that notice changed no numerical code, and both generator tests passed.
The original failing logs remain alongside the successful reruns.

## Canonical ledger

Locations identify the original source family; the original review retains the
historical line numbers and full problem descriptions. Duplicates are not counted
again here.

| Review ID | Status | Finding |
| --- | --- | --- |
| `chemistry_star_formation#2` | OPEN — DEFERRED | Shipped cosmology configs cannot reach z=0: unheated gas falls below the hard 10 K EOS floor near z~21 and the run aborts |
| `recent_commits_b#1` | OPEN — DEFERRED | Per-particle SH03 hazard and wind-decoupling limits are applied as a global SPH timestep cap (wind case drives rungs past kMaxRung) |
| `sph_infrastructure#1` | OPEN — DEFERRED | Hydro reruns after chemistry/star-formation open reuse the RUNGS neighbour lists, which leave out inactive particles whose support reaches an active one |
| `core_options_driver#2` | Implemented | Hierarchical ladder mutates scale/adot: later rung drifts and kicks use an already-advanced scale factor |
| `core_options_driver#3` | Implemented | FMM hierarchical rung drift only runs on the root locality |
| `particles_domain_drift#2` | Implemented | Remote particle position cache is never invalidated in hierarchical FMM runs |
| `core_options_driver#1` | Implemented | numberOfParticles is computed in 32-bit int and overflows for particlesPerDimension >= 1292 |
| `scenarios_configs#2` | Implemented | glass.bin writer/reader mismatch: fixed64 builds write 8-byte positions, closePack writes 2x entries under a side^3 header |
| `cuda_infra#1` | Implemented | DeviceVector never destroys its elements, so nested vectors leak (ROCKSTAR leaks every leaf's neighbor block) |
| `sph_physics_kernels#1` | Implemented | Thermal-energy timestep bound also limits heating, giving dt ~ dt_CFL/M^2 in shocks into cold gas and aborting at kMaxRung |
| `cosmology#1` | Implemented | kz=0 plane random seeds collide with the kz=1 draws and fold each axis separately (correlated, mirror-symmetric ICs) |
| `cosmology#2` | Implemented | DM-only multi-locality ICs: initX0 never runs on non-root localities, so only rank 0's lattice block exists |
| `cosmology#3` | Implemented | closePack without hydro offsets one sublattice by 0.25 box lengths instead of 0.25 cells |
| `cosmology#9` | Implemented | Truncated Fixed32 lattice coordinates shift the M6 stencil one cell below the ghost region at locality boundaries |
| `analysis_lightcone_bh_power#3` | Implemented | lc.healpix map is rewritten by every drift batch, keeping only the last one |
| `build_cmake_scripts#1` | Implemented | Default configuration with BUILD_TESTING=ON does not compile: fmm_walk test calls kickStepHierarchical, which only exists when hierarchical integration is enabled |
| `tests_python#1` | Implemented | run_gravity_error.py checkpoint decoder ignores the runtime-algorithm block (flag bit 18): every gravity_error\_\* CTest fails |
| `io_checkpoint_output_tools#4` | Implemented | Runs started from Gadget4 ICs cannot restart: units are checked against startup options and the Gadget loader re-runs after readCheckpoint |
| `core_options_driver#5` | Implemented | Global integrator runs every 'coarse-step' analysis/output on every global step |
| `core_options_driver#4` | Implemented | Driver silently overrides compute.bucketSize every step with a hard-coded 112->184 schedule (beyond MAX_BUCKET_SIZE) |
| `core_options_driver#7` | Implemented | Cosmological energy expansion term hard-codes gamma=5/3 for non-chemistry SPH |
| `core_options_driver#6` | Implemented | run.maxIterations is off by one in the cosmology driver and inconsistent with the scenario driver |
| `chemistry_star_formation#17` | OPEN — DEFERRED | supernovaTemperature has no upper bound, so dense gas later exceeds the 1e8 K CIE ceiling and aborts |
| `gravity_tree_walk_cpu#2` | Implemented | gravity.openingAngle is never validated; theta <= 0 crashes the tree build with 'Maximum depth exceeded' |
| `recent_commits_a#2` | Implemented | Global fixed scenarios never report completion when the accumulated stage end rounds below run.endTime |
| `analysis_lightcone_bh_power#2` | Implemented | GPU HEALPix lookup tables are initialized only on locality 0 |
| `scenarios_configs#7` | OPEN — DEFERRED | Cosmology INIs and default.ini enable star formation: default builds reject them, and the documented collisionless recipe fails |
| `scenarios_configs#8` | Implemented | Disc velocities hard-code GM=1 and RT relies on an INI-only acceleration; both scenarios run wrong physics without their INI |
| `scenarios_configs#3` | Implemented | Plummer radius sampler can loop forever: the numerical enclosed mass plateaus below the largest uniform deviate |
| `scenarios_configs#1` | Implemented | Polytropic-star stage viscosity (alpha 0 then 0.05) is silently ignored under the shipped viscosityEvolution=morrisAndMonaghan |
| `scenarios_configs#6` | Implemented | Scenario error norms (Sod, blast, polytrope) sample only locality-0 particles after domain decomposition |
| `scenarios_configs#4` | Implemented | Sod (and KH) density ratio depends on resolution; the shipped sod.ini runs 1:0.152, not the documented 1:0.125 |
| `chemistry_star_formation#11` | OPEN — DEFERRED | CPU chemistry batch spawns one std::thread per HPX worker from inside an HPX task (oversubscription, blocks the worker) |
| `chemistry_star_formation#3` | OPEN — DEFERRED | chemistry.maxSubsteps above 8192 is silently ineffective: hard-coded maxAttempts caps the solver |
| `recent_commits_b#5` | OPEN — DEFERRED | Decoupled winds in gas denser than n_H=1e4 cm^-3 are sent to the non-equilibrium solver, which rejects them and aborts the run |
| `core_runtime_infra#3` | Implemented | Fatal-signal handler kills the process when a second thread faults, truncating the first thread's stack trace |
| `cuda_infra#2` | Implemented | CUDA_CHECK evaluates its argument twice, only prints, and is an unbraced if |
| `cuda_infra#7` | Implemented | 2 GiB limits: int byte counts in copyCudaMemory and DeviceVector, and nullptr pool returns that callers do not check |
| `cuda_infra#3` | Implemented | CudaUnorderedMap::rehash leaks the old table and copies every entry instead of moving it |
| `cuda_infra#6` | Implemented | DeviceVector's device-side collective contract breaks for swap/move, nested element types and popBack |
| `cuda_infra#5` | Implemented | DeviceVector copy is a raw memcpy (shallow for element types that own storage), and the pool has no double-free detection |
| `cuda_infra#10` | Implemented | GPU selection uses global rank % device count, and the HPX event-pool warm-up opens a context on every visible GPU |
| `particles_domain_drift#3` | Implemented | Cache-line fetch actions read particle arrays without the lock that guards reallocation |
| `particles_domain_drift#7` | Implemented | Checkpoint writes the ParticleInteger rung boundaries with sizeof(int) |
| `io_checkpoint_output_tools#1` | Implemented | Algorithm block writes a nonzero expansion order for gravity.method.far=none, making the checkpoint unreadable |
| `io_checkpoint_output_tools#2` | Implemented | Restart from an older checkpoint aborts mid-run when the original run already wrote a checkpoint at the same iteration |
| `io_checkpoint_output_tools#6` | Implemented | groups2halofunc counts subhalos as halos and normalizes by a hard-coded 684 Mpc box |
| `gravity_nitadori_pmmm#1` | Implemented | Collective dispatch throws away the exception text and aborts the MPI job on any PMMM prepare/transform/destroy error, even with one locality |
| `gravity_nitadori_pmmm#4` | Implemented | Cell-forest tree arena reserves 4 node slots per source particle (about 1 KB per particle) and caps local particle count near 5.4e8 |
| `gravity_tree_walk_cpu#1` | Implemented | Remote TreeNode deserialization carves multipole records out of the local tree arena via a second, unsynchronized counter |
| `sph_physics_kernels#2` | Implemented | Price et al. switch counts the Hubble-flow deceleration as compression, so alpha saturates at alpha_max in essentially all cosmological gas |
| `cosmology#6` | Implemented | 1/sinc^6 deconvolution is wrong for particles on mesh nodes: near-Nyquist IC modes are over- or under-amplified, and Nyquist is kept in odd derivatives |
| `cosmology#4` | Implemented | Glass tile range uses (int)box.end \* nf (the cast binds first), dropping tiles on multi-locality runs |
| `analysis_groups_rockstar#12` | Implemented | Fetch/refresh of the last group-tree cache line reads past nodes.size() when the arena's final chunk is partial |
| `analysis_groups_rockstar#4` | Implemented | More than 32767 halos in one FoF group aborts the whole run (15-bit haloId, short pid) |
| `analysis_groups_rockstar#1` | Implemented | Spatial FoF never pulls labels from single-particle leaves: zero-volume bounding boxes fail the volume()>0 overlap test |
| `analysis_groups_rockstar#2` | Implemented | CPU phase-space linking writes labels into neighbour leaves without marking them active, so the lastActive filter can lose links |
| `analysis_groups_rockstar#8` | Implemented | Seed assignment indexes subgroups[-1] when no seed distance compares below FLT_MAX |
| `analysis_groups_rockstar#3` | Implemented | Snapshot RockstarRecord positions are not wrapped into the periodic box |
| `analysis_lightcone_bh_power#7` | Implemented | lightcone.mapSize is never validated; bad values index past the HEALPix map |
| `analysis_lightcone_bh_power#4` | Implemented | lcTimeToFlush changes rank 0's nsideValue between lcInit calls |
| `analysis_lightcone_bh_power#5` | Implemented | HEALPix decomposition depends on each locality's own hardware concurrency |
| `analysis_lightcone_bh_power#1` | Implemented | Power-spectrum error pass filters the already deconvolved FFT field a second time |
| `math_numerics#1` | Implemented | boxDistanceSquared decides inside/outside without periodic wrap but measures distance with periodic wrap |
| `math_numerics#2` | Implemented | SymmetricTracelessTensor: operator- returns other - this, and the non-const operator() aliases or overruns for n>=2 in NDEBUG builds |
| `build_cmake_scripts#3` | Implemented | COSMICTIGER_WITH_MAX_BUCKET=32/64 is accepted, but the default compute.bucketSize=128 is then rejected on every run |
| `tests_sph_chem_cpp#1` | OPEN — DEFERRED | App builds the SH03 CUDA kernel with inherited --use_fast_math, but its unit test builds it strict |
| `build_cmake_scripts#2` | Implemented | CPU-only builds register six CTests that pass --compute.cuda=true, so they always fail |
| `build_cmake_scripts#4` | Implemented | Application-wide -ffast-math silently removes the std::isfinite safety checks still left in several sources |
| `fmm_generators_tools#1` | Implemented | Generator docs prescribe retired CMake options, wrong defaults, wrong test names and wrong spherical template argument |
| `tests_gravity_cpp#7` | Implemented | analyticCompare statistics are wrong or fragile, and its result is never asserted |
| `tests_gravity_cpp#4` | Implemented | analyticCompare falls off the end of a non-void function in non-CUDA builds (UB; GCC -O2 crashes) |
| `tests_gravity_cpp#3` | Implemented | fmm_api.cpp hard-codes FMM order 7 and fails to compile when the configured degree range excludes degree 6 |
| `tests_python#2` | Implemented | PMMM checkpoint helper reads the stale viscosity layout (index 4 as CD flag): mixed gas PMMM evolution always fails |
| `tests_other_cpp#1` | Implemented | Power-spectrum test reference hard-codes 2^-32 position scaling; wrong in fixed64 builds |
| `tests_python#3` | Implemented | checkpoint_integrity 'version_one' restart expects success but C++ only accepts v1 headers in fixed32 builds |
| `core_options_driver#14` | Implemented | timestep.txt fopen unchecked; NaN inverse timesteps hit ALWAYS_ASSERT(false) |
| `core_options_driver#23` | Implemented | Restart silently ignores a changed cosmology.endRedshift |
| `particles_domain_drift#16` | Implemented | Tracer selection uses a particle total taken from options |
| `core_options_driver#10` | Implemented | int(tau / t0) % 8 misidentifies the coarse step about 4.5% of the time |
| `core_options_driver#9` | Implemented | Final view index is off by one after an interrupt at a coarse boundary and in global mode |
| `core_options_driver#8` | Implemented | run.minRung (and several other options) are never range-checked; out-of-range minRung indexes rungDt out of bounds |
| `core_options_driver#16` | Implemented | Plummer scenario silently discards explicit gravity radius options (magic constant) |
| `core_options_driver#11` | Implemented | rho.dat maximum density only covers the root locality's particles |
| `scenarios_configs#5` | Implemented | Blast energy deposit is not normalized on the particle lattice, while the Sedov reference assumes E=1 |
| `scenarios_configs#10` | Implemented | Lane-Emden derivative at the origin returns -3 instead of -1/3 |
| `scenarios_configs#14` | Implemented | RT lattice size truncates a cube root that sits just below an integer, losing 37% of the nominal particles |
| `tests_other_cpp#9` | Not established; documented/tested | Rayleigh-Taylor perturbation test copies the production formula; bracket placement looks wrong |
| `scenarios_configs#13` | Implemented | polytropicStar silently runs without self-gravity when gravity is unavailable or disabled |
| `scenarios_configs#11` | Implemented | Scenario reference solutions and initial conditions hard-code gamma=5/3 instead of using or validating hydro.gamma |
| `scenarios_configs#12` | Implemented | sedov::solution interpolates on a grid spacing that differs from the computed table; cache ignores rmax and ndim |
| `chemistry_star_formation#10` | OPEN — DEFERRED | New CUDA entry points never call cudaSetDevice(); batches run on whatever device the current OS thread has |
| `core_runtime_infra#15` | Implemented | Bounds-checked Vector::operator[] casts to std::vector<T>\*, ignoring Alloc (UB for Vector<T, PinnedAllocator<T>>); size_t printed with %i |
| `core_runtime_infra#4` | Implemented | Assertion text goes to unflushed std::cout and is lost on abort() when stdout is redirected; the SIGABRT handler then writes a second trace |
| `cuda_infra#4` | Implemented | CudaUnorderedMap iterator, erase() and clear() are wrong (unused today) |
| `cuda_infra#12` | Implemented | Pool blocks are only 8-byte aligned (every other block is 8 mod 16) |
| `cuda_infra#8` | Implemented | Pool or malloc is chosen on every call from the mutable compute.cuda option, not recorded per allocation |
| `particles_domain_drift#10` | Implemented | SPH migration leaves received particles in nondeterministic arrival order |
| `particles_domain_drift#4` | Implemented | drift() silently narrows the scale factor, dt and conformal times to float |
| `particles_domain_drift#8` | Implemented | CUDA drift enables the lightcone for tauMax <= 0, unlike the CPU path |
| `particles_domain_drift#17` | Implemented | 32-bit drift rounds every position down at each step |
| `tests_other_cpp#7` | Implemented | Lightcone crossing uses post-drag velocity while displacement uses pre-drag; test enshrines it |
| `particles_domain_drift#11` | Implemented | Grown particle slots keep stale or uninitialized lastgroup and tracer values |
| `particles_domain_drift#19` | Implemented | Misleading messages and leftover debug output in particle I/O and drift |
| `io_checkpoint_output_tools#15` | Implemented | readCheckpointSelection resolves the checkpoint path differently from readCheckpoint in run.test mode |
| `io_checkpoint_output_tools#14` | Implemented | Tracer/slice Silo writers and view.N.dat ignore I/O errors |
| `io_checkpoint_output_tools#7` | Implemented | Slice image is transposed: pixels stored y-fastest but written to Silo as x-fastest |
| `io_checkpoint_output_tools#9` | Implemented | groups2halofunc crashes on degenerate catalogues, and debug prints corrupt its stdout output |
| `io_checkpoint_output_tools#8` | Implemented | healpix2silo flips the map north/south and trusts the input header |
| `gravity_dehnen_global_fmm#7` | Implemented | CPU and CUDA Ewald acceptance clamp the near-distance floor differently |
| `gravity_dehnen_global_fmm#8` | Implemented | Global-FMM Ewald leaf criterion can be unsatisfiable, causing infinite subdivision and a misleading 'Maximum depth exceeded' |
| `gravity_dehnen_global_fmm#15` | Implemented | Staging exception paths leave tasks that capture stack locals still running |
| `gravity_cuda_kick#2` | Implemented | PC/PP sink base index truncated to int while doKick uses ParticleInteger |
| `gravity_tree_walk_cpu#12` | Implemented | Dead and uninitialized fields in KickParameters and TreeCreateParameters |
| `gravity_cuda_kick#3` | Implemented | All 32 lanes read-modify-write the same walk-phase word without lane election |
| `gravity_cuda_kick#7` | Implemented | Register residency limit ignores the four register sub-partitions, so it overestimates (9 vs 8 at order 4) |
| `gravity_cuda_kick#8` | Implemented | Static launch buffers are left dangling if a reallocation fails |
| `gravity_tree_walk_cpu#9` | Implemented | KickWorkspaceTreeIdHash multiplies in int and can overflow |
| `gravity_cuda_kick#6` | Implemented | close/direct interaction counters are rank-0, GPU-only, but printed as run-wide statistics |
| `gravity_tree_walk_cpu#5` | Implemented | Global-FMM midpoint split has no coincident-particle fallback; more than bucketSize identical positions recurse to kMaxDepth and throw |
| `sph_infrastructure#5` | Implemented | Early return on an empty checklist at a global node skips the neighbour-list reset of descendant localities |
| `sph_infrastructure#2` | Implemented | Bitwise OR instead of AND makes every interaction-selection test true |
| `sph_infrastructure#7` | Implemented | Staging futures are not drained on exception, leaving tasks with dangling references to stack atomics |
| `cosmology#10` | Implemented | twolptPhase loops with int over an int64 local volume |
| `analysis_lightcone_bh_power#13` | Implemented | Lightcone nside cap uses a radius normalized by tauMax, taken at the outer edge of the interval |
| `analysis_lightcone_bh_power#17` | Implemented | CUDA lightcone kernel reads a label that lane 0 writes in the same step without warp synchronization |
| `analysis_lightcone_bh_power#14` | Implemented | Power error estimate counts conjugate-duplicate modes of the kz=0 plane as independent |
| `math_numerics#5` | Implemented | Physical constants disagree: kH0 uses Mpc = 3.086e24 cm while lengths use kMpcToCm = 3.085678e24, and several modules keep their own constant sets |
| `math_numerics#10` | Implemented | distance(double, double) never terminates for infinite or very large separations |
| `math_numerics#7` | Implemented | rand1() can return exactly 1.0f and has only float (24-bit) resolution |
| `math_numerics#4` | Implemented | Range::shortestDim returns an uninitialized index when every span exceeds 1; split() and longestDim have the same pattern for degenerate boxes |
| `build_cmake_scripts#8` | Implemented | The project's `native` CUDA-architecture default overrides the standard CUDAARCHS environment variable |
| `build_cmake_scripts#6` | Implemented | Multi-label LABELS passed to cosmictiger_add_gtest are split by GoogleTest discovery, so every label after the first is lost |
| `build_cmake_scripts#7` | Implemented | Some multi-locality tests are not gated on the HPX MPI parcelport, and none check the available MPI slots |
| `build_cmake_scripts#12` | Implemented | SIMD benchmark pipeline (prepare_simd.py and run_simd.py) cannot work against the current tree |
| `build_cmake_scripts#20` | Implemented | summarize_ncu_source.py double-counts inlined code; NCU helper hard-codes the ncu path |
| `build_cmake_scripts#9` | Implemented | build.sh takes the default CUDA architecture from CPU-only build caches |
| `fmm_generators_tools#16` | Implemented | Nearest-image Ewald radial terms may underflow in float with FTZ for tiny non-zero separations (order 9) |
| `fmm_generators_tools#3` | Implemented | CSV export error after headers are sent appends a second HTTP response into the download |
| `tests_other_cpp#4` | Implemented | cuda_memory stub maps ALWAYS_ASSERT to assert(), which is compiled out in Release (NDEBUG) |
| `tests_python#6` | Implemented | PMMM helper decodes the Cullen-Dehnen viscosity tail as two separate arrays; writer interleaves per particle |
| `tests_python#13` | Implemented | run_smoke does not clear checkpoint.N.incomplete staging directories, so one killed run blocks all later runs |
