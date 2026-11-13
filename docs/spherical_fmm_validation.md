# Spherical mesh / Cartesian tree validation

> Historical record of the retired PMMM implementation. Its spherical kernels,
> basis conversions, and mesh/tree adapters have been removed. The descriptions
> and measurements below apply to the source version recorded here; see the
> [Cartesian global FMM documentation](generated_fmm.md) for the active solver.

Step 5 completed November 9, 2026, in the workspace checkout on `main`, based
on `61c1e0ea3057c6652923ecdddc9a05a757c1b729`. Parts 1–5 remain uncommitted.
The selected implementation reuses owned Cartesian roots for spherical source
deposition and the existing selected sink trees for far evaluation. Dipoles are
mandatory; the known local trace stays implicit. PP and far-only solves retain
the direct spherical evaluator. Frozen-interval FMM remains unsupported.

Float tree L2L/L2P passes the tested accuracy envelope. Double remains at the
mesh boundary, in the root recentering, and in the upward source construction.
Whole-application timings show no material dense-case change and a small sparse
median reduction with overlapping ranges. No general speedup is established.

Raw timings, counters, options, executable hashes, build capabilities and the
validation manifest are in
[spherical_fmm_validation_results.json](spherical_fmm_validation_results.json).
Detailed commands and logs are retained in `/tmp/cosmictiger_step5/`.

## Precision and background mathematics

Use the normalization derived in [historical basis-conversion derivation](https://github.com/dmarce1/cosmictiger/blob/38f59a72ab08e2355017728e60b338b352f1cd05/docs/fmm_basis_conversion.md):
physical offset r, tree coordinate y=κr, mesh coordinate u=r/h, κ=1000,
and mesh background coefficient βₘ. The derivative-dual map B gives

\[
H'=H+2\beta_m U_{20},\qquad
C_n=\kappa^{-1}(\kappa h)^{-n}B_n L_n,\qquad
K=-\frac{3\beta_m}{\kappa^3h^2}.
\]

The true local is Φ=H′+Kz²/2 in tree units. Recenter the harmonic proxy by d
using ordinary Cartesian L2L, then add Kd_z to its z derivative and Kd_z²/2
to its constant. Evaluation adds Ky_z²/2 to potential and −Ky_z to z
acceleration. Physical potential and acceleration receive factors κ and κ².
No local trace slot or additional FFT channel is introduced. The independent
source radial second moment still supplies its existing potential constant.

The implemented arithmetic boundary is

\[
C^{(f)}_{\rm root}=\operatorname{round}_f
 \left[\operatorname{L2L}_d(C_{\rm mesh},d_{\rm root};K_d)\right],
\quad
C^{(f)}_{\rm child}=\operatorname{L2L}_f
 (C^{(f)}_{\rm parent},d_f;K_f).
\]

Both mesh conversion and mesh-to-root translation use double. Each root local
narrows once. Child translations, leaf evaluation, physical-unit factors and
quadratic completion then use float, with curvature narrowed once per batch.
Fixed32/Fixed64 coordinates are subtracted in their native representation
before conversion; small Fixed64 offsets are not formed by subtracting rounded
absolute floating-point coordinates. Near and far float fields promote to
double for their final sum, followed by one `consumeField` call and one kick.
The downward pass is on the CPU even when the near walk uses CUDA. The existing
CUDA near walk already stores and operates on float Cartesian locals.

Source P2M/M2M, FFT coefficients and root deposition retain Part 3's double
policy. Persistent near-tree moments remain float. No second sink selection,
extra tree arena nodes, public precision/backend option or permanent double
tree has been added. FP32 and FP64 work counters follow the arithmetic split.

## Accuracy and lifecycle evidence

All entries below passed. Counts describe the named batches and overlap;
they are not a count of distinct tests across the project.

| Scope | Evidence |
| --- | --- |
| Generated operators / conversions | Generated-current check; 161 executed weighted FLOP checks and 35 minimum selections over q=3…9; 358,218 scalar/SIMD and production-helper checks over q=3…9, float/double and both fixed widths |
| Helper accuracy | Independent finite Legendre polynomials and analytic derivatives, every basis coefficient, four mesh widths, nested shifts, known curvature, native quantum offsets, potential on/off. Largest scaled discrepancy 5.368e−7 in the float-inclusive suite; unchanged double budget 3e−11, separate new float budget 2e−6 |
| Focused fixed32 CPU | 7/7 source-moment, conversion, generated-code and PMMM checks; final post-fixture batch 3/3 |
| Wider fixed32 CPU | 55/55 Cartesian/API/acceptance/geometry, spherical/Ewald, PMMM kernel/FLOP, PP fallback, mesh domains, evolution/restart, hierarchical gas, frozen interval, sorting/timestep and checkpoint checks |
| Fixed64 full application | 14/14 full PMMM runs: FMM and PP at every q=3…9; order-9 evolution/restart and checkpoint mismatch rejection also pass |
| True MPI | Full fixed32 q7 PMMM at 2 localities × 2 workers and 4 × 1; full fixed64 q3 and q9 PMMM at 2 × 2 |
| SPH / outer intervals | Distributed hierarchical gas evolution and one/two-rank outer-interval restart tests, including far-only energy/source membership and gas index maps: 2/2 |
| Gravity-disabled build | Release fixed32 SPH application builds; build-features, startup-options and SPH-tree tests 3/3. No FFTW/GSL dependency in `ldd` |
| CUDA helpers | q=3…9, both fixed widths, float/double descent, ordinary/tiny offsets and potential on/off: 448 host/device checks, maximum discrepancy 5.960e−8 |
| Full CUDA application | Fixed32/q7 full PMMM with FMM near and PP near, including dense CPU/CUDA comparisons; CUDA drift/evolution/restart/legacy/checkpoint mismatch tests pass |

PMMM regression includes unequal masses, clustered multilevel trees, sparse
active rungs and mixed active leaves, mesh sizes and near widths, signed/staged
kicks, force-only calls and PP/far-only fallbacks. Existing independent finite
expansion and full Ewald oracles remain. Audits check exactly one far conversion
per active occupied sink cell, no inactive kicks, streamed force consumption,
delayed-ghost and pre-FFT progress, source/far exceptions and successful recovery.

The all-order run exposed two fixture assumptions, reproduced with the saved
Part 4 **double** descent using the same CPU allocation before changing tests:

- The eight-particle traditional-kick fixture places particles at side-16 cell
  centers but requested mesh side 8. It now requests mesh side 16, removing a
  low-order expansion truncation artifact from this integrator test.
- The loose-angle FMM test now uses θ=0.2 for q3's dipole-only source expansion;
  higher orders retain θ=0.4. Its 2% acceptance tolerance and the θ=0.02
  convergence checks remain unchanged.

No established numerical tolerance was loosened. Initial failed matrices,
baseline reproduction logs and the corrected matrix are retained separately.
Local GTest sources were reused after a network fetch failed. MPI socket
creation required execution outside the sandbox; the same source then passed.

## Complete-solve measurements

Three balanced run rotations compare source reuse alone (direct spherical
particle evaluation), Part 4's dedicated double descent, and the current float
descent. These are real application runs with two traditional Plummer kicks,
q7, width 2, CPU near interactions, Release optimization, and two HPX workers
pinned to logical CPUs 0–1. Each case/variant has three runs. Timings include
startup, cold mesh/Green preparation, solve work and teardown; they are not
kernel microbenchmarks. The saved binaries precede only test/counter changes
where noted in the raw manifests; their solver arithmetic matches the named
variant.

| Particles / mesh side | Source reuse only | Double descent | Float descent |
| --- | ---: | ---: | ---: |
| 32,768 / 16 | 8.1275 s | 8.1294 s | 8.1272 s |
| 4,096 / 32 | 6.8764 s | 6.8266 s | 6.7809 s |

These are whole-process wall-time medians. Dense ranges are respectively
8.1267–8.2268, 8.0806–8.1316 and 8.0269–8.1774 s; sparse ranges are
6.8269–6.9253, 6.8247–6.8286 and 6.7786–6.8302 s. Float's sparse median is
0.67% below double, with overlapping ranges. Median RSS spans 316.69–317.18 MiB
in the dense case and 314.06–314.86 MiB in the sparse case. These runs establish
neither a general speedup nor a measurable whole-process memory saving.

The float coordinator phase medians locate the dominant costs: dense
`finishNearPipeline` 6.78 s, `prepare` 0.71 s and `transform` 0.01 s; sparse
`prepare` 5.60 s, `finishNearPipeline` 0.51 s and `transform` 0.07 s. Preparation
includes cold mesh/Green work; the FFT transform itself does not dominate these
runs. Coordinator rows have 0.01-second resolution.

APEX task-interval medians give more detail (milliseconds):

| Task / workload | Source only | Double | Float |
| --- | ---: | ---: | ---: |
| Owned cell-tree construction, dense | 26.202 | 26.781 | 25.791 |
| Owned source deposition, dense | 3.856 | 3.691 | 3.682 |
| Far conversion/descent/consumption, dense | — | 99.391 | 98.753 |
| Owned cell-tree construction, sparse | 3.747 | 3.667 | 3.663 |
| Owned source deposition, sparse | 1.710 | 1.639 | 1.669 |
| Far conversion/descent/consumption, sparse | — | 507.497 | 507.902 |

These task intervals overlap/nest, include suspended-task accounting, and must
not be added to coordinator wall time. The far annotation includes the final
combination and consumption; there is no independently measured combination
wall time. No per-particle timing instrumentation was added to production.

### Occupancy, active fraction and communication

The optional `tests/gravity/pmmm_profile.cpp` replaces only the test payload at
link time. It invokes complete production PMMM solves, without a production
selector. It sweeps mesh sides 8/16, 32 particles in 32 occupied cells versus
320 particles in four occupied cells, active fractions 1, 1/4 and 1/16, and
force-only versus potential requests: 24 combinations. Every combination has
one warmup and three warm repeats in each of three balanced process rotations
per schedule: 648 timed warm solves plus 216 warmup/first solves. Cold flags
record actual Green rebuilds rather than assuming every fixture starts cold.

Float/double ratios of the 24 warm medians range 0.950–1.043 with median 0.993;
all 24 run ranges overlap. The tiny cases include observer overhead, so these
numbers do not establish a precision throughput advantage. Source depths are
1 and 9, active particle counts 2–320, and active cell counts 2–32 or four.
The counters exclude inactive cells from pre-FFT progress. An additional true
2-locality run records all 96 rows: sparse-cell ghost payload is 0–288 bytes,
clustered-cell payload 4,320 bytes. All clustered active cells complete near work
before FFT readiness in its 48 warm/first measurements; sparse occupancy has
variable overlap. Deterministic overlap gates in the regression suite remain
the correctness check, rather than imposing a timing expectation on every run.

## Storage and compiler resources

At q7, each recursive coefficient local is 196 bytes rather than 392 bytes.
There are also two double root-boundary temporaries, the returned mesh local,
ordinary stack frames and the existing near results. A conservative coefficient
bound for a depth-9 example is 9×196+2×392=2,548 bytes versus a depth-plus-one
double bound of 10×392=3,920 bytes, excluding the common returned mesh local.
Source depth is an upper bound on the selected traversal, not a measured stack
high-water mark. Compiler frame layout may reuse temporary storage.

Near scratch remains approximately
`(4*sizeof(float)+sizeof(ParticleInteger))*peakBatchParticles` per active worker,
plus existing near-walk storage. The occupancy probe records peak batches up
to 240 particles. One indivisible dense cell can exceed the configured batch
target, as before. There is no new N-sized far array or stored trace channel.
Streamed solves assert zero retained diagnostic acceleration/potential arrays.

GNU `size` reports CPU q7 text 38,729,628 bytes for the saved double baseline
and 38,732,638 for current float, a 3,010-byte difference including test changes;
data 658,408 and BSS 687,912 are identical. CUDA q7 text/data/BSS are
37,675,511/518,512/380,248; the minimal fixed64 multiorder build reports
22,509,085/369,424/344,648. Those latter builds have different capabilities and
are not direct size comparisons. `cuobjdump --dump-resource-usage` reports
255 registers and 1,200 bytes thread stack for the current q7 `cudaKickKernel`,
with zero static shared bytes. Its dynamic arena is not included in that
static-shared figure; no pre-change GPU resource or performance comparison is
claimed.

## Reproduction and validation limits

The JSON manifest records exact build/test argument vectors and CPU affinity;
the log stem is `/tmp/cosmictiger_step5/<name>.log`. Application build directories
are `build/pmmm_only_cpu`, `build/spherical_step5_fixed64`,
`build/spherical_step5_no_gravity` and `build/bounded_target_cuda`. All are
Release; application degree 2…8 corresponds to order q3…9. Single-order builds
use degree 6 (q7). The fixed64 and gravity-disabled configurations reused local
GTest source via `FETCHCONTENT_SOURCE_DIR_GOOGLETEST` and disabled downloading.

The profile comparison patches are
[spherical_fmm_profile_double.patch](spherical_fmm_profile_double.patch) and
[spherical_fmm_profile_source_only.patch](spherical_fmm_profile_source_only.patch).
They are diagnostic artifacts, not production selectors. To reconstruct a
comparison, apply a patch to a temporary copy of `cell_schedule.cpp`, compile
with that translation unit's `compile_commands.json` flags plus the original
source directory as `-iquote`, and replace only its object in the application's
Ninja link command. Compile `pmmm_profile.cpp` with the `pmmm.cpp` flags, then
add that object and `-Wl,--wrap=_Z8pmmmTestv` to each comparison link. The exact
link vectors and harness sources used here are retained under the scratch root;
source hashes and argument vectors are also in the results JSON. Run the linked
binary with `--run.test=pmmm`, CPU gravity, traditional integration, two HPX
workers and `--hpx:bind=none`, under CPU affinity 0–1. The fixture emits
`PMMM_PROFILE` JSON rows. `whole_solve` and `profile_sweep` sections retain the
full application options, row samples and executable hashes.

War Thunder remained ON. Builds/tests used at most four assigned logical CPU
cores in aggregate, beneath the shared six-core ceiling. MPI descendants were
pinned to CPUs 0–1. Full CUDA tests used the optional validation-only preload
`tests/gravity/cuda_workload_guard.cpp`: serialized launches, explicit device
synchronization, at most 16 blocks on an 82-SM RTX 3090, and rejection of unknown
oversized or extended launches. Only the audited atomic-work-queue FMM kernel
and grid-stride PP kernel may have their grids reduced. Thus these tests occupy
at most 16 SMs (19.5%); they are not unrestricted GPU performance measurements.
The real CUDA 13 kernel-handle launch entrypoints were intercepted: FMM 284
launches/3 capped/max16 blocks, PP 20/0/max5. The separate parity test used 16
serialized one-block/one-thread launches. GPU/MPI execution outside the sandbox
was needed for device/socket access. All reservations are released.

Full CUDA application coverage is fixed32/q7 on one locality. CUDA helpers span
all orders and both fixed widths; a full fixed64/all-order CUDA application or
GPU MPI run was not performed. This is a focused affected integration matrix,
not the entire project suite, sanitizer run, hosted CI or an independent review.
No QB access, email, branch switch, commit or push occurred. Existing unrelated
workspace dirt and handoff files remain unstaged.
