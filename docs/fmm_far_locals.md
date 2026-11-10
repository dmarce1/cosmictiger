# Cartesian downward evaluation of completed mesh locals

Part 4 of `spherical_fmm_tree_plan.md` converts each completed spherical far
local once per occupied active sink cell and descends the same owned kd-tree
used for source deposition and near interactions. Step 5 now uses float within
the tree, with double conversion and mesh-to-root translation; see
[spherical_fmm_validation.md](spherical_fmm_validation.md). The supported PP and far-only
paths retain direct spherical particle evaluation. Frozen-FMM remains unsupported.

## Numerical boundary

The pure local map is the derivative dual map in `fmm_basis_conversion.md`.
For mesh width h, tree scale κ=`kScaleFactor`, and mesh background βₘ, the
production helper `gravity/fmm/tree_local.hpp` first adds 2βₘ to L₂₀, maps
spherical harmonics to Cartesian derivatives, and scales rank n by
κ⁻¹(κh)⁻ⁿ. This constructs the harmonic proxy H′ in tree units at the mesh
center. It carries the known residual curvature K=-3βₘ/(κ³h²) as a solve
context, with no per-local trace coefficient.

Each downward center shift d uses ordinary Cartesian L2L plus K d_z in D_z
and K d_z²/2 in D₀. At a sink with tree offset y, Cartesian L2P is completed
with K y_z²/2 in potential and -K y_z in z acceleration. Potential is then
multiplied by κ and acceleration by κ². All center and particle differences
are computed in native fixed-point coordinates before conversion. Completed
mesh locals, basis conversion and mesh-to-root translation use double. Each root
local is then narrowed once; child L2L, particle L2P, displacement conversion and
quadratic completion use float. The known curvature is narrowed once per batch.
The Step 5 report records the error and resource measurements for this boundary.

The independent source radial moment and its existing potential constant
correction are unchanged. Near fields are harmonic contributions and never
increment the background context. The final field adds the existing float
near result to the float far result after promoting both to double, then calls
`consumeField` exactly once.
That is the single gravity scaling and kick boundary. Force-only helper calls
preserve the caller's potential slots; the production combined field sets its
unused potential to zero.

## Selection, scheduling and memory

Near batches retain their existing readiness-driven execution before the FFT
finishes. Each bounded near batch then waits for `farReady`, reads each needed
mesh local, and descends the original owned tree. Its sorted selected particle
indices are reused, even if a near selection split a mixed active/inactive
source leaf into temporary sink-only leaves. No second sink selection occurs,
no additional tree arena nodes are allocated, and inactive particles are not
evaluated or kicked.

The descent prunes subtrees that contain no remaining selected index. It keeps
one q² float local per recursive level and consumes leaf fields immediately;
there is no extra per-particle far array. In addition to the existing bounded
near batch, live storage is O(q² × tree depth), plus the one returned spherical
local and two double boundary temporaries. The existing batch particle limit may be
exceeded by one indivisible dense cell, as before; no N-sized scratch has been
introduced. `gravity.pmmm.cartesianFarDescent` measures conversion, descent,
combination and consumption after any remote local-expansion read completes.

The readiness graph is unchanged: source producers do not wait for far fields,
deposition does not wait for near consumers, and failure releases far waiters
before all users drain and the arena is destroyed. A far-local observer tests
both exactly-once conversion and an injected exception followed by a successful
solve.

## Historical Part 4 validation and schedule comparison

The measurements below describe Part 4 before Step 5 changed the dedicated
tree descent from double to float. They remain the schedule-selection baseline;
current precision and broader validation are recorded in the Step 5 report.

The dedicated schedule and an experimental combined walk were measured
separately. The candidate seeded the ordinary float near walk with H′ at the
mesh center after FFT, then applied the residual using the original mesh z
coordinate. It passed the full numerical PMMM regression with unchanged
tolerances; its deliberately incompatible before-FFT gate was omitted. The
candidate source, patch and executable are retained under
`/tmp/cosmictiger_far_locals_validation/`; it is not a runtime selector.

[fmm_far_locals_benchmark.json](fmm_far_locals_benchmark.json) records three
alternating runs for each schedule and population. Both cases use order 7,
width 2, two traditional Plummer kicks, two HPX workers pinned to logical CPUs
0–1, Release optimization, CPU near interactions, and no GPU. Part 3 is the
saved executable immediately before this part. The timed dedicated binary
precedes the final counter-only accounting correction and test-only checks; its
numerical arithmetic and scheduling are the selected implementation.

| Population / mesh side | Part 3 elapsed | Dedicated elapsed | Combined elapsed |
| --- | ---: | ---: | ---: |
| 32,768 / 16 | 8.133 s | 8.126 s | 8.130 s |
| 4,096 / 32 | 6.831 s | 6.829 s | 6.781 s |

These are medians of application elapsed time. The corresponding summed PMMM
coordinator phase medians are 7.54/7.53/7.51 s and 6.23/6.21/6.19 s, respectively.
Dense-case elapsed ranges are 8.130–8.181, 8.081–8.127, and 8.079–8.131 s;
mesh-case ranges are 6.827–6.875, 6.826–6.831, and 6.778–6.830 s.
Median peak RSS differs by less than 0.3 MiB between schedules in each case
(roughly 314–317 MiB total). The new depth-sized locals do not create a visible
whole-process memory penalty at this scale.

Dedicated conversion/descent/consumption APEX interval totals are 99.4 ms for
the dense case and 508.4 ms for the mesh case. Combined seed intervals are
81.5 and 505.9 ms, with residual/consumption intervals of 8.4 and 1.2 ms.
These task intervals overlap and nest; they must not be added to coordinator
wall time. The candidate's seed annotation also contains subsequent checklist
setup. Coordinator phase rows have 0.01-second resolution.

The dedicated schedule was retained. Whole-solve time and peak memory are
essentially equal in the dense case. The combined median is 0.7% lower in the
mesh case, with overlapping run ranges; that small experiment does not establish
a general scheduling advantage. The dedicated path preserves demonstrated near
progress before far readiness and double far arithmetic without a material
whole-solve or memory penalty in these cases. No speedup or universal scheduling
superiority is claimed.

Part 4 validation on November 9, 2026 (superseded where extended by Step 5):

- All q=3..9 standalone conversion tests pass 312,914 checks. The largest
  scaled discrepancy is 5.37e-7 across the float/SIMD-inclusive suite. The new
  production helper checks each local basis against an independent finite
  Legendre polynomial and analytic derivatives, four mesh widths, nested
  shifts, both fixed widths, and potential sentinels. Explicit quantum-scaled
  displacement checks detect losing Fixed64 offsets below double absolute
  coordinate resolution.
- The Release fixed32/order-7 application builds. Six focused CTests pass:
  `basis_conversion_order_7`, `pmmm_cpu`, `pmmm_sources_cpu`, `pmmm_near_pp`,
  `pmmm_evolution_cpu`, and `pmmm_interval_cpu`. They cover unchanged
  finite-expansion/Ewald tolerances, dense multilevel trees with bucket size 8,
  mixed active leaves, stages and signed kicks, force-only output, PP/far-only
  fallback, restart, frozen intervals, repeated solves, and failure cleanup.
- The source audit verifies exactly one far-local conversion per occupied
  active cell. Both source-publication and far-local exceptions propagate, and
  the following solve reproduces its reference. An eight-isolated-cell fixture
  proves near completion while an owned producer still prevents FFT readiness;
  its bounded timed wait can add teardown time, unrelated to solver timings.
- Full PMMM tests pass with two localities × two HPX workers and four
  localities × one worker, with all descendants restricted to CPUs 0–1. This
  includes the existing deterministic delayed-ghost overlap gate. PMIx socket
  creation was denied inside the sandbox; the same tests passed outside that
  restriction without source changes.
- NVCC sm_86 compiles all new helpers at q=3..9 and both fixed widths. A bounded
  host/device parity executable passes 224 checks with maximum discrepancy
  5.55e-17, including ordinary/tiny offsets and potential on/off. The RTX 3090
  reports 82 SMs; eight serialized launches each used one block × one thread,
  occupying at most one SM. No full application CUDA near path was executed.
- Formatting and whitespace checks pass. No complete application-suite,
  sanitizer, hosted-CI, full fixed64 application runtime, or broad performance
  sweep is claimed; those remain outside this part's evidence.

Raw commands, logs, timing binaries and candidate sources are retained under
`/tmp/cosmictiger_far_locals_validation/`. Initial overlapping build invocations
were both pinned to CPUs 0–1; later builds and all validation were serialized.
Earlier Ninja metadata warnings recovered; the successful final compile/link
and runtime checks provide the build evidence. No branch change, commit, push,
email or QB access occurred.
