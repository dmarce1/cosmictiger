# Order-aware CUDA kick shared-memory policy

This document describes the runtime-selectable FMM order, the shared-memory
arena of the CUDA kick kernel, how its budget is chosen, and what was measured.
Sections marked *measured* report numbers from this workstation (RTX 3090,
compute capability 8.6, CUDA 12.8, GCC 13.3); everything else is design.

## FMM order

The FMM order equals the multipole degree p plus two: source multipoles carry
degree p, local expansions degree p + 1, and the Ewald derivative tensors ranks
0 through p + 1. Orders 3 through 9 are supported. Order 3 (dipoles with
quadratic expansions) is the smallest order whose local expansion can hold the
periodic uniform-background quadratic; a degree-1 expansion keeps only the
z-squared Laplacian slot of that term, so order 2 is not offered. Order 9
(degree-7 multipoles, degree-8 expansions) is the largest: the CUDA single
precision parity test showed order 10's host/device roundoff gap (6.9e-6)
approaching the 2e-5 budget, so single precision stops gaining there.

| order | multipole coefficients | positioned multipole bytes | local coefficients | local expansion bytes |
|---:|---:|---:|---:|---:|
| 3 | 4 | 28 | 10 | 40 |
| 4 | 9 | 48 | 17 | 68 |
| 5 | 16 | 76 | 26 | 104 |
| 6 | 25 | 112 | 37 | 148 |
| 7 | 36 | 156 | 50 | 200 |
| 8 | 49 | 208 | 65 | 260 |
| 9 | 64 | 268 | 82 | 328 |

Every order is compiled into the binary and the run selects one with
`gravity.fmmOrder` (default: the build's `COSMICTIGER_WITH_ORDER`, 7). The
order-dependent code is instantiated per order from templates:

- `cosmictiger/fmm_order.hpp`: constexpr coefficient counts, byte sizes, the
  `FmmOrder<order>` traits, the runtime `fmmOrder()` and `fmmDispatch()`, which
  switches to the matching instantiation at coarse boundaries (tree build, a
  CPU walk, a CUDA launch). Inner loops never test the order.
- `cosmictiger/fmm_types.hpp`: `FmmMultipole<Real, order>`,
  `FmmExpansion<Real, order>`, the position-first multipole record
  `FmmMultiPos<order>` (tree nodes point at its order-agnostic
  `MultipolePosition` header), and the max-size transport buffers
  `ExpansionBuffer` / `MultipoleBuffer` used by order-agnostic interfaces.
- `cosmictiger/fmm_kernels.hpp`: operator wrappers that deduce the degrees from
  their tensor arguments over the generated STF kernels.
- Generated code: `tools/generate_stf_m2l.py` emits one fully specialized
  kernel class per degree for local degrees 2 through 8 into a single header set
  in the build tree (`ninja cosmictiger_generated_kernels`), and
  `src/fmm_kernel_codegen.cpp` emits one Ewald derivative overload per order.
  Parsing the whole set costs about two seconds with g++ and three with nvcc.
- Per-order translation units: CMake configures
  `instances/fmm_instance_cpu_<order>.cpp` and `fmm_instance_cuda_<order>.cu`
  from `src/fmm_instance_*.in`; they explicitly instantiate the heavy CPU and
  CUDA templates so the orders compile in parallel.
- Ewald tables (`EwaldConst`) are filled once for the runtime order into
  buffers sized for order 9 (about 29 KB of constant memory).

## Bucket cap

`COSMICTIGER_WITH_MAX_BUCKET` (power of two, default 128) caps the runtime
`compute.bucketSize`. Leaves hold between half the runtime bucket plus one and
the runtime bucket particles. The kick arena is tuned for a runtime bucket
equal to the cap: its automatic sink-force tile covers the whole bucket
whenever the shared budget allows and only then falls back to bounded tiles.

## Shared-memory arena

One 32-thread warp per block. The dynamic shared memory of a block follows a
host-computed `CudaKickArenaLayout` (`cosmictiger/kick.hpp`):

Persistent region

- `CudaKickControl<order>`: the control objects of the interaction lists, the
  checklist stacks and the ancestor-expansion stack (their elements stay in the
  growable `cudaAllocate` pool), the staging barrier, and the resolved region
  pointers. 496 bytes at order 7.
- The active local expansion with its position (`compute.cudaKickSharedExpansion`,
  default on). Each node's L2L translates cooperatively into the arena scratch
  with `translateInto` (group leaders store their coefficients; no replicated
  per-lane result) and the warp copies the result back; ancestors awaiting their
  right child are pushed to and restored from global backing.

Reusable phase arena, sized for the largest simultaneously live set

- L2L output scratch (one expansion), used before any force exists.
- Sink-force planes gx, gy, gz, phi for `sinkCapacity` particles. They are
  initialized only after the cell interactions, so they alias the scratch.
- Source tiles following the force planes: particle planes x, y, z (and a mass
  plane only for mixed-mass runs; equal-mass runs instantiate PP/CP without it)
  or staged positioned multipoles with a 16-byte stride. The two tiles overlap,
  because PC and PP never run concurrently. PC accumulates all staged
  multipoles in registers and writes each sink once.

Lifetimes: staged tiles are refilled only after every lane has finished reading
them (`__syncwarp()` after each use); the checklist copy and CP share one
barrier, so the copy completes before CP starts; the control block is
constructed by every lane and lane zero writes the resolved pointers only after
a warp barrier. compute-sanitizer racecheck reports no hazards and no warnings
for the fmmWalk regression; memcheck is clean. synccheck reports 32 "missing
init/wait" diagnostics on the `cuda::barrier` objects in dynamic shared memory;
the unchanged baseline kernel reports the same 32, so these are a pre-existing
tool interaction, not a new hazard.

Forces for buckets larger than the sink tile are processed as consecutive
sink tiles: forces are zeroed, PC and PP run over the full interaction lists,
and the kick of that tile follows, so lists are built once per leaf and sources
are re-staged per tile. `compute.cudaKickSinkTile=-1` keeps whole-bucket forces
in global backing instead.

## Budget and tile selection

`kickArenaLayoutImpl<order>()` reads the compiled kernel's registers and static
shared memory with `cudaFuncGetAttributes` and the device limits (shared memory
per SM, the 1 KB driver reservation per block, registers per SM, blocks and
warps per SM). The register-limited residency uses the 256-register warp
allocation granularity. For the requested residency (`compute.cudaKickResidency`,
default: the register limit) the byte budget per block is the per-SM shared
memory divided by the residency, minus the reservation and static shared
memory, rounded down to the 128-byte allocation unit. The layout is then built
to fit that budget (defaults: whole-bucket sink tile, 512 staged particles, a
multipole tile that fills the remaining source bytes up to 64 records, trimmed
particles-first when the budget is exceeded) and verified with
`cudaOccupancyMaxActiveBlocksPerMultiprocessor`; the budget shrinks by one
allocation unit until the calculator agrees. The chosen policy is printed once
per order as `CUDA kick arena (FMM order N): ...`.

Options (all default to automatic): `compute.cudaKickResidency`,
`compute.cudaKickSinkTile`, `compute.cudaKickParticleTile`,
`compute.cudaKickMultipoleTile`, `compute.cudaKickSharedExpansion`,
`compute.cudaKickHandoffBlocks` (pins the CPU/GPU handoff divisor so allocation
variants can be compared at identical work division).

## Compiled kernel resources (measured)

`cuobjdump --dump-resource-usage` on the linked binary (CUDA 12.8, sm_86,
`--use_fast_math`, one 32-thread block). Registers saturate at 255 from order 6
upward; the stack frame, which holds the per-lane spills of the Ewald and
cell-interaction expansions, grows steeply with order.

| order | registers | stack bytes | static shared bytes | register-limited blocks/SM |
|---:|---:|---:|---:|---:|
| 3 | 234 | 264 | 0 | 8 |
| 4 | 224 | 296 | 0 | 9 (8 achieved) |
| 5 | 230 | 328 | 0 | 8 |
| 6 | 255 | 376 | 0 | 8 |
| 7 | 255 | 544 | 0 | 8 |
| 8 | 255 | 936 | 0 | 8 |
| 9 | 255 | 2160 | 0 | 8 |

Each per-order CUDA instance unit compiles in 22 to 29 seconds; parsing the
all-order generated headers costs about 2 seconds with g++ and 3 with nvcc.

## Validation (measured)

`docs/cuda_kick_shared_memory_results/validation/status.txt` records, for the
runtime-order binary:

- `stf_kernels_test`, `stf_kernels_cuda_test`, `stf_l2l_cuda_test` and
  `stf_ewald_cpu_test` pass for every order 3 through 9 in one executable
  (independent Cartesian polynomial, binomial-series and remainder oracles,
  SIMD and host/device parity, the cooperative L2L at every local degree with
  `translate` and `translateInto`, and the Ewald tables refilled at orders 3,
  7 and 9).
- The CUDA `fmmWalk` regression (CPU and CUDA against operator references,
  periodic Ewald sums, hybrid leaf-cloud paths, reciprocity) passes at every
  order 3 through 9.
- `fmmTree` and the mixed-mass `sphGravity` regression (which exercises the
  mass-plane instantiation) pass at orders 3, 7 and 9.

## CPU and CUDA agreement (measured)

The force exports of CPU-only and CUDA runs of the same fixtures agree to about
1e-7 in relative terms for both accelerations and potentials at orders 3, 7 and 9
(`docs/cuda_kick_shared_memory_results/agreement_cpu_cuda/`). The harness's
direct Ewald reference is only valid in CUDA-enabled runs: it is evaluated on the
device and reads the device kernel table (self potential and softening), which
`setKernelSelections` uploads only when `compute.cuda` is on. In a CPU-only run
that reference reported a spurious 1.5% potential offset; the harness now skips
the reference there and prints `STF_WALK_REFERENCE_SKIPPED`.

## Measurements

All timings: RTX 3090 shared with the desktop, two HPX workers pinned to cores
1 and 2, 32,768 particles, theta 0.8 unless stated, Ferrers-4 softening, CUDA
backend. Values are medians over paired processes of each process's median kick
time (one warmup, 15 samples per case); the paired range is the min-max of the
per-pair ratio. "baseline" is the unchanged commit 585ee6c0 (fixed order 7,
15,184-byte shared union, 6 blocks per SM, 492 blocks); "arena" is this work at
order 7. Raw data, commands, per-sample times and force exports are under
`docs/cuda_kick_shared_memory_results/` (force CSVs are not committed).

### Allocation variants at order 7, bucket 64 (`variants_order7_bucket64/`)

Speedup is baseline time over variant time (above one: variant faster).

| variant | lattice (forces) | clustered (forces) | adaptive (forces) | adaptive (forces + potential) |
|---|---:|---:|---:|---:|
| baseline, 6 blocks/SM | 33.40 ms | 8.77 ms | 19.20 ms | 19.73 ms |
| arena auto (8 blocks/SM, shared expansion and forces, 512-particle tile) | 32.14 ms, 1.046x (1.039-1.064) | 8.38 ms, 1.082x (1.036-1.082) | 20.81 ms, 0.919x (0.917-0.941) | 21.37 ms, 0.923x |
| arena, residency 6 | 32.18 ms, 1.039x | 8.38 ms, 1.041x | 21.19 ms, 0.908x | 20.98 ms, 0.941x |
| arena, residency 7 | 31.88 ms, 1.048x | 8.11 ms, 1.074x | 20.61 ms, 0.932x | 20.84 ms, 0.948x |
| arena, global active expansion | 32.28 ms, 1.037x | 8.93 ms, 1.007x | 20.41 ms, 0.941x | 20.64 ms, 0.949x |
| arena, global whole-bucket forces | 32.83 ms, 1.018x | 9.11 ms, 0.984x | 20.57 ms, 0.928x | 20.73 ms, 0.946x |
| arena, lists reserved (4096) | 31.92 ms, 1.047x | 8.41 ms, 1.043x | 20.18 ms, 0.959x | 20.35 ms, 0.959x |
| arena, residency 6 + lists reserved | 32.09 ms, 1.041x | 8.47 ms, 1.040x | 20.61 ms, 0.932x | 20.55 ms, 0.950x |

The arena is 4-5% faster than the baseline on the lattice and 4-8% faster on
the clustered fixture, and 4-9% slower on the adaptive fixture (1,082 leaves of
1 to 64 particles, depths 9 to 17), for every policy. Keeping the forces or the
active expansion in global memory, or lowering the residency to the baseline's
6 blocks, does not recover the adaptive loss, so it is not the shared-memory
layout itself; reserving the list capacities (one pool allocation per list
instead of doubling growth) recovers about half of it.

### Fixed CPU/GPU handoff (`handoff_fixed_order7/`)

With `compute.cudaKickHandoffBlocks=492` the arena divides work between CPU and
GPU exactly as the baseline does. The picture is unchanged: lattice 1.027x,
clustered 1.046x, adaptive 0.927x (0.918-0.933) for forces, 0.935x with the
potential. The adaptive gap is therefore not the handoff threshold either.

### Kernel attribution (preliminary, `fixed_order7_prelim/`)

Nsight Compute replay of the force-only adaptive kick kernel put the kernel
itself 0.3 ms slower (5.55 versus 5.25 ms) with the whole difference in the
pool allocator (`cudaAllocate` 42.0 M to 64.3 M warp instructions, `cudaRelease`
9.4 M to 14.9 M) while PP, PC, CP, CC and the walk were within 1-6%. The
remaining 1 to 1.5 ms of the adaptive gap lies outside the kernel; the Nsight
Systems host/kernel split below is the next measurement.

### Orders at bucket 64, theta 0.8 (`orders_bucket64/`)

Force and potential errors are relative L2 against the direct periodic sum over
1,024 sampled particles (the lattice's forces nearly cancel, so its relative
errors are large by construction). Times are with the potential.

| order | adaptive ms | adaptive force err | adaptive potential err | clustered ms | clustered force err | clustered potential err | lattice ms | lattice force err |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 13.40 | 2.77e-2 | 3.87e-3 | 8.10 | 4.04e-2 | 1.08e-2 | 19.40 | 4.49 |
| 4 | 14.09 | 1.63e-2 | 1.59e-3 | 7.90 | 2.77e-2 | 7.72e-3 | 20.67 | 4.38 |
| 5 | 14.82 | 6.51e-3 | 4.60e-4 | 8.02 | 1.20e-2 | 2.50e-3 | 20.31 | 7.10e-1 |
| 6 | 16.72 | 3.74e-3 | 1.83e-4 | 9.08 | 6.46e-3 | 7.72e-4 | 25.08 | 4.43e-1 |
| 7 | 21.85 | 2.11e-3 | 1.03e-4 | 9.10 | 4.82e-3 | 5.75e-4 | 33.62 | 3.42e-1 |
| 8 | 27.81 | 1.38e-3 | 6.67e-5 | 10.45 | 2.98e-3 | 4.35e-4 | 48.62 | 2.43e-1 |
| 9 | 43.18 | 9.39e-4 | 6.07e-5 | 12.61 | 2.98e-3 | 6.06e-4 | 84.50 | 2.10e-1 |

Every order runs through the same binary (`gravity.fmmOrder`). On the clustered
fixture the force error stops improving between orders 8 and 9 and the
potential error rises again: the single-precision floor the order cap is based
on. A matched-accuracy comparison needs the opening angle varied per order; the
theta sweep below provides it.

### Selected defaults and what remains

Measured basis for the defaults: residency stays at the register limit (8 blocks
per SM at orders 6 and above), since lowering it to the baseline's 6 did not
help any fixture; the active expansion and the sink forces stay in shared
memory, since moving either to global backing did not help; and
`compute.cudaKickListReserve` defaults to 4096 entries, the only variant that
recovered part of the adaptive-fixture loss (0.919x to 0.959x) without costing
the other fixtures. These are measured on one workstation with a shared GPU and
should be re-checked where the balance between pool allocation and traversal
differs.

Not yet measured (the runs were queued and then paused under a machine-wide
resource cap): the bucket sweep at 32, 64, 128, 256, 512 and 1024 with
whole-bucket, 128- and 64-sink tiles and global force backing
(`buckets_order7/`), the Nsight Systems host/kernel split of the remaining
adaptive-fixture gap (`nsys_adaptive_order7/`), and the theta sweep that
compares orders at matched force error (`theta_sweep/`). Buckets above 128
require a build with `COSMICTIGER_WITH_MAX_BUCKET=1024`; `build/order_shared_arena`
is configured that way. The commands are in `benchmarks/run_cuda_kick_arena.py`
(`--buckets`, `--orders`, `--theta`, `--policy`) and the summaries come from
`benchmarks/compare_kick_arena_runs.py` and `benchmarks/summarize_kick_profiles.py`.

Known limitation: the kernel's pool allocator (`cudaAllocate`/`cudaRelease`)
scans lock-free queues with atomics and already costs about 18% of the baseline
kernel's warp instructions; more resident blocks raise that cost. A per-launch
slab for the per-block list storage would remove it and is the natural next step
if the adaptive-fixture gap matters.
