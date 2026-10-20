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

- `cosmictiger/fmm/fmm_order.hpp`: constexpr coefficient counts, byte sizes, the
  `FmmOrder<order>` traits, the runtime `fmmOrder()` and `fmmDispatch()`, which
  switches to the matching instantiation at coarse boundaries (tree build, a
  CPU walk, a CUDA launch). Inner loops never test the order.
- `cosmictiger/fmm/fmm_types.hpp`: `FmmMultipole<Real, order>`,
  `FmmExpansion<Real, order>`, the position-first multipole record
  `FmmMultiPos<order>` (tree nodes point at its order-agnostic
  `MultipolePosition` header), and the max-size transport buffers
  `ExpansionBuffer` / `MultipoleBuffer` used by order-agnostic interfaces.
- `cosmictiger/fmm/fmm_kernels.hpp`: operator wrappers that deduce the degrees from
  their tensor arguments over the generated STF kernels.
- Generated code: `tools/generate_stf_m2l.py` emits one fully specialized
  kernel class per degree for local degrees 2 through 8 into a single header set
  in the build tree (`ninja cosmictiger_generated_kernels`), and
  `src/fmm/fmm_kernel_codegen.cpp` emits one Ewald derivative overload per order.
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
host-computed `CudaKickArenaLayout` (`cosmictiger/gravity/kick.hpp`):

Persistent region

- `CudaKickControl<order>`: the control objects of the interaction lists, the
  checklist stacks and the ancestor-expansion stack (their elements live in the
  block's region of the per-launch list slab, below), the staging barrier, and
  the resolved region pointers. 648 bytes at order 7 (528 before the slab: each
  vector carries an ownership flag and the layout two more fields).
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
  multipoles in registers and writes each sink once. PP reads the planes
  through pointers derived from the dynamic shared-memory symbol (so the
  compiler emits shared-memory loads), four sources per 16-byte load, and
  gives each lane two sinks on complete 64-sink chunks; see "PP loop schedule
  at low occupancy" below.

Lifetimes: staged tiles are refilled only after every lane has finished reading
them (`__syncwarp()` after each use); the checklist copy and CP share one
barrier, so the copy completes before CP starts; the control block is
constructed by every lane and lane zero writes the resolved pointers only after
a warp barrier. compute-sanitizer memcheck is clean for the fmmWalk regression
and racecheck reports no errors; its warnings (126 in the committed order-7
log, `sanitizer/racecheck_order_7.log`) all pair the `cuda::memcpy_async`
staging writes with the later reads of the staged tile, accesses that the
`cuda::barrier` wait orders. Whether a build gets these warnings depends on
how ptxas addresses the barrier, not on the staging code (see the racecheck
staging warnings section). synccheck reports 32
"missing init/wait" diagnostics on the `cuda::barrier` objects in dynamic
shared memory; the unchanged baseline kernel reports the same 32, so these are
a pre-existing tool interaction, not a new hazard.

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
`compute.cudaKickHandoffBlocks` (pins the CPU/GPU handoff divisor; by default
`kickBlockCount()` uses two resident blocks per multiprocessor, a measured
choice that is independent of the launch size, see the handoff sections
below).

## Per-launch list slab

A block's six interaction lists, its two checklist stacks with their level
bounds, the ancestor-expansion and search-parameter stacks and, when the
corresponding region is not in shared memory, the active expansion and the
whole-bucket force backing live in global memory. With
`compute.cudaKickListReserve` above zero (default 4096 entries per list) that
storage is one region per block of a slab the host allocates once per order
with `cudaMalloc`, sized for `residency` blocks per multiprocessor.
`kickCarveListStorage<order>()` (`cosmictiger/gravity/kick_cuda.hpp`) lays the regions
out in a fixed order on 16-byte boundaries; the host calls it with a null base
to measure the per-block stride (`CudaKickArenaLayout::listSlabBytes`, a
multiple of 128 bytes) and each block calls it on its own region to resolve
its pointers. The block then constructs its control object on those regions
(`DeviceVector::attach`: the vector uses the storage without owning it), so
constructing and destroying the lists costs no pool traffic. A list that
outgrows its region moves into a pool block as before (doubling growth), and
that pool block is released at the end of the launch. At order 7 with the
default reservation a block's region is 145,792 bytes (eight 16 KB lists, the
64-level ancestor stack of 212-byte expansions, the bounds and parameters),
95.6 MB for 656 blocks, about what the twelve pool blocks of the earlier
reservation occupied inside the managed-memory heap (`HEAP_SIZE`), which no
longer has to hold them. `compute.cudaKickListReserve=0` keeps every list in
the pool, grown on demand, as before.

Before the slab, every block reserved its lists at launch with one
`cudaAllocate` per list and released them at the end, twelve allocations and
twelve releases per block per launch, each a scan of the pool's lock-free
queues with system-scope atomics; Nsight Compute attributed 42 to 64 million
warp instructions per launch to `cudaAllocate` and 9 to 15 million to
`cudaRelease` on the adaptive fixture at bucket 64 (more with eight resident
blocks per multiprocessor than with six). The measured effect is in the
per-launch list slab section below.

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
- With the per-launch list slab, the CUDA and CPU `fmmWalk`, `fmmTree` and the
  CUDA `sphGravity` regressions pass in the 1024-bucket build
  (`validation/ctest_cap1024_list_slab.log`); the compute-sanitizer results of
  that build are in the list slab section below.
- The slab, PP loop schedule and racecheck explanation branches merged onto the
  reorganized `main` pass the STF unit tests, the CUDA `fmmWalk` regression at
  orders 3, 7 and 9 and `fmmTree` and `sphGravity` at order 7 on each merge
  head, with memcheck clean and racecheck at 0 errors on the final head
  (`validation/merge_reorganized/`, see Landing on the reorganized layout).

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
active expansion in global memory does not recover the adaptive loss, so it is
not the shared-memory layout itself; reserving the list capacities (one pool
allocation per list instead of doubling growth) recovers about half of it.
The "residency 6" and "residency 7" rows of this table were measured before
the residency option worked (see the residency sweep below): they launched
8 blocks per SM like the automatic policy and differ from it only by noise.
These runs used the launch-coupled handoff (656 blocks); the handoff sections
below change that default.

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

### Bucket cap of the benchmark build (`cap_effect_bucket64/`)

Buckets above 128 need a build with `COSMICTIGER_WITH_MAX_BUCKET=1024`. The
automatic sink tile follows the runtime bucket (`min(compute.bucketSize,
kMaxBucketSize)`), so at bucket 64 the 1024-cap build selects the same layout
as the 128-cap build of the measurements above (7,904 dynamic shared bytes,
64-particle sink tile, 512-particle source tile, 38 multipoles) and its force
exports are byte-identical to the 128-cap build's. Timings agree within the
pair spread; the 1024-cap build is used for everything below.

| variant | lattice (forces) | clustered (forces) | adaptive (forces) | adaptive (forces + potential) |
|---|---:|---:|---:|---:|
| baseline, 6 blocks/SM | 33.06 ms | 8.87 ms | 19.40 ms | 19.47 ms |
| arena, 128-particle cap build | 31.87 ms, 1.037x (1.030-1.038) | 8.44 ms, 1.057x (1.045-1.069) | 20.99 ms, 0.923x (0.917-0.937) | 21.04 ms, 0.933x (0.902-0.944) |
| arena, 1024-particle cap build | 31.64 ms, 1.037x (1.037-1.049) | 8.68 ms, 1.021x (1.017-1.067) | 20.52 ms, 0.945x (0.924-0.948) | 20.81 ms, 0.936x (0.926-0.938) |

### Bucket sweep at order 7 (`buckets_order7/`)

Baseline and arena at buckets 32 to 1024, three fixtures, forces only, three
pairs of 15 samples, direct reference over 1,024 particles. Speedup is baseline
time over arena time at the same bucket (above one: arena faster). The baseline
binary has no bucket cap and runs every bucket.

lattice:

| bucket | baseline | arena auto | arena, sink tile 128 | arena, sink tile 64 | arena, global forces |
|---:|---:|---:|---:|---:|---:|
| 32 | 33.10 ms | 90.73 ms, 0.363x (0.352-0.373) | 90.78 ms, 0.364x (0.363-0.371) | 90.40 ms, 0.364x (0.364-0.374) | 90.61 ms, 0.365x (0.364-0.374) |
| 64 | 32.39 ms | 31.03 ms, 1.048x (1.027-1.052) | 31.27 ms, 1.036x (1.025-1.042) | 30.83 ms, 1.053x (1.035-1.055) | 31.18 ms, 1.040x (1.033-1.043) |
| 128 | 14.98 ms | 14.37 ms, 1.034x (0.988-1.065) | 14.49 ms, 1.026x (1.000-1.053) | 14.37 ms, 1.044x (0.994-1.060) | 14.31 ms, 1.047x (1.016-1.069) |
| 256 | 11.44 ms | 11.71 ms, 0.977x (0.941-0.994) | 11.89 ms, 0.967x (0.918-0.978) | 11.83 ms, 0.964x (0.963-1.013) | 11.77 ms, 0.977x (0.926-1.004) |
| 512 | 10.60 ms | 12.60 ms, 0.853x (0.836-0.899) | 12.47 ms, 0.849x (0.847-0.871) | 12.39 ms, 0.854x (0.845-0.954) | 12.25 ms, 0.868x (0.866-0.959) |
| 1024 | 16.48 ms | 19.22 ms, 0.854x (0.841-0.883) | 19.25 ms, 0.862x (0.853-0.871) | 19.33 ms, 0.869x (0.853-0.937) | 19.18 ms, 0.879x (0.852-0.943) |

clustered:

| bucket | baseline | arena auto | arena, sink tile 128 | arena, sink tile 64 | arena, global forces |
|---:|---:|---:|---:|---:|---:|
| 32 | 8.80 ms | 11.34 ms, 0.744x (0.724-0.817) | 11.10 ms, 0.780x (0.769-0.796) | 10.67 ms, 0.814x (0.796-0.828) | 11.00 ms, 0.803x (0.797-0.813) |
| 64 | 8.37 ms | 8.17 ms, 1.005x (0.991-1.045) | 7.91 ms, 1.058x (0.993-1.090) | 8.03 ms, 1.043x (1.040-1.056) | 8.13 ms, 1.043x (1.005-1.060) |
| 128 | 6.19 ms | 6.18 ms, 0.991x (0.983-1.031) | 6.12 ms, 1.014x (1.000-1.032) | 6.12 ms, 1.019x (1.012-1.032) | 6.40 ms, 0.957x (0.950-1.006) |
| 256 | 7.16 ms | 7.00 ms, 1.033x (0.992-1.041) | 7.22 ms, 0.995x (0.936-1.002) | 6.81 ms, 1.020x (1.009-1.066) | 7.04 ms, 1.001x (0.977-1.018) |
| 512 | 9.92 ms | 10.02 ms, 0.991x (0.927-1.034) | 10.14 ms, 0.979x (0.941-1.036) | 10.26 ms, 1.003x (0.929-1.067) | 9.98 ms, 1.008x (0.994-1.034) |
| 1024 | 16.80 ms | 18.41 ms, 0.917x (0.888-0.917) | 16.71 ms, 1.010x (0.976-1.010) | 16.95 ms, 0.995x (0.978-1.060) | 16.43 ms, 1.023x (1.015-1.099) |

adaptive:

| bucket | baseline | arena auto | arena, sink tile 128 | arena, sink tile 64 | arena, global forces |
|---:|---:|---:|---:|---:|---:|
| 32 | 17.91 ms | 20.63 ms, 0.871x (0.806-0.882) | 20.74 ms, 0.864x (0.852-0.878) | 20.44 ms, 0.880x (0.831-0.889) | 20.50 ms, 0.877x (0.861-0.879) |
| 64 | 18.49 ms | 19.53 ms, 0.948x (0.905-0.961) | 20.26 ms, 0.917x (0.908-0.922) | 19.93 ms, 0.942x (0.907-0.964) | 20.06 ms, 0.924x (0.917-0.927) |
| 128 | 16.18 ms | 17.83 ms, 0.911x (0.881-0.915) | 17.79 ms, 0.908x (0.902-0.913) | 17.20 ms, 0.944x (0.906-0.954) | 17.47 ms, 0.927x (0.915-0.929) |
| 256 | 18.50 ms | 21.04 ms, 0.878x (0.877-0.920) | 21.17 ms, 0.876x (0.870-0.892) | 20.99 ms, 0.882x (0.879-0.947) | 20.35 ms, 0.928x (0.891-0.936) |
| 512 | 28.50 ms | 34.64 ms, 0.823x (0.785-0.907) | 35.01 ms, 0.823x (0.809-0.857) | 34.95 ms, 0.860x (0.805-0.904) | 34.19 ms, 0.880x (0.818-0.909) |
| 1024 | 82.78 ms | 98.09 ms, 0.844x (0.818-0.851) | 80.81 ms, 1.017x (1.003-1.127) | 81.68 ms, 1.013x (0.976-1.110) | 80.16 ms, 1.033x (1.022-1.085) |

Automatic layouts chosen per bucket (order 7, 11,776-byte budget):

- bucket 32: 7,392 bytes, sink tile 32, 512 particles, 38 multipoles
- bucket 64: 7,904 bytes, sink tile 64, 512 particles, 38 multipoles
- bucket 128: 8,928 bytes, sink tile 128, 512 particles, 38 multipoles
- bucket 256: 10,976 bytes, sink tile 256, 512 particles, 38 multipoles
- bucket 512: 11,648 bytes, sink tile 512, 224 particles, 17 multipoles
- bucket 1024: 3,168 bytes, sink tile 128, 32 particles, 2 multipoles

What the sweep shows:

- At buckets 64 and 128 (128 is the default) the arena is 3-5% faster on the
  lattice, level on the clustered fixture and 5-9% slower on the adaptive
  fixture, as before. The best absolute times move to bucket 128-256 on the
  lattice (15 and 11 ms against 32 ms at 64) and bucket 128 on the clustered
  and adaptive fixtures.
- At bucket 32 the arena loses badly on every fixture (lattice 0.36x, clustered
  0.74-0.81x, adaptive 0.87x) and the sink-tile policy makes no difference, so
  the shared layout is not the cause. The CPU/GPU handoff is the suspect: the
  CPU hands a subtree to the GPU once it holds at most max(N / blocks, bucket)
  particles, and the arena's 656 blocks (8 per SM) lower that threshold from
  the baseline's 66 (492 blocks) to 49 particles. At bucket 32 this is the only
  bucket where the threshold exceeds the bucket, so the arena's CPU walk goes
  one level deeper on two threads. The handoff diagnostic below tests this.
- At bucket 1024 the automatic layout is wrong: the whole-bucket sink tile
  (16 KB of forces) does not fit, the trim loop first shrinks the source tiles
  to their minimum (32 particles, 2 multipoles) and only then falls back to the
  128-particle sink tile, keeping the shrunken source tiles. The explicit
  128-particle tile with the default 512-particle source tile is 1.02x the
  baseline on the adaptive fixture where the automatic layout is 0.84x.
  `kickBuildLayout` now restores the automatic source tiles whenever it shrinks
  the sink tile; the check below measures the fixed build.
- At bucket 512 every arena policy, including global force backing, is 0.82-0.88x
  on the adaptive fixture and 0.85-0.87x on the lattice while the clustered
  fixture is level; this is not the force storage either. With 64 leaves of up
  to 512 particles only 64 of the 656 blocks have work, so per-warp throughput
  of the leaf processing decides, and the arena's PP/PC loops are slower there
  than the baseline's 928-particle staging. The Nsight Systems split below
  includes this case.

Force and potential errors against the direct periodic sum do not depend on
the binary or policy (the arena matches the baseline to 1e-7 at every bucket);
larger buckets replace multipole approximations with direct sums and lower the
adaptive error from 2.7e-3 at bucket 32 to 7.5e-4 at bucket 1024:

| bucket | lattice force | clustered force | clustered potential | adaptive force | adaptive potential |
|---:|---:|---:|---:|---:|---:|
| 32 | 2.46e-01 | 3.90e-03 | 5.18e-04 | 2.67e-03 | 1.06e-04 |
| 64 | 3.42e-01 | 4.81e-03 | 5.75e-04 | 2.11e-03 | 1.02e-04 |
| 128 | 3.42e-01 | 2.24e-03 | 5.02e-04 | 1.77e-03 | 1.04e-04 |
| 256 | 3.42e-01 | 1.88e-03 | 5.31e-04 | 1.32e-03 | 9.64e-05 |
| 512 | 3.42e-01 | 1.77e-03 | 5.58e-04 | 1.00e-03 | 9.75e-05 |
| 1024 | 3.42e-01 | 1.77e-03 | 5.58e-04 | 7.53e-04 | 9.12e-05 |

### Multi-tile sink processing (`buckets_order7/policy_forces.md`)

`benchmarks/compare_kick_arena_policies.py` compares the force export of every
policy with the automatic (whole-bucket) layout at the same bucket. Up to bucket
256, where all policies stage the same 512-particle and 38-multipole source
tiles, the 128- and 64-particle sink tiles and the global force backing
reproduce the whole-bucket forces and potentials bit for bit: each sink sums
its sources in the same order whichever tile it belongs to. At buckets 512 and
1024 the automatic layout stages smaller source tiles (224/17 and 32/2), which
changes the summation order; the differences are roundoff (adaptive and
clustered: acceleration relative L2 at most 3.6e-7, potential 6.2e-7; lattice,
whose forces nearly cancel, 6.9e-6 and 2.5e-5 with a worst pointwise relative
acceleration difference of 3.8e-4). The CUDA `fmmWalk`, `fmmTree` and mixed-mass
`sphGravity` regressions pass in the 1024-cap build
(`validation/ctest_cap1024.log`); `fmmWalk` fixes its bucket at 64 and its
fixtures hold at most 88 particles, so the sink tiling itself is validated by
the force comparisons above, not by that test.

### Theta sweep (`theta_sweep/`)

Orders 3 to 9 at opening angles 0.5 to 1.0, bucket 64, forces only, two
pairs of 15 samples per cell, direct reference over 1,024 particles, one
`run_cuda_kick_arena.py` result directory per theta
(`benchmarks/summarize_kick_profiles.py theta` builds the tables). Every cell
runs through the same binary with `gravity.fmmOrder` and
`gravity.openingAngle`.

adaptive (kick ms / force relative L2):

| order | theta 0.5 | theta 0.6 | theta 0.7 | theta 0.8 | theta 0.9 | theta 1.0 |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 19.9 / 7.4e-03 | 15.5 / 1.2e-02 | 13.0 / 1.6e-02 | 12.5 / 2.8e-02 | 11.3 / 3.7e-02 | 11.4 / 7.8e-02 |
| 4 | 21.6 / 1.8e-03 | 17.1 / 3.6e-03 | 14.4 / 5.6e-03 | 12.3 / 1.6e-02 | 13.0 / 2.5e-02 | 12.3 / 5.7e-02 |
| 5 | 23.1 / 6.2e-04 | 17.5 / 1.3e-03 | 14.3 / 2.3e-03 | 12.8 / 6.5e-03 | 12.5 / 1.5e-02 | 11.5 / 4.2e-02 |
| 6 | 27.8 / 2.2e-04 | 21.4 / 5.5e-04 | 17.0 / 1.3e-03 | 14.8 / 3.7e-03 | 14.6 / 1.2e-02 | 13.0 / 3.4e-02 |
| 7 | 38.4 / 6.8e-05 | 27.9 / 2.5e-04 | 23.1 / 6.7e-04 | 19.7 / 2.1e-03 | 18.7 / 6.6e-03 | 15.7 / 2.4e-02 |
| 8 | 50.3 / 2.2e-05 | 38.8 / 1.1e-04 | 30.7 / 4.0e-04 | 26.1 / 1.4e-03 | 24.4 / 5.0e-03 | 20.4 / 1.9e-02 |
| 9 | 75.2 / 1.3e-05 | 56.8 / 4.9e-05 | 45.1 / 2.3e-04 | 39.0 / 9.4e-04 | 37.2 / 3.6e-03 | 30.6 / 1.5e-02 |

clustered (kick ms / force relative L2):

| order | theta 0.5 | theta 0.6 | theta 0.7 | theta 0.8 | theta 0.9 | theta 1.0 |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 11.0 / 1.2e-02 | 7.8 / 2.1e-02 | 7.3 / 2.8e-02 | 6.6 / 4.0e-02 | 6.7 / 4.6e-02 | 6.8 / 5.6e-02 |
| 4 | 12.6 / 4.8e-03 | 7.7 / 1.3e-02 | 7.3 / 1.5e-02 | 6.8 / 2.8e-02 | 6.6 / 2.9e-02 | 7.4 / 4.3e-02 |
| 5 | 12.0 / 9.4e-04 | 8.2 / 1.9e-03 | 7.6 / 4.3e-03 | 6.8 / 1.2e-02 | 7.0 / 1.7e-02 | 7.2 / 3.2e-02 |
| 6 | 13.2 / 4.0e-04 | 8.6 / 7.3e-04 | 8.4 / 2.8e-03 | 7.8 / 6.5e-03 | 7.7 / 1.0e-02 | 7.8 / 2.0e-02 |
| 7 | 16.9 / 8.1e-05 | 9.8 / 2.6e-04 | 9.5 / 1.2e-03 | 7.9 / 4.8e-03 | 8.5 / 7.1e-03 | 8.2 / 1.6e-02 |
| 8 | 22.0 / 3.8e-05 | 11.3 / 1.4e-04 | 11.1 / 7.9e-04 | 9.3 / 3.0e-03 | 10.6 / 4.6e-03 | 9.6 / 1.3e-02 |
| 9 | 35.2 / 4.3e-05 | 15.1 / 2.0e-04 | 15.1 / 5.8e-04 | 11.0 / 3.0e-03 | 11.7 / 3.9e-03 | 11.7 / 8.9e-03 |

lattice (kick ms / force relative L2):

| order | theta 0.5 | theta 0.6 | theta 0.7 | theta 0.8 | theta 0.9 | theta 1.0 |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 29.4 / 4.3e-01 | 25.3 / 8.3e-01 | 21.4 / 1.6e+00 | 17.8 / 4.5e+00 | 15.8 / 4.2e+00 | 14.7 / 4.0e+00 |
| 4 | 32.4 / 2.6e-01 | 28.1 / 6.8e-01 | 23.3 / 1.4e+00 | 19.3 / 4.4e+00 | 17.2 / 4.5e+00 | 16.2 / 6.6e+00 |
| 5 | 31.9 / 5.7e-02 | 27.5 / 1.6e-01 | 22.5 / 2.1e-01 | 18.8 / 7.1e-01 | 17.4 / 1.2e+00 | 15.7 / 3.9e+00 |
| 6 | 39.4 / 3.1e-02 | 33.7 / 1.2e-01 | 28.2 / 1.3e-01 | 23.3 / 4.4e-01 | 21.7 / 1.1e+00 | 18.9 / 4.5e+00 |
| 7 | 54.8 / 1.1e-02 | 45.7 / 3.9e-02 | 37.4 / 9.5e-02 | 31.2 / 3.4e-01 | 28.9 / 5.6e-01 | 24.7 / 2.8e+00 |
| 8 | 77.2 / 5.8e-03 | 66.0 / 2.7e-02 | 53.5 / 8.5e-02 | 45.5 / 2.4e-01 | 40.2 / 4.8e-01 | 35.0 / 2.8e+00 |
| 9 | 133.3 / 5.4e-03 | 114.4 / 2.6e-02 | 94.0 / 6.2e-02 | 77.0 / 2.1e-01 | 68.9 / 3.8e-01 | 60.1 / 2.6e+00 |

The lattice's forces nearly cancel, so its relative errors are large by
construction (orders 3-6 stay above 2e-2 at every theta; order 7 at theta 0.5
reaches 1.1e-2 in 54.8 ms and order 8 at theta 0.5 5.8e-3 in 77.2 ms); it is not
a useful accuracy fixture and is left out of the matched-accuracy tables.

Fastest measured configuration at or below a force error target (adaptive):
| target force rel L2 | order | theta | kick forces (ms) | measured error |
|---:|---:|---:|---:|---:|
| 3e-02 | 4 | 0.8 | 12.344 | 1.626e-02 |
| 1e-02 | 5 | 0.8 | 12.751 | 6.512e-03 |
| 5e-03 | 5 | 0.7 | 14.267 | 2.309e-03 |
| 3e-03 | 5 | 0.7 | 14.267 | 2.309e-03 |
| 2e-03 | 6 | 0.7 | 16.969 | 1.258e-03 |
| 1e-03 | 6 | 0.6 | 21.392 | 5.481e-04 |
| 5e-04 | 6 | 0.5 | 27.826 | 2.231e-04 |

Fastest measured configuration at or below a force error target (clustered):
| target force rel L2 | order | theta | kick forces (ms) | measured error |
|---:|---:|---:|---:|---:|
| 3e-02 | 4 | 0.9 | 6.642 | 2.902e-02 |
| 1e-02 | 5 | 0.7 | 7.567 | 4.274e-03 |
| 5e-03 | 5 | 0.7 | 7.567 | 4.274e-03 |
| 3e-03 | 5 | 0.6 | 8.190 | 1.922e-03 |
| 2e-03 | 5 | 0.6 | 8.190 | 1.922e-03 |
| 1e-03 | 6 | 0.6 | 8.636 | 7.266e-04 |
| 5e-04 | 7 | 0.6 | 9.766 | 2.567e-04 |

Reading at matched accuracy (adaptive fixture): the default order 7 at theta
0.8 reaches a force error of 2.1e-3 in 19.7 ms; order 6 at theta 0.7 reaches
1.3e-3 in 17.0 ms and order 5 at theta 0.7 2.3e-3 in 14.3 ms. Down to a 5e-4
target the fastest measured configuration is order 5 or 6 with theta 0.6-0.7,
never order 7. Order 7 at theta 0.6 (2.5e-4, 27.9 ms) is level with order 6 at theta
0.5 (2.2e-4, 27.8 ms); order 7 becomes the fastest choice only below that, at
theta 0.5 (6.8e-5 in 38.4 ms), and orders 8 and 9 only below 7e-5, where they
reach 2.2e-5 and 1.3e-5 at theta 0.5 in 50 and 75 ms. The clustered fixture agrees down to 1e-3 (order 5 or 6 at theta 0.6-0.7)
and prefers order 7 at theta 0.6 for 5e-4 (9.8 ms, 2.6e-4); its single
precision floor shows at order 9, theta 0.5, whose error (4.3e-5) is above
order 8's (3.8e-5). On this grid a lower order with a smaller opening angle is
the cheaper way to a given force accuracy until roughly 2e-4, which is the
basis for choosing the production order: the default order 7 is only the right
choice when the target force error is below about 1e-3 on clustered data.

### Host and kernel split (`nsys_adaptive_order7/`)

`nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none` around the
benchmark process, three rounds per variant with alternating order, 15
samples, forces only and with the potential. The kick kernel is launched once
per kick, so `benchmarks/summarize_kick_profiles.py nsys` pairs the launches
with the per-sample kick times and reports medians of the kernel duration and
of the host share (whole kick minus its kernel). Besides the adaptive fixture
at bucket 64 the profile covers the two configurations where the bucket sweep
lost most. The residency-6 rows are the automatic policy in disguise (next
section). Medians over the three rounds:

| fixture, bucket | variant | whole kick (ms) | kick kernel (ms) | host (ms) | with potential: whole kick | kernel | host |
|---|---|---:|---:|---:|---:|---:|---:|
| adaptive, bucket 64 | baseline | 19.70 | 5.05 | 14.54 | 19.69 | 5.10 | 14.32 |
| adaptive, bucket 64 | arena auto | 20.93 | 5.62 | 14.96 | 20.84 | 5.69 | 14.84 |
| adaptive, bucket 64 | arena, residency 6 (ineffective, see below) | 21.23 | 6.06 | 15.02 | 21.06 | 6.12 | 14.87 |
| lattice, bucket 32 | baseline | 33.82 | 3.95 | 29.80 | 33.86 | 3.96 | 29.81 |
| lattice, bucket 32 | arena auto | 91.48 | 3.66 | 87.42 | 91.31 | 3.62 | 87.29 |
| lattice, bucket 32 | arena, residency 6 (ineffective, see below) | 91.36 | 3.94 | 87.56 | 91.39 | 3.64 | 87.57 |
| adaptive, bucket 512 | baseline | 30.66 | 24.81 | 5.41 | 29.43 | 23.95 | 5.42 |
| adaptive, bucket 512 | arena auto | 36.38 | 30.86 | 5.45 | 35.13 | 29.89 | 5.64 |
| adaptive, bucket 512 | arena, residency 6 (ineffective, see below) | 35.48 | 29.98 | 5.39 | 35.71 | 30.25 | 5.37 |

- Adaptive fixture, bucket 64: the host side is 14.5 ms of a 19.7 ms kick for
  the baseline, so three quarters of the kick is the two-thread CPU walk, not
  the kernel. The arena's 1.2 ms loss is 0.6 ms in the kernel (5.6 against
  5.1 ms, the pool-allocator cost that Nsight Compute attributed earlier) and
  0.4-0.5 ms on the host; the handoff diagnostic below assigns the host part to
  the finer CPU/GPU handoff. Earlier text that put 1 to 1.5 ms "outside the
  kernel" was based on a Nsight Compute replay, whose kernel timing does not
  transfer to the unprofiled run; the trace-based split above supersedes it.
- Lattice, bucket 32: the kernels are equal (3.7 against 4.0 ms); the arena's
  whole loss is host time, 87 ms against 30 ms.
- Adaptive fixture, bucket 512: the host share is identical (5.4 ms) and the
  kernel is 25% slower (30.9 against 24.8 ms). Only 64 leaves exist, so 64 of
  the launched blocks have work and per-warp leaf throughput decides. The PP
  and PC loops of the two kernels are the same code; `ncu_bucket512/` below
  attributes the difference.

### Handoff diagnostic at bucket 32 (`handoff_bucket32/`)

The CPU hands a subtree to the GPU once it holds at most
`max(N / kickBlockCount(), bucketSize)` particles, and `kickBlockCount()` is
the launch size, residency times the multiprocessor count: 656 blocks for the
arena against 492 for the baseline, so for 32,768 particles the threshold falls
from 66 to 49 particles. The diagnostic pins the divisor with
`compute.cudaKickHandoffBlocks=492` (the arena still launches 656 blocks) and
also tries no list reservation; two pairs of 11 samples, forces only:

lattice:

| bucket | baseline | arena auto (656 blocks) | arena, handoff 492 | arena, "residency 6" (ineffective) | arena, no list reservation |
|---:|---:|---:|---:|---:|---:|
| 32 | 33.71 ms | 91.29 ms, 0.369x (0.368-0.370) | 31.82 ms, 1.060x (1.054-1.065) | 91.29 ms, 0.369x (0.368-0.371) | 90.62 ms, 0.372x (0.369-0.375) |
| 64 | 33.15 ms | 31.73 ms, 1.045x (1.040-1.049) | 31.84 ms, 1.041x (1.025-1.058) | 31.53 ms, 1.051x (1.046-1.057) | 31.89 ms, 1.040x (1.037-1.042) |

clustered:

| bucket | baseline | arena auto (656 blocks) | arena, handoff 492 | arena, "residency 6" (ineffective) | arena, no list reservation |
|---:|---:|---:|---:|---:|---:|
| 32 | 9.45 ms | 11.46 ms, 0.824x (0.788-0.861) | 8.53 ms, 1.108x (1.062-1.154) | 11.16 ms, 0.847x (0.823-0.871) | 10.93 ms, 0.865x (0.813-0.918) |
| 64 | 8.82 ms | 8.39 ms, 1.051x (1.038-1.065) | 8.42 ms, 1.047x (1.046-1.048) | 8.32 ms, 1.060x (1.042-1.078) | 8.52 ms, 1.036x (1.017-1.055) |

adaptive:

| bucket | baseline | arena auto (656 blocks) | arena, handoff 492 | arena, "residency 6" (ineffective) | arena, no list reservation |
|---:|---:|---:|---:|---:|---:|
| 32 | 18.86 ms | 21.00 ms, 0.898x (0.890-0.906) | 19.19 ms, 0.983x (0.969-0.997) | 21.14 ms, 0.892x (0.880-0.905) | 21.95 ms, 0.859x (0.854-0.864) |
| 64 | 19.10 ms | 20.70 ms, 0.923x (0.908-0.938) | 19.94 ms, 0.958x (0.954-0.962) | 20.45 ms, 0.934x (0.920-0.948) | 21.09 ms, 0.906x (0.893-0.919) |

The handoff explains the bucket-32 loss entirely: with the baseline's divisor
the arena is 1.06x (lattice), 1.11x (clustered) and 0.98x (adaptive) at bucket
32, and at bucket 64 it recovers a third of the adaptive gap (0.923x to
0.958x), the host part of the Nsight split. Dropping the list reservation
costs 2-4% on the adaptive fixture, confirming the 4096-entry default.

The residency option did not do what the earlier variants assumed: a
`compute.cudaKickResidency=6` run printed `8 blocks/SM (register limit 8),
budget 16000`, because `kickSelectLayout` computed the budget for six blocks,
the default tiles fit in it, the occupancy calculator reported eight achievable
blocks and the layout recorded that number as its residency, so 656 blocks were
launched and the handoff divisor stayed at 656. Every "residency 6" and
"residency 7" measurement in this document before the residency sweep below
is therefore the automatic policy with a larger budget (the same layout at
bucket 64). `kickSelectLayout` now records the requested residency; the
residency sweep below measures the option for the first time.

### Automatic layout fix at large buckets (`layout_fix_bucket1024/`)

With `kickBuildLayout` restoring the automatic source tiles whenever it
shrinks the sink tile, the bucket-1024 layout becomes the 128-particle sink
tile with the default 512-particle and 38-multipole source tiles (8,928
bytes), and the bucket-512 layout is unchanged (512-particle sink tile, 224
particles, 17 multipoles). Two pairs of 11 samples against the baseline, forces
only, with the explicit 128-particle sink tile for comparison:

| fixture | bucket | baseline | arena auto (fixed layout) | arena, sink tile 128 |
|---|---:|---:|---:|---:|
| lattice | 512 | 11.59 ms | 12.62 ms, 0.917x (0.876-0.958) | 12.21 ms, 0.948x (0.895-1.001) |
| lattice | 1024 | 16.40 ms | 19.00 ms, 0.863x (0.861-0.865) | 19.77 ms, 0.830x (0.806-0.855) |
| clustered | 512 | 10.28 ms | 9.89 ms, 1.040x (1.022-1.059) | 10.28 ms, 1.000x (0.993-1.008) |
| clustered | 1024 | 16.41 ms | 16.49 ms, 0.996x (0.979-1.013) | 16.80 ms, 0.977x (0.956-0.998) |
| adaptive | 512 | 28.97 ms | 33.81 ms, 0.857x (0.852-0.862) | 34.60 ms, 0.837x (0.834-0.840) |
| adaptive | 1024 | 81.56 ms | 80.23 ms, 1.017x (1.015-1.018) | 80.14 ms, 1.018x (1.013-1.022) |

The automatic policy at bucket 1024 now matches the explicit tile (adaptive
1.017x against 0.844x before the fix) and its force errors against the direct
reference are unchanged (adaptive 7.53e-4, clustered 1.77e-3). The remaining
bucket-512 and lattice-1024 losses (0.86-0.92x) are the kernel-side per-leaf
throughput difference of the Nsight split, not the layout.

### Residency and handoff sweep (`residency_handoff/`)

With `kickSelectLayout` honouring the requested residency, residency 6 launches
492 blocks (6 per SM, budget 16,000 bytes, same tiles) and residency 7 574
blocks, and the handoff divisor follows the launch size. The sweep adds the
divisor pinned to 492, 328 and 164 blocks at full residency (subtrees of at
most 66, 100 and 200 particles for 32,768 particles); two pairs of 11 samples,
forces only, speedup against the baseline at the same bucket:

lattice:

| bucket | baseline (492 blocks) | arena auto (8/SM, 656) | residency 7 (574) | residency 6 (492) | handoff 492 | handoff 328 | handoff 164 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 34.22 ms | 94.17 ms, 0.363x (0.363-0.364) | 93.40 ms, 0.366x (0.364-0.369) | 32.47 ms, 1.054x (1.048-1.060) | 32.43 ms, 1.055x (1.039-1.071) | 32.79 ms, 1.043x (1.031-1.056) | 15.78 ms, 2.169x (2.111-2.227) |
| 64 | 34.29 ms | 32.77 ms, 1.046x (1.040-1.053) | 33.10 ms, 1.036x (1.027-1.044) | 32.73 ms, 1.047x (1.040-1.055) | 33.01 ms, 1.038x (1.026-1.051) | 32.71 ms, 1.048x (1.046-1.050) | 14.90 ms, 2.302x (2.238-2.365) |
| 128 | 16.01 ms | 15.38 ms, 1.041x (1.021-1.061) | 15.70 ms, 1.020x (1.016-1.025) | 15.42 ms, 1.039x (1.028-1.049) | 15.60 ms, 1.026x (1.023-1.029) | 15.14 ms, 1.058x (1.047-1.068) | 15.00 ms, 1.068x (1.037-1.098) |

clustered:

| bucket | baseline (492 blocks) | arena auto (8/SM, 656) | residency 7 (574) | residency 6 (492) | handoff 492 | handoff 328 | handoff 164 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 9.74 ms | 11.77 ms, 0.828x (0.820-0.836) | 11.38 ms, 0.856x (0.855-0.856) | 8.94 ms, 1.091x (1.049-1.133) | 8.84 ms, 1.103x (1.083-1.123) | 9.44 ms, 1.032x (1.027-1.036) | 6.59 ms, 1.480x (1.460-1.500) |
| 64 | 9.22 ms | 8.84 ms, 1.043x (1.042-1.044) | 8.95 ms, 1.031x (1.016-1.046) | 9.07 ms, 1.018x (0.997-1.039) | 8.86 ms, 1.044x (0.992-1.096) | 8.96 ms, 1.030x (1.017-1.042) | 6.23 ms, 1.480x (1.459-1.500) |
| 128 | 6.98 ms | 6.66 ms, 1.048x (1.023-1.073) | 6.95 ms, 1.006x (0.978-1.033) | 6.94 ms, 1.008x (0.951-1.066) | 6.73 ms, 1.038x (1.020-1.055) | 6.61 ms, 1.057x (1.041-1.073) | 6.41 ms, 1.089x (1.082-1.095) |

adaptive:

| bucket | baseline (492 blocks) | arena auto (8/SM, 656) | residency 7 (574) | residency 6 (492) | handoff 492 | handoff 328 | handoff 164 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 19.36 ms | 21.99 ms, 0.881x (0.879-0.882) | 22.89 ms, 0.846x (0.841-0.851) | 20.42 ms, 0.948x (0.934-0.962) | 20.09 ms, 0.963x (0.948-0.979) | 18.07 ms, 1.071x (1.051-1.092) | 16.77 ms, 1.155x (1.137-1.172) |
| 64 | 20.25 ms | 21.70 ms, 0.933x (0.926-0.940) | 20.99 ms, 0.964x (0.956-0.973) | 20.48 ms, 0.989x (0.979-0.998) | 20.64 ms, 0.981x (0.969-0.994) | 17.24 ms, 1.175x (1.155-1.194) | 16.27 ms, 1.245x (1.226-1.264) |
| 128 | 17.90 ms | 19.57 ms, 0.915x (0.910-0.920) | 19.71 ms, 0.909x (0.885-0.932) | 19.30 ms, 0.928x (0.923-0.932) | 19.72 ms, 0.908x (0.883-0.934) | 19.38 ms, 0.925x (0.893-0.958) | 16.56 ms, 1.081x (1.052-1.110) |

- Residency 6 and the pinned divisor 492 are the same thing for the handoff
  and measure the same: they remove the bucket-32 collapse (lattice 1.05x,
  clustered 1.09-1.10x, adaptive 0.95-0.96x) and leave buckets 64 and 128 as
  they were (lattice 1.03-1.05x, clustered 1.0-1.04x, adaptive 0.91-0.99x).
  Residency 7 (574 blocks, 57-particle threshold) keeps the collapse. Fewer
  resident blocks do not change the kernel time measurably at bucket 64 or
  128, so the pool-allocator contention of eight blocks per SM is not what
  the residency option can buy back; it only acts through the handoff.
- A coarser handoff is a large win for every fixture at this problem size:
  200-particle subtrees (divisor 164) run the lattice in 15 ms at buckets 32
  and 64 instead of 32-34 ms (2.2-2.3x), the clustered fixture 1.5x faster and
  the adaptive fixture 1.08-1.25x faster. The 2-thread CPU walk, not the GPU,
  bounds the kick: with 656 workitems of 50 particles the CPU refines
  checklists one or two levels deeper than it needs to. `handoff_coarse/` and
  `handoff_threads4/` below look for the optimum and check the effect with
  four CPU threads.

### Kernel attribution at bucket 512 (`ncu_bucket512/`)

Nsight Compute replay of the second force-only kick kernel on the adaptive
fixture at bucket 512 (`--set full`; 124 single-warp blocks for both binaries,
255 registers, 15,184 against 11,648 dynamic shared bytes):

| metric | baseline | arena | ratio |
|---|---:|---:|---:|
| kernel duration | 28.05 ms | 35.29 ms | 1.258 |
| warp instructions executed | 344.6 M | 334.9 M | 0.972 |
| issue-active cycles (% of active) | 23.7 | 20.9 | 0.885 |
| local-memory load sectors (spills) | 14.17 M | 0.79 M | 0.056 |
| local-memory store sectors | 14.81 M | 2.20 M | 0.149 |
| shared-memory load wavefronts | 19.8 M | 15.4 M | 0.777 |
| LSU pipe utilization (% of active) | 7.45 | 2.95 | 0.396 |
| stall, long scoreboard (per issue) | 0.65 | 0.43 | 0.672 |
| stall, short scoreboard (per issue) | 0.72 | 1.37 | 1.898 |
| stall, wait (per issue) | 1.29 | 1.45 | 1.123 |
| warps active (% of peak) | 2.56 | 2.66 | 1.041 |

The arena executes fewer instructions, spills an order of magnitude less and
loads less from shared memory, but each issued instruction waits almost twice
as long on short-scoreboard dependencies (shared-memory loads of the staged
sources and the reciprocal square root in the PP loop). With 64 leaves of up
to 512 particles the launch holds 124 warps for 82 multiprocessors, so no
other warp hides that latency and the kernel runs at the issue rate of a
single warp; the baseline's PP loop, whose staged sources sit at compile-time
shared offsets, schedules better in that regime. The PP source code is the
same in both kernels, so the fix would be an instruction-level one (unrolling
the source loop or processing two sinks per lane). The regime itself,
fewer workitems than warps the GPU can hold, belongs to small problems: at
buckets 32 to 128 every SM holds its full complement of blocks and the arena
kernel is level with or faster than the baseline.

### Where the handoff optimum lies (`handoff_coarse/`)

Divisors of 164, 82, 41 and 20 blocks (subtrees of at most 200, 400, 800 and
1,600 particles) at buckets 64 and 128, two pairs of 11 samples, forces only.
A first run of this stage coincided with other load on the workstation (every
baseline time 50% above its other measurements, pair ranges of 2x) and is kept
as `handoff_coarse_contaminated/`; the table is the clean repeat.

lattice:

| bucket | baseline (492) | arena auto (656) | handoff 164 (200) | handoff 82 (400) | handoff 41 (800) | handoff 20 (1,600) |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 32.88 ms | 31.30 ms, 1.051x (1.041-1.060) | 14.51 ms, 2.266x (2.241-2.291) | 11.92 ms, 2.761x (2.707-2.814) | 12.50 ms, 2.632x (2.612-2.651) | 19.34 ms, 1.700x (1.665-1.736) |
| 128 | 15.37 ms | 15.07 ms, 1.020x (1.017-1.023) | 14.33 ms, 1.073x (1.062-1.084) | 11.83 ms, 1.300x (1.265-1.336) | 12.06 ms, 1.275x (1.260-1.289) | 19.04 ms, 0.807x (0.801-0.814) |

clustered:

| bucket | baseline (492) | arena auto (656) | handoff 164 (200) | handoff 82 (400) | handoff 41 (800) | handoff 20 (1,600) |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 8.75 ms | 8.61 ms, 1.016x (1.015-1.017) | 6.00 ms, 1.458x (1.455-1.462) | 5.46 ms, 1.604x (1.595-1.613) | 5.13 ms, 1.709x (1.643-1.775) | 6.81 ms, 1.286x (1.253-1.319) |
| 128 | 6.81 ms | 6.40 ms, 1.064x (1.041-1.086) | 6.08 ms, 1.119x (1.103-1.135) | 5.94 ms, 1.148x (1.100-1.195) | 5.98 ms, 1.138x (1.133-1.143) | 9.01 ms, 0.755x (0.745-0.765) |

adaptive:

| bucket | baseline (492) | arena auto (656) | handoff 164 (200) | handoff 82 (400) | handoff 41 (800) | handoff 20 (1,600) |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 19.34 ms | 20.91 ms, 0.925x (0.920-0.929) | 15.69 ms, 1.232x (1.225-1.240) | 20.35 ms, 0.950x (0.931-0.969) | 34.81 ms, 0.556x (0.543-0.569) | 62.68 ms, 0.308x (0.303-0.314) |
| 128 | 17.22 ms | 18.61 ms, 0.925x (0.921-0.929) | 15.50 ms, 1.112x (1.082-1.141) | 19.68 ms, 0.875x (0.868-0.882) | 34.15 ms, 0.504x (0.503-0.506) | 63.68 ms, 0.270x (0.267-0.274) |

The adaptive fixture sets the limit: 200-particle subtrees are its optimum
(1.23x at bucket 64, 1.11x at 128), 400 already fall below the baseline, and
800 or 1,600 leave the GPU with too few, too unequal workitems (0.5x and
0.3x). The lattice and clustered fixtures keep gaining down to 400-800
particles (2.6-2.8x and 1.6-1.7x at bucket 64) but 200 particles takes most
of it (2.3x and 1.5x). The harness pinned two worker threads when this was
measured; the sweep over thread counts and particle counts
(`handoff_threads_sizes/`, below) measures the balance with 4 to 12 threads
and at 262,144 particles.

### Selected defaults

- Handoff: `kickBlockCount()` no longer returns the launch size. It divides
  the particles by `kKickHandoffBlocksPerMultiprocessor = 2` times the
  multiprocessor count (164 blocks on the RTX 3090, 200-particle subtrees for
  32,768 particles), the measured optimum of the adaptive fixture and within
  10-20% of the best division of the other two; `compute.cudaKickHandoffBlocks`
  still pins it. This replaces both the arena's 656-block divisor (which lost
  the bucket-32 case 2.7x) and the baseline's 492. The sweep over thread and
  particle counts below (`handoff_threads_sizes/`) later made the divisor
  grow as the cube root of the particle count, to four blocks per
  multiprocessor at 262,144 particles and above; the final check of this
  section measured the 164-block rule.
- Residency: the register limit (8 blocks per SM at orders 6 to 9) stays the
  default. Lowering it to 6 or 7 does not change the kernel time at any bucket
  once the handoff is separated from it, and the option now really launches
  that many blocks.
- List reservation: 4096 entries stays; removing it costs 2-4% on the adaptive
  fixture at buckets 32 and 64. It now sizes the per-launch list slab (one
  region per block, pool growth only on overflow) instead of twelve pool
  allocations per block per launch; see the list slab section below.
- Shared expansion and shared sink forces stay; the automatic sink tile falls
  back to 128 particles with full-size source tiles when the whole bucket does
  not fit (bucket 1024 at the order-7 budget).

### Final check of the selected defaults (`final_defaults/`)

The final build (the three fixes above, automatic layout, 8 blocks per SM,
4096-entry list reservation, two-blocks-per-SM handoff) against the baseline
at buckets 32 to 256, three pairs of 15 samples, direct reference over 1,024
particles. The third column pins the arena to the baseline's 492-block
division to separate the handoff from the arena itself. The CUDA `fmmWalk`,
`fmmTree`, `sphGravity` and the CPU `fmmWalk` regressions pass with this
build (`validation/ctest_cap1024_final.log`).

lattice (forces only):

| bucket | baseline | arena, selected defaults (164-block handoff) | arena, baseline handoff (492) |
|---:|---:|---:|---:|
| 32 | 33.20 ms | 14.65 ms, 2.283x (2.236-2.291) | 31.28 ms, 1.061x (1.036-1.076) |
| 64 | 32.16 ms | 14.45 ms, 2.225x (2.216-2.257) | 30.97 ms, 1.039x (1.034-1.041) |
| 128 | 14.66 ms | 14.43 ms, 1.024x (1.007-1.069) | 14.21 ms, 1.027x (1.018-1.085) |
| 256 | 11.58 ms | 11.75 ms, 0.963x (0.912-1.005) | 11.72 ms, 0.966x (0.909-0.999) |

clustered (forces only):

| bucket | baseline | arena, selected defaults (164-block handoff) | arena, baseline handoff (492) |
|---:|---:|---:|---:|
| 32 | 8.73 ms | 6.14 ms, 1.388x (1.349-1.525) | 8.06 ms, 1.082x (1.026-1.163) |
| 64 | 8.11 ms | 5.88 ms, 1.380x (1.379-1.532) | 7.88 ms, 1.053x (1.009-1.168) |
| 128 | 6.13 ms | 6.08 ms, 1.007x (1.005-1.046) | 5.97 ms, 1.025x (1.008-1.057) |
| 256 | 6.81 ms | 6.71 ms, 1.029x (1.015-1.099) | 6.83 ms, 0.995x (0.986-1.164) |

adaptive (forces only):

| bucket | baseline | arena, selected defaults (164-block handoff) | arena, baseline handoff (492) |
|---:|---:|---:|---:|
| 32 | 18.17 ms | 14.42 ms, 1.273x (1.215-1.350) | 18.09 ms, 1.013x (1.002-1.041) |
| 64 | 18.35 ms | 13.80 ms, 1.347x (1.283-1.367) | 19.05 ms, 0.973x (0.946-0.978) |
| 128 | 15.77 ms | 13.59 ms, 1.170x (1.147-1.189) | 17.34 ms, 0.919x (0.871-0.929) |
| 256 | 17.91 ms | 20.53 ms, 0.884x (0.861-0.886) | 21.42 ms, 0.825x (0.824-0.903) |

adaptive (forces + potential):

| bucket | baseline | arena, selected defaults (164-block handoff) | arena, baseline handoff (492) |
|---:|---:|---:|---:|
| 32 | 18.43 ms | 14.61 ms, 1.258x (1.210-1.325) | 18.52 ms, 0.995x (0.968-1.000) |
| 64 | 18.56 ms | 14.56 ms, 1.288x (1.250-1.308) | 19.65 ms, 0.952x (0.930-0.959) |
| 128 | 16.10 ms | 13.82 ms, 1.157x (1.136-1.187) | 17.69 ms, 0.908x (0.890-0.910) |
| 256 | 18.37 ms | 20.96 ms, 0.886x (0.869-0.886) | 22.08 ms, 0.826x (0.810-0.874) |

With the selected defaults the arena is 2.2-2.3x faster than the baseline on
the lattice at buckets 32 and 64, 1.4x on the clustered fixture and 1.3x on
the adaptive one, and 1.0-1.2x at bucket 128; nearly all of that is the
handoff. At the same division (third column) the arena itself is 3-6% faster
on the lattice and clustered fixtures at buckets 32 to 128, level on the
adaptive fixture at bucket 32 and 3-8% slower at buckets 64 and 128, the
kernel-side pool-allocator cost. At bucket 256 the bucket exceeds the
200-particle handoff for every binary, so the division is the same, and the
arena is 4% slower on the lattice, level on the clustered fixture and 12%
slower on the adaptive fixture: 128 to 256 workitems for 656 launched blocks
is the beginning of the low-occupancy regime attributed at bucket 512. The
force and potential errors against the direct reference agree between the
two binaries to 1.7e-05 relative, and the force exports differ at
roundoff (acceleration relative L2 at most 2.1e-7 adaptive, 4.3e-7
clustered, 1.1e-6 lattice at bucket 256), since the coarser handoff moves
cell interactions between the SIMD CPU walk and the GPU.

### Handoff at larger thread and particle counts (`handoff_threads_sizes/`)

The harness now runs on the configured `--hpx:threads` and builds its
fixtures from `initial.particlesPerDimension`, so the handoff divisor can be
measured away from two threads and 32,768 particles. Divisors of 82, 164,
328, 492 and 656 blocks (1, 2, 4, 6 and 8 blocks per multiprocessor) at 2,
4, 8 and 12 worker threads (pinned to CPUs 1 through the thread count, the
distinct physical cores of the 12-core workstation) for 32,768 and 262,144
particles (sides 32 and 64), all three fixtures, bucket 128, two pairs of 11
samples, forces and potential (the tables show forces only; the potential
rows agree). The binary includes the open-larger-cell walk of commit
92d735c6 from `main`, so absolute times are not comparable with the earlier
tables (the clustered fixture at bucket 128 takes 7.8 ms here against 6.1 ms
above with the same division); every comparison here is within one binary.
The column headers give the handoff threshold `max(N / divisor, bucket)`.
At 32,768 particles the bucket clamps the three largest divisors to the same
128-particle threshold, so their three columns are repeat measurements of one
division and show the noise floor (2-4%). At 262,144 particles the
thresholds are 3,197, 1,598, 799, 532 and 399 particles; the lattice
(2,048 leaves of exactly 128) hands off nodes of 2,048, 1,024, 512, 512 and
256 particles, that is 128, 256, 512, 512 and 1,024 workitems for 656
resident blocks.

Other sessions were building and timing on the workstation during this
sweep. A stage whose pairs disagreed by more than 20% (clean stages stay
within 16%) was discarded (kept under `handoff_threads_sizes_contaminated/`)
and repeated behind a shared GPU lock, a quiet half minute without builds or
GPU processes (`nvidia-smi`; a running cosmictiger is named `main-thread#0`
by HPX, so process names do not identify it) and the per-process idle wait of
the driver; `sweep_summary.md` lists each final stage's load average and worst
pair spread (`benchmarks/check_kick_arena_pairs.py`,
`benchmarks/summarize_kick_handoff_sweep.py`).

Speedup is the 164-block time over the variant's time, pairwise.

lattice (32,768 particles, forces only):

| threads | 82 (399) | 164 (199) | 328 (128) | 492 (128) | 656 (128) |
|---:|---:|---:|---:|---:|---:|
| 2 | 12.23 ms, 1.327x (1.276-1.377) | 16.20 ms | 16.70 ms, 0.970x (0.965-0.976) | 16.11 ms, 1.006x (0.988-1.024) | 16.31 ms, 0.993x (0.984-1.003) |
| 4 | 10.56 ms, 1.123x (1.076-1.169) | 11.83 ms | 11.84 ms, 1.000x (0.955-1.045) | 12.16 ms, 0.974x (0.942-1.006) | 12.06 ms, 0.984x (0.921-1.048) |
| 8 | 9.67 ms, 1.035x (0.994-1.075) | 10.00 ms | 9.96 ms, 1.004x (0.987-1.022) | 9.94 ms, 1.006x (0.998-1.013) | 10.03 ms, 0.997x (0.974-1.020) |
| 12 | 10.39 ms, 0.949x (0.946-0.952) | 9.86 ms | 10.29 ms, 0.959x (0.944-0.974) | 9.75 ms, 1.011x (1.008-1.014) | 10.28 ms, 0.960x (0.927-0.992) |

clustered (32,768 particles, forces only):

| threads | 82 (399) | 164 (199) | 328 (128) | 492 (128) | 656 (128) |
|---:|---:|---:|---:|---:|---:|
| 2 | 6.49 ms, 1.203x (1.165-1.241) | 7.81 ms | 7.89 ms, 0.990x (0.963-1.018) | 7.82 ms, 0.999x (0.998-0.999) | 7.76 ms, 1.007x (0.984-1.030) |
| 4 | 5.70 ms, 1.047x (1.038-1.055) | 5.97 ms | 5.99 ms, 0.996x (0.969-1.023) | 6.20 ms, 0.964x (0.911-1.018) | 6.01 ms, 0.993x (0.952-1.035) |
| 8 | 5.88 ms, 0.976x (0.960-0.992) | 5.74 ms | 5.67 ms, 1.012x (1.012-1.012) | 5.72 ms, 1.004x (0.974-1.034) | 5.67 ms, 1.012x (0.980-1.044) |
| 12 | 6.33 ms, 0.998x (0.978-1.019) | 6.32 ms | 6.00 ms, 1.054x (1.051-1.057) | 6.09 ms, 1.038x (1.015-1.061) | 6.07 ms, 1.041x (1.019-1.064) |

adaptive (32,768 particles, forces only):

| threads | 82 (399) | 164 (199) | 328 (128) | 492 (128) | 656 (128) |
|---:|---:|---:|---:|---:|---:|
| 2 | 19.92 ms, 0.829x (0.824-0.835) | 16.52 ms | 19.84 ms, 0.833x (0.813-0.853) | 19.86 ms, 0.832x (0.823-0.841) | 19.96 ms, 0.828x (0.813-0.842) |
| 4 | 18.88 ms, 0.735x (0.686-0.785) | 13.85 ms | 16.00 ms, 0.871x (0.789-0.953) | 16.56 ms, 0.836x (0.820-0.852) | 15.53 ms, 0.892x (0.859-0.925) |
| 8 | 18.85 ms, 0.738x (0.734-0.743) | 13.92 ms | 13.66 ms, 1.020x (0.989-1.051) | 14.13 ms, 0.985x (0.975-0.996) | 14.42 ms, 0.967x (0.926-1.008) |
| 12 | 19.47 ms, 0.712x (0.688-0.736) | 13.87 ms | 14.05 ms, 0.988x (0.940-1.036) | 13.78 ms, 1.006x (0.968-1.044) | 14.23 ms, 0.974x (0.948-1.000) |

lattice (262,144 particles, forces only):

| threads | 82 (3,196) | 164 (1,598) | 328 (799) | 492 (532) | 656 (399) |
|---:|---:|---:|---:|---:|---:|
| 2 | 38.68 ms, 0.853x (0.808-0.897) | 33.00 ms | 47.81 ms, 0.690x (0.643-0.738) | 49.18 ms, 0.671x (0.627-0.715) | 104.44 ms, 0.316x (0.297-0.335) |
| 4 | 33.41 ms, 0.787x (0.743-0.831) | 26.25 ms | 31.50 ms, 0.834x (0.799-0.868) | 31.71 ms, 0.828x (0.793-0.863) | 58.10 ms, 0.452x (0.438-0.465) |
| 8 | 32.94 ms, 0.700x (0.669-0.731) | 23.01 ms | 23.50 ms, 0.979x (0.974-0.985) | 23.70 ms, 0.971x (0.956-0.986) | 36.56 ms, 0.629x (0.628-0.631) |
| 12 | 34.90 ms, 0.665x (0.660-0.671) | 23.22 ms | 22.95 ms, 1.012x (0.983-1.042) | 22.16 ms, 1.048x (1.046-1.050) | 32.38 ms, 0.717x (0.701-0.734) |

clustered (262,144 particles, forces only):

| threads | 82 (3,196) | 164 (1,598) | 328 (799) | 492 (532) | 656 (399) |
|---:|---:|---:|---:|---:|---:|
| 2 | 25.44 ms, 0.757x (0.698-0.815) | 19.19 ms | 18.28 ms, 1.050x (1.003-1.097) | 17.55 ms, 1.097x (1.009-1.185) | 20.39 ms, 0.943x (0.878-1.008) |
| 4 | 23.24 ms, 0.744x (0.720-0.768) | 17.30 ms | 14.98 ms, 1.156x (1.100-1.211) | 15.81 ms, 1.094x (1.062-1.127) | 17.03 ms, 1.016x (1.012-1.019) |
| 8 | 23.26 ms, 0.686x (0.672-0.700) | 15.93 ms | 14.57 ms, 1.095x (1.015-1.174) | 14.39 ms, 1.106x (1.072-1.139) | 15.53 ms, 1.027x (0.948-1.107) |
| 12 | 23.99 ms, 0.727x (0.703-0.750) | 17.43 ms | 15.98 ms, 1.091x (1.050-1.132) | 15.31 ms, 1.138x (1.128-1.149) | 17.05 ms, 1.023x (0.982-1.063) |

adaptive (262,144 particles, forces only):

| threads | 82 (3,196) | 164 (1,598) | 328 (799) | 492 (532) | 656 (399) |
|---:|---:|---:|---:|---:|---:|
| 2 | 44.88 ms, 0.674x (0.656-0.692) | 30.23 ms | 28.50 ms, 1.061x (1.039-1.083) | 29.44 ms, 1.028x (0.995-1.061) | 36.27 ms, 0.834x (0.824-0.843) |
| 4 | 41.83 ms, 0.646x (0.625-0.666) | 27.00 ms | 22.77 ms, 1.186x (1.166-1.206) | 23.66 ms, 1.141x (1.105-1.177) | 25.22 ms, 1.071x (1.031-1.111) |
| 8 | 40.07 ms, 0.637x (0.628-0.646) | 25.50 ms | 21.29 ms, 1.198x (1.187-1.209) | 21.84 ms, 1.168x (1.127-1.209) | 22.47 ms, 1.135x (1.103-1.166) |
| 12 | 43.57 ms, 0.621x (0.620-0.623) | 27.08 ms | 21.68 ms, 1.250x (1.215-1.285) | 22.97 ms, 1.179x (1.161-1.198) | 21.42 ms, 1.264x (1.256-1.273) |

Time relative to the 164-block division, geometric mean over the six cases
(three fixtures, potential off and on) and, in parentheses, the worst case;
below one is faster:

| particles | threads | handoff82 | handoff164 | handoff328 | handoff492 | handoff656 |
|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | 2 | 0.930 (1.279) | 1.000 (1.000) | 1.074 (1.201) | 1.057 (1.202) | 1.061 (1.208) |
| 32,768 | 4 | 1.068 (1.452) | 1.000 (1.000) | 1.058 (1.198) | 1.068 (1.196) | 1.057 (1.187) |
| 32,768 | 8 | 1.119 (1.391) | 1.000 (1.000) | 0.993 (1.030) | 1.008 (1.044) | 1.011 (1.047) |
| 32,768 | 12 | 1.149 (1.419) | 1.000 (1.000) | 1.011 (1.043) | 1.003 (1.070) | 1.015 (1.043) |
| 262,144 | 2 | 1.327 (1.485) | 1.000 (1.000) | 1.071 (1.449) | 1.078 (1.490) | 1.540 (3.165) |
| 262,144 | 4 | 1.422 (1.549) | 1.000 (1.000) | 0.951 (1.213) | 0.991 (1.232) | 1.258 (2.216) |
| 262,144 | 8 | 1.490 (1.599) | 1.000 (1.000) | 0.906 (1.021) | 0.918 (1.042) | 1.106 (1.622) |
| 262,144 | 12 | 1.497 (1.611) | 1.000 (1.000) | 0.881 (0.988) | 0.873 (0.954) | 1.018 (1.394) |

- 32,768 particles: two blocks per multiprocessor (200-particle subtrees)
  stays the best or tied division at every thread count. The adaptive fixture
  needs it: 400-particle subtrees (82 blocks) lose 1.2-1.4x at every thread
  count, and the 128-particle division loses 1.1-1.2x at 2-4 threads and is
  level at 8-12. The coarser division's gain on the lattice and clustered
  fixtures (1.33x and 1.20x at 2 threads) shrinks to 1.12x and 1.05x at 4
  threads and is gone at 8-12 (0.95-1.04x), so more threads do not move the
  optimum here; they only remove the CPU-bound penalty of finer divisions.
- 262,144 particles: the 164-block division (256 workitems) under-fills the
  GPU. Four blocks per multiprocessor (328, 512 workitems) is 1.19-1.25x
  faster on the adaptive fixture and 1.09-1.16x on the clustered one from 4
  threads up, and level or better on the lattice from 8 threads (0.98x at 8,
  1.01x at 12). Six blocks (492) gives the lattice the same 512 workitems and
  matches 328 within a few percent (4-5% better on the lattice at 12 threads,
  worse on the adaptive fixture at 4-8). Eight blocks (656, 1,024 workitems)
  ties 328 on the adaptive fixture at 12 threads but stays CPU-bound on the
  lattice at every thread count (0.32x at 2, 0.45x at 4, 0.63x at 8, 0.72x at
  12): the CPU walk cost grows with the workitem count divided by the thread
  count, while the GPU gain saturates once the launch is full. At 2 threads
  the coarser 164 remains the better compromise (328 gains 5-11% on the
  clustered and adaptive fixtures and loses 31% on the lattice).
- The thread count therefore does not set the optimum from 4 threads up; the
  particle count does, through the number of workitems the GPU gets. The
  divisor should grow with the particle count: by a factor two between the
  two measured sizes, which is the cube root of their ratio.

Rule (implemented in `kickBlockCount(particles)`, `src/gravity/kick.cu`): the divisor
is `kKickHandoffBlocksPerMultiprocessor = 2` times the multiprocessor count,
times the cube root of the particle count over
`kKickHandoffReferenceParticles = 32,768`, clamped between 1 and
`kKickHandoffMaxScale = 2`: 164 blocks at or below 32,768 particles, 328 at or
above 262,144, interpolated between (238 at 100,000). The threshold
`N / divisor` then grows as `N^(2/3)`: the CPU walks one level deeper for
every eightfold increase in particles and the GPU receives twice the
workitems. Above 262,144 particles the divisor is held at four blocks per
multiprocessor, the largest measured value, so the 16,384-particle cap of
`CUDA_KICK_PARTS_MAX` decides above about 5.4 million particles per rank.
The thread count does not enter the rule: at 2 threads it costs the lattice
at 262,144 particles, but production runs use the workers the node has.
`compute.cudaKickHandoffBlocks` still pins the divisor. The sweep at 2.1
million particles below raises `kKickHandoffMaxScale` to 3 (six blocks per
multiprocessor from 884,736 particles).

### Check of the cube-root rule (`handoff_rule_check/`)

The rebuilt binary (`validation/ctest_cap1024_handoff_rule.log`: the CUDA
`fmmWalk`, `fmmTree` and `sphGravity` regressions and the CPU `fmmWalk` pass)
with the automatic divisor against the pinned 164 and 328 at both particle
counts, 8 worker threads, bucket 128, order 7, two pairs of 11 samples (the
32,768-particle stage repeated with three pairs). Speedup is the pinned
division's time over the automatic rule's time, pairwise (above one: the
rule is faster).

32,768 particles, 8 threads, bucket 128 (`t8_n32/`):

| fixture | potential | pinned 164 | pinned 328 | automatic rule |
|---|---:|---:|---:|---:|
| lattice | 0 | 10.14 ms, 0.938x (0.905-0.970) | 10.08 ms, 0.933x (0.891-0.975) | 10.82 ms |
| lattice | 1 | 9.99 ms, 0.956x (0.951-0.961) | 10.14 ms, 0.970x (0.954-0.987) | 10.45 ms |
| clustered | 0 | 5.72 ms, 1.008x (0.986-1.030) | 5.77 ms, 1.017x (0.987-1.047) | 5.67 ms |
| clustered | 1 | 5.98 ms, 0.904x (0.898-0.911) | 6.16 ms, 0.932x (0.926-0.939) | 6.61 ms |
| adaptive | 0 | 14.10 ms, 1.011x (0.997-1.024) | 14.73 ms, 1.057x (1.020-1.094) | 13.95 ms |
| adaptive | 1 | 13.72 ms, 0.974x (0.939-1.009) | 14.02 ms, 0.995x (0.965-1.025) | 14.09 ms |

262,144 particles, 8 threads, bucket 128 (`t8_n64/`):

| fixture | potential | pinned 164 | pinned 328 | automatic rule |
|---|---:|---:|---:|---:|
| lattice | 0 | 25.56 ms, 1.088x (1.039-1.137) | 24.55 ms, 1.045x (0.994-1.096) | 23.52 ms |
| lattice | 1 | 24.81 ms, 0.999x (0.963-1.035) | 24.37 ms, 0.981x (0.961-1.001) | 24.86 ms |
| clustered | 0 | 16.77 ms, 1.153x (1.126-1.179) | 14.85 ms, 1.020x (0.989-1.051) | 14.56 ms |
| clustered | 1 | 19.39 ms, 1.177x (1.114-1.241) | 16.20 ms, 0.983x (0.982-0.984) | 16.49 ms |
| adaptive | 0 | 26.37 ms, 1.248x (1.203-1.292) | 21.67 ms, 1.025x (1.024-1.026) | 21.14 ms |
| adaptive | 1 | 28.64 ms, 1.232x (1.209-1.254) | 23.36 ms, 1.004x (0.988-1.020) | 23.26 ms |

32,768 particles, 8 threads, three pairs, bucket 128 (`t8_n32_repeat/`):

| fixture | potential | pinned 164 | pinned 328 | automatic rule |
|---|---:|---:|---:|---:|
| lattice | 0 | 10.61 ms, 1.084x (1.043-1.230) | 10.00 ms, 0.983x (0.927-1.069) | 10.18 ms |
| lattice | 1 | 10.08 ms, 1.030x (1.021-1.041) | 9.82 ms, 0.991x (0.989-1.070) | 9.82 ms |
| clustered | 0 | 5.73 ms, 0.993x (0.987-0.993) | 5.82 ms, 1.008x (0.990-1.055) | 5.77 ms |
| clustered | 1 | 5.90 ms, 0.978x (0.926-1.028) | 5.89 ms, 0.984x (0.930-0.993) | 5.93 ms |
| adaptive | 0 | 13.16 ms, 0.983x (0.961-1.020) | 13.76 ms, 1.020x (1.005-1.049) | 13.49 ms |
| adaptive | 1 | 13.66 ms, 0.989x (0.920-1.002) | 14.00 ms, 1.014x (0.991-1.015) | 13.91 ms |

At 262,144 particles the rule reproduces the pinned 328 within 2-4% and is
1.15-1.25x faster than the previous 164-block default on the clustered and
adaptive fixtures and level to 1.09x faster on the lattice. At 32,768
particles the rule is the 164-block division by construction; the first
stage put it 5-10% behind both pinned divisions on the lattice and the
clustered potential case, the three-pair repeat puts it level with both
within 1-3%, so that was run noise (one pair of the repeat's pinned-164
lattice run was slow, hence its 1.23 range).

After merging `main` (34d4aa88 keeps empty source cells out of the CPU
walk, 327555a8 replaces `FastFuture`), the rebuilt binary passes the same
regressions (`validation/ctest_cap1024_handoff_rule_merged.log`) and the
re-check at 262,144 particles (`run_merged.sh`) gives the same picture: the
rule within 1-3% of the pinned 328 and 1.05-1.22x faster than 164 on the
clustered and adaptive fixtures, level on the lattice.

262,144 particles, 8 threads, after merging main, bucket 128 (`t8_n64_merged/`):

| fixture | potential | pinned 164 | pinned 328 | automatic rule |
|---|---:|---:|---:|---:|
| lattice | 0 | 24.48 ms, 1.042x (0.998-1.086) | 23.83 ms, 1.014x (0.991-1.037) | 23.52 ms |
| lattice | 1 | 24.46 ms, 1.007x (0.990-1.025) | 25.06 ms, 1.032x (1.012-1.053) | 24.29 ms |
| clustered | 0 | 15.80 ms, 1.050x (1.017-1.083) | 14.86 ms, 0.987x (0.919-1.054) | 15.05 ms |
| clustered | 1 | 18.18 ms, 1.152x (1.121-1.182) | 15.26 ms, 0.967x (0.961-0.973) | 15.78 ms |
| adaptive | 0 | 24.59 ms, 1.174x (1.091-1.257) | 20.61 ms, 0.982x (0.964-0.999) | 21.00 ms |
| adaptive | 1 | 26.83 ms, 1.216x (1.137-1.295) | 22.75 ms, 1.030x (0.999-1.061) | 22.10 ms |

### Handoff at 2.1 million particles (`handoff_large_sizes/`)

The rule above held four blocks per multiprocessor above 262,144 particles
only because larger sizes were unmeasured. This sweep takes the same binary
code (`main` at dd4b4b8b, frozen as `cosmictiger_b1024_handoff_large`) to
2,097,152 particles (side 128) with divisors of 328 (the rule's value there),
492, 656, 984 and 1312 blocks (4, 6, 8, 12 and 16 blocks per
multiprocessor) at 8 and 12 worker threads, all three fixtures, bucket 128,
order 7, two pairs of 11 samples, forces and potential (the tables show
forces; the potential rows agree). The thresholds `N / divisor` are 6,394,
4,262, 3,197, 2,131 and 1,598 particles, all below the 16,384-particle cap
of `CUDA_KICK_PARTS_MAX`; the lattice (16,384 leaves of exactly 128) hands
off nodes of 4,096, 4,096, 2,048, 2,048 and 1,024 particles, that is 512,
512, 1,024, 1,024 and 2,048 workitems, so 328 and 492 divide the lattice
identically and so do 656 and 984. A process takes 35-48 s (kicks of 65-115
ms). The stages ran under the same lock, quiet gate and pair-spread check as
the sweep above (`run.sh`, `driver.log`, `sweep_summary.md`); the 8-thread
stage was clean at once (worst pair spread 1.09), the 12-thread stage was
discarded twice for desktop activity during the runs (spreads 1.22 and 1.31,
kept under `handoff_large_sizes_contaminated/`) and accepted on the third
attempt (1.16). Its workers include CPU 12, the sibling of core 0 where the
desktop's threads run, which makes that stage the more sensitive one. The
force exports of these stages (900 MB per process at this size, 18 GB per
stage) were deleted after the pair-spread check; `runs.json` keeps each
export's sha256.

Speedup is the 328-block time over the variant's time, pairwise.

lattice (2,097,152 particles, forces only):

| threads | 328 (6,394) | 492 (4,262) | 656 (3,197) | 984 (2,131) | 1312 (1,598) |
|---:|---:|---:|---:|---:|---:|
| 8 | 81.81 ms | 83.29 ms, 0.982x (0.960-1.005) | 100.78 ms, 0.812x (0.789-0.835) | 104.36 ms, 0.784x (0.753-0.816) | 103.52 ms, 0.790x (0.774-0.807) |
| 12 | 84.76 ms | 77.91 ms, 1.090x (0.992-1.188) | 95.76 ms, 0.884x (0.863-0.905) | 95.91 ms, 0.883x (0.846-0.919) | 92.09 ms, 0.920x (0.865-0.975) |

clustered (2,097,152 particles, forces only):

| threads | 328 (6,394) | 492 (4,262) | 656 (3,197) | 984 (2,131) | 1312 (1,598) |
|---:|---:|---:|---:|---:|---:|
| 8 | 76.40 ms | 77.03 ms, 0.992x (0.992-0.992) | 67.95 ms, 1.126x (1.087-1.164) | 65.22 ms, 1.171x (1.156-1.187) | 63.81 ms, 1.198x (1.168-1.228) |
| 12 | 76.52 ms | 72.91 ms, 1.054x (0.959-1.148) | 71.41 ms, 1.072x (1.065-1.079) | 66.65 ms, 1.150x (1.125-1.174) | 70.24 ms, 1.090x (1.029-1.151) |

adaptive (2,097,152 particles, forces only):

| threads | 328 (6,394) | 492 (4,262) | 656 (3,197) | 984 (2,131) | 1312 (1,598) |
|---:|---:|---:|---:|---:|---:|
| 8 | 91.61 ms | 80.77 ms, 1.136x (1.070-1.201) | 75.32 ms, 1.217x (1.161-1.272) | 77.74 ms, 1.178x (1.136-1.221) | 78.94 ms, 1.160x (1.118-1.202) |
| 12 | 88.23 ms | 76.37 ms, 1.155x (1.144-1.166) | 78.04 ms, 1.130x (1.112-1.149) | 78.07 ms, 1.130x (1.105-1.154) | 79.68 ms, 1.107x (1.082-1.132) |

Time relative to the 328-block division, geometric mean over the six cases
(three fixtures, potential off and on) and, in parentheses, the worst case;
below one is faster:

| particles | threads | handoff328 | handoff492 | handoff656 | handoff984 | handoff1312 |
|---:|---:|---:|---:|---:|---:|---:|
| 2,097,152 | 8 | 1.000 (1.000) | 0.962 (1.026) | 0.969 (1.232) | 0.961 (1.276) | 0.952 (1.265) |
| 2,097,152 | 12 | 1.000 (1.000) | 0.919 (0.955) | 0.964 (1.130) | 0.944 (1.131) | 0.938 (1.086) |

- Four blocks per multiprocessor (328: 512 workitems of up to 6,394
  particles) is no longer the best division at this size. Its times at 8 and
  12 threads agree within 4% on every fixture, so the kick is GPU-bound
  there, and every finer division is faster on the clustered and adaptive
  fixtures.
- Six blocks (492) is 1.14-1.16x faster on the adaptive fixture at both
  thread counts, level (8 threads) to 1.05x (12) on the clustered one, and
  level on the lattice at 8 threads (0.98x forces, 1.01x potential: the same
  512-workitem division as 328) and 1.05-1.09x faster at 12. Its geometric
  mean is 0.962 at 8 threads with a worst case of 1.026, and 0.919 at 12
  threads with no case slower than 328: the only division that never loses.
- Eight or more blocks (656, 984, 1312: 1,024-2,048 workitems) gain
  1.05-1.20x on the clustered fixture and 1.11-1.22x on the adaptive one but
  lose on the lattice: 0.78-0.87x at 8 threads and 0.88-0.95x at 12. Unlike
  at 262,144 particles, where the 656-block lattice penalty shrank from
  0.63x at 8 threads to 0.72x at 12, four more threads barely move it here,
  so at this size it is not the CPU walk's thread budget. A per-workitem
  cost that does not shrink with the workitem (the uniform lattice carries
  the full far-field and Ewald lists for every node) doubles with the
  workitem count; that is the likely cause but was not profiled.
- Among 656, 984 and 1312 there is no monotonic trend: their geometric means
  lie within 2% of each other at each thread count and the clustered
  optimum moves between them from pair to pair.

Rule: `kKickHandoffMaxScale` is raised from 2 to 3. The divisor is still two
blocks per multiprocessor times the cube root of the particle count over
32,768, now clamped between 1 and 3: 164 blocks at or below 32,768
particles, 328 at 262,144, 492 at or above 884,736 (27 times the reference
count; 416 at 524,288). The threshold `N / divisor` grows as `N^(2/3)` up to
884,736 particles and linearly beyond, and the 16,384-particle cap of
`CUDA_KICK_PARTS_MAX` decides above about 8.1 million particles per rank
(492 times 16,384). A cap of 4 (656 blocks at 2.1 million) was not taken:
it is within 1% of 492 on the geometric mean at 8 threads and 5% behind it
at 12, with a 12-22% loss on the lattice. The clustered fixture's further
gain from 12-16 blocks per multiprocessor (up to 1.20x) would need a
division that knows how clustered the tree is; the rule has no such input.
Beyond 2.1 million particles per rank the division is unmeasured and holds
six blocks per multiprocessor.

### Check of the raised cap (`handoff_rule_check_large/`)

The rebuilt binary (`validation/ctest_cap1024_handoff_large_rule.log`: the
CUDA `fmmWalk`, `fmmTree` and `sphGravity` regressions and the CPU `fmmWalk`
pass; frozen as `cosmictiger_b1024_handoff_large_rule`) with the automatic
divisor against the pinned 328 and 492 at 2,097,152 particles, 8 worker
threads, bucket 128, order 7, two pairs of 11 samples, under the GPU lock
(worst pair spread 1.09; the driver's closing summary step failed because
the force exports were deleted while it still read them, so `runs.json`
holds the runs and the pair check was run by hand, `driver.log`). The
recorded force digests of the automatic rule equal those of the pinned 492
in every pair (and differ from 328's), so the rule hands off exactly the
492-block division. Speedup is the pinned division's time over the
automatic rule's time, pairwise (above one: the rule is faster).

2,097,152 particles, 8 threads, bucket 128 (`t8_n128/`):

| fixture | potential | pinned 328 | pinned 492 | automatic rule |
|---|---:|---:|---:|---:|
| lattice | 0 | 72.26 ms, 0.906x (0.901-0.911) | 75.97 ms, 0.953x (0.943-0.962) | 79.73 ms |
| lattice | 1 | 88.64 ms, 0.959x (0.928-0.989) | 89.33 ms, 0.966x (0.947-0.984) | 92.49 ms |
| clustered | 0 | 68.21 ms, 0.956x (0.896-1.016) | 70.19 ms, 0.984x (0.915-1.053) | 71.51 ms |
| clustered | 1 | 79.32 ms, 0.951x (0.919-0.983) | 85.56 ms, 1.027x (0.956-1.097) | 83.51 ms |
| adaptive | 0 | 80.29 ms, 1.083x (1.058-1.107) | 74.47 ms, 1.005x (0.951-1.059) | 74.18 ms |
| adaptive | 1 | 95.50 ms, 1.093x (1.051-1.135) | 88.28 ms, 1.010x (0.977-1.043) | 87.44 ms |

The rule is the 492-block division by construction (identical forces) and
reproduces its time within 1% on the geometric mean (3-5% behind on the
lattice, 1-3% ahead on the clustered potential and adaptive cases: run
noise of the kind the sweep's pair ranges show; the first automatic run
overlapped the deletion of the sweep's 36 GB of exports and took 53 s
against 36-39 s for the others). This stage ran on a quieter machine than the sweep (the
pinned 328 lattice kick took 72 ms against 82 ms there) and puts the rule
1% behind 328 on the geometric mean: 4-9% behind on the lattice and
clustered fixtures, 8-9% ahead on the adaptive one. Over the three
measurements at this size the six-block division is therefore consistently
1.08-1.16x faster on the adaptive fixture, level to 10% slower on the
lattice at 8 threads (level to 1.19x faster at 12) and level on the
clustered fixture, which is 4% faster on average at 8 threads in the sweep,
level in this check, and 8% faster at 12 threads. The adaptive fixture is
the closest of the three to a clustered production volume, so the cap of 3
stands; a run that is uniform throughout (the lattice case) can pin
`compute.cudaKickHandoffBlocks=328` at this size.

### Per-launch list slab (`list_slab/`, `nsys_list_slab/`, `nsys_list_slab_matched/`, `ncu_list_slab/`)

Three binaries: "head" is the pre-slab branch head eebab5b1 (the arena with
the fixes above merged with main's open-larger-cell walk, 92d735c6), "slab"
is the same commit with the per-launch list slab, both from one 1024-bucket
configuration, and "baseline" is 585ee6c0 as above. Because of the merged
walk change the absolute head and slab times are not comparable with the
tables above (the handoff and opening rule moved work from the CPU to the
GPU: at bucket 64 the adaptive kick fell from 13.8 to 12.4 ms while its
kernel grew from 5.6 to 7.2 ms), so the slab is judged against its own head
and, for the kernel criterion, with both arena binaries pinned to the
baseline's work division and opening rule (`compute.cudaKickHandoffBlocks=492`,
`gravity.openLargerCell=false`, the "matched" policy). The policy lines are
unchanged apart from the control block (528 to 648 bytes, dynamic shared
7,920 to 8,048 bytes at bucket 64, same tiles) and the new slab size:
`list slab 145792 bytes per block (4096 entries per list)`.

All GPU stages ran one at a time under a lock shared with the other sessions
of this workstation after their overlapping runs had contaminated a first
attempt (kept out of the tables; `nsys_list_slab/` round 3 is such a case
and its summary says so).

Whole-kick times (`list_slab/run.sh`: three pairs of 15 samples, direct
reference over 1,024 particles, the sampler saw no other GPU process during
the run, pair spreads at most 8%). The slab's force exports are bit-identical
to the head's on every fixture, bucket and policy (acceleration and potential
relative L2 of 0 in `list_slab/summary.md`), and both agree with the direct
reference as before (adaptive force error 2.3e-3 at bucket 64).

Whole-kick times, current handoff and opening rule (automatic policy); speedup is head over slab (paired range in parentheses):

| bucket | fixture | potential | baseline | head | slab | speedup slab vs head |
|---:|---|---:|---:|---:|---:|---:|
| 64 | lattice | no | 32.48 ms | 15.32 ms | 14.98 ms | 1.022x (1.020-1.032) |
| 64 | clustered | no | 8.44 ms | 6.42 ms | 6.08 ms | 1.054x (1.046-1.076) |
| 64 | adaptive | no | 18.61 ms | 12.35 ms | 11.90 ms | 1.038x (0.960-1.070) |
| 64 | adaptive | yes | 18.82 ms | 12.59 ms | 11.93 ms | 1.055x (1.027-1.112) |
| 128 | lattice | no | 14.83 ms | 15.18 ms | 14.91 ms | 1.018x (1.004-1.041) |
| 128 | clustered | no | 6.31 ms | 7.30 ms | 6.94 ms | 1.052x (1.043-1.055) |
| 128 | adaptive | no | 15.88 ms | 12.55 ms | 12.62 ms | 0.994x (0.969-1.040) |
| 128 | adaptive | yes | 16.86 ms | 12.54 ms | 12.49 ms | 1.004x (0.995-1.018) |

Whole-kick times, baseline's work division and opening rule (matched policy); speedup is head over slab (paired range in parentheses):

| bucket | fixture | potential | baseline | head | slab | speedup slab vs head |
|---:|---|---:|---:|---:|---:|---:|
| 64 | lattice | no | 32.48 ms | 31.58 ms | 30.71 ms | 1.028x (1.022-1.038) |
| 64 | clustered | no | 8.44 ms | 8.37 ms | 6.83 ms | 1.225x (1.216-1.242) |
| 64 | adaptive | no | 18.61 ms | 19.61 ms | 18.98 ms | 1.033x (1.021-1.040) |
| 64 | adaptive | yes | 18.82 ms | 20.43 ms | 19.30 ms | 1.059x (1.030-1.059) |
| 128 | lattice | no | 14.83 ms | 14.68 ms | 14.40 ms | 1.020x (1.010-1.034) |
| 128 | clustered | no | 6.31 ms | 6.19 ms | 6.27 ms | 0.987x (0.974-0.987) |
| 128 | adaptive | no | 15.88 ms | 17.70 ms | 17.01 ms | 1.041x (0.998-1.041) |
| 128 | adaptive | yes | 16.86 ms | 18.35 ms | 17.69 ms | 1.037x (1.025-1.068) |

Largest pair spread (max/min of the three pair medians) over all variants: 1.083.

Host and kernel split (`nsys_list_slab/nsys_run.sh`, as in the earlier Nsight
Systems section: kernel launches paired with the kicks, medians over the two
clean rounds; round 3 overlapped another session's GPU run and is excluded),
automatic policy, 15 samples:

| fixture, bucket | variant | whole kick (ms) | kick kernel (ms) | host (ms) | with potential: whole kick | kernel | host |
|---|---|---:|---:|---:|---:|---:|---:|
| adaptive, bucket 64 | baseline | 19.58 | 5.08 | 14.39 | 19.65 | 4.99 | 14.37 |
| adaptive, bucket 64 | head | 13.87 | 7.14 | 6.79 | 12.81 | 6.25 | 6.47 |
| adaptive, bucket 64 | slab | 13.22 | 6.33 | 6.58 | 12.36 | 5.84 | 6.45 |
| adaptive, bucket 128 | baseline | 17.53 | 7.79 | 9.47 | 16.31 | 6.91 | 9.33 |
| adaptive, bucket 128 | head | 13.80 | 6.61 | 6.93 | 12.87 | 6.00 | 6.69 |
| adaptive, bucket 128 | slab | 13.15 | 6.23 | 6.90 | 12.33 | 5.69 | 6.71 |

- At bucket 64 the slab kernel is 0.8 ms (11%) faster than the head's (6.33
  against 7.14 ms, forces; 5.84 against 6.25 ms with the potential) and the
  host share is nearly unchanged, so the whole kick gains the kernel
  difference. At bucket 128 the kernel gains 0.3 to 0.4 ms (5 to 6%): with
  200-particle subtrees the launch holds about 164 workitems for 656 blocks,
  so fewer blocks compete for the allocator at any one time.
- The head kernel is slower than the baseline's at bucket 64 (7.1 against 5.1
  ms) not because of the arena but because the two-blocks-per-SM handoff gives
  the GPU more of the walk (host 6.8 against 14.4 ms): the whole kick is 13.9
  against 19.7 ms. The kernel comparison that the earlier criterion (5.6
  against 5.1 ms) meant needs the same work division, measured next.

Host and kernel split at the baseline's work division
(`nsys_list_slab_matched/nsys_run.sh`: the arena binaries with
`compute.cudaKickHandoffBlocks=492` and `gravity.openLargerCell=false`, plus
the three automatic-policy variants again; three rounds, medians over the
rounds; one build of another session ran during the first 15 s and two
round-2 host shares at bucket 128 show CPU contention, see the summary, so
the kernel columns are the reliable ones):

| fixture, bucket | variant | whole kick (ms) | kick kernel (ms) | host (ms) | with potential: whole kick | kernel | host |
|---|---|---:|---:|---:|---:|---:|---:|
| adaptive, bucket 64 | baseline | 19.95 | 5.11 | 14.90 | 19.79 | 5.05 | 14.64 |
| adaptive, bucket 64 | head, matched | 21.34 | 6.64 | 14.87 | 21.05 | 6.29 | 14.87 |
| adaptive, bucket 64 | slab, matched | 20.39 | 5.45 | 14.61 | 20.03 | 5.07 | 14.58 |
| adaptive, bucket 64 | head | 13.60 | 7.11 | 6.97 | 13.38 | 6.64 | 6.59 |
| adaptive, bucket 64 | slab | 13.80 | 6.75 | 6.98 | 12.74 | 6.10 | 6.72 |
| adaptive, bucket 128 | baseline | 18.20 | 8.27 | 9.63 | 16.92 | 7.23 | 9.50 |
| adaptive, bucket 128 | head, matched | 19.83 | 9.53 | 9.90 | 18.30 | 8.65 | 9.89 |
| adaptive, bucket 128 | slab, matched | 19.01 | 9.12 | 9.76 | 18.18 | 8.63 | 9.52 |
| adaptive, bucket 128 | head | 14.18 | 6.84 | 7.04 | 13.52 | 6.46 | 6.89 |
| adaptive, bucket 128 | slab | 13.74 | 6.49 | 7.23 | 12.79 | 5.85 | 6.89 |

- At the baseline's division the host shares agree to 0.3 ms, as they must,
  and the kernels compare like for like. At bucket 64 the head kernel is 1.5
  ms (30%) above the baseline's and the slab kernel 0.34 ms (7%) above it,
  5.45 against 5.11 ms; with the potential the two are equal (5.07 against
  5.05 ms). The slab therefore removes about 80% of the arena kernel's gap to
  the baseline at the division the earlier criterion was stated for; the rest
  is in the PP and PC loops (the bucket-512 attribution above), not in list
  handling. At bucket 128 the arena kernel is further from the baseline (head
  9.5 against 8.3 ms) and the slab recovers a third of it (9.1 ms).
- The automatic-policy rows repeat the first split within noise (slab kernel
  6.75 against 6.33 ms there; the clean rounds of the two stages differ by
  that much for the head as well).


Nsight Compute replay of the second force-only kick kernel on the adaptive
fixture at bucket 64 (`--set full`, 656 single-warp blocks, automatic policy
with the current handoff), executed warp instructions per device function:

| device function | head | slab | difference |
|---|---:|---:|---:|
| `cudaKickKernel<7>` (everything inlined into it) | 128,944,938 | 128,912,577 | -32,361 |
| `cudaAllocate` | 17,144,740 | 0 | -17,144,740 |
| `cudaRelease` | 1,075,614 | 0 | -1,075,614 |
| `addGpuFlops` | 641,480 | 641,480 | 0 |
| total | 147,841,842 | 129,589,127 | -18,252,715 |

| metric | head | slab | ratio |
|---|---:|---:|---:|
| kernel duration (replay) | 7.57 ms | 6.90 ms | 0.912 |
| warp instructions issued | 147.7 M | 129.8 M | 0.879 |
| stall, long scoreboard (per issue) | 1.54 | 0.48 | 0.31 |
| stall, branch resolving (per issue) | 0.50 | 0.18 | 0.37 |
| stall, membar (per issue) | 0.0024 | 0 | 0 |
| stall, short scoreboard (per issue) | 1.91 | 2.23 | 1.17 |
| local-memory load requests (spills) | 726,843 | 568,582 | 0.78 |

The kernel no longer executes a single allocator instruction: the 18.2 M
warp instructions of `cudaAllocate` and `cudaRelease` (12% of the launch)
are gone, the kernel body itself is unchanged, and with them go most of the
long-scoreboard stalls (the pool's queue scans are dependent global atomics),
the branch-resolving stalls of the scan loops and the only memory-barrier
stalls. The replay runs 8.8% faster; the per-issue short-scoreboard stall
rises because the remaining instructions are the shared-memory PP and PC
loops.

`benchmarks/attribute_ncu_functions.py` produces the per-function table from
the `--page source` export of the reports (`ncu_list_slab/ncu_run.sh`).

compute-sanitizer on the CUDA `fmmWalk` regression at order 7
(`validation/*_list_slab.log`, `validation/racecheck_trials/`): memcheck is
clean for the slab build. racecheck reports 0 errors for every binary, and
its warning count is a property of the binary, not of the slab: the head
binary reports none in three runs, the slab binary reports 138 in four runs
whether its lists use the slab or, with `compute.cudaKickListReserve=0`, the
pool, and the committed log of the pre-slab arena build e2c00fdb
(`sanitizer/racecheck_order_7.log`) reports 126. Every warning pairs a
`cuda::memcpy_async` staging write with a later read of the staged tile (CP,
the PP main loop and its tail loop), accesses that are ordered by the
`cuda::barrier` wait and the `__syncwarp()` after it; the tool does not
relate them through the barrier in some builds (why is explained in the
racecheck staging warnings section). The force exports of the
slab and head binaries are bit-identical on every fixture and bucket
(`list_slab/summary.md`, acceleration and potential relative L2 of 0), so the
staging is not racing in fact.

### PP loop schedule at low occupancy (`pp_schedule/`)

The kernel-side loss at buckets 256 and 512 was in the PP loop's schedule, not
its arithmetic. The SASS view of the `ncu_bucket512/` reports shows three
causes, all in the regime where one warp per multiprocessor has to hide its
own latency:

- The staged source planes were read through the pointers resolved in the
  control block. Reloaded from shared memory those are generic pointers, so
  every staged source cost three 64-bit address computations (`IMAD.WIDE`) and
  three generic loads (`LD.E`), whereas the baseline's planes sit at
  compile-time offsets of its shared union and load as `LDS` with immediate
  offsets.
- Neither loop was unrolled: one source per iteration and one dependency chain
  per lane (load, fixed-point difference, conversion, `rsqrt`, accumulate),
  about 35 SASS instructions per pair in the arena and 31 in the baseline,
  issued at 4.2 to 4.8 cycles per instruction.
- The per-pair `if (r2 > h2)` split the body into one basic block per pair, so
  unrolling alone could not overlap the reciprocal square roots: a first
  version that only unrolled (four sources per 16-byte load, two sinks per
  lane) cut the kernel's instructions by 13% but left the latency at 4.6
  cycles per instruction and the kernel at 25.2 ms
  (`kernel_quick_bucket512/v1.txt`).

The loop now (`cosmictiger/gravity/gravity_cuda.hpp`, `detail::cudaStagedSources`,
`detail::cudaPpSinkGroup` and `detail::cudaPpInversePowers`):

- derives the plane pointers from the `extern __shared__` symbol and the
  layout offsets, so the loads compile to `LDS`;
- loads four consecutive sources per plane with one 16-byte `LDS.128` (the
  planes are 16-byte aligned by construction: the arena offset and the force
  region are multiples of the 16-byte arena alignment and the plane stride is
  a multiple of 128 bytes);
- gives each lane two sinks, k and k + 32, while every lane has two (complete
  64-sink chunks, so lane utilization on partial groups is exactly the
  one-sink pass's), then one sink per lane for the rest, and keeps the
  lane-split loop for a trailing group of fewer than 16 sinks;
- computes the plain inverse powers of all 4 x 2 pairs in one straight-line
  block (24 fixed-point differences, 8 `MUFU.RSQ`, 16 multiplies), takes the
  lane's smallest r2 with 7 `FMNMX`, and only when that is inside the support
  radius runs the per-pair softened substitution. The softened pairs are
  counted and the direct count is the pair total minus them. The potential
  accumulates unconditionally and is stored only with doPhi, which removes
  the per-pair select.

The fast path of the two-sink body is about 170 SASS instructions for 8 pairs
(21 per pair against 35): one basic block from the loads to the branch over
the softening block, then an accumulation block of 24 `FFMA` and 8 `FADD`.
Per sink the accumulation order over the staged sources is unchanged, so the
forces are bit-identical to the one-sink loop's: the force exports of the
head and the new loop agree exactly (`pass_b/head_vs_arena.md` from
`benchmarks/compare_kick_arena_binaries.py`: relative L2 and maximum pointwise
difference 0 for accelerations and potentials at every bucket and fixture),
and the direct-reference errors are the same to every printed digit. The
compiler now keeps `cudaGravityPpDirect` as a called subroutine instead of
inlining it (its prologue saves eight registers); the 16 inlined copies of
the softening kernel are what make it 18,000 instructions long.

Nsight Compute replay of the second force-only kick kernel, adaptive fixture,
bucket 512, `--set full` (`ncu_bucket512/`; the baseline column is the earlier
replay of the same binary in `../ncu_bucket512/baseline`, "arena head" the
arena before this change, same walk as the new loop):

| metric | baseline | arena head | arena, PP schedule |
|---|---:|---:|---:|
| kernel duration | 28.05 ms | 26.08 ms | 25.13 ms |
| warp instructions executed | 344.6 M | 285.4 M | 214.0 M |
| issue-active cycles (% of active) | 23.7 | 20.9 | 25.6 |
| cycles per issued instruction | 4.23 | 4.78 | 3.91 |
| stall, short scoreboard (per issue) | 0.72 | 1.16 | 1.25 |
| stall, long scoreboard (per issue) | 0.65 | 0.51 | 0.34 |
| stall, wait (per issue) | 1.29 | 1.54 | 0.83 |
| stall, no instruction (per issue) | 0.16 | 0.18 | 0.18 |
| shared-memory load wavefronts | 19.8 M | 15.0 M | 5.09 M |
| local-memory load sectors | 14.17 M | 0.80 M | 1.19 M |
| local-memory store sectors | 14.81 M | 2.25 M | 2.29 M |

The arena head already runs below the baseline here because the merged walk
(opening the larger cell on both walks, hybrids by counted flops) issues 17%
fewer instructions; the new loop removes another 25% of the instructions and
two thirds of the shared-memory loads and brings the issue rate above the
baseline's. The PP routine's share of the kernel's stall samples drops from
58% (baseline, `cudaGravityPpDirect` as its own function) to 35%
(`ncu_bucket512/baseline_sections.txt`, `arena/sections.txt`); the kernel
body holds the other 60%, and by source line
(`benchmarks/summarize_ncu_source.py` on the SASS-with-source export) the
Ewald cell interactions (`ewald_kernels.hpp` and the `simd_math.hpp` and
`cmath` functions they call) account for more than half of the attributed
samples. That is the remaining low-occupancy bound.

At small buckets the kernel does not regress (`kernel_small_buckets/`, same
replay, second force-only kick):

| fixture, bucket | arena head | arena, PP schedule | warp instructions |
|---|---:|---:|---:|
| clustered, 32 | 1.85 ms | 1.84 ms | 49.0 M to 46.7 M |
| clustered, 64 | 1.82 ms | 1.74 ms | 49.2 M to 45.2 M |
| adaptive, 64 | 7.63 ms | 7.22 ms | 145.0 M to 141.0 M |

Whole-kick times (`run.sh`, `run_pass_b.sh`: baseline, arena head and the new
loop, buckets 32 to 512, 15 samples, direct reference over 1,024 particles).
Three passes of the same three binaries are kept: the development run
(`pass_dev/`, three pairs, one disturbed by a sibling session's build), pass A
(`pass_a/`, three pairs, its first disturbed on the CPU-bound cells by profile
exports and other sessions' runs alongside it) and pass B (`pass_b/`, five
pairs, run under the sessions' shared GPU benchmark lock with
`wait_quiet_then_pass_b.sh`; no run exceeded 4.1 s and every tree build took
its quiet-machine 2 ms, but an IDE was indexing on the desktop). The tables
are pass B's:

lattice (forces only):

| bucket | baseline | arena head (old loop) | arena, PP schedule | speedup vs baseline (paired range) | speedup vs head (paired range) |
|---:|---:|---:|---:|---:|---:|
| 32 | 33.58 ms | 16.37 ms | 16.39 ms | 2.048x (2.027-2.160) | 0.998x (0.954-1.041) |
| 64 | 33.08 ms | 15.62 ms | 15.45 ms | 2.141x (2.122-2.189) | 1.011x (0.988-1.035) |
| 128 | 15.36 ms | 15.54 ms | 15.58 ms | 0.986x (0.954-1.017) | 0.997x (0.986-1.028) |
| 256 | 12.37 ms | 11.56 ms | 10.97 ms | 1.128x (0.967-1.151) | 1.054x (0.954-1.081) |
| 512 | 11.17 ms | 10.53 ms | 9.88 ms | 1.131x (0.959-1.335) | 1.067x (0.959-1.104) |

clustered (forces only):

| bucket | baseline | arena head (old loop) | arena, PP schedule | speedup vs baseline (paired range) | speedup vs head (paired range) |
|---:|---:|---:|---:|---:|---:|
| 32 | 8.89 ms | 6.95 ms | 7.03 ms | 1.264x (1.260-1.370) | 0.988x (0.954-1.019) |
| 64 | 8.77 ms | 6.69 ms | 6.80 ms | 1.289x (1.190-1.335) | 0.984x (0.956-1.000) |
| 128 | 6.44 ms | 7.61 ms | 7.40 ms | 0.869x (0.839-0.937) | 1.028x (0.992-1.037) |
| 256 | 7.36 ms | 7.82 ms | 6.33 ms | 1.163x (1.113-1.221) | 1.235x (1.168-1.313) |
| 512 | 10.19 ms | 10.36 ms | 5.97 ms | 1.708x (1.564-2.122) | 1.736x (1.604-1.966) |

adaptive (forces only):

| bucket | baseline | arena head (old loop) | arena, PP schedule | speedup vs baseline (paired range) | speedup vs head (paired range) |
|---:|---:|---:|---:|---:|---:|
| 32 | 18.47 ms | 13.04 ms | 13.48 ms | 1.371x (1.329-1.516) | 0.968x (0.924-1.146) |
| 64 | 19.24 ms | 13.01 ms | 12.70 ms | 1.514x (1.390-1.685) | 1.024x (0.923-1.086) |
| 128 | 16.23 ms | 12.76 ms | 12.79 ms | 1.269x (1.220-1.322) | 0.997x (0.969-1.049) |
| 256 | 18.73 ms | 18.16 ms | 18.30 ms | 1.024x (0.945-1.141) | 0.993x (0.947-1.052) |
| 512 | 28.71 ms | 27.26 ms | 28.54 ms | 1.006x (0.991-1.156) | 0.955x (0.933-1.044) |

adaptive (forces + potential):

| bucket | baseline | arena head (old loop) | arena, PP schedule | speedup vs baseline (paired range) | speedup vs head (paired range) |
|---:|---:|---:|---:|---:|---:|
| 32 | 18.80 ms | 13.38 ms | 13.44 ms | 1.399x (1.338-1.494) | 0.996x (0.947-1.041) |
| 64 | 19.03 ms | 12.98 ms | 13.41 ms | 1.418x (1.324-1.671) | 0.968x (0.890-1.078) |
| 128 | 16.90 ms | 13.32 ms | 12.68 ms | 1.333x (1.261-1.391) | 1.051x (0.980-1.057) |
| 256 | 18.91 ms | 18.81 ms | 18.23 ms | 1.037x (0.932-1.161) | 1.031x (0.976-1.080) |
| 512 | 29.64 ms | 28.89 ms | 29.22 ms | 1.014x (0.982-1.154) | 0.989x (0.973-1.055) |

Speedups are the baseline's or head's median over the new loop's; the paired
range is the min-max of the per-pair ratio. The new loop is level with the
head at buckets 32 to 128 (within 3%, where the kick is dominated by the
CPU walk and a fully occupied kernel), 5 to 7% faster on the lattice and
1.24x and 1.74x faster on the clustered fixture at buckets 256 and 512: the
clustered leaves hold many softened pairs, whose per-pair branches cost the
old loop most. On the adaptive fixture the whole kick at buckets 256 and 512
is level with the baseline and the head within the pair spread, which in
pass B was wide on the GPU-bound cell (new loop 26.2 to 28.6 ms at bucket
512, head 26.6 to 29.9 ms, baseline 28.3 to 33.0 ms, with an IDE indexing
on the desktop). In the quiet windows of the other two passes (`pass_a/`
pairs 1 and 2, `pass_dev/` pairs 0 and 1, the development run of the same
binaries) that cell was tight and ordered: new loop 26.1 to 26.7 ms, head
26.8 to 27.5 ms, baseline 28.6 to 29.4 ms, in line with the 25.1, 26.1 and
28.1 ms kernel replays above plus about 2 ms of host time. At bucket 256
the same windows give 17.2 to 17.8 ms for the new loop against 17.1 to
17.8 ms for the head and 18.1 to 18.8 ms for the baseline.

Direct-reference errors are unchanged from the head's to every printed digit
(the walk differs from the baseline's, so they differ from the baseline's as
documented for the merged walk), and the force exports of the new loop equal
the head's bit for bit in every pass (`pass_*/head_vs_arena.md`).

### racecheck staging warnings (`validation/racecheck_forms/`)

The racecheck warnings of the list slab section are an effect of how a given
build addresses the `cuda::barrier` in SASS, which the tool's `memcpy_async`
tracking does not follow in one of the two shapes ptxas produces; they are not
an ordering gap in the staging. The evidence, all on the CUDA `fmmWalk`
regression at order 7 with compute-sanitizer 2025.1.0 (CUDA 12.8, driver
595.99.02), one GPU job at a time under the shared lock:

- What the tool reports. With `--racecheck-report all --print-level info`
  every hazard record of the slab binary reads "(Warp Level Programming)
  Potential RAW hazard detected (invalid memcpy_async synchronization)", the
  write at the 4-byte `cp.async` inside `cuda::memcpy_async` (barrier.h:957)
  and the read at a CP or PP tile read; 17,505 of the 83,626 records dumped
  before the run was stopped name the same thread as writer and reader
  (`logs/wl_slab_hazards_tally.txt`, sample in
  `logs/wl_slab_hazards_sample.log`). That category is the tool's "copy
  target read before the commit or wait that completes it"; a thread reading
  its own copy target after the barrier wait can only land there if the tool
  did not connect the wait to the copies. With `--racecheck-memcpy-async no`
  the slab binary reports 0 warnings (`logs/wl_slab_memcpy_async_off.log`):
  there is no other hazard. The head binary reports 0 either way.

- What the builds differ in. Per `memcpy_async` the staging compiles to an
  `LDGSTS.E` loop and one `ARRIVES.LDGSTSBAR.64 [barrier]`
  (`cp.async.mbarrier.arrive`); `arrive_and_wait()` compiles to
  `ATOMS.ARRIVE.64 [barrier]` and a polling `LDS` of the barrier's upper word
  (`mbarrier.test_wait` on sm_86). The order-7 kernel of every frozen binary
  of this series (`forms_summary.txt`, from `scripts/survey_one.sh`, which
  runs `nvdisasm -g` on the extracted cubin) shows one of two shapes. In the
  clean head binary the barrier address is held in a per-thread register:
  `ARRIVES [UR4]` (that register moved to a uniform register),
  `ATOMS.ARRIVE.64 [R12+URZ]`, `LDS [R12+0x4]`. In the slab binary and in the
  four gifted-euler binaries the control block base is a uniform register and
  all three instructions address the barrier as `[UR47+0x1d0]`. The
  instruction sequences are otherwise the same, and the slab commit did not
  touch the staging source; the kernel's extra constructor path changed a
  register-allocation choice.

  | binary | barrier operands (arrive-on, arrive, test_wait) | racecheck warnings |
  | --- | --- | --- |
  | wizardly-leakey head eebab5b1 | `[UR4]`, `[R12+URZ]`, `[R12+0x4]` | 0 (five runs) |
  | wizardly-leakey slab eedcc53a | `[UR47+0x1d0]` for all three | 138 (four runs) |
  | gifted-euler final e2c00fdb | `[URn+0x1d0]` for all three | 126 (committed log) |
  | gifted-euler 231325d5, fix, fix2 | `[URn+0x1d0]` for all three | not run |
  | unruffled-robinson head cf66eacc (v1, v2 alike) | `[UR4]`, `[R+URZ]`, `[R+0x4]` | 0 (head, one run) |
  | modest-borg handoff b730980c | `[UR4]`, `[R196+UR40]`, `[R196+UR40+0x4]` | 0 (two runs) |
  | this branch, scratch build (`build/racecheck`) | kernel-inlined CP sites `[UR40+0x1d0]` for all three; out-of-line PP `[UR]`, `[R+URZ]`, `[R+0x4]` | 9 reports, all CP lines |

  Where ptxas keeps a CP or PP instantiation out of line (the handoff, the
  PP-loop v1/v2 and this branch's builds have separate
  `cudaGravityCpDirect<7, …>` and `cudaGravityPpDirect<7, …>` functions), the
  control block arrives in a register and the staging there takes the clean
  shape (`outofline_forms.txt`); that is why this branch's build reports only
  its kernel-inlined CP copies. The PC multipole copy and the checklist copy
  are never in the reports because neither issues a `cp.async`: the PC call
  passes an untyped byte count, so libcu++ deduces 1-byte alignment and takes
  its synchronous per-thread copy, and the checklist copy is global to global.

- Variants of the staging in the scratch build of this branch
  (`scripts/apply_variant.py`, diffs in `variants/`, each rebuilt from the
  committed source, racecheck in `logs/`):

  | variant | change in `cudaStageParticleSources` | arrive-on operand at the staging sites | racecheck |
  | --- | --- | --- | --- |
  | base | committed source | `[UR40+0x1d0]` | 9 reports (6,285 hazard records printed) |
  | launder_copy | the barrier reference passed to the three `memcpy_async` calls goes through an empty `asm volatile` | `[URn]`; arrive and test_wait unchanged, `[UR40+0x1d0]` | 0 |
  | launder_wait | the same for the `arrive_and_wait()` call only | `[UR40+0x1d0]`; arrive `[R4+URZ]`, test_wait `[R42+0x4]` | 9 reports |
  | launder_both | both | `[URn]`, `[R+URZ]`, `[R+0x4]` | 0 |
  | block_sync | `group.sync()` added after `arrive_and_wait()` | as base, plus a `BAR.SYNC` | 9 reports |
  | sync_copy | per-lane plain copies and `__syncwarp()`, no barrier | no `cp.async` in the staging | 0 |

  Only the operand form of the `cp.async.mbarrier.arrive` changes the
  result: with a uniform register plus an immediate offset the tool does not
  connect the copies to the barrier, and every later read of the tile, by
  the copying lane included, is reported as an unsynchronized
  `memcpy_async`; the forms of the arrive and the wait, and an extra block
  barrier, make no difference. The `fmmWalk` regression passes in every
  binary and variant above.

The staging is the documented `cuda::memcpy_async(group, …, barrier)` plus
`barrier.arrive_and_wait()` pattern: each lane's copies arrive on the barrier,
the phase completes only when every copy has landed, and the tile is read
after the wait and a `__syncwarp()`. The force exports of the slab and head
binaries are bit-identical (list slab section). The staging code is therefore
unchanged, and the warnings of builds in which ptxas addresses the control
block through a uniform register are recorded as a compute-sanitizer 2025.1.0
modelling artefact. A build that has to pass racecheck cleanly can hide the
barrier address from the compiler as `launder_copy` does (the same
instructions with a different operand form) or copy synchronously as
`sync_copy` does (no `cp.async` and no barrier in the staging); neither was
adopted here, the first because it steers a compiler choice to suit the tool
and the second because its cost was not measured.

### Landing on the reorganized layout (`validation/merge_reorganized/`)

The three follow-up branches above were cut before the source reorganization
(2ce9c788 moved the sources into subdirectories, 565759b6 split
`kick_cuda_impl.hpp` into `kick_cuda_kernel.hpp` and `kick_cuda_launch.hpp`)
and were merged onto `main` at 00909d26 in order: the list slab (eb4277f0,
from eedcc53a and a07fafd7), the PP loop schedule (a285653a, from ada60c24)
and the racecheck explanation (2db2deb4, from 9184eb68). Git's rename
detection carried the slab's `device_vector.hpp`, `stack_vector.hpp`,
`kick.hpp`, `kick_cuda_instances.hpp` and `options.cpp` hunks and the
kernel-side slab hunks of `kick_cuda_impl.hpp` to their moved files, and the
PP commit's `gravity_cuda.hpp` hunks to `cosmictiger/gravity/gravity_cuda.hpp`.
The host-side slab hunks were placed by hand in `kick_cuda_launch.hpp`:
`forceBackingCapacity` and `listSlabBytes` in `kickBuildLayout`, the slab
fields of the policy line in `kickSelectLayout`, and the per-launch slab
allocation and kernel argument in `cudaExecuteKicksImpl`; `kick_cuda.hpp` took
the moved include paths. The merged kernel and launch headers equal the
branch's `kick_cuda_impl.hpp` apart from the file split and license header,
and the merged `gravity_cuda.hpp` equals the branch's apart from the include
paths.

Validation of each merge head in a 1024-bucket Release build of the worktree
(`validation_slab.txt`, `validation_pp.txt`, `validation_final.txt`):
`stf_kernels_cuda_test`, `stf_l2l_cuda_test` and `stf_ewald_cpu_test` pass;
the CUDA `fmmWalk` regression passes at orders 3, 7 and 9, and `fmmTree` and
the mixed-mass `sphGravity` regression pass at order 7. The order-7 policy
line of the merged binary is the slab branch's (`policy_lines_final.txt`:
control 648 bytes, dynamic shared 8,048 bytes at bucket 64, `list slab 145792
bytes per block (4096 entries per list)`), and `compute.cudaKickListReserve=0`
still takes the pool path (`list slab 0 bytes per block`, `fmmWalk` passes).
compute-sanitizer on `fmmWalk` at order 7 on the final head
(`racecheck_order_7_merged.log`, `memcheck_order_7_merged.log`): memcheck 0
errors; racecheck 0 errors and 24 warnings, every one a `cuda::memcpy_async`
staging write (barrier.h:957) paired with a CP read of the staged particle
planes (`gravity_cuda.hpp` lines 313 to 315, 160 hazard records each), the
arrive-on operand-form artefact of the previous section; the PP loop, which
now reads the planes through the shared-memory symbol, is not reported. No
hazard kind appears that the slab and head logs did not already show (0, 126
and 138 warnings there, 0 errors throughout).

Not measured: the paired whole-kick timing of the merged binary against the
pre-merge `main` (order 7, bucket 64, adaptive fixture) was not run, because
the workstation's gaming cap (six CPU threads, a quarter of the GPU, no kick
arena driver runs) was in force while the merges landed. The branch
measurements in the list slab and PP loop schedule sections stand as the
record of the two changes' effect; the merged code reproduces them
unchanged, so no change beyond run noise is expected, but it is unverified
on the merged head.

### What remains

- The per-launch list slab removed the pool allocator from the kernel (above);
  at the baseline's work division the kernel is now within 7% of the
  baseline's at bucket 64 (equal with the potential) and 10% at 128, and the
  remaining difference is in the PP and PC loops themselves, not in list
  handling.
- The PC multipole staging passes an untyped byte count to
  `cuda::memcpy_async`, so libcu++ deduces 1-byte alignment and copies
  synchronously through registers; the barrier wait after it then orders
  nothing. A `cuda::aligned_size_t<16>` count, if the multipole records are
  16-byte aligned in global memory, would make it a `cp.async` copy like the
  particle planes. Not measured.
- At small problems where the launch holds fewer warps than the GPU can keep
  resident (64 leaves at bucket 512), the PP loop is no longer the bound: with
  the restructured loop it holds about a third of the kernel's stall samples
  and the walk with its Ewald cell interactions (`cudaGravityCcEwald`, one
  multipole per lane, a few multipoles per leaf) holds most of the rest. That
  loop is the next candidate for the same treatment.
- The handoff rule is measured at 32,768, 262,144 and 2,097,152 particles
  and 2-12 threads and holds six blocks per multiprocessor above 884,736
  particles; beyond 2.1 million particles per rank it is unmeasured, and the
  16,384-particle `CUDA_KICK_PARTS_MAX` cap takes over at about 8 million.
  At 2.1 million particles the clustered fixture gains up to 1.20x from 12-16
  blocks per multiprocessor while the lattice loses 10-22%, so a division
  that reads the tree's clustering could beat the fixed rule; the lattice
  penalty of the finer divisions is thread-independent there and was not
  profiled (`handoff_large_sizes/`).
- The whole-kick time of the merged binary against the pre-merge `main`
  (order 7, bucket 64, adaptive fixture) is unmeasured; the list slab and PP
  loop schedule measurements predate the merge (Landing on the reorganized
  layout).
- The theta sweep shows order 7 at theta 0.8 is not the cheapest route to its
  own force error; a lower order with a smaller opening angle is, down to
  about 2e-4. The production order and opening angle should be chosen from
  the matched-accuracy tables rather than fixed at 7 and 0.8.
