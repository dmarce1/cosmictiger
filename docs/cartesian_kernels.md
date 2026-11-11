# Trace-free Cartesian gravity kernels

`cosmictiger::gravity::cartesian::Multipole<Real>` stores symmetric
trace-free moments, and `cartesian::Expansion<Real>` stores local
coefficients. Both take only the scalar type; the build fixes the internal
FMM order. The multipole degree is `kFmmOrder - 2`, and the local/potential
degree is `kFmmOrder - 1`.
The named fields `m`, `x`, `xy`, `xyz`, etc. retain the compact rank-major
layout; components containing two or more `z` indices are derived from the
trace-free relation. A union overlays the named fields with an array of the
same length. Serialization visits the array in stored order; indexed access
supports coefficient loops and the CUDA warp translation schedule.
The transport buffer holds exactly the build-sized coefficient count and
serializes `kFmmOrder * kFmmOrder` local coefficients.
Generated Cartesian operators are ordinary scalar/SIMD overloads with the
source and displacement before the destination accumulator. `M2L` takes the target-minus-source center displacement and
evaluates the free-space interaction directly.
The FMM uses the generator imported from
[`dmarce1/experimental`](https://github.com/dmarce1/experimental/tree/7ba8f1829fdede7348a1418b7cb36c0e7a955302)
at commit `7ba8f1829fdede7348a1418b7cb36c0e7a955302`. Its Boost license is retained
in `tools/cartesian/LICENSE`; the surrounding application remains GPL.

CMake generates one operator set for `COSMICTIGER_EXPANSION_DEGREE` (default 6,
supported values 2 through 8). Degree d uses source degree d-1, local degree d,
and internal order d+1. Tree and CUDA calls use this order directly; there is
no runtime selector or order dispatcher. Generated headers reside in each
build directory, allowing independent builds with different degrees.
At potential degree 6 (internal order 7) the source stores 36 coefficients and the local stores 49 Cartesian
coefficients; in general a degree-`d` tensor stores
`(d + 1)^2` coefficients. Indices are rank-major, with `nz=0` followed by `nz=1`.
Moments are positive raw Cartesian moments multiplied by `S^rank`, and local
coefficients are derivatives of the negative potential divided by `S^(rank+1)`,
where `S=1000` preserves the existing numerical normalization.

The imported partial-contraction M2L is used directly for near-field cell-cell
interactions; particle-cell interactions use the generated M2P. Both retain
source dipoles. CPU SIMD and CUDA use the same generated arithmetic. P2M and
L2P were added to the imported rational-expression generator. The periodic
far field uses a cached spherical Ewald Green filter and Fourier contraction.

The ordinary kernel emitter uses explicit `fma` with a function-local `math` import
for single-use products
consumed by addition, subtraction, or output accumulation. Shared products stay materialized; M2L retains source degree one below the local degree.
The JSON report counts each FMA once for comparisons; generated `k*Flops`
constants use FMA = 2 for runtime performance accounting. See [the representation comparison](generated_fmm.md)
for the weighted M2L costs including geometry.

CUDA L2L has a generated cooperative schedule in `tools/cartesian/cuda.py`.
It emits order-specific `L2LWarp` and `L2LWarpInto` functions for local
degree selected by the build; the schedule table, the per-phase target table, the
basis register count, the lane-group widths and the phase code are compile-time
constants of that degree. All 32 lanes translate one expansion using aligned
power-of-two lane groups. Each 32-bit schedule entry packs three 8-bit
coefficient indices (basis, primary, secondary) and a 2-bit kind (padding,
direct term, or trace-reduced `nz = 2` term); the output coefficient of each
group comes from a separate per-phase `l2lWarpTargets` table. Both tables use
coalesced read-only global loads. The polynomial basis is distributed over
`ceil((L + 1)^2 / 32)` lane-owned registers (two at degree six, three at degree
eight): basis index `i` lives in word `i >> 5` on lane `i & 31`, and each row
fetches it with one shuffle per word plus a constant selection chain, avoiding
dynamic register-array indexing.

Two entry points share that schedule. `L2LWarp` broadcasts every completed
coefficient to all lanes' private expansions, as the production tree walk
expects today. `L2LWarpInto` writes into a non-aliasing target of
`(L + 1)^2` values (for example a shared-memory expansion): each group's
leader stores its coefficient once, strided lanes copy the unchanged top-rank
coefficients, and no broadcast shuffles are issued. Both end with a
warp barrier that completes all source reads and target writes before the caller
can overwrite the expansion. The degree-six schedule uses 22 rows, three phases,
and 86 shuffle instructions per lane (50 for `L2LWarpInto`).
Generation asserts that every symbolic Taylor term appears exactly once, that
every output below the top rank owns exactly one aligned lane group inside the
warp, that packed indices and kinds fit their fields, and that padded entries
are kind zero.

## Periodic background

PMMM builds its periodic Green filter in the spherical basis and applies the
far-field contraction on the mesh. The completed mesh local is converted to a
Cartesian trace-free local for tree descent. The uniform-background curvature
is carried separately through local translation and point evaluation; it is
not stored in Cartesian harmonic coefficients.

## Regeneration and validation

Generated kernels are checked in under `cosmictiger/gravity/`. Ordinary builds
and tests include these source headers directly; they do not regenerate or
modify them. Regenerate explicitly after changing a generator:

```sh
python3 tools/generate_fmm.py
python3 tools/generate_fmm.py --check
# Equivalent regeneration through a configured build:
cmake --build build --target generate_fmm_kernels
```

The generator defaults to the source header directory and uses the repository's
clang-format configuration. It emits Cartesian and spherical families for FMM
the requested `--degree` (default 6). `kernels/cartesian_tracefree_config.hpp`
records the generated degree. Cartesian operators live in `kernels/cartesian_tracefree/m2l.hpp`,
`m2m.hpp`, `l2l.hpp`, `p2m.hpp`, `p2l.hpp`, `m2p.hpp`, and `l2p.hpp`;
the cooperative CUDA translation is in `l2l_warp.hpp`. Spherical operators are in
`kernels/spherical_kernels.hpp` and `kernels/spherical_ewald_kernels.hpp`.
`operation_counts.json` accompanies the headers.
The same Python entry point emits the spherical Ewald Green evaluator used to
build the PMMM filter.
The compatibility targets `generate_cartesian_kernels` and
`cosmictiger_generated_kernels` invoke the same explicit regeneration command.
CMake generates only the configured degree in its build directory, automatically
before compilation; direct generator invocation updates the source references.
Regeneration requires Python 3 and clang-format. Never hand-edit generated
headers; `--check` verifies them without writing, and CTest's
`generated_fmm_current` runs that comparison against the generated build headers.

`cartesian_kernels` checks independent Cartesian polynomial references, translations,
nonzero dipoles, the truncated direct-particle remainder
series, and SIMD lanes for the build order,
printing a pass line with the worst normalized discrepancy it saw.
`cartesian_kernels_cuda` executes the same operations on the device in float,
at the build order. The PMMM gravity tests compare periodic forces
and potentials with an independent Ewald sum.

`cartesian_l2l_cuda` independently checks every coefficient and every lane after one
and 17 translations, both potential modes, linear fields, and
multiple warps per block, at each selected order's local degree. It checks
`L2LWarp` through the caller's synchronized in-place update pattern and
`L2LWarpInto` through ping-pong shared-memory source and target expansions.
The old CUDA generator omitted the last of 616 nonpotential terms,
`Lzzzzzz * dz^4 / 24` in `Lzz`; the repeated-shift checks detect that
translation error at local degree six.

The spherical Ewald kernel tests check the generated Green coefficients at the
origin, near the origin, along coordinate axes, and at general periodic
separations. See [the spherical Ewald documentation](generated_spherical_ewald.md)
for its recurrence, lattice cutoff, and validation.

CPU runtime validation uses `--gravity.cuda=false`. It can run in either a
CUDA-enabled application or a CPU-only build configured with
`COSMICTIGER_WITH_CUDA=OFF`; CUDA tests require a CUDA-enabled build and device.

## Historical benchmarks

The Cartesian comparison harness, microbenchmarks, and associated runner and
analysis scripts are archived in the Git stash named
`retired_cartesian_benchmarks_and_generic_fmm_2026_10_27`.
The [measured results](stf_benchmark_results.md) remain as historical CPU, CUDA,
and cooperative L2L performance and numerical comparisons. The CUDA L2L
correctness checks remain in `tests/gravity/cartesian_l2l_cuda.cu`.

Generated arithmetic and CUDA schedule tables are compiled in `cosmictiger_fmm`;
operator headers contain declarations. See [compiled FMM library](generated_fmm.md#compiled-library)
for supported concrete types and CPU/CUDA linking.
