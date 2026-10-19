# STF gravity kernels

The FMM uses the STF generator imported from
[`dmarce1/experimental`](https://github.com/dmarce1/experimental/tree/7ba8f1829fdede7348a1418b7cb36c0e7a955302)
at commit `7ba8f1829fdede7348a1418b7cb36c0e7a955302`. Its Boost license is retained
in `cosmictiger/gravity/stf/LICENSE`; the surrounding application remains GPL.

Only the production degrees are generated: source degree 5 and local degree 6.
The source stores 36 coefficients. The local stores 49 STF coefficients and one
constant Laplacian. Indices are rank-major, with `nz=0` followed by `nz=1`.
Moments are positive raw STF moments multiplied by `S^rank`, and local
coefficients are derivatives of the negative potential divided by `S^(rank+1)`,
where `S=1000` preserves the existing numerical normalization.

The imported partial-contraction M2L is used directly for exterior cell-cell
interactions; particle-cell interactions use the generated M2P. Both preserve
the `centered` template option, defaulting to false so geometric expansion
centers retain their dipoles. CPU SIMD and CUDA use the same generated
arithmetic. P2M, L2P, and contraction against precomputed harmonic Ewald
derivatives were added to the imported rational-expression generator.

CUDA L2L has a generated cooperative schedule in `tools/stf/stf_cuda.py`.
It emits one fully specialized `L2lWarpKernel<L, true>` for the requested local
degree `L` in 2 through 11; the schedule table, the per-phase target table, the
basis register count, the lane-group widths and the phase code are compile-time
constants of that degree. All 32 lanes translate one expansion using aligned
power-of-two lane groups. Each 32-bit schedule entry packs three 8-bit
coefficient indices (basis, primary, secondary) and a 2-bit kind (padding,
direct term, or trace-reduced `nz = 2` term); the output coefficient of each
group comes from a separate per-phase `l2lWarpTargets` table. Both tables use
coalesced read-only global loads. The polynomial basis is distributed over
`ceil((L + 1)^2 / 32)` lane-owned registers (two at degree six, five at degree
eleven): basis index `i` lives in word `i >> 5` on lane `i & 31`, and each row
fetches it with one shuffle per word plus a constant selection chain, avoiding
dynamic register-array indexing.

Two entry points share that schedule. `translate` broadcasts every completed
coefficient to all lanes' private expansions, as the production tree walk
expects today. `translateInto` writes into a non-aliasing target of
`(L + 1)^2 + 1` values (for example a shared-memory expansion): each group's
leader stores its coefficient once, strided lanes copy the unchanged top-rank
coefficients and trace, and no broadcast shuffles are issued. Both end with a
warp barrier that completes all source reads and target writes before the caller
can overwrite the expansion. The degree-six schedule uses 22 rows, three phases,
and 86 shuffle instructions per lane (50 for `translateInto`); degree eleven
uses 137 rows, 22 phases, and 873 shuffles per lane (752 for `translateInto`).
Generation asserts that every symbolic Taylor term appears exactly once, that
every output below the top rank owns exactly one aligned lane group inside the
warp, that packed indices and kinds fit their fields, and that padded entries
are kind zero.

## Constant background

For the unit periodic box and potential convention `Phi=-1/r`, the Ewald
correction has Laplacian `-4*pi`. Let `r` be the minimum-image displacement and
`R` the unwrapped displacement between the same expansion centers. Interactions
use the harmonic function

```
H(R) = Ewald(r) + (2*pi/3) |R|^2.
```

Scalar, gradient, and Hessian terms are changed together. The complete root
expansion receives the opposite quadratic

```
B(x) = -(2*pi/3) [M |x-C|^2 - 2 D dot (x-C) + Q],
D = sum m (y-C),
Q = sum m |y-C|^2.
```

`M` and `D` come from the root multipole. When potential is requested, `Q` is
summed once over the current source-particle range across localities. It is
never stored in source multipoles. This preserves the potential gauge as well
as the force. Local translations and evaluation use unwrapped coordinates so
this root polynomial translates consistently across periodic faces. The
Laplacian is carried only in local expansions; Ewald interaction locals are
harmonic.

## Regeneration and validation

```
python3 tools/generate_stf_m2l.py
python3 tools/generate_stf_m2l.py --check
cmake --build build/stf_validation --target generate_fmm_kernels
```

The last command regenerates only the Ewald derivative formulas into
`build/stf_validation/generated/cosmictiger/ewald_kernels.hpp`; formatting that
file with the project clang-format configuration must reproduce the checked-in
`cosmictiger/ewald_kernels.hpp`. The former Cartesian translation generator has
been removed from `src/fmm_kernel_codegen.cpp`.

`stf_kernels` checks independent Cartesian polynomial references, translations,
nonzero dipoles, constant Laplacians, and SIMD lanes. `stf_kernels_cuda` executes
the same operations on the device in float and double. The complete `fmmWalk`
regression compares periodic forces and absolute potentials with an independent
Ewald sum, including periodic seams and excluded inactive particles.

`stf_l2l_cuda` independently checks every coefficient and every lane after one
and 17 translations, both potential modes, nonzero traces, linear fields, and
multiple warps per block, at whichever local degree `ORDER` selects. It checks
`translate` through the caller's synchronized in-place update pattern and
`translateInto` through ping-pong shared-memory source and target expansions. The old CUDA generator omitted the last of 616 nonpotential
terms, `Lzzzzzz * dz^4 / 24` in `Lzz`; the repeated-shift checks detect that trace
drift in the unchanged legacy baseline.

`stf_ewald_cpu` also compiles the CUDA-disabled Ewald initializer and checks its
image list, self terms, normalization, and periodic potential/gradient values.
CPU runtime validation uses `--compute.cuda=false` in the CUDA-enabled
application. A complete `COSMICTIGER_WITH_CUDA=OFF` application build still has
pre-existing unconditional CUDA calls elsewhere in the application.

## Benchmarks

The opt-in `COSMICTIGER_BUILD_STF_BENCHMARKS=ON` build adds
`--run.test=stfBenchmark`, which measures deterministic 32^3-particle lattice and
clustered trees with potential both enabled and disabled. It reports tree and
kick timings separately, uses one warmup and five measured repetitions by
default, and writes forces outside the measured interval.

`benchmarks/stf_kernels.cpp` and `benchmarks/stf_kernels_cuda.cu` measure individual
operators. `COSMICTIGER_STF_BASELINE_HEADER` selects an external legacy header so
the exact same harness can compare the old and new implementations. Timings
must be collected after validation, with builds and other benchmark jobs idle.
See [benchmark instructions](../benchmarks/stf_benchmarks.md) for reproducible
builds, timing, and force comparisons, and the
[measured results](stf_benchmark_results.md) for CPU, CUDA, and cooperative L2L
performance and numerical comparisons.
