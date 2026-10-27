# Cartesian gravity kernels

`cosmictiger::gravity::cartesian::Multipole<Real, order>` stores symmetric
trace-free moments, and `cartesian::Expansion<Real, order>` stores local
coefficients and a Laplacian. Their second template argument is the application
FMM order: the multipole degree is `order - 2`, and the local degree is
`order - 1`. Internal `cartesian::detail::Coefficients<Real, degree, withTrace>`
storage and generated operator specializations retain degree-based arguments.
The same namespace contains the generated operators `P2M`, `M2M`, `M2L`, `M2P`,
`P2L`, `L2L`, and `L2P`; `M2LDerivatives` contracts precomputed harmonic derivatives.
The FMM uses the generator imported from
[`dmarce1/experimental`](https://github.com/dmarce1/experimental/tree/7ba8f1829fdede7348a1418b7cb36c0e7a955302)
at commit `7ba8f1829fdede7348a1418b7cb36c0e7a955302`. Its Boost license is retained
in `tools/cartesian/LICENSE`; the surrounding application remains GPL.

Kernels are generated for every supported FMM order, 3 through 9: source
degrees 1 through 7 paired with local degrees 2 through 8. CMake's
`COSMICTIGER_FMM_MIN_ORDER` and `COSMICTIGER_FMM_MAX_ORDER` select the compiled
range, defaulting to 4 through 8. The corresponding `FMM_MIN_ORDER` and
`FMM_MAX_ORDER` preprocessor guards exclude other orders from compilation while
retaining all orders in the source headers. A run selects one compiled
order with `gravity.fmmOrder` (default 7, the build's `COSMICTIGER_WITH_ORDER`),
and `fmmDispatch()` switches to that order's instantiation at coarse boundaries.
At order 7 the source stores 36 coefficients and the local stores 49 Cartesian
coefficients and one constant Laplacian; in general a degree-`d` tensor stores
`(d + 1)^2` coefficients. Indices are rank-major, with `nz=0` followed by `nz=1`.
Moments are positive raw Cartesian moments multiplied by `S^rank`, and local
coefficients are derivatives of the negative potential divided by `S^(rank+1)`,
where `S=1000` preserves the existing numerical normalization.

The imported partial-contraction M2L is used directly for exterior cell-cell
interactions; particle-cell interactions use the generated M2P. Both preserve
the `centered` template option, defaulting to false so geometric expansion
centers retain their dipoles. CPU SIMD and CUDA use the same generated
arithmetic. P2M, L2P, and contraction against precomputed harmonic Ewald
derivatives were added to the imported rational-expression generator.

The ordinary kernel emitter uses explicit `simdFma` for single-use products
consumed by addition, subtraction, or output accumulation. Shared products stay
materialized, and both M2L source-degree branches participate in use analysis.
`kWeightedFlops_` counts each FMA once; `kFlops_` retains FMA = 2 for runtime
performance accounting. See [the representation comparison](generated_fmm.md)
for the weighted M2L costs including geometry.

CUDA L2L has a generated cooperative schedule in `tools/cartesian/cuda.py`.
It emits one fully specialized `cartesian::detail::L2LWarp<L, true>` for the requested local
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
clang-format configuration. It emits both Cartesian and spherical families for
FMM orders 3 through 9: one specialized kernel class per degree for local
degrees 2 through 8 (`fmm_kernels/cartesian_config.hpp` records the available range as
`kMinLocalDegree`/`kMaxLocalDegree`) and one `ewaldCorrectionDerivatives` overload
per order in `fmm_kernels/cartesian_ewald_kernels.hpp`. The Cartesian free-space
operators are in `fmm_kernels/cartesian_m2l.hpp` and
`fmm_kernels/cartesian_kernels.hpp`; spherical operators are in
`fmm_kernels/spherical_kernels.hpp` and `fmm_kernels/spherical_ewald_kernels.hpp`.
`operation_counts.json` accompanies the headers.
The same Python entry point uses `tools/cartesian/ewald.py` for the Ewald
derivatives, so generation does not require compiling a separate C++ tool.
The compatibility targets `generate_cartesian_kernels` and
`cosmictiger_generated_kernels` invoke the same explicit regeneration command.
This command always emits orders 3 through 9, regardless of the selected
compiled range.
Regeneration requires Python 3 and clang-format. Never hand-edit generated
headers; `--check` verifies them without writing, and CTest's
`generated_fmm_current` runs that comparison against the source headers.

`cartesian_kernels` checks independent Cartesian polynomial references, translations,
nonzero dipoles, constant Laplacians, the truncated direct-particle remainder
series, and SIMD lanes for every order in the selected compiled range,
printing one pass line per order with the worst normalized discrepancy it saw.
`cartesian_kernels_cuda` executes the same operations on the device in float and
double, one launch per order. The complete `fmmWalk` regression compares
periodic forces and absolute potentials with an independent Ewald sum at the
runtime order, including periodic seams and excluded inactive particles.

`cartesian_l2l_cuda` independently checks every coefficient and every lane after one
and 17 translations, both potential modes, nonzero traces, linear fields, and
multiple warps per block, at each selected order's local degree. It checks
`translate` through the caller's synchronized in-place update pattern and
`translateInto` through ping-pong shared-memory source and target expansions.
The old CUDA generator omitted the last of 616 nonpotential terms,
`Lzzzzzz * dz^4 / 24` in `Lzz`; the repeated-shift checks detect that trace drift
in the unchanged legacy baseline, which exists only at local degree six.

`cartesian_ewald_cpu` also compiles the CUDA-disabled Ewald initializer and checks its
image list, self terms, normalization, periodic potential/gradient values, and
cubic axis-permutation symmetry at every selected order, re-filling the
tables through `setFmmOrder` and `EwaldConst::init`. The symmetry check includes
every derivative rank through that order's local degree. Moving the Ewald
emitter to Python removed signed-integer overflow in ten rank-eight projection
coefficients; Fourier table
generation also converts factors before multiplying to avoid the same overflow.
Equal-destination terms now accumulate in a deterministic stable order, so
floating-point results need not be bitwise identical to the former C++ emitter.

The nearest-image radial terms use a 20-term series for
`S_l(y) = integral_0^1 t^(2l) exp(-y t^2) dt` at the highest rank, followed by
the downward recurrence `S_(l-1) = (exp(-y) + 2 y S_l) / (2l - 1)`.
Here `y = alpha^2 r^2 <= 3` for wrapped separations. This avoids the
catastrophic cancellation of the former upward recurrence at small separation.
The float derivative regression compares every stored rank against double
precision at ten separations, including the origin and box corner, for every
selected order. The other periodic images retain their existing `erfc` recurrence.

CPU runtime validation uses
`--compute.cuda=false` in the CUDA-enabled application. A complete
`COSMICTIGER_WITH_CUDA=OFF` application build still has pre-existing
unconditional CUDA calls elsewhere in the application.

## Historical benchmarks

The Cartesian comparison harness, microbenchmarks, and associated runner and
analysis scripts are archived in the Git stash named
`retired_cartesian_benchmarks_and_generic_fmm_2026_10_27`.
The [measured results](stf_benchmark_results.md) remain as historical CPU, CUDA,
and cooperative L2L performance and numerical comparisons. The CUDA L2L
correctness checks remain in `tests/gravity/cartesian_l2l_cuda.cu`.
