# Generated Cartesian FMM kernels

`tools/generate_fmm.py` emits compact Cartesian operators for the global
periodic FMM. By default, a tensor through degree d stores `(d + 1)^2 + 1`
coefficients when d >= 2: the harmonic slots followed by an independent `.zz`.
Degree-two source moments are raw symmetric moments, and local/Green tensors
store the full Hessian. All ranks above two remain trace-free.

Each build generates kernels for its single potential degree, selected by
`COSMICTIGER_EXPANSION_DEGREE` (default 6, supported values 2 through 8).
Degree d uses source degree d-1 and internal FMM order d+1. Runtime order
selection is removed. Checkpoints retain the internal order for compatibility
validation. A different degree requires another build.

`COSMICTIGER_DIPOLE_FREE=ON` selects center-of-mass tree multipoles and generates
P2M/M2M kernels without rank-one output and M2M/M2L/M2P/periodic M2L kernels
without source dipole terms. The default is `OFF`, retaining geometric center
selection and general kernels. Local expansions still retain their rank-one
coefficients, which represent the field gradient. Coefficient storage and
checkpoint layouts stay unchanged; the three source dipole slots remain zero
when constructing a dipole-free tree. External kernel callers must initialize
their accumulators and supply multipoles centered at their centers of mass.

Leaves compute mass-weighted centers from native fixed-point differences;
parents combine child masses and centers. A double-precision residual travels
with each tree-build return so fixed-point center rounding does not accumulate
up the tree. Radii enclose particles about the stored center and round outward
when narrowed to float. Centers are limited by the selected fixed32/fixed64
position grid; the corresponding residual physical dipole is omitted, rather
than exactly representable at arbitrary mass-weighted coordinates. M2M also
omits the dipole of an individual translated child; its accumulated result is
valid as a source at the parent center of mass.

The generator exposes the same choice as `--dipole-free` / `--no-dipole-free`.
Builds own their generated mode, reported by `cartesian::kDipoleFree` and the
`dipole_free` field in `operation_counts.json`. Reconfiguring a build in either
direction regenerates its kernels. The checked-in references use the default
mode.

Generated build declarations live under `generated_fmm/cosmictiger/gravity/kernels`
and take precedence over the checked-in default-degree reference headers.
Build implementations live under `generated_fmm/src/gravity/kernels`.
The generator's `--degree` argument selects one potential degree; without it,
it regenerates the degree-6 reference declarations and sources. Build generation never modifies
the source tree, so independent builds may use different degrees concurrently.

The runtime walk, tree layout, checkpoint representation, and periodic correction
use these compact Cartesian coefficients. The C++ namespace
`cosmictiger::gravity::cartesian` provides distinct `Multipole`, `Expansion`,
and `Green` types, source-first tree operators, and periodic Ewald operators.
Field acceleration is minus the potential gradient. Callers include
`cartesian_tracefree_operators.hpp` and qualify `cartesian` (or use ADL with its types).

`Multipole<Real>` has source degree `kFmmOrder - 2`; `Expansion<Real>` and
`Green<Real>` have degree `kFmmOrder - 1`. The build fixes these shapes; neither
the coefficient types nor kernel calls take an order template argument.
Generated operators are ordinary scalar/SIMD overloads. Accumulators come last,
for example `M2P(M, distance, field)` and `P2M(mass, distance, M)`. Operations
continue to accumulate in place, with generated arithmetic and FLOP counts.
Types use PascalCase, functions and variables camelCase, and namespaces and
filenames snake_case, with the standard FMM operator abbreviations retained.

Numerical coefficients are declared at each kernel function's entry in ascending
order of nonnegative magnitude as scalar `constexpr` values. Concrete float
implementations use direct `f`-suffixed literals; double implementations use
unsuffixed literals. No scalar alias or cast is needed in those implementations. Names use
equal-width indices within each function: `k0` through `k8` for nine constants,
or `k00` through `k14` for fifteen. Concrete implementations declare
`constexpr float` constants with `%14.8e` literals or `constexpr double`
constants with `%23.17e` literals (8 or 17 fractional digits, lowercase `e`).
SIMD implementations use the precision of their lanes. Exact rational coefficients
are rounded to nearest, with ties to even, using integer arithmetic before
conversion to the declared scalar type. Opposite signs share one magnitude, with negation at
the use site.
Integer indices, template arguments, and loop bounds are not coefficients.

## Compiled library

`cosmictiger_fmm` is a static library containing the generated arithmetic in
`src/gravity/kernels/`. Every build compiles scalar and SIMD `.cpp` files with
ordinary CPU-only definitions. CUDA builds additionally compile scalar `.cu`
files with explicit `__device__` definitions. Each file contains complete function
bodies; neither includes the other. Declarations select the host or device
execution space for the current compiler pass. CUDA host calls link the same
C++ definitions as other CPU callers. Concrete bodies spell out their types;
scalar constants are used directly and only SIMD values use broadcasts.
The warp L2L functions and both schedule tables live in one CUDA unit. CUDA
callers use separable compilation and resolve device symbols at executable link.
The library disables implicit multiply-add contraction; generated explicit
`math::fma` calls remain fused, preserving the emitted arithmetic graph regardless of consumer flags.
Link this target when using the generated operators; including a header alone
no longer supplies their arithmetic implementations.

Concrete implementations cover `float`, `double`, `SimdVector<float, 8>` and
`SimdVector<double, 2>` used by the solver, plus four-lane float/double packs
used by numerical reference tests, and eight-lane double overloads. CUDA supports float/double. Production headers declare concrete overloads; generic declarations are enabled
only by the counting-test instrumentation macros.

`tests/gravity/kernels/instrumentation/*.cpp` contains generic definitions solely for counting-scalar
tests. Those tests explicitly include the relevant source definitions; normal
headers and production targets never include them. Concrete overloads in
those test executables still link to the library. The standalone FLOP audit uses
`COSMICTIGER_FMM_COUNTING_ONLY` with its replacement arithmetic backend.

## Generate and verify

```sh
python3 tools/generate_fmm.py
python3 tools/generate_fmm.py --check
python3 tests/gravity/check_generated_fmm_flops.py --generated-dir cosmictiger/gravity/kernels --compiler c++
```

Generated implementations are checked in under `src/gravity/kernels/`.
Declarations and their operation report remain under `cosmictiger/gravity/kernels/`:

- `cartesian_tracefree_types.hpp` contains named coefficient structs;
  `cartesian_tracefree_config.hpp` records the generated degree and trace option.
- `cartesian_tracefree/{p2m,p2l,m2m,m2l,m2p,l2l,l2p,l2l_warp}.hpp`
  declares one Cartesian operator per header, with `geometry.hpp` for radius work.
- `cartesian_tracefree/{ewald_green,ewald_m2l}.hpp` declares the periodic
  Ewald correction operators.
- `operation_counts.json` records the generated arithmetic counts and selections.

The generator defaults to these source-tree directories and emits Cartesian kernels for
the selected degree (default 6), formatting C++ with the repository's clang-format
configuration. `--quadrupole-trace` is enabled by default; `--no-quadrupole-trace`
emits the harmonic-only algebra for standalone use. The production periodic
solver uses the default full degree-two representation. Use `--output-dir` for a separate declaration/report directory and
`--source-dir` for its production implementations and `--instrumentation-dir`
for its counting-test implementations. With only a custom `--output-dir`,
these go into its `sources/` and `instrumentation/` subdirectories, keeping generation isolated
from the source tree. Never hand-edit
generated files; update their generators and regenerate them. `--check`
compares the expected output with the existing files without changing them.

CMake automatically generates one operator set before compilation; the explicit
`generate_fmm_kernels` target and its aliases use the same dependency rule.
Changing the degree or generator reruns generation. Python 3 and clang-format
are required. The numerical tests and `generated_fmm_current` check that build's
single order. `generated_fmm_flops` independently counts executed arithmetic
for the selected Cartesian operators.

## Cartesian tracefree fused arithmetic

The shared Cartesian emitter fuses single-use products into addition,
subtraction, and destination accumulation through explicit `fma` calls.
Each generated function imports `math` locally; the local `Fma` callable keeps
CPU SIMD overload resolution on the fused helper. Products used elsewhere remain
shared temporaries. The final emission pass
preserves the rational DAG's factoring and checks uses before folding a
product. Fused rounding is intentional. Fused output operands are passed
directly to `fma`; source and destination must be distinct.

Cartesian kernels expose order-specific `k*Flops` constants with FMA = 2 for application performance
counters. Its JSON report retains weighted counts with FMA = 1 for generation
comparisons. M2L source degree is always local degree minus one. The optional
`--independent-m2l-ratios` generator flag emits separately scaled source and
target ratio inputs; the default kernel computes inverse radius from its `Vector3`
displacement. In the JSON
report, arithmetic dictionaries count each explicit FMA once; accumulation
counts include only standalone additions/subtractions.
Thus weighted cost is the arithmetic sum plus accumulations; conventional cost
adds one more per FMA. The cooperative CUDA L2L schedule and periodic Green
derivative emitter retain their existing explicit FMAs and count conventions.

## Traces and periodic fields

Global FMM evaluates the Cartesian Ewald correction during the tree walk.
The primary force and its regular periodic correction use a common image
throughout each accepted expansion. The periodic checklist holds pairs with
both interactions pending. Image-consistent pairs can resolve Ewald early
and transfer their remaining primary interaction to the Newtonian checklist.
Pairs crossing an image-selection plane resolve both interactions together
with Newtonian M2L and `M2Lewald` at identical centers. The Ewald separation
and smoothing depth requirements are retained: they guarantee that crossing
leaf pairs can use this joint M2L. There is no Ewald particle-pair fallback.
`evaluateEwaldGreen(distance, green)` sets the full Hessian, including
`green.zz = -4*pi/kappa^3 - green.xx - green.yy`. The generated
`M2Lewald(source, green, local)` contracts the whole periodic field, including
the constant-Laplacian contributions. `P2M` and `M2M` retain all six raw
degree-two moments; Newtonian contractions use their STF projection.
`L2L`, cooperative CUDA `L2LWarp`, and `L2P` propagate and evaluate the full
quadratic, so no separate radial-moment or curvature field is required.
The stored trace and the coupled image choice address separate requirements:
the former retains the Poisson quadratic, while the latter keeps the primary
and periodic image contributions consistent.
