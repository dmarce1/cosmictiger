# Generated Cartesian FMM kernels

`tools/generate_fmm.py` emits Cartesian trace-free operators for the global
periodic FMM. A tensor through degree d stores `(d + 1)^2` coefficients.

Each build generates kernels for its single potential degree, selected by
`COSMICTIGER_EXPANSION_DEGREE` (default 6, supported values 2 through 8).
Degree d uses source degree d-1 and internal FMM order d+1. Runtime order
selection is removed. Checkpoints retain the internal order for compatibility
validation. A different degree requires another build.

Generated build declarations live under `generated_fmm/cosmictiger/gravity/kernels`
and take precedence over the checked-in default-degree reference headers.
Build implementations live under `generated_fmm/src/gravity/kernels`.
The generator's `--degree` argument selects one potential degree; without it,
it regenerates the degree-6 reference declarations and sources. Build generation never modifies
the source tree, so independent builds may use different degrees concurrently.

The runtime walk, tree layout, checkpoint representation, and periodic correction
use Cartesian trace-free coefficients. The C++ namespace
`cosmictiger::gravity::cartesian` provides distinct `Multipole`, `Expansion`,
and `Green` types, source-first tree operators, and periodic Ewald operators.
Field acceleration is minus the potential gradient. Callers include
`cartesian_tracefree_operators.hpp` and qualify `cartesian` (or use ADL with its types).

`Multipole<Real>` has source degree `kFmmOrder - 2`; `Expansion<Real>` and
`Green<Real>` have degree `kFmmOrder - 1`. The build fixes these shapes; neither
the coefficient types nor kernel calls take an order template argument.
Generated operators are ordinary scalar/SIMD overloads. Accumulators come last,
for example `M2P(M, distance, field)` and `P2M(mass, distance, M)`. Operations
continue to accumulate in place, with unchanged arithmetic and FLOP counts.
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

- `cartesian_tracefree_types.hpp` contains named STF coefficient structs;
  `cartesian_tracefree_config.hpp` records the generated degree.
- `cartesian_tracefree/{p2m,p2l,m2m,m2l,m2p,l2l,l2p,l2l_warp}.hpp`
  declares one Cartesian operator per header, with `geometry.hpp` for radius work.
- `cartesian_tracefree/{ewald_green,ewald_m2l}.hpp` declares the periodic
  Ewald correction operators.
- `operation_counts.json` records the generated arithmetic counts and selections.

The generator defaults to these source-tree directories and emits Cartesian kernels for
the selected degree (default 6), formatting C++ with the repository's clang-format
configuration. Use `--output-dir` for a separate declaration/report directory and
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
The nearest-image primary force and its regular periodic correction use a
single image throughout each accepted expansion. Cells crossing an image
selection plane are opened; unresolved leaf pairs are evaluated directly.
Source radial second moments and local background curvature are stored
separately from trace-free coefficients; see `gravity/periodic_fmm.hpp`.
