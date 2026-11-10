# Generated FMM representations

`tools/generate_fmm.py` emits two representations together:

| Representation | Storage through degree d | Use |
| --- | --- | --- |
| Cartesian tracefree | `(d + 1)^2` symmetric trace-free coefficients | PMMM near-field FMM |
| Spherical | `(d + 1)^2` real slots for complex-conjugate harmonic coefficients | PMMM far field: free-space and periodic Ewald kernels |

The checked-in headers contain both families for local degrees 2 through 8,
source degrees 1 through 7, corresponding to application FMM orders 3 through 9.
The Cartesian near-field M2L uses the generated trace-free expansion. The PMMM
far field contracts spherical source moments with the cached Green filter in
Fourier space. `--min-order` and `--max-order` retain the
existing generator convention: their arguments are **local degrees**, not the
application FMM order.

New CMake configurations compile potential degrees 4 through 6 by default
(FMM orders 5 through 7); the runtime default is the highest compiled degree.
`COSMICTIGER_EXPANSION_DEGREE` selects one degree or a range, and CMake derives
`FMM_MIN_ORDER`/`FMM_MAX_ORDER` as degree plus one. Other orders remain in the
source headers but are excluded from compilation. Use
`-DCOSMICTIGER_EXPANSION_DEGREE=2-8` to compile every available degree.
When reusing an older cache, remove the retired options with
`-U COSMICTIGER_FMM_MIN_ORDER -U COSMICTIGER_FMM_MAX_ORDER`.

The representations are generated and tested independently. The runtime walk,
tree layout, checkpoint representation, and softening selection use Cartesian
tracefree; the far-field mesh uses the spherical kernels.

The C++ namespaces are `cosmictiger::gravity::cartesian` and
`cosmictiger::gravity::spherical`. Each has distinct `Multipole`, `Expansion`,
and `Green` types. Cartesian provides the destination-first tree operators;
the spherical family provides `P2M`, `evaluateGreen`, `L2P` and
`evaluateEwaldGreen`. Production PMMM uses the Green evaluators and basis casts;
the direct spherical particle operators remain test references. Field acceleration is minus the potential
gradient. Explicit casts convert Cartesian multipoles to spherical multipoles and spherical
expansions to Cartesian expansions at the same order and scalar type. Cartesian
callers include `cartesian_tracefree_operators.hpp` and qualify
`cartesian` (or use ADL with its types).

`Multipole<Real, order>` has source degree `order - 2`; `Expansion` and `Green`
have degree `order - 1`. The generated implementation functions take degrees.
Types use PascalCase, functions and variables camelCase, and namespaces and
filenames snake_case, with the standard FMM operator abbreviations retained.

Numerical coefficients are declared at each kernel function's entry in ascending
order of nonnegative magnitude, using `Scalar = SimdScalar<Real>` so they remain
scalar `constexpr` values for both scalar and SIMD instantiations. Names use
equal-width indices within each function: `k0` through `k8` for nine constants,
or `k00` through `k14` for fifteen. Literals have no leading plus, use uppercase
`E`, and have 18 digits after the decimal point. Exact rational coefficients
are rounded to nearest, with ties to even, using integer arithmetic before
conversion to `Scalar`. Opposite signs share one magnitude, with negation at
the use site.
Integer indices, template arguments, and loop bounds are not coefficients.

## Generate and verify

```sh
python3 tools/generate_fmm.py
python3 tools/generate_fmm.py --check
python3 tests/gravity/check_generated_fmm_flops.py --generated-dir cosmictiger/gravity/kernels --compiler c++
```

Generated headers and their operation report are checked in under
`cosmictiger/gravity/kernels/`:

- `cartesian_tracefree_types.hpp` contains named STF coefficient structs;
  `cartesian_tracefree_config.hpp` records the generated degree range.
- `cartesian_tracefree/{p2m,p2l,m2m,m2l,m2p,l2l,l2p,l2l_warp}.hpp`
  contains one Cartesian operator per header, with `geometry.hpp` for radius work.
- `spherical_kernels.hpp` and `spherical_ewald_kernels.hpp` contain the
  spherical kernels.
- `basis_conversion.hpp` defines the explicit coefficient casts; both type headers
  include these definitions. `spherical_types.hpp` is generated with its cast declarations.
- `operation_counts.json` records the generated arithmetic counts and selections.

The generator defaults to this source directory and emits both families for
FMM orders 3 through 9, formatting headers with the repository's clang-format
configuration. Use `--output-dir` to write a separate copy. Never hand-edit
generated headers; update their generators and regenerate them. `--check`
compares the expected output with the existing files without changing them.

CMake's explicit `generate_fmm_kernels` target always regenerates every family
for orders 3 through 9 in the source tree, independently of the selected build
range. `generate_cartesian_kernels` and `cosmictiger_generated_kernels`
remain aliases to the same command. Ordinary builds compile the checked-in
headers directly and do not regenerate or modify them. Regeneration requires
Python 3 and clang-format.

The numerical tests follow the selected build range: CMake creates
`generated_spherical_degree_<degree>` for each selected degree when gravity
is enabled, and
`generated_spherical_ewald`, `cartesian_kernels`, and the corresponding CUDA
tests exercise that range. `generated_fmm_current` checks reproducibility of
the complete source output. CTest's `generated_fmm_flops` passes the configured
range through `--min-fmm-order` and `--max-fmm-order`, compiling and executing
the selected Cartesian and spherical arithmetic specializations with an
independent counting scalar. Its availability checks verify that selected
Cartesian overloads and spherical kernel classes are defined, while excluded
orders are unavailable. Cartesian checks cover
the single M2L source degree and verify FMA counts under both
operation-count conventions. The direct Python counting command above omits
the range options and checks all generated orders 3 through 9.

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

## Spherical rotation selection

For each order, L2P generates three candidates: no rotation, azimuthal
rotation, and axial alignment. Generation selects the candidate with the
smallest emitted weighted count. A comment on each specialization records its
rotation variant; there is no C++ `Variant` type or variant template parameter.
Equal counts choose the first candidate. P2M and Green use direct harmonic
recurrences. The operation report retains the costs of every L2P candidate.

Costs include geometry, rotation, local evaluation, and accumulation into
the caller's existing output. Addition, subtraction, multiplication, division,
and an explicit fused multiply-add each cost **1**. A square root costs **7**;
`1/sqrt(x)` costs **8**. Constants, copies, unary sign changes, comparisons, and
selection cost zero. Costs are per scalar lane and describe emitted arithmetic,
not instruction counts or measured timings. `operation_counts.json` records all
candidates, selected variants, and the cost model. The original representation's
existing counts retain their documented scope, which excludes wrapper geometry.

The mathematics follows [Dehnen, A fast multipole method for stellar dynamics,
Appendix A.6](https://arxiv.org/pdf/1405.2255v1). The paper was published in 2014;
the cited PDF has a March 26, 2018 title-page date. The independently written
Python emitter also uses the author's
[spherical_fmm](https://github.com/dmarce1/spherical_fmm) implementation as a
normalization and optimization reference. Exact rational rotation coefficients,
conjugate symmetry, common-expression elimination, and dead-code elimination are
performed during generation.

## Traces and periodic fields

The spherical Ewald lattice correction is built into the cached PMMM Green
filter. PMMM contracts the filter and source moments in Fourier space; its
general harmonic contraction cannot use the free-space axial shortcut. The
periodic background is handled separately from the trace-free coefficient arrays.
The formulas follow
[Nitadori's PMMM paper, Appendix B](https://arxiv.org/html/1409.5981v2).
See [the spherical API](generated_spherical.md) and
[the Ewald convention and validation](generated_spherical_ewald.md) for signs,
normalizations, cutoff/domain assumptions, and auxiliary trace arguments.

## Generated weighted counts

The following table is generated from the operation report. Parentheses give
the chosen variant: none, azimuth, or full. Every value includes output
accumulation and geometry; FMA costs 1 and sqrt costs 7.

| FMM order | L2P |
| --- | --- |
| 3 | 45 (none) |
| 4 | 89 (axial) |
| 5 | 136 (axial) |
| 6 | 198 (axial) |
| 7 | 274 (axial) |
| 8 | 369 (axial) |
| 9 | 482 (axial) |

The spherical P2M and Green functions template only the scalar and output types;
L2P additionally deduces its source type from the input argument.
