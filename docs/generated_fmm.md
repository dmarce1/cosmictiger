# Generated FMM representations

`tools/generate_fmm.py` emits two representations together:

| Representation | Storage through degree d | Use |
| --- | --- | --- |
| `Cartesian` | `(d + 1)^2` symmetric trace-free coefficients; an optional local Laplacian slot | Existing production Newtonian FMM |
| `Spherical` | `(d + 1)^2` real slots for complex-conjugate harmonic coefficients | Newtonian free-space and periodic Ewald kernels |

The checked-in headers contain both families for local degrees 2 through 8,
source degrees 1 through 7, corresponding to application FMM orders 3 through 9.
M2L keeps terms of combined source and local degree at most the local-degree
limit. `--min-order` and `--max-order` retain the
existing generator convention: their arguments are **local degrees**, not the
application FMM order.

New CMake configurations compile orders 4 through 8 by default; the runtime
default remains order 7. `COSMICTIGER_FMM_MIN_ORDER` and
`COSMICTIGER_FMM_MAX_ORDER` set the `FMM_MIN_ORDER`/`FMM_MAX_ORDER` preprocessor
guards and the per-order instantiations. Orders outside that selected range
remain in the source headers but are excluded from compilation. Existing
caches retain their configured range. Set
`-DCOSMICTIGER_FMM_MIN_ORDER=3 -DCOSMICTIGER_FMM_MAX_ORDER=9` when reconfiguring
to compile every available order.

The representations are generated and tested independently. The runtime
walk, tree layout, checkpoint representation, and softening selection still use
`Cartesian`. Connecting a new family to those paths is separate
work. `Cartesian` remains symmetric and trace-free; the shorter name does not
change its coefficient layout or mathematics. Namespaces, generator modules,
test targets, and related source filenames use `cartesian`.
Types use PascalCase, functions and variables camelCase, and namespaces and
filenames snake_case.

## Generate and verify

```sh
python3 tools/generate_fmm.py
python3 tools/generate_fmm.py --check
python3 tests/gravity/check_generated_fmm_flops.py --generated-dir cosmictiger/gravity --compiler c++
```

Generated headers and their operation report are checked in under
`cosmictiger/gravity/`:

- `generated_cartesian_m2l.hpp`, `generated_cartesian_operators.hpp`, and
  `generated_config.hpp` contain the Cartesian operators and degree configuration.
- `dehnen/generated_cartesian_ewald.hpp` contains the Cartesian Ewald derivatives.
- `nitadori/generated_spherical.hpp` and
  `nitadori/generated_spherical_ewald.hpp` contain the spherical kernels.
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
`generated_spherical_order_<order>` for each selected order, and
`generated_spherical_ewald`, `cartesian_kernels`, and corresponding Ewald/CUDA
tests exercise that range. `generated_fmm_current` checks reproducibility of
the complete source output. CTest's `generated_fmm_flops` passes the configured
range through `--min-fmm-order` and `--max-fmm-order`, compiling and executing
the selected Cartesian and spherical arithmetic specializations with an
independent counting scalar. Its `CompleteKernel` checks verify that selected
kernel classes are defined and excluded classes are not. Cartesian checks cover
both M2L source-degree branches and verify FMA counts as well as both
operation-count conventions. The direct Python counting command above omits
the range options and checks all generated orders 3 through 9.

## Cartesian fused arithmetic

The shared Cartesian emitter fuses single-use products into addition,
subtraction, and destination accumulation through explicit `simdFma` calls.
Products used elsewhere remain shared temporaries. The final emission pass
preserves the rational DAG's factoring and checks uses across both M2L source
degree branches before folding a product. Fused rounding is intentional.
Fused output operands are copied at their original evaluation point, preserving
the existing read order when input and output buffers overlap.

Ordinary Cartesian kernels expose `kWeightedFlops_` with FMA = 1, and retain
`kFlops_` with FMA = 2 for the application's existing performance counters.
The M2L `Full` constants describe the optional source-degree-equals-local-degree
branch. In the JSON report, arithmetic dictionaries count each explicit FMA
once; accumulation counts include only standalone additions/subtractions.
Thus weighted cost is the arithmetic sum plus accumulations; conventional cost
adds one more per FMA. The cooperative CUDA L2L schedule and periodic Green
derivative emitter retain their existing explicit FMAs and count conventions.

## Spherical rotation selection

For each order, M2M, M2L, M2P, L2L, and L2P generate three candidates: no rotation,
azimuthal rotation only, and full rotation. The public kernel inherits the
candidate with the smallest emitted weighted count. Equal counts choose the
first candidate in that order. P2M/P2L use direct solid-harmonic recurrences.
All candidates remain available for independent numerical comparisons.

Costs include geometry, rotation, translation/contraction, and accumulation into
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

Spherical Ewald is a lattice correction, so it uses a general harmonic
contraction rather than the free-space axial shortcut. The generator carries
separate source radial-second-moment and local background-trace values. These
are required for the uniform periodic background, including nonzero dipoles and
translation/evaluation of the quadratic local term. The formulas follow
[Nitadori's PMMM paper, Appendix B](https://arxiv.org/html/1409.5981v2).
See [the spherical API](generated_spherical.md) and
[the Ewald convention and validation](generated_spherical_ewald.md) for signs,
normalizations, cutoff/domain assumptions, and auxiliary trace arguments.

## Generated weighted counts

The following table is generated from the operation report. Parentheses give
the chosen variant: none, azimuth, or full. Every value includes output
accumulation and geometry; FMA costs 1 and sqrt costs 7.

| FMM order | M2M | M2L | M2P | L2L | L2P |
| --- | --- | --- | --- | --- | --- |
| 3 | 9 (none) | 62 (none) | 55 (none) | 50 (none) | 45 (none) |
| 4 | 46 (none) | 144 (azimuth) | 94 (full) | 130 (none) | 89 (full) |
| 5 | 124 (none) | 270 (azimuth) | 146 (full) | 279 (none) | 136 (full) |
| 6 | 271 (none) | 454 (azimuth) | 216 (full) | 464 (full) | 198 (full) |
| 7 | 465 (full) | 710 (full) | 304 (azimuth) | 695 (full) | 274 (full) |
| 8 | 696 (full) | 1001 (full) | 401 (azimuth) | 990 (full) | 369 (full) |
| 9 | 990 (full) | 1370 (full) | 510 (azimuth) | 1353 (full) | 482 (full) |

## Cartesian versus Spherical direct M2L

These counts include displacement geometry and additive output, retain arbitrary
source dipoles, and use the same degree truncation. Application coordinate-unit
conversion is excluded for both representations. Cartesian geometry adds 14 to
its generated body: one multiply, two FMAs, one square root, one division, and
three coordinate multiplies. Both generators emit explicit FMAs. FMA costs 1
and sqrt costs 7; the table counts emitted arithmetic, not measured instruction
counts or timings.

| FMM order | Cartesian | Spherical | Spherical rotation |
| --- | --- | --- | --- |
| 3 | 47 | 62 | none |
| 4 | 99 | 144 | azimuth |
| 5 | 200 | 270 | azimuth |
| 6 | 393 | 454 | azimuth |
| 7 | 707 | 710 | full |
| 8 | 1230 | 1001 | full |
| 9 | 2048 | 1370 | full |
