# Small fixed-size matrices

`cosmictiger/math/matrix.hpp` provides stack-resident, compile-time dimensions:

- `math::Matrix<T, rows, columns = rows>` stores `rows * columns` entries.
- `math::SymmetricMatrix<T, n>` stores the lower triangle, `n * (n + 1) / 2`
  entries; `(i,j)` and `(j,i)` refer to the same entry.
- `math::DiagonalMatrix<T, n>` stores `n` entries, accessed with `diagonal(i)`.
  Read-only `(i,j)` returns zero off the diagonal.

All matrix types zero-initialize. Dense rows support `matrix[i][j]`; the uniform
interface is `matrix(i,j)`. Products work with the existing `math::Vector` and
`Vector3`, with the result dimensions following the operands. Rectangular matrix
products, mixed structured products, transpose, outer products, trace, and Frobenius
norms are included. A product of two symmetric matrices is a **dense** matrix: symmetry
is not generally preserved. Converting dense storage to packed symmetric storage copies
the lower triangle; it does not symmetrize the input.

Arithmetic supports SIMD elements. Inversion, determinant, solve and eigenpair APIs
require scalar floating-point elements; there is no lane-wise pivot solver. Functions
used by kernels are host/device compatible. There is no allocation, HPX dependency, IO,
or dynamic-size matrix container in this layer.

## Algorithms and compile-time selection

| Operation | General default cutoff | Symmetric default cutoff |
| --- | ---: | ---: |
| Inverse | 4 | 4 |
| Determinant | 4 | 4 |
| Solve | 3 | 3 |

`tryInverse`, `tryDeterminant`, and `trySolve` use explicit cofactors through the cutoff
and partial row pivoting above it. Diagonal operations are always direct, regardless of
dimension or cutoff. Six CMake cache options select the defaults:

```sh
-DCOSMICTIGER_MATRIX_GENERAL_INVERSE_THRESHOLD=4
-DCOSMICTIGER_MATRIX_SYMMETRIC_INVERSE_THRESHOLD=4
-DCOSMICTIGER_MATRIX_GENERAL_DETERMINANT_THRESHOLD=4
-DCOSMICTIGER_MATRIX_SYMMETRIC_DETERMINANT_THRESHOLD=4
-DCOSMICTIGER_MATRIX_GENERAL_SOLVE_THRESHOLD=3
-DCOSMICTIGER_MATRIX_SYMMETRIC_SOLVE_THRESHOLD=3
```

Each accepts 0 through 6; 0 disables the automatic cofactor path. The header uses the
same defaults outside CMake. A custom policy derived from `DefaultMatrixPolicy` can
override its `kGeneralInverseThreshold`, `kSymmetricInverseThreshold`, and corresponding
determinant/solve constants for an individual instantiation. Explicit `try*Explicit` and
`try*Pivoted` entry points are also available.

```cpp
math::SymmetricMatrix<double, 3> moment;
// Fill each independent entry once.
math::SymmetricMatrix<double, 3> inverse;
if (math::tryInverse(moment, inverse) == math::MatrixStatus::Success) {
    auto gradient = inverse * Vector3<double>{1.0, 2.0, 3.0};
}
```

The explicit path normalizes the matrix, uses specialized 1/2/3-dimensional cofactors or
minor expansion through dimension 6, and rescales with guarded reciprocals. Automatic
dispatch screens both its inverse-scale estimate and the magnitude of the normalized
determinant with `n * sqrt(epsilon)`, then retries pivoting after any rejected explicit
attempt. The determinant check catches cancellation when several eigenvalues are small,
even if the computed inverse-scale estimate looks acceptable. This is a conditioning
safeguard, not an exact condition-number calculation. Explicit entry points use only the
caller's requested tolerance.

`MatrixOptions<T>{relativePivotTolerance, absolutePivotTolerance}` controls pivot
acceptance. A negative relative tolerance selects `n * epsilon`; zero disables the
relative floor. Its optional third field, `relativeDeterminantTolerance`, is a
cofactor-only floor on the normalized determinant (default zero). Automatic dispatch
raises that floor for its explicit attempt only. The pivoted path works in original
units, so disabling the relative floor preserves systems with physically unequal
variable scales. Its LU factors are reused for multiple right-hand sides or an identity
RHS to form an inverse. Packed symmetric inverse outputs average opposite entries of the
dense inverse to avoid amplifying asymmetric roundoff when packing. Determinants
accumulate mantissas/exponents to avoid intermediate product overflow. Outputs still
obey the scalar type's representable range.

Every `try*` operation returns `Success`, `Singular` (including rejected small pivots),
or `NonFinite`, and leaves its output unchanged on failure. A singular determinant
returns `Singular`; it does not overwrite the output with zero. Float/double finite
checks work under fast math. Long-double nonfinite checking requires finite-math-only
optimizations to be disabled.

`matrix_eigen.hpp` supplies a scaled symmetric Jacobi eigensolver, with selectable
cyclic sweeps or largest-off-diagonal rotations. Eigenvalues sort largest first;
eigenvectors are columns. `Success` reports finite output, while `converged` separately
reports whether the requested tolerance was met within the iteration budget. Absolute
off-diagonal tolerances are expressed in **normalized matrix units**. The wrapper
symmetrizes dense input. The low-level Jacobi routine accepts an already normalized
packed symmetric matrix and leaves the eigenvalues unsorted.

## Call sites

- Cullen-Dehnen viscosity and Price conductivity: shared 3x3 gradient
  reconstruction. The existing normalized determinant cutoff `128 epsilon`,
  eight cyclic Jacobi sweeps and pseudoinverse eigenvalue cutoff `64 epsilon`
  remain unchanged. Missing spatial directions are not regularized into existence.
- ROCKSTAR: halo shape eigensystems. Its adapter retains row-oriented eigenvectors,
  largest-first order, and 64 largest-pivot rotations.
- Recombination: 3x3 Newton solve, retaining the absolute pivot floor `1e-200`
  and no relative floor.
- Boltzmann: the fixed dense core uses the shared pivot solve, retaining exact-zero
  pivot rejection. Dynamically sized tridiagonal hierarchy tails remain specialized.

FMM kernels, FFTs, large/dynamic storage, and specialized polynomial interpolation are
outside this change. The design draws on the fixed-size storage and interface in
`dmarce1/astrolib` and historical OctoTiger, with corrected dimensions and standalone
CosmicTiger algorithms instead of importing their dependency trees.

## Threshold measurements

The reproducible benchmark is `benchmarks/math/matrix.cpp`, also exposed as the
non-default `matrix_benchmark` CMake target when tests are enabled. It compares checked
explicit and pivoted APIs on 256 deterministic, diagonally dominant matrices for each
size 1 through 6, independently for float/double and general/packed-symmetric matrices.
Each timing is the median of seven interleaved runs after warmup. Complete outputs are
observable to prevent dead-code removal. This measures small-matrix throughput, not a
full application speedup.

The retained runs use GCC 15.2 and GCC 13.4, `-O3 -march=native -ffp-contract=off`,
Ryzen 9 3900X, one pinned CPU, and 32,768 calls per run. A GCC 15 repeat uses 131,072
calls per run. Raw nanoseconds and ratios are in
`matrix_results_20261102/gcc15_strict.csv`, `gcc15_strict_repeat.csv`, and
`gcc13_strict.csv`. The workstation was shared with War Thunder; these medians are local
measurements, not universal constants. At dimension 4 the winner depends on compiler,
scalar, and storage: GCC 15 general inverses and determinants favor explicit formulas;
packed symmetric double inverses favor pivoting, while float 4x4 inverses favor explicit
formulas. The defaults retain a common cutoff of 4 with separate tuning knobs; by
dimension 5 both are slower in all measured scalar/storage combinations. Direct solves
are already faster with pivoting at dimension 2 and mostly close at 3/4. Their cutoff is
3, as selected by the user after reviewing these measurements. A single contiguous
cutoff is not the fastest choice at every individual size. These CPU defaults are
configurable; GPU execution is checked for correctness, not tuned with a full-device
performance benchmark.

## Validation and performance limits

Tests cover rectangular and structured products, scalar/SIMD arithmetic, singular and
nonfinite inputs, failed-output preservation, aliasing, extreme scales, multiple
right-hand sides, and near-rank-one cancellation. Ill-conditioned inverse residual
bounds scale with the matrix condition; successful inversion cannot restore information
lost from the input precision. Standard and fast-math builds exercise the production
finite checks.

This refactor centralizes algorithms and improves edge-case handling; it is **not a
demonstrated whole-application speedup**. GPU validation uses one block at a time while War Thunder is on, and does not
establish device-wide performance.