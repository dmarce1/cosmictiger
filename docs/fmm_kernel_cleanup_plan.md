# FMM generated-kernel cleanup plan

Implemented November 9, 2026 in `~/mywork/cosmictiger`. This document records
the agreed generator and kernel changes. The generated Cartesian kernels retain
the source dipole at every order; there are no dipole-free specializations.

## Shared scalar CUDA and CPU SIMD math

- Move the shared helpers from the global namespace into `math`. Rename
  `simdFma` to `math::fma`; likewise use `math::broadcast`, `math::select`,
  `math::min`, `math::max`, `math::sqrt`, `math::anyOf`, and `math::allOf` for
  `simdBroadcast`, `simdSelect`, `simdMin`, `simdMax`, `simdSqrt`, `simdAnyOf`,
  and `simdAllOf`. Retain fused rounding in `math::fma` on scalar CUDA and CPU
  SIMD paths.
- Check the existing `math::max(double, double)` overload and the separate
  `special_functions::fma`/`sqrt` implementations when making these changes.
  Update generators and generated kernels with their callers. CPU pack-only
  utilities are outside this naming decision.

## Cartesian M2L scaling

- Default generated Cartesian M2L to compute `iR` from its `Vector3`
  displacement. The production path uses `iR` for both source and target
  radial factors.
- Provide an explicit `tools/generate_fmm.py` option, disabled by default, to
  generate independent `sourceRatio` and `targetRatio` inputs for callers with
  separately scaled source moments and target local coefficients. Suggested
  spelling: `--independent-m2l-ratios`.
- Keep the typed wrapper, generator mode, generated headers, operation report,
  `--check`, and default/option-enabled tests consistent. This option concerns
  Cartesian M2L only; other kernels' scale inputs need their own decisions.

## Mathematical names and symbol directories

- Emit a `Symbol directory:` near the top of each generated kernel header,
  covering the M2L, M2M, P2L, P2M, M2P, L2L, L2P, Green, and Ewald functions
  present in that file. State the actual equations and conventions used by
  each representation. Include only symbols used by the file, such as `R` for
  separation radius and `iR` for inverse radius, and define center offsets,
  source moments, local coefficients, and Green coefficients where applicable.
- In generated functions, use short mathematical aliases with clear meanings,
  such as `auto& L = local;` and `auto const& M = source;`, when they improve the
  emitted formulas. `L` cannot be `auto const&` when the kernel writes into it.
  Generate these aliases and the directory from Python; do not hand-edit the
  checked-in generated headers. Preserve the project's equation-identifier
  conventions and avoid aliases that obscure a source/destination distinction.

## Unused arguments and emitted FMA operands

- The generated `(void) ux;`-style statements currently silence unused
  parameter warnings. Remove them. For each generated function, leave an
  unused parameter's type in the required signature but omit its name. Do not
  use `[[maybe_unused]]` or a `(void)` cast as a substitute. The default
  one-radius M2L interface removes the two ratio parameters entirely.
- `fmaOutput*First/Second` copies preserve reads at the original DAG point
  before earlier destination writes and across conditional emission. For M2L,
  source and local are separate by contract, so inline the original operand
  expressions into generated `fma` calls, with a function-local `math` import,
  where an alias and evaluation-order audit
  proves the copies unnecessary. Apply the same audit to other generated
  operators; retain a temporary only where it prevents a changed read or
  conditional-scope error. Verify generated numerical results after removal.

## FLOP metadata

- Cartesian M2L uses source degree exactly one less than local degree. Remove
  the optional source-degree-equals-local-degree kernel branch, its template
  selector, full-order tests, and `Full` cost constants. That extra branch
  affects only potential and is not used by the public production multipole
  type. The generated C++ retains `kFlops_`, which counts FMA as two
  arithmetic operations. Weighted counts use FMA = 1 and remain in JSON for
  generator comparisons and tests.
- Simplify the generated C++ interface: retain only costs required by callers
  at runtime, and keep alternate counting conventions in generator metadata or
  tests where useful. Retain accurate operation reporting without treating any
  static count as a measured hardware throughput.

## Source dipole

- Generate one general Cartesian direct M2L, derivative M2L overload, and M2P kernel per order.
  These retain rank-one source moments. Remove the `centered` template parameter,
  the dipole-free generated variants, their duplicate test outputs, and their
  operation-count entries.

## Order-specific Cartesian functions and coefficient fields

- Generate Cartesian kernels as free functions with the order fixed by their
  `Multipole<Real, order>`, `Expansion<Real, order>`, and `Green<Real, order>`
  arguments. `Real` is the only function template parameter. Overload `M2L`
  for `Vector3<Real>` displacement and `Green<Real, order>` derivatives.
- Keep each generated operator in its own header under
  `cosmictiger/gravity/kernels/cartesian_tracefree/`; place the cooperative
  CUDA L2L functions in `l2l_warp.hpp`.
- Use named coefficient fields such as `m`, `x`, `xx`, and `xy` while
  preserving rank-major compact storage. Size buffers and archives for the active
  trace-free coefficient layout. Recover
  `zz` and higher z-pair components from the trace-free identity. Overlay
  the named fields with an equal-length array in a union, and serialize
  that array in stored order. Retain indexed access for dynamic CUDA lanes
  and order-independent computation.
- Pass position differences as `Vector3<Real> const&`. Let `dx`, `dy`, and
  `dz` mean raw displacement in every Cartesian kernel, and compute inverse
  radius inside the direct distance kernels. Remove the redundant ratio aliases
  from P2L and M2P. The independent M2L ratio option remains explicit and off
  by default.
