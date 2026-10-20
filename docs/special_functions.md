# Special-function comparison

Measurements: 2026-10-20, Ryzen 9 3900X, GCC 13.3, RTX 3090 (82 SMs),
CUDA 12.8.93, compute capability 8.6. Repository base:
`d8e957281698a6773626429a5ba76a8353ff3ad6`, with the new benchmark sources and
candidate headers in the working tree.

This change supplies comparisons and accurate candidate implementations.
Existing production call sites still use their existing implementations.
There is no operation-count structure.

## CPU results

Median **nanoseconds per value**, written **custom / fastest accurate library**
in each cell. Smaller is better. `standard` omits fast-math; `fast` matches
the application's fast-math setting. The library results passed the sampled
accuracy checks, rather than being assumed accurate.

| Function | Standard scalar-source | Standard SIMD8 | Fast scalar-source | Fast SIMD8 |
| --- | ---: | ---: | ---: | ---: |
| sin | 13.195 / 7.699 | 1.974 / 1.143 | 10.203 / 0.499 | 1.989 / 1.120 |
| cos | 10.126 / 7.650 | 1.928 / 1.167 | 7.623 / 0.491 | 1.952 / 1.121 |
| sincos | 13.247 / 9.040 | 2.792 / 3.118 | 13.385 / 9.778 | 2.793 / 3.226 |
| exp | 3.129 / 2.289 | 0.800 / 2.504 | 2.959 / 0.438 | 0.828 / 0.999 |
| exp2 | 2.372 / 2.515 | 0.731 / 2.250 | 2.371 / 0.372 | 0.722 / 0.910 |
| log | 3.157 / 2.477 | 1.076 / 2.896 | 1.011 / 0.467 | 1.242 / 1.053 |
| log2 | 2.567 / 2.424 | 0.995 / 2.849 | 1.010 / 0.424 | 1.139 / 0.981 |
| erf | 13.096 / 15.050 | 3.488 / 14.627 | 11.567 / 0.611 | 3.630 / 1.160 |
| erfc | 8.415 / 15.608 | 2.664 / 16.003 | 8.697 / 0.795 | 3.070 / 1.346 |
| erfcexp | 8.377 / 20.475 | 3.522 / 20.582 | 8.445 / 1.342 | 3.583 / 2.806 |
| sqrt | 1.378 / 1.414 | 0.176 / 0.178 | 1.378 / 0.157 | 0.178 / 0.177 |
| rsqrt | 0.866 / 2.365 | 0.550 / 0.298 | 0.799 / 0.112 | 0.548 / 0.182 |

The CPU conclusion depends on compiler settings:

- In the standard build, custom `erfcexp` is about **2.4× faster scalar**
  and **5.8× faster SIMD8** than the accurate separate library pair.
- With fast-math, the compiler's vector library wins the scalar-source array
  loops. In explicit SIMD8, custom `sincos`, `exp`, and `exp2` win; library
  `sin`, `cos`, `log`, `log2`, `erf`, and `erfc` win.
- The fastest accurate fast-math SIMD8 `erfcexp` library combination uses
  double Gaussian evaluation: 2.806 versus 3.583 ns/value. The FMA-corrected
  float pair takes about 20 ns/value because of GCC's vectorization choice.
  The scalar-source corrected float pair auto-vectorizes to 1.342 ns/value.
- SIMD sqrt is effectively tied: both use hardware instructions. The
  existing/library SIMD reciprocal root beats the new guarded Newton wrapper.

The old formulas are also timed in the raw data. Their apparent speed does
not qualify them for the requested accuracy: old SIMD trig fails near phase
zeros, exp/log fail the ULP target, and old `erfcexp` has errors exceeding
500,000 ULP in the required Ewald interval. The old base-two exponential
passes the sampled check.

Raw measurements and accuracy logs:
[special_functions_results_20261020](special_functions_results_20261020/).

## CUDA results

Array/event timing spreads overlapped substantially. A second measurement
keeps values in registers and makes each evaluation depend on the preceding
result. This gives a stable comparison of calculation cost under four warps
per block, rather than a claim about saturated GPU throughput.

Median **clock cycles per dependent evaluation**, **custom / library**.
These include the common loop and input-update overhead. The trig, exp, log,
and log2 library columns use accurate libdevice entry points. The paired
entry uses `erfcf` plus the accurate exponential and square correction.
The root columns use ordinary library roots, which meet the sampled 8-ULP
threshold; the custom root wrappers demand tighter rounding.

| Function | Standard CUDA | CUDA fast-math |
| --- | ---: | ---: |
| sin | 141 / 122 | 151 / 122 |
| cos | 137 / 126 | 145 / 127 |
| sincos | 146 / 118 | 156 / 120 |
| exp | 123 / 66 | 122 / 70 |
| exp2 | 115 / 73 | 118 / 41 |
| log | 180 / 120 | 138 / 120 |
| log2 | 172 / 130 | 131 / 130 |
| erf | 275 / 100 | 200 / 102 |
| erfc | 224 / 146 | 155 / 151 |
| erfcexp | 226 / 158 | 148 / 155 |
| sqrt | 76 / 76 | 74 / 41 |
| rsqrt | 147 / 73 | 117 / 41 |

Thus the accurate **library is the CUDA default recommendation for most
functions**. The custom combined `erfcexp` wins by about **4.5% in the
fast-math dependent-loop probe**, but loses in the standard build. Its
advantage should be checked in an actual Ewald kernel before changing
production routing. Event/array medians alone did not establish that win.

Accuracy changes the comparison materially:

- Ordinary fast CUDA sin/cos/sincos fail the relative ULP target near
  multiples of π; accurate libdevice calls pass the sampled checks.
- Ordinary fast exp reached 53 ULP in the final sample. For paired outputs,
  the naive library Gaussian reached 87 ULP; square correction alone still
  reached 49 ULP. Accurate libdevice plus correction stayed within 3 ULP.
- The new combined function stayed within 4 ULP in the final CUDA samples
  and passed the larger reference suite. The legacy CUDA pair exceeded
  500,000 ULP.
- Double Gaussian evaluation was much slower on this GPU. That CPU
  fast-math optimization should not be carried over to CUDA.

All five candidate accuracy configurations passed: **8,399,776 result
comparisons** total across baseline/native/fast CPU and standard/fast CUDA.
The test suite also verifies the CUDA `pow` fix and hardware-root edge cases.

## What is compared

- `legacy`: the existing functions in `cosmictiger/math/simd.hpp`; on CUDA,
  the existing `erfcexp` formula from `cosmictiger/math/math.hpp`.
- `custom`: the new candidates exposed by
  `cosmictiger/math/special_functions.hpp`. They support scalar float on CPU
  and CUDA, and the project's eight-float CPU SIMD pack. Scalar double
  overloads preserve double precision through platform math.
- `library`: ordinary scalar or `std::experimental::simd` math. CUDA
  `--use_fast_math` can replace ordinary functions with approximate intrinsics.
- `library_fma`: `erfc(x)` and `exp(-x*x)`, correcting the Gaussian for the
  rounding of the float square using two FMAs.
- `library_double`: `erfc(x)` and a double-precision Gaussian, rounded to float.
- `libdevice_accurate`: CUDA's documented libdevice entry points, bypassing
  the fast-math substitutions. Its paired Gaussian includes the FMA correction.

The new candidate implementation uses ideas from
[dmarce1/simd at dc85446](https://github.com/dmarce1/simd/tree/dc85446d2a7de89c366c54abb6a024aed13eafc5),
particularly split range reduction, an atanh logarithm expansion, and
computing erfc through a scaled-error-function factor. The coefficients and
implementation here are independent of that library's interval tables.
`tools/generate_special_coefficients.py` reproduces the new degree-15
transformed erfcx polynomial using mpmath.

## Where the functions are used

| Function | Existing implementation and consumers |
| --- | --- |
| `erfcexp` | SIMD and CUDA custom formulas; scalar double library composition. Generated Ewald real-image derivatives and `src/gravity/analytic.cu`. Returns both **erfc(x)** and **exp(-x²)**, rather than erfcx(x). |
| `sincos`, `sin`, `cos` | Custom SIMD functions; platform scalar/CUDA calls. Generated Ewald Fourier terms; analytic gravity; random Fourier phases in initialization through `expc`. |
| `exp`, `twoPow` | Custom SIMD exponential and base-two exponential. Ewald nearest-image terms and the Gaussian within `erfcexp`. |
| `erf`, `erfc`, `log`, `log2`, `pow` | Custom SIMD convenience functions. The principal Ewald path uses the paired `erfcexp` directly. |
| `sqrt`, `rsqrt`, FMA | Platform/SIMD helpers throughout gravity, FMM, SPH, and vector norms. The candidates compare hardware roots and a refined x86 reciprocal-root estimate. |
| `sinc`, `expc` | Compositions of platform functions, used for FFT deconvolution and complex Fourier phases respectively. |
| `erfnearzero`, `expnearzero` | Older standalone Taylor expansions in `math.hpp`; no active production callers found. |
| Factorials/double factorials | Integer recurrences or lookup tables for tensor/Ewald setup; outside this floating-point transcendental benchmark. |

## Required ranges and accuracy

The real-space Ewald scalar cutoff requires `x=2r < 5.2`. SIMD's
`simdAnyOf` branch evaluates all lanes if any lane qualifies: another lane can
reach `2r = sqrt(75)`, so testing only `[0,5.2]` would miss required inputs.
The candidates cover `erfcexp` through `|x| <= 9` and exponential arguments
down to `-75`. Fourier phases are within approximately `[-4π,4π]`; the trig
fast path covers `[-16,16]`.

| Candidate | Optimized float domain | Observed extended-sweep maximum |
| --- | --- | --- |
| `sin`, `cos`, `sincos` | `[-16,16]` | 1 ULP in scalar/SIMD sweeps including π/2 neighbors |
| `exp` | `[-87,88]` | 2 ULP |
| `exp2` | `[-126,127]` | 2 ULP |
| `log`, `log2` | All positive normal floats | 4 ULP strict, 5 ULP fast math |
| `erf` | `[-9,9]` | 2 ULP |
| `erfc`, `erfcexp` complement | `[-9,9]` | 6 ULP |
| `erfcexp` Gaussian | `[-9,9]` | 2 ULP |
| `sqrt`, `rsqrt` | Positive normal floats, with edge fallbacks | 0 and 2 ULP respectively |

These are observed maxima, not exhaustive guarantees across every domain.
The reproducible CTest suite uses 65,536 random inputs per family, boundary
neighbors, mixed SIMD lanes, powers of two, normal exponent extremes, and
FMA cancellation cases. It enforces an 8-ULP limit against a separately
compiled long-double oracle. Square-root signed-zero and nonfinite cases
are checked explicitly. Float Gaussian references use the exact product of
the represented input in long double.

The broader development sweeps included ten million scalar and SIMD trig
inputs, all float encodings in `[0.5,2)` for logarithms, all encodings in
`[1,4)` for roots, and two million mixed SIMD error-function inputs.
Selected logarithm extrema were also checked against MPFR at 192-bit precision.
The raw CTest logs in the results directory are the reproducible regression
record; the extended development sweeps are additional evidence.

Outside optimized domains, candidates use platform fallbacks. No
all-input ULP promise is made for NaNs, subnormal handling under fast-math,
or the platform fallbacks. Double overloads use platform math rather than
the float polynomials.

### Why the square correction matters

With `square = x*x`, `expf(-square)` evaluates the exponential of a rounded
argument. Its error relative to mathematical `exp(-x²)` grows in the tail
even when `expf` itself is accurately rounded. The correction is:

```cpp
float const square = x * x;
float const tail = std::fma(-x, x, square);
float const gaussian = std::exp(-square);
float const corrected = std::fma(tail, gaussian, gaussian);
```

The custom pair retains `tail` during exponential range reduction and reuses
the Gaussian for its erfc result. Ordinary CUDA fast `expf` introduces a
separate error; correcting the square alone does not correct that intrinsic.
The benchmark therefore also measures the accurate libdevice exponential.
The `pow` candidate likewise uses the
[documented `__nv_powf` entry point](https://docs.nvidia.com/cuda/libdevice-users-guide/libdevice_users_guide/function_device__functions__decls_8h_1a78184906ace5dea11025b3f3d2f594cf.html)
on CUDA: ordinary `powf` under fast-math exceeded the test tolerance.

### FloatFloat and possible DoubleDouble intermediates

The existing `cosmictiger/math/float_float.hpp` supplies scalar CPU/CUDA
`FloatFloat` arithmetic. Its `twoProduct` primitive stores a rounded product
and an FMA residual. The Gaussian correction above uses the same construction:
`square` is the high word and `-tail` is the low word of `x*x`, when the
residual is representable. Keeping those words separate preserves information
that would disappear if they were added back into one float before `exp`.
The custom exponential carries the tail through argument reduction; the
library path uses a first-order correction whose omitted terms are small
enough in the tested interval.

Consequently, this accuracy issue already uses a specialized form of
float-float arithmetic, including SIMD lanes. It does not require evaluating
the entire polynomial with `FloatFloat`. The existing class is scalar, so a
general SIMD version would need a pair of SIMD packs and compatible rounding
primitives. Full `FloatFloat` evaluation has not been benchmarked here; it
remains an option for reductions or cancellation that fail the float target.
Its convenience `fma` deliberately means `a*b+c`; the residual construction
requires the genuinely fused base-type operation.

A `DoubleDouble` built from the same high/low model is a reasonable option
if custom double transcendental evaluation needs more intermediate precision.
It would need double coefficients split into high/low words, appropriate
argument reduction, and its own accuracy/performance checks. Wider arithmetic
alone does not remove polynomial approximation error. The present custom
polynomials and measured ULP results target float; double overloads delegate
to platform math, with the square residual also corrected for `erfcexp` in
`|x| <= 9`. No DoubleDouble implementation is needed for these float results.

## Timing method and limits

CPU: 131,072 values, 15 samples per path, median nanoseconds per value.
All paths use the same input arrays and output checksums; measurements are
interleaved. Explicit SIMD numbers are **per lane**. The scalar-source loops
may auto-vectorize, especially under `-ffast-math`; those timings are array
throughput, not the latency of one scalar function call.

CUDA: 262,144 values, 15 samples, 32 kernel launches per event bracket;
16 blocks of 128 threads. At most 16 of the 82 SMs can carry these blocks.
Inputs stay on the GPU, and output copies/checksums occur outside timing.
Min/median/max are saved because timing spread is material. These measurements
describe a resource-limited microbenchmark, not full-GPU saturation or an
Ewald/application speedup. Overlapping timing ranges do not establish a
reliable winner.

The GCC 13 fast-math SIMD paired-library expression is an instructive special
case: disassembly shows eight scalar `expf` and eight scalar `erfcf` calls per
pack, while the double-Gaussian form uses vector math-library entry points.
The FMA wrapper already uses the project's always-inline helper. This is a
compiler vectorization difference, not an inherent cost of doing two FMAs.

The optional CUDA register probe performs 8,192 dependent evaluations per
thread in each of 15 trials. A result-bit-derived factor in `[0.999,1]`
perturbs the original seed, preserving the input domain while preventing
loop hoisting. Subsequent inputs can differ slightly between paths because
their rounded results differ. For each trial, the reported sample is the
minimum across warps of the maximum lane clock delta, divided by the
iteration count. This reduces interference from delayed or preempted blocks;
it is a best-warp calculation probe, not an average of all threads.
Inline assembly timestamps have explicit register dependencies. Raw cycle
files include min/median/max; the common loop overhead is not subtracted.

Estrin evaluation shortens polynomial dependency chains. An isolated SIMD
polynomial probe was about 3–4× faster than Horner, but that is not a
lane-width speedup for the complete function. Range reduction, masks,
library vectorization, and memory traffic still determine end-to-end timing.

## Reproduce

Configure the project with its usual HPX and numerical-library dependencies
and `BUILD_TESTING=ON`. For the measurements here, the build directory is
`build/special_functions_validation`.

```sh
cmake --build build/special_functions_validation --parallel 4 --target \
  special_functions_baseline_test special_functions_native_test \
  special_functions_fast_math_test special_functions_cuda_standard_test \
  special_functions_cuda_fast_math_test special_functions_standard_benchmark \
  special_functions_fast_math_benchmark special_functions_cuda_standard_benchmark \
  special_functions_cuda_fast_math_benchmark
ctest --test-dir build/special_functions_validation -V -R '^special_functions_'
build/special_functions_validation/special_functions_standard_benchmark 131072 15
build/special_functions_validation/special_functions_fast_math_benchmark 131072 15
build/special_functions_validation/special_functions_cuda_standard_benchmark 262144 15 16 32
build/special_functions_validation/special_functions_cuda_fast_math_benchmark 262144 15 16 32
# Optional register/cycle mode: fifth argument is iterations per thread.
build/special_functions_validation/special_functions_cuda_standard_benchmark 262144 15 16 32 8192
build/special_functions_validation/special_functions_cuda_fast_math_benchmark 262144 15 16 32 8192
```

Benchmarks are excluded from the default build and are not CTest timing
gates. The reference translation unit always uses `-fno-fast-math
-ffp-contract=off`. CUDA accuracy tests launch one block and return skip
code 77 only when the GPU is unavailable.
