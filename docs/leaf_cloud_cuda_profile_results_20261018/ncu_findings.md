# Nsight Compute kernel comparison

These reports capture the second `cudaKickKernel` launch: the force-only adaptive
fixture after one warmup. The executables are the original before/after benchmark
binaries. Nsight Compute replays kernels to collect counters; its elapsed time
does not reproduce the benchmark's original memory residency and timing conditions.

| Metric | Previous | Modified | Change |
|---|---:|---:|---:|
| Replayed kernel duration | 3.985152 ms | 3.980256 ms | -0.12% |
| Hardware executed warp instructions | 239,279,477 | 236,666,527 | -1.09% |
| Instrumented SASS warp instructions | 238,777,936 | 232,644,210 | -2.57% |
| Predicated-on thread instructions | 3,997,484,199 | 3,902,701,829 | -2.37% |
| Register-spill instructions | 7,315,829 | 7,315,829 | 0% |
| Theoretical local-memory sectors | 28,849,953 | 29,221,958 | +1.29% |
| Theoretical global-memory sectors | 22,874,603 | 22,702,421 | -0.75% |
| Registers per thread | 255 | 255 | unchanged |
| Stack size | 1,680 | 1,680 | unchanged |
| Theoretical occupancy | 12.50% | 12.50% | unchanged |
| Achieved occupancy | 8.79% | 8.76% | -0.03 percentage points |
| Average active threads per warp | 17.25 | 17.03 | -1.28% |

## Device-function attribution

The source CSV repeats each SASS address in multiple inline source contexts.
`analyze_ncu.py` removes those duplicates, verifies identical repeated metrics,
and checks the resulting instruction totals against the raw SASS counters.
Hardware counters and instrumented SASS counters come from separate replay
passes and should not be mixed when summing attributed instructions.

| Function | Previous warp instructions | Modified warp instructions | Difference |
|---|---:|---:|---:|
| PC, `cudaGravityPcDirect` | 2,622,684 | 4,217,217 | +1,594,533 |
| CP, `cudaGravityCpDirect` | 2,901,789 | 3,886,445 | +984,656 |
| PP, `cudaGravityPpDirect` | 35,437,863 | 29,534,872 | -5,902,991 |
| CC direct | 3,248,510 | 3,182,189 | -66,321 |
| CC Ewald | 125,981,625 | 125,981,625 | 0 |
| Walk, `cudaKickKernel` excluding called functions | 7,750,065 | 7,625,295 | -124,770 |

The extra PC/CP arithmetic is more than offset by reduced PP arithmetic in this
capture. CP local-memory sectors increase by 379,703 (+62.76%), accounting for
the total local-memory increase of 372,005 sectors. This is extra stack/local
traffic in CP, but the whole-kernel change is only +1.29%; Ewald's 24,316,346
local sectors are identical. There is no increased register-spill instruction
count or changed register/shared-memory occupancy limit.

PC/CP sampled warp observations rise from 2,052 to 5,333, while PP observations
fall from 25,137 to 20,174. Overall samples are 217,485 versus 217,927 (+0.20%),
and not-issued samples are 170,156 versus 169,666 (-0.29%). Sample counts are
statistical, not exact elapsed-time attribution. There is no matching 6%
whole-kernel compute/stall growth under replay conditions.

The major sampled costs in both variants are Ewald evaluation (approximately
35% of samples) and device allocation/release (approximately 37% combined).
Long-scoreboard stalls account for approximately 41% of sampled warp states in
both versions. This identifies existing bottlenecks, but does not establish
either as the cause of the measured application regression.

The Nsight Systems timeline and managed-memory activity must determine where
the extra application time resides. In particular, replay can mask memory
migration effects and cannot be used to dismiss a cold/residency-sensitive
kernel slowdown seen in the ordinary run.

Full reproducible output: `ncu_analysis.json` and `ncu_function_comparison.csv`.
Source exports are retained as `ncu_{before,after}_source.csv.gz`; the analyzer
reads these transparently. `ncu_compression_manifest.json` records the original
and verified decompressed SHA-256 hashes. Full `.ncu-rep` captures remain in the
build-directory profiling run folders.
