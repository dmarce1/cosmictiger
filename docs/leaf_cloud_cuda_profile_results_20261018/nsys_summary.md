# CUDA trace comparison

Arithmetic means across 3 pairs x 15 post-warmup samples per phi and variant. Means preserve additive kick=kernel+residual decomposition. Median and median-of-run-medians also included.

Three alternating before/after pairs; each launch aligned one-to-one with benchmark stdout. Warmups excluded.

| Mode | Component | Before (ms) | After (ms) | Change (ms) | Change (%) |
|---|---|---:|---:|---:|---:|
| Force | Whole kick | 17.0501 | 18.2889 | +1.2389 | +7.27% |
| Force | CUDA kernel | 5.2559 | 5.4850 | +0.2291 | +4.36% |
| Force | Outside kernel | 11.7942 | 12.8039 | +1.0098 | +8.56% |
| Force + potential | Whole kick | 17.0206 | 18.1418 | +1.1212 | +6.59% |
| Force + potential | CUDA kernel | 5.3681 | 5.4676 | +0.0995 | +1.85% |
| Force + potential | Outside kernel | 11.6524 | 12.6742 | +1.0218 | +8.77% |

The nonkernel residual accounts for 81.5% and 91.1% of the mean regression, respectively.

| Mode | Diagnostic | Before | After |
|---|---|---:|---:|
| Force | Launch to kernel start (ms) | 0.1544 | 0.1558 |
| Force | Kernel end to sync return (ms) | 0.0201 | 0.0228 |
| Force | UVM HtoD bytes during kernel | 2398709 | 2397252 |
| Force | UVM HtoD transfer sum (ms) | 0.2658 | 0.2645 |
| Force | UVM DtoH bytes until next launch | 1988016 | 1986560 |
| Force | UVM DtoH transfer sum (ms) | 0.1574 | 0.1539 |
| Force | UVM DtoH migration span (ms) | 4.5617 | 4.5882 |
| Force | Explicit HtoD payload bytes | 364260 | 349480 |
| Force | Prelaunch HtoD API time (ms) | 0.0583 | 0.0528 |
| Force + potential | Launch to kernel start (ms) | 0.1261 | 0.1341 |
| Force + potential | Kernel end to sync return (ms) | 0.0193 | 0.0208 |
| Force + potential | UVM HtoD bytes during kernel | 2388514 | 2384145 |
| Force + potential | UVM HtoD transfer sum (ms) | 0.2656 | 0.2592 |
| Force + potential | UVM DtoH bytes until next launch | 1991241 | 1986560 |
| Force + potential | UVM DtoH transfer sum (ms) | 0.1576 | 0.1579 |
| Force + potential | UVM DtoH migration span (ms) | 4.5052 | 4.5151 |
| Force + potential | Explicit HtoD payload bytes | 364260 | 349480 |
| Force + potential | Prelaunch HtoD API time (ms) | 0.0465 | 0.0525 |

UVM migration volume, transfer time, and post-kernel span are essentially unchanged. Launch and synchronization latencies change by only microseconds. These traces localize most extra time outside the GPU kernel, but cannot alone identify a particular CPU routine.

Caveats:

- Nsight profiling perturbs absolute timing; use original unprofiled benchmark for headline performance.
- Nonkernel residual is measured wall kick minus the CUDA kernel; it includes host preparation, CPU computations, memory-fault handling, and synchronization.
- Launch-window UVM metrics include inter-sample reset/tree work and cannot locate CPU fault time specifically inside the kick timer.
- Final sample lacks a following launch, so post-kernel migration metrics cover 42 instead of45 samples for phi1.
- Summed transfer durations are not page-fault service durations; API durations overlap kernel execution.
