# Geometric FMM measurement results, 2026-11-14

The exported forest reduced serialized application traffic by 49–55% in these two
CPU cases. It did not demonstrate a general speedup: reconstructed path cost was
1.3% lower for the small case and 1.1% higher for the clustered case. Three repeats
on a workstation with War Thunder ON do not establish statistical significance,
production evolution performance or network scaling. No solver optimization or
transport redesign was applied on the strength of these results.

## Identity, configuration and resources

Baseline: complete-source demand traversal and exported source-forest traversal
of the **same frozen geometric trees**, over main
`b18af4112f6892ead43f5377fda115d244b80da9` plus preserved uncommitted steps 2–6
and the new measurement support. This is not an old-tree/new-tree comparison.
The measured executable SHA-256 was
`cd22d32e0ce81effd6ca7acd0fc380046fbb71ce48a75651c1d0346684bfb015`.

Actual runs used two local MPI ranks, two HPX workers per rank (four total), CPU
only, Release GCC 15, native SIMD, build order 7, Fixed32 positions, bucket size
128, all sinks at rung zero, equal mass with `gm=1/N`, and no velocity update.
The CPU was an AMD Ryzen 9 3900X; both ranks ran on the same workstation with
binding disabled. Each case used three paired repeats, alternating reference,
export, reference as the first path. No warmup was dropped. Nested OpenMP/BLAS
worker counts were set to one. There was no GPU use or QB login. After the user
reported War Thunder ON, task builds and runs remained sequential and capped at
four CPU workers, within the shared six-core ceiling.

| Case | Particles and distribution | Theta | Normalized softening | Native position payload |
|---|---|---:|---:|---:|
| Small | 96 seeded uniform points | 0.4 | 0.00436790232 | 12 bytes/particle |
| Clustered | 32³ points in two Gaussian halos, centers (0.25,0.25,0.25)/(0.75,0.75,0.75), sigma 0.035 | 0.5 | 0.000625 | 12 bytes/particle |

The runner uses Python seed 731924 and saves the exact particle text files.
The clustered case represents this authorized static workload, not a production
cosmology snapshot. Ownership was decomposed once before each case and retained
through every pair. Clustered owner counts were 16,383 and 16,385.

## Joined coordinator times

All values below are means of three repeats, in milliseconds.

| Case | Load | Ownership/migration | Local build | Exchange | Reference force | Export force | Shared cleanup | Actual pair wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Small | 0.475 | 3.933 | 1.896 | 2.286 | 143.225 | 139.103 | 0.335 | 462.595 |
| Clustered | 57.912 | 5.394 | 9.021 | 4.813 | 256.483 | 254.618 | 0.516 | 527.692 |

Load and ownership are one-time setup intervals, not per-repeat costs. Force
includes preflight and joined traversal completion. Exchange includes selection,
packing, transfer, import installation and readiness. Pair wall additionally
includes cold-cache resets, metric collection and accuracy diagnostics.

| Case | Reference reconstructed path | Export reconstructed path | Export/reference | Reference path range | Export path range |
|---|---:|---:|---:|---:|---:|
| Small | 145.456 ms | 143.620 ms | 0.9874 | 144.557–146.215 ms | 143.337–144.003 ms |
| Clustered | 266.021 ms | 268.968 ms | 1.0111 | 265.103–266.800 ms | 267.904–269.900 ms |

These paths add the same shared build/cleanup intervals to each walk; the export
path also adds exchange. They are not independently timed evolution-step totals.
They exclude diagnostics. See [the measurement guide](geometric_fmm_measurement.md)
for precise timer boundaries and counter limitations.

## Traffic, coverage and rank imbalance

Values are summed over both sending ranks per repeat, counting each transmitted
application object once. Demand traffic includes index requests and cache-line
responses; export traffic counts push payloads. HPX envelopes, descriptor/control
messages, acknowledgements, MPI framing and measurement RPCs are excluded.

| Case | Reference demand bytes / messages | Export bytes / pushes | Reduction in payload bytes | Export nodes / positions | Used/imported nodes | Coarse represented sources / all sources | Fine-detail peers per destination | Fallbacks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Small | 852,712 / 8 | 433,814 / 2 | 49.1% | 1,120 / 96 | 1,022 / 1,120 | 0 / 96 | 1 | 0 |
| Clustered | 1,027,530 / 10 | 458,706 / 2 | 55.4% | 678 / 16,388 | 330 / 678 | 16,380 / 32,768 | 1 | 0 |

The clustered export was asymmetric: rank 0 sent 403,277 bytes and 16,383 particle
positions; rank 1 sent 55,429 bytes and five positions plus 22 coarse terminals.
Imported leaves/terminals represented every remote owner's source count, and
sent/received bytes, messages and exported/imported record counts matched in both
directions. Demand caches were cold before each walk. The imported walk issued
zero demand messages, separately from its zero fallback guard.

Mean traversal max-rank/mean-rank ratios were 1.043 reference / 1.033 export for
the small case and 1.251 / 1.252 for the clustered case. The exported forest did
not remove the clustered case's traversal imbalance.

The following are maxima of per-rank task elapsed intervals over three repeats;
they overlap and must not be summed into a coordinator total.

| Case | Export selection/packing | Owner exchange including wait | Exposed wait after final dispatch | Import validation/installation | Export traversal |
|---|---:|---:|---:|---:|---:|
| Small | 0.562 ms | 2.352 ms | 1.088 ms | 0.301 ms | 139.094 ms |
| Clustered | 0.627 ms | 5.520 ms | 4.068 ms | 3.610 ms | 254.811 ms |

## Memory and accuracy

The owned arena reserved 522,190,848 bytes per rank in both cases, dominated by
the existing minimum node allocation. This is logical reserved arena storage,
not resident memory. Maximum imported map storage estimates were 100,224 bytes
(small) and 906,992 bytes (clustered). Largest export vector capacities were
131,840 and 327,680 bytes; per-owner in-flight serialized payload bounds were
226,569 and 403,277 bytes. Reference demand-cache vector storage maxima were
240,224 and 414,976 bytes. Epoch remote multipole allocation was 655,360 bytes
per rank; it can include both paths' retained multipoles. Process lifetime peak
RSS maxima were 283,238,400 and 290,119,680 bytes. Allocator, validation temporaries,
HPX/MPI buffering and other application storage are not attributed to these
container estimates. No standalone per-path RSS comparison is claimed.

Every force and potential component matched between paths exactly in all six
pairs (reported relative L2 and maximum scaled differences were zero). All
velocities and rung-zero states stayed unchanged. The independent periodic
direct check for the 96-particle input had acceleration relative L2 error
`1.694163465e-4`, below the configured order/theta bound `0.4^6 = 0.004096`.
The clustered case was checked against the complete-source FMM, not an O(N²)
direct solve. These are static CPU force checks, not mixed-rung evolution or
momentum-conservation evidence.

## Supported next investigation and validation

The clustered import action's 3.610 ms maximum and 4.068 ms exposed wait are
material relative to the 1.865 ms mean traversal saving. Packing was at most
0.627 ms. Before altering transport, split import validation, size accounting
and particle-map insertion timings. A compact range-indexed representation for
particle imports is the next optimization candidate if insertion dominates;
these data do not isolate its cost yet. Only 330 of 678 imported nodes were read,
so tighter destination covers remain a separate candidate after import costs
are understood. Two pushes do not justify a new peer handshake or relaying.
No demand-fetch fallback was removed.

Validation actually performed:

- CPU Release build and focused `fmmWalk_cpu`, `fmmWalk_mpi`, `benchmark_cpu`
  checks passed (3/3; 45.35 s). The benchmark regression also exercises the new
  one-locality comparison and bounded direct check.
- The benchmark regression passed again after worker-count reporting changes
  (35.61 s). The final focused MPI benchmark regression passed, including the
  explicit global comparison count, per-locality legacy benchmark count, and
  forest node-count relation `nodes = 2*leaves - ranks`. Its first attempt exposed
  the pre-existing single-root assertion, which was corrected before the pass.
- All six approved measured pairs passed force/potential comparisons, zero
  fallback checks, coverage, peer-direction and timing-arithmetic invariants.
- Python compilation, focused C++ formatting and `git diff --check` passed.

No full suite, Fixed64 build, CUDA build/runtime, distributed-network benchmark,
extra workload sweep, commit or push was performed in this task.

Raw inputs, CSVs, command/configuration/source hashes, logs and `summary.json`
are under `build/geometric_fmm_measurement_2026_11_14/` in this checkout. The initial
CSV header called HPX's aggregate worker count `threads`; it was corrected to
`total_worker_threads` without changing values or adding repeats. Original CSVs
and `header_correction.txt` preserve that correction. In the installed HPX,
`runtime_distributed::get_num_worker_threads()` calls
`addressing_service::get_num_overall_threads()`. Legacy benchmark rows and the
startup line now use `hpx::get_os_thread_count()` for counts local to a rank.
The final executable differs from the measured binary only in these reporting
fixes. Per-file hashes in `command.json` identify the measured source exactly.
