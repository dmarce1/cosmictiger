# Leaf–cloud CUDA regression profile

The slowdown comes mainly from additional CPU work above the CUDA handoff. The leaf–cloud extension enables internal-node CP operations while those larger sink nodes are still being processed by the host. The resulting delay before GPU launch accounts for about 82–91% of the measured regression. Removing the particle-count heuristic alone does not reproduce the slowdown.

Production sources were not modified during this investigation. All instrumentation and intermediate variants live under `build/leaf_cloud_benchmark_20261018/cuda_regression_profile/`.

## Whole-kick attribution

Three alternating before/after Nsight Systems pairs used the original comparison binaries. Each process ran the adaptive fixture, one warmup plus 15 samples in each potential mode. The table reports arithmetic means over 45 measured samples per version and mode. Profiler overhead makes these absolute times different from the original unprofiled benchmark; use the original benchmark for headline speed measurements.

| Stage | Forces before | Forces after | Change | Forces + potential before | Forces + potential after | Change |
|---|---:|---:|---:|---:|---:|---:|
| Whole kick | 17.050 ms | 18.289 ms | +1.239 ms | 17.021 ms | 18.142 ms | +1.121 ms |
| `cudaKickKernel` | 5.256 ms | 5.485 ms | +0.229 ms | 5.368 ms | 5.468 ms | +0.099 ms |
| Outside kernel | 11.794 ms | 12.804 ms | **+1.010 ms** | 11.652 ms | 12.674 ms | **+1.022 ms** |

Outside-kernel time explains 81.5% and 91.1% of the increase, respectively. This subtraction is not a sum of CUDA API durations: the post-launch `cudaMemcpyAsync` blocks while the GPU kernel runs, so its duration overlaps the kernel.

Unified-memory migration volume, transfer duration and migration span are essentially unchanged: roughly 2.39 MB HtoD and 1.99 MB DtoH per sample. Explicit HtoD payload shrinks from 364,260 to 349,480 bytes. Launch and synchronization differences are on the order of microseconds.

## Which host work increased

Identical NVTX annotations were added to isolated host-source copies. Both binaries reuse their original CUDA object byte-for-byte, along with the same synchronized adaptive benchmark harness. Two alternating profile pairs provided 30 measured samples per version and potential mode. All final forces match the corresponding uninstrumented exports byte-for-byte.

The added internal CP work is directly visible:

- **349 nonempty `cpuGravityCp` calls per kick**, versus none before, processing **1,724 source-leaf list entries**. Their summed elapsed time is approximately **1.94–2.00 ms** per kick.
- CPU direct-checklist classification rises from approximately **1.02 ms to 1.60 ms**, even though initial checklist entries fall from **33,491 to 31,686**.
- Time before kernel launch grows by **1.141 ms** for forces and **1.180 ms** with potential. CPU Ewald work and GPU preparation remain essentially unchanged. Work-list insertion becomes slightly cheaper.

These stage times are summed across two HPX workers and overlap other work; they must not be added directly to whole-kick wall time. In particular, internal CP can execute concurrently with Ewald work on the other worker. The added CP and classification work lengthen the host critical path before the GPU is launched.

The mechanism follows the current handoff rule in `src/gravity/kick.cpp`: a sink subtree is sent to CUDA only when its particle count is at most `min(16384, max(current_range_particles / kickBlockCount(), bucketSize))`, subject to the locality gates. The numerator is the local length of `particlesCurrentRange()`, which returns `[rungBegin, particlesSize())`; it is not a separate count of particles passing the kick's active-rung test. The hierarchical integrator can narrow this range, whereas the ordinary SPH path retains the full source range and filters active sinks by rung. This benchmark bypasses time evolution and resets the range to all **32,768 particles** before every kick. Here `kickBlockCount()` is 492 and the bucket size is 64, making the cutoff **66 particles**. Larger internal sinks continue through CPU acceptance and gravity operators. The new `cpuGravityCp` call before child recursion consequently evaluates accepted source-leaf contributions on the CPU before the remaining work reaches CUDA.

`cpuGravityCp` gathers source positions and types into five temporary vectors per chunk, evaluates the P2L expansion, and reduces its coefficients. The profile attributes its combined packing/allocation/arithmetic cost; it does not separate those subcomponents.

The broader CPU classification condition also computes box distances for batches containing source leaves without first requiring a CC rejection. The CUDA counterpart explicitly checks CC rejection. This is an identifiable source of avoidable host acceptance work, though this profile does not assign an exact fraction of the classification increase to already-accepted pairs.

## Isolating the two requested changes

A third executable uses the new count-free selector but retains the old leaf–leaf-only execution paths. Its CPU and CUDA kick source copies differ from baseline only in selector arguments; all remaining objects and the adaptive harness are shared. Three rounds rotate execution order among all three binaries, with 15 samples plus one warmup per case.

| Variant | Forces kick | Forces + potential kick |
|---|---:|---:|
| Previous version | 16.474 ms | 16.777 ms |
| Remove count heuristic, retain leaf–leaf only | 16.452 ms | 16.698 ms |
| Full leaf–cloud extension | 17.478 ms | 17.613 ms |

Values are medians of the three process medians. The count-free leaf–leaf variant stays close to baseline. The slowdown appears with leaf–cloud execution, consistent with the host-stage measurements.

## GPU instruction evidence

Nsight Compute profiled the first measured force-only kernel after warmup. Replay timings are diagnostic and are not substitutes for ordinary benchmark timings or managed-memory behavior.

| Source-attributed SASS instructions | Previous | Modified |
|---|---:|---:|
| PC | 2.623 million | 4.217 million |
| CP | 2.902 million | 3.886 million |
| PP | 35.438 million | 29.535 million |

The PP savings exceed the extra PC/CP instructions. Overall source-attributed instructions fall 2.57%. Both versions use 255 registers per thread, six blocks per SM, the same stack size, and the same 7,315,829 spill instructions. CP local-memory traffic increases, but total local traffic grows only 1.29%. No occupancy or broad instruction-count regression explains the whole-kick slowdown.

Ordinary Systems traces still show a smaller kernel-time increase of about 0.10–0.23 ms. The hardware-counter replay does not establish a unique cause for that residual; it should not be mistaken for the much larger host delay.

## Optimization targets

The primary target is the internal CP work executed before CUDA handoff: improve its host packing/evaluation cost or move that work into the GPU workload. A separate, smaller opportunity is to restrict CPU box-distance fallback to CC-rejected, hybrid-eligible pairs. Restoring the particle-count heuristic is not supported as a remedy by the isolated comparison.

## Evidence and reproduction

- [Original benchmark](leaf_cloud_benchmark.md)
- [Nsight Systems timing and migration summary](leaf_cloud_cuda_profile_results_20261018/nsys_summary.md)
- [Systems profile commands and binary hashes](leaf_cloud_cuda_profile_results_20261018/profile_manifest.json)
- [Host-stage timing, counts and overlap analysis](leaf_cloud_cuda_profile_results_20261018/nvtx_summary.md)
- [Host NVTX profile commands and force hashes](leaf_cloud_cuda_profile_results_20261018/host_profile_manifest.json)
- [Host instrumentation patch](leaf_cloud_cuda_profile_results_20261018/host_ranges_source_changes.patch)
- [Isolated variant measurements](leaf_cloud_cuda_profile_results_20261018/ablation_runs.json)
- [Isolated variant build provenance](leaf_cloud_cuda_profile_results_20261018/leaf_only_build_manifest.json)
- [Nsight Compute findings](leaf_cloud_cuda_profile_results_20261018/ncu_findings.md)
- [Function instruction comparison](leaf_cloud_cuda_profile_results_20261018/ncu_function_comparison.csv)

The committed results include compact per-sample CSVs, aggregate timing and transfer analyses, profiling scripts, raw metrics, and compressed source-attribution exports. The ten `host_pair*_analysis.json` and `nsys_pair*_analysis.json` event exports remain local in the results directory and are excluded from Git; the analysis scripts regenerate them from the full `.nsys-rep` and `.sqlite` captures under the build directory. Full `.ncu-rep` captures also remain under the build directory. CPU workers stayed pinned to cores 1 and 2; GPU profiles and builds ran sequentially. No remote access was used.
