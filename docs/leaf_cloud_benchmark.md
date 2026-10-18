# Leaf–cloud PC/CP benchmark, 2026-10-18

The requested changes improve the adaptive workload's CPU gravity kick by 1.8–2.1%, while increasing its CUDA kick time by 6.0–6.4%. All five paired runs agree on the direction for both backends. Including tree construction gives a 1.9–2.1% CPU improvement and a 5.5–5.6% CUDA regression.

The comparison is between commit `c6bf4a8ad7de47b456dabf028f3b852a42846218` and the six-file production/test patch recorded in `leaf_cloud_benchmark_results_20261018/changes.patch`: removal of all particle-count decisions from PC/CP selection, plus leaf–cloud PC/CP fallback after CC rejection. No production source was changed during benchmarking.

## Method

- Ryzen 9 3900X, two HPX worker threads pinned to separate physical cores 1 and 2; one locality. RTX 3090 for CUDA. Builds and performance runs were sequential; no remote machine was used.
- Release GCC 13.3, CUDA 12.8, architecture 86, same HPX installation and compiler flags for both versions. Common application objects were shared; every application consumer of the changed acceptance header was rebuilt from an archived baseline source tree, then CUDA device linking and final linking were repeated. Object, source, binary and command hashes are recorded in the build manifests.
- 32,768 particles, bucket size 64, opening angle 0.8, Ferrers degree 4; physical compact support radius 0.0008798383391969691. All particles active. Periodic tree creation and full gravity kick measured separately; reset and CSV export excluded. Both acceleration-only and acceleration-plus-potential modes measured.
- Five sequential before/after pairs for each backend, alternating AB/BA order. Each process ran one warmup and 15 measured samples per case. Reported values are the median of five process medians. The paired ranges below are observed ranges, not confidence intervals. No timing sample was discarded from the completed suite.
- Clocks were not locked, and the desktop shared the GPU. Small control-workload differences should not be treated as portable speedups. Full runtime metadata and raw per-sample measurements are retained.

The original lattice and eight equal clusters are controls: every occupied leaf has 64 particles and the same depth within its fixture. The added adaptive fixture retains 8,192 particles throughout the box and compresses 24,576 into a cube of side 0.18 centered at (0.41, 0.47, 0.53), using a deterministic index permutation. Its varying leaf depths and small leaves exercise the newly enabled decisions. Both versions use exactly the same harness object and positions. Untimed leaf histograms agree across every run.

## Results

Timings are medians of five independent process medians, each containing 15 measured samples after one warmup.
Tree + kick is the recorded median of each sample's combined tree and kick time. Negative time change means faster; speedup is before / after.

| Backend | Fixture | Potential | Kick before (ms) | Kick after (ms) | Change | Speedup | Paired speedup range | Tree + kick before (ms) | Tree + kick after (ms) | Change |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | adaptive | 0 | 123.741 | 121.470 | -1.84% | 1.0187x | 1.0102–1.0212x | 126.358 | 124.004 | -1.86% |
| cpu | adaptive | 1 | 125.735 | 123.034 | -2.15% | 1.0220x | 1.0086–1.0261x | 128.331 | 125.617 | -2.11% |
| cpu | clustered | 0 | 19.040 | 19.121 | +0.42% | 0.9958x | 0.9875–1.0002x | 21.685 | 21.781 | +0.44% |
| cpu | clustered | 1 | 20.435 | 20.471 | +0.18% | 0.9982x | 0.9911–1.0013x | 23.054 | 23.067 | +0.06% |
| cpu | lattice | 0 | 108.108 | 106.950 | -1.07% | 1.0108x | 1.0051–1.0147x | 110.032 | 108.827 | -1.10% |
| cpu | lattice | 1 | 109.504 | 108.443 | -0.97% | 1.0098x | 1.0051–1.0127x | 111.357 | 110.369 | -0.89% |
| cuda | adaptive | 0 | 16.290 | 17.338 | +6.44% | 0.9395x | 0.9247–0.9482x | 19.295 | 20.380 | +5.63% |
| cuda | adaptive | 1 | 16.476 | 17.463 | +5.99% | 0.9435x | 0.9328–0.9572x | 19.502 | 20.578 | +5.52% |
| cuda | clustered | 0 | 6.860 | 7.008 | +2.16% | 0.9789x | 0.9553–1.0197x | 9.863 | 10.024 | +1.64% |
| cuda | clustered | 1 | 7.050 | 7.002 | -0.68% | 1.0068x | 0.9887–1.0154x | 9.938 | 9.891 | -0.47% |
| cuda | lattice | 0 | 31.266 | 31.206 | -0.19% | 1.0019x | 0.9949–1.0158x | 33.445 | 33.350 | -0.28% |
| cuda | lattice | 1 | 31.347 | 31.050 | -0.95% | 1.0096x | 1.0052–1.0103x | 33.494 | 33.237 | -0.77% |

| Fixture | Occupied leaves | Depth range | Particles/leaf range | Leaves with <38 particles |
|---|---:|---:|---:|---:|
| adaptive | 1082 | 9–17 | 1–64 | 713 (65.9%) |
| clustered | 512 | 12–12 | 64–64 | 0 (0.0%) |
| lattice | 512 | 9–9 | 64–64 | 0 (0.0%) |

Force comparisons use the final potential-enabled sample. Differences from the previous version are not errors against an exact-force reference.

| Comparison | Fixture | Acceleration relative L2 | Max acceleration vector difference | Max particle-relative acceleration difference | Potential relative L2 | Max potential difference |
|---|---|---:|---:|---:|---:|---:|
| cpu before/after | adaptive | 0.00149977 | 23466.7 | 0.0600881 | 2.94057e-05 | 258.031 |
| cpu before/after | clustered | 0 | 0 | 0 | 0 | 0 |
| cpu before/after | lattice | 0 | 0 | 0 | 0 | 0 |
| cuda before/after | adaptive | 0.00149977 | 23467 | 0.0600884 | 2.94045e-05 | 258.07 |
| cuda before/after | clustered | 0 | 0 | 0 | 0 | 0 |
| cuda before/after | lattice | 0 | 0 | 0 | 0 | 0 |
| before CPU/CUDA | adaptive | 1.96579e-07 | 1.37642 | 2.98105e-06 | 1.86837e-07 | 0.140625 |
| before CPU/CUDA | clustered | 2.12944e-07 | 0.0943028 | 4.2082e-06 | 2.97824e-07 | 0.0175781 |
| before CPU/CUDA | lattice | 9.12554e-06 | 0.0247917 | 0.00041779 | 3.44915e-05 | 0.013237 |
| after CPU/CUDA | adaptive | 1.92201e-07 | 1.53761 | 2.96612e-06 | 1.84083e-07 | 0.15625 |
| after CPU/CUDA | clustered | 2.12944e-07 | 0.0943028 | 4.2082e-06 | 2.97824e-07 | 0.0175781 |
| after CPU/CUDA | lattice | 9.12554e-06 | 0.0247917 | 0.00041779 | 3.44915e-05 | 0.013237 |

cpu after: 5 repeat exports are byte-identical.
cpu before: 5 repeat exports are byte-identical.
cuda after: 5 repeat exports are byte-identical.
cuda before: 5 repeat exports are byte-identical.

## Numerical interpretation

All final force/potential exports are finite, coordinate aligned, and byte-identical across the five repeats within each backend/version. The controls are exactly unchanged between versions. The adaptive forces change by 0.150% in global relative L2, with a maximum per-particle relative vector change of 6.01%; potential changes by about 0.00294% in relative L2. These are differences between two approximations, not errors against a direct solution. No direct-reference accuracy benchmark was performed. The modified CPU and CUDA adaptive accelerations agree to 1.92e-7 relative L2.

An independent untimed Nsight Systems trace confirms four actual `cudaKickKernel` launches for adaptive acceleration-only and potential-enabled warmup/sample runs. That trace is excluded from performance statistics. Both versions report six resident blocks per SM, 15,184 shared-memory bytes, and 255 registers for the kick kernel.

## Interrupted initial suite and benchmark isolation

The initial regular-fixture suite completed three CPU pairs, then the modified binary crashed. GDB reproduced the failure at `selfPtr->localRoot`, before interaction selection, with `self.index == -1`. A baseline diagnostic process completed 100 samples per case without crashing; the failure was not reproduced in that process.

Source inspection identified a task-lifetime race in completion logic shared by both versions: the last nonempty leaf can publish particle completion while trailing empty-node descents or ancestor tasks are still active; local child futures are discarded without joining. Rebuilding the tree can then overwrite nodes those tasks still use. The benchmark-only harness now stops the ordinary HPX memory-monitor task once, and waits for HPX pool quiescence after every kick, before recording its elapsed time and before reusing the tree. This identical synchronization is present in both timed binaries. It does not modify production traversal or repair its completion contract.

All 20 processes in the completed three-fixture suite succeeded. The initial partial results and debugger logs are retained in the parent results directory, but are excluded from the reported timings. The production completion race remains a separate issue.

## Reproduction and evidence

The saved `run_comparison.py` contains the exact environment, process ordering and commands. Each process uses:

```sh
taskset -c 1,2 /absolute/path/to/cosmictiger_before_or_after \
  --run.test=stfBenchmark --initial.particlesPerDimension=32 \
  --hydro.enabled=false --gravity.openingAngle=0.8 \
  --gravity.effectiveRadius=0.01876988456953534 \
  --gravity.kernelFamily=ferrers --gravity.kernelDegree=4 \
  --hpx:threads=2 --hpx:bind=none --analysis.views=false \
  --analysis.power=false --analysis.groups=false --lightcone.enabled=false \
  --compute.cuda=false
```

Set `COSMICTIGER_STF_WALK_SAMPLES=15`, `OMP_NUM_THREADS=2`, `OMP_THREAD_LIMIT=2`; use `--compute.cuda=true` for CUDA. The public radius option is in initial particle-spacing units; the physical support value above comes from the benchmark configuration record. Run each process in a separate output directory.

- [Full table and numerical comparison](leaf_cloud_benchmark_results_20261018/adaptive_suite/summary.md)
- [All measured runs, commands and samples](leaf_cloud_benchmark_results_20261018/adaptive_suite/runs.json)
- [Adaptive/common-harness build manifest](leaf_cloud_benchmark_results_20261018/adaptive_suite/build_manifest.json)
- [Original baseline build manifest](leaf_cloud_benchmark_results_20261018/build_manifest.json)
- [Exact compared production patch](leaf_cloud_benchmark_results_20261018/changes.patch)
- [Common benchmark-only harness changes](leaf_cloud_benchmark_results_20261018/adaptive_suite/harness_changes.patch)
- [Runtime metadata](leaf_cloud_benchmark_results_20261018/adaptive_suite/runtime_metadata.json)
- [CUDA execution trace summary](leaf_cloud_benchmark_results_20261018/adaptive_suite/cuda_kernel_summary.txt)

The committed evidence includes timing logs, numerical comparison summaries, reproduction scripts, and a manifest of representative force-export hashes. The four compressed force exports remain local in the results directory and are excluded from Git. Build binaries, baseline source snapshot, and the full Nsight report remain under `build/leaf_cloud_benchmark_20261018/`.
