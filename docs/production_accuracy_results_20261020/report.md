The order-7, theta=0.61 baseline was fastest in every paired comparison. Neither tested lower-order setting earns a production change.

The initial order-6/theta=0.51 and order-5/theta=0.41 candidates failed accuracy and were excluded from timing. Their sole retries, order-6/theta=0.35 and order-5/theta=0.32, passed every gate on both fixtures but were slower. This bounded test does not rule out an intermediate angle; the one-retry budget was respected and no further tuning was performed.

Protocol: commit 788c45a0245c387cf315c4b07e9d33442d7f1ee7 plus the saved benchmark-only harness.patch; Release CPU/CUDA -O3 -DNDEBUG, CUDA sm_86, RTX 3090, 8 HPX workers on CPUs 1-8, OMP_NUM_THREADS=OMP_THREAD_LIMIT=1, one GPU/locality, 262144 particles (side 64), bucket 184, Ferrers degree 4, effective radius 0.01876988456953534, unchanged default CPU/GPU handoff policy. User explicitly reported War Thunder OFF. Build used at most 6 jobs and did not overlap runs.

Accuracy: 1024 identical physical sink coordinates per fixture, selected from a deterministic permutation before tree sorting. All configurations are scored against the saved baseline direct periodic/Ewald reference. Strict gates require force and potential relative L2 and pointwise p99, plus RMS and maximum absolute force error in the baseline lowest-force 10% of sinks, to be no worse than baseline. Reference rerun disagreement must be below 1% of baseline error; this is a reproducibility gate, not an accuracy bound. Force-only and potential-enabled force checksums matched exactly in all accuracy runs.

| Order / theta | Fixture | Force rel L2 | Force rel p99 | Potential rel L2 | Potential rel p99 | Worst gate / baseline | Pass |
|---|---|---:|---:|---:|---:|---:|---|
| 7 / 0.61 | clustered | 0.000330781 | 0.00150892 | 3.94113e-05 | 0.000261979 | 1.000 | yes |
| 7 / 0.61 | adaptive | 0.000170649 | 0.00449943 | 7.06178e-06 | 0.000906056 | 1.000 | yes |
| 6 / 0.51 | clustered | 0.000466268 | 0.00143989 | 0.000126504 | 0.00154157 | 5.884 | no |
| 6 / 0.51 | adaptive | 0.000172592 | 0.00626609 | 8.6274e-06 | 0.000793811 | 1.393 | no |
| 5 / 0.41 | clustered | 0.000389845 | 0.00182011 | 9.15755e-05 | 0.00044077 | 2.324 | no |
| 5 / 0.41 | adaptive | 0.000185529 | 0.00734148 | 8.93055e-06 | 0.00148531 | 1.639 | no |
| 6 / 0.35 | clustered | 5.45712e-05 | 0.000285661 | 4.02024e-06 | 2.18034e-05 | 0.189 | yes |
| 6 / 0.35 | adaptive | 2.27184e-05 | 0.000486556 | 5.93992e-07 | 5.74816e-05 | 0.133 | yes |
| 5 / 0.32 | clustered | 0.000158674 | 0.000676673 | 2.09536e-05 | 9.95175e-05 | 0.534 | yes |
| 5 / 0.32 | adaptive | 7.1846e-05 | 0.0025374 | 4.17842e-06 | 0.000319042 | 0.592 | yes |

Maximum reference rerun disagreement / baseline error across all recorded reproducibility metrics: 1.16e-11.

Timing: exactly three rounds with the configuration order reversed in round 2 (baseline, order 6, order 5; then order 5, order 6, baseline; then the original order). Each process measured 11 samples after one warmup for force-only and force-plus-potential modes. Reference calls and force CSV exports were disabled for timings. The primary metric is tree construction plus the complete synchronized kick; initialization/reset, diagnostics, and teardown are outside that metric. Values below are medians of the three per-process medians. Runtime ratios use paired round medians; values above 1 are slower.

| Order / theta | Fixture | Potential | Tree + kick ms | Paired runtime / baseline range | Kick ms |
|---|---|---|---:|---|---:|
| 7 / 0.61 | clustered | off | 18.101 | 1.000–1.000 | 12.924 |
| 7 / 0.61 | clustered | on | 19.610 | 1.000–1.000 | 14.673 |
| 7 / 0.61 | adaptive | off | 28.007 | 1.000–1.000 | 21.429 |
| 7 / 0.61 | adaptive | on | 28.768 | 1.000–1.000 | 22.469 |
| 6 / 0.35 | clustered | off | 28.868 | 1.455–1.738 | 24.206 |
| 6 / 0.35 | clustered | on | 30.290 | 1.429–1.607 | 25.681 |
| 6 / 0.35 | adaptive | off | 88.662 | 3.073–3.208 | 81.233 |
| 6 / 0.35 | adaptive | on | 88.556 | 3.015–3.105 | 81.381 |
| 5 / 0.32 | clustered | off | 28.553 | 1.449–1.678 | 24.045 |
| 5 / 0.32 | clustered | on | 30.983 | 1.422–1.610 | 26.717 |
| 5 / 0.32 | adaptive | off | 99.702 | 3.550–3.581 | 93.313 |
| 5 / 0.32 | adaptive | on | 100.836 | 3.433–3.530 | 94.023 |

Within-process median absolute deviation / median ranged from 0.44% to 4.78%. No candidate approached the required repeatable 5% improvement; every paired comparison was slower. Raw sample distributions are retained in timing/runs.json and timing/noise.json.

Limit: these are synthetic complete force evaluations, not a cosmological snapshot or active-rung production replay. The direct reference uses float pair calculations (including an approximate erfc) with double accumulation. Absolute reference bias was not calibrated; the reported errors and pass decisions are relative to that reference, not a claim of independently verified physical accuracy at the smallest reported residual. The retry candidates are more accurate than the gate requires, so this is an equal-or-better-accuracy screen rather than an optimized equal-error frontier.

All 10 accuracy fixture/configuration cases and 18 timing processes completed successfully. The optimized build and source whitespace checks passed. Only the benchmark harness and its new driver were changed; no production default or kernel was changed.

Evidence: plan.json, build_configuration.txt, configure.log, build.log, harness.patch, per-configuration summary.json/summary.md and sampled CSVs, timing/summary.json, timing/runs.json, timing/noise.json, and GPU metadata. Exact invocation/settings, binary SHA-256 and source hashes are in each metadata.json. Reproduction uses benchmarks/run_production_accuracy.py accuracy followed by timing with the saved qualifying evidence.

The archived metadata describes the source and absolute workspace paths at measurement time, before the benchmark changes were committed. Raw logs and the source patch retain their original bytes. Repeated startup-generated ewald.txt and soft.txt kernel-table dumps are omitted from version control; they are not inputs to the accuracy or timing summaries.
