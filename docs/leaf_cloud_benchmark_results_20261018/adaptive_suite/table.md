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
