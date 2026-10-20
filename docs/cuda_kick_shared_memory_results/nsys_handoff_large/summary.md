# Host/kernel split of the handoff divisor at 2,097,152 particles

Stage nsys_run.sh, 22:38:21 to 22:56:40 CDT 2026-10-19, under the shared GPU lock (the
quiet gate held rounds 2-3 for about 9 minutes while another session built with ninja);
binary faaf984a (frozen cosmictiger_b1024_handoff_large_rule, sha256 eef7e2f1...).
Order 7, bucket 128, 8 worker threads on CPUs 1-8, 11 samples per phase, three rounds.
The unified-memory rows of stats_cuda_gpu_trace.csv (about 2,400 per kick) were removed
after this summary was written; their per-kernel counts are in each kick_split.csv.

## summarize_kick_profiles.py nsys

| variant | round | whole kick forces (ms) | whole kick forces+potential (ms) | kernel launches | mean kernel (ms) | host share forces (ms) |
|---|---:|---:|---:|---:|---:|---:|
| adaptive128_h328 | 1 | 81.153 | 97.476 | 24 | 71.011 | 10.142 |
| adaptive128_h328 | 2 | 85.920 | 102.574 | 24 | 75.155 | 10.765 |
| adaptive128_h328 | 3 | 81.095 | 96.576 | 24 | 69.713 | 11.382 |
| adaptive128_h656 | 1 | 75.450 | 92.855 | 24 | 61.788 | 13.662 |
| adaptive128_h656 | 2 | 77.902 | 89.325 | 24 | 60.258 | 17.644 |
| adaptive128_h656 | 3 | 74.750 | 88.676 | 24 | 60.478 | 14.271 |
| lattice128_h328 | 1 | 75.188 | 89.792 | 24 | 60.078 | 15.110 |
| lattice128_h328 | 2 | 76.821 | 90.370 | 24 | 61.402 | 15.419 |
| lattice128_h328 | 3 | 81.238 | 93.619 | 24 | 63.796 | 17.441 |
| lattice128_h656 | 1 | 91.754 | 105.298 | 24 | 60.022 | 31.731 |
| lattice128_h656 | 2 | 93.871 | 104.726 | 24 | 61.096 | 32.774 |
| lattice128_h656 | 3 | 95.104 | 106.049 | 24 | 61.841 | 33.263 |

Per-phase medians over the samples (kernel launches paired with kicks in order):

| variant | round | forces: kick (ms) | kernel (ms) | host (ms) | forces+potential: kick (ms) | kernel (ms) | host (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive128_h328 | 1 | 81.153 | 65.935 | 15.791 | 97.476 | 73.109 | 24.305 |
| adaptive128_h328 | 2 | 85.920 | 70.485 | 15.880 | 102.574 | 77.978 | 24.596 |
| adaptive128_h328 | 3 | 81.095 | 64.658 | 16.069 | 96.576 | 72.217 | 24.521 |
| adaptive128_h656 | 1 | 75.450 | 55.327 | 19.158 | 92.855 | 64.671 | 28.566 |
| adaptive128_h656 | 2 | 77.902 | 58.291 | 19.554 | 89.325 | 61.669 | 28.373 |
| adaptive128_h656 | 3 | 74.750 | 55.960 | 18.790 | 88.676 | 60.899 | 27.978 |
| lattice128_h328 | 1 | 75.188 | 56.210 | 19.326 | 89.792 | 61.834 | 28.337 |
| lattice128_h328 | 2 | 76.821 | 57.518 | 19.723 | 90.370 | 62.291 | 28.041 |
| lattice128_h328 | 3 | 81.238 | 61.243 | 19.410 | 93.619 | 64.755 | 28.278 |
| lattice128_h656 | 1 | 91.754 | 56.666 | 34.902 | 105.298 | 61.120 | 44.179 |
| lattice128_h656 | 2 | 93.871 | 58.534 | 34.635 | 104.726 | 60.918 | 43.797 |
| lattice128_h656 | 3 | 95.104 | 59.684 | 35.194 | 106.049 | 62.071 | 43.979 |

## summarize_kick_handoff_profile.py

GPU-phase split, forces: medians over the samples of each round (ms)

| variant | round | kick | host | launch | kernel | UM faults | UM MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive128_h328 | 1 | 81.15 | 13.83 | 1.73 | 65.94 | 1983 | 78.81 |
| adaptive128_h328 | 2 | 85.92 | 14.18 | 1.87 | 70.48 | 2000 | 78.82 |
| adaptive128_h328 | 3 | 81.10 | 14.49 | 1.59 | 64.66 | 1960 | 78.10 |
| adaptive128_h656 | 1 | 75.45 | 17.42 | 1.72 | 55.33 | 1791 | 78.68 |
| adaptive128_h656 | 2 | 77.90 | 17.67 | 1.62 | 58.29 | 1746 | 78.81 |
| adaptive128_h656 | 3 | 74.75 | 17.20 | 1.78 | 55.96 | 1795 | 78.81 |
| lattice128_h328 | 1 | 75.19 | 17.73 | 1.62 | 56.21 | 2018 | 74.91 |
| lattice128_h328 | 2 | 76.82 | 17.66 | 1.78 | 57.52 | 2033 | 74.83 |
| lattice128_h328 | 3 | 81.24 | 17.61 | 1.67 | 61.24 | 2019 | 74.43 |
| lattice128_h656 | 1 | 91.75 | 33.22 | 1.63 | 56.67 | 1845 | 75.36 |
| lattice128_h656 | 2 | 93.87 | 33.06 | 1.67 | 58.53 | 1917 | 75.08 |
| lattice128_h656 | 3 | 95.10 | 33.36 | 1.61 | 59.68 | 1813 | 75.36 |

Median over the rounds, forces (ms)

| variant | kick | host | launch | kernel | UM faults | UM MB |
|---|---:|---:|---:|---:|---:|---:|
| adaptive128_h328 | 81.15 | 14.18 | 1.73 | 65.94 | 1983 | 78.81 |
| adaptive128_h656 | 75.45 | 17.42 | 1.72 | 55.96 | 1791 | 78.81 |
| lattice128_h328 | 76.82 | 17.66 | 1.67 | 57.52 | 2019 | 74.83 |
| lattice128_h656 | 93.87 | 33.22 | 1.63 | 58.53 | 1845 | 75.36 |

GPU-phase split, forces and potential: medians over the samples of each round (ms)

| variant | round | kick | host | launch | kernel | UM faults | UM MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive128_h328 | 1 | 97.48 | 22.73 | 1.60 | 73.11 | 2013 | 78.08 |
| adaptive128_h328 | 2 | 102.57 | 22.98 | 1.59 | 77.98 | 2041 | 78.36 |
| adaptive128_h328 | 3 | 96.58 | 22.82 | 1.59 | 72.22 | 2027 | 78.80 |
| adaptive128_h656 | 1 | 92.85 | 26.55 | 1.68 | 64.67 | 1760 | 78.38 |
| adaptive128_h656 | 2 | 89.33 | 26.50 | 1.63 | 61.67 | 1778 | 78.81 |
| adaptive128_h656 | 3 | 88.68 | 26.28 | 1.65 | 60.90 | 1743 | 78.81 |
| lattice128_h328 | 1 | 89.79 | 26.60 | 1.56 | 61.83 | 1975 | 74.83 |
| lattice128_h328 | 2 | 90.37 | 26.49 | 1.59 | 62.29 | 1939 | 74.37 |
| lattice128_h328 | 3 | 93.62 | 26.58 | 1.58 | 64.75 | 1998 | 74.60 |
| lattice128_h656 | 1 | 105.30 | 42.54 | 1.58 | 61.12 | 1874 | 74.84 |
| lattice128_h656 | 2 | 104.73 | 42.17 | 1.61 | 60.92 | 1899 | 75.62 |
| lattice128_h656 | 3 | 106.05 | 42.14 | 1.59 | 62.07 | 1891 | 75.36 |

Median over the rounds, forces and potential (ms)

| variant | kick | host | launch | kernel | UM faults | UM MB |
|---|---:|---:|---:|---:|---:|---:|
| adaptive128_h328 | 97.48 | 22.82 | 1.59 | 73.11 | 2027 | 78.36 |
| adaptive128_h656 | 89.33 | 26.50 | 1.65 | 61.67 | 1760 | 78.81 |
| lattice128_h328 | 90.37 | 26.58 | 1.58 | 62.29 | 1975 | 74.60 |
| lattice128_h656 | 105.30 | 42.17 | 1.59 | 61.12 | 1891 | 75.36 |
