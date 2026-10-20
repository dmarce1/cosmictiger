# CPU walk by routine, handoff divisor at 2,097,152 particles

Stage nsys_run.sh, 23:18:24 to 23:25:43 CDT 2026-10-19, under the shared GPU lock (taken before
the incremental build of the binary and held through the stage); binary faaf984a plus
instrumentation.patch (cosmictiger_b1024_handoff_large_rule_nvtx_walk, sha256 1222dca1...),
same build configuration as the frozen binary. Order 7, bucket 128, 8 worker threads on CPUs 1-8,
11 samples per phase, two rounds. cpu* columns are thread time summed over the CPU nodes of a
kick. The unified-memory rows of stats_cuda_gpu_trace.csv were removed after this summary was
written (counts in kick_split.csv).

## summarize_kick_handoff_profile.py --nvtx

NVTX phases, forces: medians over the samples of each round (ms; counts per kick)

| variant | round | kick | walk | prepare | pack | device | other | tail | addWork | held | cpuNode | cpuEwald | cpuDirect | cpuCp | cpuNodes | items | dchecks | echecks | treeIds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptive128_t8_h328 | 1 | 81.85 | 4.97 | 5.44 | 0.04 | 67.16 | 3.44 | 0.78 | 20.39 | 1.63 | 21.71 | 13.46 | 1.90 | 2.51 | 546 | 547 | 21035 | 6180 | 1318 |
| adaptive128_t8_h328 | 2 | 84.85 | 5.03 | 5.48 | 0.04 | 70.42 | 3.45 | 0.82 | 22.38 | 1.74 | 21.83 | 13.49 | 1.92 | 2.50 | 546 | 547 | 21035 | 6180 | 1318 |
| adaptive128_t8_h656 | 1 | 75.02 | 8.86 | 5.01 | 0.09 | 55.84 | 3.49 | 0.82 | 52.47 | 3.62 | 37.99 | 23.13 | 3.31 | 4.28 | 1075 | 1076 | 55082 | 16326 | 2050 |
| adaptive128_t8_h656 | 2 | 75.41 | 8.86 | 5.48 | 0.09 | 56.76 | 3.54 | 0.82 | 51.85 | 3.66 | 37.96 | 23.17 | 3.28 | 4.29 | 1075 | 1076 | 55082 | 16326 | 2050 |
| lattice128_t8_h328 | 1 | 76.77 | 10.45 | 3.98 | 0.06 | 57.98 | 3.27 | 0.55 | 33.47 | 1.76 | 73.57 | 66.93 | 1.72 | 0.15 | 511 | 512 | 28824 | 28824 | 762 |
| lattice128_t8_h328 | 2 | 76.72 | 10.42 | 3.98 | 0.06 | 58.40 | 3.25 | 0.68 | 29.58 | 1.79 | 73.64 | 67.04 | 1.72 | 0.15 | 511 | 512 | 28824 | 28824 | 762 |
| lattice128_t8_h656 | 1 | 94.79 | 26.09 | 4.12 | 0.08 | 60.41 | 3.35 | 0.60 | 32.96 | 2.12 | 194.10 | 181.93 | 3.76 | 0.29 | 1023 | 1024 | 45588 | 0 | 1516 |
| lattice128_t8_h656 | 2 | 91.85 | 26.08 | 3.90 | 0.07 | 57.35 | 3.36 | 0.60 | 39.56 | 2.33 | 194.03 | 181.95 | 3.76 | 0.29 | 1023 | 1024 | 45588 | 0 | 1516 |

Median over the rounds, forces (ms; counts per kick)

| variant | kick | walk | prepare | pack | device | other | tail | addWork | held | cpuNode | cpuEwald | cpuDirect | cpuCp | cpuNodes | items | dchecks | echecks | treeIds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptive128_t8_h328 | 83.35 | 5.00 | 5.46 | 0.04 | 68.79 | 3.44 | 0.80 | 21.39 | 1.69 | 21.77 | 13.48 | 1.91 | 2.51 | 546 | 547 | 21035 | 6180 | 1318 |
| adaptive128_t8_h656 | 75.22 | 8.86 | 5.25 | 0.09 | 56.30 | 3.52 | 0.82 | 52.16 | 3.64 | 37.97 | 23.15 | 3.30 | 4.29 | 1075 | 1076 | 55082 | 16326 | 2050 |
| lattice128_t8_h328 | 76.75 | 10.44 | 3.98 | 0.06 | 58.19 | 3.26 | 0.61 | 31.53 | 1.77 | 73.61 | 66.98 | 1.72 | 0.15 | 511 | 512 | 28824 | 28824 | 762 |
| lattice128_t8_h656 | 93.32 | 26.09 | 4.01 | 0.08 | 58.88 | 3.35 | 0.60 | 36.26 | 2.23 | 194.07 | 181.94 | 3.76 | 0.29 | 1023 | 1024 | 45588 | 0 | 1516 |

NVTX phases, forces and potential: medians over the samples of each round (ms; counts per kick)

| variant | round | kick | walk | prepare | pack | device | other | tail | addWork | held | cpuNode | cpuEwald | cpuDirect | cpuCp | cpuNodes | items | dchecks | echecks | treeIds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptive128_t8_h328 | 1 | 98.35 | 14.31 | 4.72 | 0.04 | 74.71 | 3.44 | 0.78 | 20.84 | 1.59 | 21.63 | 13.36 | 1.91 | 2.47 | 546 | 547 | 21035 | 6180 | 1318 |
| adaptive128_t8_h328 | 2 | 100.61 | 14.33 | 4.70 | 0.04 | 77.37 | 3.45 | 0.83 | 20.60 | 1.61 | 21.62 | 13.42 | 1.90 | 2.45 | 546 | 547 | 21035 | 6180 | 1318 |
| adaptive128_t8_h656 | 1 | 91.39 | 18.20 | 4.97 | 0.09 | 63.58 | 3.51 | 0.86 | 55.13 | 3.54 | 37.86 | 23.18 | 3.29 | 4.22 | 1075 | 1076 | 55082 | 16326 | 2050 |
| adaptive128_t8_h656 | 2 | 90.56 | 18.21 | 4.74 | 0.09 | 62.48 | 3.49 | 0.77 | 55.55 | 3.51 | 37.88 | 23.22 | 3.31 | 4.24 | 1075 | 1076 | 55082 | 16326 | 2050 |
| lattice128_t8_h328 | 1 | 90.23 | 19.84 | 3.14 | 0.06 | 63.36 | 3.23 | 0.52 | 26.57 | 1.76 | 73.57 | 67.00 | 1.72 | 0.15 | 511 | 512 | 28824 | 28824 | 762 |
| lattice128_t8_h328 | 2 | 90.95 | 19.82 | 3.20 | 0.06 | 64.00 | 3.22 | 0.57 | 30.68 | 1.76 | 73.17 | 66.61 | 1.71 | 0.15 | 511 | 512 | 28824 | 28824 | 762 |
| lattice128_t8_h656 | 1 | 105.51 | 35.93 | 3.55 | 0.07 | 61.80 | 3.27 | 0.55 | 39.61 | 2.20 | 196.10 | 183.74 | 3.86 | 0.31 | 1023 | 1024 | 45588 | 0 | 1516 |
| lattice128_t8_h656 | 2 | 105.92 | 35.81 | 3.61 | 0.07 | 62.50 | 3.31 | 0.55 | 37.25 | 2.17 | 196.50 | 184.01 | 3.84 | 0.32 | 1023 | 1024 | 45588 | 0 | 1516 |

Median over the rounds, forces and potential (ms; counts per kick)

| variant | kick | walk | prepare | pack | device | other | tail | addWork | held | cpuNode | cpuEwald | cpuDirect | cpuCp | cpuNodes | items | dchecks | echecks | treeIds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptive128_t8_h328 | 99.48 | 14.32 | 4.71 | 0.04 | 76.04 | 3.44 | 0.81 | 20.72 | 1.60 | 21.63 | 13.39 | 1.90 | 2.46 | 546 | 547 | 21035 | 6180 | 1318 |
| adaptive128_t8_h656 | 90.97 | 18.21 | 4.85 | 0.09 | 63.03 | 3.50 | 0.81 | 55.34 | 3.52 | 37.87 | 23.20 | 3.30 | 4.23 | 1075 | 1076 | 55082 | 16326 | 2050 |
| lattice128_t8_h328 | 90.59 | 19.83 | 3.17 | 0.06 | 63.68 | 3.23 | 0.55 | 28.63 | 1.76 | 73.37 | 66.80 | 1.72 | 0.15 | 511 | 512 | 28824 | 28824 | 762 |
| lattice128_t8_h656 | 105.72 | 35.87 | 3.58 | 0.07 | 62.15 | 3.29 | 0.55 | 38.43 | 2.19 | 196.30 | 183.87 | 3.85 | 0.32 | 1023 | 1024 | 45588 | 0 | 1516 |

## summarize_kick_handoff_profile.py

GPU-phase split, forces: medians over the samples of each round (ms)

| variant | round | kick | host | launch | kernel | UM faults | UM MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive128_t8_h328 | 1 | 81.86 | 14.68 | 1.67 | 65.52 | 2024 | 78.61 |
| adaptive128_t8_h328 | 2 | 84.85 | 14.81 | 1.62 | 68.72 | 2009 | 78.17 |
| adaptive128_t8_h656 | 1 | 75.03 | 18.39 | 1.64 | 54.25 | 1788 | 77.84 |
| adaptive128_t8_h656 | 2 | 75.42 | 18.61 | 1.64 | 54.96 | 1767 | 78.17 |
| lattice128_t8_h328 | 1 | 76.78 | 17.93 | 1.63 | 56.40 | 1979 | 74.38 |
| lattice128_t8_h328 | 2 | 76.73 | 18.26 | 1.65 | 56.84 | 2035 | 74.34 |
| lattice128_t8_h656 | 1 | 94.80 | 33.97 | 1.67 | 58.48 | 1895 | 75.17 |
| lattice128_t8_h656 | 2 | 91.86 | 33.99 | 1.66 | 55.71 | 1883 | 75.36 |

Median over the rounds, forces (ms)

| variant | kick | host | launch | kernel | UM faults | UM MB |
|---|---:|---:|---:|---:|---:|---:|
| adaptive128_t8_h328 | 83.36 | 14.75 | 1.65 | 67.12 | 2016 | 78.39 |
| adaptive128_t8_h656 | 75.22 | 18.50 | 1.64 | 54.60 | 1778 | 78.01 |
| lattice128_t8_h328 | 76.75 | 18.09 | 1.64 | 56.62 | 2007 | 74.36 |
| lattice128_t8_h656 | 93.33 | 33.98 | 1.66 | 57.09 | 1889 | 75.26 |

GPU-phase split, forces and potential: medians over the samples of each round (ms)

| variant | round | kick | host | launch | kernel | UM faults | UM MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive128_t8_h328 | 1 | 98.36 | 23.46 | 1.62 | 73.17 | 2036 | 78.29 |
| adaptive128_t8_h328 | 2 | 100.61 | 23.67 | 1.83 | 75.51 | 2019 | 78.40 |
| adaptive128_t8_h656 | 1 | 91.39 | 27.76 | 1.66 | 61.93 | 1731 | 78.72 |
| adaptive128_t8_h656 | 2 | 90.57 | 27.47 | 1.62 | 60.79 | 1766 | 77.62 |
| lattice128_t8_h328 | 1 | 90.24 | 26.76 | 1.62 | 61.75 | 1993 | 74.80 |
| lattice128_t8_h328 | 2 | 90.96 | 26.89 | 1.62 | 62.10 | 2078 | 74.63 |
| lattice128_t8_h656 | 1 | 105.52 | 43.58 | 1.63 | 60.16 | 1909 | 75.35 |
| lattice128_t8_h656 | 2 | 105.93 | 43.46 | 1.64 | 60.61 | 1832 | 74.84 |

Median over the rounds, forces and potential (ms)

| variant | kick | host | launch | kernel | UM faults | UM MB |
|---|---:|---:|---:|---:|---:|---:|
| adaptive128_t8_h328 | 99.49 | 23.56 | 1.72 | 74.34 | 2028 | 78.34 |
| adaptive128_t8_h656 | 90.98 | 27.62 | 1.64 | 61.36 | 1748 | 78.17 |
| lattice128_t8_h328 | 90.60 | 26.83 | 1.62 | 61.92 | 2036 | 74.72 |
| lattice128_t8_h656 | 105.73 | 43.52 | 1.63 | 60.38 | 1870 | 75.09 |
