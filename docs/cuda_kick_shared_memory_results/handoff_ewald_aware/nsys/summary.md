
GPU-phase split, forces: medians over the samples of each round (ms)

| variant | round | kick | host | launch | kernel | UM faults | UM MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive128_auto | 1 | 66.84 | 14.42 | 1.99 | 50.60 | 2163 | 80.00 |
| adaptive128_auto | 2 | 64.74 | 14.12 | 1.72 | 48.67 | 2128 | 79.80 |
| adaptive128_auto | 3 | 64.69 | 14.03 | 1.91 | 48.42 | 2153 | 79.51 |
| adaptive128_fine1312 | 1 | 76.86 | 19.35 | 1.79 | 55.38 | 1802 | 79.72 |
| adaptive128_fine1312 | 2 | 77.47 | 19.36 | 1.81 | 55.49 | 1822 | 79.78 |
| adaptive128_fine1312 | 3 | 76.16 | 19.53 | 1.77 | 54.81 | 1800 | 79.55 |
| adaptive128_fine656 | 1 | 73.88 | 15.41 | 1.75 | 56.37 | 2017 | 79.60 |
| adaptive128_fine656 | 2 | 73.98 | 15.99 | 1.76 | 55.78 | 2079 | 79.66 |
| adaptive128_fine656 | 3 | 73.72 | 16.01 | 1.93 | 55.60 | 2038 | 80.02 |
| adaptive128_h656 | 1 | 65.79 | 17.68 | 1.79 | 47.01 | 2067 | 79.56 |
| adaptive128_h656 | 2 | 66.31 | 17.63 | 1.95 | 46.87 | 2081 | 79.57 |
| adaptive128_h656 | 3 | 66.44 | 17.79 | 1.78 | 47.28 | 2054 | 79.33 |

Median over the rounds, forces (ms)

| variant | kick | host | launch | kernel | UM faults | UM MB |
|---|---:|---:|---:|---:|---:|---:|
| adaptive128_auto | 64.74 | 14.12 | 1.91 | 48.67 | 2153 | 79.80 |
| adaptive128_fine1312 | 76.86 | 19.36 | 1.79 | 55.38 | 1802 | 79.72 |
| adaptive128_fine656 | 73.88 | 15.99 | 1.76 | 55.78 | 2038 | 79.66 |
| adaptive128_h656 | 66.31 | 17.68 | 1.79 | 47.01 | 2067 | 79.56 |

GPU-phase split, forces and potential: medians over the samples of each round (ms)

| variant | round | kick | host | launch | kernel | UM faults | UM MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive128_auto | 1 | 83.98 | 22.92 | 1.61 | 59.84 | 2111 | 79.86 |
| adaptive128_auto | 2 | 82.46 | 22.62 | 1.75 | 58.02 | 2191 | 79.60 |
| adaptive128_auto | 3 | 82.79 | 22.99 | 1.74 | 58.18 | 2143 | 79.14 |
| adaptive128_fine1312 | 1 | 96.51 | 28.25 | 1.78 | 66.53 | 1747 | 80.05 |
| adaptive128_fine1312 | 2 | 96.80 | 28.14 | 1.78 | 66.76 | 1787 | 79.78 |
| adaptive128_fine1312 | 3 | 96.32 | 27.95 | 1.79 | 66.42 | 1784 | 79.79 |
| adaptive128_fine656 | 1 | 94.66 | 24.75 | 1.75 | 68.34 | 2096 | 80.54 |
| adaptive128_fine656 | 2 | 90.78 | 24.45 | 1.68 | 64.92 | 2057 | 79.54 |
| adaptive128_fine656 | 3 | 94.91 | 24.62 | 1.74 | 68.13 | 2012 | 79.51 |
| adaptive128_h656 | 1 | 86.32 | 26.30 | 1.74 | 58.20 | 2055 | 79.12 |
| adaptive128_h656 | 2 | 88.12 | 26.30 | 1.67 | 59.91 | 2019 | 79.69 |
| adaptive128_h656 | 3 | 85.82 | 25.92 | 1.73 | 57.61 | 2066 | 79.64 |

Median over the rounds, forces and potential (ms)

| variant | kick | host | launch | kernel | UM faults | UM MB |
|---|---:|---:|---:|---:|---:|---:|
| adaptive128_auto | 82.79 | 22.92 | 1.74 | 58.18 | 2143 | 79.60 |
| adaptive128_fine1312 | 96.51 | 28.14 | 1.78 | 66.53 | 1784 | 79.79 |
| adaptive128_fine656 | 94.66 | 24.62 | 1.74 | 68.13 | 2057 | 79.54 |
| adaptive128_h656 | 86.32 | 26.30 | 1.73 | 58.20 | 2055 | 79.64 |
