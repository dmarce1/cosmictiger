Round 3 of every variant (14:10:30 to 14:10:54 CDT) overlapped a GPU benchmark
of another session on this workstation (its process appeared in the foreign
process samples at 14:10:43 and 14:10:48); the whole-kick and host times of
that round are 1.5 to 2x those of rounds 1 and 2 and are excluded from the
document's tables. Rounds 1 and 2 ran on an otherwise idle GPU.

| variant | round | whole kick forces (ms) | whole kick forces+potential (ms) | kernel launches | mean kernel (ms) | host share forces (ms) |
|---|---:|---:|---:|---:|---:|---:|
| adaptive128_baseline | 1 | 17.711 | 16.500 | 32 | 7.824 | 9.887 |
| adaptive128_baseline | 2 | 17.358 | 16.119 | 32 | 7.532 | 9.826 |
| adaptive128_baseline | 3 | 29.906 | 30.242 | 32 | 9.393 | 20.513 |
| adaptive128_head | 1 | 13.867 | 12.858 | 32 | 6.835 | 7.032 |
| adaptive128_head | 2 | 13.741 | 12.872 | 32 | 6.864 | 6.877 |
| adaptive128_head | 3 | 23.517 | 22.821 | 32 | 7.404 | 16.113 |
| adaptive128_slab | 1 | 13.233 | 12.414 | 32 | 6.345 | 6.888 |
| adaptive128_slab | 2 | 13.070 | 12.252 | 32 | 6.170 | 6.900 |
| adaptive128_slab | 3 | 23.707 | 20.077 | 32 | 6.715 | 16.991 |
| adaptive64_baseline | 1 | 19.704 | 19.703 | 32 | 5.377 | 14.327 |
| adaptive64_baseline | 2 | 19.459 | 19.595 | 32 | 5.186 | 14.273 |
| adaptive64_baseline | 3 | 36.184 | 35.153 | 32 | 5.487 | 30.697 |
| adaptive64_head | 1 | 13.899 | 12.803 | 32 | 7.237 | 6.662 |
| adaptive64_head | 2 | 13.850 | 12.814 | 32 | 7.227 | 6.624 |
| adaptive64_head | 3 | 23.096 | 20.747 | 32 | 8.650 | 14.445 |
| adaptive64_slab | 1 | 13.012 | 12.207 | 32 | 6.358 | 6.654 |
| adaptive64_slab | 2 | 13.435 | 12.509 | 32 | 6.451 | 6.983 |
| adaptive64_slab | 3 | 18.967 | 18.329 | 32 | 6.639 | 12.328 |

Per-phase medians over the samples (kernel launches paired with kicks in order):

| variant | round | forces: kick (ms) | kernel (ms) | host (ms) | forces+potential: kick (ms) | kernel (ms) | host (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive128_baseline | 1 | 17.711 | 8.108 | 9.494 | 16.500 | 7.032 | 9.401 |
| adaptive128_baseline | 2 | 17.358 | 7.469 | 9.439 | 16.119 | 6.787 | 9.263 |
| adaptive128_baseline | 3 | 29.906 | 8.238 | 21.655 | 30.242 | 8.605 | 21.449 |
| adaptive128_head | 1 | 13.867 | 6.698 | 6.936 | 12.858 | 6.110 | 6.737 |
| adaptive128_head | 2 | 13.741 | 6.514 | 6.922 | 12.872 | 5.885 | 6.652 |
| adaptive128_head | 3 | 23.517 | 6.052 | 16.846 | 22.821 | 7.033 | 14.995 |
| adaptive128_slab | 1 | 13.233 | 6.380 | 6.926 | 12.414 | 5.919 | 6.635 |
| adaptive128_slab | 2 | 13.070 | 6.072 | 6.882 | 12.252 | 5.469 | 6.784 |
| adaptive128_slab | 3 | 23.707 | 6.475 | 16.620 | 20.077 | 6.839 | 13.994 |
| adaptive64_baseline | 1 | 19.704 | 5.311 | 14.400 | 19.703 | 4.960 | 14.315 |
| adaptive64_baseline | 2 | 19.459 | 4.846 | 14.384 | 19.595 | 5.021 | 14.433 |
| adaptive64_baseline | 3 | 36.184 | 5.039 | 31.124 | 35.153 | 5.562 | 29.494 |
| adaptive64_head | 1 | 13.899 | 7.100 | 6.802 | 12.803 | 6.305 | 6.528 |
| adaptive64_head | 2 | 13.850 | 7.180 | 6.782 | 12.814 | 6.205 | 6.404 |
| adaptive64_head | 3 | 23.096 | 7.314 | 14.742 | 20.747 | 7.202 | 13.091 |
| adaptive64_slab | 1 | 13.012 | 6.125 | 6.527 | 12.207 | 5.842 | 6.381 |
| adaptive64_slab | 2 | 13.435 | 6.528 | 6.641 | 12.509 | 5.847 | 6.510 |
| adaptive64_slab | 3 | 18.967 | 6.372 | 12.745 | 18.329 | 6.244 | 12.085 |
