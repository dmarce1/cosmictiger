| variant | round | whole kick forces (ms) | whole kick forces+potential (ms) | kernel launches | mean kernel (ms) | host share forces (ms) |
|---|---:|---:|---:|---:|---:|---:|
| adaptive512_arena_auto | 1 | 36.381 | 35.117 | 32 | 30.796 | 5.585 |
| adaptive512_arena_auto | 2 | 36.007 | 36.114 | 32 | 30.806 | 5.202 |
| adaptive512_arena_auto | 3 | 36.582 | 35.129 | 32 | 31.137 | 5.445 |
| adaptive512_arena_res6 | 1 | 35.055 | 35.706 | 32 | 30.897 | 4.157 |
| adaptive512_arena_res6 | 2 | 35.480 | 36.723 | 32 | 30.838 | 4.642 |
| adaptive512_arena_res6 | 3 | 35.640 | 35.511 | 32 | 30.659 | 4.981 |
| adaptive512_baseline | 1 | 29.579 | 29.029 | 32 | 24.822 | 4.758 |
| adaptive512_baseline | 2 | 30.665 | 29.460 | 32 | 24.704 | 5.960 |
| adaptive512_baseline | 3 | 31.202 | 29.429 | 32 | 24.857 | 6.345 |
| adaptive64_arena_auto | 1 | 20.920 | 20.837 | 32 | 6.324 | 14.597 |
| adaptive64_arena_auto | 2 | 20.929 | 21.142 | 32 | 6.353 | 14.576 |
| adaptive64_arena_auto | 3 | 21.311 | 20.645 | 32 | 6.555 | 14.756 |
| adaptive64_arena_res6 | 1 | 21.233 | 20.817 | 32 | 6.576 | 14.657 |
| adaptive64_arena_res6 | 2 | 20.860 | 21.058 | 32 | 6.428 | 14.432 |
| adaptive64_arena_res6 | 3 | 21.508 | 21.282 | 32 | 6.650 | 14.857 |
| adaptive64_baseline | 1 | 19.699 | 19.608 | 32 | 5.312 | 14.387 |
| adaptive64_baseline | 2 | 19.604 | 19.689 | 32 | 5.192 | 14.412 |
| adaptive64_baseline | 3 | 19.780 | 19.784 | 32 | 5.452 | 14.327 |
| lattice32_arena_auto | 1 | 92.024 | 91.618 | 32 | 4.282 | 87.741 |
| lattice32_arena_auto | 2 | 91.182 | 91.081 | 32 | 4.122 | 87.060 |
| lattice32_arena_auto | 3 | 91.485 | 91.311 | 32 | 4.416 | 87.069 |
| lattice32_arena_res6 | 1 | 92.019 | 91.474 | 32 | 4.256 | 87.763 |
| lattice32_arena_res6 | 2 | 91.272 | 91.388 | 32 | 4.311 | 86.961 |
| lattice32_arena_res6 | 3 | 91.360 | 91.035 | 32 | 4.230 | 87.129 |
| lattice32_baseline | 1 | 33.770 | 33.860 | 32 | 4.064 | 29.706 |
| lattice32_baseline | 2 | 33.817 | 33.619 | 32 | 4.211 | 29.606 |
| lattice32_baseline | 3 | 34.378 | 33.935 | 32 | 4.223 | 30.155 |

Per-phase medians over the samples (kernel launches paired with kicks in order):

| variant | round | forces: kick (ms) | kernel (ms) | host (ms) | forces+potential: kick (ms) | kernel (ms) | host (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive512_arena_auto | 1 | 36.381 | 30.855 | 5.440 | 35.117 | 29.887 | 5.266 |
| adaptive512_arena_auto | 2 | 36.007 | 29.819 | 5.465 | 36.114 | 30.399 | 5.642 |
| adaptive512_arena_auto | 3 | 36.582 | 30.918 | 5.446 | 35.129 | 29.535 | 5.698 |
| adaptive512_arena_res6 | 1 | 35.055 | 29.820 | 5.479 | 35.706 | 30.175 | 5.430 |
| adaptive512_arena_res6 | 2 | 35.480 | 29.977 | 5.388 | 36.723 | 31.421 | 5.302 |
| adaptive512_arena_res6 | 3 | 35.640 | 30.476 | 5.395 | 35.511 | 30.247 | 5.373 |
| adaptive512_baseline | 1 | 29.579 | 24.194 | 5.410 | 29.029 | 23.647 | 5.416 |
| adaptive512_baseline | 2 | 30.665 | 24.808 | 5.462 | 29.460 | 23.961 | 5.493 |
| adaptive512_baseline | 3 | 31.202 | 25.695 | 5.398 | 29.429 | 23.948 | 5.401 |
| adaptive64_arena_auto | 1 | 20.920 | 5.624 | 15.189 | 20.837 | 5.605 | 14.764 |
| adaptive64_arena_auto | 2 | 20.929 | 5.503 | 14.965 | 21.142 | 5.899 | 14.903 |
| adaptive64_arena_auto | 3 | 21.311 | 6.338 | 14.959 | 20.645 | 5.693 | 14.843 |
| adaptive64_arena_res6 | 1 | 21.233 | 6.057 | 15.020 | 20.817 | 5.679 | 14.853 |
| adaptive64_arena_res6 | 2 | 20.860 | 5.790 | 14.886 | 21.058 | 6.115 | 14.983 |
| adaptive64_arena_res6 | 3 | 21.508 | 6.268 | 15.213 | 21.282 | 6.408 | 14.868 |
| adaptive64_baseline | 1 | 19.699 | 5.051 | 14.536 | 19.608 | 5.102 | 14.613 |
| adaptive64_baseline | 2 | 19.604 | 4.799 | 14.653 | 19.689 | 5.029 | 14.321 |
| adaptive64_baseline | 3 | 19.780 | 5.406 | 14.391 | 19.784 | 5.636 | 14.217 |
| lattice32_arena_auto | 1 | 92.024 | 3.626 | 88.380 | 91.618 | 3.619 | 87.805 |
| lattice32_arena_auto | 2 | 91.182 | 3.815 | 87.341 | 91.081 | 3.587 | 87.231 |
| lattice32_arena_auto | 3 | 91.485 | 3.663 | 87.420 | 91.311 | 3.732 | 87.290 |
| lattice32_arena_res6 | 1 | 92.019 | 4.103 | 87.949 | 91.474 | 3.637 | 87.784 |
| lattice32_arena_res6 | 2 | 91.272 | 3.941 | 87.562 | 91.388 | 3.984 | 87.574 |
| lattice32_arena_res6 | 3 | 91.360 | 3.741 | 87.267 | 91.035 | 3.620 | 87.168 |
| lattice32_baseline | 1 | 33.770 | 3.830 | 29.800 | 33.860 | 3.956 | 29.905 |
| lattice32_baseline | 2 | 33.817 | 3.948 | 29.651 | 33.619 | 3.995 | 29.613 |
| lattice32_baseline | 3 | 34.378 | 4.179 | 30.019 | 33.935 | 3.964 | 29.808 |
