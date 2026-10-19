| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| arena | auto | cuda | 64 | 7 | adaptive | 0 | 21.904 | 21.841 | 21.975 | 3 |
| arena | auto | cuda | 64 | 7 | adaptive | 1 | 21.961 | 21.802 | 22.362 | 3 |
| arena | auto | cuda | 64 | 7 | clustered | 0 | 9.391 | 8.934 | 9.432 | 3 |
| arena | auto | cuda | 64 | 7 | clustered | 1 | 9.727 | 9.223 | 9.819 | 3 |
| arena | auto | cuda | 64 | 7 | lattice | 0 | 34.001 | 33.206 | 34.766 | 3 |
| arena | auto | cuda | 64 | 7 | lattice | 1 | 33.844 | 33.800 | 34.296 | 3 |
| arena | r6 | cuda | 64 | 7 | adaptive | 0 | 21.740 | 21.550 | 22.095 | 3 |
| arena | r6 | cuda | 64 | 7 | adaptive | 1 | 22.215 | 21.650 | 22.264 | 3 |
| arena | r6 | cuda | 64 | 7 | clustered | 0 | 9.216 | 8.884 | 9.492 | 3 |
| arena | r6 | cuda | 64 | 7 | clustered | 1 | 9.710 | 9.040 | 9.715 | 3 |
| arena | r6 | cuda | 64 | 7 | lattice | 0 | 34.020 | 33.173 | 34.465 | 3 |
| arena | r6 | cuda | 64 | 7 | lattice | 1 | 33.953 | 33.459 | 34.140 | 3 |
| baseline | auto | cuda | 64 | 7 | adaptive | 0 | 20.373 | 20.105 | 20.380 | 3 |
| baseline | auto | cuda | 64 | 7 | adaptive | 1 | 20.526 | 20.146 | 21.732 | 3 |
| baseline | auto | cuda | 64 | 7 | clustered | 0 | 9.866 | 9.242 | 9.924 | 3 |
| baseline | auto | cuda | 64 | 7 | clustered | 1 | 9.628 | 9.378 | 10.136 | 3 |
| baseline | auto | cuda | 64 | 7 | lattice | 0 | 34.931 | 34.782 | 34.934 | 3 |
| baseline | auto | cuda | 64 | 7 | lattice | 1 | 35.023 | 34.730 | 35.215 | 3 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| arena | auto | cuda | 64 | 7 | `` |
| arena | r6 | cuda | 64 | 7 | `` |
| baseline | auto | cuda | 64 | 7 | `Occupancy is 6 shmem size = 15184 numregs = 255` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
| vs_baseline | arena | auto | 64 | 7 | adaptive | 4.589e-08 | 1.221e-06 | 3.594e-08 |
| vs_baseline | arena | auto | 64 | 7 | clustered | 2.326e-08 | 3.579e-06 | 3.596e-08 |
| vs_baseline | arena | auto | 64 | 7 | lattice | 1.004e-06 | 6.536e-05 | 3.584e-06 |
| vs_baseline | arena | r6 | 64 | 7 | adaptive | 4.589e-08 | 1.221e-06 | 3.594e-08 |
| vs_baseline | arena | r6 | 64 | 7 | clustered | 2.326e-08 | 3.579e-06 | 3.596e-08 |
| vs_baseline | arena | r6 | 64 | 7 | lattice | 1.004e-06 | 6.536e-05 | 3.584e-06 |
