| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| head | auto | cuda | 64 | 7 | adaptive | 0 | 11.335 | 11.298 | 12.323 | 5 |
| head | auto | cuda | 64 | 7 | adaptive | 1 | 11.776 | 11.604 | 12.415 | 5 |
| head | auto | cuda | 64 | 7 | clustered | 0 | 5.551 | 5.352 | 5.713 | 5 |
| head | auto | cuda | 64 | 7 | clustered | 1 | 5.665 | 5.654 | 5.896 | 5 |
| head | auto | cuda | 64 | 7 | lattice | 0 | 14.594 | 14.420 | 14.803 | 5 |
| head | auto | cuda | 64 | 7 | lattice | 1 | 14.703 | 14.427 | 15.437 | 5 |
| head | auto | cuda | 512 | 7 | adaptive | 0 | 24.347 | 24.194 | 26.578 | 5 |
| head | auto | cuda | 512 | 7 | adaptive | 1 | 25.053 | 24.882 | 27.367 | 5 |
| head | auto | cuda | 512 | 7 | clustered | 0 | 5.333 | 5.247 | 5.396 | 5 |
| head | auto | cuda | 512 | 7 | clustered | 1 | 5.968 | 5.883 | 6.364 | 5 |
| head | auto | cuda | 512 | 7 | lattice | 0 | 8.886 | 8.862 | 9.223 | 5 |
| head | auto | cuda | 512 | 7 | lattice | 1 | 9.559 | 9.241 | 9.910 | 5 |
| shared | auto | cuda | 64 | 7 | adaptive | 0 | 11.459 | 11.378 | 12.296 | 5 |
| shared | auto | cuda | 64 | 7 | adaptive | 1 | 12.297 | 11.605 | 12.455 | 5 |
| shared | auto | cuda | 64 | 7 | clustered | 0 | 5.491 | 5.290 | 5.650 | 5 |
| shared | auto | cuda | 64 | 7 | clustered | 1 | 5.848 | 5.605 | 5.856 | 5 |
| shared | auto | cuda | 64 | 7 | lattice | 0 | 14.415 | 14.306 | 14.639 | 5 |
| shared | auto | cuda | 64 | 7 | lattice | 1 | 14.491 | 14.411 | 14.713 | 5 |
| shared | auto | cuda | 512 | 7 | adaptive | 0 | 24.341 | 24.095 | 25.921 | 5 |
| shared | auto | cuda | 512 | 7 | adaptive | 1 | 24.934 | 24.741 | 27.120 | 5 |
| shared | auto | cuda | 512 | 7 | clustered | 0 | 5.364 | 5.140 | 5.395 | 5 |
| shared | auto | cuda | 512 | 7 | clustered | 1 | 5.880 | 5.824 | 6.054 | 5 |
| shared | auto | cuda | 512 | 7 | lattice | 0 | 8.873 | 8.752 | 9.340 | 5 |
| shared | auto | cuda | 512 | 7 | lattice | 1 | 9.349 | 9.243 | 9.769 | 5 |

| binary | policy | backend | bucket | order | fixture | force rel L2 | force p99 | potential rel L2 | potential p99 |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| head | auto | cuda | 64 | 7 | lattice | 3.644e-01 | 2.101e+00 | 4.476e-02 | 1.435e-01 |
| head | auto | cuda | 64 | 7 | clustered | 5.305e-03 | 2.563e-02 | 1.096e-03 | 1.145e-02 |
| head | auto | cuda | 64 | 7 | adaptive | 2.262e-03 | 5.030e-02 | 1.163e-04 | 2.580e-02 |
| head | auto | cuda | 512 | 7 | lattice | 3.644e-01 | 2.101e+00 | 4.476e-02 | 1.435e-01 |
| head | auto | cuda | 512 | 7 | clustered | 2.146e-03 | 2.165e-02 | 1.120e-03 | 1.132e-02 |
| head | auto | cuda | 512 | 7 | adaptive | 1.015e-03 | 6.030e-02 | 1.108e-04 | 2.637e-02 |
| shared | auto | cuda | 64 | 7 | lattice | 3.644e-01 | 2.101e+00 | 4.476e-02 | 1.435e-01 |
| shared | auto | cuda | 64 | 7 | clustered | 5.305e-03 | 2.563e-02 | 1.096e-03 | 1.145e-02 |
| shared | auto | cuda | 64 | 7 | adaptive | 2.262e-03 | 5.030e-02 | 1.163e-04 | 2.580e-02 |
| shared | auto | cuda | 512 | 7 | lattice | 3.644e-01 | 2.101e+00 | 4.476e-02 | 1.435e-01 |
| shared | auto | cuda | 512 | 7 | clustered | 2.146e-03 | 2.165e-02 | 1.120e-03 | 1.132e-02 |
| shared | auto | cuda | 512 | 7 | adaptive | 1.015e-03 | 6.030e-02 | 1.108e-04 | 2.637e-02 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| head | auto | cuda | 64 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8048 bytes: control 648, expansion shared, forces shared tile 64, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |
| head | auto | cuda | 512 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 11760 bytes: control 648, expansion shared, forces shared tile 512, particle tile 224 (without mass plane), multipole tile 16 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |
| shared | auto | cuda | 64 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8048 bytes: control 648, expansion shared, forces shared tile 64, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |
| shared | auto | cuda | 512 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 11760 bytes: control 648, expansion shared, forces shared tile 512, particle tile 224 (without mass plane), multipole tile 16 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
| vs_head | shared | auto | 64 | 7 | adaptive | 5.183e-09 | 2.591e-07 | 1.287e-09 |
| vs_head | shared | auto | 64 | 7 | clustered | 5.646e-09 | 1.820e-07 | 2.845e-09 |
| vs_head | shared | auto | 64 | 7 | lattice | 2.541e-07 | 1.151e-05 | 3.761e-07 |
| vs_head | shared | auto | 512 | 7 | adaptive | 3.994e-09 | 2.544e-07 | 1.169e-09 |
| vs_head | shared | auto | 512 | 7 | clustered | 1.228e-09 | 7.704e-07 | 0.000e+00 |
| vs_head | shared | auto | 512 | 7 | lattice | 2.700e-07 | 2.301e-05 | 3.858e-07 |
