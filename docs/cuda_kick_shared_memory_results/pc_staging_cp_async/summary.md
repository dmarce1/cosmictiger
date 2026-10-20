| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| aligned | auto | cuda | 64 | 7 | adaptive | 0 | 12.077 | 11.419 | 12.319 | 5 |
| aligned | auto | cuda | 64 | 7 | adaptive | 1 | 12.317 | 12.214 | 12.430 | 5 |
| aligned | auto | cuda | 64 | 7 | clustered | 0 | 5.551 | 5.506 | 5.694 | 5 |
| aligned | auto | cuda | 64 | 7 | clustered | 1 | 5.880 | 5.760 | 5.981 | 5 |
| aligned | auto | cuda | 64 | 7 | lattice | 0 | 14.677 | 14.486 | 14.959 | 5 |
| aligned | auto | cuda | 64 | 7 | lattice | 1 | 14.728 | 14.559 | 14.937 | 5 |
| aligned | auto | cuda | 512 | 7 | adaptive | 0 | 26.139 | 25.871 | 26.434 | 5 |
| aligned | auto | cuda | 512 | 7 | adaptive | 1 | 27.151 | 26.738 | 29.691 | 5 |
| aligned | auto | cuda | 512 | 7 | clustered | 0 | 5.376 | 5.332 | 5.880 | 5 |
| aligned | auto | cuda | 512 | 7 | clustered | 1 | 6.110 | 6.058 | 6.350 | 5 |
| aligned | auto | cuda | 512 | 7 | lattice | 0 | 9.494 | 9.172 | 9.572 | 5 |
| aligned | auto | cuda | 512 | 7 | lattice | 1 | 10.103 | 9.732 | 10.286 | 5 |
| head | auto | cuda | 64 | 7 | adaptive | 0 | 12.127 | 11.771 | 12.312 | 5 |
| head | auto | cuda | 64 | 7 | adaptive | 1 | 12.403 | 12.332 | 12.531 | 5 |
| head | auto | cuda | 64 | 7 | clustered | 0 | 5.573 | 5.484 | 5.598 | 5 |
| head | auto | cuda | 64 | 7 | clustered | 1 | 5.736 | 5.679 | 5.910 | 5 |
| head | auto | cuda | 64 | 7 | lattice | 0 | 14.705 | 14.534 | 15.011 | 5 |
| head | auto | cuda | 64 | 7 | lattice | 1 | 14.810 | 14.566 | 15.113 | 5 |
| head | auto | cuda | 512 | 7 | adaptive | 0 | 26.774 | 26.187 | 27.332 | 5 |
| head | auto | cuda | 512 | 7 | adaptive | 1 | 27.475 | 26.884 | 27.589 | 5 |
| head | auto | cuda | 512 | 7 | clustered | 0 | 5.454 | 5.309 | 5.500 | 5 |
| head | auto | cuda | 512 | 7 | clustered | 1 | 6.111 | 5.959 | 6.248 | 5 |
| head | auto | cuda | 512 | 7 | lattice | 0 | 9.567 | 9.217 | 9.857 | 5 |
| head | auto | cuda | 512 | 7 | lattice | 1 | 10.089 | 9.748 | 10.337 | 5 |

| binary | policy | backend | bucket | order | fixture | force rel L2 | force p99 | potential rel L2 | potential p99 |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| head | auto | cuda | 64 | 7 | lattice | 3.644e-01 | 2.101e+00 | 4.476e-02 | 1.435e-01 |
| head | auto | cuda | 64 | 7 | clustered | 5.305e-03 | 2.563e-02 | 1.096e-03 | 1.145e-02 |
| head | auto | cuda | 64 | 7 | adaptive | 2.262e-03 | 5.030e-02 | 1.163e-04 | 2.580e-02 |
| head | auto | cuda | 512 | 7 | lattice | 3.644e-01 | 2.101e+00 | 4.476e-02 | 1.435e-01 |
| head | auto | cuda | 512 | 7 | clustered | 2.146e-03 | 2.165e-02 | 1.120e-03 | 1.132e-02 |
| head | auto | cuda | 512 | 7 | adaptive | 1.015e-03 | 6.030e-02 | 1.108e-04 | 2.637e-02 |
| aligned | auto | cuda | 64 | 7 | lattice | 3.644e-01 | 2.101e+00 | 4.476e-02 | 1.435e-01 |
| aligned | auto | cuda | 64 | 7 | clustered | 5.305e-03 | 2.563e-02 | 1.096e-03 | 1.145e-02 |
| aligned | auto | cuda | 64 | 7 | adaptive | 2.262e-03 | 5.030e-02 | 1.163e-04 | 2.580e-02 |
| aligned | auto | cuda | 512 | 7 | lattice | 3.644e-01 | 2.101e+00 | 4.476e-02 | 1.435e-01 |
| aligned | auto | cuda | 512 | 7 | clustered | 2.146e-03 | 2.165e-02 | 1.120e-03 | 1.132e-02 |
| aligned | auto | cuda | 512 | 7 | adaptive | 1.015e-03 | 6.030e-02 | 1.108e-04 | 2.637e-02 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| aligned | auto | cuda | 64 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8048 bytes: control 648, expansion shared, forces shared tile 64, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |
| aligned | auto | cuda | 512 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 11760 bytes: control 648, expansion shared, forces shared tile 512, particle tile 224 (without mass plane), multipole tile 16 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |
| head | auto | cuda | 64 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8048 bytes: control 648, expansion shared, forces shared tile 64, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |
| head | auto | cuda | 512 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 11760 bytes: control 648, expansion shared, forces shared tile 512, particle tile 224 (without mass plane), multipole tile 16 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
| vs_head | aligned | auto | 64 | 7 | adaptive | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_head | aligned | auto | 64 | 7 | clustered | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_head | aligned | auto | 64 | 7 | lattice | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_head | aligned | auto | 512 | 7 | adaptive | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_head | aligned | auto | 512 | 7 | clustered | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_head | aligned | auto | 512 | 7 | lattice | 0.000e+00 | 0.000e+00 | 0.000e+00 |
