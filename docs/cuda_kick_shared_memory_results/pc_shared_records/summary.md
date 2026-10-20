| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| head | auto | cuda | 64 | 7 | adaptive | 0 | 11.381 | 11.182 | 12.314 | 5 |
| head | auto | cuda | 64 | 7 | adaptive | 1 | 11.601 | 11.479 | 12.324 | 5 |
| head | auto | cuda | 64 | 7 | clustered | 0 | 5.401 | 5.284 | 5.773 | 5 |
| head | auto | cuda | 64 | 7 | clustered | 1 | 5.709 | 5.591 | 5.911 | 5 |
| head | auto | cuda | 64 | 7 | lattice | 0 | 14.342 | 14.300 | 14.886 | 5 |
| head | auto | cuda | 64 | 7 | lattice | 1 | 14.415 | 14.397 | 15.114 | 5 |
| head | auto | cuda | 512 | 7 | adaptive | 0 | 24.075 | 23.859 | 25.743 | 5 |
| head | auto | cuda | 512 | 7 | adaptive | 1 | 24.624 | 24.525 | 26.227 | 5 |
| head | auto | cuda | 512 | 7 | clustered | 0 | 5.203 | 5.116 | 5.494 | 5 |
| head | auto | cuda | 512 | 7 | clustered | 1 | 5.831 | 5.808 | 6.096 | 5 |
| head | auto | cuda | 512 | 7 | lattice | 0 | 8.802 | 8.727 | 9.126 | 5 |
| head | auto | cuda | 512 | 7 | lattice | 1 | 9.381 | 9.298 | 9.533 | 5 |
| shared | auto | cuda | 64 | 7 | adaptive | 0 | 11.408 | 11.219 | 12.631 | 5 |
| shared | auto | cuda | 64 | 7 | adaptive | 1 | 11.549 | 11.464 | 12.435 | 5 |
| shared | auto | cuda | 64 | 7 | clustered | 0 | 5.398 | 5.283 | 5.637 | 5 |
| shared | auto | cuda | 64 | 7 | clustered | 1 | 5.636 | 5.547 | 5.968 | 5 |
| shared | auto | cuda | 64 | 7 | lattice | 0 | 14.424 | 14.309 | 14.753 | 5 |
| shared | auto | cuda | 64 | 7 | lattice | 1 | 14.536 | 14.436 | 14.752 | 5 |
| shared | auto | cuda | 512 | 7 | adaptive | 0 | 23.976 | 23.874 | 25.732 | 5 |
| shared | auto | cuda | 512 | 7 | adaptive | 1 | 24.637 | 24.552 | 26.657 | 5 |
| shared | auto | cuda | 512 | 7 | clustered | 0 | 5.129 | 5.112 | 5.741 | 5 |
| shared | auto | cuda | 512 | 7 | clustered | 1 | 5.856 | 5.779 | 6.275 | 5 |
| shared | auto | cuda | 512 | 7 | lattice | 0 | 8.872 | 8.753 | 9.281 | 5 |
| shared | auto | cuda | 512 | 7 | lattice | 1 | 9.328 | 9.211 | 9.823 | 5 |

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
| vs_head | shared | auto | 64 | 7 | adaptive | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_head | shared | auto | 64 | 7 | clustered | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_head | shared | auto | 64 | 7 | lattice | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_head | shared | auto | 512 | 7 | adaptive | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_head | shared | auto | 512 | 7 | clustered | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_head | shared | auto | 512 | 7 | lattice | 0.000e+00 | 0.000e+00 | 0.000e+00 |
