| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| head | auto | cuda | 64 | 7 | adaptive | 0 | 10.434 | 10.219 | 10.547 | 5 |
| head | auto | cuda | 64 | 7 | adaptive | 1 | 10.713 | 10.556 | 10.768 | 5 |
| head | auto | cuda | 64 | 7 | clustered | 0 | 5.746 | 5.502 | 5.815 | 5 |
| head | auto | cuda | 64 | 7 | clustered | 1 | 5.866 | 5.791 | 6.055 | 5 |
| head | auto | cuda | 64 | 7 | lattice | 0 | 14.604 | 14.264 | 15.224 | 5 |
| head | auto | cuda | 64 | 7 | lattice | 1 | 14.679 | 14.541 | 14.926 | 5 |
| head | auto | cuda | 512 | 7 | adaptive | 0 | 19.480 | 19.157 | 19.810 | 5 |
| head | auto | cuda | 512 | 7 | adaptive | 1 | 20.573 | 20.018 | 20.638 | 5 |
| head | auto | cuda | 512 | 7 | clustered | 0 | 5.283 | 5.126 | 5.400 | 5 |
| head | auto | cuda | 512 | 7 | clustered | 1 | 6.038 | 5.853 | 6.197 | 5 |
| head | auto | cuda | 512 | 7 | lattice | 0 | 7.618 | 7.425 | 7.742 | 5 |
| head | auto | cuda | 512 | 7 | lattice | 1 | 8.322 | 8.055 | 8.555 | 5 |
| shared | auto | cuda | 64 | 7 | adaptive | 0 | 10.332 | 10.241 | 10.754 | 5 |
| shared | auto | cuda | 64 | 7 | adaptive | 1 | 10.474 | 10.317 | 10.656 | 5 |
| shared | auto | cuda | 64 | 7 | clustered | 0 | 5.729 | 5.569 | 5.769 | 5 |
| shared | auto | cuda | 64 | 7 | clustered | 1 | 5.971 | 5.892 | 6.067 | 5 |
| shared | auto | cuda | 64 | 7 | lattice | 0 | 14.339 | 14.296 | 14.505 | 5 |
| shared | auto | cuda | 64 | 7 | lattice | 1 | 14.574 | 14.499 | 14.749 | 5 |
| shared | auto | cuda | 512 | 7 | adaptive | 0 | 19.735 | 19.308 | 19.826 | 5 |
| shared | auto | cuda | 512 | 7 | adaptive | 1 | 20.455 | 19.887 | 20.660 | 5 |
| shared | auto | cuda | 512 | 7 | clustered | 0 | 5.282 | 4.991 | 5.407 | 5 |
| shared | auto | cuda | 512 | 7 | clustered | 1 | 5.987 | 5.830 | 6.152 | 5 |
| shared | auto | cuda | 512 | 7 | lattice | 0 | 7.736 | 7.679 | 7.768 | 5 |
| shared | auto | cuda | 512 | 7 | lattice | 1 | 8.315 | 8.021 | 8.430 | 5 |

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
