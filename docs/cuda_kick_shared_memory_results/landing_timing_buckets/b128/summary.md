| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| merged | auto | cuda | 128 | 7 | adaptive | 0 | 12.477 | 11.833 | 12.646 | 5 |
| merged | auto | cuda | 128 | 7 | adaptive | 1 | 12.725 | 12.669 | 13.041 | 5 |
| merged | auto | cuda | 128 | 7 | clustered | 0 | 6.262 | 6.138 | 6.662 | 5 |
| merged | auto | cuda | 128 | 7 | clustered | 1 | 6.529 | 6.464 | 6.565 | 5 |
| merged | auto | cuda | 128 | 7 | lattice | 0 | 15.208 | 15.126 | 15.445 | 5 |
| merged | auto | cuda | 128 | 7 | lattice | 1 | 15.308 | 15.206 | 15.510 | 5 |
| premerge | auto | cuda | 128 | 7 | adaptive | 0 | 12.668 | 12.561 | 12.761 | 5 |
| premerge | auto | cuda | 128 | 7 | adaptive | 1 | 12.725 | 12.683 | 12.806 | 5 |
| premerge | auto | cuda | 128 | 7 | clustered | 0 | 6.837 | 6.774 | 6.947 | 5 |
| premerge | auto | cuda | 128 | 7 | clustered | 1 | 7.075 | 7.070 | 7.264 | 5 |
| premerge | auto | cuda | 128 | 7 | lattice | 0 | 15.337 | 15.020 | 15.652 | 5 |
| premerge | auto | cuda | 128 | 7 | lattice | 1 | 15.253 | 14.969 | 15.887 | 5 |

| binary | policy | backend | bucket | order | fixture | force rel L2 | force p99 | potential rel L2 | potential p99 |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| premerge | auto | cuda | 128 | 7 | lattice | 3.644e-01 | 2.101e+00 | 4.476e-02 | 1.435e-01 |
| premerge | auto | cuda | 128 | 7 | clustered | 2.962e-03 | 1.798e-02 | 1.017e-03 | 1.036e-02 |
| premerge | auto | cuda | 128 | 7 | adaptive | 1.869e-03 | 6.030e-02 | 1.191e-04 | 2.637e-02 |
| merged | auto | cuda | 128 | 7 | lattice | 3.644e-01 | 2.101e+00 | 4.476e-02 | 1.435e-01 |
| merged | auto | cuda | 128 | 7 | clustered | 2.962e-03 | 1.798e-02 | 1.017e-03 | 1.036e-02 |
| merged | auto | cuda | 128 | 7 | adaptive | 1.869e-03 | 6.030e-02 | 1.191e-04 | 2.637e-02 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| merged | auto | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 9072 bytes: control 648, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |
| premerge | auto | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
| vs_premerge | merged | auto | 128 | 7 | adaptive | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_premerge | merged | auto | 128 | 7 | clustered | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_premerge | merged | auto | 128 | 7 | lattice | 0.000e+00 | 0.000e+00 | 0.000e+00 |
