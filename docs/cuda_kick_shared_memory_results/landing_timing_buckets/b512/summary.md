| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| merged | auto | cuda | 512 | 7 | adaptive | 0 | 26.744 | 26.430 | 29.196 | 5 |
| merged | auto | cuda | 512 | 7 | adaptive | 1 | 27.479 | 27.288 | 28.024 | 5 |
| merged | auto | cuda | 512 | 7 | clustered | 0 | 5.400 | 5.334 | 5.514 | 5 |
| merged | auto | cuda | 512 | 7 | clustered | 1 | 6.095 | 5.982 | 6.208 | 5 |
| merged | auto | cuda | 512 | 7 | lattice | 0 | 9.574 | 9.249 | 9.708 | 5 |
| merged | auto | cuda | 512 | 7 | lattice | 1 | 10.320 | 10.158 | 10.402 | 5 |
| premerge | auto | cuda | 512 | 7 | adaptive | 0 | 27.103 | 26.363 | 28.732 | 5 |
| premerge | auto | cuda | 512 | 7 | adaptive | 1 | 27.444 | 27.205 | 29.240 | 5 |
| premerge | auto | cuda | 512 | 7 | clustered | 0 | 10.107 | 9.969 | 10.949 | 5 |
| premerge | auto | cuda | 512 | 7 | clustered | 1 | 10.818 | 10.772 | 11.112 | 5 |
| premerge | auto | cuda | 512 | 7 | lattice | 0 | 10.114 | 10.025 | 11.197 | 5 |
| premerge | auto | cuda | 512 | 7 | lattice | 1 | 10.714 | 10.593 | 10.899 | 5 |

| binary | policy | backend | bucket | order | fixture | force rel L2 | force p99 | potential rel L2 | potential p99 |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| premerge | auto | cuda | 512 | 7 | lattice | 3.644e-01 | 2.101e+00 | 4.476e-02 | 1.435e-01 |
| premerge | auto | cuda | 512 | 7 | clustered | 2.146e-03 | 2.165e-02 | 1.120e-03 | 1.132e-02 |
| premerge | auto | cuda | 512 | 7 | adaptive | 1.015e-03 | 6.030e-02 | 1.108e-04 | 2.637e-02 |
| merged | auto | cuda | 512 | 7 | lattice | 3.644e-01 | 2.101e+00 | 4.476e-02 | 1.435e-01 |
| merged | auto | cuda | 512 | 7 | clustered | 2.146e-03 | 2.165e-02 | 1.120e-03 | 1.132e-02 |
| merged | auto | cuda | 512 | 7 | adaptive | 1.015e-03 | 6.030e-02 | 1.108e-04 | 2.637e-02 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| merged | auto | cuda | 512 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 11760 bytes: control 648, expansion shared, forces shared tile 512, particle tile 224 (without mass plane), multipole tile 16 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |
| premerge | auto | cuda | 512 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 11664 bytes: control 528, expansion shared, forces shared tile 512, particle tile 224 (without mass plane), multipole tile 17 x 160 bytes` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
| vs_premerge | merged | auto | 512 | 7 | adaptive | 1.381e-09 | 1.128e-07 | 1.805e-09 |
| vs_premerge | merged | auto | 512 | 7 | clustered | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_premerge | merged | auto | 512 | 7 | lattice | 2.383e-07 | 1.148e-05 | 1.760e-06 |
