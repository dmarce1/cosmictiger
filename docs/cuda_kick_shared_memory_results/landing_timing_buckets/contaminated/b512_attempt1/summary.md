| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| merged | auto | cuda | 512 | 7 | adaptive | 0 | 26.701 | 26.160 | 26.727 | 5 |
| merged | auto | cuda | 512 | 7 | adaptive | 1 | 27.044 | 26.638 | 28.605 | 5 |
| merged | auto | cuda | 512 | 7 | clustered | 0 | 5.385 | 5.352 | 5.571 | 5 |
| merged | auto | cuda | 512 | 7 | clustered | 1 | 6.185 | 6.009 | 6.312 | 5 |
| merged | auto | cuda | 512 | 7 | lattice | 0 | 9.700 | 9.324 | 9.985 | 5 |
| merged | auto | cuda | 512 | 7 | lattice | 1 | 10.298 | 10.172 | 10.370 | 5 |
| premerge | auto | cuda | 512 | 7 | adaptive | 0 | 26.581 | 26.072 | 26.705 | 5 |
| premerge | auto | cuda | 512 | 7 | adaptive | 1 | 27.037 | 26.917 | 29.704 | 5 |
| premerge | auto | cuda | 512 | 7 | clustered | 0 | 10.054 | 9.994 | 10.849 | 5 |
| premerge | auto | cuda | 512 | 7 | clustered | 1 | 11.014 | 10.739 | 11.376 | 5 |
| premerge | auto | cuda | 512 | 7 | lattice | 0 | 10.078 | 9.807 | 11.425 | 5 |
| premerge | auto | cuda | 512 | 7 | lattice | 1 | 10.797 | 10.670 | 11.373 | 5 |

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
