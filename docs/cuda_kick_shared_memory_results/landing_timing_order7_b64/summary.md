| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| merged | auto | cuda | 64 | 7 | adaptive | 0 | 12.120 | 11.861 | 12.619 | 5 |
| merged | auto | cuda | 64 | 7 | adaptive | 1 | 12.398 | 12.308 | 12.467 | 5 |
| merged | auto | cuda | 64 | 7 | clustered | 0 | 5.562 | 5.477 | 5.899 | 5 |
| merged | auto | cuda | 64 | 7 | clustered | 1 | 5.900 | 5.769 | 5.918 | 5 |
| merged | auto | cuda | 64 | 7 | lattice | 0 | 14.616 | 14.492 | 14.677 | 5 |
| merged | auto | cuda | 64 | 7 | lattice | 1 | 14.652 | 14.571 | 14.794 | 5 |
| premerge | auto | cuda | 64 | 7 | adaptive | 0 | 12.299 | 12.028 | 12.356 | 5 |
| premerge | auto | cuda | 64 | 7 | adaptive | 1 | 12.397 | 12.326 | 12.476 | 5 |
| premerge | auto | cuda | 64 | 7 | clustered | 0 | 6.256 | 5.998 | 6.324 | 5 |
| premerge | auto | cuda | 64 | 7 | clustered | 1 | 6.154 | 6.107 | 6.202 | 5 |
| premerge | auto | cuda | 64 | 7 | lattice | 0 | 15.271 | 15.035 | 15.483 | 5 |
| premerge | auto | cuda | 64 | 7 | lattice | 1 | 15.170 | 15.076 | 15.515 | 5 |

| binary | policy | backend | bucket | order | fixture | force rel L2 | force p99 | potential rel L2 | potential p99 |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| premerge | auto | cuda | 64 | 7 | lattice | 3.644e-01 | 2.101e+00 | 4.476e-02 | 1.435e-01 |
| premerge | auto | cuda | 64 | 7 | clustered | 5.305e-03 | 2.563e-02 | 1.096e-03 | 1.145e-02 |
| premerge | auto | cuda | 64 | 7 | adaptive | 2.262e-03 | 5.030e-02 | 1.163e-04 | 2.580e-02 |
| merged | auto | cuda | 64 | 7 | lattice | 3.644e-01 | 2.101e+00 | 4.476e-02 | 1.435e-01 |
| merged | auto | cuda | 64 | 7 | clustered | 5.305e-03 | 2.563e-02 | 1.096e-03 | 1.145e-02 |
| merged | auto | cuda | 64 | 7 | adaptive | 2.262e-03 | 5.030e-02 | 1.163e-04 | 2.580e-02 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| merged | auto | cuda | 64 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8048 bytes: control 648, expansion shared, forces shared tile 64, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |
| premerge | auto | cuda | 64 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 7920 bytes: control 528, expansion shared, forces shared tile 64, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
| vs_premerge | merged | auto | 64 | 7 | adaptive | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_premerge | merged | auto | 64 | 7 | clustered | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_premerge | merged | auto | 64 | 7 | lattice | 0.000e+00 | 0.000e+00 | 0.000e+00 |
