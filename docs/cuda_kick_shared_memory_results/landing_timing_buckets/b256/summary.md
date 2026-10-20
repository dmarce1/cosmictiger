| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| merged | auto | cuda | 256 | 7 | adaptive | 0 | 17.411 | 17.298 | 17.917 | 5 |
| merged | auto | cuda | 256 | 7 | adaptive | 1 | 18.403 | 17.541 | 18.872 | 5 |
| merged | auto | cuda | 256 | 7 | clustered | 0 | 5.715 | 5.645 | 5.824 | 5 |
| merged | auto | cuda | 256 | 7 | clustered | 1 | 6.091 | 6.062 | 6.227 | 5 |
| merged | auto | cuda | 256 | 7 | lattice | 0 | 10.612 | 10.592 | 10.785 | 5 |
| merged | auto | cuda | 256 | 7 | lattice | 1 | 11.104 | 10.764 | 11.407 | 5 |
| premerge | auto | cuda | 256 | 7 | adaptive | 0 | 17.285 | 17.132 | 17.761 | 5 |
| premerge | auto | cuda | 256 | 7 | adaptive | 1 | 17.916 | 17.798 | 18.497 | 5 |
| premerge | auto | cuda | 256 | 7 | clustered | 0 | 7.013 | 6.992 | 7.085 | 5 |
| premerge | auto | cuda | 256 | 7 | clustered | 1 | 7.460 | 7.373 | 8.184 | 5 |
| premerge | auto | cuda | 256 | 7 | lattice | 0 | 10.645 | 10.566 | 11.406 | 5 |
| premerge | auto | cuda | 256 | 7 | lattice | 1 | 10.911 | 10.777 | 11.485 | 5 |

| binary | policy | backend | bucket | order | fixture | force rel L2 | force p99 | potential rel L2 | potential p99 |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| premerge | auto | cuda | 256 | 7 | lattice | 3.644e-01 | 2.101e+00 | 4.476e-02 | 1.435e-01 |
| premerge | auto | cuda | 256 | 7 | clustered | 2.677e-03 | 2.816e-02 | 1.153e-03 | 1.118e-02 |
| premerge | auto | cuda | 256 | 7 | adaptive | 1.295e-03 | 6.030e-02 | 1.114e-04 | 2.637e-02 |
| merged | auto | cuda | 256 | 7 | lattice | 3.644e-01 | 2.101e+00 | 4.476e-02 | 1.435e-01 |
| merged | auto | cuda | 256 | 7 | clustered | 2.677e-03 | 2.816e-02 | 1.153e-03 | 1.118e-02 |
| merged | auto | cuda | 256 | 7 | adaptive | 1.295e-03 | 6.030e-02 | 1.114e-04 | 2.637e-02 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| merged | auto | cuda | 256 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 11120 bytes: control 648, expansion shared, forces shared tile 256, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes; list slab 145792 bytes per block (4096 entries per list)` |
| premerge | auto | cuda | 256 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 10992 bytes: control 528, expansion shared, forces shared tile 256, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
| vs_premerge | merged | auto | 256 | 7 | adaptive | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_premerge | merged | auto | 256 | 7 | clustered | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| vs_premerge | merged | auto | 256 | 7 | lattice | 0.000e+00 | 0.000e+00 | 0.000e+00 |
