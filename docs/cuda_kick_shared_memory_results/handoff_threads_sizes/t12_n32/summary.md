| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| arena | handoff164 | cuda | 128 | 7 | adaptive | 0 | 13.867 | 13.371 | 14.362 | 2 |
| arena | handoff164 | cuda | 128 | 7 | adaptive | 1 | 13.995 | 13.771 | 14.220 | 2 |
| arena | handoff164 | cuda | 128 | 7 | clustered | 0 | 6.322 | 6.161 | 6.484 | 2 |
| arena | handoff164 | cuda | 128 | 7 | clustered | 1 | 6.182 | 6.078 | 6.286 | 2 |
| arena | handoff164 | cuda | 128 | 7 | lattice | 0 | 9.862 | 9.755 | 9.968 | 2 |
| arena | handoff164 | cuda | 128 | 7 | lattice | 1 | 9.564 | 9.510 | 9.619 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 0 | 14.046 | 13.863 | 14.229 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 1 | 14.406 | 13.962 | 14.850 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 0 | 5.998 | 5.862 | 6.133 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 1 | 6.280 | 6.155 | 6.405 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 0 | 10.288 | 10.018 | 10.558 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 1 | 9.769 | 9.725 | 9.813 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 0 | 13.784 | 13.756 | 13.812 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 1 | 14.974 | 14.916 | 15.033 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 0 | 6.090 | 6.070 | 6.110 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 1 | 6.174 | 6.164 | 6.185 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 0 | 9.754 | 9.621 | 9.887 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 1 | 9.640 | 9.602 | 9.679 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 0 | 14.232 | 14.106 | 14.358 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 1 | 14.429 | 13.909 | 14.948 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 0 | 6.071 | 6.046 | 6.096 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 1 | 6.412 | 6.277 | 6.548 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 0 | 10.283 | 10.046 | 10.521 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 1 | 9.517 | 9.430 | 9.604 | 2 |
| arena | handoff82 | cuda | 128 | 7 | adaptive | 0 | 19.474 | 19.435 | 19.512 | 2 |
| arena | handoff82 | cuda | 128 | 7 | adaptive | 1 | 19.862 | 19.663 | 20.062 | 2 |
| arena | handoff82 | cuda | 128 | 7 | clustered | 0 | 6.333 | 6.302 | 6.363 | 2 |
| arena | handoff82 | cuda | 128 | 7 | clustered | 1 | 6.588 | 6.518 | 6.657 | 2 |
| arena | handoff82 | cuda | 128 | 7 | lattice | 0 | 10.393 | 10.252 | 10.535 | 2 |
| arena | handoff82 | cuda | 128 | 7 | lattice | 1 | 9.815 | 9.739 | 9.891 | 2 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| arena | handoff164 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff328 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff492 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff656 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff82 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
