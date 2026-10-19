| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| arena | handoff164 | cuda | 128 | 7 | adaptive | 0 | 32.947 | 26.321 | 39.572 | 2 |
| arena | handoff164 | cuda | 128 | 7 | adaptive | 1 | 34.353 | 29.119 | 39.586 | 2 |
| arena | handoff164 | cuda | 128 | 7 | clustered | 0 | 20.299 | 16.326 | 24.272 | 2 |
| arena | handoff164 | cuda | 128 | 7 | clustered | 1 | 23.514 | 18.323 | 28.705 | 2 |
| arena | handoff164 | cuda | 128 | 7 | lattice | 0 | 31.959 | 26.558 | 37.359 | 2 |
| arena | handoff164 | cuda | 128 | 7 | lattice | 1 | 33.076 | 26.670 | 39.483 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 0 | 36.328 | 33.336 | 39.320 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 1 | 40.603 | 37.487 | 43.719 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 0 | 24.193 | 21.949 | 26.437 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 1 | 24.723 | 23.909 | 25.538 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 0 | 51.534 | 44.553 | 58.515 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 1 | 49.891 | 46.041 | 53.741 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 0 | 29.291 | 24.244 | 34.338 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 1 | 31.041 | 25.678 | 36.404 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 0 | 23.042 | 22.644 | 23.440 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 1 | 20.921 | 17.150 | 24.691 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 0 | 42.534 | 36.829 | 48.240 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 1 | 41.321 | 36.379 | 46.263 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 0 | 40.439 | 39.407 | 41.472 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 1 | 37.708 | 36.888 | 38.527 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 0 | 23.512 | 23.501 | 23.522 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 1 | 27.520 | 26.344 | 28.696 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 0 | 70.875 | 60.180 | 81.571 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 1 | 74.061 | 61.651 | 86.471 | 2 |
| arena | handoff82 | cuda | 128 | 7 | adaptive | 0 | 47.886 | 41.302 | 54.471 | 2 |
| arena | handoff82 | cuda | 128 | 7 | adaptive | 1 | 55.491 | 44.631 | 66.350 | 2 |
| arena | handoff82 | cuda | 128 | 7 | clustered | 0 | 29.180 | 23.111 | 35.248 | 2 |
| arena | handoff82 | cuda | 128 | 7 | clustered | 1 | 30.350 | 26.298 | 34.403 | 2 |
| arena | handoff82 | cuda | 128 | 7 | lattice | 0 | 37.712 | 33.421 | 42.003 | 2 |
| arena | handoff82 | cuda | 128 | 7 | lattice | 1 | 40.130 | 35.760 | 44.499 | 2 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| arena | handoff164 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff328 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff492 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff656 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff82 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
