| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| arena | handoff164 | cuda | 128 | 7 | adaptive | 0 | 30.232 | 29.953 | 30.511 | 2 |
| arena | handoff164 | cuda | 128 | 7 | adaptive | 1 | 34.377 | 32.639 | 36.114 | 2 |
| arena | handoff164 | cuda | 128 | 7 | clustered | 0 | 19.192 | 18.421 | 19.963 | 2 |
| arena | handoff164 | cuda | 128 | 7 | clustered | 1 | 21.402 | 20.667 | 22.136 | 2 |
| arena | handoff164 | cuda | 128 | 7 | lattice | 0 | 33.000 | 30.896 | 35.103 | 2 |
| arena | handoff164 | cuda | 128 | 7 | lattice | 1 | 33.738 | 32.297 | 35.178 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 0 | 28.504 | 28.185 | 28.823 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 1 | 30.884 | 30.703 | 31.065 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 0 | 18.279 | 18.192 | 18.367 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 1 | 19.429 | 18.278 | 20.579 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 0 | 47.811 | 47.554 | 48.068 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 1 | 48.094 | 48.067 | 48.121 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 0 | 29.440 | 28.221 | 30.658 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 1 | 32.564 | 31.755 | 33.372 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 0 | 17.553 | 16.847 | 18.260 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 1 | 18.879 | 17.885 | 19.874 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 0 | 49.177 | 49.092 | 49.262 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 1 | 47.753 | 47.580 | 47.926 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 0 | 36.267 | 36.200 | 36.333 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 1 | 37.513 | 36.904 | 38.122 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 0 | 20.395 | 19.808 | 20.982 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 1 | 20.747 | 20.553 | 20.941 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 0 | 104.438 | 103.944 | 104.933 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 1 | 105.572 | 104.913 | 106.232 | 2 |
| arena | handoff82 | cuda | 128 | 7 | adaptive | 0 | 44.884 | 44.117 | 45.652 | 2 |
| arena | handoff82 | cuda | 128 | 7 | adaptive | 1 | 48.953 | 47.992 | 49.915 | 2 |
| arena | handoff82 | cuda | 128 | 7 | clustered | 0 | 25.441 | 24.480 | 26.401 | 2 |
| arena | handoff82 | cuda | 128 | 7 | clustered | 1 | 28.571 | 27.492 | 29.650 | 2 |
| arena | handoff82 | cuda | 128 | 7 | lattice | 0 | 38.684 | 38.243 | 39.126 | 2 |
| arena | handoff82 | cuda | 128 | 7 | lattice | 1 | 41.999 | 41.670 | 42.327 | 2 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| arena | handoff164 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff328 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff492 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff656 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff82 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
