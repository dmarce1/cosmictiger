| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| arena | handoff164 | cuda | 128 | 7 | adaptive | 0 | 25.498 | 24.740 | 26.255 | 2 |
| arena | handoff164 | cuda | 128 | 7 | adaptive | 1 | 27.335 | 26.378 | 28.292 | 2 |
| arena | handoff164 | cuda | 128 | 7 | clustered | 0 | 15.928 | 15.086 | 16.770 | 2 |
| arena | handoff164 | cuda | 128 | 7 | clustered | 1 | 18.183 | 17.417 | 18.949 | 2 |
| arena | handoff164 | cuda | 128 | 7 | lattice | 0 | 23.013 | 22.989 | 23.037 | 2 |
| arena | handoff164 | cuda | 128 | 7 | lattice | 1 | 23.823 | 23.018 | 24.628 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 0 | 21.291 | 20.468 | 22.114 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 1 | 21.897 | 21.730 | 22.065 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 0 | 14.571 | 14.284 | 14.858 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 1 | 15.843 | 15.636 | 16.050 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 0 | 23.499 | 23.397 | 23.601 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 1 | 24.257 | 23.795 | 24.719 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 0 | 21.835 | 21.716 | 21.955 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 1 | 22.664 | 22.627 | 22.701 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 0 | 14.394 | 14.069 | 14.718 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 1 | 15.781 | 15.694 | 15.869 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 0 | 23.703 | 23.369 | 24.036 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 1 | 24.825 | 24.732 | 24.918 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 0 | 22.469 | 22.427 | 22.510 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 1 | 24.151 | 23.765 | 24.538 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 0 | 15.531 | 15.155 | 15.906 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 1 | 17.016 | 16.917 | 17.115 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 0 | 36.560 | 36.419 | 36.700 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 1 | 38.639 | 37.915 | 39.363 | 2 |
| arena | handoff82 | cuda | 128 | 7 | adaptive | 0 | 40.075 | 38.319 | 41.831 | 2 |
| arena | handoff82 | cuda | 128 | 7 | adaptive | 1 | 43.703 | 42.563 | 44.843 | 2 |
| arena | handoff82 | cuda | 128 | 7 | clustered | 0 | 23.256 | 21.538 | 24.973 | 2 |
| arena | handoff82 | cuda | 128 | 7 | clustered | 1 | 25.739 | 24.748 | 26.730 | 2 |
| arena | handoff82 | cuda | 128 | 7 | lattice | 0 | 32.938 | 31.525 | 34.351 | 2 |
| arena | handoff82 | cuda | 128 | 7 | lattice | 1 | 35.089 | 34.059 | 36.120 | 2 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| arena | handoff164 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff328 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff492 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff656 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff82 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
