| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| arena | handoff164 | cuda | 128 | 7 | adaptive | 0 | 29.354 | 29.340 | 29.369 | 2 |
| arena | handoff164 | cuda | 128 | 7 | adaptive | 1 | 31.209 | 31.153 | 31.266 | 2 |
| arena | handoff164 | cuda | 128 | 7 | clustered | 0 | 18.539 | 18.153 | 18.926 | 2 |
| arena | handoff164 | cuda | 128 | 7 | clustered | 1 | 20.686 | 20.446 | 20.926 | 2 |
| arena | handoff164 | cuda | 128 | 7 | lattice | 0 | 41.158 | 31.262 | 51.054 | 2 |
| arena | handoff164 | cuda | 128 | 7 | lattice | 1 | 33.072 | 32.235 | 33.909 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 0 | 42.221 | 30.297 | 54.144 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 1 | 43.380 | 29.453 | 57.308 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 0 | 18.126 | 18.072 | 18.180 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 1 | 25.915 | 19.414 | 32.416 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 0 | 56.684 | 47.793 | 65.574 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 1 | 56.172 | 48.256 | 64.089 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 0 | 30.980 | 29.824 | 32.137 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 1 | 42.025 | 31.856 | 52.195 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 0 | 18.665 | 18.350 | 18.981 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 1 | 20.705 | 20.601 | 20.808 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 0 | 47.258 | 46.413 | 48.102 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 1 | 49.815 | 48.644 | 50.985 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 0 | 37.514 | 37.025 | 38.003 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 1 | 39.316 | 38.684 | 39.949 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 0 | 20.990 | 20.115 | 21.866 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 1 | 22.522 | 22.041 | 23.003 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 0 | 105.681 | 104.431 | 106.930 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 1 | 107.198 | 106.438 | 107.957 | 2 |
| arena | handoff82 | cuda | 128 | 7 | adaptive | 0 | 42.678 | 42.640 | 42.716 | 2 |
| arena | handoff82 | cuda | 128 | 7 | adaptive | 1 | 46.180 | 46.085 | 46.274 | 2 |
| arena | handoff82 | cuda | 128 | 7 | clustered | 0 | 24.631 | 24.457 | 24.805 | 2 |
| arena | handoff82 | cuda | 128 | 7 | clustered | 1 | 28.356 | 28.131 | 28.580 | 2 |
| arena | handoff82 | cuda | 128 | 7 | lattice | 0 | 35.095 | 34.249 | 35.941 | 2 |
| arena | handoff82 | cuda | 128 | 7 | lattice | 1 | 37.746 | 37.717 | 37.774 | 2 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| arena | handoff164 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff328 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff492 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff656 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff82 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
