| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| arena | handoff164 | cuda | 128 | 7 | adaptive | 0 | 27.080 | 26.874 | 27.285 | 2 |
| arena | handoff164 | cuda | 128 | 7 | adaptive | 1 | 29.191 | 28.446 | 29.935 | 2 |
| arena | handoff164 | cuda | 128 | 7 | clustered | 0 | 17.429 | 16.989 | 17.869 | 2 |
| arena | handoff164 | cuda | 128 | 7 | clustered | 1 | 19.879 | 19.774 | 19.983 | 2 |
| arena | handoff164 | cuda | 128 | 7 | lattice | 0 | 23.219 | 23.060 | 23.378 | 2 |
| arena | handoff164 | cuda | 128 | 7 | lattice | 1 | 24.799 | 24.710 | 24.887 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 0 | 21.679 | 21.234 | 22.123 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 1 | 23.872 | 23.709 | 24.035 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 0 | 15.979 | 15.780 | 16.177 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 1 | 17.253 | 17.215 | 17.291 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 0 | 22.948 | 22.440 | 23.456 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 1 | 22.591 | 22.558 | 22.624 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 0 | 22.969 | 22.439 | 23.498 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 1 | 24.213 | 23.935 | 24.492 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 0 | 15.306 | 15.059 | 15.554 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 1 | 17.059 | 16.877 | 17.240 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 0 | 22.157 | 22.056 | 22.258 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 1 | 21.766 | 21.568 | 21.965 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 0 | 21.418 | 21.113 | 21.722 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 1 | 24.112 | 22.913 | 25.311 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 0 | 17.050 | 16.804 | 17.296 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 1 | 18.173 | 17.882 | 18.464 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 0 | 32.375 | 31.861 | 32.889 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 1 | 33.873 | 33.499 | 34.247 | 2 |
| arena | handoff82 | cuda | 128 | 7 | adaptive | 0 | 43.575 | 43.323 | 43.827 | 2 |
| arena | handoff82 | cuda | 128 | 7 | adaptive | 1 | 47.022 | 46.039 | 48.006 | 2 |
| arena | handoff82 | cuda | 128 | 7 | clustered | 0 | 23.989 | 23.825 | 24.153 | 2 |
| arena | handoff82 | cuda | 128 | 7 | clustered | 1 | 27.537 | 27.105 | 27.969 | 2 |
| arena | handoff82 | cuda | 128 | 7 | lattice | 0 | 34.899 | 34.377 | 35.421 | 2 |
| arena | handoff82 | cuda | 128 | 7 | lattice | 1 | 37.513 | 37.493 | 37.532 | 2 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| arena | handoff164 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff328 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff492 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff656 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff82 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
