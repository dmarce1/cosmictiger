| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| arena | handoff164 | cuda | 128 | 7 | adaptive | 0 | 26.449 | 25.476 | 27.422 | 2 |
| arena | handoff164 | cuda | 128 | 7 | adaptive | 1 | 28.112 | 27.929 | 28.296 | 2 |
| arena | handoff164 | cuda | 128 | 7 | clustered | 0 | 17.194 | 16.224 | 18.163 | 2 |
| arena | handoff164 | cuda | 128 | 7 | clustered | 1 | 18.394 | 18.087 | 18.701 | 2 |
| arena | handoff164 | cuda | 128 | 7 | lattice | 0 | 26.012 | 25.917 | 26.107 | 2 |
| arena | handoff164 | cuda | 128 | 7 | lattice | 1 | 27.603 | 26.781 | 28.424 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 0 | 23.303 | 22.922 | 23.684 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 1 | 25.836 | 25.492 | 26.180 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 0 | 16.458 | 15.819 | 17.098 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 1 | 16.962 | 16.565 | 17.360 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 0 | 31.447 | 30.730 | 32.165 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 1 | 32.918 | 31.078 | 34.757 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 0 | 25.271 | 23.933 | 26.608 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 1 | 26.085 | 26.008 | 26.163 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 0 | 16.056 | 14.167 | 17.944 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 1 | 17.099 | 16.613 | 17.584 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 0 | 33.447 | 31.475 | 35.419 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 1 | 32.912 | 31.719 | 34.105 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 0 | 27.051 | 26.945 | 27.157 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 1 | 28.182 | 28.131 | 28.232 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 0 | 16.383 | 15.903 | 16.863 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 1 | 18.178 | 16.883 | 19.473 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 0 | 59.505 | 58.017 | 60.992 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 1 | 61.515 | 59.842 | 63.189 | 2 |
| arena | handoff82 | cuda | 128 | 7 | adaptive | 0 | 43.129 | 41.883 | 44.375 | 2 |
| arena | handoff82 | cuda | 128 | 7 | adaptive | 1 | 44.615 | 43.449 | 45.782 | 2 |
| arena | handoff82 | cuda | 128 | 7 | clustered | 0 | 23.046 | 22.790 | 23.302 | 2 |
| arena | handoff82 | cuda | 128 | 7 | clustered | 1 | 25.672 | 25.661 | 25.683 | 2 |
| arena | handoff82 | cuda | 128 | 7 | lattice | 0 | 32.726 | 31.382 | 34.070 | 2 |
| arena | handoff82 | cuda | 128 | 7 | lattice | 1 | 37.412 | 37.384 | 37.440 | 2 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| arena | handoff164 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff328 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff492 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff656 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff82 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
