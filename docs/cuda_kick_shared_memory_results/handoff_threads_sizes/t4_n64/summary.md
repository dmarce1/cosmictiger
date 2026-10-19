| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| arena | handoff164 | cuda | 128 | 7 | adaptive | 0 | 27.000 | 26.361 | 27.638 | 2 |
| arena | handoff164 | cuda | 128 | 7 | adaptive | 1 | 30.112 | 29.176 | 31.048 | 2 |
| arena | handoff164 | cuda | 128 | 7 | clustered | 0 | 17.296 | 16.849 | 17.744 | 2 |
| arena | handoff164 | cuda | 128 | 7 | clustered | 1 | 18.506 | 18.034 | 18.978 | 2 |
| arena | handoff164 | cuda | 128 | 7 | lattice | 0 | 26.253 | 25.378 | 27.129 | 2 |
| arena | handoff164 | cuda | 128 | 7 | lattice | 1 | 27.190 | 26.900 | 27.480 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 0 | 22.766 | 22.605 | 22.926 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 1 | 23.649 | 22.814 | 24.485 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 0 | 14.984 | 14.647 | 15.321 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 1 | 16.387 | 16.328 | 16.446 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 0 | 31.502 | 31.253 | 31.752 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 1 | 32.980 | 32.565 | 33.394 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 0 | 23.660 | 23.473 | 23.847 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 1 | 25.907 | 25.776 | 26.039 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 0 | 15.806 | 15.743 | 15.869 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 1 | 17.085 | 16.770 | 17.400 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 0 | 31.715 | 31.442 | 31.987 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 1 | 33.485 | 33.326 | 33.643 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 0 | 25.221 | 24.876 | 25.566 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 1 | 27.657 | 27.138 | 28.176 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 0 | 17.031 | 16.533 | 17.530 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 1 | 17.727 | 17.681 | 17.773 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 0 | 58.095 | 57.909 | 58.281 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 1 | 60.262 | 60.125 | 60.399 | 2 |
| arena | handoff82 | cuda | 128 | 7 | adaptive | 0 | 41.829 | 41.477 | 42.182 | 2 |
| arena | handoff82 | cuda | 128 | 7 | adaptive | 1 | 46.429 | 45.698 | 47.160 | 2 |
| arena | handoff82 | cuda | 128 | 7 | clustered | 0 | 23.242 | 23.095 | 23.389 | 2 |
| arena | handoff82 | cuda | 128 | 7 | clustered | 1 | 27.485 | 27.016 | 27.955 | 2 |
| arena | handoff82 | cuda | 128 | 7 | lattice | 0 | 33.409 | 32.655 | 34.164 | 2 |
| arena | handoff82 | cuda | 128 | 7 | lattice | 1 | 37.004 | 35.849 | 38.160 | 2 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| arena | handoff164 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff328 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff492 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff656 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff82 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
