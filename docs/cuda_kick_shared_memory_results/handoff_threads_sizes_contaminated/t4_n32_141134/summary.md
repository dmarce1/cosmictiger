| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| arena | handoff164 | cuda | 128 | 7 | adaptive | 0 | 22.027 | 17.097 | 26.957 | 2 |
| arena | handoff164 | cuda | 128 | 7 | adaptive | 1 | 24.271 | 22.817 | 25.725 | 2 |
| arena | handoff164 | cuda | 128 | 7 | clustered | 0 | 11.485 | 11.034 | 11.937 | 2 |
| arena | handoff164 | cuda | 128 | 7 | clustered | 1 | 12.307 | 11.796 | 12.817 | 2 |
| arena | handoff164 | cuda | 128 | 7 | lattice | 0 | 19.109 | 17.919 | 20.298 | 2 |
| arena | handoff164 | cuda | 128 | 7 | lattice | 1 | 18.229 | 18.198 | 18.260 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 0 | 20.360 | 19.156 | 21.563 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 1 | 22.622 | 22.306 | 22.937 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 0 | 10.416 | 9.851 | 10.981 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 1 | 10.308 | 8.704 | 11.911 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 0 | 19.119 | 18.682 | 19.555 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 1 | 18.270 | 18.028 | 18.511 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 0 | 23.984 | 23.365 | 24.603 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 1 | 21.509 | 20.939 | 22.078 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 0 | 9.388 | 8.907 | 9.869 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 1 | 11.436 | 10.489 | 12.383 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 0 | 19.434 | 18.104 | 20.763 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 1 | 19.427 | 19.179 | 19.675 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 0 | 21.105 | 20.901 | 21.309 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 1 | 21.767 | 21.463 | 22.070 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 0 | 10.406 | 10.269 | 10.542 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 1 | 10.170 | 10.133 | 10.207 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 0 | 18.397 | 18.074 | 18.719 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 1 | 18.377 | 18.296 | 18.457 | 2 |
| arena | handoff82 | cuda | 128 | 7 | adaptive | 0 | 25.521 | 24.407 | 26.634 | 2 |
| arena | handoff82 | cuda | 128 | 7 | adaptive | 1 | 24.669 | 21.977 | 27.362 | 2 |
| arena | handoff82 | cuda | 128 | 7 | clustered | 0 | 11.187 | 11.131 | 11.243 | 2 |
| arena | handoff82 | cuda | 128 | 7 | clustered | 1 | 9.579 | 9.046 | 10.112 | 2 |
| arena | handoff82 | cuda | 128 | 7 | lattice | 0 | 15.789 | 13.361 | 18.216 | 2 |
| arena | handoff82 | cuda | 128 | 7 | lattice | 1 | 18.639 | 17.655 | 19.622 | 2 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| arena | handoff164 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff328 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff492 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff656 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff82 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
