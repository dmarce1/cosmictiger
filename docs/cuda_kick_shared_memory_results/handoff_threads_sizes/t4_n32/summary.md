| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| arena | handoff164 | cuda | 128 | 7 | adaptive | 0 | 13.849 | 13.410 | 14.288 | 2 |
| arena | handoff164 | cuda | 128 | 7 | adaptive | 1 | 13.358 | 13.196 | 13.520 | 2 |
| arena | handoff164 | cuda | 128 | 7 | clustered | 0 | 5.967 | 5.832 | 6.101 | 2 |
| arena | handoff164 | cuda | 128 | 7 | clustered | 1 | 6.402 | 6.361 | 6.442 | 2 |
| arena | handoff164 | cuda | 128 | 7 | lattice | 0 | 11.828 | 11.693 | 11.964 | 2 |
| arena | handoff164 | cuda | 128 | 7 | lattice | 1 | 11.869 | 11.690 | 12.048 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 0 | 15.999 | 14.993 | 17.005 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 1 | 16.004 | 15.221 | 16.787 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 0 | 5.989 | 5.962 | 6.016 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 1 | 6.446 | 6.283 | 6.608 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 0 | 11.844 | 11.446 | 12.242 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 1 | 11.913 | 11.692 | 12.134 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 0 | 16.564 | 16.358 | 16.771 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 1 | 15.389 | 14.962 | 15.815 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 0 | 6.198 | 5.994 | 6.403 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 1 | 6.547 | 6.418 | 6.676 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 0 | 12.156 | 11.896 | 12.416 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 1 | 11.674 | 11.576 | 11.772 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 0 | 15.531 | 15.447 | 15.614 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 1 | 15.852 | 15.579 | 16.125 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 0 | 6.012 | 5.897 | 6.128 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 1 | 6.565 | 6.364 | 6.766 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 0 | 12.058 | 11.418 | 12.697 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 1 | 11.824 | 11.616 | 12.031 | 2 |
| arena | handoff82 | cuda | 128 | 7 | adaptive | 0 | 18.877 | 18.208 | 19.546 | 2 |
| arena | handoff82 | cuda | 128 | 7 | adaptive | 1 | 19.392 | 18.476 | 20.307 | 2 |
| arena | handoff82 | cuda | 128 | 7 | clustered | 0 | 5.700 | 5.618 | 5.783 | 2 |
| arena | handoff82 | cuda | 128 | 7 | clustered | 1 | 6.125 | 6.051 | 6.199 | 2 |
| arena | handoff82 | cuda | 128 | 7 | lattice | 0 | 10.561 | 9.998 | 11.123 | 2 |
| arena | handoff82 | cuda | 128 | 7 | lattice | 1 | 10.927 | 10.898 | 10.956 | 2 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| arena | handoff164 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff328 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff492 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff656 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff82 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
