| binary | policy | backend | bucket | order | fixture | potential | kick ms (median) | min | max | pairs |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| arena | handoff164 | cuda | 128 | 7 | adaptive | 0 | 13.919 | 13.892 | 13.947 | 2 |
| arena | handoff164 | cuda | 128 | 7 | adaptive | 1 | 14.088 | 13.989 | 14.186 | 2 |
| arena | handoff164 | cuda | 128 | 7 | clustered | 0 | 5.737 | 5.597 | 5.878 | 2 |
| arena | handoff164 | cuda | 128 | 7 | clustered | 1 | 5.943 | 5.896 | 5.990 | 2 |
| arena | handoff164 | cuda | 128 | 7 | lattice | 0 | 9.995 | 9.931 | 10.060 | 2 |
| arena | handoff164 | cuda | 128 | 7 | lattice | 1 | 9.842 | 9.742 | 9.942 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 0 | 13.658 | 13.276 | 14.041 | 2 |
| arena | handoff328 | cuda | 128 | 7 | adaptive | 1 | 13.561 | 13.519 | 13.602 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 0 | 5.669 | 5.531 | 5.806 | 2 |
| arena | handoff328 | cuda | 128 | 7 | clustered | 1 | 5.966 | 5.853 | 6.079 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 0 | 9.955 | 9.720 | 10.191 | 2 |
| arena | handoff328 | cuda | 128 | 7 | lattice | 1 | 10.134 | 9.987 | 10.281 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 0 | 14.129 | 13.948 | 14.311 | 2 |
| arena | handoff492 | cuda | 128 | 7 | adaptive | 1 | 13.998 | 13.868 | 14.128 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 0 | 5.715 | 5.686 | 5.745 | 2 |
| arena | handoff492 | cuda | 128 | 7 | clustered | 1 | 5.989 | 5.814 | 6.164 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 0 | 9.941 | 9.799 | 10.083 | 2 |
| arena | handoff492 | cuda | 128 | 7 | lattice | 1 | 10.278 | 10.016 | 10.540 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 0 | 14.421 | 13.840 | 15.002 | 2 |
| arena | handoff656 | cuda | 128 | 7 | adaptive | 1 | 13.946 | 13.887 | 14.005 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 0 | 5.670 | 5.630 | 5.710 | 2 |
| arena | handoff656 | cuda | 128 | 7 | clustered | 1 | 5.961 | 5.935 | 5.988 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 0 | 10.028 | 9.861 | 10.195 | 2 |
| arena | handoff656 | cuda | 128 | 7 | lattice | 1 | 10.301 | 10.169 | 10.433 | 2 |
| arena | handoff82 | cuda | 128 | 7 | adaptive | 0 | 18.854 | 18.783 | 18.925 | 2 |
| arena | handoff82 | cuda | 128 | 7 | adaptive | 1 | 19.591 | 19.456 | 19.725 | 2 |
| arena | handoff82 | cuda | 128 | 7 | clustered | 0 | 5.885 | 5.645 | 6.125 | 2 |
| arena | handoff82 | cuda | 128 | 7 | clustered | 1 | 6.226 | 6.204 | 6.248 | 2 |
| arena | handoff82 | cuda | 128 | 7 | lattice | 0 | 9.674 | 9.354 | 9.994 | 2 |
| arena | handoff82 | cuda | 128 | 7 | lattice | 1 | 9.849 | 9.840 | 9.857 | 2 |

| binary | policy | backend | bucket | order | kernel policy line |
|---|---|---|---:|---:|---|
| arena | handoff164 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff328 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff492 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff656 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |
| arena | handoff82 | cuda | 128 | 7 | `CUDA kick arena (FMM order 7): 255 registers, 0 static shared bytes, 8 blocks/SM (register limit 8), budget 11776, dynamic shared 8944 bytes: control 528, expansion shared, forces shared tile 128, particle tile 512 (without mass plane), multipole tile 38 x 160 bytes` |

| comparison | binary | policy | bucket | order | fixture | acceleration rel L2 | acceleration max pointwise | potential rel L2 |
|---|---|---|---:|---:|---|---:|---:|---:|
