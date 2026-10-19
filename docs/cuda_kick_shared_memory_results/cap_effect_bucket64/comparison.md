| backend | bucket | fixture | potential | binary | policy | order | kick ms | speedup vs reference | paired range | force rel L2 | potential rel L2 |
|---|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|
| cuda | 64 | adaptive | 0 | arena_cap1024 | auto | 7 | 20.516 | 0.945 | 0.924-0.948 |  |  |
| cuda | 64 | adaptive | 0 | arena_cap128 | auto | 7 | 20.986 | 0.923 | 0.917-0.937 |  |  |
| cuda | 64 | adaptive | 0 | baseline | auto | 7 | 19.396 |  |  |  |  |
| cuda | 64 | adaptive | 1 | arena_cap1024 | auto | 7 | 20.806 | 0.936 | 0.926-0.938 |  |  |
| cuda | 64 | adaptive | 1 | arena_cap128 | auto | 7 | 21.038 | 0.933 | 0.902-0.944 |  |  |
| cuda | 64 | adaptive | 1 | baseline | auto | 7 | 19.472 |  |  |  |  |
| cuda | 64 | clustered | 0 | arena_cap1024 | auto | 7 | 8.678 | 1.021 | 1.017-1.067 |  |  |
| cuda | 64 | clustered | 0 | arena_cap128 | auto | 7 | 8.441 | 1.057 | 1.045-1.069 |  |  |
| cuda | 64 | clustered | 0 | baseline | auto | 7 | 8.865 |  |  |  |  |
| cuda | 64 | clustered | 1 | arena_cap1024 | auto | 7 | 8.753 | 1.039 | 1.010-1.049 |  |  |
| cuda | 64 | clustered | 1 | arena_cap128 | auto | 7 | 8.638 | 1.049 | 1.008-1.062 |  |  |
| cuda | 64 | clustered | 1 | baseline | auto | 7 | 8.977 |  |  |  |  |
| cuda | 64 | lattice | 0 | arena_cap1024 | auto | 7 | 31.635 | 1.037 | 1.037-1.049 |  |  |
| cuda | 64 | lattice | 0 | arena_cap128 | auto | 7 | 31.875 | 1.037 | 1.030-1.038 |  |  |
| cuda | 64 | lattice | 0 | baseline | auto | 7 | 33.062 |  |  |  |  |
| cuda | 64 | lattice | 1 | arena_cap1024 | auto | 7 | 31.665 | 1.037 | 1.023-1.049 |  |  |
| cuda | 64 | lattice | 1 | arena_cap128 | auto | 7 | 32.120 | 1.031 | 1.023-1.038 |  |  |
| cuda | 64 | lattice | 1 | baseline | auto | 7 | 32.940 |  |  |  |  |
