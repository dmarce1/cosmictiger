| backend | bucket | fixture | potential | binary | policy | order | kick ms | speedup vs reference | paired range | force rel L2 | potential rel L2 |
|---|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|
| cuda | 512 | adaptive | 0 | merged | auto | 7 | 26.744 | 0.989 | 0.976-1.080 | 1.015e-03 | 1.108e-04 |
| cuda | 512 | adaptive | 0 | premerge | auto | 7 | 27.103 |  |  | 1.015e-03 | 1.108e-04 |
| cuda | 512 | adaptive | 1 | merged | auto | 7 | 27.479 | 0.990 | 0.979-1.072 | 1.015e-03 | 1.108e-04 |
| cuda | 512 | adaptive | 1 | premerge | auto | 7 | 27.444 |  |  | 1.015e-03 | 1.108e-04 |
| cuda | 512 | clustered | 0 | merged | auto | 7 | 5.400 | 1.881 | 1.808-2.028 | 2.146e-03 | 1.120e-03 |
| cuda | 512 | clustered | 0 | premerge | auto | 7 | 10.107 |  |  | 2.146e-03 | 1.120e-03 |
| cuda | 512 | clustered | 1 | merged | auto | 7 | 6.095 | 1.805 | 1.735-1.825 | 2.146e-03 | 1.120e-03 |
| cuda | 512 | clustered | 1 | premerge | auto | 7 | 10.818 |  |  | 2.146e-03 | 1.120e-03 |
| cuda | 512 | lattice | 0 | merged | auto | 7 | 9.574 | 1.054 | 1.042-1.209 | 3.644e-01 | 4.476e-02 |
| cuda | 512 | lattice | 0 | premerge | auto | 7 | 10.114 |  |  | 3.644e-01 | 4.476e-02 |
| cuda | 512 | lattice | 1 | merged | auto | 7 | 10.320 | 1.051 | 1.018-1.055 | 3.644e-01 | 4.476e-02 |
| cuda | 512 | lattice | 1 | premerge | auto | 7 | 10.714 |  |  | 3.644e-01 | 4.476e-02 |
