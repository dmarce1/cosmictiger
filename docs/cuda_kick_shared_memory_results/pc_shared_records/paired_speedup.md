| backend | bucket | fixture | potential | binary | policy | order | kick ms | speedup vs reference | paired range | force rel L2 | potential rel L2 |
|---|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|
| cuda | 64 | adaptive | 0 | head | auto | 7 | 11.381 |  |  | 2.262e-03 | 1.163e-04 |
| cuda | 64 | adaptive | 0 | shared | auto | 7 | 11.408 | 0.998 | 0.905-1.009 | 2.262e-03 | 1.163e-04 |
| cuda | 64 | adaptive | 1 | head | auto | 7 | 11.601 |  |  | 2.262e-03 | 1.163e-04 |
| cuda | 64 | adaptive | 1 | shared | auto | 7 | 11.549 | 1.001 | 0.934-1.009 | 2.262e-03 | 1.163e-04 |
| cuda | 64 | clustered | 0 | head | auto | 7 | 5.401 |  |  | 5.305e-03 | 1.096e-03 |
| cuda | 64 | clustered | 0 | shared | auto | 7 | 5.398 | 1.000 | 0.985-1.053 | 5.305e-03 | 1.096e-03 |
| cuda | 64 | clustered | 1 | head | auto | 7 | 5.709 |  |  | 5.305e-03 | 1.096e-03 |
| cuda | 64 | clustered | 1 | shared | auto | 7 | 5.636 | 0.999 | 0.990-1.029 | 5.305e-03 | 1.096e-03 |
| cuda | 64 | lattice | 0 | head | auto | 7 | 14.342 |  |  | 3.644e-01 | 4.476e-02 |
| cuda | 64 | lattice | 0 | shared | auto | 7 | 14.424 | 0.995 | 0.972-1.017 | 3.644e-01 | 4.476e-02 |
| cuda | 64 | lattice | 1 | head | auto | 7 | 14.415 |  |  | 3.644e-01 | 4.476e-02 |
| cuda | 64 | lattice | 1 | shared | auto | 7 | 14.536 | 0.993 | 0.981-1.030 | 3.644e-01 | 4.476e-02 |
| cuda | 512 | adaptive | 0 | head | auto | 7 | 24.075 |  |  | 1.015e-03 | 1.108e-04 |
| cuda | 512 | adaptive | 0 | shared | auto | 7 | 23.976 | 1.000 | 0.995-1.011 | 1.015e-03 | 1.108e-04 |
| cuda | 512 | adaptive | 1 | head | auto | 7 | 24.624 |  |  | 1.015e-03 | 1.108e-04 |
| cuda | 512 | adaptive | 1 | shared | auto | 7 | 24.637 | 1.000 | 0.984-1.003 | 1.015e-03 | 1.108e-04 |
| cuda | 512 | clustered | 0 | head | auto | 7 | 5.203 |  |  | 2.146e-03 | 1.120e-03 |
| cuda | 512 | clustered | 0 | shared | auto | 7 | 5.129 | 1.005 | 0.951-1.043 | 2.146e-03 | 1.120e-03 |
| cuda | 512 | clustered | 1 | head | auto | 7 | 5.831 |  |  | 2.146e-03 | 1.120e-03 |
| cuda | 512 | clustered | 1 | shared | auto | 7 | 5.856 | 0.992 | 0.949-1.012 | 2.146e-03 | 1.120e-03 |
| cuda | 512 | lattice | 0 | head | auto | 7 | 8.802 |  |  | 3.644e-01 | 4.476e-02 |
| cuda | 512 | lattice | 0 | shared | auto | 7 | 8.872 | 0.996 | 0.948-1.007 | 3.644e-01 | 4.476e-02 |
| cuda | 512 | lattice | 1 | head | auto | 7 | 9.381 |  |  | 3.644e-01 | 4.476e-02 |
| cuda | 512 | lattice | 1 | shared | auto | 7 | 9.328 | 1.008 | 0.947-1.018 | 3.644e-01 | 4.476e-02 |
