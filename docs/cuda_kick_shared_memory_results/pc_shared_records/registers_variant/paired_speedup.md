| backend | bucket | fixture | potential | binary | policy | order | kick ms | speedup vs reference | paired range | force rel L2 | potential rel L2 |
|---|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|
| cuda | 64 | adaptive | 0 | head | auto | 7 | 11.335 |  |  | 2.262e-03 | 1.163e-04 |
| cuda | 64 | adaptive | 0 | shared | auto | 7 | 11.459 | 0.990 | 0.988-1.023 | 2.262e-03 | 1.163e-04 |
| cuda | 64 | adaptive | 1 | head | auto | 7 | 11.776 |  |  | 2.262e-03 | 1.163e-04 |
| cuda | 64 | adaptive | 1 | shared | auto | 7 | 12.297 | 1.000 | 0.947-1.015 | 2.262e-03 | 1.163e-04 |
| cuda | 64 | clustered | 0 | head | auto | 7 | 5.551 |  |  | 5.305e-03 | 1.096e-03 |
| cuda | 64 | clustered | 0 | shared | auto | 7 | 5.491 | 1.011 | 0.982-1.043 | 5.305e-03 | 1.096e-03 |
| cuda | 64 | clustered | 1 | head | auto | 7 | 5.665 |  |  | 5.305e-03 | 1.096e-03 |
| cuda | 64 | clustered | 1 | shared | auto | 7 | 5.848 | 1.008 | 0.967-1.019 | 5.305e-03 | 1.096e-03 |
| cuda | 64 | lattice | 0 | head | auto | 7 | 14.594 |  |  | 3.644e-01 | 4.476e-02 |
| cuda | 64 | lattice | 0 | shared | auto | 7 | 14.415 | 1.002 | 0.999-1.027 | 3.644e-01 | 4.476e-02 |
| cuda | 64 | lattice | 1 | head | auto | 7 | 14.703 |  |  | 3.644e-01 | 4.476e-02 |
| cuda | 64 | lattice | 1 | shared | auto | 7 | 14.491 | 1.004 | 0.999-1.065 | 3.644e-01 | 4.476e-02 |
| cuda | 512 | adaptive | 0 | head | auto | 7 | 24.347 |  |  | 1.015e-03 | 1.108e-04 |
| cuda | 512 | adaptive | 0 | shared | auto | 7 | 24.341 | 1.010 | 0.933-1.092 | 1.015e-03 | 1.108e-04 |
| cuda | 512 | adaptive | 1 | head | auto | 7 | 25.053 |  |  | 1.015e-03 | 1.108e-04 |
| cuda | 512 | adaptive | 1 | shared | auto | 7 | 24.934 | 1.006 | 0.944-1.098 | 1.015e-03 | 1.108e-04 |
| cuda | 512 | clustered | 0 | head | auto | 7 | 5.333 |  |  | 2.146e-03 | 1.120e-03 |
| cuda | 512 | clustered | 0 | shared | auto | 7 | 5.364 | 1.004 | 0.973-1.039 | 2.146e-03 | 1.120e-03 |
| cuda | 512 | clustered | 1 | head | auto | 7 | 5.968 |  |  | 2.146e-03 | 1.120e-03 |
| cuda | 512 | clustered | 1 | shared | auto | 7 | 5.880 | 1.017 | 0.972-1.093 | 2.146e-03 | 1.120e-03 |
| cuda | 512 | lattice | 0 | head | auto | 7 | 8.886 |  |  | 3.644e-01 | 4.476e-02 |
| cuda | 512 | lattice | 0 | shared | auto | 7 | 8.873 | 0.999 | 0.964-1.044 | 3.644e-01 | 4.476e-02 |
| cuda | 512 | lattice | 1 | head | auto | 7 | 9.559 |  |  | 3.644e-01 | 4.476e-02 |
| cuda | 512 | lattice | 1 | shared | auto | 7 | 9.349 | 1.022 | 0.946-1.060 | 3.644e-01 | 4.476e-02 |
