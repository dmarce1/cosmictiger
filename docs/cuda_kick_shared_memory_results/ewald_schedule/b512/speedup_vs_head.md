| backend | bucket | fixture | potential | binary | policy | order | kick ms | speedup vs reference | paired range | force rel L2 | potential rel L2 |
|---|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|
| cuda | 512 | adaptive | 0 | baseline | auto | 7 | 28.442 | 0.920 | 0.908-0.947 | 1.001e-03 | 9.754e-05 |
| cuda | 512 | adaptive | 0 | ewald | auto | 7 | 19.649 | 1.335 | 1.332-1.374 | 1.015e-03 | 1.108e-04 |
| cuda | 512 | adaptive | 0 | head | auto | 7 | 26.385 |  |  | 1.015e-03 | 1.108e-04 |
| cuda | 512 | adaptive | 1 | baseline | auto | 7 | 29.607 | 0.914 | 0.906-0.926 | 1.001e-03 | 9.754e-05 |
| cuda | 512 | adaptive | 1 | ewald | auto | 7 | 20.575 | 1.318 | 1.303-1.352 | 1.015e-03 | 1.108e-04 |
| cuda | 512 | adaptive | 1 | head | auto | 7 | 27.187 |  |  | 1.015e-03 | 1.108e-04 |
| cuda | 512 | clustered | 0 | baseline | auto | 7 | 9.651 | 0.560 | 0.557-0.567 | 1.767e-03 | 5.584e-04 |
| cuda | 512 | clustered | 0 | ewald | auto | 7 | 5.223 | 1.043 | 1.002-1.077 | 2.146e-03 | 1.120e-03 |
| cuda | 512 | clustered | 0 | head | auto | 7 | 5.469 |  |  | 2.146e-03 | 1.120e-03 |
| cuda | 512 | clustered | 1 | baseline | auto | 7 | 10.432 | 0.589 | 0.563-0.612 | 1.767e-03 | 5.584e-04 |
| cuda | 512 | clustered | 1 | ewald | auto | 7 | 6.042 | 1.011 | 0.986-1.046 | 2.146e-03 | 1.120e-03 |
| cuda | 512 | clustered | 1 | head | auto | 7 | 6.086 |  |  | 2.146e-03 | 1.120e-03 |
| cuda | 512 | lattice | 0 | baseline | auto | 7 | 10.340 | 0.893 | 0.823-0.906 | 3.418e-01 | 3.613e-02 |
| cuda | 512 | lattice | 0 | ewald | auto | 7 | 7.630 | 1.230 | 1.193-1.271 | 3.644e-01 | 4.476e-02 |
| cuda | 512 | lattice | 0 | head | auto | 7 | 9.244 |  |  | 3.644e-01 | 4.476e-02 |
| cuda | 512 | lattice | 1 | baseline | auto | 7 | 10.888 | 0.900 | 0.887-0.930 | 3.418e-01 | 3.613e-02 |
| cuda | 512 | lattice | 1 | ewald | auto | 7 | 8.106 | 1.195 | 1.191-1.248 | 3.644e-01 | 4.476e-02 |
| cuda | 512 | lattice | 1 | head | auto | 7 | 10.041 |  |  | 3.644e-01 | 4.476e-02 |
