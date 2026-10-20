| backend | bucket | fixture | potential | binary | policy | order | kick ms | speedup vs reference | paired range | force rel L2 | potential rel L2 |
|---|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|
| cuda | 512 | adaptive | 0 | baseline | auto | 7 | 28.442 |  |  | 1.001e-03 | 9.754e-05 |
| cuda | 512 | adaptive | 0 | ewald | auto | 7 | 19.649 | 1.453 | 1.429-1.470 | 1.015e-03 | 1.108e-04 |
| cuda | 512 | adaptive | 0 | head | auto | 7 | 26.385 | 1.087 | 1.056-1.101 | 1.015e-03 | 1.108e-04 |
| cuda | 512 | adaptive | 1 | baseline | auto | 7 | 29.607 |  |  | 1.001e-03 | 9.754e-05 |
| cuda | 512 | adaptive | 1 | ewald | auto | 7 | 20.575 | 1.437 | 1.432-1.470 | 1.015e-03 | 1.108e-04 |
| cuda | 512 | adaptive | 1 | head | auto | 7 | 27.187 | 1.094 | 1.080-1.103 | 1.015e-03 | 1.108e-04 |
| cuda | 512 | clustered | 0 | baseline | auto | 7 | 9.651 |  |  | 1.767e-03 | 5.584e-04 |
| cuda | 512 | clustered | 0 | ewald | auto | 7 | 5.223 | 1.862 | 1.799-1.900 | 2.146e-03 | 1.120e-03 |
| cuda | 512 | clustered | 0 | head | auto | 7 | 5.469 | 1.785 | 1.765-1.795 | 2.146e-03 | 1.120e-03 |
| cuda | 512 | clustered | 1 | baseline | auto | 7 | 10.432 |  |  | 1.767e-03 | 5.584e-04 |
| cuda | 512 | clustered | 1 | ewald | auto | 7 | 6.042 | 1.712 | 1.655-1.794 | 2.146e-03 | 1.120e-03 |
| cuda | 512 | clustered | 1 | head | auto | 7 | 6.086 | 1.698 | 1.635-1.775 | 2.146e-03 | 1.120e-03 |
| cuda | 512 | lattice | 0 | baseline | auto | 7 | 10.340 |  |  | 3.418e-01 | 3.613e-02 |
| cuda | 512 | lattice | 0 | ewald | auto | 7 | 7.630 | 1.360 | 1.346-1.544 | 3.644e-01 | 4.476e-02 |
| cuda | 512 | lattice | 0 | head | auto | 7 | 9.244 | 1.120 | 1.103-1.215 | 3.644e-01 | 4.476e-02 |
| cuda | 512 | lattice | 1 | baseline | auto | 7 | 10.888 |  |  | 3.418e-01 | 3.613e-02 |
| cuda | 512 | lattice | 1 | ewald | auto | 7 | 8.106 | 1.341 | 1.302-1.343 | 3.644e-01 | 4.476e-02 |
| cuda | 512 | lattice | 1 | head | auto | 7 | 10.041 | 1.111 | 1.075-1.128 | 3.644e-01 | 4.476e-02 |
