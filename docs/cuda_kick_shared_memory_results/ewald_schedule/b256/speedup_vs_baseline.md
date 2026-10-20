| backend | bucket | fixture | potential | binary | policy | order | kick ms | speedup vs reference | paired range | force rel L2 | potential rel L2 |
|---|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|
| cuda | 256 | adaptive | 0 | baseline | auto | 7 | 18.324 |  |  | 1.317e-03 | 9.639e-05 |
| cuda | 256 | adaptive | 0 | ewald | auto | 7 | 13.354 | 1.358 | 1.343-1.374 | 1.295e-03 | 1.114e-04 |
| cuda | 256 | adaptive | 0 | head | auto | 7 | 17.010 | 1.088 | 1.039-1.091 | 1.295e-03 | 1.114e-04 |
| cuda | 256 | adaptive | 1 | baseline | auto | 7 | 18.566 |  |  | 1.317e-03 | 9.639e-05 |
| cuda | 256 | adaptive | 1 | ewald | auto | 7 | 13.838 | 1.334 | 1.293-1.380 | 1.295e-03 | 1.114e-04 |
| cuda | 256 | adaptive | 1 | head | auto | 7 | 17.815 | 1.037 | 1.027-1.070 | 1.295e-03 | 1.114e-04 |
| cuda | 256 | clustered | 0 | baseline | auto | 7 | 6.842 |  |  | 1.884e-03 | 5.309e-04 |
| cuda | 256 | clustered | 0 | ewald | auto | 7 | 5.688 | 1.202 | 1.192-1.229 | 2.677e-03 | 1.153e-03 |
| cuda | 256 | clustered | 0 | head | auto | 7 | 5.672 | 1.216 | 1.197-1.230 | 2.677e-03 | 1.153e-03 |
| cuda | 256 | clustered | 1 | baseline | auto | 7 | 7.343 |  |  | 1.884e-03 | 5.309e-04 |
| cuda | 256 | clustered | 1 | ewald | auto | 7 | 6.098 | 1.203 | 1.187-1.239 | 2.677e-03 | 1.153e-03 |
| cuda | 256 | clustered | 1 | head | auto | 7 | 6.032 | 1.205 | 1.191-1.260 | 2.677e-03 | 1.153e-03 |
| cuda | 256 | lattice | 0 | baseline | auto | 7 | 11.369 |  |  | 3.418e-01 | 3.613e-02 |
| cuda | 256 | lattice | 0 | ewald | auto | 7 | 9.735 | 1.168 | 1.158-1.259 | 3.644e-01 | 4.476e-02 |
| cuda | 256 | lattice | 0 | head | auto | 7 | 10.413 | 1.093 | 1.079-1.155 | 3.644e-01 | 4.476e-02 |
| cuda | 256 | lattice | 1 | baseline | auto | 7 | 11.653 |  |  | 3.418e-01 | 3.613e-02 |
| cuda | 256 | lattice | 1 | ewald | auto | 7 | 9.998 | 1.155 | 1.133-1.339 | 3.644e-01 | 4.476e-02 |
| cuda | 256 | lattice | 1 | head | auto | 7 | 10.722 | 1.088 | 1.057-1.243 | 3.644e-01 | 4.476e-02 |
