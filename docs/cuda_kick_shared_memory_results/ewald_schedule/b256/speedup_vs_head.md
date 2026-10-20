| backend | bucket | fixture | potential | binary | policy | order | kick ms | speedup vs reference | paired range | force rel L2 | potential rel L2 |
|---|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|
| cuda | 256 | adaptive | 0 | baseline | auto | 7 | 18.324 | 0.919 | 0.917-0.962 | 1.317e-03 | 9.639e-05 |
| cuda | 256 | adaptive | 0 | ewald | auto | 7 | 13.354 | 1.257 | 1.232-1.295 | 1.295e-03 | 1.114e-04 |
| cuda | 256 | adaptive | 0 | head | auto | 7 | 17.010 |  |  | 1.295e-03 | 1.114e-04 |
| cuda | 256 | adaptive | 1 | baseline | auto | 7 | 18.566 | 0.965 | 0.934-0.974 | 1.317e-03 | 9.639e-05 |
| cuda | 256 | adaptive | 1 | ewald | auto | 7 | 13.838 | 1.286 | 1.248-1.304 | 1.295e-03 | 1.114e-04 |
| cuda | 256 | adaptive | 1 | head | auto | 7 | 17.815 |  |  | 1.295e-03 | 1.114e-04 |
| cuda | 256 | clustered | 0 | baseline | auto | 7 | 6.842 | 0.822 | 0.813-0.836 | 1.884e-03 | 5.309e-04 |
| cuda | 256 | clustered | 0 | ewald | auto | 7 | 5.688 | 0.994 | 0.981-1.003 | 2.677e-03 | 1.153e-03 |
| cuda | 256 | clustered | 0 | head | auto | 7 | 5.672 |  |  | 2.677e-03 | 1.153e-03 |
| cuda | 256 | clustered | 1 | baseline | auto | 7 | 7.343 | 0.830 | 0.793-0.840 | 1.884e-03 | 5.309e-04 |
| cuda | 256 | clustered | 1 | ewald | auto | 7 | 6.098 | 0.997 | 0.983-1.003 | 2.677e-03 | 1.153e-03 |
| cuda | 256 | clustered | 1 | head | auto | 7 | 6.032 |  |  | 2.677e-03 | 1.153e-03 |
| cuda | 256 | lattice | 0 | baseline | auto | 7 | 11.369 | 0.915 | 0.866-0.927 | 3.418e-01 | 3.613e-02 |
| cuda | 256 | lattice | 0 | ewald | auto | 7 | 9.735 | 1.069 | 1.067-1.090 | 3.644e-01 | 4.476e-02 |
| cuda | 256 | lattice | 0 | head | auto | 7 | 10.413 |  |  | 3.644e-01 | 4.476e-02 |
| cuda | 256 | lattice | 1 | baseline | auto | 7 | 11.653 | 0.920 | 0.805-0.946 | 3.418e-01 | 3.613e-02 |
| cuda | 256 | lattice | 1 | ewald | auto | 7 | 9.998 | 1.077 | 1.061-1.093 | 3.644e-01 | 4.476e-02 |
| cuda | 256 | lattice | 1 | head | auto | 7 | 10.722 |  |  | 3.644e-01 | 4.476e-02 |
