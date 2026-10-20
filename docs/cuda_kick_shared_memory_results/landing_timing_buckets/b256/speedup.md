| backend | bucket | fixture | potential | binary | policy | order | kick ms | speedup vs reference | paired range | force rel L2 | potential rel L2 |
|---|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|
| cuda | 256 | adaptive | 0 | merged | auto | 7 | 17.411 | 0.988 | 0.965-1.002 | 1.295e-03 | 1.114e-04 |
| cuda | 256 | adaptive | 0 | premerge | auto | 7 | 17.285 |  |  | 1.295e-03 | 1.114e-04 |
| cuda | 256 | adaptive | 1 | merged | auto | 7 | 18.403 | 0.990 | 0.956-1.015 | 1.295e-03 | 1.114e-04 |
| cuda | 256 | adaptive | 1 | premerge | auto | 7 | 17.916 |  |  | 1.295e-03 | 1.114e-04 |
| cuda | 256 | clustered | 0 | merged | auto | 7 | 5.715 | 1.228 | 1.206-1.242 | 2.677e-03 | 1.153e-03 |
| cuda | 256 | clustered | 0 | premerge | auto | 7 | 7.013 |  |  | 2.677e-03 | 1.153e-03 |
| cuda | 256 | clustered | 1 | merged | auto | 7 | 6.091 | 1.226 | 1.196-1.346 | 2.677e-03 | 1.153e-03 |
| cuda | 256 | clustered | 1 | premerge | auto | 7 | 7.460 |  |  | 2.677e-03 | 1.153e-03 |
| cuda | 256 | lattice | 0 | merged | auto | 7 | 10.612 | 1.000 | 0.986-1.075 | 3.644e-01 | 4.476e-02 |
| cuda | 256 | lattice | 0 | premerge | auto | 7 | 10.645 |  |  | 3.644e-01 | 4.476e-02 |
| cuda | 256 | lattice | 1 | merged | auto | 7 | 11.104 | 0.996 | 0.956-1.054 | 3.644e-01 | 4.476e-02 |
| cuda | 256 | lattice | 1 | premerge | auto | 7 | 10.911 |  |  | 3.644e-01 | 4.476e-02 |
