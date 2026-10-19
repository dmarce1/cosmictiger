| backend | bucket | fixture | potential | binary | policy | order | kick ms | speedup vs reference | paired range | force rel L2 | potential rel L2 |
|---|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|
| cuda | 64 | adaptive | 0 | arena | auto | 7 | 20.808 | 0.919 | 0.917-0.941 | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 0 | arena | globalexp | 7 | 20.414 | 0.941 | 0.940-0.941 | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 0 | arena | globalforce | 7 | 20.574 | 0.928 | 0.917-0.942 | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 0 | arena | r6 | 7 | 21.191 | 0.908 | 0.901-0.913 | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 0 | arena | r6_reserve4k | 7 | 20.606 | 0.932 | 0.926-0.952 | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 0 | arena | r7 | 7 | 20.605 | 0.932 | 0.922-0.946 | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 0 | arena | reserve4k | 7 | 20.179 | 0.959 | 0.907-0.965 | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 0 | baseline | auto | 7 | 19.196 |  |  | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 1 | arena | auto | 7 | 21.371 | 0.923 | 0.913-0.954 | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 1 | arena | globalexp | 7 | 20.635 | 0.949 | 0.948-0.956 | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 1 | arena | globalforce | 7 | 20.734 | 0.946 | 0.943-0.952 | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 1 | arena | r6 | 7 | 20.981 | 0.941 | 0.916-0.953 | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 1 | arena | r6_reserve4k | 7 | 20.548 | 0.950 | 0.944-0.984 | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 1 | arena | r7 | 7 | 20.836 | 0.948 | 0.926-0.955 | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 1 | arena | reserve4k | 7 | 20.352 | 0.959 | 0.956-0.972 | 2.111e-03 | 1.025e-04 |
| cuda | 64 | adaptive | 1 | baseline | auto | 7 | 19.733 |  |  | 2.111e-03 | 1.025e-04 |
| cuda | 64 | clustered | 0 | arena | auto | 7 | 8.381 | 1.082 | 1.036-1.082 | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 0 | arena | globalexp | 7 | 8.928 | 1.007 | 0.967-1.015 | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 0 | arena | globalforce | 7 | 9.112 | 0.984 | 0.962-1.010 | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 0 | arena | r6 | 7 | 8.382 | 1.041 | 1.039-1.065 | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 0 | arena | r6_reserve4k | 7 | 8.467 | 1.040 | 0.856-1.071 | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 0 | arena | r7 | 7 | 8.106 | 1.074 | 1.046-1.118 | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 0 | arena | reserve4k | 7 | 8.410 | 1.043 | 1.007-1.103 | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 0 | baseline | auto | 7 | 8.771 |  |  | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 1 | arena | auto | 7 | 8.597 | 1.050 | 1.037-1.064 | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 1 | arena | globalexp | 7 | 8.652 | 1.036 | 1.017-1.052 | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 1 | arena | globalforce | 7 | 8.922 | 1.000 | 0.966-1.027 | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 1 | arena | r6 | 7 | 8.491 | 1.067 | 1.050-1.076 | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 1 | arena | r6_reserve4k | 7 | 8.670 | 1.028 | 0.708-1.129 | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 1 | arena | r7 | 7 | 8.348 | 1.070 | 1.033-1.090 | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 1 | arena | reserve4k | 7 | 8.572 | 1.040 | 1.014-1.085 | 4.815e-03 | 5.752e-04 |
| cuda | 64 | clustered | 1 | baseline | auto | 7 | 8.992 |  |  | 4.815e-03 | 5.752e-04 |
| cuda | 64 | lattice | 0 | arena | auto | 7 | 32.144 | 1.046 | 1.039-1.064 | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 0 | arena | globalexp | 7 | 32.284 | 1.037 | 1.034-1.066 | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 0 | arena | globalforce | 7 | 32.826 | 1.018 | 0.994-1.066 | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 0 | arena | r6 | 7 | 32.179 | 1.039 | 1.038-1.062 | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 0 | arena | r6_reserve4k | 7 | 32.094 | 1.041 | 1.037-1.068 | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 0 | arena | r7 | 7 | 31.883 | 1.048 | 1.046-1.057 | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 0 | arena | reserve4k | 7 | 31.916 | 1.047 | 1.040-1.086 | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 0 | baseline | auto | 7 | 33.403 |  |  | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 1 | arena | auto | 7 | 32.621 | 1.038 | 1.030-1.041 | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 1 | arena | globalexp | 7 | 32.473 | 1.047 | 1.034-1.047 | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 1 | arena | globalforce | 7 | 32.830 | 1.028 | 1.022-1.059 | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 1 | arena | r6 | 7 | 32.491 | 1.054 | 1.032-1.058 | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 1 | arena | r6_reserve4k | 7 | 32.210 | 1.048 | 1.040-1.071 | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 1 | arena | r7 | 7 | 32.235 | 1.042 | 1.035-1.052 | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 1 | arena | reserve4k | 7 | 31.799 | 1.056 | 1.041-1.084 | 3.418e-01 | 3.613e-02 |
| cuda | 64 | lattice | 1 | baseline | auto | 7 | 33.762 |  |  | 3.418e-01 | 3.613e-02 |
