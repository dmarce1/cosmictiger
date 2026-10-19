| backend | bucket | fixture | potential | binary | policy | order | kick ms | speedup vs reference | paired range | force rel L2 | potential rel L2 |
|---|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|
| cuda | 512 | adaptive | 0 | arena_fix | auto | 7 | 33.811 | 0.857 | 0.852-0.862 | 1.001e-03 | 9.754e-05 |
| cuda | 512 | adaptive | 0 | arena_fix | tile128 | 7 | 34.597 | 0.837 | 0.834-0.840 | 1.001e-03 | 9.754e-05 |
| cuda | 512 | adaptive | 0 | baseline | auto | 7 | 28.971 |  |  | 1.001e-03 | 9.754e-05 |
| cuda | 512 | adaptive | 1 | arena_fix | auto | 7 | 35.629 | 0.822 | 0.808-0.837 | 1.001e-03 | 9.754e-05 |
| cuda | 512 | adaptive | 1 | arena_fix | tile128 | 7 | 35.029 | 0.836 | 0.829-0.844 | 1.001e-03 | 9.754e-05 |
| cuda | 512 | adaptive | 1 | baseline | auto | 7 | 29.298 |  |  | 1.001e-03 | 9.754e-05 |
| cuda | 512 | clustered | 0 | arena_fix | auto | 7 | 9.890 | 1.040 | 1.022-1.059 | 1.767e-03 | 5.584e-04 |
| cuda | 512 | clustered | 0 | arena_fix | tile128 | 7 | 10.281 | 1.000 | 0.993-1.008 | 1.767e-03 | 5.584e-04 |
| cuda | 512 | clustered | 0 | baseline | auto | 7 | 10.284 |  |  | 1.767e-03 | 5.584e-04 |
| cuda | 512 | clustered | 1 | arena_fix | auto | 7 | 10.577 | 0.973 | 0.922-1.025 | 1.767e-03 | 5.584e-04 |
| cuda | 512 | clustered | 1 | arena_fix | tile128 | 7 | 11.009 | 0.934 | 0.898-0.971 | 1.767e-03 | 5.584e-04 |
| cuda | 512 | clustered | 1 | baseline | auto | 7 | 10.287 |  |  | 1.767e-03 | 5.584e-04 |
| cuda | 512 | lattice | 0 | arena_fix | auto | 7 | 12.621 | 0.917 | 0.876-0.958 | 3.418e-01 | 3.613e-02 |
| cuda | 512 | lattice | 0 | arena_fix | tile128 | 7 | 12.210 | 0.948 | 0.895-1.001 | 3.418e-01 | 3.613e-02 |
| cuda | 512 | lattice | 0 | baseline | auto | 7 | 11.586 |  |  | 3.418e-01 | 3.613e-02 |
| cuda | 512 | lattice | 1 | arena_fix | auto | 7 | 12.487 | 0.890 | 0.850-0.929 | 3.418e-01 | 3.613e-02 |
| cuda | 512 | lattice | 1 | arena_fix | tile128 | 7 | 12.601 | 0.883 | 0.829-0.937 | 3.418e-01 | 3.613e-02 |
| cuda | 512 | lattice | 1 | baseline | auto | 7 | 11.086 |  |  | 3.418e-01 | 3.613e-02 |
| cuda | 1024 | adaptive | 0 | arena_fix | auto | 7 | 80.229 | 1.017 | 1.015-1.018 | 7.528e-04 | 9.124e-05 |
| cuda | 1024 | adaptive | 0 | arena_fix | tile128 | 7 | 80.139 | 1.018 | 1.013-1.022 | 7.528e-04 | 9.124e-05 |
| cuda | 1024 | adaptive | 0 | baseline | auto | 7 | 81.558 |  |  | 7.528e-04 | 9.124e-05 |
| cuda | 1024 | adaptive | 1 | arena_fix | auto | 7 | 81.257 | 1.021 | 1.015-1.028 | 7.528e-04 | 9.124e-05 |
| cuda | 1024 | adaptive | 1 | arena_fix | tile128 | 7 | 81.911 | 1.013 | 1.012-1.014 | 7.528e-04 | 9.124e-05 |
| cuda | 1024 | adaptive | 1 | baseline | auto | 7 | 82.981 |  |  | 7.528e-04 | 9.124e-05 |
| cuda | 1024 | clustered | 0 | arena_fix | auto | 7 | 16.486 | 0.996 | 0.979-1.013 | 1.767e-03 | 5.584e-04 |
| cuda | 1024 | clustered | 0 | arena_fix | tile128 | 7 | 16.800 | 0.977 | 0.956-0.998 | 1.767e-03 | 5.584e-04 |
| cuda | 1024 | clustered | 0 | baseline | auto | 7 | 16.415 |  |  | 1.767e-03 | 5.584e-04 |
| cuda | 1024 | clustered | 1 | arena_fix | auto | 7 | 17.985 | 0.989 | 0.966-1.012 | 1.767e-03 | 5.584e-04 |
| cuda | 1024 | clustered | 1 | arena_fix | tile128 | 7 | 17.819 | 0.998 | 0.992-1.004 | 1.767e-03 | 5.584e-04 |
| cuda | 1024 | clustered | 1 | baseline | auto | 7 | 17.788 |  |  | 1.767e-03 | 5.584e-04 |
| cuda | 1024 | lattice | 0 | arena_fix | auto | 7 | 19.001 | 0.863 | 0.861-0.865 | 3.418e-01 | 3.613e-02 |
| cuda | 1024 | lattice | 0 | arena_fix | tile128 | 7 | 19.768 | 0.830 | 0.806-0.855 | 3.418e-01 | 3.613e-02 |
| cuda | 1024 | lattice | 0 | baseline | auto | 7 | 16.399 |  |  | 3.418e-01 | 3.613e-02 |
| cuda | 1024 | lattice | 1 | arena_fix | auto | 7 | 20.466 | 0.849 | 0.846-0.852 | 3.418e-01 | 3.613e-02 |
| cuda | 1024 | lattice | 1 | arena_fix | tile128 | 7 | 20.312 | 0.856 | 0.855-0.857 | 3.418e-01 | 3.613e-02 |
| cuda | 1024 | lattice | 1 | baseline | auto | 7 | 17.382 |  |  | 3.418e-01 | 3.613e-02 |
