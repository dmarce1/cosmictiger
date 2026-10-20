Order 6, theta 0.35; baseline order 7, theta 0.61.

| fixture | force L2 | force p99 | potential L2 | potential p99 | low-force absolute RMS | low-force absolute max | passes |
|---|---:|---:|---:|---:|---:|---:|---|
| clustered | 5.45712e-05 | 0.000285661 | 4.02024e-06 | 2.18034e-05 | 57.7727 | 201.578 | True |
| adaptive | 2.27184e-05 | 0.000486556 | 5.93992e-07 | 5.74816e-05 | 64.277 | 269 | True |

Qualified on both fixtures: True.
Reference discrepancies and strict baseline gates are in summary.json.
Reference disagreement measures reproducibility, not true reference accuracy.
The direct reference evaluates pair contributions in float and accumulates in double.
