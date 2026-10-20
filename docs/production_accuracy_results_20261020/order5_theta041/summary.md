Order 5, theta 0.41; baseline order 7, theta 0.61.

| fixture | force L2 | force p99 | potential L2 | potential p99 | low-force absolute RMS | low-force absolute max | passes |
|---|---:|---:|---:|---:|---:|---:|---|
| clustered | 0.000389845 | 0.00182011 | 9.15755e-05 | 0.00044077 | 393.734 | 1261.41 | False |
| adaptive | 0.000185529 | 0.00734148 | 8.93055e-06 | 0.00148531 | 889.605 | 3140.9 | False |

Qualified on both fixtures: False.
Reference discrepancies and strict baseline gates are in summary.json.
Reference disagreement measures reproducibility, not true reference accuracy.
The direct reference evaluates pair contributions in float and accumulates in double.
