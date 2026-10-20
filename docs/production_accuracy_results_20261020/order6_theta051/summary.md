Order 6, theta 0.51; baseline order 7, theta 0.61.

| fixture | force L2 | force p99 | potential L2 | potential p99 | low-force absolute RMS | low-force absolute max | passes |
|---|---:|---:|---:|---:|---:|---:|---|
| clustered | 0.000466268 | 0.00143989 | 0.000126504 | 0.00154157 | 253.039 | 880.788 | False |
| adaptive | 0.000172592 | 0.00626609 | 8.6274e-06 | 0.000793811 | 694.572 | 2219.89 | False |

Qualified on both fixtures: False.
Reference discrepancies and strict baseline gates are in summary.json.
Reference disagreement measures reproducibility, not true reference accuracy.
The direct reference evaluates pair contributions in float and accumulates in double.
