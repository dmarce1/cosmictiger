Order 5, theta 0.32; baseline order 7, theta 0.61.

| fixture | force L2 | force p99 | potential L2 | potential p99 | low-force absolute RMS | low-force absolute max | passes |
|---|---:|---:|---:|---:|---:|---:|---|
| clustered | 0.000158674 | 0.000676673 | 2.09536e-05 | 9.95175e-05 | 145.957 | 700.491 | True |
| adaptive | 7.1846e-05 | 0.0025374 | 4.17842e-06 | 0.000319042 | 322.035 | 1179.56 | True |

Qualified on both fixtures: True.
Reference discrepancies and strict baseline gates are in summary.json.
Reference disagreement measures reproducibility, not true reference accuracy.
The direct reference evaluates pair contributions in float and accumulates in double.
