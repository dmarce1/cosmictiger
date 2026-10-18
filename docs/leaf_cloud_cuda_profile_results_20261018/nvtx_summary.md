# Host-stage attribution

The extra CUDA-mode wall time arises mainly before the GPU launch: new internal-node CP work on the CPU plus more expensive direct-walk classification.

Two alternating pairs × 15 measured samples/mode/variant; these timings include NVTX profiling overhead.

| Mode | Root phase | Before ms | After ms | Change ms |
|---|---|---:|---:|---:|
| Force | Root kick | 18.0100 | 19.0845 | +1.0745 |
| Force | Before kernel | 12.5126 | 13.6539 | +1.1413 |
| Force | GPU kernel | 5.4019 | 5.3380 | -0.0639 |
| Force | After kernel | 0.0955 | 0.0926 | -0.0030 |
| Force + potential | Root kick | 17.7867 | 19.0542 | +1.2676 |
| Force + potential | Before kernel | 12.3940 | 13.5738 | +1.1798 |
| Force + potential | GPU kernel | 5.3084 | 5.4010 | +0.0926 |
| Force + potential | After kernel | 0.0843 | 0.0794 | -0.0049 |

Stage wall coverage below is the interval union across both CPU workers. Different rows overlap and must not be added. Summed scope durations are in the JSON.

| Stage | Force before → after ms | Force + potential before → after ms | Calls/kick before → after | Node/workitem payload before → after |
|---|---:|---:|---:|---:|
| `cpu.gravity.cp.internal` | 0.0000 → 1.8561 | 0.0000 → 1.8477 | 0 → 349 | 0 → 1724 |
| `cpu.walk.direct` | 1.0037 → 1.5530 | 0.9967 → 1.5413 | 737 → 737 | 33491 → 31686 |
| `cpu.gravity.cc.direct` | 1.4986 → 1.4068 | 1.4972 → 1.4036 | 706 → 705 | 14484 → 13708 |
| `cpu.gravity.cc.ewald` | 9.3482 → 9.4822 | 9.3170 → 9.6111 | 166 → 166 | 3494 → 3494 |
| `cpu.walk.ewald` | 0.4430 → 0.4299 | 0.4431 → 0.4279 | 737 → 737 | 8735 → 8735 |
| `gpu.add_work.set_insert` | 1.2078 → 1.1234 | 1.1965 → 1.1144 | 736 → 736 | 46878 → 43183 |
| `gpu.prepare` | 0.9359 → 0.9274 | 0.7808 → 0.7829 | 1 → 1 | 736 → 736 |
| `gpu.advice_to_gpu` | 0.1408 → 0.1390 | 0.1375 → 0.1332 | 1 → 1 | 0 → 0 |
| `gpu.execute` | 5.6978 → 5.6197 | 5.6135 → 5.6834 | 1 → 1 | 736 → 736 |
| `gpu.restore` | 0.0283 → 0.0271 | 0.0231 → 0.0214 | 1 → 1 | 0 → 0 |

New internal CP costs about 2 ms in summed scope time per kick. Its wall coverage is about 1.85 ms, of which 0.82–0.89 ms overlaps Ewald CC on the other worker. Direct classification adds about 0.54 ms of wall coverage despite receiving fewer checklist nodes. These overlapping costs produce a net 1.14–1.18 ms increase before kernel start.

No CPU leaf CP/PC/PP scopes execute in either variant on this CUDA workload; those leaves are handed to the GPU. Both versions hand over 736 workitems. GPU preparation stays unchanged and checklist set insertion becomes slightly faster.

Caveats:

- Per-label union_ms removes self-overlap across workers, but different labels still overlap and must not be added.
- Summed range time can include HPX suspension and does not measure active CPU utilization.
- NVTX instrumentation changes absolute timing; original uninstrumented comparison is the performance result.
- CPU direct-walk range covers acceptance/list classification as a whole; individual opening tests are not separately timed.
- CP-without-Ewald is an overlap diagnostic, not proof of a task dependency critical path.
