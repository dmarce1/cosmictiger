# racecheck staging warnings: operand-form survey and staging variants

Companion data of the "racecheck staging warnings" section of
`docs/cuda_kick_shared_memory.md`. compute-sanitizer 2025.1.0 (CUDA 12.8,
driver 595.99.02), RTX 3090, the CUDA `fmmWalk` regression at order 7
(`--run.test=fmmWalk --initial.particlesPerDimension=8 --hydro.enabled=false
--compute.cuda=true --hpx:threads=2 --hpx:bind=none --gravity.fmmOrder=7`).

- `forms_summary.txt`: for every frozen kick binary of this series and for the
  scratch build of each variant, the operand forms of the mbarrier instructions
  of the order-7 kick kernel (`ARRIVES.LDGSTSBAR.64`, `ATOMS.ARRIVE.64`, the
  polling `LDS` of `mbarrier.test_wait`), produced by `scripts/survey_one.sh`
  (cuobjdump -xelf, nvdisasm -g, `scripts/condense2.py`).
- `logs/`: racecheck runs. `*_all.log` were run with `--racecheck-report all
  --print-level info` (hazard records plus analysis reports; the slab dump was
  stopped after 83,626 records and is kept as a tally and a sample),
  `*_memcpy_async_off.log` with `--racecheck-memcpy-async no`, the variant runs
  in the default analysis mode with `--print-limit 0`. Function names in the
  reports are mis-resolved by the tool (lambda and SPH symbols); the file:line
  and the kernel-relative offsets are what the SASS was matched on.
  `scripts/analyze_hazards.py` tallies a hazard log.
- `variants/`: the diagnostic changes to `cosmictiger/gravity_cuda.hpp`
  (`scripts/apply_variant.py NAME` applies one to the committed text;
  `scripts/build_variants.sh` builds, freezes and surveys them in
  `build/racecheck`). None is committed to the kernel.
- `scripts/run_one.sh NAME BINARY [sanitizer options]`: one racecheck run after
  a quiet GPU, under the shared GPU lock.
- `outofline_forms.txt`: the same survey for the out-of-line order-7 `cudaGravityCpDirect` and `cudaGravityPpDirect` functions of the builds that keep them out of line.
