# Cartesian M2L ratio generator option

Recorded November 9, 2026. This is a design note; no code change is requested yet.

For the generated Cartesian tracefree M2L kernel, remove `sourceRatio` and
`targetRatio` from the default generated interface. In the current production
call both equal `inverseRadius`, so the default kernel needs only
`inverseRadius` and should use it wherever either ratio is currently used.
Keep independent source and target ratios available as an explicit option in
`tools/generate_fmm.py`, disabled by default. A possible command-line name is
`--independent-m2l-ratios`; with that option enabled, emit the existing
three-parameter scaling interface and behavior.

Limit this choice to Cartesian M2L. Other operators' scaling parameters and
the spherical and Ewald kernels are separate decisions. When implementing,
update the generator, checked-in generated headers, typed M2L wrapper, and
tests together. The default tests should cover the one-radius interface;
option-enabled tests should retain coverage of unequal source and target
scales. Keep `--check` and generated operation counts consistent with the
selected generator mode.
