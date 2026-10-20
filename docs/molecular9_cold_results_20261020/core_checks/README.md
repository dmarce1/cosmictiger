These checks cover the molecular9 10–100 K extension. They use production C++ headers, strict floating-point flags, and CPU execution. The parent results directory records application and CUDA checks separately. The 12-test campaign predates the final EOS optimization that reuses the already computed 100 K lower-bound energy; the final warm comparison below confirms identical outputs after that optimization. Final rebuilt focused-test logs are retained by the parent results directory.

- `focused_ctest.log`: 12 tests passed, including cold rates/EOS/solver, existing SciPy Radau, native recombination, backgrounds, and Boltzmann checks.
- `primordial_rates_test.log`: final additional binary64 H₂⁺ bridge test passed through every integer temperature from 30 to 70 K, including the subnormal LTE interval.
- `primordial_solver_test.log`: independent analytic Compton heating/cooling and binary64 molecular-cooling references, with three tolerance refinements.
- `manifest.json`: source hashes and measured coverage.

The cold RHS grid contains 6,552 states: every integer temperature from 10 to 100 K, six compositions, four densities (0–10⁴ cm⁻³), and three CMB temperatures. It includes exact-zero, trace, and finite H₂⁺. Tests also check all 29 nonnegative finite collisional rates, EOS round trips and independent partition sums, the 80/100 K joins and first derivatives, and rejection below 10 K. Tight binary64 CDS polynomial checks are separate from comparison to the published tabulated upward rates; the largest Table 1 versus Table 2 fit discrepancy sampled here is 3.1662% at 20 K.

The warm baseline sampled 19,920 states across 415 temperatures from 100 to 6000 K, six mixtures, four densities, and two CMB temperatures. Rates, EOS, reaction fluxes, thermal ledger, and RHS produced identical 12,908,160-byte outputs before and after the extension. SHA-256 is `8f3b19e7be19c7c1616bb0fe343a985197f1786e3d619906139686f3b57cff58` for both. Binary outputs are omitted; the archived header snapshot and generator reproduce them:

```sh
python3 docs/molecular9_cold_results_20261020/core_checks/compare_warm.py
python3 docs/molecular9_cold_results_20261020/core_checks/cold_reference.py
```

`cold_reference.py` reads the archived primary CDS Table 2 data, evaluates the rates and heat capacity in binary64, and integrates the fully molecular 80 K cooling problem with RK4. Its 1,000/2,000-step temperatures agree within 2×10⁻¹³ K. It does not import or call production C++ code. `table1_reference.json` separately records the published upward-rate anchors and fixed 3:1 pair weights.

`guard_only_probe.csv` records the earlier reproduction: lowering only the temperature guards, with unchanged physics expressions, caused H₂⁺ 0/0 NaNs at 10–35 K even when H₂⁺ abundance was zero. `guard_only_probe.cpp` reproduces that behavior when compiled against an archived header copy with only those guards lowered. The completed implementation passes both the finite-RHS and independent subnormal-value checks.

These checks do not add molecular CMB pumping, ortho/para conversion, or high-density collisional physics. Cold convergence runs use an energy absolute tolerance of 10⁻¹⁰ eV per H nucleus; the default 10⁻⁶ budget otherwise dominates the relative tolerance at low temperature.
