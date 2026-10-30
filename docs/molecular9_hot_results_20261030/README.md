# Molecular9 hot extension: comparison generators (2026-10-30)

Companion to [`docs/molecular9_hot.md`](../molecular9_hot.md). The two
programs here are the generators behind that note's "What is unchanged" and
"Molecular thermodynamics" claims, archived so the comparison can be
repeated instead of trusted.

## Bit-for-bit comparison over 10–6000 K

`bitwise_compare.cpp` evaluates, for `molecular9`, every rate coefficient and
H₂ cooling coefficient at 1661 temperatures from 10 K to 6000 K (20 per
decade of the ratio), and at every twentieth temperature the thermal energy
function, the RHS at that temperature and the EOS inversion for four
compositions (neutral atomic; pure H₂; a mixture with H⁺ 0.3, He⁺ 0.1,
H⁻ 10⁻⁶, 2H₂/H 0.2, 2H₂⁺/H 10⁻⁵; and H⁺ 0.99, He⁺⁺ 0.5, 2H₂⁺/H 10⁻³) at
0, 1 and 10⁴ cm⁻³, 1008 states in all. Two FNV-1a hashes of the binary32
results are printed: one over the fixed-temperature quantities, one over the
inverted EOS.

Build it twice with the same flags, once against the headers of commit
`5350c5ca` (the tree before the extension) and once against the current
tree:

```sh
mkdir -p /tmp/old/cosmictiger/chemistry
for f in primordial_types.hpp primordial_eos.hpp primordial_rates.hpp primordial_solver.hpp; do
  git show 5350c5ca:cosmictiger/chemistry/$f > /tmp/old/cosmictiger/chemistry/$f
done
g++ -std=c++20 -O2 -fno-fast-math -ffp-contract=off -I/tmp/old -I. \
  -o /tmp/compare_old docs/molecular9_hot_results_20261030/bitwise_compare.cpp
g++ -std=c++20 -O2 -fno-fast-math -ffp-contract=off -I. \
  -o /tmp/compare_new docs/molecular9_hot_results_20261030/bitwise_compare.cpp
/tmp/compare_old; /tmp/compare_new
```

Recorded on 2026-10-30 with GCC 13 (Ubuntu), `-O2 -fno-fast-math
-ffp-contract=off`:

| Build | States | Fixed-temperature hash | Inversion hash |
| --- | --- | --- | --- |
| `5350c5ca` headers | 1008 | `42247e8ee3553aec` | `1329dcc68f1b5dc1` |
| current tree | 1008 | `42247e8ee3553aec` | `c85f02ef66195320` |

The fixed-temperature hash is identical: rate coefficients, cooling
coefficients, the thermal energy function and the RHS at a given temperature
are unchanged below 6000 K. The inversion hash differs because the Newton
start of `evaluateEos` is no longer clipped at 6000 K; for molecule-rich
states from about 3200 K the recovered temperature can differ by up to three
binary32 ulp, inside the 3 × 10⁻⁷ inversion tolerance. An earlier run of the
same generator that printed every value as `%a` showed exactly that: only
inverted temperatures, heat capacities and pressures differed, by one ulp, at
4408 K, 5090 K and 6000 K for the molecule-rich mixtures.

## Rotor asymptote

`rotor_asymptote.cpp` compares the runtime `molecularModes` (129-level
ladders between 6000 K and 10⁴ K, Mulholland asymptote above) with converged
binary64 sums for H₂ (θ_r = 85.4 K, θ_v = 5987 K) and H₂⁺ at 6000, 9000,
9999, 10 000, 10 001, 12 000, 20 000, 10⁵, 10⁶, 10⁷ and 10⁸ K. Recorded
maximum relative differences: 1.0 × 10⁻⁷ in energy and 1.2 × 10⁻⁷ in heat
capacity (binary32 rounding of the stored values); the even and odd ladder
partition functions agree to 10⁻¹⁵ at every temperature, as the
exp(−π²/4x) estimate predicts.

## Checksums

```text
1e173e217164696bdf5a400255edbcd3b4c34e13e831ff1e0a2c1e15beea4d96  bitwise_compare.cpp
2c8d5d5c676df117e8ee9845af843a0c5b0c87ec5272bb2df18f6409680f58b5  rotor_asymptote.cpp
```
