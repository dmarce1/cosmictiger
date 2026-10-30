# Molecular9 up to 10⁸ K

The nine-species network now shares the atomic ceiling: both `atomic6` and
`molecular9` accept 10–10⁸ K at proper hydrogen-nucleus densities up to
10⁴ cm⁻³. Heated molecular gas keeps all nine species while it dissociates,
ionizes, cools, recombines and reforms H₂; there is no temperature switch to
the atomic network or to equilibrium, and no chemical history is reset.
Temperature above 10⁸ K remains an error, not a request to clamp.

This note records exactly which published expressions are used beyond their
fitted ranges, which continuations are ours, and the evidence behind each
claim. The former 6000 K limit was set by the molecular fits, not by SH03 or
any other consumer; see [star formation](star_formation.md) for how the two
models now combine.

## Reaction coefficients above 6000 K

Atomic channels C01–C06 are unchanged and already covered 10–10⁸ K. Every
molecular channel was audited against Glover & Abel (2008) Table A1, Abel et
al. (1997), Galli & Palla (1998) and Savin et al. (2004). The rules are:

1. **Published high-temperature branches are used.** The two branches of a
   pair do not meet exactly, so the hot branch is multiplied by the binary64
   ratio of the two published expressions at the splice. The ratios are
   within the stated accuracy of the fits and remove a value jump that would
   otherwise sit inside the implicit solver's Jacobian.
2. **Fits with a published upper temperature are held constant above it.**
3. **Fits whose published form diverges are held constant above a stated
   temperature.** Diverging means exponential growth or a power law steep
   enough to exceed any collisional rate inside the domain (√T forms for
   the two mutual-neutralization fits, T², T³·⁸ and T⁴·⁷⁴ for the others).
   The default freeze is 10⁴ K, the upper end of the GA08 models that the
   table was assembled for.
4. **Everything else is evaluated as published.** Several of these forms
   still grow slowly (T⁰·¹¹, T⁰·¹³⁶, T⁰·²⁵, √T e⁻¹) but all stay below
   3.5 × 10⁻⁸ cm³ s⁻¹ at 10⁸ K, below the electron-impact rates they compete
   with, and the molecular abundances they govern are negligible there (see
   the atomic limit below). Above about 2 × 10⁴ K they are extrapolations of
   the cited fits.

| ID | Reaction | Published range used | Treatment above it |
| --- | --- | --- | --- |
| C07 | H + e → H⁻ | GA08 A1/1 (Wishart), T ≤ 6000 K and T > 6000 K; Abel97 states 1–10⁸ K | hot branch × 1.0056449 (6000 K splice) |
| C09 | H + H⁺ → H₂⁺ | GP98 Table 1 H8, 1 ≤ T ≤ 32 000 K | held at 32 000 K |
| C11 | H₂ + H⁺ → H₂⁺ + H | Savin 2004, 100–30 000 K | held at 30 000 K (the polynomial diverges) |
| C12, C13 | H₂ + e, H₂ + H dissociation | GA08 A1/8–9, Trevisan & Tennyson; Mac Low & Shull / Lepp & Shull | as published (slow growth, T⁰·¹¹ and √T e⁻¹; LTE branches T⁰·¹³⁶ and bounded) |
| C14, C15 | H⁻ + e, H⁻ + H detachment | Janev 1987 polynomials in ln Tₑ | as published; 10⁸ K is 8.6 keV, inside Janev's range |
| C16 | H⁻ + H⁺ → 2H | Croft 1999, grows as √T | held at 10⁴ K |
| C17 | H⁻ + H⁺ → H₂⁺ + e | GA08 A1/16, T ≤ 8000 K and T > 8000 K | hot branch × 1.0075749 (8000 K splice) |
| C20 | H₂ + H₂ dissociation | GA08 A1/10 | as published (decays as T⁻¹·⁵) |
| C21 | H₂ + He dissociation, v = 0 | GA08 A1/11 (Dove 1987), grows as T³·⁸ | held at 10⁴ K; LTE branch as published |
| C27 | He + H⁺ → He⁺ + H | GA08 A1/27 (Kimura 1993), T ≤ 10⁴ K and T > 10⁴ K | hot branch × 1.0024343 (10⁴ K splice), held at 10⁵ K where the T⁴·⁷⁴ form is already 10⁻³ of electron-impact He ionization |
| C28 | He⁺ + H⁻ → He + H | GA08 A1/28, contains exp(+T/22 400) | held at 10⁴ K, like the other mutual-neutralization fit |
| C29 | He + H⁻ → He + H + e | GA08 A1/29, grows as T² | held at 10⁴ K |
| others | C08, C10, C18, C19, C22–C26 | constants, decaying forms, or C26's slow (T/300)⁰·²⁵ growth | as published |

The GA08 critical densities (eqs. 15–17) that bridge the v = 0 and LTE
dissociation rates are held at 10⁴ K; the linear He expression would otherwise
pass through zero near 8.3 × 10⁴ K and underflow near 7 × 10⁵ K.

## Molecular cooling above 6000 K

The GA08 Table 8 polynomials are published to 6000 K (H, H₂, He) and 10⁴ K
(H⁺, e). Between 6000 and 10⁴ K the first three are evaluated beyond their
range; above 10⁴ K all five coefficients and the Hollenbach–McKee LTE
asymptote are held at their 10⁴ K values. This is Grackle's practice
(`GAHI`, `GAH2`, `GAHe`, `GAHp`, `GAel`, `H2LTE` in `rate_functions.c`,
revision `01261db0cefc69d8d5558e76a671b0c1f6e48271`), adopted here as a
documented continuation rather than a published fit. The H₂⁺ vibrational
cooling fits (GA08 eqs. 44–48) are held at 10⁴ K; GA08 give them no upper
limit and discuss them to about 10⁴ K, so the hold is this project's choice. The Yoshida et al. quenching fraction and all binding
energies are exact at any temperature.

Held cooling coefficients bound the cooling per molecule by its LTE value at
10⁴ K, about 1.2 × 10⁻¹⁸ erg s⁻¹. Hot gas with a trace of H₂ therefore cannot
radiate more than that trace times this rate while the trace is being
destroyed; it cannot create a spurious cooling catastrophe.

## Molecular thermodynamics

The rigid-rotor ladders summed to J = 64 stop converging above about
2 × 10⁴ K for H₂ and 10⁴ K for H₂⁺. Between 6000 and 10⁴ K the sums run to
J = 128. Above 10⁴ K each ladder uses Mulholland's asymptotic partition
function, Z = (T/θ)[1 + x/3 + x²/15 + 4x³/315], x = θ/T, giving per ladder
E/k = T − θ/3 − θ²/(45T) − 8θ³/(945T²) and c_v/k = 1 + θ²/(45T²) +
16θ³/(945T³). The even and odd ladders each hold half of Z to within
exp(−π²/4x), which is below 10⁻¹²⁰ in this range, so both have the same mean
energy and the odd ladder's normalized ground state subtracts 2θ as before.
Converged binary64 sums agree with the runtime to about 10⁻⁷ in energy and
heat capacity (the binary32 rounding of the stored values; the test tolerance
is 4 × 10⁻⁷) at 6000, 8000, 9999, 10⁴, 10 001, 2 × 10⁴, 10⁵, 10⁶ and 10⁸ K
for both molecules. The harmonic vibrational closure is unchanged and tends
to kT per molecule. These remain the approximate frozen-spin thermodynamics of
the existing model: at temperatures where H₂ would be fully dissociated the
EOS is merely well defined and smooth, which is what the implicit solver needs.

## What is unchanged

Rate coefficients, cooling coefficients, the thermal energy function and the
RHS evaluated at a given temperature are bit for bit identical to the
previous implementation for every state at 10–6000 K; a 1661-point
temperature grid with 1008 mixed states was compared between the old and new
headers with the archived generator in
[`molecular9_hot_results_20261030`](molecular9_hot_results_20261030/README.md).
The EOS *inversion* can differ by up to three binary32 ulp in temperature for
molecule-rich states from about 3200 K (pure H₂; later for diluted mixtures),
because the Newton start no longer clips at 6000 K; the RHS of such a state,
which takes its temperature from the inversion, inherits that roundoff. The
solver's line-search projection (below) can also change substep sequences,
within the local tolerance, wherever a Newton candidate used to leave its
branch's bound. The physics version stays 2: no previously accepted state
evolves differently beyond solver roundoff and tolerance, and runs that used
to stop with `UnsupportedState` above 6000 K now continue.

## Solver: positivity projection

Exercising hot atomic states exposed a pre-existing solver defect that is
independent of the network: for a species at exactly zero (He⁺⁺ below about
2 × 10⁴ K, for example) the numerical Jacobian can hand Newton a noise-level
correction of the wrong sign (about 10⁻³⁷). The composition check rejected
every damped candidate, so the substep was rejected at every useful step size;
the step collapsed until the correction underflowed, and the solver exhausted
its 4096 accepted substeps long before the interval ended (half-ionized
`atomic6` gas at 12 000 K and 0.05 cm⁻³ advanced 1.4 % of a 10¹² s interval).
Both the line search and the stagnation test now project each fraction onto
its branch's feasible half-line (fraction ≥ 0 on the lower branch, ≤ 1 on the
upper one) before decoding; the residual is evaluated at the projected point,
the acceptance criteria and error control are unchanged, and every accepted
state still satisfies the residual test at a feasible point. The same state
now integrates in 132 substeps (`atomic6`) and 573 (`molecular9`), and a
synthetic model with a −10⁻³⁷ derivative on a zero species, which the old
solver could not integrate at all, completes in a handful of substeps
(`PrimordialSolver.PositivityProjection`).

## Cost

The first-order source solver spends hundreds of substeps on gas that cools
and recombines through 10⁴ K, and the molecular network evaluates seven
coupled variables where the atomic one evaluates three. Measured on one
core at the application tolerances (relative 10⁻⁴, abundance floor 10⁻¹²)
over a 10¹³ s interval, starting half ionized below 10⁵ K and fully ionized
above:

| nH (cm⁻³) | T₀ (K) | atomic6 substeps / s | molecular9 substeps / s |
| --- | --- | --- | --- |
| 0.05 | 10⁴ | 39 / 0.02 | 369 / 2.0 |
| 0.05 | 3 × 10⁴ | 307 / 0.19 | 415 / 1.4 |
| 0.05 | 10⁵ | 433 / 0.29 | 434 / 1.1 |
| 0.05 | 10⁶ | 324 / 0.21 | 324 / 0.65 |
| 10 | 10⁴ | 272 / 0.14 | 884 / 8.4 |
| 10 | 3 × 10⁴ | 581 / 0.33 | 909 / 7.1 |
| 10 | 10⁵ | 1637 / 0.92 | 1802 / 10.7 |
| 10 | 10⁶ | 2346 / 2.3 | 2492 / 14.5 |

The dense molecular runs cool all the way to a few hundred kelvin and form
H₂, which the atomic network cannot do; the extra substeps are that
evolution, and the extra time per substep is the larger network. Hot,
ionized gas in a `molecular9` run therefore costs three to ten times what
it costs in `atomic6`. Trace tolerances matter: with the unit-test defaults
(10⁻¹⁶ on ions, 10⁻²⁰ on H⁻ and H₂⁺) the step control resolves H⁻ at
10⁻¹³ to one part in 10³ and hot dense gas needs tens of thousands of
substeps; the application default 10⁻¹² does not. The Radau reference test
raises its substep budget for that reason. The `molecular9_hot` production
fixture uses 64 particles, relative tolerance 10⁻³ and abundance floor
10⁻¹⁰ to stay inside its 240 s run limit.

## Star formation

The SH03 closure never required `atomic6`; the restriction only reflected the
old molecular ceiling. With `molecular9`, ordinary gas and decoupled winds keep
their nine-species state through the hot, ionized regime. Gas entering the
dense multiphase branch still receives the neutral-cold plus atomic-CIE-hot
projection with zero molecular fractions, and its SH03 effective energy;
that projection discards the parcel's molecular history exactly as it already
discarded its atomic nonequilibrium history, and it does not charge the
binding energy of the discarded H⁻, H₂ and H₂⁺ (0.754, 4.48 and 2.65 eV) to
anything: the SH03 ledger tracks thermal and kinetic energy only, unlike the
ordinary-gas ledger. Leaving the dense branch resumes nine-species chemistry
from the projected atomic composition. Dense-phase cooling remains the SH03
atomic CIE closure with no molecular contribution.

## Validation

- `primordial_rates` (`HotMolecularPhysics`): binary64 anchors for every hot
  branch, held fit and extrapolated cooling polynomial; splice continuity;
  constancy above each freeze; finite nonnegative coefficients on
  10–10⁸ K; EOS anchors against converged partition sums for H₂ and H₂⁺;
  inversion round trips and finite RHS for four mixtures at three densities
  from 6000 K to 10⁸ K; hot dissociation ledger and H₂⁺ bridge identities.
- Atomic limit: for molecule-free states at 2 × 10⁴–10⁸ K the molecular9 and
  atomic6 RHS differ exactly, to 2 × 10⁻⁶ of the atomic rates, by the
  molecular9-only channels C07, C09, C26 and C27, and only C27 enters the
  thermal ledger. Those channels are at most 0.2 % of the H ionization plus
  recombination traffic (C07 at 2 × 10⁴ K, falling with temperature), and the
  C27 cost is at most 2 × 10⁻⁶ of the net thermal rate across the grid
  (8 × 10⁻⁸ at 2 × 10⁴ K, 1.6 × 10⁻⁶ at 10⁵ K, about 5 × 10⁻⁷ above).
- `primordial_solver` (`HotMolecularReference`, `ThermalEndpoints`): the
  production solver against the independent Radau reference below, and the
  molecular network at the 10⁸ K endpoint with trace H₂ and H₂⁺ that
  dissociate rather than reset.
- `primordial_radau_reference` cases `molecular_hot` and `ionized_hot`:
  SciPy Radau integrates the production RHS. Half-ionized gas at 30 000 K with
  2H₂/H = 2 × 10⁻³ at 1 cm⁻³ cools to 9002 K in 10¹² s while its H₂ falls to
  1.5 × 10⁻⁵ (destroyed, then partly reformed through H⁻); the production
  solver's scaled global error is 6.6 × 10⁻³, 2.1 × 10⁻³ and 6.7 × 10⁻⁴ at
  local tolerances 10⁻³, 10⁻⁴ and 10⁻⁵. Fully ionized gas at 10⁵ K and
  10 cm⁻³ cools to 7400 K, recombines to H⁺/H = 0.18 and forms H₂ to
  2H₂/H = 1.1 × 10⁻⁴ with its H⁻ and H₂⁺ traces evolving continuously. Every
  component except He⁺⁺ meets the usual 0.4 √tolerance budget; He⁺⁺ decays
  through four decades and the first-order solver's error on it is 10 %,
  3.4 % and 1.1 % at the three tolerances, refining with the same
  square-root law. The reference integrates the stored minority for
  fractions above one half, as the production solver does; Radau's relative
  error on the saturated fraction would otherwise push the trace complement
  out of the domain and resolve the majority instead of the physics.
- Application: `run_primordial_integration.py` adds a `molecular9_hot` case
  (30 000 K, half ionized, 2H₂/H = 10⁻³, 64 particles) through production
  stepping, exact restart and incompatible-network rejection, and rejects
  2 × 10⁸ K for both networks; its incompatible-network restart now keeps the
  checkpoint's particle layout so the rejection really comes from the
  chemistry comparison. `run_options.py` accepts star formation with either
  network and checks that a hot manual `molecular9` state keeps its molecular
  fraction. `run_star_formation.py` runs the three SH03 fixtures with
  `molecular9` as well as `atomic6`, plus one `molecular9` cosmology run with
  restart and rejection checks; the fixture's own guard no longer demands
  `atomic6`.
- CUDA parity and GPU execution were **not** run for this extension: GPU work
  was paused at the time. `primordial_cuda` carries the same shared headers and
  must be run before any combined CUDA claim.

## Sources

- [Glover & Abel (2008), Table A1, Table 8, equations (13)–(17) and (44)–(50)](https://arxiv.org/pdf/0803.1768v1).
- [Abel, Anninos, Zhang & Norman (1997), Appendix A and Table 2](https://arxiv.org/pdf/astro-ph/9608040v1), attachment fit range.
- [Galli & Palla (1998), Table 1 and Appendix A](https://arxiv.org/pdf/astro-ph/9803315), H8 range and LTE cooling.
- [Savin, Krstić, Haiman & Stancil (2004), ApJ 606, L167](https://doi.org/10.1086/421108), charge transfer 100–30 000 K.
- [Grackle `rate_functions.c`](https://github.com/grackle-project/grackle/blob/main/src/clib/rate_functions.c), revision pinned above, for the cooling clamp convention.
- Mulholland (1928) rotational partition-function expansion, as in standard statistical-mechanics texts.
