# Primordial reaction and thermal catalogue

This catalogue describes physics version **3**, implemented in
`primordial_types.hpp`, `primordial_rates.hpp`, and `primordial_eos.hpp`. The
shared source evaluates both CPU and CUDA physics. It is a defined, optically
thin primordial model, with the limitations below; nine species do not imply
that every primordial process or density regime is represented.

## State, units, and numerical representation

The state directly stores nine particle counts per hydrogen nucleus, in order:
H, H+, He, He+, He++, electrons, H−, H2 and H2+. Molecules count once per
molecule; helium is normalized to hydrogen. The environment supplies proper
hydrogen-nucleus density in cm⁻³, the He/H number ratio, CMB temperature in
kelvin, network choice and explicit radiation scalars.

The state, coefficients, EOS and source integrator share one template scalar
`Real`. The application selects ordinary float; the same implementation can be
instantiated with double. Signed-complement encoding, FloatFloat products and
compensated source sums have been removed. Dominant-species conservation
corrections preserve the explicitly stored depleted species; they do not require
reconstructing neutral abundances from rounded ion fractions.

The solver evolves thermal energy in eV per H nucleus, excluding chemical binding
energy. Rates are cm³ s⁻¹, photo coefficients are s⁻¹ per absorber, and radiation
heating inputs are net eV per absorber per second. The typed host boundary
converts these quantities to and from hydrodynamic specific energy.

The adopted reaction and thermal fits below remain the model catalogue. The
Anninos source integrator uses algebraic equilibrium for H− and H2+ and neglects
collisions between those trace intermediates when computing H− equilibrium,
following equation (32). Thus the molecular solver no longer evolves their
independent non-equilibrium transients. See [the solver description](primordial_chemistry.md).

## Implemented domains

| Network | Temperature | Proper H-nucleus density | Collision channels |
| --- | --- | --- | --- |
| `atomic6` | 10–10⁸ K | 0–10⁴ cm⁻³ | C01–C06 |
| `molecular9` | 10–10⁸ K | 0–10⁴ cm⁻³ | C01–C29 |

These are implementation/model bounds, not claims that every individual
laboratory measurement covers every point. The molecular temperature interval
combines the original warm fits with the [10 K cold extension](molecular9_cold.md)
and the [hot extension](molecular9_hot.md), which lists every molecular fit
used beyond its published range, the published high-temperature branches, and
the temperatures at which unbounded fits are held constant.
Published approximations
and extrapolations retained below are identified. Outside these bounds the
kernel returns `UnsupportedState`; it does not clamp temperature, density, or
negative abundances. Density zero is the well-defined radiation-only limit of
rates per H nucleus. Floating-point underflow of an exponentially suppressed
coefficient remains possible; no artificial positive rate floor is added.

The molecular network includes the 29 two-body H/He processes in Glover–Abel's
Table A1, with the atomic rate choices specified here. Three-body processes,
deuterium, HD, metals, dust, cosmic rays, and high-density opacity are absent.
The six-species atomic model deliberately has the conventional electron-impact
ionization/recombination channels; the cold H/He charge-transfer additions are
part of the molecular model. An exactly electron-free neutral state has no
collisional-ionization seed in `atomic6`; neutral-impact ionization is not in
this model.

## Collision reactions

Numbers C01–C19 correspond to the original Abel network ordering. C20–C29
are the additional Glover–Abel H/He channels. In the implementation's
stoichiometry table, C01–C19 occupy rows 0–18, photons occupy rows 19–26,
and C20–C29 occupy rows 27–36. Every row conserves H nuclei, He nuclei, and
charge by integer arithmetic. Reaction fluxes per H nucleus are
`k * nHydrogen * abundanceA * abundanceB` using particle abundances, so H₂
and H₂⁺ each contribute half their stored nuclei fraction.

| ID | Reaction | Coefficient provenance |
| --- | --- | --- |
| C01 | H + e → H⁺ + 2e | Cen fit reproduced in KWH96 Table 1 |
| C02 | H⁺ + e → H + photon | HG97 case A |
| C03 | He + e → He⁺ + 2e | Cen/KWH96 |
| C04 | He⁺ + e → He + photon | VF96 Table 1 He I(b), plus Cen dielectronic contribution |
| C05 | He⁺ + e → He²⁺ + 2e | Cen/KWH96 |
| C06 | He²⁺ + e → He⁺ + photon | HG97 case A |
| C07 | H + e → H⁻ + photon | GA08 A1, reaction 1, Wishart fit; both branches, joined at 6000 K |
| C08 | H⁻ + H → H₂ + e | GA08 fiducial associative-detachment coefficient, 1.3×10⁻⁹ |
| C09 | H + H⁺ → H₂⁺ + photon | GA08 A1/3, Galli–Palla logarithmic fit, held above 32 000 K |
| C10 | H₂⁺ + H → H₂ + H⁺ | GA08 A1/4, 6.4×10⁻¹⁰ |
| C11 | H₂ + H⁺ → H₂⁺ + H | GA08 A1/7, Savin polynomial in **natural** log T, held above 30 000 K |
| C12 | H₂ + e → 2H + e | GA08 A1/8, ground-state/LTE fits |
| C13 | H₂ + H → 3H | GA08 A1/9, ground-state/LTE fits |
| C14 | H⁻ + e → H + 2e | Abel97/GA08 A1/14, polynomial in log electron temperature in eV |
| C15 | H⁻ + H → 2H + e | Abel97/GA08 A1/15, branch at electron temperature 0.1 eV |
| C16 | H⁻ + H⁺ → 2H | GA08 §2.1.1, Croft mutual-neutralization fit, held above 10⁴ K |
| C17 | H⁻ + H⁺ → H₂⁺ + e | GA08 A1/16, both branches, joined at 8000 K |
| C18 | H₂⁺ + e → 2H | GA08 A1/6, branch at 617 K |
| C19 | H₂⁺ + H⁻ → H₂ + H | GA08 A1/21, molecular product branch |
| C20 | H₂ + H₂ → H₂ + 2H | GA08 A1/10, ground-state/LTE fits |
| C21 | H₂ + He → 2H + He | GA08 A1/11, ground-state fit held above 10⁴ K, LTE fit |
| C22 | H₂⁺ + H⁻ → 3H | GA08 A1/22, atomic product branch |
| C23 | H₂ + e → H⁻ + H | GA08 A1/23 |
| C24 | H₂ + He⁺ → He + H + H⁺ | GA08 A1/24 |
| C25 | H₂ + He⁺ → H₂⁺ + He | GA08 A1/25 |
| C26 | He⁺ + H → He + H⁺ | GA08 A1/26 |
| C27 | He + H⁺ → He⁺ + H | GA08 A1/27, both branches, joined at 10⁴ K and held above 10⁵ K |
| C28 | He⁺ + H⁻ → He + H | GA08 A1/28, held above 10⁴ K |
| C29 | He + H⁻ → He + H + e | GA08 A1/29, held above 10⁴ K |

Dissociation C12, C13, C20 and C21 uses GA08 equations (13)–(17) to interpolate
log rates between vibrational-ground-state and LTE populations. The H₂ term in
the critical-density expression counts **nuclei**, while fluxes count molecules.
The prescription assumes electrons do not dominate high-density vibrational
excitation. It is an approximate population closure, not level-resolved chemistry.
Log interpolation occurs before exponentiation, so an underflowed low-density
coefficient is never passed to `log`. The critical-density fits are held at
their 10⁴ K values above that temperature.

## Radiation channels

All radiation fields default to zero. Rates and net heating are supplied
separately; the kernel does not infer a spectrum or multiply a heating scalar
by its corresponding ionization coefficient a second time.

| ID | Channel |
| --- | --- |
| P01 | H → H⁺ + e |
| P02 | He → He⁺ + e |
| P03 | He⁺ → He²⁺ + e |
| P04 | H⁻ → H + e |
| P05 | H₂ → 2H; aggregate direct and Lyman–Werner dissociation |
| P06 | H₂ → H₂⁺ + e |
| P07 | H₂⁺ → H + H⁺ |
| P08 | H₂⁺ → 2H⁺ + e |

The reactions follow Abel97 §4 and Appendix A. Absorbed photon thresholds belong in the
external radiation model, while the heating inputs supply the remaining thermal
deposition. There is no built-in UV-background table, shielding estimator,
secondary-ionization model, or radiative transport. Externally attenuated scalars
are accepted, but this does not implement their attenuation physics.

## Thermal ledger and equation of state

Atomic electron-impact ionization consumes 13.5984, 24.5874, and 54.4178 eV per
corresponding reaction. Case-A recombination uses HG97 H/He²⁺ cooling fits;
He⁺ radiative recombination uses the conventional `kBT * alpha` approximation
and a separate dielectronic cooling fit. Recombination binding energy escapes
in photons, so its full ionization threshold is not subtracted from thermal
energy again.

H and He⁺ excitation and common-Gaunt-factor free-free emission use KWH96.
Neutral He ground-state excitation uses GJ07 Table 7 (Bray et al. 2000).
The metastable-He contribution is not included. Free-free emission weights
`H+ + He+ + 4 He++`. Compton exchange is proportional to
`xe * Tcmb^4 * (Tcmb - T)` and can heat as well as cool; it is not a CMB floor.

H₂ line cooling includes H, H₂, He, H⁺, and electron collision partners from
GA08 Table 8, fixed ortho:para 3:1. Above 6000 K the H, H₂ and He polynomials
are extrapolated to 10⁴ K and every coefficient, including the LTE asymptote
and the H₂⁺ fits, is held at its 10⁴ K value; this follows Grackle and is
recorded in the [hot extension](molecular9_hot.md). Below 100 K the H₂ collider uses Hernández21
Table 2 and detailed balance, and the H collider uses the GA08 10–100 K branch.
An explicitly numerical correction over 80–100 K matches the existing warm
values and slopes. Cold logarithmic coefficients are converted to eV before
exponentiation, and zero LTE coefficients have a finite zero-cooling limit.
The [cold extension](molecular9_cold.md) records these assumptions. The
Hollenbach–McKee LTE expression is reproduced in Yoshida06 equations
(18)–(20), and the low-density/LTE bridge follows GP98 Appendix A. Proton-driven ortho/para
conversion is omitted because the spin mixture is frozen. H₂⁺ vibrational
cooling follows GA08 equations (44)–(50), including its H/e partners and LTE
saturation. GA08 explicitly identifies the low-temperature H–H₂⁺ expression as
an extrapolation. Neither molecular line model includes absorption/stimulated
emission from the CMB or other background radiation. Such heating must be
provided externally when relevant; the code does not establish molecular
radiative equilibrium at `T = Tcmb`.

All H₂ collisional dissociation partners consume 4.48 eV per destroyed molecule.
Reverse molecular charge transfer consumes 1.83 eV; collisional H⁻ detachment
consumes 0.7542 eV. Dissociative attachment and reverse He/H charge transfer use
the corresponding binding-energy differences. Formation through C08 and C10
uses Yoshida06 equations (24)–(26): 3.53 and 1.83 eV respectively, multiplied by
the collisional quenching fraction. Assigning all formation binding energy to
the gas would overheat low-density gas, where excited products radiate.

The thermal closure follows the dominant-process approximation of GA08 §2.3.9.
It **does not resolve** kinetic/excitation/photon partitions for C16–C19, C22,
C24–C26, or C28. `ThermalRates::unresolvedChemicalEnergyRelease` gives an upper
bound on the available binding energy from those channels, in eV/H/s; it is
not included in `netHeating()`. It is not a bound on all model error. Applications
with abundant H⁻ or H₂⁺, or where this quantity competes with resolved thermal
sources, need product-state thermal modelling before a science claim. Radiative
attachment/association kinetic cooling is also omitted. No unconditional
closed-system conservation of binding plus gas energy is claimed for this
open, radiating approximate network.

The EOS integrates translational, rotational, and vibrational thermal energies;
it does not use `energy = cv(T)*T`. Even/odd rigid-rotor ladders have frozen 3:1
weights and separately normalized ground states, following Boley07 equations
(1)–(5). H₂ uses rotational/vibrational temperatures 85.4 and 5987 K. H₂⁺ uses
the NIST constants `B0 = Be - alpha_e/2` and fundamental vibration
`omega_e - 2 omega_e*x_e`. A harmonic vibrational closure and rigid rotor are
approximations, particularly toward the high-temperature end; these are not
state-resolved molecular populations. The ladders are summed to J = 64 up to
6000 K and J = 128 up to 10⁴ K, and use Mulholland's asymptotic partition
function above 10⁴ K. The same energy function supplies positive
heat capacity, temperature inversion, pressure, mean molecular weight, and
frozen-composition sound-speed gamma.

## Provenance and validation boundary

The cited editions below pin this catalogue. Different choices in current
Grackle or another chemistry package are not automatically equivalent reference
models. A numerical comparison must align rates, recombination case, radiation,
spin mixture, energy convention, and temperature/domain handling.

- **Abel97:** [Abel et al., 1997, original arXiv v1](https://arxiv.org/pdf/astro-ph/9608040v1), reaction and photon catalogue.
- **KWH96:** [Katz, Weinberg & Hernquist, 1996, Table 1](https://arxiv.org/pdf/astro-ph/9509107), Cen atomic fits.
- **HG97:** [Hui & Gnedin, 1997, Appendix A](https://arxiv.org/pdf/astro-ph/9612232), case-A recombination and cooling.
- **VF96:** [Verner & Ferland, 1996, equation (4), Table 1(b)](https://arxiv.org/pdf/astro-ph/9509083), wide-range He⁺ radiative recombination.
- **GA08:** [Glover & Abel, 2008, arXiv v1](https://arxiv.org/pdf/0803.1768v1), Table 8, Table A1, and equations (13)–(17), (44)–(50). Natural logarithms are used in the Savin polynomial; decimal logs in the cooling fits.
- **GJ07:** [Glover & Jappsen, 2007, revised arXiv paper, Table 7](https://arxiv.org/pdf/0705.0182), neutral-helium excitation and thermal conventions.
- **GP98:** [Galli & Palla, 1998, Appendix A](https://arxiv.org/pdf/astro-ph/9803315), molecular LTE interpolation and Hollenbach–McKee cooling.
- **Yoshida06:** [Yoshida et al., 2006, equations (23)–(26)](https://arxiv.org/pdf/astro-ph/0606106), molecular formation/dissociation heating and quenching.
- **Boley07:** [Boley et al., 2007](https://arxiv.org/pdf/astro-ph/0701146), molecular partition-function thermodynamics.
- **Hernández21:** [Hernández et al. (2021)](https://doi.org/10.1051/0004-6361/202039790), equations (5)–(7), Table 2; [CDS numerical tables](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/647/A155).
- **NIST:** [CCCBDB experimental diatomic constants](https://cccbdb.nist.gov/expdiatomicsx.asp), H₂⁺ row, accessed 2026-10-18; numeric constants are recorded in the EOS source.

`tests/primordial_rates.cpp` checks all reaction rows, each photon source,
coefficient anchors calculated independently in binary64, all-temperature
coefficient scans, direct species preservation, partition-energy/cv anchors,
EOS inversion and derivatives, He²⁺ free-free weighting, signed Compton exchange,
the thermal ledger, explicit domain rejection, and the hot-extension anchors,
splices, freezes and the molecule-free atomic-limit identity. These tests establish local
formula and representation checks. They do not establish cosmological,
full-trajectory, MPI, or independent stiff-integrator agreement. See
`primordial_chemistry.md` for actual build/runtime evidence and
`primordial_chemistry_plan.md` for remaining release gates.
