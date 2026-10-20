# Molecular9 down to 10 K

The molecular network supports 10–6000 K at proper hydrogen-nucleus densities
up to 10⁴ cm⁻³. This extends the existing nine-species model without adding
species or changing its source integrator. Temperature outside this range is
an error, not a request to clamp or heat the gas to a floor.

## Cold cooling model

The original 100 K boundary came from the intersection of the adopted
H₂ cooling fits. The cold extension supplies the missing branches:

- For collisions with H, use the 10–100 K branch in Glover & Abel (2008),
  Table 8. Their low-temperature prescription is itself based on the
  extrapolation described in equations (28)–(29).
- For collisions with H₂, use Hernández et al. (2021), Table 2, for
  de-excitation rates, and equation (5) for the reverse rates by detailed
  balance. The four ground-ladder transitions have energy gaps 509.86 and
  844.60 K. Their probabilities use the existing frozen ortho:para mixture,
  3:1. The tabulated rates already contain the identical-particle factors
  specified by equation (6); no additional factor of one half is inserted.
- The adopted He, H⁺, and electron cooling fits in Glover & Abel Table 8
  already extend down to 10 K.

The new low-temperature H and H₂ prescriptions do not exactly meet the old
fits at 100 K. A cubic Hermite correction to log cooling, expressed in
temperature, joins the cold values and slopes at 80 K to the old values and
slopes at 100 K. This 80–100 K matching interval is our numerical prescription,
not a fit published in either paper. It avoids extrapolating the old polynomial
below its valid range and keeps the existing formulas at and above 100 K.

The H₂–H₂ calculation retains only excitation out of the two ground spin
ladders. It is a low-density cooling closure, combined with the existing
approximate LTE saturation prescription, rather than an evolution of individual
rotational level populations. The other chemical rates retain their published
low-temperature prescriptions and existing approximations; the 10 K model
boundary does not claim that every individual laboratory measurement covers
the entire temperature range.

Cold coefficients are converted to eV units before exponentiation where needed
to avoid losing representable rates through an intermediate underflow in erg.
The H₂⁺ saturation calculation handles zero coefficients explicitly; its
exponentially suppressed low-density and LTE rates can both underflow at low
temperature. No artificial positive rate floor is introduced.

## Initialization and remaining scope

Native recombination gives a fiducial temperature of 48.855 K and
H⁺/H = 2.3788 × 10⁻⁴ at z=49. Molecular9 can now start from that state with its
existing zero default molecular abundances. Explicit initial fractions and
temperature still override the native values independently.

This remains an optically thin H/He network with a frozen spin mixture.
It includes Compton exchange, but not absorption or stimulated emission in
molecular lines from the CMB. Molecular-rich gas colder than the radiation
field can therefore require a level-population and radiation model. A CMB
temperature floor would not be a substitute: dilute cosmological gas can be
colder than the CMB. HD/deuterium, dust, metals, shielding, and high-density
collapse remain outside this network.

A 10 K lower bound also does not cover an unheated background all the way to
the present. In the fiducial native history, the gas is 10.20 K at z=21 and
9.31 K at z=20.

## Validation

The [validation archive](molecular9_cold_results_20261020/README.md) separates
independent physics references, CPU tests, CUDA parity, and application tests.
The cold checks include 6,552 RHS/EOS states, comparison with the published
H₂–H₂ upward rates, and independent thermal integrations. The old and new
implementations produce bitwise-identical output on a 19,920-state grid from
100 to 6000 K. Production SPH tests check native initialization, molecular
formation from the cold initial state, and exact checkpoint restart.

## Sources

- [Glover & Abel (2008), Table 8 and equations (28)–(29)](https://academic.oup.com/mnras/article/388/4/1627/981339).
- [Hernández et al. (2021), equations (5)–(7) and Table 2](https://doi.org/10.1051/0004-6361/202039790),
  with [CDS catalogue J/A+A/647/A155](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/647/A155),
  including calculated forward/reverse rates and polynomial coefficients.
- The existing [reaction catalogue](primordial_reactions.md) records the other
  reaction, thermal, EOS, and radiation approximations.
