# SPH thermal formulations

`hydro.enabled=true` enables SPH. Select its evolved thermal variable with:

```ini
[hydro]
enabled = true
sphFormulation = densityEnergy
```

The equivalent command-line option is `--hydro.sphFormulation=densityEnergy`.
The default, `densityEntropy`, preserves entropy evolution. Both use the same
summed density, variational pressure force, viscosity, conduction, and midpoint
time integration.

| Formulation | Persistent thermal field | Derived quantity |
| --- | --- | --- |
| `densityEntropy` | `sphParticlesEntropies`: entropy tracer A | Scaled specific internal energy E |
| `densityEnergy` | `sphParticlesInternalEnergies`: E = a²u | Entropy tracer A |

Here u is physical specific internal energy, A = P/ρ^γ uses physical pressure
and density, and ρ_c = a³ρ is the stored density. The ideal-gas conversion is

    E = A ρ_c^(γ−1) a^(5−3γ)/(γ−1).

At γ = 5/3, the scale-factor exponent is exactly zero. Lower-dimensional SPH
remains noncosmological, with a = 1.

Density-entropy evolves only the entropy change from viscosity and conduction.
Density-energy evolves the full specific thermal-energy rate:

    dE_i/dτ = pressure work_i + viscous heating_i + conduction_i
              + (5−3γ) (da/dτ)/a E_i.

Each pair uses the pressure-work coefficient belonging to its own endpoint,
consistent with the [variational pressure force](variational_sph.md). Conduction
adds opposite energy increments to the equal-mass pair. Homogeneous expansion
belongs to an active particle's own event, so a fine particle's reciprocal kick
does not repeat the coarse particle's expansion update. The existing explicit
midpoint operator evolves velocity and the selected thermal variable together.

Initial conditions and source solvers supply E in either mode. Entropy mode
converts the initial E after the first density solve; energy mode keeps E.
Cooling and star formation prepare and validate the selected float state before
committing it, and their energy ledger uses the energy actually represented by
that state. Output continues to expose both energy and entropy, deriving the
quantity that is not stored.

`hydro.hopkins=true` retains both arrays for future MFM work and still requires
`hydro.enabled=true`. There is no MFM evolution solver yet: `sphFormulation`
selects the canonical variable for the current SPH updates, and the auxiliary
array is synchronized after thermal, density, or scale-factor changes.

Migration, particle reordering, and checkpoints carry only the enabled arrays.
The default entropy checkpoint layout is preserved. Restart rejects a different
formulation or a different one-field/two-field layout. These options cannot be
changed after gas storage has been allocated.
