"""Independent analytic limits for the Anninos split source method.

The adapter instantiates the same scalar algorithm with double. This script uses
closed-form photoionization, photodissociation, heating, and Compton solutions;
it does not use the removed Newton solver or a nonequilibrium H-/H2+ closure.
"""
import ctypes
import math
import sys

library = ctypes.CDLL(sys.argv[1])
array = ctypes.POINTER(ctypes.c_double)
library.primordialReferenceInitial.argtypes = [ctypes.c_int, array]
library.primordialReferenceAdvance.argtypes = [
    ctypes.c_int, ctypes.c_double, ctypes.c_double, array,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint)]
library.primordialReferenceTemperature.argtypes = [ctypes.c_int, array, array]


def initial(case):
    state = (ctypes.c_double * 10)()
    assert library.primordialReferenceInitial(case, state) == 0
    return state


def conserved(state, helium):
    assert all(math.isfinite(x) and x >= 0 for x in state)
    assert math.isclose(state[0] + state[1] + state[6] + 2*(state[7]+state[8]),
                        1, rel_tol=0, abs_tol=4e-15)
    assert math.isclose(sum(state[2:5]), helium, rel_tol=0, abs_tol=4e-15)
    assert math.isclose(state[1] + state[3] + 2*state[4] + state[8],
                        state[5] + state[6], rel_tol=0, abs_tol=4e-15)


def evolve(case, interval, limit, pieces=1):
    state = initial(case)
    helium = sum(state[2:5])
    work = 0
    for _ in range(pieces):
        accepted = ctypes.c_double()
        substeps = ctypes.c_uint()
        status = library.primordialReferenceAdvance(
            case, interval/pieces, limit, state,
            ctypes.byref(accepted), ctypes.byref(substeps))
        assert status == 0, (case, limit, pieces, status)
        assert accepted.value == interval/pieces
        conserved(state, helium)
        work += substeps.value
    return state, work


def temperature(case, state):
    value = ctypes.c_double()
    assert library.primordialReferenceTemperature(case, state, ctypes.byref(value)) == 0
    return value.value


# The chemical BDF step converges at first order as the electron bound shrinks.
photo_initial = initial(0)
photo_exact = 1 - photo_initial[0] * math.exp(-1)
errors = []
for limit in (.1, .05, .025, .0125, .00625):
    state, work = evolve(0, 1e12, limit)
    errors.append(abs(state[1] - photo_exact))
assert all(b < .65*a for a, b in zip(errors, errors[1:])), errors
assert errors[-1] < .002, errors
print(f"Photoionization refinement: errors={errors}")

# At zero density and fixed ionization the Compton ODE is exactly linear.
compton_initial = initial(1)
capacity = 1.5 * 8.617333262145e-5 * sum(compton_initial[:9])
decay = 6.35264354e-26 * 100**4 / capacity
compton_exact = 100 + 9900 * math.exp(-1)
errors = []
for limit in (.1, .05, .025, .0125, .00625):
    state, work = evolve(1, 1/decay, limit)
    errors.append(abs(temperature(1, state) - compton_exact))
assert all(b < .65*a for a, b in zip(errors, errors[1:])), errors
assert errors[-1] < 25, errors
print(f"Compton refinement: errors={errors}")

# H2 dissociation changes no electrons, so the electron criterion alone does
# not refine it. Outer subdivision tests the molecular update independently.
molecular_initial = initial(2)
molecular_exact = molecular_initial[7] * math.exp(-1)
errors = []
for pieces in (8, 16, 32, 64):
    state, work = evolve(2, 1e12, .1, pieces)
    errors.append(abs(state[7] - molecular_exact))
assert all(b < .55*a for a, b in zip(errors, errors[1:])), errors
assert errors[-1] < 3e-4, errors
print(f"Molecular photodissociation refinement: errors={errors}")

# Thermal-before-chemistry splitting samples the old absorber population.
# As the outer interval shrinks it approaches the exact absorbed energy.
heated_initial = initial(3)
exact_energy = heated_initial[9] + 2*heated_initial[0]*(1-math.exp(-1))
errors = []
for pieces in (32, 64, 128, 256):
    state, work = evolve(3, 1e12, .1, pieces)
    errors.append(abs(state[9] - exact_energy))
assert all(b < .55*a for a, b in zip(errors, errors[1:])), errors
assert errors[-1] < .005, errors
print(f"Split photoheating refinement: errors={errors}")
