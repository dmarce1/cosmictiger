#!/usr/bin/env python3
"""Independent integration reference; usage: primordial_reference.py adapter.so.

The adapter invokes the production RHS. SciPy Radau supplies an independent
integrator/Jacobian/error controller; this checks integration, not rate physics.
The molecular cooling case evolves all seven coordinates and checks reference
stability as well as backward Euler's tolerance-dependent global error.
"""
import ctypes
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.integrate import solve_ivp

library = ctypes.CDLL(sys.argv[1])
array = ctypes.POINTER(ctypes.c_double)
library.primordialReferenceInitial.argtypes = [ctypes.c_int, array]
library.primordialReferenceRhs.argtypes = [ctypes.c_int, array, array]
library.primordialReferenceAdvance.argtypes = [
    ctypes.c_int, ctypes.c_float, array, ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_uint)]
library.primordialReferenceTemperature.argtypes = [
    ctypes.c_int, array, ctypes.POINTER(ctypes.c_float)]
interval = float(np.float32(1.e12))
absolute = np.array([1.e-14, 1.e-18, 1.e-25, 1.e-23, 1.e-16, 1.e-24, 1.e-10])


def temperature(caseIndex, values):
    value = ctypes.c_float()
    status = library.primordialReferenceTemperature(
        caseIndex, values.ctypes.data_as(array), ctypes.byref(value))
    assert status == 0, (caseIndex, status)
    return value.value


def radauReference(caseIndex, initial, active, tolerance):
    calls = 0
    # Integrate the stored minority for fractions that start above one half,
    # as the production solver does: Radau's relative error on a saturated
    # fraction would otherwise push the trace complement negative or exceed
    # unity, and its step control would resolve the majority instead of the
    # physically meaningful minority. The transform is the identity for the
    # three original cases, whose fractions all start below one half.
    upper = initial[:6] > 0.5
    sign = np.where(np.concatenate([upper, [False]]), -1.0, 1.0)
    offset = np.where(np.concatenate([upper, [False]]), 1.0, 0.0)

    def rhs(time, reduced):
        nonlocal calls
        values = initial.copy()
        values[active] = offset[active] + sign[active] * reduced
        derivative = np.zeros(7, dtype=np.float64)
        status = library.primordialReferenceRhs(
            caseIndex, values.ctypes.data_as(array), derivative.ctypes.data_as(array))
        if status != 0:
            raise RuntimeError(f"RHS status {status} at {time}: {values}")
        calls += 1
        return sign[active] * derivative[active] * interval

    start = sign[active] * (initial[active] - offset[active])
    reference = solve_ivp(rhs, (0.0, 1.0), start, method="Radau",
                          rtol=tolerance, atol=absolute[active])
    assert reference.success, reference.message
    exact = initial.copy()
    exact[active] = offset[active] + sign[active] * reference.y[:, -1]
    return exact, calls


for caseIndex, name in enumerate(("atomic", "molecular_trace", "molecular_cooling",
                                  "molecular_hot", "ionized_hot")):
    initial = np.zeros(7, dtype=np.float64)
    assert library.primordialReferenceInitial(caseIndex, initial.ctypes.data_as(array)) == 0
    active = [0, 1, 2, 6] if caseIndex == 0 else (
        [0, 3, 4, 5, 6] if caseIndex == 1 else list(range(7)))
    # The FP32 rate coefficients make the RHS noisy at the 1e-7 level; the hot
    # cases span six decades in temperature and would need millions of Radau
    # evaluations at 2e-8. Their 1e-6 references are checked against a 2.5e-7
    # refinement below, which is still far inside the production error budget.
    referenceTolerance = 2.e-8 if caseIndex < 3 else 1.e-6
    exact, calls = radauReference(caseIndex, initial, active, referenceTolerance)
    scales = np.maximum(np.abs(exact), np.array([1.e-12] * 6 + [1.e-5]))
    print(f"case={name} Radau calls={calls} reference={exact.tolist()}", flush=True)
    if caseIndex >= 3:
        refined, refinedCalls = radauReference(caseIndex, initial, active, 2.5e-7)
        referenceDifference = np.max(np.abs(refined - exact) / scales)
        print(f"  Radau refinement calls={refinedCalls} max_scaled_difference="
              f"{referenceDifference:.9g}", flush=True)
        assert referenceDifference < 1.e-5, referenceDifference
        initialTemperature = temperature(caseIndex, initial)
        finalTemperature = temperature(caseIndex, exact)
        print(f"  T={initialTemperature:.6g} -> {finalTemperature:.6g} K; "
              f"2H2/H={initial[4]:.6g} -> {exact[4]:.6g}; H+/H={initial[0]:.6g} -> {exact[0]:.6g}",
              flush=True)
        # Hot molecular gas crosses the former 6000 K ceiling downward with its
        # H2 mostly dissociated; hot ionized gas cools, recombines and forms H2
        # without any reset of its nonequilibrium abundances.
        if caseIndex == 3:
            assert 29999.0 < initialTemperature < 30001.0
            assert 6000.0 < finalTemperature < 12000.0, finalTemperature
            assert exact[4] < 0.05 * initial[4] and exact[4] > 0.0
            assert 0.1 < exact[0] < 0.5
        else:
            assert 99999.0 < initialTemperature < 100001.0
            assert 6000.0 < finalTemperature < 10000.0, finalTemperature
            assert exact[0] < 0.5 and exact[4] > 1.e-5 and exact[3] > 0.0 and exact[5] > 0.0
    if caseIndex == 2:
        coarser, coarseCalls = radauReference(caseIndex, initial, active, 1.e-7)
        referenceDifference = np.max(np.abs(coarser - exact) / scales)
        # Both evaluations use FP32 rate coefficients. This independently
        # checks the reference floor is far below the production error budget.
        assert referenceDifference < 3.e-6, referenceDifference
        initialTemperature = temperature(caseIndex, initial)
        finalTemperature = temperature(caseIndex, exact)
        assert 1999.0 < initialTemperature < 2001.0
        assert 500.0 < finalTemperature < 1500.0
        assert exact[6] < 0.75 * initial[6]
        assert abs(exact[4] / initial[4] - 1.0) < 0.01
        print(f"  Radau refinement calls={coarseCalls} max_scaled_difference="
              f"{referenceDifference:.9g}; T={initialTemperature:.6g} -> "
              f"{finalTemperature:.6g} K; thermal_energy="
              f"{initial[6]:.9g} -> {exact[6]:.9g} eV/H", flush=True)
    previous = float("inf")
    for tolerance in (1.e-3, 1.e-4, 1.e-5):
        values = initial.copy()
        accepted = ctypes.c_float()
        substeps = ctypes.c_uint()
        status = library.primordialReferenceAdvance(
            caseIndex, tolerance, values.ctypes.data_as(array), ctypes.byref(accepted),
            ctypes.byref(substeps))
        assert status == 0 and accepted.value == interval, (status, accepted.value)
        componentErrors = np.abs(values[active] - exact[active]) / scales[active]
        relative = np.max(componentErrors)
        energyError = abs(values[6] / exact[6] - 1.0)
        print(f"  local_tolerance={tolerance:g} max_scaled_global_error={relative:.9g} "
              f"thermal_relative_error={energyError:.9g} "
              f"temperature={temperature(caseIndex, values):.9g} K "
              f"accepted_substeps={substeps.value} "
              f"component_errors={np.array2string(componentErrors, precision=3)} "
              f"state={values.tolist()}", flush=True)
        # A BE step-doubling controller bounds an O(h^2) local difference;
        # first-order accumulated error therefore scales as sqrt(tolerance).
        # Require refinement, while allowing the coefficient/storage floor in
        # the nearly equilibrium trace case. Do not equate local and global error.
        assert relative < max(previous * .7, 2.e-6), (relative, previous)
        if caseIndex == 4:
            # He++ decays through four decades while the gas recombines; the
            # first-order solver's error on that trace is ten times the usual
            # budget (10% at 1e-3) and refines with the same square-root law.
            # Every other component meets the usual budget.
            others = np.delete(componentErrors, active.index(2))
            assert np.max(others) < 0.4 * np.sqrt(tolerance), others
            assert componentErrors[active.index(2)] < 4.0 * np.sqrt(tolerance), componentErrors
            assert energyError < 0.2 * np.sqrt(tolerance), energyError
        elif caseIndex >= 2:
            assert relative < 0.4 * np.sqrt(tolerance), relative
            assert energyError < 0.2 * np.sqrt(tolerance), energyError
        else:
            assert relative < (0.03 if caseIndex == 0 else 3.e-6)
        previous = relative
print("Independent Radau references passed")
