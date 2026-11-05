"""Check the dimension capability contract without configuring dependencies."""
import pathlib
import subprocess
import sys
import tempfile

source = pathlib.Path(sys.argv[1]).resolve()
features = source / "cmake/features.cmake"
solvers = ("FMM", "PMMM_PP", "PMMM_FMM", "PM", "P3M_PP", "P3M_FMM")
with tempfile.TemporaryDirectory(prefix="sph_dimension_config_") as directory:
    script = pathlib.Path(directory) / "check.cmake"
    script.write_text(f'include("{features}")\n')
    common = ["cmake", "-DCOSMICTIGER_ENABLE_CHEMISTRY=OFF", "-DCOSMICTIGER_ENABLE_STARS=OFF"]
    common += [f"-DCOSMICTIGER_ENABLE_{solver}=OFF" for solver in solvers]
    cases = [(str(dimension), {}, True) for dimension in (1, 2, 3)]
    cases += [(value, {}, False) for value in ("0", "4", "-1", "2.0", "", "ON", "1;2")]
    for dimension in (1, 2):
        cases += [(str(dimension), {solver: "ON"}, False) for solver in solvers]
        cases += [(str(dimension), {"SPH": "OFF"}, False),
                  (str(dimension), {"CHEMISTRY": "ON"}, False)]
    cases += [("3", {solver: "ON"}, True) for solver in solvers]
    for dimension, changes, succeeds in cases:
        command = common + [f"-DCOSMICTIGER_DIMENSION_COUNT={dimension}"]
        command += [f"-DCOSMICTIGER_ENABLE_{key}={value}" for key, value in changes.items()]
        result = subprocess.run(command + ["-P", str(script)], capture_output=True, text=True)
        if (result.returncode == 0) != succeeds:
            raise RuntimeError(f"Unexpected configure result D={dimension}, {changes}:\n{result.stdout}\n{result.stderr}")
    result = subprocess.run(common + ["-P", str(script)], capture_output=True, text=True, check=True)
    if "spatial dimensions: 3" not in result.stdout:
        raise RuntimeError("Default dimension is not 3")
    print(f"{len(cases) + 1} dimension capability checks passed")
