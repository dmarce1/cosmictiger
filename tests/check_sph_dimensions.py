"""Check retirement of the compile-time SPH dimension selector."""
import pathlib
import subprocess
import sys
import tempfile

source = pathlib.Path(sys.argv[1]).resolve()
features = source / "cmake/features.cmake"
with tempfile.TemporaryDirectory(prefix="sph_dimension_config_") as directory:
    script = pathlib.Path(directory) / "check.cmake"
    script.write_text(f'''include("{features}")
if(DEFINED COSMICTIGER_DIMENSION_COUNT OR DEFINED CACHE{{COSMICTIGER_DIMENSION_COUNT}})
    message(FATAL_ERROR "Retired dimension selector remained defined")
endif()
''')
    capabilities = ({}, {"GRAVITY": "OFF", "CHEMISTRY": "OFF", "STARS": "OFF"},
                    {"SPH": "OFF", "GRAVITY": "ON"})
    cases = [(dimension, changes, True) for dimension in (None, "3")
             for changes in capabilities]
    cases += [(value, {}, False) for value in ("1", "2", "0", "4", "-1", "2.0", "", "ON", "1;2")]
    for dimension, changes, succeeds in cases:
        command = ["cmake"]
        if dimension is not None:
            command.append(f"-DCOSMICTIGER_DIMENSION_COUNT={dimension}")
        command += [f"-DCOSMICTIGER_ENABLE_{key}={value}" for key, value in changes.items()]
        result = subprocess.run(command + ["-P", str(script)], capture_output=True, text=True)
        output = result.stdout + result.stderr
        if (result.returncode == 0) != succeeds:
            raise RuntimeError(f"Unexpected configure result D={dimension}, {changes}:\n{output}")
        if succeeds:
            if "SPH dimensions: runtime selection (1, 2, or 3)" not in output:
                raise RuntimeError(f"Missing runtime dimension guidance:\n{output}")
            if dimension == "3" and "Removing retired COSMICTIGER_DIMENSION_COUNT=3" not in output:
                raise RuntimeError(f"Missing migration notice:\n{output}")
        elif "COSMICTIGER_DIMENSION_COUNT is retired" not in output or "--hydro.dimensions" not in output:
            raise RuntimeError(f"Missing migration error:\n{output}")
    print(f"{len(cases)} dimension-selector migration checks passed")
