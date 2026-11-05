"""Exercise Cartesian SPH startup, evolution, padded storage, and restart."""
import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess

from checkpoint_format import Header, decode_checkpoint
from run_checkpoint import fnv1a64

parser = argparse.ArgumentParser()
parser.add_argument("executable", type=Path)
parser.add_argument("workdir", type=Path)
parser.add_argument("dimension", type=int, choices=(1, 2))
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--integrator", choices=("traditional", "global"), default="traditional")
args = parser.parse_args()
root = args.workdir.resolve()
root.mkdir(parents=True, exist_ok=True)
common = [str(args.executable.resolve()), "--hpx:threads=2", "--hpx:bind=none",
          f"--run.timeIntegrator={args.integrator}", "--run.darkMatterParticleCount=0",
          f"--run.gasParticleCount={128 if args.dimension == 1 else 1024}",
          "--gravity.enabled=false", "--chemistry.enabled=false", "--starFormation.enabled=false",
          "--hydro.enabled=true", f"--hydro.cuda={'true' if args.cuda else 'false'}",
          "--drift.cuda=false", "--output.silo=false", "--output.directory=.",
          "--run.endTime=0.001", "--run.numberOfSteps=2", "--checkpoint.intervalSeconds=0"]
environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

def run(name, scenario="hydroWave", options=(), succeeds=True, directory=None):
    if directory is None:
        directory = root / name
        if directory.exists():
            shutil.rmtree(directory)
    directory.mkdir(exist_ok=True)
    result = subprocess.run(common + [f"--run.scenario={scenario}", *options], cwd=directory,
                            env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=120)
    (directory / f"{name}.log").write_text(result.stdout)
    if (result.returncode == 0) != succeeds:
        raise RuntimeError(f"{name}: exit {result.returncode}\n{result.stdout[-4000:]}")
    return directory

def state(directory):
    checkpoints = sorted(directory.glob("checkpoint.*/checkpoint.0.dat"),
                         key=lambda path: int(path.parent.name.split(".")[1]))
    result = decode_checkpoint(checkpoints[-1].read_bytes())
    assert result.dimension_count == args.dimension
    assert result.gas_count == result.count > 0
    assert all(math.isfinite(value) and value > 0 for value in result.densities)
    assert all(math.isfinite(value) and value > 0 for value in result.entropies)
    assert result.thermal_scale_factor == 1
    for dim in range(args.dimension, 3):
        assert set(result.positions[dim]) == {1 << (8 * result.position_bytes - 1)}
        assert all(result.velocities[3 * particle + dim] == 0 for particle in range(result.count))
    for dim in range(args.dimension):
        assert abs(sum(result.velocities[dim::3])) / result.count < 2e-5
    return result

for viscosity in ("none", "morrisAndMonaghan", "cullenAndDehnen", "huEtal", "priceEtAl"):
    result = state(run("wave_" + viscosity, options=(f"--hydro.viscosityEvolution={viscosity}",)))
    assert min(result.densities) > 0.5 and max(result.densities) < 2
for scenario in ("sod", "blast") + (("kelvinHelmholtz", "gresho") if args.dimension == 2 else ()):
    directory = run(scenario, scenario)
    # KH's unequal layer densities intentionally carry net streamwise momentum.
    if scenario == "kelvinHelmholtz":
        checkpoint = sorted(directory.glob("checkpoint.*/checkpoint.0.dat"))[-1]
        result = decode_checkpoint(checkpoint.read_bytes())
        assert all(math.isfinite(value) and value > 0 for value in result.densities)
        assert all(value == 0 for value in result.velocities[2::3])
    else:
        state(directory)

continuous = state(root / "wave_none")
stopped = run("restart", options=("--hydro.viscosityEvolution=none", "--run.maxIterations=1"))
run("resumed", options=("--hydro.viscosityEvolution=none", "--checkpoint.readIndex=1"), directory=stopped)
resumed = state(stopped)
assert continuous.positions == resumed.positions
# Near-zero transverse velocities arise from cancellation of finite pair
# impulses. Bound their roundoff by the vector velocity scale, not the tiny
# residual in that component; positions and thermodynamics remain exact.
velocity_scale = max(max(map(abs, continuous.velocities)), 1e-12)
velocity_error = max(abs(first - second) for first, second in
                     zip(continuous.velocities, resumed.velocities))
assert velocity_error <= 4 * 2**-23 * velocity_scale
assert continuous.densities == resumed.densities
assert continuous.smoothing_lengths == resumed.smoothing_lengths
print(f"Restart maximum velocity difference: {velocity_error:.3e}")
assert continuous.entropies == resumed.entropies

# Alter only the saved dimension tag to exercise rejection before payload reads.
mismatched = root / "wrong_dimension"
if mismatched.exists():
    shutil.rmtree(mismatched)
shutil.copytree(root / "wave_none", mismatched)
checkpoint = sorted(mismatched.glob("checkpoint.*/checkpoint.0.dat"))[-1]
data = bytearray(checkpoint.read_bytes())
header = Header.from_buffer(data)
other_dimension = 2 if args.dimension == 1 else 1
header.reserved = (header.reserved & ~(3 << 28)) | ((3 - other_dimension) << 28)
checkpoint.write_bytes(data)
manifest = checkpoint.parent / "checkpoint.meta"
lines = manifest.read_text().splitlines()
lines[1] = f"0 {len(data)} {fnv1a64(data):016x}"
manifest.write_text("\n".join(lines) + "\n")
run("wrong_dimension", options=("--hydro.viscosityEvolution=none",
    f"--checkpoint.readIndex={checkpoint.parent.name.split('.')[1]}"),
    succeeds=False, directory=mismatched)
assert "Checkpoint spatial dimension" in (mismatched / "wrong_dimension.log").read_text()

# Runtime capability rejection complements the configure-time gravity checks.
run("central_gravity_rejected", options=("--hydro.centralAcceleration=1",), succeeds=False)
if args.dimension == 1:
    run("gresho_rejected", "gresho", succeeds=False)
print(f"D={args.dimension} SPH scenarios, viscosity modes, inactive components, and restart within float accumulation roundoff passed")
