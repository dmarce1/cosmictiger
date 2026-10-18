import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile

repo = Path('/home/dmarce1/workspace/cosmictiger')
build = repo / 'build/sph_validation'
work = Path(__file__).resolve().parent
baseline = work / 'baseline_source'
baseline.mkdir()
output = repo / 'docs/leaf_cloud_benchmark_results_20261018'
output.mkdir()
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
archive = subprocess.check_output(['git', 'archive', commit], cwd=repo)
with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
    stream.extractall(baseline, filter='data')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

changed = subprocess.check_output(['git', 'diff', '--name-only'], cwd=repo, text=True).splitlines()
allowed = {'cosmictiger/fmm_acceptance.hpp', 'src/kick.cpp', 'src/kick.cu',
           'tests/fmm_walk.cpp', 'tests/fmm_acceptance.cpp', 'tests/fmm_acceptance_cuda.cu'}
assert set(changed) == allowed, changed
patch = subprocess.check_output(['git', 'diff', '--binary'], cwd=repo)
(output / 'changes.patch').write_bytes(patch)
assert (repo / 'benchmarks/stf_walk.cpp').read_bytes() == (baseline / 'benchmarks/stf_walk.cpp').read_bytes()
pending = subprocess.check_output(['ninja', '-C', str(build), '-n', 'cosmictiger'], text=True)
assert 'no work to do' in pending, pending

app_sources = ['src/kick.cpp', 'src/kick.cu', 'tests/fmm_walk.cpp']
objects = {f'CMakeFiles/cosmictiger.dir/{path}.o': work / 'baseline_objects' / (path + '.o') for path in app_sources}
dependencies = subprocess.check_output(['ninja', '-C', str(build), '-t', 'deps'], text=True)
consumers = []
current = None
for line in dependencies.splitlines():
    if line and not line.startswith(' '):
        current = line.split(': #deps')[0]
    elif line.strip() == str(repo / 'cosmictiger/fmm_acceptance.hpp') and current.startswith('CMakeFiles/cosmictiger.dir/'):
        consumers.append(current)
assert set(consumers) == set(objects), consumers
entries = json.loads((build / 'compile_commands.json').read_text())
commands = []
for source in app_sources:
    entry = next(row for row in entries if row['file'] == str(repo / source))
    command = shlex.split(entry['command'])
    command = [part.replace(str(repo), str(baseline)) for part in command]
    target = objects[f'CMakeFiles/cosmictiger.dir/{source}.o']
    target.parent.mkdir(parents=True, exist_ok=True)
    for flag in ['-o', '-MF', '-MT']:
        if flag in command:
            command[command.index(flag) + 1] = str(target) + ('.d' if flag == '-MF' else '')
    commands.append(command)

def compile_object(item):
    source, command = item
    with (work / (Path(source).name + '.build.log')).open('w') as log:
        subprocess.run(command, cwd=build, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Built baseline ' + source, flush=True)

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(compile_object, zip(app_sources, commands)))

all_commands = subprocess.check_output(['ninja', '-C', str(build), '-t', 'commands', 'cosmictiger'], text=True).splitlines()
device = shlex.split(next(line for line in all_commands if ' -dlink ' in line))
link = all_commands[-1]
assert link.startswith(': && ') and link.endswith(' && :')
link = shlex.split(link[len(': && '):-len(' && :')])
device_output = 'CMakeFiles/cosmictiger.dir/cmake_device_link.o'
objects[device_output] = work / 'baseline_objects/cmake_device_link.o'
common_objects = {part: digest(build / part) for part in link if part.endswith('.o') and part not in objects}
device = [str(objects.get(part, part)) for part in device]
link = [str(objects.get(part, part)) for part in link]
link[link.index('-o') + 1] = str(work / 'cosmictiger_before')
for name, command in [('device_link', device), ('link', link)]:
    with (work / (name + '.build.log')).open('w') as log:
        subprocess.run(command, cwd=build, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Completed baseline ' + name, flush=True)
shutil.copy2(build / 'cosmictiger', work / 'cosmictiger_after')
manifest = {
    'baseline_commit': commit,
    'baseline_archive_sha256': hashlib.sha256(archive).hexdigest(),
    'patch_sha256': hashlib.sha256(patch).hexdigest(),
    'build_directory': str(build),
    'build_method': 'Fresh common objects; rebuild every changed-header application consumer from archived HEAD, then regenerate CUDA device link and final link.',
    'baseline_compile_commands': commands,
    'baseline_device_link_command': device,
    'baseline_link_command': link,
    'shared_object_sha256': common_objects,
    'replacement_objects': {part: {'before': digest(target), 'after': digest(build / part)} for part, target in objects.items()},
    'changed_source_sha256': {path: {'before': digest(baseline / path), 'after': digest(repo / path)} for path in changed},
    'harness_sha256': digest(repo / 'benchmarks/stf_walk.cpp'),
    'binaries': {variant: {'path': str(work / ('cosmictiger_' + variant)), 'sha256': digest(work / ('cosmictiger_' + variant))} for variant in ['before', 'after']},
}
(output / 'build_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
shutil.copy2(__file__, output / 'build_comparison.py')
shutil.copy2(build / 'CMakeCache.txt', output / 'CMakeCache.txt')
print('Comparison binaries ready: ' + str(output), flush=True)
