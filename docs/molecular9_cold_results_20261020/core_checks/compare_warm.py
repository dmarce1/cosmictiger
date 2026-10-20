"""Compile archived pre-extension and current headers and compare warm outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[3])
parser.add_argument('--compiler', default='c++')
args = parser.parse_args()
archive = Path(__file__).resolve().parent
results = {}
with tempfile.TemporaryDirectory(prefix='molecular9_warm_comparison_') as temporary:
    temporary = Path(temporary)
    for variant in ('baseline', 'current'):
        executable = temporary / variant
        includes = ['-I', str(args.source_root)]
        if variant == 'baseline':
            includes = ['-I', str(archive / 'baseline_include')] + includes
        subprocess.run([args.compiler, '-std=c++20', '-O2', '-fno-fast-math',
                        '-ffp-contract=off', *includes, str(archive / 'warm_baseline.cpp'),
                        '-o', str(executable)], check=True)
        result = subprocess.run([str(executable)], check=True, capture_output=True)
        results[variant] = {'bytes': len(result.stdout),
                            'sha256': hashlib.sha256(result.stdout).hexdigest(),
                            'grid': result.stderr.decode().strip()}
    results['identical'] = results['baseline'] == results['current']
print(json.dumps(results, indent=2))
if not results['identical']:
    raise SystemExit('Warm outputs changed')
