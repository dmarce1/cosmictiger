import json, os, subprocess
from pathlib import Path
repo=Path('/home/dmarce1/workspace/cosmictiger')
work=repo/'build/leaf_cloud_benchmark_20261018'
output=repo/'docs/leaf_cloud_benchmark_results_20261018'
runs=json.loads((output/'runs.json').read_text())
for variant in ['after','before']:
    directory=work/('diagnostic_'+variant)
    directory.mkdir(exist_ok=True)
    command=next(r['command'] for r in runs if r['variant']==variant)
    command=command[:3]+['gdb','--batch','-x',str(work/'crash_capture.gdb'),'--args']+command[3:]
    with (output/('diagnostic_'+variant+'.log')).open('w') as log:
        status=subprocess.run(command,cwd=directory,env=dict(os.environ,COSMICTIGER_STF_WALK_SAMPLES='100',OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2'),stdout=log,stderr=subprocess.STDOUT,timeout=180).returncode
    print(variant+' diagnostic status '+str(status),flush=True)
