import hashlib, json, os, shutil, subprocess, time
from pathlib import Path
repo = Path('/home/dmarce1/workspace/cosmictiger')
work = Path(__file__).resolve().parent
output = repo / 'docs/leaf_cloud_cuda_profile_results_20261018'
output.mkdir(exist_ok=True)
prior = repo / 'docs/leaf_cloud_benchmark_results_20261018/adaptive_suite'
manifest = json.loads((prior / 'build_manifest.json').read_text())
runs = json.loads((prior / 'runs.json').read_text())
for variant, info in manifest['binaries'].items():
    assert hashlib.sha256(Path(info['path']).read_bytes()).hexdigest() == info['sha256']
commands = {variant:next(r['command'] for r in runs if r['backend']=='cuda' and r['variant']==variant) for variant in ['before','after']}
env = dict(os.environ, COSMICTIGER_STF_WALK_FIXTURE='adaptive', OMP_NUM_THREADS='2', OMP_THREAD_LIMIT='2')
metadata = {'baseline_commit':manifest['baseline_commit'], 'binaries':manifest['binaries'], 'profiles':[]}
(output / 'profile_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
shutil.copy2(__file__, output / 'run_profiles.py')
for pair in range(3):
    for variant in (['before','after'] if pair%2==0 else ['after','before']):
        name=f'nsys_pair{pair}_{variant}'
        folder=work/name
        folder.mkdir()
        prefix=folder/'walk'
        command=['/usr/local/cuda/bin/nsys','profile','--trace=cuda','--sample=none','--cpuctxsw=none','--force-overwrite=true','-o',str(prefix)] + commands[variant]
        env['COSMICTIGER_STF_WALK_SAMPLES']='15'
        start=time.monotonic()
        with (output/(name+'.log')).open('w') as log:
            subprocess.run(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
        export=['/usr/local/cuda/bin/nsys','export','--type=sqlite','--force-overwrite=true','-o',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep']
        with (output/(name+'.export.log')).open('w') as log:
            subprocess.run(export,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
        metadata['profiles'].append({'name':name,'kind':'nsys','variant':variant,'pair':pair,'command':command,'export_command':export,'environment':{k:env[k] for k in ['COSMICTIGER_STF_WALK_SAMPLES','COSMICTIGER_STF_WALK_FIXTURE','OMP_NUM_THREADS','OMP_THREAD_LIMIT']},'sqlite':str(prefix)+'.sqlite','log':str(output/(name+'.log')),'wall_seconds':time.monotonic()-start})
        (output/'profile_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(name+' complete',flush=True)
for variant in ['before','after']:
    name='ncu_'+variant
    folder=work/name
    folder.mkdir()
    prefix=folder/'walk'
    sections=['SpeedOfLight','LaunchStats','Occupancy','ComputeWorkloadAnalysis','MemoryWorkloadAnalysis','InstructionStats','SchedulerStats','WarpStateStats','SourceCounters']
    command=['/usr/local/cuda/bin/ncu','--kernel-name','cudaKickKernel','--launch-skip','1','--launch-count','1','--clock-control','none','--cache-control','none','--import-source','yes','--force-overwrite','-o',str(prefix)]
    for section in sections:
        command+=['--section',section]
    command += commands[variant]
    env['COSMICTIGER_STF_WALK_SAMPLES']='1'
    with (output/(name+'.log')).open('w') as log:
        result=subprocess.run(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    metadata['profiles'].append({'name':name,'kind':'ncu','variant':variant,'command':command,'samples':1,'launch_skip':1,'launch_count':1,'report':str(prefix)+'.ncu-rep','exit_code':result.returncode})
    (output/'profile_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(name+' exit '+str(result.returncode),flush=True)
    if result.returncode:
        continue
    for page in ['details','raw','source']:
        export=['/usr/local/cuda/bin/ncu','--import',str(prefix)+'.ncu-rep','--page',page,'--csv']
        if page=='source':
            export+=['--print-source','cuda,sass']
        with (output/(name+'_'+page+'.csv')).open('w') as log:
            subprocess.run(export,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
print('Profiling finished.',flush=True)
