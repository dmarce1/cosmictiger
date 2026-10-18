import hashlib,json,os,shutil,subprocess
from pathlib import Path
repo=Path('/home/dmarce1/workspace/cosmictiger')
work=Path(__file__).resolve().parent
out=repo/'docs/leaf_cloud_cuda_profile_results_20261018'
base=json.loads((repo/'docs/leaf_cloud_benchmark_results_20261018/adaptive_suite/runs.json').read_text())
manifest=json.loads((work/'host_ranges/build_manifest.json').read_text())
assert manifest['completed']
ref=json.loads((out/'ablation_runs.json').read_text())
profiles=[]
for pair in range(2):
    for variant in (['before','after'] if pair==0 else ['after','before']):
        info=manifest['binaries'][variant]
        assert hashlib.sha256(Path(info['path']).read_bytes()).hexdigest()==info['sha256']
        name=f'host_pair{pair}_{variant}'
        folder=work/name
        folder.mkdir()
        prefix=folder/'walk'
        run_command=list(next(r['command'] for r in base if r['backend']=='cuda' and r['variant']==variant))
        run_command[3]=info['path']
        command=['/usr/local/cuda/bin/nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--force-overwrite=true','-o',str(prefix)]+run_command
        env=dict(os.environ,COSMICTIGER_STF_WALK_SAMPLES='15',COSMICTIGER_STF_WALK_FIXTURE='adaptive',OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2')
        log=out/(name+'.log')
        with log.open('w') as stream:
            subprocess.run(command,cwd=folder,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=120)
        forcehash=hashlib.sha256((folder/'stf_walk_forces.csv').read_bytes()).hexdigest()
        assert forcehash==next(r['force_sha256'] for r in ref if r['variant']==variant)
        export=['/usr/local/cuda/bin/nsys','export','--type=sqlite','--force-overwrite=true','-o',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep']
        with (out/(name+'.export.log')).open('w') as stream:
            subprocess.run(export,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=120)
        profiles.append({'name':name,'variant':variant,'pair':pair,'command':command,'samples':15,'sqlite':str(prefix)+'.sqlite','log':str(log),'force_csv_sha256':forcehash})
        (out/'host_profile_manifest.json').write_text(json.dumps(profiles,indent=2)+'\n')
        print(name+' complete; forces match',flush=True)
for source in ['build_manifest.json','source_changes.patch','build_host_ranges.py','profile_ranges.hpp']:
    shutil.copy2(work/'host_ranges'/source,out/('host_ranges_'+source))
shutil.copy2(__file__,out/'run_host_profiles.py')
print('Host profiles completed.',flush=True)
