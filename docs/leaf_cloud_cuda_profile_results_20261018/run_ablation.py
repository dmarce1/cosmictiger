import hashlib,json,os,shutil,subprocess
from pathlib import Path
repo=Path('/home/dmarce1/workspace/cosmictiger')
work=Path(__file__).resolve().parent
out=repo/'docs/leaf_cloud_cuda_profile_results_20261018'
prior=repo/'docs/leaf_cloud_benchmark_results_20261018/adaptive_suite'
base=json.loads((prior/'build_manifest.json').read_text())
leaf=json.loads((work/'leaf_only/build_manifest.json').read_text())
assert leaf['completed']
binaries=dict(base['binaries'],leaf_only=leaf['binary'])
for info in binaries.values():
    assert hashlib.sha256(Path(info['path']).read_bytes()).hexdigest()==info['sha256']
base_command=next(r['command'] for r in json.loads((prior/'runs.json').read_text()) if r['backend']=='cuda')
orders=[['before','leaf_only','after'],['after','leaf_only','before'],['leaf_only','before','after']]
runs=[]
for pair,order in enumerate(orders):
    for variant in order:
        name=f'ablation_pair{pair}_{variant}'
        folder=work/name
        folder.mkdir()
        command=list(base_command)
        command[3]=binaries[variant]['path']
        env=dict(os.environ,COSMICTIGER_STF_WALK_SAMPLES='15',COSMICTIGER_STF_WALK_FIXTURE='adaptive',OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2')
        log=out/(name+'.log')
        with log.open('w') as stream:
            subprocess.run(command,cwd=folder,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=90)
        medians=[]
        samples=[]
        for line in log.read_text().splitlines():
            fields=line.split(',')
            if fields[0]=='STF_WALK_MEDIAN':
                medians.append({'potential':int(fields[3]),'tree_seconds':float(fields[7]),'kick_seconds':float(fields[8]),'total_seconds':float(fields[9])})
            elif fields[0]=='STF_WALK_SAMPLE':
                samples.append({'potential':int(fields[3]),'sample':int(fields[4]),'kick_seconds':float(fields[7])})
        assert len(medians)==2 and len(samples)==32
        force=folder/'stf_walk_forces.csv'
        runs.append({'variant':variant,'pair':pair,'command':command,'medians':medians,'samples':samples,'force_csv':str(force),'force_sha256':hashlib.sha256(force.read_bytes()).hexdigest()})
        (out/'ablation_runs.json').write_text(json.dumps(runs,indent=2)+'\n')
        print(name+': '+', '.join(f'phi={r["potential"]} {1000*r["kick_seconds"]:.3f}ms' for r in medians),flush=True)
shutil.copy2(__file__,out/'run_ablation.py')
shutil.copy2(work/'leaf_only/build_manifest.json',out/'leaf_only_build_manifest.json')
shutil.copy2(work/'leaf_only/source_changes.patch',out/'leaf_only_source_changes.patch')
print('Ablation complete.',flush=True)
