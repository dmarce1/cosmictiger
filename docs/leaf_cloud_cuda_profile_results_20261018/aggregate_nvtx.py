#!/usr/bin/env python3
"""Aggregate existing host NVTX analyses, with overlap-preserving wall metrics."""
import collections
import json
import pathlib
import statistics

base=pathlib.Path(__file__).resolve().parent
manifest=json.loads((base/'host_profile_manifest.json').read_text())
records=[]
labels=set()
for profile in manifest:
    data=json.loads((base/(profile['name']+'_analysis.json')).read_text())
    for original in data['launches']:
        if original['warmup']:
            continue
        record=dict(original,variant=profile['variant'],pair=profile['pair'])
        labels.update(record['labels'])
        # Overlap is retained: CP outside Ewald still can overlap another CPU stage.
        overlap=collections.Counter()
        for segment in record['wall_by_active_labels_ms']:
            active=set(segment['labels']);ms=segment['duration_ms']
            cp='cpu.gravity.cp.internal' in active
            walk='cpu.walk.direct' in active
            ewald='cpu.gravity.cc.ewald' in active
            if cp:
                overlap['cp_with_ewald_ms' if ewald else 'cp_without_ewald_ms']+=ms
            if walk:
                overlap['walk_with_ewald_ms' if ewald else 'walk_without_ewald_ms']+=ms
            if cp or walk:
                overlap['cp_or_walk_with_ewald_ms' if ewald else 'cp_or_walk_without_ewald_ms']+=ms
        record['overlap']=overlap
        records.append(record)
scalar_metrics=['kick_ms','root_ms','kernel_ms','pre_kernel_ms','post_kernel_ms','root_minus_stdout_ms','uncovered_wall_ms']
label_metrics=['count','summed_range_ms','union_ms','union_pre_kernel_ms','union_during_kernel_ms','union_post_kernel_ms']
overlap_metrics=['cp_with_ewald_ms','cp_without_ewald_ms','walk_with_ewald_ms','walk_without_ewald_ms','cp_or_walk_with_ewald_ms','cp_or_walk_without_ewald_ms']

def compare(group,value):
    variants={}
    for variant in ['before','after']:
        values=[value(r) for r in group if r['variant']==variant]
        variants[variant]={'n':len(values),'mean':statistics.mean(values),'median':statistics.median(values)}
    return dict(**variants,mean_delta=variants['after']['mean']-variants['before']['mean'])

result={}
for phi in [0,1]:
    group=[r for r in records if r['do_phi']==phi]
    value={'scalars':{metric:compare(group,lambda r,m=metric:r[m]) for metric in scalar_metrics},
           'labels':{},'overlap':{metric:compare(group,lambda r,m=metric:r['overlap'][m]) for metric in overlap_metrics}}
    for label in sorted(labels):
        value['labels'][label]={metric:compare(group,lambda r,l=label,m=metric:r['labels'][l][m]) for metric in label_metrics}
        value['labels'][label]['payload_uint64_sum']=compare(group,lambda r,l=label:r['labels'][l].get('payload_sums',{}).get('uint64Value',0))
    value['pairs']={str(pair):{metric:compare([r for r in group if r['pair']==pair],lambda r,m=metric:r[m])
                   for metric in ['root_ms','pre_kernel_ms','kernel_ms','post_kernel_ms']} for pair in [0,1]}
    result[str(phi)]=value
payload={'method':'Two alternating pairs, 15 measured samples per phi and variant per pair, one warmup excluded per phi. Arithmetic means of 30 samples; scope payloads are counts of nodes/workitems, not particle interactions.',
         'conclusion':'New CPU internal CP operators and increased direct-walk classification time delay the GPU launch. CPU Ewald operator workload, GPU preparation, transfers, and workitem count stay essentially unchanged.',
         'caveats':['Per-label union_ms removes self-overlap across workers, but different labels still overlap and must not be added.',
                    'Summed range time can include HPX suspension and does not measure active CPU utilization.',
                    'NVTX instrumentation changes absolute timing; original uninstrumented comparison is the performance result.',
                    'CPU direct-walk range covers acceptance/list classification as a whole; individual opening tests are not separately timed.',
                    'CP-without-Ewald is an overlap diagnostic, not proof of a task dependency critical path.'],
         'groups':result}
(base/'nvtx_aggregate.json').write_text(json.dumps(payload,indent=2)+'\n')
lines=['# Host-stage attribution','','The extra CUDA-mode wall time arises mainly before the GPU launch: new internal-node CP work on the CPU plus more expensive direct-walk classification.','','Two alternating pairs × 15 measured samples/mode/variant; these timings include NVTX profiling overhead.','','| Mode | Root phase | Before ms | After ms | Change ms |','|---|---|---:|---:|---:|']
for phi in [0,1]:
    for metric,label in [('root_ms','Root kick'),('pre_kernel_ms','Before kernel'),('kernel_ms','GPU kernel'),('post_kernel_ms','After kernel')]:
        m=result[str(phi)]['scalars'][metric]
        lines.append(f"| {'Force + potential' if phi else 'Force'} | {label} | {m['before']['mean']:.4f} | {m['after']['mean']:.4f} | {m['mean_delta']:+.4f} |")
lines+=['','Stage wall coverage below is the interval union across both CPU workers. Different rows overlap and must not be added. Summed scope durations are in the JSON.','','| Stage | Force before → after ms | Force + potential before → after ms | Calls/kick before → after | Node/workitem payload before → after |','|---|---:|---:|---:|---:|']
for label in ['cpu.gravity.cp.internal','cpu.walk.direct','cpu.gravity.cc.direct','cpu.gravity.cc.ewald','cpu.walk.ewald','gpu.add_work.set_insert','gpu.prepare','gpu.advice_to_gpu','gpu.execute','gpu.restore']:
    q0=result['0']['labels'][label];q1=result['1']['labels'][label]
    def pair(q,key,digits=4):
        m=q[key];return f"{m['before']['mean']:.{digits}f} → {m['after']['mean']:.{digits}f}"
    lines.append(f"| `{label}` | {pair(q0,'union_ms')} | {pair(q1,'union_ms')} | {pair(q0,'count',0)} | {pair(q0,'payload_uint64_sum',0)} |")
lines+=['','New internal CP costs about 2 ms in summed scope time per kick. Its wall coverage is about 1.85 ms, of which 0.82–0.89 ms overlaps Ewald CC on the other worker. Direct classification adds about 0.54 ms of wall coverage despite receiving fewer checklist nodes. These overlapping costs produce a net 1.14–1.18 ms increase before kernel start.','','No CPU leaf CP/PC/PP scopes execute in either variant on this CUDA workload; those leaves are handed to the GPU. Both versions hand over 736 workitems. GPU preparation stays unchanged and checklist set insertion becomes slightly faster.','','Caveats:','']
lines+=['- '+c for c in payload['caveats']]
(base/'nvtx_summary.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
