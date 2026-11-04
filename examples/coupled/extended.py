"""600-second regression; positive service floor fixed before execution.

Not a complete orbital mission, operational station deployment or unseen SLA.
"""
import argparse
import json
from pathlib import Path
from space_env.scaling.benchmark import run_scale
from space_env.scaling.coupled import coupled_workload
from space_env.simulation.snapshot import canonical_json


def manifest(configuration):
    workloads=[]
    for mode in ['nominal','all_outage']:
        cfg=configuration['development'] | {'id':'extended-clock-'+mode,
            'controller_clock':'fixed-tick/1','horizon_sec':600.,'decision_sec':2.,
            'mode':mode,'min_fulfilled':1 if mode=='nominal' else 0,'min_acquired':1}
        w=coupled_workload(cfg) | {'comparison_group':cfg['id']}
        restored=w | {'id':w['id']+'-restored','resume_after_decisions':80,'resume_in_new_process':True}
        workloads.extend([w,restored])
    return {'id':'extended-600s-fixed-clock','workers':2,'workloads':workloads}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('spec',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    with a.spec.open('rb') as stream:raw=stream.read(2_000_001)
    if len(raw)>2_000_000:raise ValueError('input exceeds 2MB')
    result=run_scale(manifest(json.loads(raw)),a.output)
    print(canonical_json({'passed':result['ok'],'sha256':result['sha256']}))
    raise SystemExit(0 if result['ok'] else 1)
