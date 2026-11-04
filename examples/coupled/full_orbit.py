"""Two complete orbit cycles, paid kinematic ADCS and separate release epochs.

Synthetic hardware and fixed global grid; not an operational station network.
All service floors are frozen before execution. Failed attempts remain in output.
"""
import argparse
import copy
import json
from pathlib import Path

from space_env.scaling.benchmark import run_scale
from space_env.scaling.full_orbit import full_orbit_workload, full_orbit_geometry
from space_env.simulation.snapshot import atomic_write, canonical_json


def configuration():
    sites=[{'name':f'g{i}','lat_deg':0.,'lon_deg':float(lon)} for i,lon in enumerate(range(-180,180,45))]
    return {'id':'two-orbit-kinematic','planes':[
        {'id':'a','inclination_deg':55.,'raan_deg':0.,'phases_deg':[0.]},
        {'id':'b','inclination_deg':75.,'raan_deg':120.,'phases_deg':[0.]}],
        'targets':[{'id':f't{i}','lat_deg':s['lat_deg'],'lon_deg':s['lon_deg']} for i,s in enumerate(sites)],
        'stations':sites,'horizon_sec':12600.,'cycles':2,'min_fulfilled':2,'min_acquired':2,
        'min_delivery_epochs':2,'attitude_model':'rate-limited-kinematic/1'}


def manifest(value):
    workloads=[]
    for mode in ('nominal','all_outage'):
        cfg=copy.deepcopy(value)
        cfg.update(id=value['id']+'-'+mode,mode=mode)
        if mode=='all_outage':cfg.update(min_fulfilled=0,min_delivery_epochs=0)
        w=full_orbit_workload(cfg)
        w['comparison_group']=cfg['id']
        prefix=int((cfg['horizon_sec']/cfg['cycles'])/w['task']['decision_sec'])
        restored=w|{'id':w['id']+'-restored','resume_after_decisions':prefix,'resume_in_new_process':True}
        workloads.extend((w,restored))
    return {'id':'full-orbit-campaign','workers':4,'workloads':workloads}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    p.add_argument('--write-spec',action='store_true');p.add_argument('--spec',type=Path)
    p.add_argument('--geometry-only',action='store_true')
    args=p.parse_args()
    cfg=configuration()
    if args.spec:
        with args.spec.open('rb') as stream:raw=stream.read(2_000_001)
        if len(raw)>2_000_000:raise ValueError('input exceeds 2MB')
        cfg=json.loads(raw)
    if args.write_spec:
        atomic_write(args.output,canonical_json(cfg))
    elif args.geometry_only:
        diagnostic=full_orbit_geometry(cfg);atomic_write(args.output,canonical_json(diagnostic))
        raise SystemExit(0 if diagnostic['passed'] else 1)
    else:
        result=run_scale(manifest(cfg),args.output)
        print(canonical_json({'passed':result['ok'],'sha256':result['sha256']}))
        raise SystemExit(0 if result['ok'] else 1)
