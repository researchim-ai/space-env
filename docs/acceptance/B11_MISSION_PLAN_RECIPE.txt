"""Retain a proposal, execute its fixed plan in fresh processes, export RL/SFT.

The candidate comes from the existing causal rolling baseline, not an optimal
forecast planner. Native goal feasibility and teacher expertise are distinct.
"""
import argparse
import hashlib
import math
import os
import subprocess
import sys
from pathlib import Path

from space_env.planning import PlanningRun
from space_env.planning.mission_plan import recorded_mission_plan
from space_env.planning.mission_bundle import read_json
from space_env.scaling.coupled import coupled_workload
from space_env.scaling.full_orbit import full_orbit_workload
from space_env.simulation.snapshot import atomic_write,canonical_json,digest


def configuration(full=False,negative=False):
    step=45 if full else 30
    sites=[{'name':f'g{i:02d}','lat_deg':0.,'lon_deg':float(lon)}
           for i,lon in enumerate(range(-180,180,step))]
    cfg={'id':'retained-plan','planes':[
        {'id':'a','inclination_deg':55.,'raan_deg':0.,'phases_deg':[0.]},
        {'id':'b','inclination_deg':75.,'raan_deg':120.,'phases_deg':[0.]}],
        'targets':[{'id':f't{i:02d}','lat_deg':s['lat_deg'],'lon_deg':s['lon_deg']} for i,s in enumerate(sites)],
        'stations':sites,'controller_clock':'fixed-tick/1',
        'mode':'all_outage' if negative else 'nominal',
        'min_fulfilled':0 if negative else (2 if full else 1),'min_acquired':2 if full else 1}
    if full:cfg.update(horizon_sec=12600.,attitude_model='rate-limited-kinematic/1',
                       min_delivery_epochs=0 if negative else 2)
    return cfg


def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    parser.add_argument('--full-orbit',action='store_true');parser.add_argument('--negative',action='store_true')
    parser.add_argument('--native-audit',action='store_true')
    args=parser.parse_args();root=args.output
    if root.exists():raise ValueError('output must not exist')
    cfg=configuration(args.full_orbit,args.negative)
    w=(full_orbit_workload if args.full_orbit else coupled_workload)(cfg)
    n=len(w['scenario']['definition']['lifecycle']['requests'])
    goals={'min_fulfilled':1 if args.negative else math.ceil(w['min_completion_fraction']*n-1e-9),
           'coupled':w['coupled_gates']}
    root.mkdir(parents=True)
    def stage(name):atomic_write(root/'progress.json',canonical_json({'stage':name,'pid':os.getpid()}))
    atomic_write(root/'configuration.json',canonical_json(cfg))
    atomic_write(root/'workload.json',canonical_json(w));atomic_write(root/'goals.json',canonical_json(goals))
    stage('causal_candidate')
    run=PlanningRun(w['scenario'],w['task'],w['policy']);run.run()
    candidate=run.save(root/'proposal.json',compact=True)
    plan=recorded_mission_plan(candidate,goals=goals)
    atomic_write(root/'plan.json',canonical_json(plan))
    del run,candidate
    prefix=max(1,len(plan['steps'])//2)
    def call(name,arguments,expected=0):
        cmd=[sys.executable,*(['-I'] if sys.flags.isolated else []),'-m','space_env.cli',*map(str,arguments)]
        process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:stdout,stderr=process.communicate(timeout=3600)
        except subprocess.TimeoutExpired:
            process.kill();stdout,stderr=process.communicate()
            atomic_write(root/(name+'.stderr.txt'),stderr)
            raise RuntimeError(name+' timed out')
        atomic_write(root/(name+'.stdout.txt'),stdout);atomic_write(root/(name+'.stderr.txt'),stderr)
        if process.returncode!=expected:raise RuntimeError(name+' unexpected exit '+str(process.returncode)+': '+stderr[-2048:])
        return process.pid
    stage('prefix')
    producer=call('prefix',['mission-plan-run',root/'plan.json',root/'prefix.json','--decisions',prefix])
    stage('fresh_process_continuation')
    consumer=call('resume',['mission-plan-resume',root/'plan.json',root/'prefix.json',root/'execution.json'])
    record=read_json(root/'execution.json');recovery=record['recovery']
    if recovery['producer_pid']!=producer or recovery['consumer_pid']!=consumer or producer==consumer:
        raise ValueError('parent-observed recovery PIDs mismatch')
    stage('native_replay_and_export')
    call('export',['mission-plan-export',root/'plan.json',root/'execution.json',root/'dataset'],expected=1 if args.negative else 0)
    stage('audit')
    call('audit',['verify-mission-plan-bundle',root/'dataset',*(['--native'] if args.native_audit else [])],expected=1 if args.negative else 0)
    cert=read_json(root/'dataset/certificate.json');audit=read_json(root/'audit.stdout.txt')
    expected_gates=all(value for name,value in cert['gates'].items()
                       if not (args.negative and name=='minimum_service'))
    observed=(cert['passed'] is not args.negative and audit['integrity_passed']
              and recovery['fresh_process'] and cert['gates']['expected_native_state']
              and expected_gates)
    stage('complete' if observed else 'failed')
    files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
           for p in root.rglob('*') if p.is_file()}
    body={'schema':'space-env.retained-plan-recipe/1','passed':observed,
          'mission_goal_passed':cert['passed'],'expected_negative_control':args.negative,
          'plan_sha256':plan['sha256'],'certificate_sha256':cert['sha256'],
          'fulfilled':cert['fulfilled'],'total':cert['total'],'parent_observed_pids':{'producer':producer,'consumer':consumer},
          'native_audit':args.native_audit,'files_sha256':files,'expert_teacher_qualified':False}
    atomic_write(root/'result.json',canonical_json(body|{'sha256':digest(body)}))
    print(canonical_json(body|{'files_sha256':'retained in result.json'}))
    raise SystemExit(0 if observed else 1)


if __name__=='__main__':main()
