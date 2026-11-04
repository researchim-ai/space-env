"""Fresh-process recovery during concurrent rotating burn and joint RCS control.

The repeated-action baseline is a recovery regression, not a service/teacher policy.
"""
import argparse
import json
from pathlib import Path

from space_env.scaling.benchmark import run_scale
from space_env.validation.coupled_actuators import actuator_scenario
from space_env.simulation.snapshot import canonical_json,digest


def manifest(configuration):
    scenario=actuator_scenario(configuration['development'])
    a,b=sorted(scenario['actor_order'])[:2]
    task={'id':'actuator-midphase','runtime_audit':'compact',
        'roles':{a:{'actors':[a]},b:{'actors':[b]}},'decision_sec':.057,
        'max_decision_sec':.057,'max_decisions':60,'max_actions':3,'max_joint_actions':2,
        'actions':[
            {'id':'engine','role':a,'kind':'submit','commands':[{'command_id':'engine','actor_id':a,
             'kind':'body_burn','duration_sec':1.,'body_burn':{'frame':'BODY',
             'direction_body':[0.,0.,1.],'thrust_n':.01,'firing_sec':.4}}]},
            {'id':'unload','role':b,'kind':'submit','commands':[{'command_id':'unload','actor_id':b,
             'kind':'momentum_unload','duration_sec':2.,'momentum_unload':{'frame':'BODY',
             'target_momentum_nms':[0.,0.,0.],'tolerance_nms':1e-5}}]},
        ]}
    w={'id':'actuator-midphase','family':'fixed-clock-actuator-restore/1',
       'source':'frozen coupled synthetic hardware; mid-firing and mid-RCS tick recovery',
       'scenario':scenario,'task':task,'policy':{'name':'priority'},'timeout_sec':600,
       'sample_numerics':True,'comparison_group':'actuator-midphase',
       'coupled_gates':{'native_spec_sha256':digest(scenario['definition'])}}
    restored=w|{'id':w['id']+'-restored','resume_after_decisions':3,'resume_in_new_process':True}
    return {'id':'actuator-midphase-recovery','workers':1,'workloads':[w,restored]}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('spec',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    with a.spec.open('rb') as stream:raw=stream.read(2_000_001)
    if len(raw)>2_000_000:raise ValueError('input exceeds 2MB')
    result=run_scale(manifest(json.loads(raw)),a.output)
    print(canonical_json({'passed':result['ok'],'sha256':result['sha256']}))
    raise SystemExit(0 if result['ok'] else 1)
