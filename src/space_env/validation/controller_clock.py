"""Frozen-input command/interrupt partition check; no policy or oracle tuning."""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

from space_env.runtime import AccessPolicy, MissionRuntime
from space_env.scaling.coupled import CoupledGates, coupled_evidence
from space_env.scaling.numerics import _qdistance
from space_env.simulation.snapshot import atomic_write, canonical_json, digest


BUDGETS = {'attitude_deg': 1e-7, 'body_rate_rad_s': 1e-9,
           'wheel_momentum_nms': 1e-9, 'dc_energy_j': 1e-8, 'time_sec': 1e-9}
PARTITIONS = (.1, .025, .01)
CHECKPOINTS = (.137, .211, .263, .41, .6)


def qualify_controller_clock(scenario, output):
    """Run identical native commands under three caller partitions and replay.

    Clock period is fixed by the input; varying caller boundaries is not changing
    controller cadence. This short actuator check does not certify imaging/SLA.
    """
    scenario = copy.deepcopy(scenario)
    native = scenario['definition']
    actors = sorted(native['orbits'])
    if (scenario.get('backend') != 'reference-spacecraft' or not 2 <= len(actors) <= 4
        or not 1. <= native['horizon_sec'] <= 600. or native['operations']['products']
        or len(native.get('lifecycle', {}).get('requests', [])) > 64
        or len(native.get('lifecycle', {}).get('jobs', [])) > 64
        or any(c['attitude'].get('rigid', {}).get('clock_mode') != 'fixed-tick/1'
               for c in native['spacecraft_bus'].values())):
        raise ValueError('clock program requires bounded empty-onboard fixed-tick coupled scenario')
    path = Path(output)
    if path.exists() and any(path.iterdir()):
        raise ValueError('clock program output must be empty')
    path.mkdir(parents=True, exist_ok=True)
    atomic_write(path/'scenario.json', canonical_json(scenario))
    rows, hashes = [], {}
    for index, partition in enumerate(PARTITIONS):
        try:
            m = MissionRuntime(scenario, access=AccessPolicy(truth=True), audit_mode='compact')
            serial = 0
            def call(op, **kwargs):
                nonlocal serial
                serial += 1
                return getattr(m,op)(expected_revision=m.revision,
                                     idempotency_key=f'program/{serial}', **kwargs)
            def submit(id, actor, kind, **kwargs):
                call('submit', commands=[{'command_id':id, 'actor_id':actor,'kind':kind,
                                          'duration_sec':1., **kwargs}])
            submit('slew',actors[0],'slew',slew={'frame':'GCRS', 'quaternion_wxyz':[math.sqrt(.5),math.sqrt(.5),0.,0.]})
            submit('wheel',actors[1],'wheel_torque',wheel_torque={'frame':'BODY','motor_torque_nm':[.02,-.01,0.]})
            snapshots = []
            for target in CHECKPOINTS:
                while target - m.time_sec > 1e-9:
                    call('advance', dt_sec=min(partition,target-m.time_sec))
                snapshots.append({'time_sec':m.time_sec,'actors':{
                    a:m.observe(a)['actors'][a]['spacecraft_bus']['rigid_attitude'] for a in actors}})
                if target == .137:
                    call('cancel',command_id='wheel')
                    submit('wheel2',actors[1],'wheel_torque',wheel_torque={'frame':'BODY','motor_torque_nm':[-.015,0.,0.]})
                elif target == .211:
                    prefix=m.checkpoint(compact=True)
                    m=MissionRuntime.from_checkpoint(prefix,access=AccessPolicy(truth=True))
                elif target == .263:
                    call('cancel',command_id='slew')
                    submit('slew2',actors[0],'slew',slew={'frame':'GCRS','quaternion_wxyz':[math.sqrt(.5),0.,math.sqrt(.5),0.]})
                elif target == .41:
                    call('inject',events=[{'id':'safe','type':'spacecraft_safe_mode','time_sec':m.time_sec,
                                          'source':'frozen-clock-program/1','reason':'mid-tick protection',
                                          'payload':{'actor':actors[1]}}])
            final=path/f'final-{index}.json'
            m.save(final,compact=True)
            restored=MissionRuntime.load(final,access=AccessPolicy(truth=True))
            replay=restored.checkpoint(compact=True)==m.checkpoint(compact=True)
            evidence=coupled_evidence(scenario,m._backend.report(),CoupledGates(native_spec_sha256=digest(native)))
            hashes[final.name]=hashlib.sha256(final.read_bytes()).hexdigest()
            rows.append({'partition_sec':partition,'snapshots':snapshots,'native_replay':replay,
                         'witness':evidence,'final_sha256':m.checkpoint(compact=True)['sha256']})
        except (ValueError, RuntimeError, KeyError) as exc:
            rows.append({'partition_sec':partition, 'snapshots':[], 'native_replay':False,
                         'witness':{'passed':False}, 'error':str(exc)})
    comparisons=[]
    for row in rows[1:]:
        if len(rows[0]['snapshots']) != len(CHECKPOINTS) or len(row['snapshots']) != len(CHECKPOINTS):
            comparisons.append({'partition_sec':row['partition_sec'], 'passed':False,
                                'reason':'missing_execution; retained in denominator'})
            continue
        maxima=dict.fromkeys(BUDGETS,0.)
        for a,b in zip(rows[0]['snapshots'],row['snapshots']):
            maxima['time_sec']=max(maxima['time_sec'],abs(a['time_sec']-b['time_sec']))
            for actor in actors:
                x,y=a['actors'][actor],b['actors'][actor]
                maxima['attitude_deg']=max(maxima['attitude_deg'],_qdistance(x['quaternion_wxyz'],y['quaternion_wxyz']))
                for metric,key in [('body_rate_rad_s','omega_body_rad_s'),('wheel_momentum_nms','wheel_momentum_nms')]:
                    maxima[metric]=max(maxima[metric],math.dist(x[key],y[key]))
                maxima['dc_energy_j']=max(maxima['dc_energy_j'],abs(x['dc_energy_j']-y['dc_energy_j']))
        comparisons.append({'partition_sec':row['partition_sec'],'max_errors':maxima,
                            'passed':all(v<=BUDGETS[k] for k,v in maxima.items())})
    body={'schema':'space-env.controller-clock-program/1','scenario_sha256':digest(scenario),
          'passed':all(r['native_replay'] and r['witness']['passed'] for r in rows)
                   and all(c['passed'] for c in comparisons),
          'budgets':BUDGETS,'rows':rows,'comparisons':comparisons,'files_sha256':hashes,
          'scope':'identical native commands and off-tick interrupts; partition invariance, not physical truth or service qualification'}
    result=body|{'sha256':digest(body)}
    atomic_write(path/'result.json',canonical_json(result))
    return copy.deepcopy(result)
