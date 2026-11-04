"""Fixed-command RK/partition checks for fixed-clock RCS and rotating thrust.

Declared synthetic hardware, constant inertia and ideal balanced nozzle pairs.
No flight calibration, total chemical-energy or imaging/SLA claim is made.
"""
from __future__ import annotations

import copy
import hashlib
import math
import json
from pathlib import Path

from space_env.runtime import AccessPolicy, MissionRuntime, spacecraft_scenario
from space_env.scaling.coupled import CoupledMissionSpec, coupled_workload
from space_env.scaling.numerics import _qdistance
from space_env.simulation.snapshot import atomic_write, canonical_json, digest

BUDGETS = {'attitude_deg':1e-5, 'body_rate_rad_s':1e-7, 'wheel_momentum_nms':1e-7,
           'position_m':.01, 'velocity_m_s':1e-5, 'dc_energy_j':1e-5,
           'battery_wh':1e-6, 'temperature_k':1e-4, 'fuel_kg':1e-8}
MESHES=(.02,.01,.005)
PARTITIONS=(.1,.025,.01)
TIMES=(.171,.337,.613,1.137,2.137,2.537,3.)


def actuator_scenario(value, *, mode='nominal', mesh=.01):
    cfg=CoupledMissionSpec.model_validate(value)
    if cfg.controller_clock != 'fixed-tick/1' or mode not in {'nominal','limited'} or mesh not in MESHES:
        raise ValueError('actuator qualification requires declared fixed-tick mode and frozen mesh')
    d=coupled_workload(cfg.model_dump(mode='json') | {'id':'actuator-'+mode,
        'horizon_sec':30.,'control_step_sec':.1,'rigid_step_sec':mesh,
        'min_fulfilled':0,'min_acquired':0})['scenario']
    native=d['definition'];actors=sorted(native['orbits'])
    authority=.1 if mode=='nominal' else .02
    native['max_maneuvers']=1024
    engines,rcs={},{}
    for a in actors:
        native['orbits'][a]['propulsion'].update(propellant_kg=.1,max_thrust_n=.02)
        rigid=native['spacecraft_bus'][a]['attitude']['rigid']
        rigid['initial_omega_body_rad_s']=[0.,.02,0.] if a==actors[0] else [0.,0.,.001]
        rigid['initial_wheel_momentum_nms']=[0.,0.,0.] if a==actors[0] else [.08,0.,0.]
        engines[a]={'kind':'chemical','model_id':'synthetic-offset-engine/1','source_status':'synthetic',
            'source':'declared verification hardware; not calibrated flight engine',
            'max_thrust_n':.02,'max_gimbal_deg':0.,'gimbal_settling_sec':0.,
            'ignition_sec':.137,'ignition_power_w':3.,'firing_aux_power_w':2.,
            'chemical_wall_heat_w_at_max':.1,
            'rigid_coupling':{'lever_arm_body_m':[.2,0.,0.],'max_mass_loss_fraction':.001}}
        rcs[a]={'parameter_status':'synthetic','propellant_pool':'shared-lumped',
            'model_id':'synthetic-balanced-pairs/1','source':'declared ideal balanced RCS pairs',
            'max_total_nozzle_thrust_n':1.,'max_torque_nm':[authority]*3,
            'lever_arm_m':[.5]*3,'isp_s':60.,'startup_sec':.137,
            'startup_power_w':3.,'firing_power_w':4.}
    native['actuation']={'engines':engines,'max_segments':1024}
    native['unloading']={'actors':rcs,'max_segments':1024}
    return spacecraft_scenario(native,id="fixed-actuators-"+mode,lineage="D2e-frozen-actuators")


def _snapshot(m):
    backend=m._backend
    return {'time_sec':m.time_sec,'actors':{a:{
        'state_si':backend.mission.ephemerides[a].state(m.time_sec).tolist(),
        'mass_kg':float(backend.mission.ephemerides[a].mass(m.time_sec)),
        'battery_wh':float(backend.mission.energy_wh[a]),
        'bus':m.observe(a)['actors'][a]['spacecraft_bus'],
    } for a in backend.actors}}


def _execute(scenario,partition,path):
    m=MissionRuntime(scenario,access=AccessPolicy(truth=True),audit_mode='compact')
    actors=sorted(scenario['definition']['orbits']);a,b=actors[:2];serial=0
    def call(op,**kwargs):
        nonlocal serial
        serial+=1
        return getattr(m,op)(expected_revision=m.revision,idempotency_key=f'act/{serial}',**kwargs)
    def burn(actor,id,firing):
        call('submit',commands=[{'command_id':id,'actor_id':actor,'kind':'body_burn',
            'duration_sec':1.,'body_burn':{'frame':'BODY','direction_body':[0.,0.,1.],
                                        'thrust_n':.01,'firing_sec':firing}}])
    burn(a,'engine-a',.4)
    call('submit',commands=[{'command_id':'unload','actor_id':b,'kind':'momentum_unload',
        'duration_sec':2.,'momentum_unload':{'frame':'BODY','target_momentum_nms':[0.,0.,0.],
                                           'tolerance_nms':1e-5}}])
    snapshots=[]
    for t in TIMES:
        while t-m.time_sec > 1e-9:
            call('advance',dt_sec=min(partition,t-m.time_sec))
        snapshots.append(_snapshot(m))
        if t==.171:
            m.save(path/'prefix.json',compact=True)
            m=MissionRuntime.load(path/'prefix.json',access=AccessPolicy(truth=True))
        elif t==.613:
            call('submit',commands=[{'command_id':'recover-attitude','actor_id':a,'kind':'slew',
                'duration_sec':1.5,'slew':{'frame':'GCRS','quaternion_wxyz':[1.,0.,0.,0.]}}])
        elif t==2.137:
            burn(b,'engine-b',.2)
    m.save(path/'final.json',compact=True)
    replay=MissionRuntime.load(path/'final.json',access=AccessPolicy(truth=True)).checkpoint(compact=True)==m.checkpoint(compact=True)
    return snapshots,m._backend.report(),replay


def _witness(scenario,report,mode):
    actors=sorted(scenario['definition']['orbits']);a,b=actors[:2]
    life=report['lifecycle'];engine,unload=life['actuation'],life['unloading'];op=unload['operations']['unload']
    close=lambda x,tol: isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and abs(x)<=tol
    rigid=[s['rigid_attitude'] for s in report['spacecraft_bus']['states'].values()]
    gates={
        'engine_completion':all(o['status']=='completed' for o in engine['operations'].values()) and len(engine['operations'])==2,
        'rotating_force':bool(engine['intervals']) and all(i.get('direction_model')=='space-env.rotating-body-thrust/1' for i in engine['intervals']),
        'expected_unload':op['status']==('completed' if mode=='nominal' else 'failed'),
        'paid_fuel':all(engine['states'][k]['consumed_propellant_kg']>0 for k in [a,b]) and unload['fuel_used_kg'][b]>0,
        'shared_tank_balance':all(close(v,1e-9) for v in [*engine['fuel_balance_error_kg'].values(),*unload['fuel_balance_error_kg'].values()]),
        'electrical_heat':all(close(v,1e-6) for v in [*engine['electrical_heat_balance_error_j'].values(),*unload['energy_balance_error_j'].values()]),
        'rigid_balances':all(close(s['kinetic_work_balance_error_j'],1e-6) and close(s['dc_heat_work_balance_error_j'],1e-6)
                             and all(close(v,1e-7) for v in s['angular_momentum_balance_error_gcrs_nms']) for s in rigid),
        'battery_inventory':all(close(v['balance_error_wh'],1e-7) for v in report['energy_ledger'].values()) and all(close(v['balance_error_bits'],1e-3) for v in life['ledger'].values()),
        'balanced_pairs':bool(unload['intervals']) and all(i['net_force_gcrs_n']==[0.,0.,0.] for i in unload['intervals']),
        'analytic_engine_impulse':close(report['spacecraft_bus']['states'][a]['rigid_attitude']['external_angular_impulse_gcrs_nms'][1]+.0008,1e-9),
    }
    if mode=='limited':
        gates['retained_limited_failure']=op['reason']=='unload_window_ended' and max(abs(x) for x in op['remaining_momentum_nms'])>.04
    else:
        gates['target_momentum']=max(abs(x) for x in op['remaining_momentum_nms'])<=1e-5
    return {'passed':all(gates.values()),'gates':gates,'unloading':copy.deepcopy(op),
            'joint_clocks':unload.get('control_clocks'),
            'scope':'declared component ledgers and one analytic axial impulse; no total chemical-energy/flight qualification'}


def _compare(left,right):
    if len(left)!=len(TIMES) or len(right)!=len(TIMES):
        raise ValueError("incomplete actuator comparison grid")
    maxima=dict.fromkeys(BUDGETS,0.)
    # Reject non-finite values before max(): max(0, NaN) can hide corrupt data.
    for samples in (left,right):
        canonical_json(samples)
    for expected,x,y in zip(TIMES,left,right):
        if (abs(x['time_sec']-expected)>1e-9 or abs(y['time_sec']-expected)>1e-9
                or set(x['actors'])!=set(y['actors']) or len(x['actors'])<2):
            raise ValueError('actuator comparison grid/identity mismatch')
        for a,u in x['actors'].items():
            v=y['actors'][a];p,q=u['bus']['rigid_attitude'],v['bus']['rigid_attitude']
            values={'attitude_deg':_qdistance(p['quaternion_wxyz'],q['quaternion_wxyz']),
                'body_rate_rad_s':math.dist(p['omega_body_rad_s'],q['omega_body_rad_s']),
                'wheel_momentum_nms':math.dist(p['wheel_momentum_nms'],q['wheel_momentum_nms']),
                'position_m':math.dist(u['state_si'][:3],v['state_si'][:3]),
                'velocity_m_s':math.dist(u['state_si'][3:],v['state_si'][3:]),
                'dc_energy_j':abs(p['dc_energy_j']-q['dc_energy_j']),
                'battery_wh':abs(u['battery_wh']-v['battery_wh']),
                'temperature_k':abs(u['bus']['temperature_k']-v['bus']['temperature_k']),
                'fuel_kg':abs(u['mass_kg']-v['mass_kg'])}
            for k,value in values.items(): maxima[k]=max(maxima[k],value)
    return {'passed':all(v<=BUDGETS[k] for k,v in maxima.items()),'max_errors':maxima}


def qualify_coupled_actuators(configuration,output):
    cfg=CoupledMissionSpec.model_validate(configuration)
    if cfg.controller_clock != 'fixed-tick/1':
        raise ValueError('actuator qualification requires fixed-tick/1')
    path=Path(output)
    if path.exists() and any(path.iterdir()):raise ValueError('actuator output must be empty')
    path.mkdir(parents=True,exist_ok=True)
    atomic_write(path/'configuration.json',canonical_json(cfg.model_dump(mode='json')))
    rows=[];files={}
    for mode in ['nominal','limited']:
        reference=None
        for mesh in MESHES:
            for partition in PARTITIONS:
                directory=path/f'{mode}-{mesh}-{partition}';directory.mkdir()
                scenario=actuator_scenario(cfg,mode=mode,mesh=mesh)
                atomic_write(directory/'scenario.json',canonical_json(scenario))
                try:
                    samples,report,replay=_execute(scenario,partition,directory)
                    witness=_witness(scenario,report,mode)
                    if mesh==MESHES[0] and partition==PARTITIONS[0]:reference=samples
                    comparison=(_compare(reference,samples) if reference is not None else
                                {'passed':False,'reason':'designated reference execution missing'})
                    row={'mode':mode,'mesh_sec':mesh,'partition_sec':partition,'samples':samples,
                         'witness':witness,'comparison':comparison,'native_replay':replay,
                         'passed':replay and witness['passed'] and comparison['passed']}
                    atomic_write(directory/'report.json',canonical_json(report))
                except (ValueError,RuntimeError,KeyError) as exc:
                    row={'mode':mode,'mesh_sec':mesh,'partition_sec':partition,'passed':False,'error':str(exc)}
                rows.append(row)
                atomic_write(path/'progress.json',canonical_json({'expected':18,'finished':len(rows),'rows':rows}))
    for p in path.rglob('*.json'):files[str(p.relative_to(path))]=hashlib.sha256(p.read_bytes()).hexdigest()
    body={'schema':'space-env.coupled-actuator-qualification/1','passed':len(rows)==18 and all(r['passed'] for r in rows),
          'rows':rows,'budgets':dict(BUDGETS),'files_sha256':files,
          'fixed_controller_period_sec':.1,'program_times_sec':list(TIMES),
          'meshes_sec':list(MESHES),'partitions_sec':list(PARTITIONS),
          'scope':'fixed native commands, RK/caller partition bounds and retained negative control; not convergence order, flight hardware or service qualification'}
    result=body|{'sha256':digest(body)};atomic_write(path/'result.json',canonical_json(result))
    return copy.deepcopy(result)


def verify_coupled_actuators(directory):
    """Recompute evidence without native execution; checksums are not signatures.

    A valid bundle may contain failed executions. ``valid`` concerns its integrity;
    ``passed`` additionally requires every frozen member to pass. Native replay is
    retained execution evidence, not independently rerun by this offline audit.
    """
    root=Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('actuator bundle must be a real directory')
    root=root.resolve()
    total=0
    def read(name):
        nonlocal total
        p=root/name
        if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root):
            raise ValueError('unsafe or missing actuator artifact: '+name)
        with p.open('rb') as stream:
            raw=stream.read(32_000_001)
        total+=len(raw)
        if len(raw)>32_000_000 or total>256_000_000:
            raise ValueError('actuator bundle exceeds audit byte budget')
        def pairs(items):
            result={}
            for key,value in items:
                if key in result:raise ValueError('duplicate JSON key')
                result[key]=value
            return result
        value=json.loads(raw,object_pairs_hook=pairs)
        canonical_json(value)  # rejects NaN/Infinity, including unvisited fields
        return value,hashlib.sha256(raw).hexdigest()
    result,_=read('result.json')
    body={k:v for k,v in result.items() if k!='sha256'}
    if result.get('sha256')!=digest(body):raise ValueError('actuator result digest mismatch')
    if (result.get('schema')!='space-env.coupled-actuator-qualification/1'
            or result.get('budgets')!=BUDGETS
            or result.get('fixed_controller_period_sec')!=.1
            or result.get('program_times_sec')!=list(TIMES)
            or result.get('meshes_sec')!=list(MESHES)
            or result.get('partitions_sec')!=list(PARTITIONS)):
        raise ValueError('actuator recipe or budgets mismatch')
    expected=[(mode,mesh,part) for mode in ('nominal','limited') for mesh in MESHES for part in PARTITIONS]
    rows=result.get('rows',[])
    if len(rows)!=len(expected) or [(r.get('mode'),r.get('mesh_sec'),r.get('partition_sec')) for r in rows]!=expected:
        raise ValueError('actuator member grid mismatch')
    paths={'configuration.json','progress.json'}
    for mode,mesh,part in expected:
        prefix=f'{mode}-{mesh}-{part}'
        paths.add(prefix+'/scenario.json')
        for name in ('prefix.json','final.json','report.json'):
            if (root/prefix/name).exists():paths.add(prefix+'/'+name)
    actual=set()
    for p in root.rglob('*'):
        if p.is_symlink():raise ValueError('symlinks are forbidden in actuator bundles')
        if p.is_file():actual.add(p.relative_to(root).as_posix())
    hashes=result.get('files_sha256',{})
    if set(hashes)!=paths or actual!=paths|{'result.json'}:
        raise ValueError('actuator artifact inventory mismatch')
    artifacts={}
    for name in sorted(paths):
        value,sha=read(name)
        if hashes[name]!=sha:raise ValueError('actuator artifact digest mismatch: '+name)
        artifacts[name]=value
        if name.endswith(('/prefix.json','/final.json')):
            if value.get('sha256')!=digest({k:v for k,v in value.items() if k!='sha256'}):
                raise ValueError('native checkpoint digest mismatch: '+name)
    progress=artifacts['progress.json']
    if progress!={'expected':18,'finished':18,'rows':rows}:
        raise ValueError('actuator progress/result mismatch')
    cfg=CoupledMissionSpec.model_validate(artifacts['configuration.json'])
    if cfg.controller_clock!='fixed-tick/1':raise ValueError('actuator clock mismatch')
    reference={};passed=[]
    for row,(mode,mesh,part) in zip(rows,expected):
        prefix=f'{mode}-{mesh}-{part}'
        scenario=artifacts[prefix+'/scenario.json']
        if scenario!=actuator_scenario(cfg,mode=mode,mesh=mesh):
            raise ValueError('frozen actuator scenario mismatch: '+prefix)
        if 'error' in row:
            if row.get('passed') is not False:raise ValueError('failed member marked passed')
            passed.append(False)
            continue
        if any(prefix+'/'+name not in artifacts for name in ('prefix.json','final.json','report.json')):
            raise ValueError('missing completed actuator evidence: '+prefix)
        witness=_witness(scenario,artifacts[prefix+'/report.json'],mode)
        _compare(row['samples'],row['samples'])  # validate grid even if reference failed
        if mesh==MESHES[0] and part==PARTITIONS[0]:reference[mode]=row['samples']
        comparison=(_compare(reference[mode],row['samples']) if mode in reference else
                    {'passed':False,'reason':'designated reference execution missing'})
        ok=row.get('native_replay') is True and witness['passed'] and comparison['passed']
        if row.get('witness')!=witness or row.get('comparison')!=comparison or row.get('passed') is not ok:
            raise ValueError('actuator recomputation mismatch: '+prefix)
        passed.append(ok)
    if result.get('passed') is not all(passed):raise ValueError('actuator aggregate mismatch')
    return {'schema':'space-env.coupled-actuator-audit/1','valid':True,'passed':all(passed),
            'attempts':len(rows),'failed':sum(not p for p in passed),'result_sha256':result['sha256'],
            'native_execution_rerun':False}
