"""Atomic retained plan/episode/certificate/RL/SFT bundles and bounded audit.

Checksums provide consistency, not authorship. Integrity-only audit does not
claim native physics replay; native audit independently regenerates every row.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from space_env.eval.statistics import assign_split
from space_env.factory.quality import quality_report, public_view_gate
from space_env.planning.mission_plan import validate_plan, validate_execution, _sealed
from space_env.scaling.coupled import coupled_evidence
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.simulation.streaming import atomic_jsonl, file_digest
from space_env.tasks import VerifiedEpisode

MAX_JSON = 256_000_000
MAX_BUNDLE = 4_294_967_296


def strict_json(raw):
    def pairs(items):
        value = {}
        for key,item in items:
            if key in value:raise ValueError('duplicate JSON key')
            value[key] = item
        return value
    def constant(value):raise ValueError('nonfinite JSON value: '+value)
    try:
        result = json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)
        canonical_json(result)
    except RecursionError as exc:
        raise ValueError('JSON nesting exceeds parser budget') from exc
    return result


def read_json(path, *, limit=MAX_JSON):
    path = Path(path)
    if path.is_symlink() or not path.is_file():raise ValueError('missing or symlink JSON artifact')
    with path.open('rb') as stream:raw = stream.read(limit+1)
    if len(raw)>limit:raise ValueError('JSON artifact exceeds byte budget')
    return strict_json(raw)


def _operations(native):
    life = native.get('lifecycle',{})
    layers = {'attitude':native.get('spacecraft_bus',{}).get('operations',{}),
              'lifecycle':life.get('operations',{}),
              'engine':life.get('actuation',{}).get('operations',{}),
              'unloading':life.get('unloading',{}).get('operations',{})}
    return [{'layer':layer,'id':key,'status':row['status'],'reason':row.get('reason')}
            for layer,operations in layers.items() for key,row in operations.items()
            if row.get('status') not in {None,'completed'}]


def certificate(plan, execution, quality):
    """Recompute declared goals from retained evidence; caller establishes replay."""
    spec = validate_plan(plan,check_build=False)
    validate_execution(spec,execution)
    episode = execution['episode'];native = episode['runtime']['backend_report']
    rows = episode['transitions']
    total = (len(native['lifecycle']['requests']) if 'lifecycle' in native else len(native['products']))
    fulfilled = (native['lifecycle']['metrics']['fulfilled_requests'] if 'lifecycle' in native else
                 sum(p['status']=='delivered_on_time' for p in native['products']))
    rejected = sum(not r['transition']['feedback']['accepted'] for r in rows) if episode['schema'].endswith('/1') else None
    # Compact task rows bind feedback through transition hashes. Execution keeps
    # a small feedback witness; native audit regenerates and checks it below.
    feedback = quality['feedback']
    if len(feedback)!=len(rows) or any(type(x) is not bool for x in feedback):
        raise ValueError('invalid execution feedback witness')
    if rejected is not None and rejected!=sum(not x for x in feedback):
        raise ValueError('execution feedback witness mismatch')
    rejected = sum(not x for x in feedback)
    unfinished = _operations(native)
    failed = [r for r in unfinished if r['status'] in {'failed','rejected'}]
    transitions_match = all(s.expected_transition_sha256 is None or s.expected_transition_sha256 ==
        r.get('transition_sha256',digest(r.get('transition'))) for s,r in zip(spec.steps,rows))
    goals = spec.goals
    gates = {'complete_schedule':len(rows)==len(spec.steps),
             'full_horizon':abs(native['time_sec']-spec.horizon_sec)<=1e-8,
             'execution_without_error':execution['error'] is None,
             'expected_transitions':transitions_match,
             'expected_native_state':spec.expected_native_sha256 is None or spec.expected_native_sha256==episode['runtime']['sha256'],
             'native_quality':quality['report']['passed'] is True,
             'minimum_service':fulfilled>=goals.min_fulfilled,
             'maximum_service':goals.max_fulfilled is None or fulfilled<=goals.max_fulfilled,
             'rejection_limit':rejected<=goals.max_rejections,
             'operation_failure_limit':not goals.require_no_failed_operations or not failed}
    coupled = None
    if goals.coupled is not None:
        coupled = coupled_evidence(spec.scenario,native,goals.coupled)
        gates['coupled_mission'] = coupled['passed']
    body = {'schema':'space-env.mission-plan-certificate/1','plan_sha256':spec.sha256,
        'execution_sha256':execution['sha256'],'episode_sha256':episode['sha256'],
        'native_report_sha256':digest(native),'goals':goals.model_dump(mode='json'),
        'passed':all(gates.values()),'gates':gates,'fulfilled':fulfilled,'total':total,
        'positive_service_declared':goals.min_fulfilled>0,'rejections':rejected,
        'noncompleted_operations':unfinished,'all_reported_operations_successful':not unfinished,
        'coupled_evidence':coupled,'quality':copy.deepcopy(quality),
        'expert_teacher_qualified':False,
        'scope':'native model feasibility for this fixed schedule/build and declared goals; not optimality, independent physical truth, causality of the source proposal or flight qualification'}
    return body | {'sha256':digest(body)}


def _quality(verified):
    session = verified._session  # trusted internal consumer after constructor cold replay
    return {'report':quality_report(SimpleNamespace(session=session)),
            'feedback':[r['transition']['feedback']['accepted'] for r in session._transitions]}


def _bound_rows(verified,plan_sha,cert,format):
    for row in verified.iter_rows(format=format):
        row.update(mission_plan_sha256=plan_sha,mission_certificate_sha256=cert['sha256'],
                   mission_goal_passed=cert['passed'])
        yield row


def publish_plan_bundle(plan, execution, output, *, max_bytes=1_073_741_824):
    if type(max_bytes) is not int or not 1<=max_bytes<=MAX_BUNDLE:
        raise ValueError('invalid mission bundle byte budget')
    plan,execution = copy.deepcopy(plan),copy.deepcopy(execution)
    spec = validate_plan(plan)
    validate_execution(spec,execution)
    output = Path(output)
    if output.exists() or output.is_symlink():raise ValueError('mission bundle output must not exist')
    verified = VerifiedEpisode(execution['episode'])
    cert = certificate(plan,execution,_quality(verified))
    output.parent.mkdir(parents=True,exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.'+output.name+'.',dir=output.parent))
    try:
        files,total = {},0
        for name,value in [('plan.json',plan),('execution.json',execution),('certificate.json',cert)]:
            raw = canonical_json(value)
            if len(raw.encode())>MAX_JSON:raise ValueError('mission JSON artifact exceeds byte budget')
            atomic_write(stage/name,raw)
            meta = {'bytes':(stage/name).stat().st_size,'file_sha256':file_digest(stage/name)}
            files[name]=meta;total+=meta['bytes']
            if total>max_bytes:raise ValueError('mission bundle byte budget exceeded')
        for format in (('rl','sft') if cert['passed'] else ('rl',)):
            name = format+'.jsonl'
            if total>=max_bytes:raise ValueError('mission bundle byte budget exceeded')
            meta=atomic_jsonl(stage/name,_bound_rows(verified,spec.sha256,cert,format),
                              max_rows=10000,max_bytes=max_bytes-total)
            files[name]=meta;total+=meta['bytes']
        body={'schema':'space-env.mission-plan-bundle/1','plan_sha256':spec.sha256,
            'execution_sha256':execution['sha256'],'certificate_sha256':cert['sha256'],
            'episode_sha256':execution['episode']['sha256'],'mission_goal_passed':cert['passed'],
            'lineage':spec.scenario['lineage'],'split':assign_split(spec.scenario['lineage'],''),
            'files':files,'native_export_replay':True,'sft_exported':cert['passed'],
            'teacher_quality':'verified_execution_not_expert_qualification'}
        result=body|{'sha256':digest(body)}
        if total+len(canonical_json(result).encode())>max_bytes:
            raise ValueError('mission bundle byte budget exceeded including receipt')
        atomic_write(stage/'bundle.json',canonical_json(result))
        if output.exists() or output.is_symlink():raise ValueError('mission bundle output appeared during build')
        os.rename(stage,output)
        return copy.deepcopy(result)
    finally:
        if stage.exists():shutil.rmtree(stage)


def verify_plan_bundle(directory, *, native=False, max_bytes=MAX_BUNDLE):
    if type(native) is not bool or type(max_bytes) is not int or not 1<=max_bytes<=MAX_BUNDLE:
        raise ValueError('invalid mission bundle audit options')
    root=Path(directory)
    if root.is_symlink() or not root.is_dir():raise ValueError('mission bundle requires real directory')
    receipt=read_json(root/'bundle.json',limit=2_000_000)
    if not _sealed(receipt) or receipt.get('schema')!='space-env.mission-plan-bundle/1':
        raise ValueError('mission bundle root checksum/schema mismatch')
    if set(receipt)!={'schema','plan_sha256','execution_sha256','certificate_sha256',
            'episode_sha256','mission_goal_passed','lineage','split','files',
            'native_export_replay','sft_exported','teacher_quality','sha256'}:
        raise ValueError('mission bundle root fields mismatch')
    if (type(receipt.get('mission_goal_passed')) is not bool
            or receipt.get('native_export_replay') is not True
            or receipt.get('teacher_quality')!='verified_execution_not_expert_qualification'):
        raise ValueError('mission bundle verification/quality declaration mismatch')
    expected={'plan.json','execution.json','certificate.json','rl.jsonl'}
    if receipt['mission_goal_passed'] is True:expected.add('sft.jsonl')
    actual=set()
    for p in root.iterdir():
        if p.is_symlink():raise ValueError('mission bundle symlinks forbidden')
        if not p.is_file() or len(actual)>=6:raise ValueError('mission bundle artifact inventory mismatch')
        actual.add(p.name)
    if actual!=expected|{'bundle.json'} or set(receipt['files'])!=expected:
        raise ValueError('mission bundle artifact inventory mismatch')
    if sum((root/name).stat().st_size for name in actual)>max_bytes:
        raise ValueError('mission bundle total byte budget exceeded')
    for name,meta in receipt['files'].items():
        if type(meta.get('bytes')) is not int or meta['bytes']<0:
            raise ValueError('invalid mission file byte count')
        if meta['bytes']!=(root/name).stat().st_size or meta['file_sha256']!=file_digest(root/name):
            raise ValueError('mission bundle file checksum/size mismatch: '+name)
    plan,execution,cert=[read_json(root/name) for name in ('plan.json','execution.json','certificate.json')]
    if not _sealed(cert):raise ValueError('mission certificate checksum mismatch')
    spec=validate_plan(plan,check_build=native)
    validate_execution(spec,execution)
    expected_cert=certificate(plan,execution,cert['quality'])
    if canonical_json(cert)!=canonical_json(expected_cert):raise ValueError('mission certificate recomputation mismatch')
    if (receipt['plan_sha256']!=plan['sha256'] or receipt['execution_sha256']!=execution['sha256']
            or receipt['certificate_sha256']!=cert['sha256'] or receipt['episode_sha256']!=execution['episode']['sha256']
            or receipt['mission_goal_passed'] is not cert['passed']
            or receipt['sft_exported'] is not cert['passed']
            or receipt['lineage']!=spec.scenario['lineage']
            or receipt['split']!=assign_split(spec.scenario['lineage'],'')):
        raise ValueError('mission bundle cross-artifact binding mismatch')
    verified=None
    if native:
        verified=VerifiedEpisode(execution['episode'])
        if canonical_json(certificate(plan,execution,_quality(verified)))!=canonical_json(cert):
            raise ValueError('mission certificate native replay mismatch')
    for format in ('rl','sft'):
        name=format+'.jsonl'
        if name not in expected:continue
        meta=receipt['files'][name];count=0;last=-1;checksum=hashlib.sha256(b'[')
        iterator=iter(_bound_rows(verified,spec.sha256,cert,format)) if native else None
        with (root/name).open('rb') as stream:
            while raw:=stream.readline(33_554_433):
                if len(raw)>33_554_432:raise ValueError('mission dataset row byte budget exceeded')
                row=strict_json(raw);i=row['index']
                if (type(i) is not int or i<=last or not 0<=i<len(execution['episode']['transitions'])
                        or row.get('schema')!='space-env.task-dataset/1' or row.get('view_role')!='@central'
                        or row.get('replay_verified') is not True
                        or row.get('policy_origin')!=spec.task['policy_origin']
                        or row['mission_plan_sha256']!=plan['sha256']
                        or row['mission_certificate_sha256']!=cert['sha256']
                        or row['mission_goal_passed'] is not cert['passed']
                        or row['episode_sha256']!=execution['episode']['sha256']
                        or row['lineage']!=receipt['lineage'] or row['split']!=receipt['split']
                        or row['quality']!='verified_execution_not_expert_qualification'):
                    raise ValueError('mission dataset row binding/order mismatch')
                request=execution['episode']['transitions'][i]['request']
                if format=='rl':
                    if row['action']!={'choices':request['choices'],'advance_sec':request['advance_sec']}:
                        raise ValueError('mission RL action/plan mismatch')
                    if row['feedback']['accepted'] is not cert['quality']['feedback'][i]:
                        raise ValueError('mission RL feedback/certificate mismatch')
                    public_view_gate(row['observation']);public_view_gate(row['next_observation'])
                else:
                    if cert['quality']['feedback'][i] is not True:
                        raise ValueError('rejected decision cannot be an SFT row')
                    action=strict_json(row['messages'][1]['tool_calls'][0]['function']['arguments'])
                    if action!=request:raise ValueError('mission SFT action/plan mismatch')
                    public_view_gate(strict_json(row['messages'][0]['content']))
                    tool=strict_json(row['messages'][2]['content'])
                    if tool['feedback']['accepted'] is not cert['quality']['feedback'][i]:
                        raise ValueError('mission SFT feedback/certificate mismatch')
                    public_view_gate(tool['observation'])
                if iterator is not None:
                    expected_row=next(iterator,None)
                    if expected_row is None or canonical_json(expected_row)!=canonical_json(row):
                        raise ValueError('mission dataset differs from independently replayed native row')
                if count:checksum.update(b',')
                checksum.update(canonical_json(row).encode());count+=1;last=i
                if count>10000:raise ValueError('mission dataset row count budget exceeded')
        checksum.update(b']')
        expected_rows=len(execution['episode']['transitions']) if format=='rl' else sum(cert['quality']['feedback'])
        if (type(meta['rows']) is not int or count!=expected_rows or count!=meta['rows'] or checksum.hexdigest()!=meta['rows_sha256']
                or iterator is not None and next(iterator,None) is not None):
            raise ValueError('mission dataset row inventory/digest mismatch')
    return {'schema':'space-env.mission-plan-bundle-audit/1','integrity_passed':True,
        'mission_goal_passed':cert['passed'],'native_execution_replayed':native,
        'bundle_sha256':receipt['sha256'],'plan_sha256':plan['sha256'],
        'expert_teacher_qualified':False,'dataset_rows':{k:v['rows'] for k,v in receipt['files'].items() if k.endswith('.jsonl')}}
