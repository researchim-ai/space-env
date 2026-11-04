"""Fixed schedules, physical replay, atomic retained datasets and tamper gates."""
import copy
import hashlib
import json
import subprocess
import sys

import pytest

from space_env.planning import PlanningRun
from space_env.planning.mission_plan import (
    PlanExecution, make_mission_plan, recorded_mission_plan, validate_plan,
)
from space_env.planning.mission_bundle import (
    publish_plan_bundle, verify_plan_bundle, strict_json, read_json,
)
from space_env.simulation.snapshot import canonical_json,digest
from space_env.simulation.streaming import array_digest
from tests.test_planning import scenario,task


@pytest.fixture(scope='module')
def proposal():
    run=PlanningRun(scenario.__wrapped__(),task());run.run()
    return recorded_mission_plan(run.checkpoint(compact=True))


def seal(value):
    value['sha256']=digest({k:v for k,v in value.items() if k!='sha256'})
    return value


def execution(plan):
    run=PlanExecution(plan);run.run();return run.checkpoint()


def test_fixed_plan_reproduces_source_native_state_and_owned_results(proposal):
    source=copy.deepcopy(proposal);run=PlanExecution(source)
    source['steps'].clear()
    assert run.run()['complete']
    record=run.checkpoint()
    assert record['episode']['runtime']['sha256']==proposal['expected_native_sha256']
    before=copy.deepcopy(record)
    record['build']['source_data_sha256']='f'*64
    run.plan['steps'].clear()
    assert run.checkpoint()==before


def test_declared_plan_without_recorded_teacher_claim(proposal):
    p=make_mission_plan(proposal['scenario'],proposal['task'],
        [{'at_sec':float(i),'advance_sec':1.,'action_ids':['wait']} for i in range(8)])
    assert p['origin']=='declared' and p['source_planning_sha256'] is None
    assert PlanExecution(p).run()['complete']


@pytest.mark.parametrize('mutate',[
    lambda p:p['steps'][0].__setitem__('at_sec',1.),
    lambda p:p['steps'][0].__setitem__('action_ids',['unknown']),
    lambda p:p['steps'][0].__setitem__('action_ids',['wait','wait']),
    lambda p:p['steps'].pop(),
    lambda p:p['build'].__setitem__('source_data_sha256','f'*64),
    lambda p:p.__setitem__('scenario_sha256','f'*64),
    lambda p:p.__setitem__('task_sha256','f'*64),
    lambda p:p.__setitem__('horizon_sec',7.),
])
def test_resealed_invalid_plan_rejected(proposal,mutate):
    p=copy.deepcopy(proposal);mutate(p);seal(p)
    with pytest.raises(ValueError):validate_plan(p)


def test_source_compact_and_full_records_compile_same_actions():
    run=PlanningRun(scenario.__wrapped__(),task());run.run()
    a=recorded_mission_plan(run.checkpoint(compact=True))
    b=recorded_mission_plan(run.checkpoint(compact=False))
    assert a['steps']==b['steps']
    with pytest.raises(ValueError):recorded_mission_plan(seal({'schema':'wrong'}))


def test_prefix_resume_and_wrong_plan_binding(proposal):
    run=PlanExecution(proposal);run.run(decisions=3);prefix=run.checkpoint()
    restored=PlanExecution.from_checkpoint(proposal,prefix)
    assert not restored._recovery['fresh_process']
    restored.run();run.run()
    assert restored.checkpoint()['episode']==run.checkpoint()['episode']
    bad=copy.deepcopy(prefix);bad['plan_sha256']='0'*64;seal(bad)
    with pytest.raises(ValueError):PlanExecution.from_checkpoint(proposal,bad)


def test_recovery_metadata_must_match_reported_processes(proposal):
    first=PlanExecution(proposal);first.run(decisions=2)
    second=PlanExecution.from_checkpoint(proposal,first.checkpoint());second.run()
    record=second.checkpoint();record['recovery']['fresh_process']=True;seal(record)
    with pytest.raises(ValueError,match='recovery'):PlanExecution.from_checkpoint(proposal,record)


def test_expected_hash_divergence_retains_executed_failure(proposal,tmp_path):
    p=copy.deepcopy(proposal);p['steps'][0]['expected_transition_sha256']='0'*64;seal(p)
    run=PlanExecution(p);status=run.run()
    assert not status['complete'] and status['steps']==1
    assert status['error']['code']=='expected_transition_mismatch'
    result=publish_plan_bundle(p,run.checkpoint(),tmp_path/'failed')
    assert not result['mission_goal_passed'] and not result['sft_exported']
    assert 'sft.jsonl' not in result['files']
    assert verify_plan_bundle(tmp_path/'failed',native=True)['integrity_passed']


def test_successful_bundle_and_native_audit(proposal,tmp_path):
    p=tmp_path/'bundle';result=publish_plan_bundle(proposal,execution(proposal),p)
    assert result['mission_goal_passed'] and result['sft_exported']
    assert result['files']['rl.jsonl']['rows']==result['files']['sft.jsonl']['rows']==8
    assert not verify_plan_bundle(p)['native_execution_replayed']
    native=verify_plan_bundle(p,native=True)
    assert native['native_execution_replayed'] and not native['expert_teacher_qualified']
    row=json.loads((p/'sft.jsonl').read_text().splitlines()[0])
    assert row['mission_plan_sha256']==proposal['sha256']
    assert 'scenario' not in json.loads(row['messages'][0]['content'])


def test_unsatisfied_service_keeps_rl_and_suppresses_sft(proposal,tmp_path):
    p=copy.deepcopy(proposal);p['goals']['min_fulfilled']=100;seal(p)
    result=publish_plan_bundle(p,execution(p),tmp_path/'failed')
    assert not result['mission_goal_passed']
    assert result['files']['rl.jsonl']['rows']==8 and 'sft.jsonl' not in result['files']
    assert not verify_plan_bundle(tmp_path/'failed',native=True)['mission_goal_passed']


def test_partial_prefix_is_never_a_full_horizon_teacher(proposal,tmp_path):
    run=PlanExecution(proposal);run.run(decisions=0)
    result=publish_plan_bundle(proposal,run.checkpoint(),tmp_path/'prefix')
    assert not result['mission_goal_passed'] and not result['sft_exported']
    assert result['files']['rl.jsonl']['rows']==0
    assert verify_plan_bundle(tmp_path/'prefix',native=True)['integrity_passed']


def reseal_files(root):
    result=json.loads((root/'bundle.json').read_text())
    for name,meta in result['files'].items():
        raw=(root/name).read_bytes();meta.update(bytes=len(raw),file_sha256=hashlib.sha256(raw).hexdigest())
        if name.endswith('.jsonl'):
            rows=[json.loads(line) for line in raw.splitlines()]
            meta.update(rows=len(rows),rows_sha256=array_digest(rows))
    (root/'bundle.json').write_text(canonical_json(seal(result)))


def test_native_audit_rejects_resealed_forged_reward(proposal,tmp_path):
    root=tmp_path/'bundle';publish_plan_bundle(proposal,execution(proposal),root)
    path=root/'rl.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines()]
    rows[0]['reward']+=1.
    path.write_text(''.join(canonical_json(r)+'\n' for r in rows));reseal_files(root)
    assert verify_plan_bundle(root)['integrity_passed']  # checksums are not physics
    with pytest.raises(ValueError,match='independently replayed'):verify_plan_bundle(root,native=True)


def test_native_audit_compares_json_types_not_python_numeric_equality(proposal,tmp_path):
    root=tmp_path/'bundle';publish_plan_bundle(proposal,execution(proposal),root)
    p=root/'rl.jsonl';rows=[json.loads(line) for line in p.read_text().splitlines()]
    # Python considers 0.0 == False. Canonical native JSON must distinguish them.
    rows[0]['elapsed_sec']=int(rows[0]['elapsed_sec'])
    p.write_text(''.join(canonical_json(r)+'\n' for r in rows));reseal_files(root)
    with pytest.raises(ValueError,match='independently replayed'):verify_plan_bundle(root,native=True)


def test_unsealed_certificate_cannot_use_python_bool_int_equivalence(proposal,tmp_path):
    root=tmp_path/'bundle';publish_plan_bundle(proposal,execution(proposal),root)
    p=root/'certificate.json';cert=read_json(p)
    cert['expert_teacher_qualified']=0  # leave old SHA to exercise seal check
    p.write_text(canonical_json(cert));reseal_files(root)
    with pytest.raises(ValueError,match='certificate checksum'):verify_plan_bundle(root)


def test_resealed_action_substitution_and_privileged_view_rejected(proposal,tmp_path):
    root=tmp_path/'bundle';publish_plan_bundle(proposal,execution(proposal),root)
    path=root/'rl.jsonl';original=path.read_text();rows=[json.loads(s) for s in original.splitlines()]
    rows[0]['action']['choices']={'@central':['wrong']}
    path.write_text(''.join(canonical_json(r)+'\n' for r in rows));reseal_files(root)
    with pytest.raises(ValueError,match='action/plan'):verify_plan_bundle(root)
    rows=[json.loads(s) for s in original.splitlines()];rows[0]['observation']['seed']=123
    path.write_text(''.join(canonical_json(r)+'\n' for r in rows));reseal_files(root)
    with pytest.raises(ValueError,match='privileged'):verify_plan_bundle(root)


def test_resealed_reward_does_not_hide_row_deletion(proposal,tmp_path):
    root=tmp_path/'bundle';publish_plan_bundle(proposal,execution(proposal),root)
    p=root/'rl.jsonl';rows=p.read_text().splitlines();p.write_text('\n'.join(rows[1:])+'\n');reseal_files(root)
    with pytest.raises(ValueError,match='inventory'):verify_plan_bundle(root)


def test_resealed_expert_claim_and_nested_checkpoint_corruption_rejected(proposal,tmp_path):
    root=tmp_path/'bundle';record=execution(proposal);publish_plan_bundle(proposal,record,root)
    p=root/'certificate.json';value=read_json(p);value['expert_teacher_qualified']=True
    p.write_text(canonical_json(seal(value)));reseal_files(root)
    with pytest.raises(ValueError,match='certificate'):verify_plan_bundle(root)
    record['episode']['runtime']['backend_report']['time_sec']=999.
    seal(record['episode']);seal(record)
    with pytest.raises(ValueError,match='envelope'):PlanExecution.from_checkpoint(proposal,record)


@pytest.mark.parametrize('change',['extra','truncate','corrupt','symlink','directory'])
def test_bundle_artifact_inventory_and_corruption(proposal,tmp_path,change):
    root=tmp_path/'bundle';publish_plan_bundle(proposal,execution(proposal),root)
    if change=='extra':(root/'extra.json').write_text('{}')
    elif change=='truncate':(root/'rl.jsonl').write_text('')
    elif change=='corrupt':(root/'certificate.json').write_text('{}')
    elif change=='directory':(root/'nested').mkdir()
    else:(root/'link').symlink_to(tmp_path/'outside')
    with pytest.raises(ValueError):verify_plan_bundle(root)


def test_duplicate_nonfinite_and_read_budget(tmp_path):
    for raw in ('{"x":1,"x":2}','{"x":NaN}','{"x":Infinity}'):
        with pytest.raises(ValueError):strict_json(raw)
    p=tmp_path/'big';p.write_text('12345')
    with pytest.raises(ValueError):read_json(p,limit=4)


def test_cancelled_operation_is_not_reported_as_successful():
    from space_env.planning.mission_bundle import _operations
    result=_operations({'spacecraft_bus':{'operations':{'slew':{'status':'cancelled'}}}})
    assert len(result)==1 and result[0]['status']=='cancelled'


def test_atomic_budget_failure_and_existing_directory(proposal,tmp_path):
    record=execution(proposal)
    with pytest.raises(ValueError,match='budget'):publish_plan_bundle(proposal,record,tmp_path/'budget',max_bytes=1)
    assert not list(tmp_path.iterdir())
    root=tmp_path/'mine';root.mkdir();(root/'keep').write_text('keep')
    with pytest.raises(ValueError):publish_plan_bundle(proposal,record,root)
    assert (root/'keep').read_text()=='keep'


def test_late_export_failure_does_not_publish(proposal,tmp_path,monkeypatch):
    import space_env.planning.mission_bundle as module
    def fail(*args,**kwargs):raise OSError('disk failure')
    monkeypatch.setattr(module,'atomic_jsonl',fail)
    with pytest.raises(OSError):publish_plan_bundle(proposal,execution(proposal),tmp_path/'out')
    assert not list(tmp_path.iterdir())


def test_cli_roundtrip_fresh_process_and_bundle(proposal,tmp_path):
    plan=tmp_path/'plan.json';plan.write_text(canonical_json(proposal))
    prefix=tmp_path/'prefix.json';final=tmp_path/'final.json';bundle=tmp_path/'bundle'
    def cli(*args):
        return subprocess.run([sys.executable,'-m','space_env.cli',*map(str,args)],capture_output=True,text=True,timeout=60)
    producer=cli('mission-plan-run',plan,prefix,'--decisions','3');assert producer.returncode==0,producer.stderr
    consumer=cli('mission-plan-resume',plan,prefix,final);assert consumer.returncode==0,consumer.stderr
    record=read_json(final);recovery=record['recovery'];assert recovery['fresh_process']
    assert recovery['producer_pid']!=recovery['consumer_pid'] and recovery['prefix_steps']==3
    result=cli('mission-plan-export',plan,final,bundle);assert result.returncode==0,result.stderr
    audit=cli('verify-mission-plan-bundle',bundle,'--native');assert audit.returncode==0,audit.stderr
    assert json.loads(audit.stdout)['native_execution_replayed']
    duplicate=cli('mission-plan-run',plan,final)
    assert duplicate.returncode==2
