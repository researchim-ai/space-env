"""Joint RCS clock, paid rotating thrust, replay and retained limitations."""
import copy
import json
import pytest

from examples.coupled.study import configuration
from examples.coupled.extended import manifest
from space_env.runtime import AccessPolicy,MissionRuntime,RuntimeContractError
from space_env.scaling.benchmark import ScaleSpec
from space_env.validation.coupled_actuators import actuator_scenario,_execute,_witness,_compare


def runtime():
    return MissionRuntime(actuator_scenario(configuration(True)['development']),access=AccessPolicy(truth=True))


def unload(m):
    m.submit([{'command_id':'unload','actor_id':'b-00','kind':'momentum_unload','duration_sec':2.,
        'momentum_unload':{'frame':'BODY','target_momentum_nms':[0.,0.,0.],'tolerance_nms':1e-5}}],
        expected_revision=m.revision,idempotency_key='unload')


def advance(m,dt,key):
    return m.advance(dt,expected_revision=m.revision,idempotency_key=key)


def test_joint_hold_and_off_tick_startup():
    m=runtime();unload(m);advance(m,.171,'prefix')
    u=m._backend.lifecycle.unloader
    assert u.control_samples['b-00']==1
    assert u.control_latches['b-00']['sample_sec']==pytest.approx(.137)
    motor=u.control_latches['b-00']['motor'].copy()
    advance(m,.01,'extra-event')
    assert u.control_samples['b-00']==1
    assert (u.control_latches['b-00']['motor']==motor).all()
    advance(m,.03,'tick')
    assert u.control_samples['b-00']==2
    assert u.control_latches['b-00']['sample_sec']==pytest.approx(.2)


def test_joint_latch_replay_and_fork_ownership():
    m=runtime();unload(m);advance(m,.171,'prefix')
    restored=MissionRuntime.from_checkpoint(m.checkpoint(),access=AccessPolicy(truth=True))
    child=m.fork();child._backend.lifecycle.unloader.control_latches['b-00']['motor'][0]=99.
    assert m._backend.lifecycle.unloader.control_latches['b-00']['motor'][0]!=99.
    for model in [m,restored]:advance(model,.07,'resume')
    assert m.checkpoint()==restored.checkpoint()


def test_cancel_clears_joint_latch():
    m=runtime();unload(m);advance(m,.171,'prefix')
    m.cancel('unload',expected_revision=m.revision,idempotency_key='cancel')
    assert 'b-00' not in m._backend.lifecycle.unloader.control_latches
    energy=m._backend.bus.rigid.states['b-00'].dc_energy_j
    advance(m,.01,'coast')
    assert m._backend.bus.rigid.states['b-00'].dc_energy_j==energy


def test_fault_interrupt_clears_joint_latch():
    m=runtime();unload(m);advance(m,.171,'prefix')
    m.inject([{'id':'safe','type':'spacecraft_safe_mode','time_sec':m.time_sec,
               'source':'test','reason':'protection','payload':{'actor':'b-00'}}],
             expected_revision=m.revision,idempotency_key='safe')
    energy=m._backend.bus.rigid.states['b-00'].dc_energy_j
    advance(m,.01,'protected')
    assert not m._backend.lifecycle.unloader.control_latches
    assert m._backend.bus.rigid.states['b-00'].dc_energy_j==energy


@pytest.mark.parametrize('mode',['nominal','limited'])
def test_program_shared_tank_impulse_and_expected_outcomes(tmp_path,mode):
    scenario=actuator_scenario(configuration(True)['development'],mode=mode)
    samples,report,replay=_execute(scenario,.1,tmp_path)
    witness=_witness(scenario,report,mode)
    assert replay and witness['passed'],witness
    assert _compare(samples,samples)['passed']
    before=copy.deepcopy(report)
    report['lifecycle']['actuation']['fuel_balance_error_kg']['b-00']=.01
    assert not _witness(scenario,report,mode)['passed']
    assert before['lifecycle']['actuation']['states']['b-00']['consumed_propellant_kg']>0
    with pytest.raises(ValueError,match='incomplete'):_compare(samples[:-1],samples)


def test_extended_regression_declares_positive_service_and_keeps_denominator():
    cfg=configuration(True);before=copy.deepcopy(cfg)
    spec=ScaleSpec.model_validate(manifest(cfg))
    assert cfg==before and len(spec.workloads)==4
    assert all(w.scenario['definition']['horizon_sec']==600. for w in spec.workloads)
    assert all(len(w.scenario['definition']['lifecycle']['requests'])==24 for w in spec.workloads)
    positive=[w for w in spec.workloads if 'nominal' in w.id]
    assert all(w.min_completion_fraction==1/24 for w in positive)
    assert all(w.resume_after_decisions==80 for w in spec.workloads if w.resume_in_new_process)


def test_failed_advance_rolls_back_joint_clock_and_mass():
    from space_env.runtime import spacecraft_scenario
    scenario=actuator_scenario(configuration(True)['development'])
    d=scenario['definition']
    for model in d['spacecraft_bus'].values():
        model['attitude']['rigid']['max_total_substeps']=30
    m=MissionRuntime(spacecraft_scenario(d,id='rollback',lineage='D2e'),access=AccessPolicy(truth=True))
    unload(m);advance(m,.171,'prefix');before=m.checkpoint()
    with pytest.raises(RuntimeContractError):advance(m,.2,'exhaust')
    assert m.checkpoint()==before


def test_propellant_exhaustion_stops_joint_controller_without_resetting_wheels():
    from space_env.runtime import spacecraft_scenario
    d=actuator_scenario(configuration(True)['development'])['definition']
    d['orbits']['b-00']['propulsion']['propellant_kg']=1e-7
    m=MissionRuntime(spacecraft_scenario(d,id='fuel-limit',lineage='D2e'),access=AccessPolicy(truth=True))
    unload(m);advance(m,.3,'run')
    report=m._backend.report()['lifecycle']['unloading']
    assert report['operations']['unload']['reason']=='propellant_exhausted'
    assert max(abs(v) for v in report['operations']['unload']['remaining_momentum_nms'])>.07
    assert abs(report['fuel_balance_error_kg']['b-00'])<1e-10
    assert 'b-00' not in m._backend.lifecycle.unloader.control_latches


def test_invalid_clock_rejected_before_creating_output(tmp_path):
    from space_env.validation.coupled_actuators import qualify_coupled_actuators
    output=tmp_path/'invalid'
    with pytest.raises(ValueError,match='fixed-tick'):
        qualify_coupled_actuators(configuration()['development'],output)
    assert not output.exists()


@pytest.fixture
def failed_bundle(tmp_path,monkeypatch):
    import space_env.validation.coupled_actuators as module
    def fail(*args):raise RuntimeError('deliberate retained execution failure')
    monkeypatch.setattr(module,'_execute',fail)
    result=module.qualify_coupled_actuators(configuration(True)['development'],tmp_path)
    assert not result['passed'] and len(result['rows'])==18
    return tmp_path


def reseal_bundle(path,edit):
    """Reseal changed contents so tests exercise semantics, not just checksums."""
    import hashlib
    from space_env.simulation.snapshot import canonical_json,digest
    result=json.loads((path/'result.json').read_text())
    edit(result)
    (path/'progress.json').write_text(canonical_json({'expected':18,'finished':18,'rows':result['rows']}))
    for name in result['files_sha256']:
        result['files_sha256'][name]=hashlib.sha256((path/name).read_bytes()).hexdigest()
    result['sha256']=digest({k:v for k,v in result.items() if k!='sha256'})
    (path/'result.json').write_text(canonical_json(result))


def test_offline_audit_preserves_failed_denominator(failed_bundle):
    from space_env.validation.coupled_actuators import verify_coupled_actuators
    audit=verify_coupled_actuators(failed_bundle)
    assert audit['valid'] and not audit['passed']
    assert audit['failed']==audit['attempts']==18
    assert audit['native_execution_rerun'] is False


@pytest.mark.parametrize('change',[
    lambda r:r['rows'].pop(),
    lambda r:r['rows'].__setitem__(1,copy.deepcopy(r['rows'][0])),
    lambda r:r['budgets'].__setitem__('attitude_deg',1.),
    lambda r:r.__setitem__('passed',True),
    lambda r:r['rows'][0].__setitem__('passed',True),
])
def test_offline_audit_rejects_resealed_false_claims(failed_bundle,change):
    from space_env.validation.coupled_actuators import verify_coupled_actuators
    reseal_bundle(failed_bundle,change)
    with pytest.raises(ValueError):verify_coupled_actuators(failed_bundle)


def test_offline_audit_rejects_scenario_substitution(failed_bundle):
    from space_env.validation.coupled_actuators import verify_coupled_actuators
    file=failed_bundle/'nominal-0.02-0.1/scenario.json'
    scenario=json.loads(file.read_text());scenario['id']='substituted'
    file.write_text(json.dumps(scenario))
    reseal_bundle(failed_bundle,lambda r:None)
    with pytest.raises(ValueError,match='scenario mismatch'):verify_coupled_actuators(failed_bundle)


def test_offline_audit_rejects_corruption_and_extra_files(failed_bundle):
    from space_env.validation.coupled_actuators import verify_coupled_actuators
    file=failed_bundle/'configuration.json';raw=file.read_bytes()
    file.write_bytes(raw+b' ')
    with pytest.raises(ValueError,match='digest mismatch'):verify_coupled_actuators(failed_bundle)
    file.write_bytes(raw)
    (failed_bundle/'extra.json').write_text('{}')
    with pytest.raises(ValueError,match='inventory'):verify_coupled_actuators(failed_bundle)


def test_offline_audit_rejects_symlink(failed_bundle,tmp_path_factory):
    from space_env.validation.coupled_actuators import verify_coupled_actuators
    outside=tmp_path_factory.mktemp('outside')/'payload.json';outside.write_text('{}')
    (failed_bundle/'link.json').symlink_to(outside)
    with pytest.raises(ValueError,match='symlink'):verify_coupled_actuators(failed_bundle)


def test_offline_audit_rejects_duplicate_json_keys(failed_bundle):
    from space_env.validation.coupled_actuators import verify_coupled_actuators
    (failed_bundle/'result.json').write_text('{"passed":true,"passed":false}')
    with pytest.raises(ValueError,match='duplicate'):verify_coupled_actuators(failed_bundle)


def test_comparison_rejects_nonfinite_and_wrong_common_grid():
    from space_env.validation.coupled_actuators import TIMES
    samples=[{'time_sec':t,'actors':{'a':{},'b':{}}} for t in TIMES]
    samples[0]['time_sec']=float('nan')
    with pytest.raises(ValueError):_compare(samples,samples)
    samples[0]['time_sec']=.172
    with pytest.raises(ValueError,match='grid'):_compare(samples,samples)


def test_audit_cli_distinguishes_failed_execution_from_corruption(failed_bundle,monkeypatch,capsys):
    import sys
    from space_env.cli import main
    monkeypatch.setattr(sys,'argv',['space-env','verify-coupled-actuators',str(failed_bundle)])
    with pytest.raises(SystemExit) as failed:main()
    assert failed.value.code==1
    response=json.loads(capsys.readouterr().out)
    assert response['valid'] and response['failed']==18
    (failed_bundle/'result.json').write_text('{}')
    with pytest.raises(SystemExit) as corrupt:main()
    assert corrupt.value.code==2
    response=json.loads(capsys.readouterr().out)
    assert not response['valid'] and not response['passed']
