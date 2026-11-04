"""Fixed digital ticks, command interrupts and paid sample-and-hold actuation."""
import copy
import numpy as np
import pytest

from space_env.runtime import AccessPolicy, MissionRuntime, spacecraft_scenario
from space_env.reference.rigid_bus import RigidControllerSettings
from tests import test_rigid_mission as rigid
from tests import test_reference_lifecycle as life


def scenario():
    d = copy.deepcopy(rigid.scenario.__wrapped__()['definition'])
    d['spacecraft_bus']['a']['attitude']['rigid']['clock_mode'] = 'fixed-tick/1'
    return spacecraft_scenario(d, id='fixed-clock', lineage='D2d')


def run():
    m = MissionRuntime(scenario(), access=AccessPolicy(truth=True))
    life.submit(m,'slew',5.,'slew',slew={'frame':'GCRS','quaternion_wxyz':[.7071067811865476,.7071067811865476,0.,0.]})
    return m


def view(m):
    return rigid.view(m)


def test_legacy_serialization():
    model = RigidControllerSettings.model_validate(rigid.config())
    assert 'clock_mode' not in model.model_dump(mode='json')
    assert RigidControllerSettings.model_validate(rigid.config(clock_mode='fixed-tick/1')).model_dump()['clock_mode'] == 'fixed-tick/1'


def test_hold_across_non_tick_boundaries_and_tick_update():
    m = run()
    life.advance(m,.03,'first')
    first = view(m)
    life.advance(m,.02,'event')
    held = view(m)
    assert first['sample_clock']['samples'] == held['sample_clock']['samples'] == 1
    assert held['controller']['torque'] == first['controller']['torque']
    life.advance(m,.06,'past-tick')
    tick = view(m)
    assert tick['sample_clock']['samples'] == 2
    assert tick['sample_clock']['last_sample_sec'] == pytest.approx(.1)
    assert tick['controller']['torque'] != first['controller']['torque']


def test_cancel_and_new_command_interrupt_between_ticks():
    m = run()
    life.advance(m,.03,'first')
    m.cancel('slew',expected_revision=m.revision,idempotency_key='cancel')
    life.submit(m,'wheel_torque',1.,'manual',wheel_torque={'frame':'BODY','motor_torque_nm':[.02,0.,0.]})
    life.advance(m,.01,'command-interrupt')
    v = view(m)
    assert v['sample_clock']['samples'] == 2
    assert v['sample_clock']['last_sample_sec'] == pytest.approx(.03)
    assert v['controller']['torque'] == [.02,0.,0.]
    m.cancel('manual',expected_revision=m.revision,idempotency_key='cancel2')
    energy = view(m)['dc_energy_j']
    life.advance(m,.01,'coast')
    assert view(m)['dc_energy_j'] == energy
    assert not view(m)['controller']['enabled']


def test_mid_tick_checkpoint_replay_preserves_latch():
    m = run();life.advance(m,.03,'prefix')
    restored = MissionRuntime.from_checkpoint(m.checkpoint(),access=AccessPolicy(truth=True))
    for model in [m,restored]:
        life.advance(model,.08,'continuation')
    assert m.checkpoint() == restored.checkpoint()


def test_extra_native_boundaries_do_not_resample_controller():
    a,b = run(),run()
    life.advance(a,.4,'whole')
    for i in range(16):
        life.advance(b,.025,'part'+str(i))
    va,vb = view(a),view(b)
    assert va['sample_clock']['samples'] == vb['sample_clock']['samples'] == 4
    np.testing.assert_allclose(va['quaternion_wxyz'],vb['quaternion_wxyz'],atol=1e-11,rtol=0)
    np.testing.assert_allclose(va['omega_body_rad_s'],vb['omega_body_rad_s'],atol=1e-11,rtol=0)
    assert va['dc_energy_j'] == pytest.approx(vb['dc_energy_j'],abs=1e-10)


def test_protection_interrupt_stops_power_immediately():
    m = run();life.advance(m,.03,'prefix')
    rigid.inject(m,'spacecraft_safe_mode')
    energy = view(m)['dc_energy_j']
    life.advance(m,.01,'protected')
    assert view(m)['dc_energy_j'] == energy
    assert not view(m)['controller']['enabled']
    assert 'a' not in m._backend.bus.rigid.latches


def test_fork_owns_latched_torque():
    m = run();life.advance(m,.03,'prefix')
    child = m.fork()
    child._backend.bus.rigid.latches['a']['torque'][0] = 99.
    assert m._backend.bus.rigid.latches['a']['torque'][0] != 99.


def test_failed_advance_rolls_back_clock_and_latch():
    from space_env.runtime import RuntimeContractError
    s = scenario()
    s['definition']['spacecraft_bus']['a']['attitude']['rigid']['max_total_substeps'] = 15
    m = MissionRuntime(s,access=AccessPolicy(truth=True))
    life.submit(m,'wheel_torque',1.,'wheel',wheel_torque={'frame':'BODY','motor_torque_nm':[.02,0.,0.]})
    life.advance(m,.03,'prefix')
    before = m.checkpoint()
    with pytest.raises(RuntimeContractError):
        life.advance(m,.17,'exhaust')
    assert m.checkpoint() == before


def test_unknown_clock_mode_fails_closed():
    with pytest.raises(ValueError):
        RigidControllerSettings.model_validate(rigid.config(clock_mode='ideal-teleport'))


def test_fixed_study_makes_event_stability_mandatory():
    from examples.coupled.study import configuration
    from space_env.scaling.study import MissionStudySpec, study_manifest
    config = configuration(True)
    spec = MissionStudySpec.model_validate(config)
    assert spec.require_event_stability
    manifest,_ = study_manifest(config)
    assert len(manifest['workloads']) == 13
    assert all(c['attitude']['rigid']['clock_mode'] == 'fixed-tick/1'
               for w in manifest['workloads'] for c in w['scenario']['definition']['spacecraft_bus'].values())
    assert 'require_event_stability' not in MissionStudySpec.model_validate(configuration()).model_dump(mode='json')


def test_frozen_native_program_and_replay(tmp_path):
    from examples.coupled.study import configuration
    from space_env.scaling.coupled import coupled_workload
    from space_env.validation.controller_clock import qualify_controller_clock
    w = coupled_workload(configuration(True)['development'] | {'horizon_sec':30.,'min_fulfilled':0,'min_acquired':0})
    result=qualify_controller_clock(w['scenario'],tmp_path/'program')
    assert result['passed']
    assert len(result['rows']) == 3
    assert all(row['native_replay'] for row in result['rows'])
    from space_env.validation.controller_clock import BUDGETS
    result['budgets']['attitude_deg'] = 99.
    assert BUDGETS['attitude_deg'] == 1e-7
    with pytest.raises(ValueError,match='empty'):
        qualify_controller_clock(w['scenario'],tmp_path/'program')


def test_program_retains_all_failed_partitions(tmp_path):
    from examples.coupled.study import configuration
    from space_env.scaling.coupled import coupled_workload
    from space_env.validation.controller_clock import qualify_controller_clock
    w=coupled_workload(configuration(True)['development'] | {'horizon_sec':30.,'min_fulfilled':0,'min_acquired':0})
    for model in w['scenario']['definition']['spacecraft_bus'].values():
        model['attitude']['rigid']['max_total_substeps']=1
    result=qualify_controller_clock(w['scenario'],tmp_path/'failed')
    assert not result['passed']
    assert len(result['rows']) == 3
    assert all('error' in row for row in result['rows'])
    assert all(not c['passed'] for c in result['comparisons'])


def test_latched_motor_still_obeys_momentum_capacity():
    s=scenario()
    s['definition']['spacecraft_bus']['a']['attitude']['rigid']['model']['max_momentum_nms']=[.001]*3
    m=MissionRuntime(s,access=AccessPolicy(truth=True))
    life.submit(m,'wheel_torque',1.,'wheel',wheel_torque={'frame':'BODY','motor_torque_nm':[.1,0.,0.]})
    life.advance(m,.2,'saturate')
    v=view(m)
    assert max(abs(x) for x in v['wheel_momentum_nms']) <= .001 + 1e-12
    assert v['capacity_limited_sec'][0] > 0.
    assert abs(v['dc_heat_work_balance_error_j']) < 1e-9
    assert abs(v['kinetic_work_balance_error_j']) < 1e-9
    assert v['dc_energy_j'] == pytest.approx(2.4)


def test_event_clock_gate_requires_same_native_command_schedule():
    from tests.test_mission_study import synthetic_scale, seal
    from space_env.scaling.study import analyze_study, study_manifest
    cfg,scale=synthetic_scale()
    cfg['require_event_stability']=True
    _,ids=study_manifest(cfg)
    row=next(r for r in scale['rows'] if r['id']==ids['events'][1])
    row['numerical_samples']['samples'][1]['control']['action_ids']=['different']
    row['numerical_samples']=seal(row['numerical_samples']['samples'])
    result=analyze_study(cfg,scale)
    assert result['event_sensitivity']['passed']
    assert not result['gates']['event_clock_stability']
    assert not result['passed']
