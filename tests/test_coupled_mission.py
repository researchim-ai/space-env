"""Independent coupled inputs, causal steering policy and retained failures."""
import copy

import pytest

from examples.coupled.demo import configuration
from space_env.scaling.coupled import CoupledMissionSpec, coupled_workload
from space_env.scaling.benchmark import Workload
from space_env.runtime import MissionRuntime
from space_env.planning import PlanningRun
from space_env.planning.model import make_input
from space_env.simulation.snapshot import digest


def test_compiler_is_deterministic_owned_and_preserves_denominator():
    cfg = configuration()
    saved = copy.deepcopy(cfg)
    w = coupled_workload(cfg)
    assert cfg == saved
    assert w == coupled_workload(cfg)
    assert Workload.model_validate(w)
    d = w['scenario']['definition']
    assert d['operations']['products'] == []
    assert len(d['lifecycle']['requests']) == 24
    assert len(w['task']['actions']) == 96
    assert d['acquisition_targets'] == [t.model_dump(mode='json') for t in CoupledMissionSpec.model_validate(cfg).targets]
    assert all(b['attitude']['initial_quaternion_wxyz'] == [1., 0., 0., 0.] for b in d['spacecraft_bus'].values())


@pytest.mark.parametrize('change', ['weak_wheels', 'all_outage'])
def test_negative_changes_do_not_condition_geometry(change):
    a = coupled_workload(configuration())['scenario']['definition']
    b = coupled_workload(configuration() | {'mode': change})['scenario']['definition']
    for key in ['orbits', 'acquisition_targets', 'acquisition_sensors', 'lifecycle']:
        assert a[key] == b[key]
    assert a['operations']['stations'] == b['operations']['stations']
    if change == 'weak_wheels':
        assert b['spacecraft_bus']['a-00']['attitude']['rigid']['model']['max_torque_nm'] == [1e-6] * 3
    else:
        assert len(b['operations']['outages']) == 12


@pytest.mark.parametrize('case', ['duplicate', 'same_plane', 'excess_actors', 'gate'])
def test_reject_invalid_spec(case):
    cfg = configuration()
    if case == 'duplicate':
        cfg['targets'].append(cfg['targets'][0])
    elif case == 'same_plane':
        cfg['planes'][1].update(inclination_deg=55., raan_deg=0.)
    elif case == 'excess_actors':
        cfg['planes'][0]['phases_deg'] = [0., 20., 40., 60.]
    else:
        cfg['min_fulfilled'] = 25
    with pytest.raises(ValueError):
        coupled_workload(cfg)


def test_short_native_replay_and_policy_uses_public_pointing():
    w = coupled_workload(configuration() | {'horizon_sec': 30.})
    run = PlanningRun(w['scenario'], w['task'], w['policy'])
    obs = run.session.observe()
    pointings = [a['pointing'] for a in obs['catalogue'] if a.get('pointing')]
    assert pointings
    before = run.session._runtime.checkpoint()
    public = make_input(run.session)
    assert run.session._runtime.checkpoint() == before
    decision = run.policy.decide(public)
    assert decision['action_ids']
    run.run(2)
    restored = PlanningRun.from_checkpoint(run.checkpoint())
    assert restored.session.observe() == run.session.observe()
    assert restored.metrics()['total'] == 24


@pytest.fixture(scope='module')
def evidence_case():
    from space_env.scaling.coupled import coupled_evidence
    w = coupled_workload(configuration() | {'horizon_sec': 30.})
    run = PlanningRun(w['scenario'], w['task'], w['policy'])
    run.run(2)
    report = run.session._runtime._backend.report()
    assert coupled_evidence(w['scenario'], report, w['coupled_gates'])['passed']
    return w, report


@pytest.mark.parametrize('case', ['missing_request', 'missing_actor', 'balance', 'energy', 'wrong_target', 'false_completion'])
def test_evidence_rejects_tampering(evidence_case, case):
    from space_env.scaling.coupled import coupled_evidence
    w, original = evidence_case
    report = copy.deepcopy(original)
    life = report['lifecycle']
    rid = next(iter(life['requests']))
    if case == 'missing_request':
        del life['requests'][rid]
    elif case == 'missing_actor':
        del report['spacecraft_bus']['states']['a-00']
    elif case == 'balance':
        life['ledger']['a-00']['balance_error_bits'] = 1000.
    elif case == 'energy':
        report['spacecraft_bus']['states']['a-00']['rigid_attitude']['dc_heat_work_balance_error_j'] = 2.
    elif case == 'wrong_target':
        life['requests'][rid]['target_id'] = 'other'
    else:
        life['requests'][rid]['status'] = 'completed'
        life['metrics']['fulfilled_requests'] = 1
    assert not coupled_evidence(w['scenario'], report, w['coupled_gates'])['passed']


def test_evidence_binding_and_explicit_minimum(evidence_case):
    from space_env.scaling.coupled import coupled_evidence
    w, report = evidence_case
    with pytest.raises(ValueError, match='binding'):
        coupled_evidence(w['scenario'], report, w['coupled_gates'] | {'native_spec_sha256': '0' * 64})
    assert not coupled_evidence(w['scenario'], report, w['coupled_gates'] | {'min_acquired': 1})['passed']
    legacy = Workload.model_validate({k:v for k,v in w.items() if k != 'coupled_gates'})
    assert 'coupled_gates' not in legacy.model_dump(mode='json')


def test_public_policy_allows_settling_instead_of_restarting_slew():
    from space_env.planning.model import PlanningInput
    from space_env.planning.policy import BaselinePolicy
    from space_env.simulation.snapshot import canonical_json
    w = coupled_workload(configuration() | {'horizon_sec': 30.})
    run = PlanningRun(w['scenario'], w['task'], w['policy'])
    value = make_input(run.session).view()
    for action in value['observation']['catalogue']:
        for p in action.get('pointing', []):
            p.update(boresight_error_deg=0., body_rate_rad_s=0.)
    decision = BaselinePolicy(w['policy']).decide(PlanningInput(canonical_json(value)))
    assert decision['action_ids'] == ['wait']
