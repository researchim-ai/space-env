"""Snapshot reference commands preserve paid control and legacy serialization."""
import copy

import pytest
from pydantic import ValidationError

from space_env.runtime import AccessPolicy, MissionRuntime, RuntimeContractError
from space_env.runtime.schema import SlewSpec
from tests import test_spacecraft_bus as fixtures
from tests import test_reference_lifecycle as life


@pytest.mark.parametrize('fields', [{}, {'target_id': ' '}, {'target_id': 'site', 'station_id': 'Ground'}, {'quaternion_wxyz': [0., 0., 0., 0.]}])
def test_invalid_goals(fields):
    with pytest.raises(ValidationError):
        SlewSpec(frame='GCRS', **fields)


def test_legacy_serialization():
    old = {'frame': 'GCRS', 'quaternion_wxyz': [1., 0., 0., 0.]}
    assert SlewSpec.model_validate(old).model_dump(mode='json') == old


@pytest.mark.parametrize('reference', [{'target_id': 'site'}, {'station_id': 'Ground'}])
def test_snapshot_goal_paid_execution_and_replay(reference):
    m = fixtures.runtime(fixtures.scenario.__wrapped__())
    before = m.checkpoint()
    view = m.pointing_observation('a', **reference)
    view['goal_quaternion_wxyz'][0] = 99.
    assert m.checkpoint() == before
    expected = m.pointing_observation('a', **reference)['goal_quaternion_wxyz']
    attitude = copy.deepcopy(m._backend.bus.states['a'])
    life.submit(m, 'slew', 5., 'reference', slew={'frame': 'GCRS', **reference})
    assert m._backend.bus.states['a'] == attitude
    goal = copy.deepcopy(m._backend.bus.operations['reference']['goal'])
    assert goal == expected
    restored = MissionRuntime.from_checkpoint(m.checkpoint(), access=AccessPolicy(truth=True, forecast=True))
    for model in (m, restored):
        life.advance(model, 1., 'execute')
        assert model._backend.bus.operations['reference']['goal'] == goal
        assert model._backend.bus.states['a']['electrical_load_wh'] > 0.
    assert m.checkpoint() == restored.checkpoint()
    assert m.pointing_observation('a', **reference)['goal_quaternion_wxyz'] != goal


def test_bad_reference_atomic():
    m = fixtures.runtime(fixtures.scenario.__wrapped__())
    before = m.checkpoint()
    with pytest.raises(RuntimeContractError):
        life.submit(m, 'slew', 1., 'bad', slew={'frame': 'GCRS', 'target_id': 'missing'})
    assert m.checkpoint() == before
    with pytest.raises(RuntimeContractError):
        m.pointing_observation('a', target_id='missing')
    assert m.checkpoint() == before


def task(mask=True):
    return {
        'id': 'causal-pointing', 'pointing_masks': mask,
        'roles': {'controller': {'actors': ['a']}},
        'actions': [
            {'id': 'image', 'role': 'controller', 'kind': 'submit', 'commands': [
                {'command_id': 'image', 'actor_id': 'a', 'kind': 'imaging',
                 'duration_sec': 2., 'request_id': 'request'}]},
            {'id': 'slew', 'role': 'controller', 'kind': 'submit', 'commands': [
                {'command_id': 'slew', 'actor_id': 'a', 'kind': 'slew',
                 'duration_sec': 2., 'slew': {'frame': 'GCRS', 'target_id': 'site'}}]},
        ],
    }


def test_current_masks_do_not_probe_future_or_mutate():
    from space_env.tasks import TaskSession
    from space_env.tasks.observation import Projection
    from space_env.tasks.schema import TaskSpec

    scenario = fixtures.scenario.__wrapped__()
    m = fixtures.runtime(scenario)
    spec = TaskSpec.model_validate(task())
    before = m.checkpoint()
    result = Projection(spec).build(m, 'controller', 0, spec.actions)
    assert result['catalogue'][1]['enabled']
    assert result['catalogue'][2]['enabled']
    assert m.checkpoint() == before
    # Point the payload away without changing its geometric access.
    m._backend.bus.states['a']['quaternion_wxyz'] = [1., 0., 0., 0.]
    result = Projection(spec).build(m, 'controller', 0, spec.actions)
    assert result['catalogue'][1]['reason'] == 'current_pointing_error'
    # Legacy task behaviour is explicitly opt-out and byte-compatible.
    old = TaskSpec.model_validate(task(False))
    assert 'pointing_masks' not in old.model_dump(mode='json')
    assert Projection(old).build(m, 'controller', 0, old.actions)['catalogue'][1]['enabled']
    session = TaskSession(scenario, task())
    restored = TaskSession.from_checkpoint(session.checkpoint())
    assert restored.observe() == session.observe()


def test_unreleased_target_is_not_exposed_by_reference_action():
    from space_env.tasks import TaskSession
    scenario = copy.deepcopy(fixtures.scenario.__wrapped__())
    scenario['definition']['lifecycle']['requests'][0]['release_sec'] = 10.
    obs = TaskSession(scenario, task()).observe('controller')
    assert obs['catalogue'][2]['reason'] == 'request_not_available'
    assert 'commands' not in obs['catalogue'][2]


def test_reference_slew_rigid_controller_still_pays_energy():
    from tests import test_rigid_mission as rigid
    m = fixtures.runtime(rigid.scenario.__wrapped__())
    life.submit(m, 'slew', 1., 'reference', slew={'frame': 'GCRS', 'target_id': 'site'})
    life.advance(m, 1., 'execute')
    state = rigid.view(m)
    assert state['dc_energy_j'] > 0.
    assert abs(state['dc_heat_work_balance_error_j']) < 1e-8


@pytest.mark.parametrize('case', ['settling', 'geometry', 'rate'])
def test_masks_current_limits(case):
    from space_env.tasks.observation import Projection
    from space_env.tasks.schema import TaskSpec
    from tests import test_rigid_mission as rigid
    scenario = (rigid.scenario.__wrapped__() if case == 'rate'
                else fixtures.scenario.__wrapped__())
    m = fixtures.runtime(scenario)
    if case == 'settling':
        m._backend.bus.states['a']['settle_remaining_sec'] = 1.
        reason = 'current_settling'
    elif case == 'geometry':
        from space_env.reference.acquisition import AcquisitionTarget
        m._backend.acquisition_targets['site'] = AcquisitionTarget(
            id='site', lat_deg=89., lon_deg=0.)
        reason = 'current_geometry_unavailable'
    else:
        m._backend.bus.rigid.states['a'] = m._backend.bus.rigid.states['a'].model_copy(
            update={'omega_body_rad_s': (1., 0., 0.)})
        reason = 'current_body_rate'
    spec = TaskSpec.model_validate(task())
    obs = Projection(spec).build(m, 'controller', 0, spec.actions)
    assert obs['catalogue'][1]['reason'] == reason
