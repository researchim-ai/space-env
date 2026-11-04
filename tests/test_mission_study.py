"""Refinement is conditional on common grids and unchanged discrete commands."""
import copy
import math

import pytest

from examples.coupled.study import configuration
from space_env.scaling.benchmark import ScaleSpec, Workload
from space_env.scaling.numerics import compare_numerics, refinement_triplet, numerical_snapshot
from space_env.scaling.study import MissionStudySpec, study_manifest, analyze_study
from space_env.simulation.snapshot import digest


def envelope(offset=0.):
    samples = []
    for i in range(3):
        t = float(i * 2)
        samples.append({'time_sec': t, 'actors': {'a': {
            'state_si': [7000000.+offset, 0., 0., 0., 7000., 0.],
            'quaternion_wxyz': [1., 0., 0., 0.], 'omega_body_rad_s': [0.,0.,0.],
            'wheel_momentum_nms': [0.,0.,0.], 'battery_wh': 100.,
            'temperature_k': 285., 'dc_energy_j': t*12.}},
            'requests': {'r': {'status': 'released', 'delivered_sec': None}},
            'control': None if i == 0 else {'action_ids': ['wait'], 'advance_sec': 2., 'accepted': True}})
    return seal(samples)


def seal(samples):
    return {'samples': samples, 'sha256': digest(samples), 'scope': 'test fixture'}


def test_converging_triplet():
    result = refinement_triplet(envelope(.3), envelope(.1), envelope())
    assert result['passed']
    assert result['coarse_middle']['max_errors']['position_m'] == pytest.approx(.2)
    assert result['middle_fine']['rms_errors']['position_m'] == pytest.approx(.1)


def test_growth_is_not_convergence():
    assert not refinement_triplet(envelope(), envelope(.1), envelope(.4))['passed']


def test_large_error_is_not_hidden_by_ratio():
    assert not refinement_triplet(envelope(20.), envelope(10.), envelope())['passed']


def test_quaternion_sign_invariance():
    b = envelope()
    for s in b['samples']:
        s['actors']['a']['quaternion_wxyz'] = [-1., 0., 0., 0.]
    assert compare_numerics(envelope(), seal(b['samples']))['max_errors']['attitude_deg'] == 0.


def test_quaternion_angle_in_physical_degrees():
    b = envelope()
    for s in b['samples']:
        s['actors']['a']['quaternion_wxyz'] = [math.cos(math.pi/360), math.sin(math.pi/360), 0., 0.]
    result = compare_numerics(envelope(), seal(b['samples']))
    assert result['max_errors']['attitude_deg'] == pytest.approx(1.)
    assert not result['passed']


def test_controller_switch_is_separate_from_state_error():
    b = envelope();b['samples'][1]['control']['action_ids'] = ['different']
    assert not compare_numerics(envelope(), seal(b['samples']))['passed']
    result = compare_numerics(envelope(), seal(b['samples']), require_same_schedule=False)
    assert result['passed'] and not result['same_schedule']


def test_outcome_change_is_not_zero_timing_error():
    b = envelope();b['samples'][-1]['requests']['r'] = {'status': 'completed', 'delivered_sec': 4.}
    assert not compare_numerics(envelope(), seal(b['samples']))['passed']


@pytest.mark.parametrize('case', ['checksum', 'unordered', 'nonunit', 'actors', 'control_dt'])
def test_reject_corrupt_sample(case):
    b = envelope()
    if case == 'checksum':
        b['sha256'] = '0'*64
    elif case == 'unordered':
        b['samples'][1]['time_sec'] = 0.
    elif case == 'nonunit':
        b['samples'][1]['actors']['a']['quaternion_wxyz'] = [2.,0.,0.,0.]
    elif case == 'actors':
        b['samples'][1]['actors']['other'] = b['samples'][1]['actors'].pop('a')
    else:
        b['samples'][1]['control']['advance_sec'] = 1.
    if case != 'checksum':
        b = seal(b['samples'])
    with pytest.raises(ValueError):
        compare_numerics(envelope(), b)


def test_partial_restore_grid_cannot_claim_full_span():
    b = envelope();b = seal(b['samples'][1:])
    result = compare_numerics(b, b)
    assert not result['passed']
    assert result['reason'] == 'incomplete_or_mismatched_grid'


def test_default_sampling_does_not_change_legacy_workload_identity():
    manifest, groups = study_manifest(configuration())
    assert len(manifest['workloads']) == 13
    assert len(set(groups['kernel'] + groups['controller'] + groups['events'])) == 7
    spec = ScaleSpec.model_validate(manifest)
    assert len(spec.workloads) == 13
    legacy = Workload.model_validate(manifest['workloads'][0] | {'sample_numerics': False})
    assert 'sample_numerics' not in legacy.model_dump(mode='json')


@pytest.mark.parametrize('case', ['same', 'duplicate_id', 'numeric_only'])
def test_holdouts_must_change_geometric_inputs(case):
    cfg = configuration()
    cfg['holdouts'][0] = copy.deepcopy(cfg['development'])
    if case != 'duplicate_id':
        cfg['holdouts'][0]['id'] = 'different-name'
    if case == 'numeric_only':
        cfg['holdouts'][0]['control_step_sec'] = .05
    with pytest.raises(ValueError):
        MissionStudySpec.model_validate(cfg)


def test_failure_denominator_is_retained():
    cfg = configuration()
    manifest, _ = study_manifest(cfg)
    rows = [{'id': w['id'], 'passed': False, 'error': 'timeout'} for w in manifest['workloads']]
    scale = {'manifest_sha256': digest(ScaleSpec.model_validate(manifest).model_dump(mode='json', by_alias=True)),
             'rows': rows, 'ok': False}
    result = analyze_study(cfg, scale)
    assert not result['passed']
    assert result['expected_attempts'] == result['observed_attempts'] == len(result['execution_errors']) == 13
    assert len(result['holdouts']) == 3
    scale['rows'] = rows[:-1]
    with pytest.raises(ValueError, match='denominator'):
        analyze_study(cfg, scale)


def test_sampling_is_owned_and_does_not_change_policy_or_runtime():
    from space_env.planning import PlanningRun
    from space_env.scaling.coupled import coupled_workload
    from space_env.scaling.worker import _run
    cfg = configuration()['development'] | {'horizon_sec': 30., 'min_fulfilled': 0, 'min_acquired': 0}
    w = coupled_workload(cfg)
    run = PlanningRun(w['scenario'], w['task'], w['policy'])
    before = run.checkpoint()
    sample = numerical_snapshot(run)
    sample['actors']['a-00']['state_si'][0] = -1.
    assert run.checkpoint() == before
    samples = []
    _run(run, None, decisions=2, samples=samples)
    assert [s['time_sec'] for s in samples] == [0.,2.,4.]
    other = PlanningRun(w['scenario'], w['task'], w['policy'])
    other.run(2)
    assert run.trace == other.trace
    assert run.session._runtime.checkpoint() == other.session._runtime.checkpoint()


def synthetic_scale():
    """Aggregation fixture only; does not purport to be native execution."""
    cfg = configuration()
    manifest, groups = study_manifest(cfg)
    rows = []
    for w in manifest['workloads']:
        native = w['scenario']['definition']
        minimum = math.ceil(w['min_completion_fraction'] * len(native['lifecycle']['requests']))
        requests = {r['id']: {'status': 'expired', 'delivered_sec': None} for r in native['lifecycle']['requests']}
        for rid in list(requests)[:minimum]:
            requests[rid] = {'status': 'completed', 'delivered_sec': native['horizon_sec']}
        sample = envelope()['samples'][0]
        actors = {id: copy.deepcopy(sample['actors']['a']) for id in native['orbits']}
        data = []
        start = 10 if w.get('resume_in_new_process') else 0
        for t in range(start, 181, 2):
            data.append({'time_sec': float(t), 'actors': copy.deepcopy(actors),
                         'requests': copy.deepcopy(requests),
                         'control': None if t == 0 else {'action_ids': ['wait'], 'advance_sec': 2., 'accepted': True}})
        gates = {k: True for k in ['native_replay','quality','step_latency_budget','execution_budget',
                                  'checkpoint_budget','peak_rss_budget','completion','coupled_mission']}
        if w.get('max_completion_fraction') is not None:
            gates['completion_upper_bound'] = True
        if w.get('resume_in_new_process'):
            gates['process_restart'] = True
        total = len(requests)
        rows.append({'id': w['id'], 'passed': True, 'gates': gates, 'metrics': {
            'total': total, 'fulfilled': minimum, 'completion_fraction': minimum/total, 'physical_sec': 180.},
            'trace_sha256': 'a'*64, 'native_backend_sha256': 'b'*64, 'numerical_samples': seal(data)})
    return cfg, {'schema': 'space-env.scale-result/1', 'manifest_sha256': digest(ScaleSpec.model_validate(manifest).model_dump(mode='json', by_alias=True)),
                 'rows': rows, 'ok': True}


def test_synthetic_aggregation_and_artifact_audit(tmp_path):
    from space_env.scaling.study import audit_mission_study
    from space_env.scaling.restart import build_identity
    from space_env.simulation.snapshot import canonical_json
    cfg, scale = synthetic_scale()
    assessment = analyze_study(cfg, scale)
    assert assessment['passed']
    frozen = MissionStudySpec.model_validate(cfg).model_dump(mode='json', by_alias=True)
    scale['sha256'] = digest(scale)
    result = {'schema': 'space-env.mission-study-result/1', 'spec_sha256': digest(frozen), 'scale_sha256': scale['sha256'],
              'assessment': assessment, 'build': build_identity()}
    result['sha256'] = digest(result)
    (tmp_path/'execution').mkdir()
    (tmp_path/'study.json').write_text(canonical_json(frozen))
    (tmp_path/'execution/result.json').write_text(canonical_json(scale))
    manifest, _ = study_manifest(cfg)
    manifest = ScaleSpec.model_validate(manifest).model_dump(mode='json', by_alias=True)
    (tmp_path/'execution/manifest.json').write_text(canonical_json({'manifest': manifest, 'sha256': digest(manifest)}))
    (tmp_path/'result.json').write_text(canonical_json(result))
    assert audit_mission_study(tmp_path)['qualification_passed']
    result['assessment']['kernel']['passed'] = False
    result['sha256'] = digest({k:v for k,v in result.items() if k != 'sha256'})
    (tmp_path/'result.json').write_text(canonical_json(result))
    with pytest.raises(ValueError, match='recomputation'):
        audit_mission_study(tmp_path)


@pytest.mark.parametrize('case', ['grid', 'identity', 'gates', 'mission', 'duplicate'])
def test_reject_invalid_raw_aggregation(case):
    cfg, scale = synthetic_scale()
    row = scale['rows'][0]
    if case == 'grid':
        data = row['numerical_samples']['samples']
        del data[1]
        data[1]['control']['advance_sec'] = 4.
        row['numerical_samples'] = seal(data)
        assert not analyze_study(cfg, scale)['gates']['kernel_common_grid']
        return
    if case == 'identity':
        for s in row['numerical_samples']['samples']:
            s['actors'].pop('a-00')
        row['numerical_samples'] = seal(row['numerical_samples']['samples'])
    elif case == 'gates':
        row['gates']['native_replay'] = False
    elif case == 'mission':
        row['metrics']['fulfilled'] = 0
    else:
        scale['rows'].append(row)
    with pytest.raises(ValueError):
        analyze_study(cfg, scale)


def test_passive_coast_allows_zero_adcs_without_free_power():
    from space_env.planning import PlanningRun
    from space_env.scaling.coupled import coupled_workload, coupled_evidence
    cfg = configuration()['holdouts'][2] | {'horizon_sec': 30.}
    w = coupled_workload(cfg)
    run = PlanningRun(w['scenario'], w['task'], w['policy'])
    run.run(2)
    report = run.session._runtime._backend.report()
    assert all(s['rigid_attitude']['dc_energy_j'] == 0. for s in report['spacecraft_bus']['states'].values())
    assert coupled_evidence(w['scenario'], report, w['coupled_gates'])['passed']
    bad = copy.deepcopy(report)
    bad['spacecraft_bus']['states']['a-00']['rigid_attitude']['dc_energy_j'] = 12.
    assert not coupled_evidence(w['scenario'], bad, w['coupled_gates'])['passed']


@pytest.mark.parametrize('field', ['slew_energy_wh', 'slew_sec'])
def test_active_adcs_must_match_both_energy_and_powered_duration(field):
    from tests.test_coupled_mission import evidence_case
    from space_env.scaling.coupled import coupled_evidence
    w, report = evidence_case.__wrapped__()
    report['spacecraft_bus']['states']['a-00'][field] = 0.
    assert not coupled_evidence(w['scenario'], report, w['coupled_gates'])['passed']
