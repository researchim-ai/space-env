"""Frozen mission refinement/sensitivity studies using isolated scale workers."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import Field, model_serializer, model_validator

from space_env.runtime.schema import StrictModel
from space_env.scaling.coupled import CoupledMissionSpec, coupled_workload
from space_env.scaling.numerics import NumericalLimits, compare_numerics, refinement_triplet, _series
from space_env.simulation.snapshot import atomic_write, canonical_json, digest


class MissionStudySpec(StrictModel):
    schema_id: Literal['space-env.mission-study/1'] = Field(default='space-env.mission-study/1', alias='schema')
    id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,30}$')
    development: CoupledMissionSpec
    holdouts: tuple[CoupledMissionSpec, ...] = Field(min_length=1, max_length=4)
    limits: NumericalLimits = Field(default_factory=NumericalLimits)
    workers: int = Field(default=1, ge=1, le=4, strict=True)
    require_temporal_stability: bool = Field(default=False, strict=True)
    require_event_stability: bool = Field(default=False, strict=True)

    @model_serializer(mode='wrap')
    def legacy_requirements(self, handler):
        value = handler(self)
        if not self.require_event_stability:
            value.pop('require_event_stability', None)
        return value


    @model_validator(mode='after')
    def independent_geometry(self):
        def geometry(c):
            return digest(c.model_dump(mode='json', include={'epoch_utc', 'planes', 'targets', 'stations'}))
        cases = [self.development, *self.holdouts]
        if len({c.id for c in cases}) != len(cases) or len({geometry(c) for c in cases}) != len(cases):
            raise ValueError('holdout cases require distinct IDs and distinct geometric inputs')
        return self


def study_manifest(value):
    cfg = MissionStudySpec.model_validate(value)
    development = cfg.development.model_dump(mode='json')
    workloads = []
    ids = {}
    # Absolute levels, frozen before execution. Only the named axis changes.
    for family, key, levels in [
        ('kernel', 'rigid_step_sec', [.02, .01, .005]),
        ('controller', 'control_step_sec', [.2, .1, .05]),
        ('events', 'lifecycle_step_sec', [.5, .25, .125]),
    ]:
        ids[family] = []
        for index, level in enumerate(levels):
            # The configured development numerical settings are reset explicitly.
            c = development | {'rigid_step_sec': .01, 'control_step_sec': .1,
                               'lifecycle_step_sec': .5, key: level}
            identity = digest({k:v for k,v in c.items() if k != 'id'})
            previous = next((w for fingerprint, w in workloads if fingerprint == identity), None)
            if previous is None:
                c['id'] = f'{cfg.id}-{family}-{index}'
                w = coupled_workload(c) | {'sample_numerics': True}
                workloads.append((identity, w))
            else:
                w = previous
            ids[family].append(w['id'])
    ids['holdouts'] = []
    for index, case in enumerate(cfg.holdouts):
        c = case.model_dump(mode='json') | {'id': f'{cfg.id}-holdout-{index}'}
        w = coupled_workload(c) | {'sample_numerics': True}
        w['comparison_group'] = w['id']
        restored = w | {'id': w['id'] + '-restored', 'resume_after_decisions': 5,
                        'resume_in_new_process': True}
        workloads.extend([(None, w), (None, restored)])
        ids['holdouts'].append({'case_id': case.id, 'direct': w['id'], 'restored': restored['id']})
    return {'id': cfg.id, 'workers': cfg.workers, 'workloads': [w for _, w in workloads]}, ids


def analyze_study(value, scale_result):
    cfg = MissionStudySpec.model_validate(value)
    manifest, ids = study_manifest(cfg)
    from space_env.scaling.benchmark import ScaleSpec
    frozen = ScaleSpec.model_validate(manifest).model_dump(mode='json', by_alias=True)
    if scale_result.get('manifest_sha256') != digest(frozen):
        raise ValueError('study scale-manifest binding mismatch')
    expected = {w['id']: w for w in manifest['workloads']}
    rows = scale_result.get('rows', [])
    indexed = {r['id']: r for r in rows}
    if len(rows) != len(indexed) or set(indexed) != set(expected):
        raise ValueError('study attempt denominator mismatch')
    errors = {}
    samples = {}
    for id, row in indexed.items():
        gates = row.get('gates')
        if isinstance(gates, dict):
            required = {'native_replay', 'quality', 'step_latency_budget', 'execution_budget',
                        'checkpoint_budget', 'peak_rss_budget', 'completion', 'coupled_mission'}
            if expected[id].get('max_completion_fraction') is not None:
                required.add('completion_upper_bound')
            if expected[id].get('resume_in_new_process'):
                required.add('process_restart')
            if (set(gates) != required or any(type(v) is not bool for v in gates.values())
                or row.get('passed') != (all(gates.values()) and row.get('error') is None)):
                raise ValueError('study worker gate consistency mismatch')
        elif row.get('passed'):
            raise ValueError('successful study worker has no gates')
        if row.get('passed'):
            metrics = row['metrics']
            total = len(expected[id]['scenario']['definition']['lifecycle']['requests'])
            count = metrics['fulfilled']
            fraction = metrics['completion_fraction']
            if (type(count) is not int or not 0 <= count <= total or metrics['total'] != total
                or fraction != count / total
                or fraction < expected[id]['min_completion_fraction']
                or (expected[id].get('max_completion_fraction') is not None
                    and fraction > expected[id]['max_completion_fraction'])):
                raise ValueError('study worker mission metrics mismatch')
        if not row.get('passed') or 'numerical_samples' not in row:
            errors[id] = row.get('error') or 'execution_gate_failed_or_samples_missing'
        elif row['numerical_samples']['sha256'] != digest(row['numerical_samples']['samples']):
            raise ValueError('study numerical sample checksum mismatch')
        else:
            series = _series(row['numerical_samples'])
            native = expected[id]['scenario']['definition']
            if (set(series[0].actors) != set(native['orbits'])
                or set(series[0].requests) != {r['id'] for r in native['lifecycle']['requests']}
                or series[-1].time_sec != row['metrics']['physical_sec']):
                raise ValueError('study sample identity/time binding mismatch')
            samples[id] = row['numerical_samples']
    def triplet(family):
        names = ids[family]
        if any(n not in samples for n in names):
            return {'passed': False, 'reason': 'missing_execution', 'ids': names}
        if family == 'kernel':
            result = refinement_triplet(*(samples[n] for n in names), limits=cfg.limits)
        else:
            result = {'coarse_middle': compare_numerics(samples[names[0]], samples[names[1]], cfg.limits, require_same_schedule=False),
                      'middle_fine': compare_numerics(samples[names[1]], samples[names[2]], cfg.limits, require_same_schedule=False),
                      'scope': 'discrete-model/event-resolution sensitivity; not numerical convergence'}
            result['passed'] = result['coarse_middle']['passed'] and result['middle_fine']['passed']
        result['ids'] = names
        return result
    kernel, controller, events = (triplet(f) for f in ['kernel', 'controller', 'events'])
    def full_grid(id):
        if id not in samples:
            return False
        horizon = expected[id]['scenario']['definition']['horizon_sec']
        step = expected[id]['task']['decision_sec']
        times = [min(horizon, i*step) for i in range(math.ceil(horizon/step)+1)]
        actual = samples[id]['samples']
        return len(actual) == len(times) and all(
            math.isclose(s['time_sec'], t, rel_tol=0, abs_tol=1e-8) for s,t in zip(actual,times))
    full_coverage = all(full_grid(id) for id in set(ids['kernel'] + ids['controller'] + ids['events']))
    temporal_stable = controller['passed'] and events['passed']
    holdouts = []
    for case in ids['holdouts']:
        direct, restored = indexed[case['direct']], indexed[case['restored']]
        match = (direct.get('passed') is True and restored.get('passed') is True
                 and direct.get('trace_sha256') == restored.get('trace_sha256')
                 and direct.get('native_backend_sha256') == restored.get('native_backend_sha256'))
        holdouts.append({**case, 'paired_native_match': match,
                         'metrics': direct.get('metrics'),
                         'mission_expectation_minimum': expected[case['direct']]['min_completion_fraction'],
                         'witness': direct.get('coupled_evidence'),
                         'error': errors.get(case['direct']) or errors.get(case['restored'])})
    gates = {'native_execution': scale_result.get('ok') is True and all(r.get('passed') is True for r in rows) and not errors,
             'kernel_common_grid': full_coverage, 'kernel_self_refinement': kernel['passed'],
             'holdout_restore_pairs': all(c['paired_native_match'] for c in holdouts),
             'temporal_stability_if_required': temporal_stable or not cfg.require_temporal_stability}
    if cfg.require_event_stability:
        gates['event_clock_stability'] = (events['passed']
            and events.get('coarse_middle', {}).get('same_schedule') is True
            and events.get('middle_fine', {}).get('same_schedule') is True)
    return {'passed': all(gates.values()), 'gates': gates, 'kernel': kernel,
            'controller_sensitivity': controller, 'event_sensitivity': events,
            'temporal_stable': temporal_stable, 'temporal_stability_required': cfg.require_temporal_stability,
            'holdouts': holdouts, 'execution_errors': errors,
            'expected_attempts': len(expected), 'observed_attempts': len(rows),
            'scope': 'fixed engineering evaluation cases; not statistical generalization, calibrated truth or accuracy certification'}


def run_mission_study(value, output):
    from space_env.scaling.benchmark import run_scale
    from space_env.scaling.restart import build_identity
    cfg = MissionStudySpec.model_validate(value)
    path = Path(output)
    if path.exists() and any(path.iterdir()):
        raise ValueError('study output must be empty')
    path.mkdir(parents=True, exist_ok=True)
    spec = cfg.model_dump(mode='json', by_alias=True)
    atomic_write(path / 'study.json', canonical_json(spec))
    manifest, _ = study_manifest(cfg)
    result = run_scale(manifest, path / 'execution')
    assessment = analyze_study(cfg, result)
    body = {'schema': 'space-env.mission-study-result/1', 'spec_sha256': digest(spec),
            'scale_sha256': result['sha256'], 'build': build_identity(), 'assessment': assessment,
            'scope': 'local model verification, not flight/hardware qualification'}
    sealed = body | {'sha256': digest(body)}
    atomic_write(path / 'result.json', canonical_json(sealed))
    return sealed


def audit_mission_study(directory):
    """Offline checksum/recomputation audit; no fresh mission or authentication."""
    from space_env.scaling.restart import build_identity
    path = Path(directory)
    from space_env.scaling.restart import read_json
    spec = read_json(path / 'study.json', 2_000_000)
    result = read_json(path / 'result.json', 64_000_000)
    scale = read_json(path / 'execution/result.json', 64_000_000)
    manifest = read_json(path / 'execution/manifest.json', 64_000_000)
    if (result.get('schema') != 'space-env.mission-study-result/1'
        or scale.get('schema') != 'space-env.scale-result/1'
        or manifest.get('sha256') != digest(manifest.get('manifest'))
        or manifest['sha256'] != scale.get('manifest_sha256')):
        raise ValueError('study schema/manifest checksum mismatch')
    for record in [result, scale]:
        if record.get('sha256') != digest({k:v for k,v in record.items() if k != 'sha256'}):
            raise ValueError('study result checksum mismatch')
    if result['spec_sha256'] != digest(spec) or result['scale_sha256'] != scale['sha256']:
        raise ValueError('study artifact binding mismatch')
    assessment = analyze_study(spec, scale)
    if assessment != result['assessment']:
        raise ValueError('study assessment recomputation mismatch')
    return {'passed': True, 'qualification_passed': assessment['passed'],
            'current_build_matches': build_identity() == result['build'],
            'scope': 'offline integrity/recomputation only; no new mission replay or authentication'}
