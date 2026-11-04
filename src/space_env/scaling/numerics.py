"""Evaluator-only state sampling and conditional closed-loop self-refinement.

Changing controller or event-check cadence changes the discrete model. Those
comparisons are sensitivity evidence, not an accuracy or convergence proof.
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from space_env.runtime.schema import FiniteScalar, StrictModel
from space_env.simulation.snapshot import digest


class ActorSample(StrictModel):
    state_si: tuple[FiniteScalar, FiniteScalar, FiniteScalar, FiniteScalar, FiniteScalar, FiniteScalar]
    quaternion_wxyz: tuple[FiniteScalar, FiniteScalar, FiniteScalar, FiniteScalar]
    omega_body_rad_s: tuple[FiniteScalar, FiniteScalar, FiniteScalar]
    wheel_momentum_nms: tuple[FiniteScalar, FiniteScalar, FiniteScalar]
    battery_wh: float = Field(ge=0, strict=True)
    temperature_k: float = Field(gt=0, strict=True)
    dc_energy_j: float = Field(ge=0, strict=True)

    @model_validator(mode='after')
    def unit(self):
        if not math.isclose(math.hypot(*self.quaternion_wxyz), 1., abs_tol=1e-8, rel_tol=0):
            raise ValueError('numerical sample quaternion must be unit')
        return self


class ControlSample(StrictModel):
    action_ids: tuple[str, ...] = Field(min_length=1, max_length=64)
    advance_sec: float = Field(gt=0, strict=True)
    accepted: bool = Field(strict=True)


class RequestSample(StrictModel):
    status: str = Field(min_length=1)
    delivered_sec: float | None = Field(default=None, ge=0, strict=True)


class NumericalSample(StrictModel):
    time_sec: float = Field(ge=0, strict=True)
    actors: dict[str, ActorSample] = Field(min_length=1, max_length=4)
    requests: dict[str, RequestSample] = Field(max_length=64)
    control: ControlSample | None


class NumericalLimits(StrictModel):
    position_m: float = Field(default=1., gt=0, strict=True)
    velocity_m_s: float = Field(default=.01, gt=0, strict=True)
    attitude_deg: float = Field(default=.1, gt=0, strict=True)
    body_rate_rad_s: float = Field(default=.001, gt=0, strict=True)
    wheel_momentum_nms: float = Field(default=.005, gt=0, strict=True)
    battery_wh: float = Field(default=.01, gt=0, strict=True)
    temperature_k: float = Field(default=.05, gt=0, strict=True)
    dc_energy_j: float = Field(default=10., gt=0, strict=True)
    delivery_time_sec: float = Field(default=.5, gt=0, strict=True)


def numerical_snapshot(run):
    """Owned privileged diagnostics; never appended to actor/public features."""
    backend = run.session._runtime._backend
    t = float(backend.now())
    actors = {}
    for actor, bus in backend.bus.states.items():
        rigid = backend.bus.rigid.states[actor]
        actors[actor] = {
            'state_si': backend.mission.ephemerides[actor].state(t).tolist(),
            'quaternion_wxyz': list(rigid.quaternion_wxyz),
            'omega_body_rad_s': list(rigid.omega_body_rad_s),
            'wheel_momentum_nms': list(rigid.wheel_momentum_nms),
            'battery_wh': float(backend.mission.energy_wh[actor]),
            'temperature_k': float(bus['temperature_k']),
            'dc_energy_j': float(rigid.dc_energy_j),
        }
    last = run.trace[-1] if run.trace else None
    return NumericalSample.model_validate({
        'time_sec': t, 'actors': actors,
        'requests': {rid: {'status': state['status'], 'delivered_sec': state.get('delivered_sec')}
                     for rid, state in backend.lifecycle.request_state.items()},
        'control': None if last is None else {
            'action_ids': sorted(last['decision']['action_ids']),
            'advance_sec': float(last['decision']['advance_sec']),
            'accepted': last['feedback']['accepted']},
    }).model_dump(mode='json')


def _series(envelope):
    if not isinstance(envelope, dict) or set(envelope) != {'samples', 'sha256', 'scope'}:
        raise ValueError('invalid numerical sample envelope')
    raw = envelope['samples']
    if not isinstance(raw, list) or not 2 <= len(raw) <= 602 or digest(raw) != envelope['sha256']:
        raise ValueError('numerical sample checksum/size mismatch')
    samples = [NumericalSample.model_validate(s) for s in raw]
    if any(a.time_sec >= b.time_sec for a, b in zip(samples, samples[1:])):
        raise ValueError('numerical sample grid must strictly increase')
    if samples[0].time_sec == 0. and samples[0].control is not None:
        raise ValueError('initial sample cannot contain a completed control')
    for a, b in zip(samples, samples[1:]):
        if b.control is None or not math.isclose(b.control.advance_sec, b.time_sec-a.time_sec, rel_tol=0, abs_tol=1e-8):
            raise ValueError('numerical grid/control interval mismatch')
    if any(set(s.actors) != set(samples[0].actors) or set(s.requests) != set(samples[0].requests) for s in samples):
        raise ValueError('numerical sample identities changed')
    return samples


def _qdistance(left, right):
    a, b = np.asarray(left), np.asarray(right)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    lo, hi = sorted([float(np.linalg.norm(a - b)), float(np.linalg.norm(a + b))])
    return math.degrees(4 * math.atan2(lo, hi))


def compare_numerics(left, right, limits=None, *, require_same_schedule=True):
    if type(require_same_schedule) is not bool:
        raise TypeError("schedule requirement must be boolean")
    limits = NumericalLimits.model_validate({} if limits is None else limits)
    a, b = _series(left), _series(right)
    if ([s.time_sec for s in a] != [s.time_sec for s in b]
        or a[0].time_sec != 0. or set(a[0].actors) != set(b[0].actors)
        or set(a[0].requests) != set(b[0].requests)):
        return {'passed': False, 'reason': 'incomplete_or_mismatched_grid',
                'coverage_sec': [[a[0].time_sec, a[-1].time_sec], [b[0].time_sec, b[-1].time_sec]]}
    errors = {k: [] for k in type(limits).model_fields if k != 'delivery_time_sec'}
    same_schedule = all(x.control == y.control for x, y in zip(a, b))
    for x, y in zip(a, b):
        for actor in x.actors:
            u, v = x.actors[actor], y.actors[actor]
            errors['position_m'].append(math.dist(u.state_si[:3], v.state_si[:3]))
            errors['velocity_m_s'].append(math.dist(u.state_si[3:], v.state_si[3:]))
            errors['attitude_deg'].append(_qdistance(u.quaternion_wxyz, v.quaternion_wxyz))
            errors['body_rate_rad_s'].append(math.dist(u.omega_body_rad_s, v.omega_body_rad_s))
            errors['wheel_momentum_nms'].append(math.dist(u.wheel_momentum_nms, v.wheel_momentum_nms))
            for field in ['battery_wh', 'temperature_k', 'dc_energy_j']:
                errors[field].append(abs(getattr(u, field) - getattr(v, field)))
    maximum = {k: max(v) for k, v in errors.items()}
    rms = {k: math.sqrt(sum(x*x for x in v) / len(v)) for k, v in errors.items()}
    outcomes = all(x.status == b[-1].requests[rid].status for rid, x in a[-1].requests.items())
    timing = []
    for rid, x in a[-1].requests.items():
        y = b[-1].requests[rid]
        if (x.delivered_sec is None) != (y.delivered_sec is None):
            outcomes = False
        elif x.delivered_sec is not None:
            timing.append(abs(x.delivered_sec - y.delivered_sec))
    maximum['delivery_time_sec'] = max(timing, default=0.)
    gates = {k: v <= getattr(limits, k) for k, v in maximum.items()}
    gates['same_outcomes'] = outcomes
    gates['same_schedule'] = same_schedule or not require_same_schedule
    return {'passed': all(gates.values()), 'gates': gates, 'same_schedule': same_schedule,
            'max_errors': maximum, 'rms_errors': rms, 'samples': len(a),
            'actor_samples': len(a) * len(a[0].actors),
            'coverage_sec': [a[0].time_sec, a[-1].time_sec],
            'scope': 'empirical common-grid closed-loop differences; not independent physical truth'}


def refinement_triplet(coarse, middle, fine, limits=None, *, max_ratio=1.25):
    if isinstance(max_ratio, bool) or not math.isfinite(max_ratio) or not 1 <= max_ratio <= 2:
        raise ValueError('invalid self-refinement ratio')
    limits = NumericalLimits.model_validate({} if limits is None else limits)
    first = compare_numerics(coarse, middle, limits)
    second = compare_numerics(middle, fine, limits)
    shrinking = {}
    if 'max_errors' in first and 'max_errors' in second:
        shrinking = {k: (second['max_errors'][k] <= first['max_errors'][k] * max_ratio
                          or second['max_errors'][k] <= getattr(limits, k) * .01)
                     for k in type(limits).model_fields}
    return {'passed': first['passed'] and second['passed'] and bool(shrinking) and all(shrinking.values()),
            'coarse_middle': first, 'middle_fine': second, 'shrinking_or_below_floor': shrinking,
            'max_ratio': max_ratio, 'absolute_floor_fraction_of_tolerance': .01,
            'scope': 'conditional self-refinement with identical discrete schedule; no claimed order or continuum error bound'}
