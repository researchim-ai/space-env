"""Independent-input coupled missions. Synthetic hardware, metadata products only."""
from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_validator, model_serializer

from space_env.reference.acquisition import AcquisitionTarget
from space_env.reference.frames import FrameContext
from space_env.reference.spacecraft_bus import bus_preset
from space_env.reference.time import EOPSeries
from space_env.runtime import spacecraft_scenario
from space_env.runtime.schema import StrictModel
from space_env.scaling.independent import FixedSite, OrbitPlane, circular_state
from space_env.simulation.snapshot import digest


class CoupledMissionSpec(StrictModel):
    id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,48}$')
    epoch_utc: str = '2026-07-18T12:00:00Z'
    planes: tuple[OrbitPlane, ...] = Field(min_length=2, max_length=4)
    targets: tuple[AcquisitionTarget, ...] = Field(min_length=1, max_length=16)
    stations: tuple[FixedSite, ...] = Field(min_length=1, max_length=16)
    horizon_sec: float = Field(default=180., ge=30, le=600, strict=True)
    decision_sec: float = Field(default=2., ge=1, le=5, strict=True)
    control_step_sec: float = Field(default=.1, ge=.025, le=.2, strict=True)
    rigid_step_sec: float = Field(default=.01, ge=.0025, le=.02, strict=True)
    lifecycle_step_sec: float = Field(default=.5, ge=.125, le=.5, strict=True)
    controller_clock: Literal['event-driven/1', 'fixed-tick/1'] = 'event-driven/1'
    mode: Literal['nominal', 'weak_wheels', 'all_outage'] = 'nominal'
    min_fulfilled: int = Field(default=0, ge=0, le=64, strict=True)
    min_acquired: int = Field(default=0, ge=0, le=64, strict=True)

    @model_validator(mode='after')
    def bounded(self):
        actors = sum(len(p.phases_deg) for p in self.planes)
        if actors > 4 or actors * len(self.targets) > 64:
            raise ValueError('coupled mission supports at most 4 actors and 64 requests')
        for rows, key in [(self.planes, 'id'), (self.targets, 'id'), (self.stations, 'name')]:
            if len({getattr(r, key) for r in rows}) != len(rows):
                raise ValueError('duplicate input IDs')
        if max(self.min_fulfilled, self.min_acquired) > actors * len(self.targets):
            raise ValueError('service gate exceeds request denominator')
        # Reuse the geometric distinct-plane contract, independent of site placement.
        from space_env.scaling.independent import IndependentFleetSpec
        IndependentFleetSpec(id=self.id, epoch_utc=self.epoch_utc, planes=self.planes,
                             stations=self.stations[:8], min_delivered=0)
        return self


def coupled_workload(value):
    cfg = CoupledMissionSpec.model_validate(value)
    return _compile_coupled(cfg)


def _compile_coupled(cfg):
    """Shared compiler for separately validated short and full-orbit profiles."""
    orbits, buses, requests, jobs, actions = {}, {}, [], [], []
    for plane in cfg.planes:
        for j, phase in enumerate(plane.phases_deg):
            actor = f'{plane.id}-{j:02d}'
            r, v = circular_state(plane, phase)
            orbits[actor] = {
                'initial': {'frame': 'GCRS', 'epoch_utc': cfg.epoch_utc,
                            'r_m': r.tolist(), 'v_m_s': v.tolist()},
                'propulsion': {'dry_mass_kg': 20., 'propellant_kg': 0.,
                               'isp_s': 200., 'max_thrust_n': 1.},
                'forces': {'j2': True},
            }
            bus = bus_preset('synthetic-agile/1')
            torque = 1e-6 if cfg.mode == 'weak_wheels' else .1
            bus['attitude']['rigid'] = {
                'model': {'model_id': 'declared-coupled-wheel/1',
                          'source': 'synthetic analytical assumptions; not flight hardware',
                          'inertia_kg_m2': [[2., 0., 0.], [0., 3., 0.], [0., 0., 4.]],
                          'wheel_inertia_kg_m2': [.01] * 3,
                          'max_torque_nm': [torque] * 3,
                          'max_momentum_nms': [.2] * 3,
                          **({'max_step_sec': cfg.rigid_step_sec} if cfg.rigid_step_sec != .01 else {})},
                'supply_w': 12., 'proportional_nm_per_rad': 10.,
                'damping_nm_per_rad_s': 3., 'control_step_sec': cfg.control_step_sec,
                'payload_max_body_rate_rad_s': .1,
                **({'clock_mode': cfg.controller_clock} if cfg.controller_clock != 'event-driven/1' else {}),
            }
            buses[actor] = bus
            for i, target in enumerate(cfg.targets):
                rid = f'{actor}-{target.id}'
                raw, final, job = 'raw-' + rid, 'final-' + rid, 'cpu-' + rid
                station = cfg.stations[i % len(cfg.stations)].name
                requests.append({'id': rid, 'target_id': target.id,
                                 'eligible_actors': [actor], 'deadline_sec': cfg.horizon_sec,
                                 'acquisition_sec': 2., 'raw_product_id': raw,
                                 'raw_size_bits': 1000000, 'delivery_product_id': final,
                                 'destination_id': station})
                jobs.append({'id': job, 'inputs': [raw],
                             'outputs': [{'id': final, 'size_bits': 500000}],
                             'work_sec': 1., 'workspace_bits': 100000, 'power_w': 5.})
                for kind, duration, fields in [
                    ('slew', 12., {'slew': {'frame': 'GCRS', 'target_id': target.id}}),
                    ('imaging', 2., {'request_id': rid}),
                    ('transform', 1., {'job_id': job}),
                    ('downlink', 2., {'product_id': final, 'destination_id': station}),
                ]:
                    key = kind + '-' + rid
                    actions.append({'id': key, 'role': actor, 'kind': 'submit',
                                    'commands': [{'command_id': key, 'actor_id': actor,
                                                  'kind': kind, 'duration_sec': duration, **fields}]})
    actors = sorted(orbits)
    native = {
        'schema': 'space-env.reference-dynamic-spec/1', 'orbits': orbits,
        'eop': FrameContext(EOPSeries.packaged()).manifest()['eop'],
        'horizon_sec': cfg.horizon_sec,
        'operations': {'schema': 'space-env.reference-downlink-spec/1',
                       'vehicles': {a: {'memory_bits': 100000000., 'battery_wh': 100.,
                                       'base_power_w': 1., 'tx_power_w': 20.,
                                       'rate_bps': 2000000., 'solar_mode': 'body_fixed_bus'}
                                    for a in actors},
                       'stations': [s.model_dump(mode='json') for s in cfg.stations],
                       'products': [],
                       'outages': [{'station': s.name, 'start_sec': 0., 'end_sec': cfg.horizon_sec}
                                   for s in cfg.stations] if cfg.mode == 'all_outage' else []},
        'acquisition_targets': [t.model_dump(mode='json') for t in cfg.targets],
        'acquisition_sensors': {a: {'max_off_nadir_deg': 75., 'min_elevation_deg': 5.}
                                for a in actors},
        'spacecraft_bus': buses,
        'execution': {'actors': {a: {'memory_bandwidth_bps': 100000000.} for a in actors}},
        'lifecycle': {'requests': requests, 'jobs': jobs,
                      'payloads': {a: {'power_w': 10.} for a in actors},
                      **({'execution_step_sec': cfg.lifecycle_step_sec}
                         if cfg.lifecycle_step_sec != .5 else {})},
    }
    task = {
        'id': cfg.id, 'pointing_masks': True, 'runtime_audit': 'compact',
        'roles': {a: {'actors': [a]} for a in actors}, 'actions': actions,
        'decision_sec': cfg.decision_sec, 'max_decision_sec': cfg.decision_sec,
        'max_decisions': math.ceil(cfg.horizon_sec / cfg.decision_sec) + 1,
        'max_actions': len(actions) + 1, 'max_joint_actions': len(actors),
        'max_entities': 1024, 'max_relations': 8192,
    }
    return {'id': cfg.id, 'family': 'independent-coupled-metadata/1',
            'source': 'fixed independent inputs; synthetic ADCS, metadata acquisition, ideal geometric radio',
            'scenario': spacecraft_scenario(native, id=cfg.id, lineage='D2b-coupled'),
            'task': task, 'policy': {'name': 'route_aware_rolling'}, 'timeout_sec': 600,
            'min_completion_fraction': cfg.min_fulfilled / len(requests),
            'coupled_gates': {'native_spec_sha256': digest(native),
                              'min_acquired': cfg.min_acquired},
            **({'max_completion_fraction': 0.} if cfg.mode == 'all_outage' else {})}


class CoupledGates(StrictModel):
    native_spec_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    min_acquired: int = Field(default=0, ge=0, le=64, strict=True)
    delivery_epoch_sec: float | None = Field(default=None, gt=0, le=21600, strict=True)
    min_delivery_epochs: int | None = Field(default=None, ge=1, le=3, strict=True)

    @model_validator(mode='after')
    def epochs(self):
        if (self.delivery_epoch_sec is None) != (self.min_delivery_epochs is None):
            raise ValueError('delivery epoch gate requires period and minimum together')
        return self

    @model_serializer(mode='wrap')
    def legacy(self, handler):
        value = handler(self)
        if self.delivery_epoch_sec is None:
            value.pop('delivery_epoch_sec', None)
            value.pop('min_delivery_epochs', None)
        return value


def coupled_evidence(scenario, report, gates):
    """Evaluator-only lineage/conservation witnesses, never policy observations."""
    gates = CoupledGates.model_validate(gates)
    native = scenario['definition']
    if scenario['backend'] != 'reference-spacecraft' or digest(native) != gates.native_spec_sha256:
        raise ValueError('coupled witness scenario binding mismatch')
    expected = {r['id']: r for r in native['lifecycle']['requests']}
    life = report['lifecycle']
    artifacts = life['artifacts']
    acquired = [a for a in artifacts.values() if a['source'] == 'acquired']
    transformed = [a for a in artifacts.values() if a['source'] == 'transformed_out']
    checks = {
        'no_preload': not native['operations']['products'],
        'request_denominator': set(life['requests']) == set(expected)
            and life['metrics']['total_requests'] == len(expected),
        'minimum_acquired': len(acquired) >= gates.min_acquired,
        'actor_denominator': all(set(rows) == set(native['orbits']) for rows in [
            report['energy_ledger'], life['ledger'], report['spacecraft_bus']['states']]),
    }
    lineage = True
    for rid, original in expected.items():
        state = life['requests'].get(rid, {})
        lineage &= all(state.get(k) == original[k] for k in [
            'target_id', 'eligible_actors', 'raw_product_id', 'delivery_product_id', 'destination_id'])
        if state.get('status') == 'completed':
            raw = artifacts.get(original['raw_product_id'], {})
            final = artifacts.get(original['delivery_product_id'], {})
            lineage &= (raw.get('source') == 'acquired'
                        and final.get('source') == 'transformed_out'
                        and rid in raw.get('request_ids', [])
                        and rid in final.get('request_ids', [])
                        and raw.get('actor') in original['eligible_actors']
                        and raw.get('actor') == final.get('actor')
                        and final.get('status') == 'delivered')
    epoch_evidence = None
    if gates.delivery_epoch_sec is not None:
        epoch = gates.delivery_epoch_sec
        if epoch * gates.min_delivery_epochs > native['horizon_sec'] + 1e-9:
            raise ValueError('delivery epoch gate exceeds mission horizon')
        served = set(); epoch_rows = []
        for rid, original in expected.items():
            state = life['requests'].get(rid, {})
            raw = artifacts.get(original['raw_product_id'], {})
            release_epoch = math.floor(original.get('release_sec', 0.) / epoch)
            if (state.get('status') == 'completed'
                    and math.floor(raw.get('created_sec', -1.) / epoch) == release_epoch
                    and math.floor(state.get('delivered_sec', -1.) / epoch) == release_epoch):
                served.add(release_epoch)
                epoch_rows.append({'request_id':rid,'epoch':release_epoch,
                    'actor':raw.get('actor'),'acquired_sec':raw['created_sec'],
                    'delivered_sec':state['delivered_sec']})
        checks['minimum_delivery_epochs'] = len(served) >= gates.min_delivery_epochs
        epoch_evidence={'period_sec':epoch,'served_epochs':sorted(served),'deliveries':epoch_rows}
    jobs = {j['id']: j for j in native['lifecycle']['jobs']}
    for artifact in artifacts.values():
        ids = artifact.get('request_ids', [])
        lineage &= bool(ids) and all(rid in expected for rid in ids)
        for rid in ids:
            if rid not in expected:
                continue
            request = expected[rid]
            lineage &= artifact['actor'] in request['eligible_actors']
            if artifact['source'] == 'acquired':
                acquisition = artifact.get('acquisition', {})
                lineage &= (artifact['id'] == request['raw_product_id']
                            and artifact['size_bits'] == request['raw_size_bits']
                            and acquisition.get('target_id') == request['target_id']
                            and acquisition.get('validated_dwell_sec', 0) >= request['acquisition_sec'] - 1e-6)
            elif artifact['source'] == 'transformed_out':
                job = jobs.get(artifact.get('job_id'), {})
                raw = artifacts.get(request['raw_product_id'], {})
                lineage &= (artifact['id'] == request['delivery_product_id']
                            and request['raw_product_id'] in job.get('inputs', [])
                            and raw.get('created_sec', math.inf) <= artifact['created_sec']
                            and life['jobs'].get(artifact.get('job_id'), {}).get('status') == 'completed')
            else:
                lineage = False
    checks['completed_lineage'] = bool(lineage)
    checks['fulfilled_count'] = life['metrics']['fulfilled_requests'] == sum(
        s['status'] == 'completed' for s in life['requests'].values())
    def close(x, tolerance):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and abs(x) <= tolerance
    checks['balances'] = all(
        close(row['balance_error_wh'], 1e-7) for row in report['energy_ledger'].values()
    ) and all(close(row['balance_error_bits'], 1e-3) for row in life['ledger'].values())
    rigid = [bool(model['attitude'].get('rigid')) for model in native['spacecraft_bus'].values()]
    if any(rigid) and not all(rigid):
        raise ValueError('coupled witness requires a uniform declared attitude tier')
    if not any(rigid):
        checks['kinematic_paid_slew'] = all(
            s['slew_sec'] >= 0 and s['slew_energy_wh'] >= 0
            and close(s['slew_energy_wh'] * 3600. - s['slew_sec'] *
                      native['spacecraft_bus'][actor]['attitude']['slew_power_w'], 1e-6)
            for actor,s in report['spacecraft_bus']['states'].items())
    else:
        checks['rigid_paid_conservation'] = all(
            s['slew_sec'] >= 0 and s['slew_energy_wh'] >= 0
            and close(s['rigid_attitude']['dc_energy_j'] - s['slew_energy_wh'] * 3600., 1e-6)
            and close(s['rigid_attitude']['dc_energy_j'] - s['slew_sec'] *
                      native['spacecraft_bus'][actor]['attitude']['rigid']['supply_w'], 1e-6)
            and close(s['rigid_attitude']['dc_heat_work_balance_error_j'], 1e-6)
            and close(s['rigid_attitude']['kinetic_work_balance_error_j'], 1e-6)
            and all(close(v, 1e-6) for v in s['rigid_attitude']['angular_momentum_balance_error_gcrs_nms'])
            for actor, s in report['spacecraft_bus']['states'].items()
        )
    return {'passed': all(checks.values()), 'checks': checks,
            'total_requests': len(expected), 'acquired': len(acquired),
            'transformed': len(transformed), 'fulfilled': life['metrics']['fulfilled_requests'],
            'failed_control_windows': sum(o['status'] == 'failed' for o in report['spacecraft_bus']['operations'].values()),
            **({'delivery_epochs':epoch_evidence} if epoch_evidence is not None else {}),
            'scope': 'native model lineage/conservation; not independent physical calibration'}
