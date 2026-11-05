"""Sealed, fixed-input mission schedules over the existing native TaskSession.

Plans are proposals, never admission or physics oracles. A recorded proposal
does not certify its source policy; feasibility is evaluated by fresh execution.
"""
from __future__ import annotations

import copy
import os
from threading import RLock
from typing import Literal

from pydantic import Field, model_serializer, model_validator

from space_env.planning.demand_contract import DemandGoals, validate_demand_schedule
from space_env.runtime.schema import ScenarioSpec, StrictModel
from space_env.scaling.coupled import CoupledGates
from space_env.scaling.restart import build_identity
from space_env.simulation.snapshot import atomic_write, canonical_json, digest
from space_env.tasks import TaskSession, TaskSpec
from space_env.tasks.schema import TaskDecision

HASH = r'^[a-f0-9]{64}$'


class PlanGoals(StrictModel):
    min_fulfilled: int = Field(default=0, ge=0, le=10000, strict=True)
    max_fulfilled: int | None = Field(default=None, ge=0, le=10000, strict=True)
    max_rejections: int = Field(default=0, ge=0, le=10000, strict=True)
    require_no_failed_operations: bool = Field(default=False, strict=True)
    coupled: CoupledGates | None = None
    demand: DemandGoals | None = None

    @model_serializer(mode='wrap')
    def legacy_demand(self,handler):
        value=handler(self)
        if self.demand is None:value.pop('demand',None)
        return value

    @model_validator(mode='after')
    def bounds(self):
        if self.max_fulfilled is not None and self.max_fulfilled < self.min_fulfilled:
            raise ValueError('plan completion upper bound below minimum')
        return self


class PlanStep(StrictModel):
    at_sec: float = Field(ge=0, le=259200, strict=True)
    advance_sec: float = Field(gt=1e-9, le=3600, strict=True)
    action_ids: tuple[str, ...] = Field(min_length=1, max_length=64)
    expected_transition_sha256: str | None = Field(default=None, pattern=HASH)


class RecoveryWitness(StrictModel):
    source_checkpoint_sha256: str = Field(pattern=HASH)
    producer_pid: int = Field(gt=0,strict=True)
    consumer_pid: int = Field(gt=0,strict=True)
    fresh_process: bool = Field(strict=True)
    prefix_steps: int = Field(ge=0,le=10000,strict=True)
    method: Literal['native TaskSession audited prefix replay; reported PIDs']


class MissionPlan(StrictModel):
    schema_id: Literal['space-env.mission-plan/1'] = Field(alias='schema')
    scenario: dict
    task: dict
    scenario_sha256: str = Field(pattern=HASH)
    task_sha256: str = Field(pattern=HASH)
    build: dict
    episode_seed: int = Field(ge=0, lt=2**63, strict=True)
    horizon_sec: float = Field(gt=0, le=259200, strict=True)
    origin: Literal['declared', 'recorded-candidate']
    source_planning_sha256: str | None = Field(pattern=HASH)
    expected_native_sha256: str | None = Field(pattern=HASH)
    goals: PlanGoals
    steps: tuple[PlanStep, ...] = Field(min_length=1, max_length=10000)
    sha256: str = Field(pattern=HASH)


def _sealed(value):
    return isinstance(value, dict) and value.get('sha256') == digest(
        {k:v for k,v in value.items() if k != 'sha256'})


def validate_plan(value, *, check_build=True):
    value = copy.deepcopy(value)
    spec = MissionPlan.model_validate(value)
    normalized = spec.model_dump(mode='json', by_alias=True)
    if canonical_json(value) != canonical_json(normalized) or not _sealed(value):
        raise ValueError('mission plan must have a complete canonical sealed envelope')
    scenario = ScenarioSpec.model_validate(spec.scenario)
    task = TaskSpec.model_validate(spec.task)
    if (digest(spec.scenario) != spec.scenario_sha256 or digest(spec.task) != spec.task_sha256
            or not scenario.backend.startswith('reference-')
            or scenario.definition.get('horizon_sec') != spec.horizon_sec):
        raise ValueError('mission plan scenario/task/horizon binding mismatch')
    if spec.origin == 'recorded-candidate' and spec.source_planning_sha256 is None:
        raise ValueError('recorded candidate requires its source checksum')
    if set(spec.build) != {'source_data_sha256','package_version','python_version','dependencies'}:
        raise ValueError('mission plan build envelope mismatch')
    if check_build and spec.build != build_identity():
        raise ValueError('mission plan requires matching source/data/Python/dependency build')
    if len(spec.steps) > task.max_decisions:
        raise ValueError('mission plan decision budget exceeded')
    known = {'wait', *[a.id for a in task.actions]}
    now = 0.
    for i, step in enumerate(spec.steps):
        ids = step.action_ids
        if (abs(step.at_sec-now) > 1e-8 or now >= spec.horizon_sec-1e-9
                or step.advance_sec > task.max_decision_sec
                or len(ids) != len(set(ids)) or set(ids)-known
                or len(ids) > task.max_joint_actions
                or ('wait' in ids and len(ids) != 1)):
            raise ValueError('mission plan time grid/action catalogue mismatch at step '+str(i))
        now = min(spec.horizon_sec, now+step.advance_sec)
    if abs(now-spec.horizon_sec) > 1e-8:
        raise ValueError('mission plan must explicitly cover its full horizon')
    if spec.goals.coupled is not None and (
        scenario.backend != 'reference-spacecraft'
        or digest(scenario.definition) != spec.goals.coupled.native_spec_sha256
    ):
        raise ValueError('mission plan coupled goals/scenario mismatch')
    if spec.goals.demand is not None:
        validate_demand_schedule(spec.scenario, task, spec.steps, spec.goals.demand)
    return spec


def make_mission_plan(scenario, task, steps, *, goals=None, episode_seed=None,
                      origin='declared', source_planning_sha256=None,
                      expected_native_sha256=None):
    scenario = copy.deepcopy(scenario)
    task = TaskSpec.model_validate(copy.deepcopy(task)).model_dump(mode='json', by_alias=True)
    body = {'schema':'space-env.mission-plan/1','scenario':scenario,'task':task,
        'scenario_sha256':digest(scenario),'task_sha256':digest(task),'build':build_identity(),
        'episode_seed':scenario.get('seed',0) if episode_seed is None else episode_seed,
        'horizon_sec':float(scenario['definition']['horizon_sec']),'origin':origin,
        'source_planning_sha256':source_planning_sha256,'expected_native_sha256':expected_native_sha256,
        'goals':PlanGoals.model_validate(goals or {}).model_dump(mode='json'),
        'steps':[PlanStep.model_validate(s).model_dump(mode='json') for s in steps]}
    result = body | {'sha256':digest(body)}
    validate_plan(result)
    return copy.deepcopy(result)


def recorded_mission_plan(record, *, goals=None):
    """Compile an unverified recorded proposal; do not promote it to a teacher.

    No cold policy replay is implied here. Independent plan execution and bundle
    native replay subsequently check transitions, final state and declared goals.
    """
    record = copy.deepcopy(record)
    if not _sealed(record) or record.get('schema') not in {'space-env.planning-run/1','space-env.planning-run/2'}:
        raise ValueError('invalid source planning envelope')
    episode = record['episode']
    if not _sealed(episode) or not _sealed(episode['runtime']):
        raise ValueError('invalid source episode/runtime envelope')
    if len(record['trace']) != len(episode['transitions']):
        raise ValueError('source proposal trace length mismatch')
    steps, now = [], 0.
    horizon = episode['scenario']['definition']['horizon_sec']
    for row in episode['transitions']:
        request = row['request']
        if set(request['choices']) != {'@central'}:
            raise ValueError('recorded plan requires central joint requests')
        dt = request['advance_sec']
        if dt is None:dt = episode['task']['decision_sec']
        steps.append({'at_sec':now,'advance_sec':dt,'action_ids':request['choices']['@central'],
            'expected_transition_sha256':row.get('transition_sha256') or digest(row['transition'])})
        now = min(horizon, now+dt)
    return make_mission_plan(episode['scenario'],episode['task'],steps,goals=goals,
        episode_seed=episode['episode_seed'],origin='recorded-candidate',
        source_planning_sha256=record['sha256'],expected_native_sha256=episode['runtime']['sha256'])


def check_episode_plan(spec, episode):
    """Validate request-to-schedule binding, including partial failed executions."""
    if (set(episode)!={'schema','scenario','task','episode_seed','transitions','runtime','sha256'}
            or type(episode['episode_seed']) is not int or not isinstance(episode['transitions'],list)
            or canonical_json(episode['scenario']) != canonical_json(spec.scenario)
            or canonical_json(episode['task']) != canonical_json(spec.task)
            or episode['episode_seed'] != spec.episode_seed
            or len(episode['transitions']) > len(spec.steps)):
        raise ValueError('execution episode differs from mission plan')
    for i, row in enumerate(episode['transitions']):
        step, request = spec.steps[i], row['request']
        TaskDecision.model_validate(request)
        if canonical_json(request) != canonical_json({'choices':{'@central':list(step.action_ids)},'expected_revision':i,
                       'idempotency_key':f'mission-plan/{i}','advance_sec':step.advance_sec}):
            raise ValueError('execution request differs from mission plan step')


class PlanExecution:
    """Single-owner fixed schedule executor; it never changes native state directly."""

    def __init__(self, plan):
        self._spec = validate_plan(plan)
        self._lock = RLock()
        self._session = TaskSession(self._spec.scenario,self._spec.task,seed=self._spec.episode_seed)
        self._error = None
        self._recovery = None

    @property
    def plan(self):
        return self._spec.model_dump(mode='json',by_alias=True)

    @property
    def cursor(self):
        return self._session.revision

    @property
    def complete(self):
        return self.cursor == len(self._spec.steps) and self._error is None

    def step(self):
        with self._lock:
            if self._error is not None or self.cursor >= len(self._spec.steps):
                raise ValueError('mission plan execution is stopped or complete')
            i = self.cursor
            step = self._spec.steps[i]
            if abs(self._session._runtime.time_sec-step.at_sec) > 1e-8:
                raise ValueError('mission plan execution clock mismatch')
            transition = self._session.step({'@central':list(step.action_ids)},
                expected_revision=i,idempotency_key=f'mission-plan/{i}',advance_sec=step.advance_sec)
            if step.expected_transition_sha256 is not None and digest(transition) != step.expected_transition_sha256:
                self._error = {'code':'expected_transition_mismatch','index':i}
            return transition

    def run(self, *, decisions=None):
        if decisions is not None and (type(decisions) is not int or decisions < 0):
            raise ValueError('invalid mission plan execution budget')
        count = 0
        while not self.complete and self._error is None and (decisions is None or count < decisions):
            try:self.step()
            except ValueError as exc:
                self._error = {'code':'execution_error','index':self.cursor,'message':str(exc)[:1024]}
                break
            count += 1
        return {'complete':self.complete,'steps':self.cursor,'time_sec':self._session._runtime.time_sec,
                'error':copy.deepcopy(self._error)}

    def checkpoint(self):
        with self._lock:
            episode = self._session.checkpoint(compact=True)
            check_episode_plan(self._spec,episode)
            body = {'schema':'space-env.mission-plan-execution/1','plan_sha256':self._spec.sha256,
                    'build':self._spec.build,'producer_pid':os.getpid(),'episode':episode,
                    'error':copy.deepcopy(self._error),'recovery':copy.deepcopy(self._recovery)}
            return copy.deepcopy(body | {'sha256':digest(body)})

    def save(self,path):
        result = self.checkpoint();atomic_write(path,canonical_json(result));return result

    @classmethod
    def from_checkpoint(cls,plan,record):
        spec = validate_plan(plan)
        record = copy.deepcopy(record)
        validate_execution(spec,record)
        result = object.__new__(cls)
        result._spec, result._lock = spec, RLock()
        result._session = TaskSession.from_checkpoint(record['episode'])
        result._error = record['error']
        result._recovery = {'source_checkpoint_sha256':record['sha256'],
            'producer_pid':record['producer_pid'],'consumer_pid':os.getpid(),
            'fresh_process':record['producer_pid'] != os.getpid(),
            'prefix_steps':result.cursor,'method':'native TaskSession audited prefix replay; reported PIDs'}
        return result


def validate_execution(spec,record):
    keys = {'schema','plan_sha256','build','producer_pid','episode','error','recovery','sha256'}
    if (not isinstance(record,dict) or set(record) != keys or not _sealed(record)
            or record['schema'] != 'space-env.mission-plan-execution/1'
            or record['plan_sha256'] != spec.sha256 or record['build'] != spec.build
            or type(record['producer_pid']) is not int or record['producer_pid'] <= 0
            or not _sealed(record['episode']) or not _sealed(record['episode']['runtime'])
            or record['episode'].get('schema') != 'space-env.task-episode/2'):
        raise ValueError('invalid mission plan execution envelope')
    check_episode_plan(spec,record['episode'])
    if record['recovery'] is not None:
        recovery=RecoveryWitness.model_validate(record['recovery'])
        if (recovery.fresh_process is not (recovery.producer_pid!=recovery.consumer_pid)
                or recovery.prefix_steps>len(record['episode']['transitions'])
                or recovery.consumer_pid!=record['producer_pid']):
            raise ValueError('invalid mission recovery process witness')
    if record['error'] is not None and (not isinstance(record['error'],dict)
            or set(record['error'])-{'code','index','message'}
            or not isinstance(record['error'].get('code'),str)
            or type(record['error'].get('index')) is not int
            or not 0<=record['error']['index']<=len(spec.steps)):
        raise ValueError('invalid mission execution error witness')
