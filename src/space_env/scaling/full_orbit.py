"""Bounded multi-cycle metadata missions with explicitly selected ADCS fidelity.

No geometry search, target relocation or relaxation of service floors is done.
The diagnostic is evaluator-only and is never passed to the baseline policy.
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from space_env.core.constants import MU_EARTH
from space_env.runtime import AccessPolicy, MissionRuntime, spacecraft_scenario
from space_env.scaling.coupled import CoupledMissionSpec, _compile_coupled
from space_env.simulation.snapshot import digest


class FullOrbitSpec(CoupledMissionSpec):
    horizon_sec: float = Field(default=12600., ge=12000, le=21600, strict=True)
    decision_sec: float = Field(default=10., ge=1, le=30, strict=True)
    lifecycle_step_sec: float = Field(default=2., ge=.125, le=5, strict=True)
    cycles: int = Field(default=2, ge=2, le=3, strict=True)
    attitude_model: Literal['rate-limited-kinematic/1','fixed-tick-rigid/1'] = 'rate-limited-kinematic/1'
    rigid_step_sec: float = Field(default=.02, ge=.01, le=.02, strict=True)
    controller_clock: Literal['fixed-tick/1'] = 'fixed-tick/1'
    min_fulfilled: int = Field(default=2, ge=0, le=64, strict=True)
    min_acquired: int = Field(default=2, ge=0, le=64, strict=True)
    min_delivery_epochs: int = Field(default=2, ge=0, le=3, strict=True)

    @model_validator(mode='after')
    def full_cycle_bounds(self):
        actors=sum(len(p.phases_deg) for p in self.planes)
        if actors*len(self.targets)*self.cycles>64:
            raise ValueError('full-orbit expanded request budget exceeds 64')
        longest=max(2*math.pi*math.sqrt(p.radius_m**3/MU_EARTH) for p in self.planes)
        if self.horizon_sec < self.cycles*longest*1.025:
            raise ValueError('each cycle requires 1.025 nominal Kepler periods')
        if math.ceil(self.horizon_sec/self.decision_sec)+1>10000:
            raise ValueError('full-orbit decision budget exceeds 10000')
        if self.min_delivery_epochs>self.cycles:
            raise ValueError('delivery epoch minimum exceeds cycles')
        if self.attitude_model=='fixed-tick-rigid/1' and self.horizon_sec/self.rigid_step_sec>900000:
            raise ValueError('rigid full-orbit nominal substep budget exceeds reserved limit')
        if self.mode=='all_outage' and (self.min_fulfilled or self.min_delivery_epochs):
            raise ValueError('outage negative control requires explicit zero delivery floors')
        return self


def full_orbit_workload(value):
    cfg=FullOrbitSpec.model_validate(value)
    width=cfg.horizon_sec/cfg.cycles
    targets=[]
    for cycle in range(cfg.cycles):
        for target in cfg.targets:
            start=max(cycle*width,target.available_from_sec)
            end=min((cycle+1)*width,target.available_until_sec or cfg.horizon_sec)
            if start>=end:
                raise ValueError('target availability has an empty cycle intersection')
            targets.append(type(target).model_validate(target.model_dump(mode='json') | {
                'id':f'{target.id}-c{cycle}','available_from_sec':start,'available_until_sec':end}))
    # Derived targets are independently validated above and bounded to <=64
    # requests. The short-profile 600s validator is deliberately not bypassed by
    # callers: this compiler has its own strict FullOrbitSpec.
    workload=_compile_coupled(cfg.model_copy(update={'targets':tuple(targets)}))
    native=workload['scenario']['definition']
    destinations={t.id:cfg.stations[i%len(cfg.stations)].name for i,t in enumerate(cfg.targets)}
    products={}
    for request in native['lifecycle']['requests']:
        cycle=int(request['target_id'].rsplit('-c',1)[1])
        request['release_sec']=cycle*width
        request['deadline_sec']=(cycle+1)*width
        request['destination_id']=destinations[request['target_id'].rsplit('-c',1)[0]]
        products[request['delivery_product_id']]=request['destination_id']
    for action in workload['task']['actions']:
        command=action['commands'][0]
        if command['kind']=='downlink':
            command['destination_id']=products[command['product_id']]
    for model in native['spacecraft_bus'].values():
        if cfg.attitude_model=='rate-limited-kinematic/1':
            model['attitude'].pop('rigid',None)
        else:
            model['attitude']['rigid']['max_total_substeps']=1_000_000
    workload['scenario']=spacecraft_scenario(native,id=cfg.id,lineage='D2f-full-orbit')
    workload.update(family='full-orbit-coupled-metadata/1',timeout_sec=3600,
                    source=f'fixed independent sites; {cfg.cycles} release epochs; declared {cfg.attitude_model}; synthetic metadata hardware')
    workload['coupled_gates']['native_spec_sha256']=digest(native)
    if cfg.min_delivery_epochs:
        workload['coupled_gates'].update(delivery_epoch_sec=width,min_delivery_epochs=cfg.min_delivery_epochs)
    return workload


def full_orbit_geometry(value):
    """Measure trajectory winding and all pair contacts; not a service certificate.

    Phase is projected onto the initial osculating plane and unwrapped on a
    <=60s grid. J2 precession is retained. Nominal Kepler period only determines
    the requested horizon; measured winding is a separate diagnostic.
    """
    cfg=FullOrbitSpec.model_validate(value)
    workload=full_orbit_workload(cfg)
    m=MissionRuntime(workload['scenario'],access=AccessPolicy(truth=True))
    grid=np.linspace(0.,cfg.horizon_sec,math.ceil(cfg.horizon_sec/60.)+1)
    actors={};contacts=[]
    for actor,provider in m._backend.mission.ephemerides.items():
        initial=provider.state(0.);x=initial[:3]/np.linalg.norm(initial[:3])
        normal=np.cross(initial[:3],initial[3:]);normal/=np.linalg.norm(normal)
        y=np.cross(normal,x)
        positions=np.array([provider.state(float(t))[:3] for t in grid])
        phase=np.unwrap(np.arctan2(positions@y,positions@x))
        winding=float((phase[-1]-phase[0])/(2*math.pi))
        actors[actor]={'projected_revolutions':winding,'at_least_requested_cycles':winding>=cfg.cycles}
        for station in m._backend.mission.stations:
            windows=m._backend.mission.windows[actor,station]
            contacts.append({'actor':actor,'station':station,'windows_sec':[list(w) for w in windows],
                             'contact_sec':sum(b-a for a,b in windows)})
    body={'schema':'space-env.full-orbit-geometry/1','scenario_sha256':digest(workload['scenario']),
          'horizon_sec':cfg.horizon_sec,'cycles':cfg.cycles,'attitude_model':cfg.attitude_model,
          'actors':actors,'contacts':contacts,'sample_sec_max':60.,
          'passed':all(a['at_least_requested_cycles'] for a in actors.values()),
          'scope':'unforced initial trajectories and geometric contacts, including zero pairs; not executable service feasibility or an RF link model'}
    return body|{'sha256':digest(body)}
