"""Focused read-only audit cases for 0.86. Run from repository root.

Produces JSON observations, not a passing regression suite or physics qualification.
Synthetic fixtures keep the cases independent of the 72-hour scenario runtime.
"""
import json
import tempfile
from pathlib import Path
import numpy as np
import yaml
from space_env import __version__
from space_env.core.orbit import Orbit
from space_env.core.time import Time
from space_env.core.spacecraft import Spacecraft, SpacecraftConfig, PropulsionConfig
from space_env.mission.events import EventKind, PlannedEvent
from space_env.mission.scenarios import ScenarioConfig
from space_env.mission.scenario_v2 import load_scenario_v2
from space_env.mission.tasking import TaskRequest, StationOutage, delivery_score, mission_kpis, run_policy
from space_env.core.access import subsatellite_latlon
from space_env.core.ground import GroundStation
from space_env.simulation.engine import SimulationEngine
from space_env.simulation.world import MissionWorld

EPOCH = '2026-10-20T10:00:00Z'

def engine(name='a', planned=None):
    orbit=Orbit.from_keplerian(6800000.,0.,.9,0.,0.,0.,Time(EPOCH))
    craft=Spacecraft(config=SpacecraftConfig(name=name,propulsion=PropulsionConfig(delta_v_total_m_s=20.)))
    craft.reset_state()
    return SimulationEngine(orbit,craft,scenario=ScenarioConfig(ground_stations=[],planned_events=list(planned or [])))

def world(engines=None, horizon=60.):
    return MissionWorld(engines or {'a':engine()},epoch=EPOCH,horizon_sec=horizon,seed=7)

def main():
    out={'version':__version__}
    w=world({'a':engine(planned=[PlannedEvent(kind=EventKind.MANEUVER,scheduled_at_sec=.5,payload={'dv_m_s':1.,'direction':'prograde'})])})
    before=w.engines['a'].spacecraft.state.propulsion.delta_v_remaining_m_s
    w.forecast('a',1.)
    after=w.engines['a'].spacecraft.state.propulsion.delta_v_remaining_m_s
    restored_clock=w.engines['a'].time_sec
    _snap,events=w.engines['a'].step(1.,'wait')
    out['forecast_side_effects']={'before_dv':before,'after_forecast_dv':after,'forecast_restored_time_before_real_step':restored_clock,'real_step_maneuver_events':sum(e.kind==EventKind.MANEUVER for e in events)}
    w=world()
    w.engines['a'].preload_bits(1000.)
    snap=w.snapshot()
    first=w.estimate('a',noise=.1)['stored_bits']
    w.restore(snap)
    again=w.estimate('a',noise=.1)['stored_bits']
    out['snapshot_rng']={'first':first,'after_restore':again,'equal':first==again}
    w=world()
    w.step(1.,{'ghost':'imaging'})
    out['unknown_actor']={'accepted_without_error':True,'clock_after':w.engines['a'].time_sec}
    a,b=engine('a'),engine('b')
    b.step(1.)
    w=world({'a':a,'b':b})
    w.step(1.)
    out['mixed_clocks']={k:e.time_sec for k,e in w.engines.items()}
    resumed=world(horizon=2.)
    resumed.step(1.)
    run_policy(resumed,[],step_sec=1.)
    out['resume_horizon']={'horizon_sec':2.,'end_time_sec':resumed.engines['a'].time_sec}
    request=TaskRequest('r',0.,0.,0.,10.,100.,1,'optical')
    receipt={'product_id':'p','request_id':'r','destination':'x','time_sec':5.,'offset_bits':0.,'bits':60.}
    one=delivery_score([request],[receipt],[])
    duplicate=delivery_score([request],[receipt,dict(receipt)],[])
    out['duplicate_receipts']={'single_on_time':one['on_time'],'duplicate_on_time':duplicate['on_time']}
    premature=dict(receipt,bits=100.,time_sec=1.)
    unreleased=TaskRequest('later',0.,0.,20.,30.,100.,1,'optical')
    premature['request_id']='later'
    out['before_release_receipt']=delivery_score([unreleased],[premature],[])['on_time']
    w=world()
    w.engines['a'].spacecraft.state.power.battery_soc=.01
    _snap,events=w.engines['a'].step(1.)
    score=delivery_score([],[],[])
    kpis=mission_kpis(w,[],[],score,runtime_sec=0.,replans=0,relay_bits=0.)
    out['safety_metric']={'constraint_events':sum(e.kind==EventKind.CONSTRAINT_VIOLATION for e in events),'reported_violations':kpis['safety_violations']}
    baseline={'version':2,'epoch':EPOCH,'horizon_sec':60,'instances':[{'id':'a','model':'cubesat-6u'}],'stations':[],'requests':[]}
    bad={'negative_horizon':dict(baseline,horizon_sec=-1),'nan_anomaly':dict(baseline,instances=[{'id':'a','model':'cubesat-6u','nu_deg':float('nan')}]),'bad_coordinates_and_deadline':dict(baseline,requests=[{'id':'r','lat_deg':120,'lon_deg':300,'release_sec':50,'deadline_sec':10,'size_bits':-1}])}
    checks={}
    with tempfile.TemporaryDirectory() as tmp:
        for name,config in bad.items():
            path=Path(tmp)/f'{name}.yaml'
            path.write_text(yaml.safe_dump(config))
            try:
                load_scenario_v2(path)
                checks[name]='accepted'
            except Exception as exc:
                checks[name]=type(exc).__name__
    out['invalid_scenarios']=checks
    def contacted(name):
        e=engine(name)
        lat,lon=subsatellite_latlon(e.orbit.r,e.now())
        e.stations=[GroundStation(name='Shared',lat_deg=lat,lon_deg=lon,min_elevation_deg=5.)]
        e._refresh_environment()
        e.accept_bits(100e6,origin='imaging',request_id=name)
        return e
    w=world({'a':contacted('a'),'b':contacted('b')})
    w.step(1.,{'a':'comms','b':'comms'})
    out['shared_station_direct_world']={name:e.delivered_bits for name,e in w.engines.items()}
    w=world({'a':contacted('a')},horizon=3.)
    req=TaskRequest('a',0.,0.,0.,10.,100e6,1,'optical')
    report=run_policy(w,[req],step_sec=1.,outages=[StationOutage('Shared',1.,3.)])
    out['outage_after_arming']={'receipts_times':[r['time_sec'] for r in report['receipts']],'delivered_bits':w.engines['a'].delivered_bits,'outage_start':1.}
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
