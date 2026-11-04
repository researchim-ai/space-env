"""Multi-cycle compilation and conservative trajectory/geometry fast paths."""
import copy
import math
from unittest.mock import patch

import numpy as np
import pytest

from examples.coupled.full_orbit import configuration,manifest
from space_env.reference.providers import DynamicStateProvider
from space_env.reference.orbit import ForceModel,ReferenceState
from space_env.reference.thrust import BalancedCoupleThrust,FiniteThrust
from space_env.reference.frames import FrameContext
from space_env.reference.time import EOPSeries,Epoch
from space_env.scaling.benchmark import ScaleSpec
from space_env.scaling.coupled import CoupledMissionSpec,CoupledGates,coupled_evidence
from space_env.scaling.full_orbit import FullOrbitSpec,full_orbit_workload,full_orbit_geometry
from space_env.core.ground import GroundStation
from space_env.simulation.snapshot import digest


def provider(**forces):
    epoch=Epoch.utc('2026-07-18T12:00:00Z')
    return DynamicStateProvider(ReferenceState(epoch,[7000000.,0.,0.],[0.,7500.,0.]),
        60.,FrameContext(EOPSeries.packaged()),model=ForceModel(mass_kg=20.1,**forces),
        integration={'sample_sec':1.,'reuse_balanced_coast':True},dry_mass_kg=20.,isp_s=200.)


def couple():
    return BalancedCoupleThrust(.04,60.,20.,(0.,0.,0.),(.02,0.,0.),(.5,.5,.5))


def test_balanced_fast_path_preserves_translation_and_spends_mass():
    p=provider(j2=True);times=np.linspace(0.,60.,101)
    before=np.array([p.state(float(t)) for t in times]);sha=p.sha256
    with patch('space_env.reference.providers.propagate',side_effect=AssertionError('no propagation needed')):
        clock=0.
        for _ in range(200):
            assert p.burn(clock,.01,couple()) is False
            clock+=.01
    assert p.sha256!=sha and p.revision==200
    assert p.mass(2.)==pytest.approx(couple().mass(20.1,2.),abs=1e-11)
    assert np.max(np.abs(np.array([p.state(float(t)) for t in times])-before))<1e-8
    assert p.mass(.005)>p.mass(1.)>p.mass(3.)
    # Flattened slices avoid recursion depth proportional to number of segments.
    assert not hasattr(p.segments[-1][1].parent,'parent')


@pytest.mark.parametrize('force',['drag','srp'])
def test_mass_dependent_forces_require_full_propagation(force):
    p=provider(**{force:True})
    with patch('space_env.reference.providers.propagate',wraps=__import__('space_env.reference.providers',fromlist=['propagate']).propagate) as spy:
        assert p.burn(0.,.1,couple()) is True
    assert spy.call_count==2


def test_active_burn_and_subclass_do_not_use_balanced_fast_path():
    p=provider();p.burn(0.,1.,FiniteThrust(.01,200.,20.,(0.,1.,0.)))
    assert p.burn(.1,.1,couple()) is True
    class OtherCouple(BalancedCoupleThrust):pass
    p=provider()
    assert p.burn(0.,.1,OtherCouple(.04,60.,20.,(0.,0.,0.),(.02,0.,0.),(.5,.5,.5))) is True


def test_balanced_fast_path_matches_eager_model_on_common_grid():
    fast,eager=provider(j2=True),provider(j2=True)
    fast.burn(1.,.5,couple())
    with patch('space_env.reference.providers.BalancedCoupleThrust',object):
        assert eager.burn(1.,.5,couple()) is True
    for t in np.linspace(0.,60.,121):
        a,b=fast.state(float(t)),eager.state(float(t))
        assert np.linalg.norm(a[:3]-b[:3])<.01
        assert np.linalg.norm(a[3:]-b[3:])<1e-5
        assert fast.mass(float(t))==pytest.approx(eager.mass(float(t)),abs=1e-12)


def test_balanced_coast_is_opt_in_and_requires_dense_cadence():
    from space_env.reference.dynamic import IntegratorSettings
    assert 'reuse_balanced_coast' not in IntegratorSettings().model_dump()
    with pytest.raises(ValueError,match='sample_sec'):
        IntegratorSettings(reuse_balanced_coast=True)
    p=provider();p.reuse_balanced_coast=False
    assert p.burn(0.,.1,couple()) is True


def test_repeated_decimal_burns_use_actual_sealed_endpoints():
    p=provider();p.reuse_balanced_coast=False
    clock=1.
    for _ in range(20):
        assert p.burn(clock,.025,couple()) is True
        clock+=.025
    assert np.isfinite(p.state(clock)).all() and p.mass(clock)<20.1


def test_opt_in_native_unloading_spends_fuel_and_replays(tmp_path):
    from tests.test_momentum_unloading import scenario,runtime,unload,report
    from space_env.runtime import spacecraft_scenario,MissionRuntime
    native=copy.deepcopy(scenario.__wrapped__()['definition'])
    for orbit in native['orbits'].values():
        orbit['integration']={'sample_sec':1.,'reuse_balanced_coast':True}
    m=runtime(spacecraft_scenario(native,id='reuse',lineage='D2f'))
    unload(m);m.advance(1.5,expected_revision=m.revision,idempotency_key='advance')
    r=report(m)
    assert r['fuel_used_kg']['a']>0 and abs(r['fuel_balance_error_kg']['a'])<1e-9
    assert m._backend.geometry_revision==0
    m.save(tmp_path/'prefix.json')
    restored=MissionRuntime.load(tmp_path/'prefix.json',access=m.access)
    assert restored.checkpoint()==m.checkpoint()
    fork=m.fork();fork._backend.mission.ephemerides['a'].maneuvers[0]['thrust_n']=999.
    assert m._backend.mission.ephemerides['a'].maneuvers[0]['thrust_n']!=999.


def test_suffix_access_uses_global_scan_lattice_and_rejects_invalid_start():
    p=provider();s=GroundStation(name='test',lat_deg=0.,lon_deg=0.)
    full=p.access(s);suffix=p.access(s,start_sec=30.)
    assert suffix['coverage_sec']==[30.,60.]
    expected=[(max(30.,w['aos_sec']),w['los_sec']) for w in full['windows'] if w['los_sec']>30.]
    assert np.allclose([(w['aos_sec'],w['los_sec']) for w in suffix['windows']],expected,atol=.001)
    for start in (True,float('nan'),-1.,60.):
        with pytest.raises(ValueError):p.access(s,start_sec=start)


def test_full_orbit_compiler_keeps_all_requests_and_legacy_short_bound():
    cfg=configuration();before=copy.deepcopy(cfg);w=full_orbit_workload(cfg)
    assert cfg==before
    native=w['scenario']['definition'];requests=native['lifecycle']['requests']
    assert len(requests)==32 and not native['operations']['products']
    assert {r['release_sec'] for r in requests}=={0.,6300.}
    assert {r['deadline_sec'] for r in requests}=={6300.,12600.}
    assert all('rigid' not in b['attitude'] for b in native['spacecraft_bus'].values())
    assert w['coupled_gates']['min_delivery_epochs']==2
    assert w['coupled_gates']['native_spec_sha256']==digest(native)
    with pytest.raises(ValueError):CoupledMissionSpec.model_validate(cfg)
    s=ScaleSpec.model_validate(manifest(cfg));assert len(s.workloads)==4
    assert all(w.task['max_decisions']==1261 for w in s.workloads)
    assert all(w.resume_after_decisions==630 for w in s.workloads if w.resume_in_new_process)


def test_cycle_destinations_do_not_rotate_with_target_count():
    cfg=configuration();cfg['targets']=cfg['targets'][:3]
    w=full_orbit_workload(cfg)
    destinations={r['target_id']:r['destination_id'] for r in w['scenario']['definition']['lifecycle']['requests']}
    assert all(destinations[f't{i}-c0']==destinations[f't{i}-c1']==f'g{i}' for i in range(3))


@pytest.mark.parametrize('update',[
    {'horizon_sec':12000.,'cycles':3}, {'decision_sec':1.},
    {'cycles':2,'min_delivery_epochs':3}, {'mode':'all_outage'},
])
def test_full_orbit_rejects_infeasible_budgets_before_compilation(update):
    with pytest.raises(ValueError):FullOrbitSpec.model_validate(configuration()|update)


def test_expanded_denominator_and_rigid_tier():
    cfg=configuration();cfg['targets']*=2
    # Unique targets but too many actor x target x cycle products.
    cfg['targets']=[dict(t,id=f'x{i}') for i,t in enumerate(cfg['targets'])]
    cfg['planes'][0]['phases_deg']=[0.,180.]
    with pytest.raises(ValueError,match='expanded'):FullOrbitSpec.model_validate(cfg)
    native=full_orbit_workload(configuration()|{'attitude_model':'fixed-tick-rigid/1'})['scenario']['definition']
    assert all(b['attitude']['rigid']['max_total_substeps']==1000000 for b in native['spacecraft_bus'].values())


def test_epoch_gate_preserves_legacy_serialization():
    assert CoupledGates(native_spec_sha256='0'*64).model_dump()=={'native_spec_sha256':'0'*64,'min_acquired':0}
    with pytest.raises(ValueError):CoupledGates(native_spec_sha256='0'*64,min_delivery_epochs=2)


def test_epoch_gate_rejects_carryover_delivery_and_forged_kinematic_energy():
    from space_env.runtime import MissionRuntime,AccessPolicy
    w=full_orbit_workload(configuration());m=MissionRuntime(w['scenario'],access=AccessPolicy(truth=True))
    report=m._backend.report();life=report['lifecycle']
    requests=w['scenario']['definition']['lifecycle']['requests']
    chosen=[next(r for r in requests if r['release_sec']==t) for t in (0.,6300.)]
    for r in chosen:
        state=life['requests'][r['id']];state.update(status='completed',delivered_sec=r['release_sec']+100.)
        life['artifacts'][r['raw_product_id']]={'id':r['raw_product_id'],'source':'acquired','request_ids':[r['id']],
            'actor':r['eligible_actors'][0],'size_bits':r['raw_size_bits'],'created_sec':r['release_sec']+90.,
            'acquisition':{'target_id':r['target_id'],'validated_dwell_sec':2.}}
    result=coupled_evidence(w['scenario'],report,w['coupled_gates'])
    assert result['checks']['minimum_delivery_epochs']
    # Deliberately fabricated report stays invalid overall: this tests the epoch
    # predicate only and does not claim execution of the synthetic rows.
    assert not result['passed']
    life['requests'][chosen[0]['id']]['delivered_sec']=6400.
    result=coupled_evidence(w['scenario'],report,w['coupled_gates'])
    assert not result['checks']['minimum_delivery_epochs']
    report['spacecraft_bus']['states']['a-00']['slew_energy_wh']=1.
    assert not coupled_evidence(w['scenario'],report,w['coupled_gates'])['checks']['kinematic_paid_slew']


def test_full_orbit_cli_compiles_and_preserves_existing_output(tmp_path,monkeypatch,capsys):
    import sys,json
    from space_env.cli import main
    spec=tmp_path/'spec.json';out=tmp_path/'workload.json'
    spec.write_text(json.dumps(configuration()))
    monkeypatch.setattr(sys,'argv',['space-env','compile-full-orbit',str(spec),str(out)])
    main();assert json.loads(capsys.readouterr().out)['requests']==32
    before=out.read_bytes()
    with pytest.raises(SystemExit) as err:main()
    assert err.value.code==2 and out.read_bytes()==before


def test_full_orbit_geometry_measures_winding_and_retains_zero_pairs():
    cfg=configuration();d=full_orbit_geometry(cfg)
    assert d['passed'] and all(row['projected_revolutions']>=2 for row in d['actors'].values())
    assert len(d['contacts'])==16
    assert all('contact_sec' in row for row in d['contacts'])
    assert len(d['sha256'])==64


def test_future_refresh_matches_full_eager_access_and_preserves_history():
    from tests.test_reference_dynamic import dynamic_spec,runtime,burn
    spec=dynamic_spec.__wrapped__();spec['horizon_sec']=1800.
    m=runtime(spec)
    m.advance(600.,expected_revision=m.revision,idempotency_key='coast')
    backend=m._backend;past=copy.deepcopy(backend.mission.windows)
    provider=backend.mission.ephemerides['a']
    with patch.object(provider,'access',wraps=provider.access) as spy:
        m.submit([burn()],expected_revision=m.revision,idempotency_key='burn')
    assert all(call.kwargs.get('start_sec')==600. for call in spy.call_args_list)
    for name,station in backend.mission.stations.items():
        eager=provider.access(station)['windows']
        actual=backend.mission.windows['a',name]
        assert np.allclose(actual,[(r['aos_sec'],r['los_sec']) for r in eager],atol=.002)
        assert [(a,b) for a,b in actual if b<=600.]==[(a,b) for a,b in past['a',name] if b<=600.]
    restored=type(m).from_checkpoint(m.checkpoint(),access=m.access)
    assert m.checkpoint()==restored.checkpoint()
