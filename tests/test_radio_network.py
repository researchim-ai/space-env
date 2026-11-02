"""Independent RF identities, causal ground admission, budgets and mission replay."""
from __future__ import annotations

import copy
import math

import pytest

from space_env.reference.operations import ReferenceDownlinkMission
from space_env.reference.radio import (
    BOLTZMANN_J_K,
    C_M_S,
    RadioNetworkSettings,
    rf_budget,
)
from space_env.runtime import (
    AccessPolicy,
    MissionRuntime,
    RuntimeContractError,
    reference_scenario,
)
from tests.test_reference_operations import spec as operations_fixture


def settings(actors=('a','b')):
    return {'source':'synthetic engineering fixture', 'parameter_status':'synthetic', 'sample_sec':.5,
            'transmitters':{a:{'frequency_hz':2.2e9,'rf_power_w':1.,'gain_dbi':6.,
                'modes':[{'id':str(r),'information_rate_bps':r,'bits_per_symbol':2,'code_rate':.5,
                          'required_eb_n0_db':5.} for r in (250000.,500000.,1000000.)]} for a in actors},
            'receivers':{'Ground':{'gt_db_k':0.,'min_frequency_hz':2.1e9,'max_frequency_hz':2.3e9,
                                 'bandwidth_hz':2e6}}}


@pytest.fixture(scope='module')
def spec():
    data = copy.deepcopy(operations_fixture.__wrapped__())
    for a in data['vehicles']:
        data['vehicles'][a]['tx_power_w'] = 10.
    data['radio_network'] = settings()
    return data


def txrx():
    c = RadioNetworkSettings.model_validate(settings())
    return c.transmitters['a'],c.receivers['Ground']


def test_friis_noise_density_and_information_energy_independent_linear_reference():
    tx,rx = txrx()
    r = 1e6
    b = rf_budget(tx,rx,r)
    received = tx.rf_power_w * 10**(tx.gain_dbi/10) * (C_M_S/tx.frequency_hz/(4*math.pi*r))**2
    assert b['cn0_db_hz'] == pytest.approx(10*math.log10(received/BOLTZMANN_J_K),abs=1e-12)
    for mode in b['modes']:
        ebn0 = 10*math.log10(received/BOLTZMANN_J_K/mode['information_rate_bps'])
        assert mode['margin_db'] == pytest.approx(ebn0-5.)


def test_inverse_square_range_and_frequency_scaling():
    tx,rx = txrx()
    base = rf_budget(tx,rx,1e6)
    assert base['cn0_db_hz']-rf_budget(tx,rx,2e6)['cn0_db_hz'] == pytest.approx(20*math.log10(2))
    tx = tx.model_copy(update={'frequency_hz':4.4e9})
    assert base['cn0_db_hz']-rf_budget(tx,rx,1e6)['cn0_db_hz'] == pytest.approx(20*math.log10(2))


def test_coding_rolloff_overhead_bandwidth_and_net_information_rate():
    tx,rx = txrx()
    m = tx.modes[-1]
    assert m.bandwidth_hz == 1.25e6
    m = m.model_copy(update={'payload_fraction':.5})
    assert m.bandwidth_hz == 2.5e6
    tx = tx.model_copy(update={'modes':(m,)})
    assert rf_budget(tx,rx,1e6)['rate_bps'] == 0


def test_modem_uses_discrete_feasible_modes_and_reports_thresholds():
    tx,rx = txrx()
    b = rf_budget(tx,rx,1e6)
    excess = b['modes'][-1]['margin_db']
    assert rf_budget(tx,rx,1e6,fade_db=excess+1.)['rate_bps'] == 500000.
    assert rf_budget(tx,rx,1e6,bandwidth_hz=400000.)['rate_bps'] == 250000.
    assert rf_budget(tx,rx,1e6,rate_cap_bps=750000.)['rate_bps'] == 500000.
    assert rf_budget(tx,rx,1e6,fade_db=200.)['rate_bps'] == 0


def test_full_occupied_carrier_must_fit_receiver_passband():
    tx,rx = txrx()
    tx = tx.model_copy(update={'frequency_hz':rx.min_frequency_hz})
    b = rf_budget(tx,rx,1e6)
    assert b['rate_bps'] == 0 and not any(m['band_compatible'] for m in b['modes'])


def test_beamwidth_half_power_and_explicit_pointing_loss():
    tx,rx = txrx()
    tx = tx.model_copy(update={'beamwidth_deg':10.})
    b = rf_budget(tx,rx,1e6,pointing_error_deg=5.)
    assert b['pattern_loss_db'] == pytest.approx(3.)
    assert rf_budget(tx,rx,1e6,pointing_error_deg=180.)['pattern_loss_db'] == 60.


@pytest.mark.parametrize('kwargs',[{'range_m':0.},{'range_m':float('nan')},{'range_m':True},
    {'range_m':1.,'fade_db':-1.},{'range_m':1.,'pointing_error_deg':181.},
    {'range_m':1.,'rate_cap_bps':-1.},{'range_m':1.,'bandwidth_hz':float('inf')}])
def test_invalid_numeric_inputs_rejected(kwargs):
    tx,rx = txrx()
    with pytest.raises(ValueError):
        rf_budget(tx,rx,**kwargs)


@pytest.mark.parametrize('mutation',[
    lambda c:c['receivers']['Ground'].update(min_frequency_hz=3e9),
    lambda c:c['receivers']['Ground'].update(bandwidth_policy='equal-share'),
    lambda c:c['transmitters']['a']['modes'].append(c['transmitters']['a']['modes'][0]),
    lambda c:c.update(sample_sec=True),
    lambda c:c['transmitters']['a'].update(unknown=1.),
])
def test_strict_configuration(mutation):
    c = settings(); mutation(c)
    with pytest.raises(ValueError):
        RadioNetworkSettings.model_validate(c)


def test_visible_does_not_imply_rf_closure(spec):
    d = copy.deepcopy(spec)
    d['radio_network']['receivers']['Ground']['gt_db_k'] = -100.
    m = ReferenceDownlinkMission(d)
    assert 'Ground' in m.observation()['actors']['a']['contacts']
    m.step(3.,{'a':'Ground'})
    assert not m.receipts and m.tx_seconds['a'] == 0


def test_ground_acquisition_costs_time_and_separate_ground_energy(spec):
    d = copy.deepcopy(spec)
    d['radio_network']['receivers']['Ground'].update(acquisition_sec=2.,setup_power_w=100.,tracking_power_w=50.)
    m = ReferenceDownlinkMission(d)
    m.step(2.,{'a':'Ground'})
    assert m.remaining['a'] == 2e6 and m.tx_seconds['a'] == 0
    assert m.radio_network.ground_energy_j['Ground'] == 200.
    m.step(2.,{'a':'Ground'})
    assert m.remaining['a'] == 0 and m.tx_seconds['a'] == pytest.approx(2.)
    assert m.radio_network.ground_energy_j['Ground'] == pytest.approx(300.)
    assert m.report()['on_time_bits'] == 0  # ACK is still in flight.
    m.step(.1,{})
    assert m.report()['on_time_bits'] == 2e6
    assert abs(m.report()['energy_ledger']['a']['balance_error_wh']) < 1e-12


def test_late_ack_and_no_double_credit(spec):
    d = copy.deepcopy(spec)
    d['radio_network']['receivers']['Ground']['ack_processing_sec'] = 4.
    d['products'][0]['deadline_sec'] = 5.
    m = ReferenceDownlinkMission(d)
    m.step(2.,{'a':'Ground'})
    assert m.remaining['a'] == 0 and m.acknowledged_time('a') is None
    m.step(5.,{'a':'Ground'})
    p = next(p for p in m.report()['products'] if p['id']=='a')
    assert p['status'] == 'delivered_late' and p['completed_sec'] > 6.
    assert sum(r['bits'] for r in m.receipts) == 2e6


@pytest.mark.parametrize('policy',['independent','equal-share'])
def test_aggregate_rate_and_bandwidth_are_not_per_terminal(spec,policy):
    d = copy.deepcopy(spec)
    d['stations'][0]['terminals'] = 2
    d['radio_network']['receivers']['Ground'].update(aggregate_rate_bps=1.5e6,aggregate_bandwidth_hz=2e6,bandwidth_policy=policy)
    m = ReferenceDownlinkMission(d)
    m.step(1.,{'a':'Ground','b':'Ground'})
    for row in m.radio_network.intervals:
        assert sum(g['rate_bps'] for g in row['grants'].values()) <= 1.5e6
        assert sum(g['bandwidth_hz'] for g in row['grants'].values()) <= 2e6
        for bits in row['station_bits'].values():
            assert bits <= 1.5e6*(row['end_sec']-row['start_sec'])+1e-6
    assert m.remaining['b'] < 2e6


def test_terminal_handover_is_not_instant_even_after_product_completion(spec):
    d = copy.deepcopy(spec)
    d['radio_network']['receivers']['Ground']['acquisition_sec'] = 1.
    m = ReferenceDownlinkMission(d)
    m.step(5.,{'a':'Ground','b':'Ground'})
    assert min(r['start_sec'] for r in m.receipts if r['actor']=='a') == 1.
    assert min(r['start_sec'] for r in m.receipts if r['actor']=='b') >= 4.


def test_seeded_weather_is_causal_station_specific_and_partition_independent(spec):
    d = copy.deepcopy(spec)
    d['radio_network']['weather'] = {'Ground':{'source':'synthetic uniform fade','seed':42,'step_sec':1.,'max_fade_db':40.,'outage_probability':.2}}
    m,n = ReferenceDownlinkMission(d),ReferenceDownlinkMission(d)
    for t in (0.,.9,1.,2.):
        assert m.radio_network.weather('Ground',t) == n.radio_network.weather('Ground',t)
    assert m.radio_network.weather('Ground',0.) == m.radio_network.weather('Ground',.9)
    m.step(5.,{'a':'Ground'})
    for dt in (.3,.7,1.2,2.8):
        n.step(dt,{'a':'Ground'})
    assert m.remaining == pytest.approx(n.remaining,abs=1e-6)
    assert m.energy_wh == pytest.approx(n.energy_wh,abs=1e-12)
    view = m.observation()['actors']['a']['radio_network']
    assert all(row['time_sec'] <= m.clock for row in view['weather'].values())


def test_outage_resets_lock_requires_reacquisition(spec):
    d = copy.deepcopy(spec)
    d['outages'] = [{'station':'Ground','start_sec':1.,'end_sec':2.}]
    d['radio_network']['receivers']['Ground']['acquisition_sec'] = .5
    m = ReferenceDownlinkMission(d)
    m.step(3.,{'a':'Ground'})
    after = [r for r in m.receipts if r['start_sec'] >= 2.]
    assert min(r['start_sec'] for r in after) >= 2.5


def test_readonly_selection_does_not_mutate_ground_state(spec):
    m = ReferenceDownlinkMission(spec)
    before = copy.deepcopy(m.radio_network.__dict__)
    for _ in range(3):
        m.select_transmissions({'a':'Ground'},{},generation={'a':0.,'b':0.},ignore_radio=True)
    assert m.radio_network.__dict__ == before


def test_history_budget_failure_rolls_back_terminal_resources_receipts_and_ack(spec):
    d = copy.deepcopy(spec); d['radio_network']['max_intervals'] = 1
    m = ReferenceDownlinkMission(d)
    before = m.report()
    with pytest.raises(ValueError,match='budget'):
        m.step(2.,{'a':'Ground'})
    assert m.report() == before


def test_runtime_canonical_config_mid_acquisition_and_mid_transmission_replay(spec,tmp_path):
    d = copy.deepcopy(spec)
    d['radio_network']['receivers']['Ground']['acquisition_sec'] = 1.
    model = MissionRuntime(reference_scenario(d,id='rf-test',lineage='B05'),access=AccessPolicy(truth=True))
    model.submit([{'command_id':'radio','actor_id':'a','kind':'downlink','destination_id':'Ground','product_id':'a','duration_sec':5.}],expected_revision=0,idempotency_key='s')
    for i,dt in enumerate((.5,1.,3.5)):
        model.advance(dt,expected_revision=model.revision,idempotency_key=f'a{i}')
        path = tmp_path/'state.json'; model.save(path)
        loaded = MissionRuntime.load(path,access=AccessPolicy(truth=True))
        assert loaded.checkpoint() == model.checkpoint()
        model = loaded
    assert model._backend.mission.report()['on_time_bits'] == 2e6


def test_missing_actor_and_rf_power_above_dc_are_rejected(spec):
    d = copy.deepcopy(spec); d['radio_network']['transmitters'].pop('b')
    with pytest.raises(ValueError,match='cover'):
        ReferenceDownlinkMission(d)
    d = copy.deepcopy(spec); d['radio_network']['transmitters']['a']['rf_power_w'] = 100.
    with pytest.raises(ValueError,match='DC'):
        ReferenceDownlinkMission(d)


def event(mission,kind,payload,id='event'):
    return {'id':id,'type':kind,'time_sec':mission.clock,'source':'test','reason':'test policy','payload':payload}


def reservation(revision=1,previous=0,**values):
    return {'provider':'scheduler','external_id':'booking-1','revision':revision,'expected_previous_revision':previous,
            'status':'confirmed','actor':'a','station':'Ground','start_sec':0.,'end_sec':10.,
            'source':'synthetic provider export',**values}


def test_external_and_local_bookings_share_one_capacity_sweep(spec):
    m = ReferenceDownlinkMission(spec)
    m.apply_events([event(m,'external_reservations_imported',{'records':[reservation()]})])
    before = m.report()
    with pytest.raises(ValueError,match='conflict'):
        m.apply_events([event(m,'terminal_booking_added',{'id':'local','actor':'b','station':'Ground','start_sec':1.,'end_sec':2.},'local')])
    assert m.report() == before
    # Non-reserved b cannot consume a's reservation.
    m.step(1.,{'b':'Ground'})
    assert m.tx_seconds['b'] == 0


def test_external_revision_history_stale_rollback_and_idempotent_import(spec):
    m = ReferenceDownlinkMission(spec)
    r = reservation()
    m.apply_events([event(m,'external_reservations_imported',{'records':[r]})])
    m.apply_events([event(m,'external_reservations_imported',{'records':[r]},'duplicate-import')])
    assert len(m.bookings) == 1
    m.step(1.,{})
    r2 = reservation(2,1,start_sec=1.,end_sec=12.)
    m.apply_events([event(m,'external_reservations_imported',{'records':[r2]},'revised')])
    assert len(m.bookings) == 2 and len(m.booking_cancelled) == 1
    before = m.report()
    with pytest.raises(ValueError,match='stale'):
        m.apply_events([event(m,'external_reservations_imported',{'records':[reservation(3,0,start_sec=1.)]},'stale')])
    assert m.report() == before
    r3 = reservation(3,2,status='cancelled',start_sec=1.,end_sec=12.)
    m.apply_events([event(m,'external_reservations_imported',{'records':[r3]},'cancel')])
    assert all(v['booking_id'] is None for v in m.external_reservations.values())


def test_external_conflicting_batch_is_atomic(spec):
    m = ReferenceDownlinkMission(spec); before = m.report()
    records = [reservation(),reservation(external_id='second',actor='b')]
    with pytest.raises(ValueError,match='conflict'):
        m.apply_events([event(m,'external_reservations_imported',{'records':records})])
    assert m.report() == before and not m.external_reservations


def test_actual_body_antenna_orientation_changes_link(spec):
    from tests.test_spacecraft_bus import scenario as bus_fixture
    d = copy.deepcopy(bus_fixture.__wrapped__())
    actors = d['actor_order']
    config = settings(actors)
    config['transmitters']['a']['beamwidth_deg'] = 5.
    for v in d['definition']['operations']['vehicles'].values():
        v['tx_power_w'] = 10.
    for v in d['definition']['operations']['vehicles'].values():
        v['tx_power_w'] = 10.
    d['definition']['operations']['radio_network'] = config
    model = MissionRuntime(d,access=AccessPolicy(truth=True))
    m = model._backend.mission
    b = m.radio_network.budget(m,'a','Ground')
    assert b['pointing_error_deg'] < .001
    model._backend.bus.states['a']['quaternion_wxyz'] = [1.,0.,0.,0.]
    assert m.radio_network.budget(m,'a','Ground')['rate_bps'] == 0


def test_coupled_rf_heat_output_is_physically_bounded():
    from tests.test_spacecraft_bus import scenario as bus_fixture
    d = copy.deepcopy(bus_fixture.__wrapped__())
    c = settings(d['actor_order'])
    c['transmitters']['a']['rf_power_w'] = 3.
    for v in d['definition']['operations']['vehicles'].values():
        v['tx_power_w'] = 10.
    d['definition']['operations']['radio_network'] = c
    with pytest.raises(RuntimeContractError,match='heat exceeds DC'):
        MissionRuntime(d)


def test_lifecycle_waits_for_ack_before_reward_and_records_exact_ack_time():
    from tests import test_reference_lifecycle as life
    from tests.test_resource_lanes import scenario as lane_fixture
    d = copy.deepcopy(lane_fixture.__wrapped__())
    c = settings(d['actor_order'])
    c['receivers']['Ground'].update(ack_processing_sec=2.,acquisition_sec=.5)
    for v in d['definition']['operations']['vehicles'].values():
        v['tx_power_w'] = 10.
    d['definition']['operations']['radio_network'] = c
    d['definition']['lifecycle']['requests'][0]['delivery_product_id'] = 'raw'
    d['definition']['lifecycle']['jobs'] = []
    model = life.runtime(d)
    life.acquire(model)
    model.submit([{'command_id':'radio','actor_id':'a','kind':'downlink','destination_id':'Ground',
                   'product_id':'raw','duration_sec':2.}],expected_revision=model.revision,idempotency_key='radio')
    # The customer raw product is serialized, but not acknowledged yet.
    life.advance(model,2.,'serialize')
    assert model._backend.mission.remaining['raw'] == 0
    assert len([r for r in model._backend.mission.radio_network.transitions if r['state']=='acquiring']) == 1
    assert life.lifecycle(model)['artifacts']['raw']['status'] != 'delivered'
    assert life.lifecycle(model)['metrics']['fulfilled_requests'] == 0
    life.advance(model,3.,'ack')
    assert life.lifecycle(model)['artifacts']['raw']['status'] == 'delivered'
    assert life.lifecycle(model)['metrics']['fulfilled_requests'] == 1
    acks = [r for r in life.lifecycle(model)['acknowledgements'] if r['product_id']=='raw']
    # Raw may not have a destination obligation; native confirmation remains causal.
    assert model._backend.mission.acknowledged_time('raw') > 5.
    assert not acks or acks[0]['acknowledged_sec'] > 5.


def test_doppler_tracking_and_shifted_passband():
    tx,rx = txrx()
    rx = rx.model_copy(update={'doppler_tracking_limit_hz':1000.})
    assert rf_budget(tx,rx,1e6,doppler_hz=1001.)['rate_bps'] == 0
    assert rf_budget(tx,rx,1e6,doppler_hz=-999.)['rate_bps'] > 0
    rx = rx.model_copy(update={'max_frequency_hz':tx.frequency_hz+625000.})
    b = rf_budget(tx,rx,1e6,doppler_hz=100.)
    assert b['rate_bps'] == 500000.  # Largest occupied band crosses the edge.


def test_ground_slew_uses_spherical_separation_and_wrap():
    from space_env.reference.radio_network import angular_separation
    assert angular_separation([359.,0.],[1.,0.]) == pytest.approx(2.)
    assert angular_separation([0.,90.],[180.,90.]) == pytest.approx(0.)
    assert angular_separation([0.,0.],[90.,0.]) == pytest.approx(90.)


def test_actual_antenna_slew_delays_handover(spec):
    d = copy.deepcopy(spec)
    d['radio_network']['receivers']['Ground'].update(acquisition_sec=.5,slew_rate_deg_s=30.)
    m = ReferenceDownlinkMission(d)
    for a,az in [('a',0.),('b',90.)]:
        original = m.ephemerides[a].look
        def shifted(station,clock,*,frequency_hz=None,original=original,az=az):
            return {**original(station,clock,frequency_hz=frequency_hz),'azimuth_deg':az,'elevation_deg':0.}
        m.ephemerides[a].look = shifted
    m.step(7.,{'a':'Ground','b':'Ground'})
    acquisitions = [r for r in m.radio_network.transitions if r['state']=='acquiring']
    second = next(r for r in acquisitions if r['actor']=='b')
    assert second['ready_sec']-second['time_sec'] == pytest.approx(3.5)
    assert min(r['start_sec'] for r in m.receipts if r['actor']=='b') == pytest.approx(6.)


def test_operator_rate_change_obeys_modem_and_can_disable_link(spec):
    m = ReferenceDownlinkMission(spec)
    m.apply_events([event(m,'station_rate_changed',{'station':'Ground','rate_bps':750000.})])
    m.step(1.,{'a':'Ground'})
    assert sum(r['bits'] for r in m.receipts) == pytest.approx(500000.)
    m.apply_events([event(m,'station_rate_changed',{'station':'Ground','rate_bps':0.},'disable')])
    m.step(1.,{'a':'Ground'})
    assert sum(r['bits'] for r in m.receipts) == pytest.approx(500000.)
    assert m.radio_network.terminals['Ground'][0]['actor'] is None


def test_runtime_failure_rolls_back_scheduler_ground_aliases():
    from tests import test_reference_lifecycle as life
    from tests.test_resource_lanes import scenario as lane_fixture
    d = copy.deepcopy(lane_fixture.__wrapped__())
    c = settings(d['actor_order']); c['max_intervals'] = 1
    for v in d['definition']['operations']['vehicles'].values():
        v['tx_power_w'] = 10.
    d['definition']['operations']['radio_network'] = c
    m = life.runtime(d); before = m.checkpoint()
    with pytest.raises(RuntimeContractError,match='budget'):
        life.advance(m,1.,'too-many-intervals')
    assert m.checkpoint() == before
    assert m._backend.bus is m._backend.mission.bus


def test_external_import_revision_is_portable_runtime_history(spec,tmp_path):
    m = MissionRuntime(reference_scenario(spec,id='booking-portable',lineage='B05'),access=AccessPolicy(truth=True))
    e = event(m._backend.mission,'external_reservations_imported',{'records':[reservation()]})
    m.inject([e],expected_revision=0,idempotency_key='import')
    m.advance(1.,expected_revision=1,idempotency_key='advance')
    path = tmp_path/'imported.json'; m.save(path)
    assert MissionRuntime.load(path,access=AccessPolicy(truth=True)).checkpoint() == m.checkpoint()
