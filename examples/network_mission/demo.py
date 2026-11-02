"""Capture/compute/deliver with RF, thermal coupling, bookings, acquisition and ACK."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from space_env.runtime import AccessPolicy, MissionRuntime
from space_env.runtime.cli import apply_plan


def build_scenario(mode='nominal'):
    if mode not in {'nominal','outage','blocked'}:
        raise ValueError('mode must be nominal, outage or blocked')
    loader = importlib.util.spec_from_file_location('network_thermal_demo',Path(__file__).resolve().parents[1]/'thermal_network/demo.py')
    module = importlib.util.module_from_spec(loader); loader.loader.exec_module(module)
    scenario,_ = module.build_scenario('nominal')
    scenario.update(id='network-'+mode,lineage='B05-A')
    d = scenario['definition']; operations = d['operations']
    for v in operations['vehicles'].values():
        v['tx_power_w'] = 10.
    operations['radio_network'] = {
        'source':'synthetic S-band ground service; no modem/hardware qualification',
        'parameter_status':'synthetic','sample_sec':.25,
        'transmitters':{a:{'frequency_hz':2.2e9,'rf_power_w':1.,'gain_dbi':6.,
            'beamwidth_deg':None,
            'modes':[{'id':str(rate),'information_rate_bps':rate,'bits_per_symbol':2,
                      'code_rate':.5,'required_eb_n0_db':5.,'implementation_loss_db':1.}
                     for rate in (250000.,500000.,1000000.)]} for a in scenario['actor_order']},
        'receivers':{'Ground':{'gt_db_k':-100. if mode=='blocked' else 0.,
            'min_frequency_hz':2.1e9,'max_frequency_hz':2.3e9,'bandwidth_hz':2e6,
            'aggregate_rate_bps':1e6,'aggregate_bandwidth_hz':2e6,
            'acquisition_sec':.75,'ack_processing_sec':.1,'setup_power_w':100.,'tracking_power_w':50.}},
        'weather':{'Ground':{'source':'synthetic bounded uniform fade','seed':42,'step_sec':1.,'max_fade_db':2.}}
    }
    if mode == 'outage':
        operations['outages'] = [{'station':'Ground','start_sec':4.,'end_sec':5.},
                                {'station':'Ground','start_sec':9.5,'end_sec':10.25}]
    plan = []
    def add(op,key,**kw):
        plan.append({'op':op,'expected_revision':len(plan),'idempotency_key':key,**kw})
    def command(id,kind,duration,**kw):
        return {'command_id':id,'actor_id':'a','kind':kind,'duration_sec':duration,**kw}
    def ev(id,kind,payload):
        return {'id':id,'type':kind,'time_sec':0.,'source':'network demo','reason':'declared scheduled ground service','payload':payload}
    add('inject','import-service',events=[ev('reservation','external_reservations_imported',{'records':[{
        'provider':'synthetic-ground-provider','external_id':'service-a','revision':1,'expected_previous_revision':0,
        'status':'confirmed','actor':'a','station':'Ground','start_sec':4.,'end_sec':14.,'source':'synthetic provider export'}]}),
        ev('reservation-policy','station_reservation_policy_changed',{'station':'Ground','required':True})])
    add('submit','capture',commands=[command('capture','imaging',2.,request_id='customer-a')])
    add('advance','captured',dt_sec=2.)
    add('submit','process',commands=[command('process','transform',2.,job_id='features-a')])
    add('advance','computed',dt_sec=2.)
    add('submit','radio-first',commands=[command('radio-first','downlink',3.,destination_id='Ground',product_id='final-a')])
    add('advance','mid-acquisition',dt_sec=.25)
    prefix = len(plan)
    add('advance','mid-transmission',dt_sec=1.)
    add('advance','first-delivery',dt_sec=1.75)
    add('submit','capture-second',commands=[command('capture-second','imaging',2.,request_id='after-thermal')])
    add('advance','captured-second',dt_sec=2.)
    add('submit','radio-second',commands=[command('radio-second','downlink',3.5,destination_id='Ground',product_id='raw-after')])
    add('advance','second-delivery',dt_sec=3.5)
    add('advance','drain-ack',dt_sec=.25)
    return scenario,({'operations':plan[:prefix]},{'operations':plan[prefix:]})


def verify(model,mode):
    r = model._backend.report(); n = r['radio_network']; life = r['lifecycle']
    expected = 0 if mode=='blocked' else 2
    assert life['metrics'] == {'fulfilled_requests':expected,'total_requests':2}
    for values in life['ledger'].values():
        assert abs(values['balance_error_bits']) < 1e-6
    for values in r['spacecraft_bus']['ledgers'].values():
        assert abs(values['battery_balance_error_wh']) < 1e-9
        assert abs(values['thermal_balance_error_j']) < 1e-6
    for row in n['intervals']:
        assert sum(g['rate_bps'] for g in row['grants'].values()) <= 1e6
    if mode == 'blocked':
        assert not r['receipts'] and n['ground_energy_j']['Ground'] == 0.
    else:
        assert n['ground_energy_j']['Ground'] > 0
        assert all(a['acknowledged_sec'] > 0 for a in life['acknowledgements'])
    return {'ok':True,'mode':mode,'metrics':life['metrics'],'ground_energy_j':n['ground_energy_j'],
            'setup_terminal_seconds':n['setup_terminal_seconds'],'station_transmitted_bits':n['station_transmitted_bits'],
            'acknowledgements':life['acknowledgements'],'limits':'sampled threshold RF/ideal ACK; no concurrent ISL/DTN custody or calibrated weather'}


def main(directory,mode='nominal'):
    scenario,(prefix,resume) = build_scenario(mode)
    directory.mkdir(parents=True,exist_ok=True)
    for name,data in [('scenario',scenario),('plan',prefix),('resume',resume)]:
        (directory/(name+'.json')).write_text(json.dumps(data,allow_nan=False))
    m = MissionRuntime(scenario,access=AccessPolicy(truth=True))
    apply_plan(m,prefix); m.save(directory/'prefix.json')
    m = MissionRuntime.load(directory/'prefix.json',access=AccessPolicy(truth=True))
    for operation in resume['operations']:
        apply_plan(m,{'operations':[operation]})
        if operation['idempotency_key']=='mid-transmission':
            m.save(directory/'transmitting.json')
            m = MissionRuntime.load(directory/'transmitting.json',access=AccessPolicy(truth=True))
    m.save(directory/'final.json')
    assert MissionRuntime.load(directory/'final.json',access=AccessPolicy(truth=True)).checkpoint() == m.checkpoint()
    print(json.dumps(verify(m,mode),allow_nan=False))


if __name__ == '__main__':
    main(Path(sys.argv[1]),sys.argv[2] if len(sys.argv)>2 else 'nominal')
