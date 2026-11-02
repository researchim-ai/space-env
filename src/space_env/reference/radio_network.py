"""Persistent ground-terminal acquisition and deterministic sampled RF admission."""
from __future__ import annotations

import copy
import hashlib
import math

import numpy as np

from space_env.core.frames import geodetic_to_ecef
from space_env.reference.radio import RadioNetworkSettings, rf_budget


def angular_separation(left, right):
    def unit(angles):
        az, el = np.deg2rad(angles)
        return np.array([math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el)])
    return math.degrees(math.acos(float(np.clip(np.dot(unit(left), unit(right)), -1, 1))))


class RadioGroundNetwork:
    def __init__(self, config, mission):
        self.config = RadioNetworkSettings.model_validate(config)
        if set(self.config.transmitters) != set(mission.actors) or set(self.config.receivers) != set(mission.stations):
            raise ValueError('RF transmitters/receivers must cover every actor/station')
        if set(self.config.weather) - set(mission.stations):
            raise ValueError('RF weather references unknown station')
        for a, t in self.config.transmitters.items():
            if t.beamwidth_deg is not None and not getattr(mission,'supports_body_fixed_solar',False):
                raise ValueError('body antenna pattern requires coupled spacecraft profile')
            v = mission.vehicles[a]
            if v.thermal is not None and t.rf_power_w+v.thermal.heat_tx_w > v.tx_power_w+1e-12:
                raise ValueError('RF output plus transmitter heat exceeds DC input')
            if t.rf_power_w > mission.vehicles[a].tx_power_w:
                raise ValueError('RF output exceeds transmitter DC power')
        if any(g.terminals > 64 for g in mission.stations.values()) or sum(g.terminals for g in mission.stations.values()) > 512:
            raise ValueError('RF profile supports at most 64 terminals per station and 512 total')
        self.terminals = {s:[{'actor':None, 'ready_sec':None, 'pointing':None} for _ in range(g.terminals)]
                          for s,g in mission.stations.items()}
        self.grants = {}
        self.links = {}
        self.ack_boundaries = []
        self.intervals = []
        self.transitions = []
        self.ground_energy_j = dict.fromkeys(mission.stations, 0.)
        self.setup_seconds = dict.fromkeys(mission.stations, 0.)
        self.tracking_seconds = dict.fromkeys(mission.stations, 0.)
        self.transmitted_bits = dict.fromkeys(mission.stations, 0.)

    def weather(self, station, now):
        row = self.config.weather.get(station)
        if row is None:
            return {'time_sec':now, 'fade_db':0., 'outage':False}
        index = math.floor((now+1e-9)/row.step_sec)
        def sample(label):
            payload = f'{row.seed}:{station}:{index}:{label}'.encode()
            return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') / 2**64
        return {'time_sec':index*row.step_sec, 'fade_db':row.max_fade_db*sample('fade'),
                'outage':sample('outage') < row.outage_probability}

    def boundary(self, now, right):
        step = self.config.sample_sec
        right = min(right, (math.floor((now+1e-9)/step)+1)*step)
        for row in self.config.weather.values():
            right = min(right, (math.floor((now+1e-9)/row.step_sec)+1)*row.step_sec)
        for rows in self.terminals.values():
            for row in rows:
                if row['ready_sec'] is not None and row['ready_sec'] > now+1e-9:
                    right = min(right, row['ready_sec'])
        right = min(right, *[t for t in self.ack_boundaries if now+1e-9 < t < right], right)
        return right

    def budget(self, mission, actor, station, bandwidth=None, rate_cap=None):
        # Values are held on the absolute RF grid; geometry is never extrapolated.
        sample = math.floor((mission.clock+1e-9)/self.config.sample_sec)*self.config.sample_sec
        e = mission.ephemerides[actor]
        g = mission.stations[station]
        look = e.look(g, sample)
        weather = self.weather(station, mission.clock)
        error = 0.
        bus = getattr(mission, 'bus', None)
        tx = self.config.transmitters[actor]
        if tx.beamwidth_deg is not None:
            if bus is None:
                raise ValueError('body antenna pattern requires coupled spacecraft attitude')
            from space_env.reference.spacecraft_bus import rotate_body
            r = e.state(mission.clock)[:3]
            ground = e.context.matrix(e.epoch.shifted(mission.clock)).T @ geodetic_to_ecef(g.lat_deg,g.lon_deg,g.alt_m)
            direction = ground-r
            direction /= np.linalg.norm(direction)
            bore = rotate_body(bus.states[actor]['quaternion_wxyz'], (0.,0.,1.))
            error = math.degrees(math.acos(float(np.clip(np.dot(bore, direction),-1,1))))
        cap = mission.transmitter_rates[actor]
        if mission.station_rates[station] is not None:
            cap = min(cap, mission.station_rates[station])
        if rate_cap is not None:
            cap = min(cap, rate_cap)
        result = rf_budget(tx, self.config.receivers[station], look['range_m'], fade_db=weather['fade_db'],
                           pointing_error_deg=error, bandwidth_hz=bandwidth, rate_cap_bps=cap,
                           doppler_hz=-tx.frequency_hz*look['range_rate_m_s']/299792458.)
        result.update(actor=actor,station=station,sample_sec=sample,range_m=look['range_m'],weather=weather)
        if weather['outage']:
            result.update(rate_bps=0., occupied_bandwidth_hz=0., selected_mode=None)
        return result

    def prepare(self, mission, candidates):
        """Ground acquisition consumes a terminal, not an active spacecraft TX.

        Candidate filtering already enforces product/booking/power/bus admission.
        Read-only transmission selectors never advance this terminal state.
        """
        now = mission.clock
        self.grants, self.links = {}, {}
        requested = {}
        for a,(s,_) in candidates.items():
            b = self.budget(mission,a,s)
            self.links[a] = {**b,'state':'link_unavailable'}
            if b['rate_bps'] > 0:
                requested[a] = s
        for s,rows in self.terminals.items():
            rx = self.config.receivers[s]
            for i,row in enumerate(rows):
                a = row['actor']
                if a is not None and requested.get(a) != s:
                    # Retain the physical antenna direction for the next handover.
                    look = mission.ephemerides[a].look(mission.stations[s],now)
                    row['pointing'] = [look['azimuth_deg'],look['elevation_deg']]
                    self.transitions.append({'time_sec':now,'station':s,'terminal':i,'actor':a,'state':'released'})
                    row.update(actor=None,ready_sec=None)
            occupied = {r['actor'] for r in rows}
            waiting = sorted(a for a,station in requested.items() if station==s and a not in occupied)
            for a in waiting:
                free = next((i for i,r in enumerate(rows) if r['actor'] is None),None)
                if free is None:
                    break
                row = rows[free]
                look = mission.ephemerides[a].look(mission.stations[s],now)
                point = [look['azimuth_deg'],look['elevation_deg']]
                delay = rx.acquisition_sec
                if row['pointing'] is not None:
                    delay += angular_separation(row['pointing'],point)/rx.slew_rate_deg_s
                row.update(actor=a, ready_sec=now+delay, pointing=point)
                self.transitions.append({'time_sec':now,'station':s,'terminal':free,'actor':a,
                                         'state':'acquiring','ready_sec':now+delay})
            active = [r['actor'] for r in rows if r['actor'] is not None]
            share_bw = (rx.aggregate_bandwidth_hz/len(active) if active and rx.bandwidth_policy=='equal-share' else None)
            bw_left = rx.aggregate_bandwidth_hz
            rate_left = rx.aggregate_rate_bps
            # Preserve ongoing allocations; no optimal/fair spectrum scheduler claim.
            for row in rows:
                a = row['actor']
                if a is None:
                    continue
                budget_bw = share_bw if share_bw is not None else bw_left
                b = self.budget(mission,a,s,bandwidth=budget_bw,rate_cap=rate_left)
                state = 'acquiring' if row['ready_sec'] > now+1e-9 else 'locked'
                if b['rate_bps'] <= 0:
                    state = 'capacity_blocked'
                self.links[a] = {**b,'state':state,'ready_sec':row['ready_sec']}
                if state != 'locked':
                    continue
                self.grants[a] = {'station':s,'rate_bps':b['rate_bps'],'bandwidth_hz':b['occupied_bandwidth_hz']}
                if bw_left is not None:
                    bw_left = max(0.,bw_left-b['occupied_bandwidth_hz'])
                if rate_left is not None:
                    rate_left = max(0.,rate_left-b['rate_bps'])

    def rate(self, actor, station):
        row = self.grants.get(actor)
        return row['rate_bps'] if row is not None and row['station']==station else 0.

    def integrate(self, mission, start, span, chosen, receipt_start):
        if len(self.intervals) >= self.config.max_intervals or len(self.transitions) > self.config.max_intervals*4:
            raise ValueError('radio network history budget exhausted')
        station_bits = dict.fromkeys(mission.stations,0.)
        for receipt in mission.receipts[receipt_start:]:
            station = receipt['station']
            actor = receipt['actor']
            distance = self.links[actor]['range_m']
            from space_env.reference.radio import C_M_S
            receipt['ack_due_sec'] = start+span+2*distance/C_M_S+self.config.receivers[station].ack_processing_sec
            receipt['ack_model'] = 'ideal-round-trip-plus-processing/1'
            if mission.remaining[receipt['product_id']] == 0:
                self.ack_boundaries.append(receipt['ack_due_sec'])
            station_bits[receipt['station']] += receipt['bits']
        for s,rows in self.terminals.items():
            rx = self.config.receivers[s]
            for row in rows:
                if row['actor'] is None:
                    continue
                setup = row['ready_sec'] > start+1e-9
                (self.setup_seconds if setup else self.tracking_seconds)[s] += span
                self.ground_energy_j[s] += span*(rx.setup_power_w if setup else rx.tracking_power_w)
            self.transmitted_bits[s] += station_bits[s]
        self.intervals.append({'start_sec':start,'end_sec':start+span,
                               'grants':copy.deepcopy(self.grants),
                               'transmitters':{a:s for a,(s,_) in chosen.items()},
                               'station_bits':station_bits})

    def observation(self, mission, actor):
        return {'profile':self.config.model,'link':copy.deepcopy(self.links.get(actor)),
                'weather':{s:self.weather(s,mission.clock) for s in self.config.receivers}}

    def report(self, mission):
        return {'schema':'space-env.radio-ground/1','model':self.config.model,
                'terminals':copy.deepcopy(self.terminals),'links':copy.deepcopy(self.links),
                'grants':copy.deepcopy(self.grants),'transitions':copy.deepcopy(self.transitions),
                'intervals':copy.deepcopy(self.intervals),'ground_energy_j':dict(self.ground_energy_j),
                'setup_terminal_seconds':dict(self.setup_seconds),'tracking_terminal_seconds':dict(self.tracking_seconds),
                'station_transmitted_bits':dict(self.transmitted_bits),
                'weather_now':{s:self.weather(s,mission.clock) for s in mission.stations},
                'scope':'sampled declared link budget and ideal ground tracking; threshold modem, aggregate capacity pool; no waveform/interference or ITU rain calibration'}
