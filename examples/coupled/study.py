"""Frozen engineering evaluation cases; no outcome-based coordinate search."""
import argparse
import copy
from pathlib import Path

from space_env.scaling.study import run_mission_study
from space_env.simulation.snapshot import atomic_write, canonical_json


def configuration(fixed_clock=False):
    base = {'id': 'development',
            'planes': [{'id': 'a', 'inclination_deg': 55., 'raan_deg': 0., 'phases_deg': [0.]},
                       {'id': 'b', 'inclination_deg': 75., 'raan_deg': 120., 'phases_deg': [0.]}],
            'targets': [{'id': f't{i:02d}', 'lat_deg': 0., 'lon_deg': float(lon)}
                        for i, lon in enumerate(range(-180, 180, 30))],
            'stations': [{'name': f'g{i:02d}', 'lat_deg': 0., 'lon_deg': float(lon)}
                         for i, lon in enumerate(range(-180, 180, 30))],
            'min_fulfilled': 1, 'min_acquired': 1}
    shifted = copy.deepcopy(base)
    shifted.update(id='shifted-northern', min_fulfilled=0, min_acquired=0)
    for plane in shifted['planes']:
        plane['raan_deg'] += 35.
        plane['phases_deg'] = [40.]
    for row in shifted['targets']:
        row['lat_deg'] = 30.
        row['lon_deg'] += 7.5
    southern = copy.deepcopy(base)
    southern.update(id='southern-misaligned-ground', min_fulfilled=0, min_acquired=0)
    for plane in southern['planes']:
        plane['phases_deg'] = [220.]
    for row in southern['targets']:
        row['lat_deg'] = -30.
        row['lon_deg'] += 12.
    for row in southern['stations']:
        row['lat_deg'] = 45.
    polar = copy.deepcopy(base)
    polar.update(id='polar-no-opportunity', min_fulfilled=0, min_acquired=0)
    polar['targets'] = [{'id': 'pole', 'lat_deg': 89., 'lon_deg': 0.}]
    result = {'id': 'coupled-study', 'development': base,
            'holdouts': [shifted, southern, polar], 'workers': 2,
            'require_temporal_stability': False}
    if fixed_clock:
        result.update(id='fixed-clock-study', require_event_stability=True)
        for case in [base, shifted, southern, polar]:
            case['controller_clock'] = 'fixed-tick/1'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--write-spec', action='store_true')
    parser.add_argument('--fixed-clock', action='store_true')
    args = parser.parse_args()
    if args.write_spec:
        atomic_write(args.output, canonical_json(configuration(args.fixed_clock)))
    else:
        result = run_mission_study(configuration(args.fixed_clock), args.output)
        print(canonical_json({'sha256': result['sha256'], 'assessment': result['assessment']}))
        raise SystemExit(0 if result['assessment']['passed'] else 1)
