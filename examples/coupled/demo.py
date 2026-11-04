"""Fixed geographic inputs; all actor/target requests remain in denominator."""
import argparse
from pathlib import Path

from space_env.scaling.benchmark import run_scale
from space_env.scaling.coupled import coupled_workload
from space_env.simulation.snapshot import canonical_json


def configuration():
    return {'id': 'coupled-nominal',
            'planes': [{'id': 'a', 'inclination_deg': 55., 'raan_deg': 0., 'phases_deg': [0.]},
                       {'id': 'b', 'inclination_deg': 75., 'raan_deg': 120., 'phases_deg': [0.]}],
            'targets': [{'id': f't{i:02d}', 'lat_deg': 0., 'lon_deg': float(lon)}
                        for i, lon in enumerate(range(-180, 180, 30))],
            'stations': [{'name': f'g{i:02d}', 'lat_deg': 0., 'lon_deg': float(lon)}
                         for i, lon in enumerate(range(-180, 180, 30))]}


def manifest():
    workloads = []
    for mode in ['nominal', 'weak_wheels', 'all_outage']:
        cfg = configuration() | {'id': 'coupled-' + mode, 'mode': mode,
                                 'min_fulfilled': 1 if mode == 'nominal' else 0,
                                 'min_acquired': 0 if mode == 'weak_wheels' else 1}
        for restore in [False, True]:
            w = coupled_workload(cfg)
            w['comparison_group'] = cfg['id']
            if restore:
                w.update(id=w['id'] + '-restored', resume_after_decisions=20,
                         resume_in_new_process=True)
            workloads.append(w)
    return {'id': 'independent-coupled', 'workers': 1, 'workloads': workloads}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    value = manifest()
    result = run_scale(value, args.output)
    print(canonical_json(result))
    raise SystemExit(0 if result['ok'] else 1)
