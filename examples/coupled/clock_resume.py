"""Separate-process restoration strictly between controller ticks, not an SLA."""
import argparse
import json
from pathlib import Path

from space_env.scaling.benchmark import run_scale
from space_env.scaling.coupled import coupled_workload
from space_env.simulation.snapshot import canonical_json


def manifest(configuration):
    c = configuration['development'] | {
        'id': 'clock-midtick', 'controller_clock': 'fixed-tick/1',
        'control_step_sec': .1, 'decision_sec': 1.03, 'horizon_sec': 30.,
        'min_fulfilled': 0, 'min_acquired': 0,
    }
    w = coupled_workload(c) | {'sample_numerics': True, 'comparison_group': 'clock-midtick'}
    restored = w | {'id': 'clock-midtick-restored', 'resume_after_decisions': 5,
                    'resume_in_new_process': True}
    return {'id': 'fixed-clock-midtick-restart', 'workers': 1, 'workloads': [w,restored]}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('spec',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    with args.spec.open('rb') as stream:
        raw=stream.read(2_000_001)
    if len(raw)>2_000_000:
        raise ValueError('input exceeds 2MB')
    result=run_scale(manifest(json.loads(raw)),args.output)
    print(canonical_json({'passed':result['ok'],'sha256':result['sha256']}))
    raise SystemExit(0 if result['ok'] else 1)
