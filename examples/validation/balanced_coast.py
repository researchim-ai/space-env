"""Frozen dense-coast comparison; call counts are deterministic, timings are not."""
import argparse
import math
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

from space_env.reference.frames import FrameContext
from space_env.reference.orbit import ForceModel,ReferenceState
from space_env.reference.providers import DynamicStateProvider
from space_env.reference.thrust import BalancedCoupleThrust
from space_env.reference.time import EOPSeries,Epoch
from space_env.simulation.snapshot import atomic_write,canonical_json,digest


def qualify():
    import space_env.reference.providers as module
    result={};samples={}
    grid=np.linspace(0.,60.,601)
    law=BalancedCoupleThrust(.04,60.,20.,(0.,0.,0.),(.02,0.,0.),(.5,.5,.5))
    for label,opt in [('eager',False),('reuse',True)]:
        p=DynamicStateProvider(ReferenceState(Epoch.utc('2026-07-18T12:00:00Z'),
            [7000000.,0.,0.],[0.,7500.,0.]),60.,FrameContext(EOPSeries.packaged()),
            model=ForceModel(mass_kg=20.1,j2=True),dry_mass_kg=20.,isp_s=200.,
            integration={'sample_sec':1.,'reuse_balanced_coast':opt})
        before=p.mass(0.);changed=[];clock=1.
        with patch.object(module,'propagate',wraps=module.propagate) as spy:
            start=time.perf_counter()
            for _ in range(20):
                changed.append(p.burn(clock,.025,law));clock+=.025
            elapsed=time.perf_counter()-start
        samples[label]=np.array([p.state(float(t)) for t in grid])
        result[label]={'propagation_calls':spy.call_count,'wall_sec':elapsed,
            'geometry_changed_segments':sum(changed),'maneuvers':len(p.maneuvers),
            'fuel_used_kg':before-p.mass(60.),'trajectory_sha256':p.sha256}
    delta=samples['eager']-samples['reuse']
    errors={'position_m':float(np.max(np.linalg.norm(delta[:,:3],axis=1))),
            'velocity_m_s':float(np.max(np.linalg.norm(delta[:,3:],axis=1))),
            'fuel_kg':abs(result['eager']['fuel_used_kg']-result['reuse']['fuel_used_kg'])}
    budgets={'position_m':.01,'velocity_m_s':1e-5,'fuel_kg':1e-10}
    passed=(all(math.isfinite(v) and v<=budgets[k] for k,v in errors.items())
            and result['eager']['propagation_calls']==40 and result['reuse']['propagation_calls']==0
            and result['reuse']['geometry_changed_segments']==0
            and result['reuse']['fuel_used_kg']>0)
    body={'schema':'space-env.balanced-coast-program/1','passed':passed,'modes':result,
          'max_errors':errors,'budgets':budgets,'sample_sec':1.,'duration_sec':60.,
          'burn_segments':20,'segment_sec':.025,'comparison_samples':601,
          'scope':'fixed dense mass-independent trajectory and scalar propellant ledger; not rigid attitude, nozzle calibration or universal interpolation accuracy'}
    return body|{'sha256':digest(body)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
    if a.output.exists():raise ValueError('output already exists')
    result=qualify();atomic_write(a.output,canonical_json(result))
    print(canonical_json({'passed':result['passed'],'sha256':result['sha256']}))
    raise SystemExit(0 if result['passed'] else 1)
