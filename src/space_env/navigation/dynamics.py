"""Declared-force estimator inputs and nominal mass; never a native state provider."""

import math
from dataclasses import replace
from itertools import pairwise

import numpy as np

from space_env.core.propulsion import G0_M_S2
from space_env.navigation.models import predict
from space_env.reference.covariance import state_jacobian
from space_env.reference.frames import FrameContext
from space_env.reference.orbit import acceleration
from space_env.reference.time import EOPSeries, Epoch


class DeclaredDynamics:
    def __init__(self, config, actor):
        orbit = config.orbits[actor]
        self.model = orbit.force_model()
        self.epoch = Epoch.utc(orbit.initial.epoch_utc)
        self.context = FrameContext(
            EOPSeries(**config.eop.model_dump(exclude={"interpolation"}))
        )
        self.propulsion = orbit.propulsion
        self.perturbed = any(
            getattr(self.model, k) for k in ["j2", "drag", "srp", "third_body"]
        )

    def nominal_mass(self, time, impulses, seen, burns):
        """Chronological propellant bookkeeping from configuration and known controls."""
        exhaust = self.propulsion.isp_s * G0_M_S2
        selected = [
            r
            for r in impulses
            if r["command"]["command_id"] in seen and r["start_sec"] <= time
        ]
        times = {0.0, float(time)}
        times.update(r["start_sec"] for r in selected)
        for b in burns.values():
            times.update(t for t in [b["start"], b["end"]] if 0 <= t <= time)
        m = self.propulsion.dry_mass_kg + self.propulsion.propellant_kg
        clock = 0.0
        for right in sorted(times):
            midpoint = (clock + right) / 2
            flow = sum(
                b["thrust_n"] / exhaust
                for b in burns.values()
                if b["start"] <= midpoint < b["end"]
            )
            m -= flow * (right - clock)
            for row in selected:
                if row["start_sec"] == right:
                    m *= math.exp(
                        -math.hypot(*row["command"]["impulse"]["delta_v_m_s"]) / exhaust
                    )
            clock = right
        if not math.isfinite(m) or m < self.propulsion.dry_mass_kg - 1e-9:
            raise ValueError("declared estimator mass exceeds propellant envelope")
        return m

    def propagate(self, x, p, start, end, psd, impulses, seen, burns):
        if end <= start:
            return predict(x, p, 0.0, psd)
        cuts = sorted(
            {
                start,
                end,
                *[
                    t
                    for b in burns.values()
                    for t in [b["start"], b["end"]]
                    if start < t < end
                ],
            }
        )
        for left, right in pairwise(cuts):
            active = [
                b for b in burns.values() if b["start"] <= (left + right) / 2 < b["end"]
            ]
            if len(active) > 1:
                raise ValueError("overlapping finite burns unsupported by estimator")
            if not self.perturbed and not active:
                x, p = predict(x, p, right - left, psd)
                continue
            force = (
                np.zeros(3)
                if not active
                else active[0]["thrust_n"] * np.asarray(active[0]["direction_gcrs"])
            )

            def dynamics(t, state, left=left, force=force):
                clock = left + t
                mass = self.nominal_mass(clock, impulses, seen, burns)
                model = replace(self.model, mass_kg=mass)
                epoch = self.epoch.shifted(clock)
                value = acceleration(state, epoch, self.context, model) + force / mass
                jacobian = state_jacobian(state, epoch, self.context, model)
                return value, jacobian

            x, p = predict(x, p, right - left, psd, dynamics=dynamics)
        return x, p
