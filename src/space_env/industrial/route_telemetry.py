"""Decision-boundary samples with deterministic route-channel transport delay.

No hidden-time interpolation: only actually captured snapshots can be delivered.
Queries are pure; checkpoint replay reconstructs the same bounded sample queue.
"""

import copy

from pydantic import Field

from space_env.runtime.schema import FiniteScalar, StrictModel


class RouteTelemetrySpec(StrictModel):
    delay_sec: FiniteScalar = Field(default=0.0, ge=0, le=3600)


class RouteTelemetry:
    def __init__(self, spec):
        self.spec = RouteTelemetrySpec.model_validate(spec)
        self.samples = []

    def capture(self, now, routes):
        if self.samples and now < self.samples[-1]["measured_at"]:
            raise ValueError("route sample time must be monotonic")
        sample = {"measured_at": now, "routes": copy.deepcopy(routes or [])}
        if self.samples and now == self.samples[-1]["measured_at"]:
            self.samples[-1] = sample
        else:
            self.samples.append(sample)
        # Keep the latest delivered snapshot and all still-in-transit samples.
        eligible = [
            i
            for i, s in enumerate(self.samples)
            if s["measured_at"] + self.spec.delay_sec <= now + 1e-9
        ]
        if eligible:
            self.samples = self.samples[eligible[-1] :]

    def observation(self, now, visible_ids):
        delivered = [
            s
            for s in self.samples
            if s["measured_at"] + self.spec.delay_sec <= now + 1e-9
        ]
        sample = delivered[-1] if delivered else None
        pending = [
            s["measured_at"] + self.spec.delay_sec
            for s in self.samples
            if s["measured_at"] + self.spec.delay_sec > now + 1e-9
        ]
        age = None if sample is None else max(0.0, now - sample["measured_at"])
        routes = (
            []
            if sample is None
            else [
                {
                    **copy.deepcopy(r),
                    "age_sec": age,
                    "delivered_at": sample["measured_at"] + self.spec.delay_sec,
                }
                for r in sample["routes"]
                if r["action_id"] in visible_ids
            ]
        )
        return routes, {
            "model": "decision-boundary-route-channel/1",
            "delay_sec": self.spec.delay_sec,
            "measured_at": None if sample is None else sample["measured_at"],
            "age_sec": age,
            "next_delivery_sec": min(pending) if pending else None,
            "scope": "route channel only; native catalogue, masks and operational feedback remain current",
        }
