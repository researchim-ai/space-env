"""Common fixed/dynamic trajectory surface; discontinuities never cross a spline."""

from __future__ import annotations

import bisect
from dataclasses import replace
from typing import Protocol, runtime_checkable

import numpy as np

from space_env.core.propulsion import G0_M_S2
from space_env.reference.access import ReferenceEphemeris
from space_env.reference.covariance import CovarianceArc
from space_env.reference.orbit import ForceModel, ReferenceState, propagate
from space_env.reference.thrust import BalancedCoupleThrust
from space_env.simulation.snapshot import digest


@runtime_checkable
class StateProvider(Protocol):
    epoch: object
    context: object
    duration_sec: float
    sha256: str

    def state(self, clock: float) -> np.ndarray: ...
    def look(self, station, clock: float, *, frequency_hz=None) -> dict: ...
    def access(self, station, *, scan_sec=30.0, root_tolerance_sec=0.001, start_sec=0.0) -> dict: ...


class _CoastSlice:
    """Owned time coordinates over an immutable, already propagated coast arc."""

    def __init__(self, parent, offset, duration):
        if isinstance(parent, _CoastSlice):
            offset += parent.offset
            parent = parent.parent
        self.parent, self.offset = parent, float(offset)
        self.epoch = parent.epoch.shifted(offset)
        self.duration_sec = float(duration)
        self.max_interval_sec = parent.max_interval_sec
        self.sha256 = digest({'coast_sha256':parent.sha256,'offset_sec':self.offset,
                              'duration_sec':self.duration_sec})

    def state(self, clock):
        if not 0 <= clock <= self.duration_sec:
            raise ValueError('coast slice query outside coverage')
        return self.parent.state(min(self.parent.duration_sec,self.offset+clock))


class DynamicStateProvider(ReferenceEphemeris):
    """Piecewise sealed arcs, right-continuous impulse state, preserved past arcs.

    Future coasts are forecasts conditional on accepted commands. Each maneuver
    truncates only the future. The base access algorithm calls this provider's
    state; it therefore never interpolates across an impulsive velocity jump.
    """

    def __init__(
        self,
        initial,
        horizon_sec,
        context,
        *,
        model=None,
        integration=None,
        dry_mass_kg,
        isp_s,
        navigation=None,
    ):
        self.epoch, self.context = initial.epoch, context
        self.duration_sec = float(horizon_sec)
        self.model = model or ForceModel()
        self.integration = dict(integration or {})
        self.reuse_balanced_coast = self.integration.pop('reuse_balanced_coast',False)
        if type(self.reuse_balanced_coast) is not bool or (
            self.reuse_balanced_coast and self.integration.get('sample_sec',30.) > 1.
        ):
            raise ValueError('balanced coast reuse requires boolean opt-in and sample_sec <= 1')
        self.dry_mass_kg, self.isp_s = float(dry_mass_kg), float(isp_s)
        if not 0 < self.dry_mass_kg <= self.model.mass_kg or self.isp_s <= 0:
            raise ValueError("invalid dynamic mass/propulsion configuration")
        self.segments = []
        self.navigation = navigation
        self.navigation_arcs = []
        self.maneuvers = []
        self.revision = 0
        self.initial_payload = self._append(0.0, initial, self.model.mass_kg)
        self._seal()

    def _append(
        self,
        start,
        initial,
        mass,
        *,
        end=None,
        thrust=None,
        covariance=None,
        origin_stm=None,
    ):
        end = self.duration_sec if end is None else end
        if end <= start:
            return
        result = propagate(
            initial,
            end - start,
            self.context,
            model=replace(self.model, mass_kg=mass),
            thrust=thrust,
            **self.integration,
        )
        arc = ReferenceEphemeris.from_result(result)
        self.segments.append((float(start), arc, float(mass), thrust))
        if self.navigation is not None:
            navigation_arc = CovarianceArc(
                arc,
                self.navigation,
                model=replace(self.model, mass_kg=mass),
                thrust=thrust,
                initial_covariance=covariance,
                origin_stm=origin_stm,
            )
            self.navigation_arcs.append((float(start), navigation_arc))
        return {**result, "sha256": arc.sha256}

    def _seal(self):
        self.sha256 = digest(
            {
                "arcs": [(s, a.sha256, m) for s, a, m, _ in self.segments],
                "maneuvers": self.maneuvers,
                "revision": self.revision,
                "navigation": [(s, arc.sha256) for s, arc in self.navigation_arcs],
            }
        )
        self.max_interval_sec = max(a.max_interval_sec for _, a, _, _ in self.segments)

    def _segment(self, clock):
        if (
            isinstance(clock, bool)
            or not np.isfinite(clock)
            or not 0 <= clock <= self.duration_sec
        ):
            raise ValueError("dynamic state query outside coverage")
        index = bisect.bisect_right([row[0] for row in self.segments], clock) - 1
        return self.segments[index]

    def state(self, clock):
        start, arc, _, _ = self._segment(clock)
        return arc.state(clock - start)

    def mass(self, clock):
        start, _, mass, thrust = self._segment(clock)
        return thrust.mass(mass, clock - start) if thrust else mass

    def _cut(self, clock):
        # Keep past segments including the pre-impulse endpoint. The new segment
        # at clock is selected by bisect_right; old times retain their old arc.
        self.segments = [row for row in self.segments if row[0] < clock]
        self.navigation_arcs = [row for row in self.navigation_arcs if row[0] < clock]

    def _navigation_arc(self, clock):
        self._segment(clock)  # Common coverage/type validation.
        if self.navigation is None:
            raise ValueError("trajectory has no covariance propagation capability")
        index = bisect.bisect_right([row[0] for row in self.navigation_arcs], clock) - 1
        return self.navigation_arcs[index]

    def covariance(self, clock):
        start, arc = self._navigation_arc(clock)
        return arc.covariance(clock - start)

    def stm(self, clock):
        start, arc = self._navigation_arc(clock)
        return arc.stm(clock - start)

    def _navigation_state(self, clock):
        if self.navigation is None:
            return {}
        return {"covariance": self.covariance(clock), "origin_stm": self.stm(clock)}

    def impulse(self, clock, delta_v):
        if not 0 <= clock < self.duration_sec:
            raise ValueError("impulse requires future trajectory coverage")
        state, mass = self.state(clock), self.mass(clock)
        dv = np.asarray(delta_v, dtype=float)
        if dv.shape != (3,) or not np.isfinite(dv).all():
            raise ValueError("impulse requires finite GCRS delta-v")
        new_mass = mass * np.exp(-np.linalg.norm(dv) / (self.isp_s * G0_M_S2))
        if new_mass < self.dry_mass_kg - 1e-10:
            raise ValueError("insufficient propellant for impulse")
        initial = ReferenceState(self.epoch.shifted(clock), state[:3], state[3:] + dv)
        uncertainty = self._navigation_state(clock)
        if uncertainty:
            uncertainty["covariance"][3:, 3:] += np.asarray(
                self.navigation.impulse_velocity_cov_m2_s2
            )
        self._cut(clock)
        self._append(clock, initial, new_mass, **uncertainty)
        self.revision += 1
        self.maneuvers.append(
            {
                "kind": "impulse",
                "time_sec": clock,
                "delta_v_m_s": dv.tolist(),
                "mass_before_kg": mass,
                "mass_after_kg": float(new_mass),
            }
        )
        self._seal()

    def burn(self, clock, duration, thrust):
        """Apply burn; return whether translational geometry must be refreshed.

        Only exact built-in balanced couples on mass-independent gravity/coasts
        reuse sealed position/velocity. Mass, maneuver history and identity change.
        Drag, SRP, navigation or an already active/future burn use the full path.
        """
        state, mass = self.state(clock), self.mass(clock)
        end = clock + duration
        if (
            not 0 <= clock < end <= self.duration_sec
            or duration > thrust.fuel_seconds(mass) + 1e-9
        ):
            raise ValueError("finite burn exceeds time or propellant budget")
        # Propagation covers the representable endpoint interval, which can be
        # a few ulps shorter than the requested duration. Never query beyond the
        # resulting sealed arc or charge a different interval to its mass law.
        span = end-clock
        active = self._segment(clock)
        mass_only = (self.reuse_balanced_coast and type(thrust) is BalancedCoupleThrust and not self.model.drag
                     and not self.model.srp and self.navigation is None
                     and active[3] is None
                     and all(row[3] is None for row in self.segments if row[0] > clock))
        uncertainty = self._navigation_state(clock)
        self._cut(clock)
        if mass_only:
            start, coast, _, _ = active
            self.segments.append((float(clock),_CoastSlice(coast,clock-start,span),mass,thrust))
            final_mass = thrust.mass(mass,span)
            if end < self.duration_sec:
                self.segments.append((float(end),_CoastSlice(coast,end-start,self.duration_sec-end),final_mass,None))
        else:
            self._append(
                clock,
                ReferenceState(self.epoch.shifted(clock), state[:3], state[3:]),
                mass,
                end=end,
                thrust=thrust,
                **uncertainty,
            )
            burn_arc = self.segments[-1][1]
            final = burn_arc.state(span)
            final_mass = thrust.mass(mass, span)
            uncertainty = self._navigation_state(end)
            self._append(
                end,
                ReferenceState(self.epoch.shifted(end), final[:3], final[3:]),
                final_mass,
                **uncertainty,
            )
        self.revision += 1
        self.maneuvers.append(
            {
                "kind": "finite_burn",
                "time_sec": clock,
                "end_sec": end,
                "thrust_n": thrust.thrust_n,
                "direction_gcrs": list(thrust.direction_gcrs),
                "mass_before_kg": mass,
                "planned_mass_after_kg": final_mass,
                **(
                    {
                        "direction_model": thrust.manifest()["direction_model"],
                        "attitude_path_sha256": digest(thrust.manifest()),
                    }
                    if "direction_model" in thrust.manifest()
                    else {}
                ),
            }
        )
        self._seal()
        return not mass_only

    def stop_burn(self, clock):
        if clock >= self.duration_sec:
            return
        state, mass = self.state(clock), self.mass(clock)
        uncertainty = self._navigation_state(clock)
        self._cut(clock)
        self._append(
            clock,
            ReferenceState(self.epoch.shifted(clock), state[:3], state[3:]),
            mass,
            **uncertainty,
        )
        self.revision += 1
        self.maneuvers.append(
            {"kind": "burn_cancelled", "time_sec": clock, "mass_kg": mass}
        )
        self._seal()
