"""
Движок симуляции: шаг по времени, пропагация, питание, термика, связь, события.
Команды исполняются по наблюдению на начало шага, затем время и орбита сдвигаются.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from space_env.core.access import off_nadir_deg
from space_env.core.attitude import align_z, slew_step
from space_env.core.encounter import PropagationContext, secondary_from_geometry
from space_env.core.environment import in_eclipse, shadow_state, solar_flux_at_satellite_W_m2
from space_env.core.frames import eci_to_ecef, elevation_deg, geodetic_to_ecef
from space_env.core.ground import DEFAULT_GROUND_STATIONS, GroundStation
from space_env.core.link import link_margin_db, stepped_rate_bps
from space_env.core.passes import PassWindow, find_passes
from space_env.core.orbit import Orbit
from space_env.core.power import solar_generation_w, step_battery_soc
from space_env.core.propagate import propagate_state
from space_env.core.relative import mean_motion_rad_s
from space_env.core.spacecraft import Spacecraft, SubsystemHealth
from space_env.core.sun import sun_position_m
from space_env.core.thermal import environmental_heat_w, step_temperature_c
from space_env.core.time import Time
from space_env.mission.constraints import Constraints
from space_env.mission.events import EventKind, MissionEvent, PlannedEvent
from space_env.mission.goals import MissionGoal
from space_env.mission.scenarios import AnomalyInject, ConjunctionInject, ScenarioConfig
from space_env.simulation.state import SimulationState

_LOAD_FACTOR = {
    "nominal": 1.0,
    "reduced": 0.6,
    "minimal": 0.3,
    "safe": 0.25,
}


@dataclass
class _Restore:
    at_sec: float
    subsystem: str
    done: bool = False


@dataclass
class SimulationEngine:
    """
    Пошаговая симуляция одного КА.
    step(dt, command) применяет команду, пропагирует орбиту и обновляет подсистемы.
    """

    orbit: Orbit
    spacecraft: Spacecraft
    mission_goal: MissionGoal = field(default_factory=MissionGoal)
    constraints: Constraints = field(default_factory=Constraints)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._time_sec = 0.0
        self._epoch = self.orbit.epoch
        self._log: list[tuple[float, SimulationState, list[MissionEvent]]] = []
        self._imaging_count = 0
        self._comms_sessions = 0
        self._data_downlink_Mb = 0.0
        self._load_mode = "nominal"
        self._fired_anomaly: set[int] = set()
        self._fired_conjunction: set[int] = set()
        self._fired_planned: set[int] = set()
        self._restores: list[_Restore] = []
        self._encounter = None
        self._active_violations: set[str] = set()
        self._faults: set[str] = set()
        self._radio_recovery_steps = 0
        self._downlink_armed = False
        self._imaging_requested = False
        self._captured_targets: set[str] = set()
        self._prev_soc = 1.0
        self._soc_rate = 0.0
        self._station_lat = 0.0
        self._station_lon = 0.0
        self._station_alt = 0.0
        self._link_margin_db = -99.0
        self._slew_remaining_deg = 0.0
        self._slew_span_deg = 1.0
        self._attitude = align_z(-self.orbit.r / np.linalg.norm(self.orbit.r))
        self._passes: list[PassWindow] = []
        self.last_command_ok = True
        self._in_umbra = False
        self._shadow_kind = "sun"
        self._flux = 0.0
        self._in_contact = False
        self._elevation = -90.0
        self._station_name = ""
        self.stations: list[GroundStation] = (
            list(DEFAULT_GROUND_STATIONS)
            if self.scenario.ground_stations is None
            else list(self.scenario.ground_stations)
        )
        self._apply_scenario_initial()
        self._refresh_environment()
        init_events: list[MissionEvent] = []
        self._prev_soc = self.spacecraft.state.power.battery_soc
        self._refresh_passes()
        self._fire_injects(0.0, 0.0, init_events, include_zero=True)
        self._log.append((0.0, self._snapshot(), init_events))

    def _apply_scenario_initial(self) -> None:
        if self.scenario.initial_battery_soc is not None:
            self.spacecraft.state.power.battery_soc = self.scenario.initial_battery_soc
        if self.scenario.initial_delta_v_fraction is not None:
            total = self.spacecraft.config.propulsion.delta_v_total_m_s
            self.spacecraft.state.propulsion.delta_v_remaining_m_s = total * self.scenario.initial_delta_v_fraction

    def _now(self) -> Time:
        return self._epoch + self._time_sec

    def _health_factor(self, subsystem: str) -> float:
        health = self.spacecraft.state.health.get(subsystem, SubsystemHealth.NOMINAL)
        value = health.value if isinstance(health, SubsystemHealth) else str(health)
        if value == SubsystemHealth.FAILED.value:
            return 0.0
        if value == SubsystemHealth.DEGRADED.value:
            return 0.5
        return 1.0

    def _set_health(self, subsystem: str, severity: str) -> None:
        kind = SubsystemHealth.FAILED if severity == "failed" else SubsystemHealth.DEGRADED
        self.spacecraft.state.health[subsystem] = kind

    def _refresh_environment(self) -> None:
        r_sun = sun_position_m(self._now())
        r = self.orbit.r
        shadow = shadow_state(r, r_sun)
        self._shadow_kind = shadow.kind
        self._in_umbra = shadow.kind == "umbra"
        self._flux = solar_flux_at_satellite_W_m2(r, r_sun)
        best_el = -90.0
        best_station: GroundStation | None = None
        best_margin = -99.0
        contact_station: GroundStation | None = None
        contact_el = -90.0
        contact_margin = -99.0
        for station in self.stations:
            el = elevation_deg(r, station.lat_deg, station.lon_deg, station.alt_m, self._now())
            sat = eci_to_ecef(r, self._now())
            slant = float(np.linalg.norm(sat - geodetic_to_ecef(station.lat_deg, station.lon_deg, station.alt_m)))
            margin = self._margin_db(slant)
            closes = el >= station.min_elevation_deg and margin >= self.spacecraft.config.comms.min_margin_dB
            if closes and margin > contact_margin:
                contact_station = station
                contact_el = el
                contact_margin = margin
            if el > best_el:
                best_el = el
                best_station = station
                best_margin = margin
        chosen = contact_station or best_station
        self._elevation = contact_el if contact_station is not None else best_el
        self._link_margin_db = contact_margin if contact_station is not None else best_margin
        self._station_name = chosen.name if chosen is not None else ""
        if chosen is not None:
            self._station_lat = chosen.lat_deg
            self._station_lon = chosen.lon_deg
            self._station_alt = chosen.alt_m
        in_contact = contact_station is not None and self.radio_ready()
        was_contact = self._in_contact
        self._in_contact = in_contact
        self.spacecraft.state.comms.in_contact = in_contact
        self.spacecraft.state.power.in_sun = not self._in_umbra
        if was_contact and not in_contact:
            self.spacecraft.state.comms.data_sent_this_pass_Mb = 0.0
            self._downlink_armed = False

    def _command_pointing(self, mode: str) -> None:
        if mode == self.spacecraft.state.pointing and self._slew_remaining_deg <= 0.0:
            return
        if mode != self.spacecraft.state.pointing:
            self._slew_span_deg = max(self._boresight_separation_deg(), 1.0)
            self._slew_remaining_deg = self._slew_span_deg
        self.spacecraft.state.pointing = mode

    def _boresight_separation_deg(self) -> float:
        radius = np.linalg.norm(self.orbit.r)
        nadir = -self.orbit.r / radius
        sun = sun_position_m(self._now())
        sun = sun / np.linalg.norm(sun)
        cosine = float(np.clip(np.dot(nadir, sun), -1.0, 1.0))
        return float(np.rad2deg(np.arccos(cosine)))

    def _target_boresight(self) -> np.ndarray:
        if self.spacecraft.state.pointing == "sun":
            sun = sun_position_m(self._now())
            return sun / np.linalg.norm(sun)
        return -self.orbit.r / np.linalg.norm(self.orbit.r)

    def boresight_eci(self) -> np.ndarray:
        """Куда сейчас смотрит +Z прибора, единичный вектор в ECI."""
        from space_env.core.attitude import boresight

        return boresight(self._attitude)

    def _advance_slew(self, dt_sec: float) -> None:
        target = align_z(self._target_boresight())
        self._attitude, self._slew_remaining_deg = slew_step(
            self._attitude,
            target,
            self.spacecraft.config.slew_rate_deg_s,
            dt_sec,
        )

    def _panel_cosine(self) -> float:
        """1 на Солнце, косинус зенита в надире, между ними — доля незаконченного разворота."""
        sun = sun_position_m(self._now())
        sun = sun / np.linalg.norm(sun)
        zenith = self.orbit.r / np.linalg.norm(self.orbit.r)
        nadir_cos = max(0.0, float(np.dot(zenith, sun)))
        target_cos = 1.0 if self.spacecraft.state.pointing == "sun" else nadir_cos
        if self._slew_remaining_deg <= 0.0:
            return target_cos
        origin_cos = nadir_cos if self.spacecraft.state.pointing == "sun" else 1.0
        done = 1.0 - self._slew_remaining_deg / max(self._slew_span_deg, 1.0)
        return origin_cos + done * (target_cos - origin_cos)

    def _event(self, kind: EventKind, payload: dict | None = None) -> MissionEvent:
        return MissionEvent(kind=kind, time=self._now(), time_sec=self._time_sec, payload=payload or {})

    def _apply_command(
        self,
        command: str | None,
        payload: dict,
        events: list[MissionEvent],
        dt_sec: float,
    ) -> None:
        self.last_command_ok = True
        if self.spacecraft.state.payload_mode == "imaging" and command != "imaging":
            self.spacecraft.state.payload_mode = self.spacecraft.config.payload.default_mode
        if command is None or command == "wait":
            return
        if command == "nominal_power":
            self._load_mode = "nominal"
            return
        if command == "reduced_power":
            self._load_mode = "reduced"
            self._faults.discard("overload")
            return
        if command == "minimal_power":
            self._load_mode = "minimal"
            self._faults.discard("overload")
            return
        if command == "safe_mode":
            self._load_mode = "safe"
            self._command_pointing("sun")
            self.spacecraft.state.payload_mode = "idle"
            self._faults.discard("overload")
            self._faults.discard("stuck_heater")
            return
        if command == "point_sun":
            self._command_pointing("sun")
            return
        if command == "point_nadir":
            self._command_pointing("nadir")
            return
        if command == "reboot":
            # Перезагрузка БЦВМ и радиолинии занимает несколько шагов и не лечит шину и нагреватель.
            self._radio_recovery_steps = 3
            self._command_pointing("nadir")
            events.append(self._event(EventKind.RECOVERY, {"action": "reboot", "steps_remaining": 3}))
            return
        if command == "imaging":
            self._command_imaging(events)
            return
        if command == "comms":
            self._command_comms(events, dt_sec)
            return
        if command == "maneuver":
            dv = float(payload.get("dv_m_s", 5.0))
            direction = str(payload.get("direction", "prograde"))
            self._command_maneuver(dv, direction, events)
            return
        self.last_command_ok = False

    def _command_imaging(self, events: list[MissionEvent]) -> None:
        del events
        payload_ok = self._health_factor("payload") > 0.0
        nadir = self.spacecraft.state.pointing == "nadir"
        if not payload_ok or not nadir or not self.scenario.ground_targets:
            self.last_command_ok = False
            return
        self._imaging_requested = True
        self.spacecraft.state.payload_mode = "imaging"

    def _command_comms(self, events: list[MissionEvent], dt_sec: float) -> None:
        del dt_sec
        if not self._in_contact or not self._link_ok():
            self.last_command_ok = False
            return
        if not self._downlink_armed:
            self._comms_sessions += 1
            self._downlink_armed = True
            events.append(
                self._event(
                    EventKind.COMMS_PASS,
                    {"station": self._station_name, "sessions": self._comms_sessions},
                )
            )

    def _command_maneuver(self, dv_m_s: float, direction: str, events: list[MissionEvent]) -> None:
        if dv_m_s <= 0.0 or self._health_factor("propulsion") <= 0.0:
            self.last_command_ok = False
            return
        remaining = self.spacecraft.state.propulsion.delta_v_remaining_m_s
        if dv_m_s > remaining + 1e-6:
            self.last_command_ok = False
            return
        r = self.orbit.r
        v = self.orbit.v
        speed = float(np.linalg.norm(v))
        if speed < 1e-6:
            self.last_command_ok = False
            return
        vhat = v / speed
        if direction == "retrograde":
            axis = -vhat
        elif direction == "normal":
            h = np.cross(r, v)
            h_norm = float(np.linalg.norm(h))
            axis = h / h_norm if h_norm > 0 else vhat
        elif direction == "radial":
            axis = r / np.linalg.norm(r)
        else:
            axis = vhat
        dv_vec = axis * dv_m_s
        # Второе тело численное: импульс меняет только свой аппарат, относительная скорость выходит сама.
        self.orbit = self.orbit.with_delta_v(dv_vec)
        self.spacecraft.state.propulsion.delta_v_remaining_m_s = remaining - dv_m_s
        events.append(self._event(EventKind.MANEUVER, {"dv_m_s": dv_m_s, "direction": direction}))
        self._refresh_passes()

    def _fire_injects(
        self,
        t0: float,
        t1: float,
        events: list[MissionEvent],
        *,
        include_zero: bool,
    ) -> None:
        for index, inj in enumerate(self.scenario.anomaly_injects):
            if index in self._fired_anomaly:
                continue
            if not _is_due(inj.at_sec, t0, t1, include_zero=include_zero):
                continue
            self._fired_anomaly.add(index)
            self._set_health(inj.subsystem, inj.severity)
            events.append(
                self._event(EventKind.ANOMALY, {"subsystem": inj.subsystem, "severity": inj.severity})
            )
            if inj.duration_sec is not None:
                self._restores.append(_Restore(inj.at_sec + inj.duration_sec, inj.subsystem))
        for index, inj in enumerate(self.scenario.conjunction_injects):
            if index in self._fired_conjunction:
                continue
            if not _is_due(inj.at_sec, t0, t1, include_zero=include_zero):
                continue
            self._fired_conjunction.add(index)
            self._start_conjunction(inj)
            events.append(
                self._event(
                    EventKind.CONJUNCTION_WARNING,
                    {
                        "miss_distance_km": inj.miss_distance_km,
                        "probability": inj.probability,
                    },
                )
            )

    def arm_fault(self, name: str) -> None:
        """Процедурный отказ: overload, stuck_heater или radio_latch. В наблюдение не копируется."""
        self._faults.add(name)

    def faults_active(self) -> bool:
        return bool(self._faults) or self._radio_recovery_steps > 0

    def radio_ready(self) -> bool:
        if self._radio_recovery_steps > 0 or "radio_latch" in self._faults:
            return False
        return self._health_factor("comms") > 0.0

    def _link_ok(self) -> bool:
        return self.radio_ready() and self._in_contact

    def _propagation_context(
        self,
        *,
        mass_kg: float,
        drag_area_m2: float,
        drag_coefficient: float,
        srp_area_m2: float = 0.0,
        srp_coefficient: float = 1.2,
    ) -> PropagationContext:
        return PropagationContext(
            mass_kg=mass_kg,
            drag_area_m2=drag_area_m2,
            drag_coefficient=drag_coefficient,
            atmosphere=self.scenario.atmosphere,
            f107=self.scenario.f107,
            f107a=self.scenario.f107a,
            ap=self.scenario.ap,
            density_scale=self.scenario.density_scale,
            lat_deg=self._station_lat,
            lon_deg=self._station_lon,
            when=self._now().datetime.replace(tzinfo=None),
            srp_area_m2=srp_area_m2,
            srp_coefficient=srp_coefficient,
            include_third_body=True,
        )

    def _primary_context(self) -> PropagationContext:
        prop = self.spacecraft.config.propulsion
        return self._propagation_context(
            mass_kg=prop.mass_kg,
            drag_area_m2=prop.drag_area_m2,
            drag_coefficient=prop.drag_coefficient,
            srp_area_m2=prop.srp_area_m2,
            srp_coefficient=prop.srp_coefficient,
        )

    def _start_conjunction(self, inj: ConjunctionInject) -> None:
        if inj.miss_radial_m is None and inj.miss_cross_m is None:
            radial_m = inj.miss_distance_km * 1000.0
            cross_m = 0.0
        else:
            radial_m = 0.0 if inj.miss_radial_m is None else inj.miss_radial_m
            cross_m = 0.0 if inj.miss_cross_m is None else inj.miss_cross_m
        debris = self._propagation_context(mass_kg=50.0, drag_area_m2=0.2, drag_coefficient=2.2)
        self._encounter = secondary_from_geometry(
            self.orbit.r,
            self.orbit.v,
            radial_m,
            cross_m,
            inj.approach_m_s,
            inj.tca_in_sec,
            inj.combined_sigma_m,
            self._time_sec + inj.tca_in_sec,
            inj.hard_body_radius_m,
            debris,
        )

    def use_tracking_covariance(self, covariance_rtn_m2: np.ndarray) -> None:
        """Подменяет ковариацию сближения оценкой дуги. Промах остаётся от геометрии инжекта."""
        if self._encounter is None:
            raise RuntimeError("сближение не начато")
        matrix = np.asarray(covariance_rtn_m2, dtype=float)
        if matrix.shape != (6, 6):
            raise ValueError("ковариация дуги должна быть 6×6")
        self._encounter.covariance = 0.5 * (matrix + matrix.T)

    def _advance_encounter(self, dt_sec: float) -> None:
        enc = self._encounter
        if enc is None:
            return
        n = mean_motion_rad_s(float(np.linalg.norm(self.orbit.r)))
        enc.advance(dt_sec, n)
        if self._time_sec >= enc.tca_sec + 30.0:
            enc.passed = True

    def _run_planned(self, t0: float, t1: float, events: list[MissionEvent], dt_sec: float) -> None:
        for index, planned in enumerate(self.scenario.planned_events):
            if index in self._fired_planned:
                continue
            if not (t0 < planned.scheduled_at_sec <= t1):
                continue
            self._fired_planned.add(index)
            self._execute_planned(planned, events, dt_sec)

    def _execute_planned(self, planned: PlannedEvent, events: list[MissionEvent], dt_sec: float) -> None:
        if planned.kind == EventKind.IMAGING:
            self._command_imaging(events)
        elif planned.kind == EventKind.COMMS_PASS:
            self._command_comms(events, dt_sec)
        elif planned.kind == EventKind.MANEUVER:
            dv = float(planned.payload.get("dv_m_s", 1.0))
            direction = str(planned.payload.get("direction", "prograde"))
            self._command_maneuver(dv, direction, events)

    def _restore_anomalies(self, events: list[MissionEvent]) -> None:
        for item in self._restores:
            if item.done or self._time_sec < item.at_sec:
                continue
            item.done = True
            self.spacecraft.state.health[item.subsystem] = SubsystemHealth.NOMINAL
            events.append(self._event(EventKind.RECOVERY, {"subsystem": item.subsystem}))

    def _update_subsystems(self, dt_sec: float, flux_w_m2: float) -> None:
        cfg_p = self.spacecraft.config.power
        st_p = self.spacecraft.state.power
        cfg_t = self.spacecraft.config.thermal
        st_t = self.spacecraft.state.thermal
        load = _LOAD_FACTOR[self._load_mode]
        base_w = cfg_p.consumption_nominal_W * load
        heater_w = 0.0
        thermal_factor = self._health_factor("thermal")
        if (
            thermal_factor > 0.0
            and st_p.battery_soc > 0.05
            and st_t.temp_C < cfg_t.temp_min_C + 5.0
            and cfg_t.heater_power_W > 0.0
        ):
            heater_w = cfg_t.heater_power_W * thermal_factor
            st_t.heater_on = True
        else:
            st_t.heater_on = False
        consumption = base_w + heater_w + (450.0 if "overload" in self._faults else 0.0)
        st_p.consumption_W = consumption
        generation = solar_generation_w(
            cfg_p.solar_panel_area_m2,
            flux_w_m2,
            cfg_p.panel_efficiency,
            self._health_factor("power") * self._panel_cosine(),
        )
        st_p.battery_soc = step_battery_soc(
            st_p.battery_soc,
            cfg_p.battery_capacity_Wh,
            consumption,
            generation,
            dt_sec,
        )
        q_solar, q_albedo, q_ir = environmental_heat_w(
            r_sat_m=self.orbit.r,
            r_sun_m=sun_position_m(self._now()),
            solar_flux_w_m2=flux_w_m2,
            absorptivity=cfg_t.absorptivity,
            emissivity=cfg_t.emissivity,
            solar_area_m2=cfg_t.projected_area_m2,
            earth_area_m2=cfg_t.earth_facing_area_m2,
        )
        q_fault = 250.0 if self._health_factor("thermal") <= 0.0 else 0.0
        if "stuck_heater" in self._faults:
            q_fault += 180.0
        st_t.temp_C = step_temperature_c(
            st_t.temp_C,
            dt_sec,
            q_sun_w=q_solar + q_albedo + q_ir,
            q_diss_w=consumption + q_fault,
            q_heater_w=heater_w,
            radiating_area_m2=cfg_t.radiating_area_m2,
            emissivity=cfg_t.emissivity,
            heat_capacity_j_k=cfg_t.heat_capacity_J_K,
        )

    def _emit_edges(self, was_umbra: bool, was_contact: bool, events: list[MissionEvent]) -> None:
        if not was_umbra and self._in_umbra:
            events.append(self._event(EventKind.ECLIPSE_START))
        elif was_umbra and not self._in_umbra:
            events.append(self._event(EventKind.ECLIPSE_END))
        if not was_contact and self._in_contact:
            events.append(self._event(EventKind.AOS, {"station": self._station_name, "elevation_deg": self._elevation}))
        elif was_contact and not self._in_contact:
            events.append(self._event(EventKind.LOS, {"station": self._station_name}))

    def _check_constraints(self, events: list[MissionEvent]) -> None:
        st = self.spacecraft.state
        used = self.spacecraft.config.propulsion.delta_v_total_m_s - st.propulsion.delta_v_remaining_m_s
        current: dict[str, float] = {}
        if st.power.battery_soc < self.constraints.battery_soc_min:
            current["battery_soc"] = st.power.battery_soc
        if self.constraints.temp_min_C is not None and st.thermal.temp_C < self.constraints.temp_min_C:
            current["temp_min"] = st.thermal.temp_C
        if self.constraints.temp_max_C is not None and st.thermal.temp_C > self.constraints.temp_max_C:
            current["temp_max"] = st.thermal.temp_C
        if self.constraints.delta_v_max_m_s is not None and used > self.constraints.delta_v_max_m_s:
            current["delta_v"] = used
        for key, value in current.items():
            if key not in self._active_violations:
                events.append(self._event(EventKind.CONSTRAINT_VIOLATION, {"constraint": key, "value": value}))
        self._active_violations = set(current)

    def _conjunction_view(self) -> tuple[float | None, float | None, bool, float | None, float | None]:
        enc = self._encounter
        if enc is None:
            return None, None, False, None, None
        if enc.passed:
            return (
                enc.frozen_miss_m / 1000.0,
                enc.frozen_pc,
                True,
                enc.frozen_radial_m,
                enc.frozen_cross_m,
            )
        horizon = max(60.0, enc.tca_sec - self._time_sec + 120.0)
        distance, probability, radial, cross = enc.predict(self.orbit.r, self.orbit.v, self._primary_context(), horizon)
        if self._time_sec >= enc.tca_sec:
            enc.passed = True
            enc.frozen_miss_m = distance
            enc.frozen_pc = probability
            enc.frozen_radial_m = radial
            enc.frozen_cross_m = cross
        return distance / 1000.0, probability, enc.passed, radial, cross

    def _refresh_passes(self) -> None:
        if not self.stations:
            self._passes = []
            return
        prop = self.spacecraft.config.propulsion
        comms = self.spacecraft.config.comms
        raw = find_passes(
            self.orbit.r,
            self.orbit.v,
            self._now(),
            self.stations,
            horizon_sec=6.0 * 3600.0,
            mass_kg=prop.mass_kg,
            drag_area_m2=prop.drag_area_m2,
            drag_coefficient=prop.drag_coefficient,
            frequency_hz=comms.frequency_Hz,
            tx_power_w=comms.tx_power_W,
            tx_gain_dbi=comms.tx_gain_dBi,
            rx_gain_dbi=comms.rx_gain_dBi,
            system_loss_db=comms.system_loss_dB,
            required_power_dbw=comms.required_power_dBW,
            min_margin_db=comms.min_margin_dB,
            atmosphere=self.scenario.atmosphere,
            f107=self.scenario.f107,
            density_scale=self.scenario.density_scale,
            include_third_body=True,
        )
        origin = self._time_sec
        self._passes = [
            PassWindow(item.station, origin + item.aos_sec, origin + item.los_sec, item.max_elevation_deg)
            for item in raw
        ]

    def seconds_to_next_aos(self) -> float | None:
        """Секунды до следующего AOS. 0, если сеанс уже идёт. None, если окна нет на горизонте 6 ч."""
        if self._in_contact:
            return 0.0
        for window in self._passes:
            if window.aos_sec <= self._time_sec < window.los_sec:
                return 0.0
            if window.aos_sec > self._time_sec:
                return window.aos_sec - self._time_sec
        return None

    def _margin_db(self, range_m: float) -> float:
        cfg = self.spacecraft.config.comms
        return link_margin_db(
            range_m,
            frequency_hz=cfg.frequency_Hz,
            tx_power_w=cfg.tx_power_W,
            tx_gain_dbi=cfg.tx_gain_dBi,
            rx_gain_dbi=cfg.rx_gain_dBi,
            system_loss_db=cfg.system_loss_dB,
            required_power_dbw=cfg.required_power_dBW,
        )

    def _accrue_downlink(self, dt_sec: float) -> None:
        if not self._downlink_armed or not self._link_ok():
            return
        rate = stepped_rate_bps(self._link_margin_db, self.spacecraft.config.comms.data_rate_max_bps)
        mbit = rate * dt_sec / 1e6
        self._data_downlink_Mb += mbit
        self.spacecraft.state.comms.data_sent_this_pass_Mb += mbit

    def _resolve_imaging(self, samples: list[tuple[float, np.ndarray, np.ndarray]], t0: float, events: list[MissionEvent]) -> None:
        if not self._imaging_requested:
            return
        self._imaging_requested = False
        fov = self.spacecraft.config.payload.half_fov_deg
        captured_now = False
        if self.spacecraft.state.pointing != "nadir" or self._slew_remaining_deg > 1.0:
            self.last_command_ok = False
            return
        for offset, position, _velocity in samples:
            moment = self._epoch + (t0 + offset)
            sun = sun_position_m(moment)
            if in_eclipse(position, sun):
                continue
            for target in self.scenario.ground_targets:
                if target.name in self._captured_targets:
                    continue
                if off_nadir_deg(position, target.lat_deg, target.lon_deg, moment) <= fov:
                    self._captured_targets.add(target.name)
                    self._imaging_count += 1
                    captured_now = True
                    events.append(self._event(EventKind.IMAGING, {"target": target.name, "count": self._imaging_count}))
        if not captured_now:
            self.last_command_ok = False

    def _snapshot(self) -> SimulationState:
        st = self.spacecraft.state
        miss_km, prob, tca, radial_m, cross_m = self._conjunction_view()
        total_dv = self.spacecraft.config.propulsion.delta_v_total_m_s
        voltage = 24.0 + 8.0 * st.power.battery_soc
        if "overload" in self._faults:
            voltage -= 3.0
        if self._in_umbra:
            voltage -= 1.5
        return SimulationState(
            time_sec=self._time_sec,
            altitude_km=self.orbit.altitude_km(),
            in_eclipse=self._in_umbra,
            shadow=self._shadow_kind,
            battery_soc=st.power.battery_soc,
            delta_v_remaining_m_s=st.propulsion.delta_v_remaining_m_s,
            temp_C=st.thermal.temp_C,
            heater_on=st.thermal.heater_on,
            imaging_count=self._imaging_count,
            comms_sessions=self._comms_sessions,
            data_downlink_Mb=self._data_downlink_Mb,
            delta_v_used_m_s=total_dv - st.propulsion.delta_v_remaining_m_s,
            in_contact=self._in_contact,
            elevation_deg=self._elevation,
            contact_station=self._station_name,
            payload_mode=st.payload_mode,
            pointing=st.pointing,
            conjunction_miss_km=miss_km,
            conjunction_probability=prob,
            conjunction_tca_passed=tca,
            link_ok=self._link_ok(),
            bus_voltage_V=voltage,
            soc_rate_per_s=self._soc_rate,
            miss_radial_m=radial_m,
            miss_cross_m=cross_m,
            link_margin_db=self._link_margin_db,
            slew_remaining_deg=self._slew_remaining_deg,
            seconds_to_aos=self.seconds_to_next_aos(),
        )

    def step(
        self,
        dt_sec: float,
        command: str | None = None,
        command_payload: dict | None = None,
    ) -> tuple[SimulationState, list[MissionEvent]]:
        """Один шаг. command — действие агента или плана на начало шага."""
        if dt_sec <= 0.0:
            raise ValueError("dt_sec должен быть положительным")
        events: list[MissionEvent] = []
        was_umbra = self._in_umbra
        was_contact = self._in_contact
        flux_start = self._flux
        self._apply_command(command, command_payload or {}, events, dt_sec)
        prop = self.spacecraft.config.propulsion
        r_end, v_end, samples = propagate_state(
            self.orbit.r,
            self.orbit.v,
            dt_sec,
            mass_kg=prop.mass_kg,
            drag_area_m2=prop.drag_area_m2,
            drag_coefficient=prop.drag_coefficient,
            atmosphere=self.scenario.atmosphere,
            f107=self.scenario.f107,
            f107a=self.scenario.f107a,
            ap=self.scenario.ap,
            density_scale=self.scenario.density_scale,
            lat_deg=self._station_lat,
            lon_deg=self._station_lon,
            when=self._now().datetime.replace(tzinfo=None),
            srp_area_m2=prop.srp_area_m2,
            srp_coefficient=prop.srp_coefficient,
            include_third_body=True,
        )
        self._advance_slew(dt_sec)
        prev_t = self._time_sec
        self._time_sec += dt_sec
        self.orbit = Orbit.from_vectors(r_end, v_end, self._epoch + self._time_sec)
        self._refresh_environment()
        self._advance_encounter(dt_sec)
        self._fire_injects(prev_t, self._time_sec, events, include_zero=False)
        self._run_planned(prev_t, self._time_sec, events, dt_sec)
        self._resolve_imaging(samples, prev_t, events)
        self._restore_anomalies(events)
        if self._radio_recovery_steps > 0:
            self._radio_recovery_steps -= 1
            if self._radio_recovery_steps == 0:
                self._faults.discard("radio_latch")
        self._update_subsystems(dt_sec, 0.5 * (flux_start + self._flux))
        self._accrue_downlink(dt_sec)
        self._soc_rate = (self.spacecraft.state.power.battery_soc - self._prev_soc) / dt_sec
        self._prev_soc = self.spacecraft.state.power.battery_soc
        self._emit_edges(was_umbra, was_contact, events)
        self._check_constraints(events)
        snap = self._snapshot()
        self._log.append((self._time_sec, snap, events))
        return snap, events

    def get_state_snapshot(self) -> SimulationState:
        """Текущий снимок без шага."""
        return self._snapshot()

    @property
    def time_sec(self) -> float:
        return self._time_sec

    @property
    def log(self) -> list[tuple[float, SimulationState, list[MissionEvent]]]:
        return self._log


def _is_due(at_sec: float, t0: float, t1: float, *, include_zero: bool) -> bool:
    if include_zero and at_sec == 0.0 and t0 == 0.0 and t1 == 0.0:
        return True
    return t0 < at_sec <= t1
