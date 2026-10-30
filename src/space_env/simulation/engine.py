"""
Движок симуляции: шаг по времени, пропагация, питание, термика, связь, события.
Команды исполняются по наблюдению на начало шага, затем время и орбита сдвигаются.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np

from space_env.core.access import target_access
from space_env.core.attitude import align_z, slew_step
from space_env.core.encounter import (
    NumericalEncounter,
    PropagationContext,
    secondary_from_geometry,
)
from space_env.core.environment import (
    in_eclipse,
    shadow_state,
    solar_flux_at_satellite_W_m2,
)
from space_env.core.frames import (
    LEGACY_PHYSICS_PROFILE,
    ecef_to_geodetic,
    eci_to_ecef,
    elevation_deg,
    geodetic_to_ecef,
    require_numerical_frame,
)
from space_env.core.ground import DEFAULT_GROUND_STATIONS, GroundStation
from space_env.core.link import link_margin_db, stepped_rate_bps
from space_env.core.orbit import Orbit
from space_env.core.passes import PassWindow, find_passes
from space_env.core.power import solar_generation_w, step_battery_soc
from space_env.core.propagate import propagate_state
from space_env.core.relative import mean_motion_rad_s
from space_env.core.spacecraft import Spacecraft, SpacecraftState, SubsystemHealth
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
        require_numerical_frame(self.orbit.frame)
        self._rng = random.Random(self.seed)
        self._time_sec = 0.0
        self._epoch = self.orbit.epoch
        self._log: list[tuple[float, SimulationState, list[MissionEvent]]] = []
        self._imaging_count = 0
        self._comms_sessions = 0
        self._data_downlink_Mb = 0.0
        self._stored_bits = 0.0
        self._generated_bits = 0.0
        self._delivered_bits = 0.0
        self._lost_bits = 0.0
        self._produced_bits = 0.0
        self._initial_bits = 0.0
        self._received_bits = 0.0
        self._relayed_out_bits = 0.0
        self._products: list[dict] = []
        self._imaging_cuts: list[float] = []
        self._product_seq = 0
        self._terminal_used: list[tuple[float, float]] = []
        self._receipts: list[dict] = []
        self._payload_energy_j = 0.0
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
        self._setup_until = 0.0
        self._link_outages: list[tuple[str, float, float]] = []
        self._imaging_requested = False
        self._imaging_since: float | None = None
        self._burn: dict[str, float | str] | None = None
        self._safe_latched = False
        self._load_before_safe: str | None = None
        self._process_queue: list[dict[str, float | str]] = []
        self._looks: dict[str, list[tuple[float, float]]] = {}
        self._sar_used_sec = 0.0
        self._exposure_reason = ""
        self._symptom = ""
        self._symptom_factor = 1.0
        self._hidden_cause = ""
        self._recovery_sec = 0.0
        self._recover_until = 0.0
        self._symptom_until = 0.0
        self._false_latch = False
        self._message_miss_m: float | None = None
        self._avoid_plan: dict | None = None
        self._avoid_reason = ""
        self._keeping_block = False
        self._keeping_reason = ""
        self._pass_capacity_bits = 0.0
        self._rate_share = 1.0
        self._downlink_mask: list[tuple[float, float]] | None = None
        self._phase_plan: dict | None = None
        self._phase_reason = ""
        self._captured_targets: set[str] = set()
        self._prev_soc = 1.0
        self._soc_rate = 0.0
        self._station_lat = 0.0
        self._station_lon = 0.0
        self._station_alt = 0.0
        self._link_margin_db = -99.0
        self._slew_remaining_deg = 0.0
        self._slew_rate_deg_s = 0.0
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

    @property
    def physics_profile(self) -> str:
        """Explicitly identify historical frame/time and mission-force conventions."""
        return LEGACY_PHYSICS_PROFILE

    def now(self) -> Time:
        """Время миссии: эпоха старта плюс секунды симуляции."""
        return self._epoch + self._time_sec

    def captured_targets(self) -> frozenset[str]:
        """Имена целей, уже засчитанных съёмкой."""
        return frozenset(self._captured_targets)

    def clear_captured_targets(self) -> None:
        """Сбрасывает засчитанные цели. Нужно, когда сценарий подменяет точки съёмки."""
        self._captured_targets.clear()

    def _now(self) -> Time:
        return self.now()

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
            closes = (
                el >= station.min_elevation_deg
                and margin >= self.spacecraft.config.comms.min_margin_dB
                and self._band_matches(station)
            )
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
            self._setup_until = 0.0

    def _command_pointing(self, mode: str) -> None:
        if mode == self.spacecraft.state.pointing and self._slew_remaining_deg <= 0.0:
            return
        if mode != self.spacecraft.state.pointing:
            self._slew_span_deg = max(self._boresight_separation_deg(), 1.0)
            self._slew_remaining_deg = self._slew_span_deg
            self._slew_rate_deg_s = 0.0
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
        from space_env.core.attitude import slew_limited

        target = align_z(self._target_boresight())
        accel = float(self.spacecraft.config.slew_accel_deg_s2)
        if accel <= 0.0:
            self._attitude, self._slew_remaining_deg = slew_step(
                self._attitude,
                target,
                self.spacecraft.config.slew_rate_deg_s,
                dt_sec,
            )
            self._slew_rate_deg_s = 0.0 if self._slew_remaining_deg <= 1e-6 else float(self.spacecraft.config.slew_rate_deg_s)
            return
        self._attitude, self._slew_remaining_deg, self._slew_rate_deg_s = slew_limited(
            self._attitude,
            target,
            self._slew_rate_deg_s,
            self.spacecraft.config.slew_rate_deg_s,
            accel,
            dt_sec,
        )

    def _cosine_at(self, position: np.ndarray, moment: Time) -> float:
        """1 при наведении на Солнце, косинус зенита в надире, между ними — доля разворота."""
        sun = sun_position_m(moment)
        sun_norm = float(np.linalg.norm(sun))
        radius = float(np.linalg.norm(position))
        if sun_norm < 1.0 or radius < 1.0:
            return 0.0
        nadir_cos = max(0.0, float(np.dot(position / radius, sun / sun_norm)))
        target_cos = 1.0 if self.spacecraft.state.pointing == "sun" else nadir_cos
        if self._slew_remaining_deg <= 0.0:
            return target_cos
        origin_cos = nadir_cos if self.spacecraft.state.pointing == "sun" else 1.0
        done = 1.0 - self._slew_remaining_deg / max(self._slew_span_deg, 1.0)
        return origin_cos + done * (target_cos - origin_cos)

    def _panel_cosine(self) -> float:
        """Косинус в текущем положении. Шаг питания считает его на каждом отсчёте."""
        return self._cosine_at(self.orbit.r, self._now())

    def _event(self, kind: EventKind, payload: dict | None = None) -> MissionEvent:
        return self._event_at(kind, self._time_sec, payload)

    def _event_at(self, kind: EventKind, time_sec: float, payload: dict | None = None) -> MissionEvent:
        return MissionEvent(kind=kind, time=self._epoch + time_sec, time_sec=time_sec, payload=payload or {})

    def cancel_imaging(self) -> None:
        """Обрывает съёмку, которая уже идёт. Уже записанный продукт остаётся."""
        self._imaging_cuts.append(self._time_sec)
        self._imaging_requested = False

    def _apply_command(
        self,
        command: str | None,
        payload: dict,
        events: list[MissionEvent],
        dt_sec: float,
    ) -> None:
        self.last_command_ok = True
        if command != "imaging":
            self._imaging_since = None
        if self.spacecraft.state.payload_mode == "imaging" and command != "imaging":
            self.spacecraft.state.payload_mode = self.spacecraft.config.payload.default_mode
        if command is None or command == "wait":
            return
        if self._safe_latched and command not in {"safe_mode", "point_sun", "recover"}:
            self.last_command_ok = False
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
            self._burn = None
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
        if command == "avoid":
            self._command_avoid(payload, events)
            return
        if command == "phase":
            self._command_phase(payload, events)
            return
        if command == "recover":
            self._begin_recovery(events)
            return
        self.last_command_ok = False

    def _command_imaging(self, events: list[MissionEvent]) -> None:
        del events
        if self._safe_latched or self.spacecraft.state.power.battery_soc <= 0.0:
            self.last_command_ok = False
            self._imaging_since = None
            return
        if self.spacecraft.config.payload.exclusive_with_downlink and self._downlink_armed:
            self.last_command_ok = False
            self._imaging_since = None
            return
        payload_ok = self._health_factor("payload") > 0.0
        nadir = self.spacecraft.state.pointing == "nadir"
        if not payload_ok or not nadir or not self.scenario.ground_targets:
            self.last_command_ok = False
            self._imaging_since = None
            return
        if self._imaging_since is None:
            self._imaging_since = self._time_sec
        self._imaging_requested = True
        self.spacecraft.state.payload_mode = "imaging"

    def _downlink_overlaps(self, dt_sec: float) -> bool:
        """План съёмки пересекает шаг. Команда съёмки проверяется отдельно, по флагу передатчика."""
        origin = self._time_sec
        end = origin + dt_sec
        for planned in self.scenario.planned_events:
            if planned.kind != EventKind.IMAGING:
                continue
            start = float(planned.scheduled_at_sec)
            stop = start + float(planned.duration_sec)
            if planned.duration_sec <= 0.0:
                if origin - 1e-9 <= start < end:
                    return True
                continue
            if stop > origin and start < end:
                return True
        return False

    def _command_comms(self, events: list[MissionEvent], dt_sec: float) -> None:
        if self._safe_latched or self.spacecraft.state.power.battery_soc <= 0.0:
            self.last_command_ok = False
            return
        if self.spacecraft.config.payload.exclusive_with_downlink and self._downlink_overlaps(dt_sec):
            self.last_command_ok = False
            return
        if not self._in_contact or not self._link_ok():
            self.last_command_ok = False
            return
        if not self._downlink_armed:
            self._comms_sessions += 1
            self._downlink_armed = True
            self._setup_until = self._time_sec + float(self.spacecraft.config.comms.setup_sec)
            events.append(
                self._event(
                    EventKind.COMMS_PASS,
                    {"station": self._station_name, "sessions": self._comms_sessions},
                )
            )

    def _propulsion_profile(self) -> str:
        prop = self.spacecraft.config.propulsion
        touched = prop.thrust_N > 0.0 or prop.isp_s > 0.0 or prop.dry_mass_kg > 0.0
        if not touched:
            return "impulsive"
        if prop.thrust_N > 0.0 and prop.isp_s > 0.0 and 0.0 < prop.dry_mass_kg < prop.mass_kg:
            return "finite"
        return "invalid"

    def _command_maneuver(self, dv_m_s: float, direction: str, events: list[MissionEvent]) -> None:
        if self._safe_latched or dv_m_s <= 0.0 or self._health_factor("propulsion") <= 0.0:
            self.last_command_ok = False
            return
        if self._burn is not None:
            self.last_command_ok = False
            return
        remaining = self.spacecraft.state.propulsion.delta_v_remaining_m_s
        if dv_m_s > remaining + 1e-6:
            self.last_command_ok = False
            return
        profile = self._propulsion_profile()
        if profile == "invalid":
            self.last_command_ok = False
            return
        if profile == "finite":
            from space_env.core.propulsion import exhaust_speed_m_s

            prop = self.spacecraft.config.propulsion
            ceiling = exhaust_speed_m_s(prop.isp_s) * math.log(prop.mass_kg / prop.dry_mass_kg)
            if dv_m_s > ceiling + 1e-6:
                self.last_command_ok = False
                return
            self._burn = {"direction": direction, "dv_left": float(dv_m_s)}
            events.append(
                self._event(
                    EventKind.MANEUVER,
                    {"dv_m_s": dv_m_s, "direction": direction, "profile": "finite"},
                )
            )
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

    def _arm_symptom(self, inj) -> None:
        """Симптом виден в отчёте. Скрытая причина в отчёт не входит и health не переписывает."""
        self._symptom = str(inj.symptom)
        self._symptom_factor = 0.25 if inj.symptom == "rate" else 1.0
        self._hidden_cause = str(inj.cause) if bool(inj.hidden) else ""
        self._recovery_sec = float(inj.recovery_sec)
        self._recover_until = 0.0
        if inj.duration_sec is not None:
            self._symptom_until = float(inj.at_sec) + float(inj.duration_sec)
        if inj.symptom == "false_safe":
            self._false_latch = True
            self._safe_latched = True

    def symptom_report(self) -> dict[str, str | float | bool]:
        """То, что можно показать агенту. Поля причины здесь нет."""
        return {
            "symptom": self._symptom,
            "rate_factor": self._symptom_factor if self._symptom == "rate" else 1.0,
            "false_safe": bool(self._false_latch and self._safe_latched),
        }

    def _begin_recovery(self, events: list[MissionEvent]) -> None:
        if not self._symptom:
            self.last_command_ok = False
            return
        if self._recovery_sec <= 0.0:
            self._clear_symptom(events)
            return
        self._recover_until = self._time_sec + self._recovery_sec
        events.append(self._event(EventKind.RECOVERY, {"action": "recover_started", "until_sec": self._recover_until}))

    def _tick_symptom(self, events: list[MissionEvent]) -> None:
        if not self._symptom:
            return
        recover_due = self._recover_until > 0.0 and self._time_sec + 1e-9 >= self._recover_until
        timer_due = self._symptom_until > 0.0 and self._time_sec + 1e-9 >= self._symptom_until
        if recover_due or timer_due:
            self._clear_symptom(events)

    def _clear_symptom(self, events: list[MissionEvent]) -> None:
        self._symptom = ""
        self._symptom_factor = 1.0
        self._hidden_cause = ""
        self._recover_until = 0.0
        self._symptom_until = 0.0
        if self._false_latch:
            self._false_latch = False
            power = self.spacecraft.config.power
            if float(power.safe_soc_enter) == 0.0 and float(power.safe_soc_exit) == 0.0:
                self._safe_latched = False
        events.append(self._event(EventKind.RECOVERY, {"action": "symptom_cleared"}))

    def _command_avoid(self, payload: dict, events: list[MissionEvent]) -> None:
        """Решение по промаху сообщения. Принятый импульс нормальный и тратит бюджет миссии."""
        from space_env.mission.avoidance import avoidance_plan

        if self._encounter is None or self._message_miss_m is None:
            self.last_command_ok = False
            self._avoid_reason = "нет сближения"
            self._avoid_plan = None
            return
        miss = float(self._message_miss_m)
        plan = avoidance_plan(
            miss,
            hard_body_radius_m=float(payload.get("hard_body_radius_m", self._encounter.hard_body_m)),
            sigma_m=float(payload.get("sigma_m", 50.0)),
            dv_budget_m_s=float(self.spacecraft.state.propulsion.delta_v_remaining_m_s),
            mission_reserve_m_s=float(payload.get("mission_reserve_m_s", 0.0)),
            target_miss_m=float(payload.get("target_miss_m", miss + 1_000.0)),
            dv_per_m=float(payload.get("dv_per_m", 0.001)),
        )
        self._avoid_plan = plan
        if not plan["accepted"]:
            self.last_command_ok = False
            self._avoid_reason = str(plan["reason"])
            return
        before_km, _probability, _passed, _radial, _cross = self._conjunction_view()
        self._command_maneuver(float(plan["dv_m_s"]), "normal", events)
        if not self.last_command_ok:
            self._avoid_reason = "манёвр не принят"
            plan["accepted"] = False
            return
        self._avoid_reason = ""
        plan["numerical_miss_before_km"] = before_km

    def _command_phase(self, payload: dict, events: list[MissionEvent]) -> None:
        """Вековой уход вдоль трека. Отказ линейной модели импульс не тратит."""
        from space_env.mission.phasing import phase_plan

        radius = float(np.linalg.norm(self.orbit.r))
        plan = phase_plan(
            float(payload.get("along_error_m", 0.0)),
            radius_m=radius,
            horizon_sec=float(payload.get("horizon_sec", 600.0)),
            dv_budget_m_s=float(self.spacecraft.state.propulsion.delta_v_remaining_m_s),
            mission_reserve_m_s=float(payload.get("mission_reserve_m_s", 0.0)),
        )
        self._phase_plan = plan
        if not plan["accepted"]:
            self.last_command_ok = False
            self._phase_reason = str(plan["reason"])
            return
        self._command_maneuver(abs(float(plan["dv_m_s"])), str(plan["direction"]), events)
        if not self.last_command_ok:
            self._phase_reason = "манёвр не принят"
            plan["accepted"] = False
            return
        self._phase_reason = ""

    def commit_drag_makeup(self, horizon_sec: float, burns: int, mission_reserve_m_s: float) -> dict:
        """Ставит включения компенсации, если после резерва импульс ещё помещается."""
        from space_env.core.constants import R_EARTH_EQUATOR
        from space_env.mission.stationkeep import drag_makeup_schedule

        prop = self.spacecraft.config.propulsion
        altitude = float(np.linalg.norm(self.orbit.r) - R_EARTH_EQUATOR)
        plan = drag_makeup_schedule(
            altitude,
            mass_kg=prop.mass_kg,
            area_m2=prop.drag_area_m2,
            drag_coefficient=prop.drag_coefficient,
            horizon_sec=horizon_sec,
            burns=burns,
            dv_budget_m_s=float(self.spacecraft.state.propulsion.delta_v_remaining_m_s),
            mission_reserve_m_s=mission_reserve_m_s,
            f107=float(self.scenario.f107),
        )
        self._keeping_reason = str(plan["reason"])
        if not plan["accepted"]:
            return plan
        for slot in plan["burns"]:
            self.scenario.planned_events.append(
                PlannedEvent(
                    kind=EventKind.MANEUVER,
                    scheduled_at_sec=float(slot["at_sec"]),
                    duration_sec=0.0,
                    payload={
                        "dv_m_s": float(slot["dv_m_s"]),
                        "direction": str(slot["direction"]),
                        "purpose": "drag_makeup",
                    },
                )
            )
        return plan

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
            if inj.symptom:
                self._arm_symptom(inj)
            else:
                self._set_health(inj.subsystem, inj.severity)
            events.append(
                self._event(EventKind.ANOMALY, {"subsystem": inj.subsystem, "severity": inj.severity, "symptom": inj.symptom})
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

    def _body_latlon(self) -> tuple[float, float]:
        """Геодезическая точка аппарата. Для атмосферы, не координаты станции."""
        lat, lon, _alt = ecef_to_geodetic(eci_to_ecef(self.orbit.r, self._now()))
        return lat, lon

    def _propagation_context(
        self,
        *,
        mass_kg: float,
        drag_area_m2: float,
        drag_coefficient: float,
        srp_area_m2: float = 0.0,
        srp_coefficient: float = 1.2,
    ) -> PropagationContext:
        lat, lon = self._body_latlon()
        return PropagationContext(
            mass_kg=mass_kg,
            drag_area_m2=drag_area_m2,
            drag_coefficient=drag_coefficient,
            atmosphere=self.scenario.atmosphere,
            f107=self.scenario.f107,
            f107a=self.scenario.f107a,
            ap=self.scenario.ap,
            density_scale=self.scenario.density_scale,
            lat_deg=lat,
            lon_deg=lon,
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
        self._message_miss_m = math.hypot(radial_m, cross_m)

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

    def _apply_power_hysteresis(
        self,
        events: list[MissionEvent],
        windows: list[tuple[float, float]] | None = None,
    ) -> None:
        """Держит безопасный режим между порогами. Оба нуля оставляют прежнее поведение."""
        cfg = self.spacecraft.config.power
        enter = float(cfg.safe_soc_enter)
        leave = float(cfg.safe_soc_exit)
        if enter == 0.0 and leave == 0.0:
            return
        if not (0.0 < enter < leave <= 1.0):
            raise ValueError("порог выхода из безопасного режима должен быть выше порога входа")
        soc = float(self.spacecraft.state.power.battery_soc)
        if self._safe_latched:
            if soc + 1e-12 < leave:
                return
            self._safe_latched = False
            if self._load_before_safe is not None:
                self._load_mode = self._load_before_safe
            self._load_before_safe = None
            events.append(self._event(EventKind.RECOVERY, {"action": "power_safe_cleared", "soc": soc}))
            return
        if soc > enter + 1e-12:
            return
        self._load_before_safe = self._load_mode
        self._safe_latched = True
        self._load_mode = "safe"
        self._burn = None
        self._downlink_armed = False
        self._setup_until = 0.0
        self._imaging_since = None
        self._imaging_requested = False
        self._command_pointing("sun")
        self.spacecraft.state.payload_mode = "idle"
        if windows is not None:
            windows.clear()
        events.append(self._event(EventKind.CONSTRAINT_VIOLATION, {"action": "power_safe", "soc": soc}))

    def _run_planned(self, t0: float, t1: float, events: list[MissionEvent], dt_sec: float) -> None:
        for index, planned in enumerate(self.scenario.planned_events):
            if index in self._fired_planned:
                continue
            if not _is_due(planned.scheduled_at_sec, t0, t1, include_zero=(t0 == 0.0 and t1 == t0)):
                continue
            self._fired_planned.add(index)
            if planned.kind == EventKind.IMAGING:
                continue
            before = len(events)
            self._execute_planned(planned, events, max(dt_sec, 0.0))
            if len(events) > before:
                stamped = events[-1]
                events[-1] = self._event_at(stamped.kind, planned.scheduled_at_sec, stamped.payload)

    def _execute_planned(self, planned: PlannedEvent, events: list[MissionEvent], dt_sec: float) -> None:
        if planned.kind == EventKind.IMAGING:
            self._command_imaging(events)
        elif planned.kind == EventKind.COMMS_PASS:
            self._command_comms(events, dt_sec)
        elif planned.kind == EventKind.MANEUVER:
            if str(planned.payload.get("purpose", "")) == "drag_makeup":
                self._keeping_block = True
            dv = float(planned.payload.get("dv_m_s", 1.0))
            direction = str(planned.payload.get("direction", "prograde"))
            self._command_maneuver(dv, direction, events)

    def _restore_anomalies(self, events: list[MissionEvent]) -> None:
        for item in self._restores:
            if item.done or self._time_sec < item.at_sec:
                continue
            item.done = True
            self.spacecraft.state.health[item.subsystem] = SubsystemHealth.NOMINAL
            events.append(self._event_at(EventKind.RECOVERY, item.at_sec, {"subsystem": item.subsystem}))

    def _flux_of(self, position: np.ndarray, moment: Time) -> float:
        return solar_flux_at_satellite_W_m2(position, sun_position_m(moment))

    def _side_loads(self, position: np.ndarray, moment: Time) -> tuple[float, float]:
        """Выработка батарей и внешний теплоприток в одной точке, Вт."""
        cfg_p = self.spacecraft.config.power
        cfg_t = self.spacecraft.config.thermal
        flux = self._flux_of(position, moment)
        generation = solar_generation_w(
            cfg_p.solar_panel_area_m2,
            flux,
            cfg_p.panel_efficiency,
            self._health_factor("power") * self._cosine_at(position, moment),
        )
        solar, albedo, earth = environmental_heat_w(
            r_sat_m=position,
            r_sun_m=sun_position_m(moment),
            solar_flux_w_m2=flux,
            absorptivity=cfg_t.absorptivity,
            emissivity=cfg_t.emissivity,
            solar_area_m2=cfg_t.projected_area_m2,
            earth_area_m2=cfg_t.earth_facing_area_m2,
        )
        return generation, solar + albedo + earth

    def _averaged_loads(
        self,
        samples: list[tuple[float, np.ndarray, np.ndarray]],
        seg: float,
    ) -> tuple[float, float]:
        """Средние выработка и внешний нагрев. Конец шага не заменяет освещённое начало."""
        if seg <= 0.0 or len(samples) < 2:
            return self._side_loads(self.orbit.r, self._now())
        start = self._time_sec - seg
        generation = 0.0
        heat = 0.0
        span = 0.0
        for left, right in zip(samples, samples[1:]):
            dt = float(right[0] - left[0])
            if dt <= 0.0:
                continue
            left_moment = self._epoch + (start + float(left[0]))
            right_moment = self._epoch + (start + float(right[0]))
            left_generation, left_heat = self._side_loads(left[1], left_moment)
            right_generation, right_heat = self._side_loads(right[1], right_moment)
            generation += 0.5 * (left_generation + right_generation) * dt
            heat += 0.5 * (left_heat + right_heat) * dt
            span += dt
        if span <= 0.0:
            return self._side_loads(self.orbit.r, self._now())
        return generation / span, heat / span

    def _update_subsystems(
        self,
        dt_sec: float,
        samples: list[tuple[float, np.ndarray, np.ndarray]],
        *,
        imaging_sec: float = 0.0,
    ) -> None:
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
        imaging_sec = min(max(float(imaging_sec), 0.0), dt_sec)
        payload_w = 0.0
        if dt_sec > 0.0:
            payload_w = self.spacecraft.config.payload.imaging_power_w * imaging_sec / dt_sec
        self._payload_energy_j += self.spacecraft.config.payload.imaging_power_w * imaging_sec
        payload_heat = payload_w
        overload_w = 450.0 if "overload" in self._faults else 0.0
        dissipation = base_w + payload_heat + overload_w
        consumption = dissipation + heater_w
        st_p.consumption_W = consumption
        generation, environment_w = self._averaged_loads(samples, dt_sec)
        st_p.battery_soc = step_battery_soc(
            st_p.battery_soc,
            cfg_p.battery_capacity_Wh,
            consumption,
            generation,
            dt_sec,
        )
        q_fault = 250.0 if self._health_factor("thermal") <= 0.0 else 0.0
        if "stuck_heater" in self._faults:
            q_fault += 180.0
        st_t.temp_C = step_temperature_c(
            st_t.temp_C,
            dt_sec,
            q_sun_w=environment_w,
            q_diss_w=dissipation + q_fault,
            q_heater_w=heater_w,
            radiating_area_m2=cfg_t.radiating_area_m2,
            emissivity=cfg_t.emissivity,
            heat_capacity_j_k=cfg_t.heat_capacity_J_K,
        )

    def _emit_edges(
        self,
        was_umbra: bool,
        was_contact: bool,
        events: list[MissionEvent],
        samples: list[tuple[float, np.ndarray, np.ndarray]],
        origin: float,
    ) -> None:
        contact_at = self._time_sec
        contact_name = self._station_name
        for offset, position, _velocity in samples:
            opened, _margin, station = self._link_sample(position, origin + offset)
            if opened != was_contact:
                contact_at = origin + offset
                if station:
                    contact_name = station
                break
        if not was_umbra and self._in_umbra:
            events.append(self._event_at(EventKind.ECLIPSE_START, self._time_sec))
        elif was_umbra and not self._in_umbra:
            events.append(self._event_at(EventKind.ECLIPSE_END, self._time_sec))
        if not was_contact and self._in_contact:
            events.append(self._event_at(EventKind.AOS, contact_at, {"station": contact_name, "elevation_deg": self._elevation}))
        elif was_contact and not self._in_contact:
            events.append(self._event_at(EventKind.LOS, contact_at, {"station": contact_name}))

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
            f107a=self.scenario.f107a,
            ap=self.scenario.ap,
            density_scale=self.scenario.density_scale,
            include_third_body=True,
            srp_area_m2=prop.srp_area_m2,
            srp_coefficient=prop.srp_coefficient,
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

    def preload_bits(self, bits: float) -> float:
        """Кладёт уже существующий объём. Это не новая съёмка и не увеличивает число кадров."""
        return self._credit(bits, origin="preload", request_id="preload")

    def accept_bits(self, bits: float, *, origin: str, request_id: str) -> float:
        """Принимает биты от группировки: collect или relay. Съёмкой не считается."""
        return self._credit(bits, origin=origin, request_id=request_id)

    def release_bits(
        self, bits: float, *, destination: str = "",
        eligible: dict[tuple[str, float], float] | None = None,
    ) -> list[dict]:
        """Отдаёт куски продуктов соседу. Идентификатор заявки у куска не меняется."""
        if bits <= 0.0 or not math.isfinite(bits):
            return []
        left = bits
        budgets = None if eligible is None else dict(eligible)
        handed: list[dict] = []
        for product in self._products:
            if left <= 1e-9:
                break
            if product.get("delivered_to") or product.get("custodian") != self.spacecraft.config.name:
                continue
            available = float(product.get("available_time_sec", 0.0))
            if product.get("forward_held") and available > self._time_sec + 1e-9:
                continue
            onboard = float(product["onboard_bits"])
            if onboard <= 1e-9:
                continue
            key = (str(product.get("product_id", "")), float(product.get("offset_bits", 0.0)))
            take = min(onboard, left, math.inf if budgets is None else budgets.get(key, 0.0))
            if take <= 1e-9:
                continue
            if budgets is not None:
                budgets[key] -= take
            if take + 1e-9 < onboard:
                moved = dict(product)
                moved["offset_bits"] = float(product["offset_bits"]) + onboard - take
                moved["size_bits"] = take
                moved["onboard_bits"] = take
                moved["accepted_bits"] = take
                product["size_bits"] = float(product["size_bits"]) - take
                product["onboard_bits"] = onboard - take
                product["accepted_bits"] = product["onboard_bits"]
            else:
                moved = dict(product)
                product["onboard_bits"] = 0.0
                product["delivered_to"] = destination or "relay"
            moved["custodian"] = destination or moved["custodian"]
            moved["delivered_to"] = ""
            moved["origin"] = "relay"
            handed.append(moved)
            left -= take
        moved_bits = bits - left
        self._stored_bits -= moved_bits
        self._relayed_out_bits += moved_bits
        return handed

    def accept_custody(self, records: list[dict]) -> float:
        """Принимает те же куски продуктов. Новая заявка не создаётся."""
        total = 0.0
        owner = self.spacecraft.config.name
        for record in records:
            piece = dict(record)
            piece["custodian"] = owner
            piece["delivered_to"] = ""
            piece["onboard_bits"] = float(piece.get("onboard_bits", piece.get("accepted_bits", 0.0)))
            hold = float(self.spacecraft.config.comms.forward_hold_sec)
            if hold > 0.0:
                ready = self._time_sec + hold
                piece["available_time_sec"] = max(float(piece.get("available_time_sec", ready)), ready)
                piece["forward_held"] = True
            else:
                piece["available_time_sec"] = float(piece.get("available_time_sec", self._time_sec))
            self._products.append(piece)
            total += piece["onboard_bits"]
        if total <= 0.0:
            return 0.0
        self._received_bits += total
        self._generated_bits += total
        self._stored_bits += total
        return total

    def _room_bits(self) -> float:
        memory = self.spacecraft.config.payload.memory_bits
        if memory <= 0.0:
            return math.inf
        return max(0.0, memory - self._stored_bits)

    def _evict_for(self, needed_bits: float, keep: str | None) -> None:
        """Снимает целые старые продукты, пока не хватит места под новый объём."""
        while self._room_bits() + 1e-9 < needed_bits:
            victim = next(
                (
                    product
                    for product in self._products
                    if float(product.get("onboard_bits", 0.0)) > 1e-9
                    and not product.get("delivered_to")
                    and product.get("product_id") != keep
                ),
                None,
            )
            if victim is None:
                return
            onboard = float(victim["onboard_bits"])
            victim["onboard_bits"] = 0.0
            victim["delivered_to"] = "evicted"
            self._stored_bits = max(0.0, self._stored_bits - onboard)
            self._lost_bits += onboard
            victim_id = victim.get("product_id")
            self._process_queue = [item for item in self._process_queue if item.get("product_id") != victim_id]

    def _accept_volume(self, bits: float, *, keep: str | None = None) -> tuple[float, float]:
        policy = self.spacecraft.config.payload.memory_policy
        if policy == "drop_oldest":
            self._evict_for(bits, keep)
        elif policy != "reject":
            raise ValueError("неизвестная политика памяти")
        room = self._room_bits()
        added = bits if math.isinf(room) else min(bits, max(room, 0.0))
        return added, bits - added

    def _hold_for_processing(self, product: dict, added: float, ready: float) -> float:
        """Ставит принятый объём в очередь. Ноль бит/с оставляет прежний момент доступности."""
        rate = float(self.spacecraft.config.payload.process_rate_bps)
        if rate <= 0.0 or added <= 1e-12:
            return ready
        self._process_queue.append({"product_id": str(product["product_id"]), "remaining_bits": float(added)})
        return 1.0e30

    def _advance_processor(self, origin: float, dt_sec: float) -> None:
        payload = self.spacecraft.config.payload
        rate = float(payload.process_rate_bps)
        if rate <= 0.0 or dt_sec <= 0.0 or not self._process_queue:
            return
        budget = rate * dt_sec
        elapsed = 0.0
        while budget > 1e-9 and self._process_queue:
            item = self._process_queue[0]
            remaining = float(item["remaining_bits"])
            take = min(remaining, budget)
            item["remaining_bits"] = remaining - take
            budget -= take
            elapsed += take / rate
            if float(item["remaining_bits"]) <= 1e-6:
                product_id = item["product_id"]
                self._process_queue.pop(0)
                if any(queued.get("product_id") == product_id for queued in self._process_queue):
                    continue
                done_at = origin + elapsed
                for product in self._products:
                    if product.get("product_id") == product_id and not product.get("delivered_to"):
                        product["available_time_sec"] = done_at
                        break
        power = float(payload.process_power_w)
        if elapsed <= 0.0 or power <= 0.0:
            return
        capacity = self.spacecraft.config.power.battery_capacity_Wh
        state = self.spacecraft.state.power
        state.battery_soc = step_battery_soc(state.battery_soc, capacity, power, 0.0, elapsed)
        state.consumption_W += power * elapsed / dt_sec

    def _credit(self, bits: float, *, origin: str, request_id: str) -> float:
        if bits == 0.0:
            return 0.0
        if bits < 0.0 or not math.isfinite(bits):
            raise ValueError("объём бит должен быть положительным")
        added, lost = self._accept_volume(bits)
        if origin == "preload":
            self._initial_bits += bits
        elif origin == "relay":
            self._received_bits += added
            lost = 0.0
            bits = added
        else:
            self._produced_bits += bits
        self._generated_bits += bits
        self._stored_bits += added
        self._lost_bits += lost
        owner = self.spacecraft.config.name
        self._product_seq += 1
        self._products.append(
            {
                "product_id": f"{owner}-{self._product_seq}",
                "request_id": request_id,
                "offset_bits": 0.0,
                "generated_time_sec": self._time_sec,
                "available_time_sec": self._time_sec,
                "size_bits": bits,
                "accepted_bits": added,
                "onboard_bits": added,
                "quality": "full" if lost <= 1e-9 else "partial",
                "custodian": owner,
                "delivered_to": "",
                "owner": owner,
                "origin": origin,
            }
        )
        return added

    def _store_capture(self, request_id: str, when_sec: float, duration_sec: float, *, grade: str = "full", coverage: dict | None = None) -> None:
        """Пишет один продукт. Поток интегрируется по видимому времени, кадр без потока берётся один раз."""
        if grade == "none":
            return
        payload = self.spacecraft.config.payload
        streaming = payload.store_rate_bps > 0.0
        existing = next(
            (item for item in self._products if item["origin"] == "imaging" and item["request_id"] == request_id),
            None,
        )
        if not streaming:
            if existing is not None or duration_sec < 0.0:
                return
            bits = payload.image_bits
        else:
            if duration_sec <= 0.0:
                return
            bits = payload.store_rate_bps * duration_sec
        bits *= float(payload.compression_ratio)
        if bits <= 0.0:
            return
        ready = min(float(self._time_sec), float(when_sec) + max(float(duration_sec), 0.0))
        keep = None if existing is None else str(existing["product_id"])
        added, lost = self._accept_volume(bits, keep=keep)
        self._produced_bits += bits
        self._generated_bits += bits
        self._stored_bits += added
        self._lost_bits += lost
        if existing is None:
            self._captured_targets.add(request_id)
            self._imaging_count += 1
            owner = self.spacecraft.config.name
            self._product_seq += 1
            product = {
                "product_id": f"{owner}-{self._product_seq}",
                "request_id": request_id,
                "offset_bits": 0.0,
                "generated_time_sec": when_sec,
                "available_time_sec": ready,
                "size_bits": bits,
                "accepted_bits": added,
                "onboard_bits": added,
                "quality": "partial" if grade == "partial" or lost > 1e-9 else "full",
                "custodian": owner,
                "delivered_to": "",
                "owner": owner,
                "origin": "imaging",
            }
            if coverage:
                product.update(coverage)
            product["available_time_sec"] = self._hold_for_processing(product, added, ready)
            self._products.append(product)
            return
        existing["size_bits"] += bits
        existing["accepted_bits"] += added
        existing["onboard_bits"] = float(existing.get("onboard_bits", 0.0)) + added
        existing["available_time_sec"] = self._hold_for_processing(
            existing,
            added,
            max(float(existing.get("available_time_sec", when_sec)), ready),
        )
        existing["quality"] = (
            "partial"
            if grade == "partial" or existing["accepted_bits"] + 1e-6 < existing["size_bits"]
            else "full"
        )

    def _move_downlink(
        self,
        open_sec: float,
        margin_db: float,
        station: str = "",
        start_sec: float | None = None,
        end_sec: float | None = None,
    ) -> float:
        margin_db = float(margin_db) - float(self.spacecraft.config.comms.weather_loss_db)
        rate = stepped_rate_bps(margin_db, self.spacecraft.config.comms.data_rate_max_bps)
        if self._symptom == "rate":
            rate *= self._symptom_factor
        rate *= float(self._rate_share)
        rate *= self._health_factor("comms")
        if open_sec > 0.0 and margin_db >= 0.0 and rate > 0.0:
            self._pass_capacity_bits += rate * float(open_sec)
        if open_sec <= 0.0 or self._stored_bits <= 0.0 or margin_db < 0.0:
            return 0.0
        if start_sec is None or end_sec is None:
            end_sec = self._time_sec
            start_sec = end_sec - open_sec
        ready = 0.0
        owner = self.spacecraft.config.name
        for product in self._products:
            if product.get("delivered_to") or product.get("custodian") != owner:
                continue
            available = float(product.get("available_time_sec", product.get("generated_time_sec", 0.0)))
            if available > float(start_sec) + 1e-9:
                continue
            ready += float(product.get("onboard_bits", 0.0))
        moved = min(ready, rate * open_sec)
        if moved <= 0.0:
            return 0.0
        air = moved / rate if rate > 0.0 else 0.0
        self._stored_bits -= moved
        self._delivered_bits += moved
        self._data_downlink_Mb = self._delivered_bits / 1.0e6
        self.spacecraft.state.comms.data_sent_this_pass_Mb += moved / 1.0e6
        self._mark_delivered(moved, station or self._station_name or "ground", float(start_sec), float(start_sec) + air)
        return moved

    def _mark_delivered(self, bits: float, station: str, start_sec: float, end_sec: float) -> None:
        """Квитанция несёт отрезок передачи. time_sec — момент, когда эти биты дошли."""
        left = bits
        span = max(float(end_sec) - float(start_sec), 0.0)
        speed = bits / span if span > 1e-12 else 0.0
        cursor = float(start_sec)
        stop = float(end_sec)
        for product in self._products:
            if left <= 1e-9:
                return
            if product.get("delivered_to") or product.get("custodian") != self.spacecraft.config.name:
                continue
            onboard = float(product.get("onboard_bits", 0.0))
            ready = float(product.get("available_time_sec", product.get("generated_time_sec", 0.0)))
            if onboard <= 1e-9 or ready > float(start_sec) + 1e-9:
                continue
            take = min(onboard, left)
            offset = float(product.get("offset_bits", 0.0)) + float(product.get("size_bits", onboard)) - onboard
            product["onboard_bits"] = onboard - take
            if product["onboard_bits"] <= 1e-9:
                product["delivered_to"] = station
            piece_end = cursor + (take / speed if speed > 0.0 else 0.0)
            piece_end = min(stop, piece_end) if speed > 0.0 else stop
            self._receipts.append(
                {
                    "product_id": product["product_id"],
                    "request_id": product["request_id"],
                    "destination": station,
                    "time_sec": piece_end + float(self.spacecraft.config.comms.ack_sec),
                    "start_sec": cursor,
                    "end_sec": piece_end,
                    "offset_bits": offset,
                    "bits": take,
                }
            )
            cursor = piece_end
            left -= take

    def _terminal_free(self, start: float, end: float) -> float:
        span = end - start
        if span <= 0.0:
            return 0.0
        for used_start, used_end in self._terminal_used:
            span -= max(0.0, min(end, used_end) - max(start, used_start))
        return max(0.0, span)

    def _terminal_reserve(self, start: float, end: float) -> None:
        if end <= start:
            return
        self._terminal_used.append((start, end))
        self._terminal_used = _merge_intervals(self._terminal_used)

    def transmit_downlink(self, dt_sec: float) -> float:
        """Сдача за ещё не занятый интервал терминала. Повтор того же времени возвращает 0."""
        if dt_sec <= 0.0 or not math.isfinite(dt_sec) or not self._link_ok():
            return 0.0
        start = self._time_sec
        free = self._terminal_free(start, start + dt_sec)
        if free <= 0.0:
            return 0.0
        self._terminal_reserve(start, start + dt_sec)
        moved = self._move_downlink(free, self._link_margin_db, self._station_name or "ground", start, start + free)
        return moved / 1.0e6

    def _accrue_downlink(
        self,
        samples: list[tuple[float, np.ndarray, np.ndarray]],
        t0: float,
        *,
        armed: bool,
    ) -> None:
        if not armed:
            return
        moved = 0.0
        for left, right in zip(samples, samples[1:]):
            open_left, margin_left, name_left = self._link_sample(left[1], t0 + left[0])
            open_right, margin_right, name_right = self._link_sample(right[1], math.nextafter(t0 + right[0], -math.inf))
            span = right[0] - left[0]
            left_t = t0 + left[0]
            right_t = t0 + right[0]
            if open_left and open_right and name_left == name_right:
                moved += self._dump_after_setup(span, min(margin_left, margin_right), name_left, left_t, right_t)
            else:
                if open_left:
                    moved += self._dump_after_setup(0.5 * span, margin_left, name_left, left_t, left_t + 0.5 * span)
                if open_right:
                    moved += self._dump_after_setup(0.5 * span, margin_right, name_right, right_t - 0.5 * span, right_t)
        if self._time_sec > t0:
            self._terminal_reserve(t0, self._time_sec)
        del moved

    def _dump_after_setup(self, span: float, margin: float, station: str, start: float, end: float) -> float:
        begin = max(float(start), float(self._setup_until))
        finish = float(end)
        if finish <= begin + 1e-12 or span <= 0.0:
            return 0.0
        moved = 0.0
        for lo, hi in self._granted_spans(begin, finish):
            usable = min(float(span), hi - lo)
            moved += self._move_downlink(usable, margin, station, lo, hi)
        return moved

    def _granted_spans(self, begin: float, finish: float) -> list[tuple[float, float]]:
        if self._downlink_mask is None:
            return [(begin, finish)]
        pieces = []
        for start, end in self._downlink_mask:
            lo = max(float(begin), float(start))
            hi = min(float(finish), float(end))
            if hi > lo + 1e-12:
                pieces.append((lo, hi))
        return pieces

    def preview_contact_intervals(self, dt_sec: float) -> list[tuple[str, float, float]]:
        """Интервалы контакта на будущем шаге. Часы и орбита не двигаются."""
        origin = float(self._time_sec)
        samples = self.preview_positions(dt_sec)
        spans: list[tuple[str, float, float]] = []
        open_name = ""
        open_at: float | None = None
        previous = origin
        for offset, position in samples:
            moment = origin + float(offset)
            name, _margin = self.contact_at(position, self._epoch + moment, time_sec=moment)
            if name and name == open_name:
                previous = moment
                continue
            if open_name and open_at is not None and previous > open_at:
                spans.append((open_name, open_at, previous))
            open_name = name
            open_at = moment if name else None
            previous = moment
        if open_name and open_at is not None and previous > open_at:
            spans.append((open_name, open_at, previous))
        return spans

    def contact_at(self, position: np.ndarray, moment: Time, *, time_sec: float | None = None) -> tuple[str, float]:
        """Станция с лучшим запасом в этот момент. Пустое имя — линия закрыта."""
        if not self.radio_ready():
            return "", -99.0
        clock = self.time_sec if time_sec is None else float(time_sec)
        best_margin = -99.0
        best = ""
        floor = self.spacecraft.config.comms.min_margin_dB
        for station in self.stations:
            if any(station.name == name and start <= clock < end for name, start, end in self._link_outages):
                continue
            elevation = elevation_deg(position, station.lat_deg, station.lon_deg, station.alt_m, moment)
            slant = float(
                np.linalg.norm(
                    eci_to_ecef(position, moment) - geodetic_to_ecef(station.lat_deg, station.lon_deg, station.alt_m)
                )
            )
            margin = self._margin_db(slant)
            if elevation >= station.min_elevation_deg and margin >= floor and margin > best_margin:
                if not self._band_matches(station):
                    continue
                best_margin = margin
                best = station.name
        return best, best_margin

    def _band_matches(self, station: GroundStation) -> bool:
        band = float(station.band_hz)
        if band <= 0.0:
            return True
        freq = float(self.spacecraft.config.comms.frequency_Hz)
        return abs(freq - band) / band <= 0.02

    def _expire_products(self) -> None:
        ttl = float(self.spacecraft.config.comms.ttl_sec)
        if ttl <= 0.0:
            return
        now = float(self._time_sec)
        owner = self.spacecraft.config.name
        for product in self._products:
            if product.get("delivered_to") or product.get("custodian") != owner:
                continue
            onboard = float(product.get("onboard_bits", 0.0))
            born = float(product.get("generated_time_sec", 0.0))
            if onboard <= 1e-9 or now + 1e-9 < born + ttl:
                continue
            self._stored_bits = max(0.0, self._stored_bits - onboard)
            self._lost_bits += onboard
            product["onboard_bits"] = 0.0
            product["delivered_to"] = "expired"

    def _link_sample(self, position: np.ndarray, time_sec: float) -> tuple[bool, float, str]:
        name, margin = self.contact_at(position, self._epoch + time_sec, time_sec=time_sec)
        return bool(name), margin, name

    def _imaging_windows(self, origin: float, end: float, *, agent: bool) -> list[tuple[float, float]]:
        windows: list[tuple[float, float]] = []
        prep = float(self.spacecraft.config.payload.prep_sec)
        if agent:
            since = self._imaging_since if self._imaging_since is not None else origin
            ready = since + prep
            if ready < end - 1e-12:
                windows.append((max(ready, origin), end))
        for planned in self.scenario.planned_events:
            if planned.kind != EventKind.IMAGING:
                continue
            start = float(planned.scheduled_at_sec)
            stop = start + float(planned.duration_sec)
            for cut in self._imaging_cuts:
                if start < cut <= stop:
                    stop = cut
            if planned.duration_sec <= 0.0:
                if origin - 1e-9 <= start <= end + 1e-9 and not any(cut <= start for cut in self._imaging_cuts):
                    windows.append((start, start))
                continue
            ready = start + prep
            left = max(ready, origin)
            right = min(stop, end)
            if right <= left:
                continue
            windows.append((left, right))
        return _merge_intervals(windows)

    def _visible_seconds(
        self,
        samples: list[tuple[float, np.ndarray, np.ndarray]],
        t0: float,
        lat_deg: float,
        lon_deg: float,
        windows: list[tuple[float, float]],
    ) -> tuple[float, bool, float | None]:
        fov = self.spacecraft.config.payload.half_fov_deg
        images_in_eclipse = self.spacecraft.config.payload.images_in_eclipse
        duration = 0.0
        point = False
        opened_at: float | None = None

        def seen(position: np.ndarray, moment_sec: float) -> bool:
            return target_access(
                position,
                lat_deg,
                lon_deg,
                self._epoch + moment_sec,
                half_fov_deg=fov,
                frame=self.orbit.frame,
                images_in_eclipse=images_in_eclipse,
            ).visible

        for left, right in zip(samples, samples[1:]):
            left_t = t0 + left[0]
            right_t = t0 + right[0]
            visible = [seen(left[1], left_t), seen(right[1], right_t)]
            overlap = _covered_seconds(left_t, right_t, windows)
            if visible[0] and visible[1]:
                duration += overlap
            elif visible[0] or visible[1]:
                duration += 0.5 * overlap
            if opened_at is None and overlap > 0.0 and any(visible):
                opened_at = left_t if visible[0] else right_t
            for mark, flag, position in ((left_t, visible[0], left[1]), (right_t, visible[1], right[1])):
                if flag and any(abs(mark - start) <= 1e-6 and abs(stop - start) <= 1e-6 for start, stop in windows):
                    point = True
                    if opened_at is None:
                        opened_at = mark
                    del position
        return duration, point, opened_at

    def _resolve_imaging(
        self,
        samples: list[tuple[float, np.ndarray, np.ndarray]],
        t0: float,
        events: list[MissionEvent],
        windows: list[tuple[float, float]],
        *,
        agent: bool,
    ) -> None:
        self._imaging_requested = False
        if not windows or self._safe_latched or self._keeping_block:
            if agent:
                self.last_command_ok = False
            return
        if self.spacecraft.state.pointing != "nadir" or self._slew_remaining_deg > self.spacecraft.config.slew_settle_deg:
            if agent:
                self.last_command_ok = False
            return
        captured_now = False
        for target in self.scenario.ground_targets:
            duration, point, opened_at = self._visible_seconds(samples, t0, target.lat_deg, target.lon_deg, windows)
            if duration <= 0.0 and not point:
                continue
            when = opened_at if opened_at is not None else t0
            if self._exposure_block(target, when):
                continue
            if self._sar_block(target, duration if duration > 0.0 else 0.0):
                continue
            before = self._imaging_count
            self._store_capture(
                target.name,
                opened_at if opened_at is not None else t0,
                duration if duration > 0.0 else 0.0,
                grade=self._frame_grade(samples, target, opened_at if opened_at is not None else t0),
                coverage=self._frame_coverage(target),
            )
            if self._imaging_count > before:
                captured_now = True
                events.append(
                    self._event_at(
                        EventKind.IMAGING,
                        opened_at if opened_at is not None else t0,
                        {"target": target.name, "count": self._imaging_count},
                    )
                )
            elif duration > 0.0 and target.name in self._captured_targets:
                captured_now = True
        if agent and not captured_now:
            self.last_command_ok = False

    def _exposure_block(self, target, when_sec: float) -> str:
        """Полоса, полигон, повтор, стереопара и предшественник не пишут продукт, пока вердикт отрицательный."""
        geometry = str(getattr(target, "geometry", "point"))
        depends = str(getattr(target, "depends_on", "") or "")
        if geometry == "point" and not depends:
            self._exposure_reason = ""
            return ""
        from space_env.core.access import footprint_diameter_m, off_nadir_deg
        from space_env.core.constants import R_EARTH_EQUATOR
        from space_env.mission.geometry_requests import classify_request

        altitude = float(np.linalg.norm(self.orbit.r) - R_EARTH_EQUATOR)
        diameter = footprint_diameter_m(max(altitude, 0.0), self.spacecraft.config.payload.half_fov_deg)
        angle = off_nadir_deg(self.orbit.r, target.lat_deg, target.lon_deg, self._epoch + when_sec)
        history = self._looks.setdefault(target.name, [])
        history.append((float(when_sec), float(angle)))
        done = {str(item["request_id"]) for item in self._receipts}
        for product in self._products:
            dest = str(product.get("delivered_to") or "")
            if dest and dest not in {"expired", "evicted"}:
                done.add(str(product["request_id"]))
        verdict = classify_request(
            {
                "geometry": geometry,
                "lat_deg": target.lat_deg,
                "lon_deg": target.lon_deg,
                "length_km": float(getattr(target, "length_km", 0.0)),
                "vertices": list(getattr(target, "vertices", []) or []),
                "looks": int(getattr(target, "looks", 1)),
                "gap_sec": float(getattr(target, "gap_sec", 0.0)),
                "min_stereo_deg": float(getattr(target, "min_stereo_deg", 0.0)),
                "max_stereo_deg": float(getattr(target, "max_stereo_deg", 0.0)),
                "depends_on": depends,
            },
            footprint_diameter_m=diameter,
            hit_times_sec=[item[0] for item in history],
            off_nadir_deg=[item[1] for item in history],
            done_ids=done,
        )
        self._exposure_reason = "" if verdict["reachable"] else str(verdict["reason"])
        return self._exposure_reason

    def _sar_block(self, target, seconds: float) -> str:
        """Радар с заданной скважностью не пишет кадр ниже горизонта, сверх доли витка или без энергии."""
        payload = self.spacecraft.config.payload
        if not payload.images_in_eclipse or payload.sar_duty_fraction <= 0.0:
            return ""
        from space_env.core.constants import R_EARTH_EQUATOR
        from space_env.mission.sar_ops import SarDuty, sar_look

        access = target_access(
            self.orbit.r,
            target.lat_deg,
            target.lon_deg,
            self._epoch + self._time_sec,
            half_fov_deg=payload.half_fov_deg,
            frame=self.orbit.frame,
            images_in_eclipse=True,
        )
        period = 2.0 * math.pi / mean_motion_rad_s(float(np.linalg.norm(self.orbit.r)))
        duty = SarDuty(orbit_sec=period, duty_fraction=payload.sar_duty_fraction, used_sec=self._sar_used_sec)
        verdict = sar_look(
            elevation_deg=float(access.elevation_deg),
            altitude_m=max(float(np.linalg.norm(self.orbit.r) - R_EARTH_EQUATOR), 0.0),
            beam_half_deg=payload.half_fov_deg,
            power_w=payload.imaging_power_w,
            seconds=max(float(seconds), 0.0),
            soc=float(self.spacecraft.state.power.battery_soc),
            capacity_wh=float(self.spacecraft.config.power.battery_capacity_Wh),
            duty=duty,
        )
        if not verdict["ok"]:
            self._exposure_reason = str(verdict["reason"])
            return self._exposure_reason
        self._sar_used_sec = duty.used_sec
        self._exposure_reason = ""
        return ""

    def _frame_grade(self, samples, target, when_sec: float) -> str:
        from space_env.core.access import off_nadir_deg
        from space_env.mission.quality import frame_grade

        position = self.orbit.r
        if samples:
            origin = self._time_sec - float(samples[-1][0])
            position = min(samples, key=lambda item: abs(origin + float(item[0]) - when_sec))[1]
        payload = self.spacecraft.config.payload
        angle = off_nadir_deg(position, target.lat_deg, target.lon_deg, self._epoch + when_sec)
        grade = frame_grade(
            off_nadir_deg=angle,
            half_fov_deg=payload.half_fov_deg,
            cloud_fraction=self._cloud_cover(target),
            images_in_eclipse=bool(payload.images_in_eclipse),
            quality_off_nadir_deg=payload.quality_off_nadir_deg,
            slew_rate_deg_s=float(self._slew_rate_deg_s),
            smear_rate_deg_s=float(payload.smear_rate_deg_s),
        )
        factor = self._health_factor("payload")
        if grade == "full" and 0.0 < factor < 1.0:
            return "partial"
        return grade

    def _frame_coverage(self, target) -> dict:
        """Короткая полоса закрывается одним кадром. Длинная сюда не доходит: её режет вердикт."""
        geometry = str(getattr(target, "geometry", "point"))
        if geometry == "point":
            return {}
        from space_env.core.access import footprint_diameter_m
        from space_env.core.constants import R_EARTH_EQUATOR

        altitude = max(float(np.linalg.norm(self.orbit.r) - R_EARTH_EQUATOR), 0.0)
        diameter = footprint_diameter_m(altitude, self.spacecraft.config.payload.half_fov_deg)
        if geometry == "strip":
            length = float(getattr(target, "length_km", 0.0)) * 1000.0
            return {"geometry": "strip", "coverage_m": min(length, diameter)}
        if geometry == "polygon":
            return {"geometry": "polygon", "vertices": len(getattr(target, "vertices", []) or [])}
        return {"geometry": geometry}

    def _cloud_cover(self, target) -> float:
        from space_env.mission.quality import cloud_cover

        fraction = float(getattr(target, "cloud_fraction", 0.0))
        sigma = float(getattr(target, "cloud_sigma", 0.0))
        unit = self._rng.random() if sigma > 0.0 else 0.5
        return cloud_cover(fraction, sigma, unit)

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
            stored_bits=self._stored_bits,
            generated_bits=self._generated_bits,
            delivered_bits=self._delivered_bits,
            lost_bits=self._lost_bits,
            produced_bits=self._produced_bits,
            initial_bits=self._initial_bits,
            received_bits=self._received_bits,
            relayed_out_bits=self._relayed_out_bits,
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

    def _deadlines(self, t0: float, t1: float) -> list[float]:
        marks: list[float] = []
        for planned in self.scenario.planned_events:
            if t0 < planned.scheduled_at_sec < t1:
                marks.append(planned.scheduled_at_sec)
            stop = planned.scheduled_at_sec + planned.duration_sec
            if planned.duration_sec > 0.0 and t0 < stop < t1:
                marks.append(stop)
        for inj in self.scenario.anomaly_injects:
            if t0 < inj.at_sec < t1:
                marks.append(inj.at_sec)
        for inj in self.scenario.conjunction_injects:
            if t0 < inj.at_sec < t1:
                marks.append(inj.at_sec)
        return sorted(set(marks))

    def preview_positions(self, dt_sec: float) -> list[tuple[float, np.ndarray]]:
        """Положения на будущем интервале. Часы и орбита остаются на месте."""
        if dt_sec <= 0.0:
            return [(0.0, self.orbit.r.copy())]
        prop = self.spacecraft.config.propulsion
        lat, lon = self._body_latlon()
        _end_r, _end_v, samples = propagate_state(
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
            lat_deg=lat,
            lon_deg=lon,
            when=self._now().datetime.replace(tzinfo=None),
            srp_area_m2=prop.srp_area_m2,
            srp_coefficient=prop.srp_coefficient,
            include_third_body=True,
        )
        return [(float(offset), position) for offset, position, _velocity in samples]

    def _move(self, dt_sec: float, *, burn: dict[str, float | str] | None) -> list[tuple[float, np.ndarray, np.ndarray]]:
        prop = self.spacecraft.config.propulsion
        if prop.mass_kg <= 0.0 or not math.isfinite(prop.mass_kg):
            raise ValueError("масса аппарата должна быть положительной")
        lat, lon = self._body_latlon()
        mass_box = [prop.mass_kg]
        thrust_n = prop.thrust_N if burn is not None else 0.0
        direction = str(burn["direction"]) if burn is not None else None
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
            lat_deg=lat,
            lon_deg=lon,
            when=self._now().datetime.replace(tzinfo=None),
            srp_area_m2=prop.srp_area_m2,
            srp_coefficient=prop.srp_coefficient,
            include_third_body=True,
            thrust_n=thrust_n,
            thrust_direction=direction,
            isp_s=prop.isp_s,
            dry_mass_kg=prop.dry_mass_kg,
            mass_out=mass_box,
        )
        if not np.all(np.isfinite(r_end)) or not np.all(np.isfinite(v_end)):
            raise RuntimeError("состояние орбиты перестало быть конечным")
        if burn is not None and mass_box[0] < prop.mass_kg:
            from space_env.core.propulsion import exhaust_speed_m_s

            delivered = exhaust_speed_m_s(prop.isp_s) * math.log(prop.mass_kg / mass_box[0])
            spent = min(float(burn["dv_left"]), delivered)
            burn["dv_left"] = float(burn["dv_left"]) - spent
            self.spacecraft.state.propulsion.delta_v_remaining_m_s -= spent
            prop.mass_kg = mass_box[0]
        self._time_sec += dt_sec
        self.orbit = self.orbit.with_state(r_end, v_end, self._epoch + self._time_sec)
        self._refresh_environment()
        return samples

    def _propagate_segment(self, dt_sec: float) -> list[tuple[float, np.ndarray, np.ndarray]]:
        from space_env.core.propulsion import burn_seconds

        samples: list[tuple[float, np.ndarray, np.ndarray]] = []
        left = float(dt_sec)
        covered = 0.0
        while left > 1e-8:
            burn = self._burn
            if burn is None:
                part = self._move(left, burn=None)
                self._append_samples(samples, part, covered)
                break
            prop = self.spacecraft.config.propulsion
            span = burn_seconds(
                prop.mass_kg,
                float(burn["dv_left"]),
                prop.thrust_N,
                prop.isp_s,
                prop.dry_mass_kg,
                left,
            )
            if span <= 1e-8:
                self._burn = None
                continue
            part = self._move(span, burn=burn)
            self._append_samples(samples, part, covered)
            covered += span
            left -= span
            if float(burn["dv_left"]) <= 1e-4:
                self._burn = None
                self._refresh_passes()
        return samples

    def _append_samples(
        self,
        samples: list[tuple[float, np.ndarray, np.ndarray]],
        part: list[tuple[float, np.ndarray, np.ndarray]],
        offset: float,
    ) -> None:
        for mark, position, velocity in part:
            if samples and mark <= 1e-9:
                continue
            samples.append((offset + mark, position, velocity))

    def step(
        self,
        dt_sec: float,
        command: str | None = None,
        command_payload: dict | None = None,
    ) -> tuple[SimulationState, list[MissionEvent]]:
        """Один шаг. Команда действует с начала шага, событие плана — в свой момент."""
        if not math.isfinite(dt_sec) or dt_sec <= 0.0:
            raise ValueError("dt_sec должен быть положительным и конечным")
        events: list[MissionEvent] = []
        was_umbra = self._in_umbra
        was_contact = self._in_contact
        origin = self._time_sec
        self._keeping_block = False
        self._apply_power_hysteresis(events)
        self._apply_command(command, command_payload or {}, events, dt_sec)
        armed = self._downlink_armed
        windows = self._imaging_windows(
            origin,
            origin + dt_sec,
            agent=command == "imaging" and self._imaging_requested,
        )
        if origin == 0.0:
            self._fire_injects(0.0, 0.0, events, include_zero=True)
            self._run_planned(0.0, 0.0, events, 0.0)
        combined: list[tuple[float, np.ndarray, np.ndarray]] = []
        cursor = origin
        end = origin + dt_sec
        for mark in [*self._deadlines(origin, end), end]:
            seg = mark - cursor
            if seg > 1e-8:
                samples = self._propagate_segment(seg)
                base = cursor - origin
                for offset, position, velocity in samples:
                    if combined and offset <= 1e-9:
                        continue
                    combined.append((base + offset, position, velocity))
            self._fire_injects(cursor, mark, events, include_zero=False)
            self._run_planned(cursor, mark, events, seg)
            if seg > 1e-8:
                self._update_subsystems(
                    seg,
                    samples,
                    imaging_sec=_covered_seconds(cursor, mark, windows),
                )
                self._apply_power_hysteresis(events, windows)
            cursor = mark
        self._advance_slew(dt_sec)
        self._advance_encounter(dt_sec)
        self._resolve_imaging(
            combined,
            origin,
            events,
            windows,
            agent=command == "imaging" and bool(windows),
        )
        self._advance_processor(origin, dt_sec)
        self._restore_anomalies(events)
        self._tick_symptom(events)
        if self._radio_recovery_steps > 0:
            self._radio_recovery_steps -= 1
            if self._radio_recovery_steps == 0:
                self._faults.discard("radio_latch")
        self._expire_products()
        self._accrue_downlink(combined, origin, armed=armed and not self._safe_latched)
        self._soc_rate = (self.spacecraft.state.power.battery_soc - self._prev_soc) / dt_sec
        self._prev_soc = self.spacecraft.state.power.battery_soc
        self._emit_edges(was_umbra, was_contact, events, combined, origin)
        self._check_constraints(events)
        self._downlink_mask = None
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
    def stored_bits(self) -> float:
        return self._stored_bits

    @property
    def delivered_bits(self) -> float:
        return self._delivered_bits

    @property
    def pass_capacity_bits(self) -> float:
        """Сколько бит модем мог пропустить за уже открытые куски. Это не объём буфера."""
        return self._pass_capacity_bits

    @property
    def produced_bits(self) -> float:
        return self._produced_bits

    @property
    def receipts(self) -> tuple[dict, ...]:
        return tuple(self._receipts)

    def products(self) -> tuple[dict, ...]:
        """Снимок регистратора. Политика читает его и не пишет в борт."""
        return tuple(dict(item) for item in self._products)

    def request_report(self, request_id: str, *, deadline_sec: float | None = None, destination: str | None = None) -> dict:
        """Полнота заявки по квитанциям, а не по общей сумме регистратора."""
        produced = [
            item
            for item in self._products
            if item["request_id"] == request_id and item["origin"] == "imaging"
        ]
        size = sum(float(item["size_bits"]) for item in produced)
        slips = [
            item
            for item in self._receipts
            if item["request_id"] == request_id and (destination is None or item["destination"] == destination)
        ]
        delivered = sum(float(item["bits"]) for item in slips)
        on_time = deadline_sec is None or all(float(item["time_sec"]) <= deadline_sec + 1e-6 for item in slips)
        complete = size > 0.0 and delivered + 1e-6 >= size and on_time and bool(slips)
        return {
            "request_id": request_id,
            "size_bits": size,
            "delivered_bits": delivered,
            "on_time": on_time,
            "complete": complete,
            "receipts": slips,
        }

    @property
    def log(self) -> list[tuple[float, SimulationState, list[MissionEvent]]]:
        return self._log

    def set_link_outages(self, intervals: list[tuple[str, float, float]]) -> None:
        """Станция не принимает в полуинтервалах [start, end) по часам аппарата."""
        self._link_outages = [(str(name), float(start), float(end)) for name, start, end in intervals]

    def capture(self) -> dict:
        """Снимок, с которого шаг можно продолжить."""
        import copy

        return {
            "physics_profile": self.physics_profile,
            "time_sec": self.time_sec,
            "spacecraft_state": self.spacecraft.state.model_dump(mode="json"),
            "runtime": {
                name: getattr(self, name) for name in (
                    "_comms_sessions", "_data_downlink_Mb", "_pass_capacity_bits",
                    "_imaging_requested", "_prev_soc", "_soc_rate", "last_command_ok",
                )
            },
            "passes": [asdict(window) for window in self._passes],
            "encounter": _capture_encounter(self._encounter),
            "r": self.orbit.r.copy(),
            "v": self.orbit.v.copy(),
            "stored_bits": self._stored_bits,
            "produced_bits": self._produced_bits,
            "delivered_bits": self._delivered_bits,
            "lost_bits": self._lost_bits,
            "initial_bits": self._initial_bits,
            "received_bits": self._received_bits,
            "relayed_out_bits": self._relayed_out_bits,
            "generated_bits": self._generated_bits,
            "soc": self.spacecraft.state.power.battery_soc,
            "temp_c": self.spacecraft.state.thermal.temp_C,
            "products": copy.deepcopy(self._products),
            "receipts": copy.deepcopy(self._receipts),
            "terminal_used": list(self._terminal_used),
            "imaging_cuts": list(self._imaging_cuts),
            "captured": set(self._captured_targets),
            "imaging_count": self._imaging_count,
            "payload_energy_j": self._payload_energy_j,
            "downlink_armed": self._downlink_armed,
            "setup_until": self._setup_until,
            "delta_v_remaining_m_s": self.spacecraft.state.propulsion.delta_v_remaining_m_s,
            "fired_planned": set(self._fired_planned),
            "fired_anomaly": set(self._fired_anomaly),
            "fired_conjunction": set(self._fired_conjunction),
            "faults": set(self._faults),
            "radio_recovery_steps": self._radio_recovery_steps,
            "load_mode": self._load_mode,
            "pointing": self.spacecraft.state.pointing,
            "payload_mode": self.spacecraft.state.payload_mode,
            "slew_remaining_deg": self._slew_remaining_deg,
            "slew_rate_deg_s": self._slew_rate_deg_s,
            "slew_span_deg": self._slew_span_deg,
            "product_seq": self._product_seq,
            "log_len": len(self._log),
            "health": {
                key: value.value if isinstance(value, SubsystemHealth) else str(value)
                for key, value in self.spacecraft.state.health.items()
            },
            "restores": [(item.at_sec, item.subsystem, item.done) for item in self._restores],
            "active_violations": set(self._active_violations),
            "attitude": self._attitude.copy(),
            "rng_state": self._rng.getstate(),
            "imaging_since": self._imaging_since,
            "mass_kg": self.spacecraft.config.propulsion.mass_kg,
            "burn": None if self._burn is None else dict(self._burn),
            "safe_latched": self._safe_latched,
            "load_before_safe": self._load_before_safe,
            "process_queue": [dict(item) for item in self._process_queue],
            "looks": {name: list(samples) for name, samples in self._looks.items()},
            "sar_used_sec": self._sar_used_sec,
            "exposure_reason": self._exposure_reason,
            "symptom": self._symptom,
            "symptom_factor": self._symptom_factor,
            "hidden_cause": self._hidden_cause,
            "recovery_sec": self._recovery_sec,
            "recover_until": self._recover_until,
            "symptom_until": self._symptom_until,
            "false_latch": self._false_latch,
            "message_miss_m": self._message_miss_m,
            "avoid_reason": self._avoid_reason,
            "keeping_block": self._keeping_block,
            "keeping_reason": self._keeping_reason,
            "rate_share": self._rate_share,
            "phase_reason": self._phase_reason,
        }

    def restore(self, payload: dict) -> None:
        """Возвращает снимок capture(). Дальнейший шаг совпадает с непрерывным прогоном."""
        import copy

        if payload.get("physics_profile", LEGACY_PHYSICS_PROFILE) != self.physics_profile:
            raise ValueError("incompatible physics profile; no implicit frame relabeling")
        self._time_sec = float(payload["time_sec"])
        self.orbit = self.orbit.with_state(payload["r"], payload["v"], self._epoch + self._time_sec)
        self._stored_bits = float(payload["stored_bits"])
        self._produced_bits = float(payload["produced_bits"])
        self._delivered_bits = float(payload["delivered_bits"])
        self._lost_bits = float(payload["lost_bits"])
        self._initial_bits = float(payload["initial_bits"])
        self._received_bits = float(payload["received_bits"])
        self._relayed_out_bits = float(payload["relayed_out_bits"])
        self._generated_bits = float(payload["generated_bits"])
        self.spacecraft.state.power.battery_soc = float(payload["soc"])
        self.spacecraft.state.thermal.temp_C = float(payload["temp_c"])
        self._products = copy.deepcopy(payload["products"])
        self._receipts = copy.deepcopy(payload["receipts"])
        self._terminal_used = list(payload["terminal_used"])
        self._imaging_cuts = list(payload["imaging_cuts"])
        self._captured_targets = set(payload["captured"])
        self._imaging_count = int(payload["imaging_count"])
        self._payload_energy_j = float(payload["payload_energy_j"])
        self._downlink_armed = bool(payload["downlink_armed"])
        if "setup_until" in payload:
            self._setup_until = float(payload["setup_until"])
        if "delta_v_remaining_m_s" in payload:
            self.spacecraft.state.propulsion.delta_v_remaining_m_s = float(payload["delta_v_remaining_m_s"])
        if "fired_planned" in payload:
            self._fired_planned = set(payload["fired_planned"])
        if "fired_anomaly" in payload:
            self._fired_anomaly = set(payload["fired_anomaly"])
        if "fired_conjunction" in payload:
            self._fired_conjunction = set(payload["fired_conjunction"])
        if "faults" in payload:
            self._faults = set(payload["faults"])
        if "radio_recovery_steps" in payload:
            self._radio_recovery_steps = int(payload["radio_recovery_steps"])
        if "load_mode" in payload:
            self._load_mode = str(payload["load_mode"])
        if "pointing" in payload:
            self.spacecraft.state.pointing = str(payload["pointing"])
        if "payload_mode" in payload:
            self.spacecraft.state.payload_mode = str(payload["payload_mode"])
        if "slew_remaining_deg" in payload:
            self._slew_remaining_deg = float(payload["slew_remaining_deg"])
        if "slew_rate_deg_s" in payload:
            self._slew_rate_deg_s = float(payload["slew_rate_deg_s"])
        if "slew_span_deg" in payload:
            self._slew_span_deg = float(payload["slew_span_deg"])
        if "product_seq" in payload:
            self._product_seq = int(payload["product_seq"])
        if "log_len" in payload:
            self._log = self._log[: int(payload["log_len"])]
        if "health" in payload:
            self.spacecraft.state.health = {
                str(key): SubsystemHealth(str(value)) for key, value in payload["health"].items()
            }
        if "restores" in payload:
            self._restores = [
                _Restore(float(item[0]), str(item[1]), bool(item[2])) for item in payload["restores"]
            ]
        if "active_violations" in payload:
            self._active_violations = set(payload["active_violations"])
        if "attitude" in payload:
            self._attitude = np.array(payload["attitude"], dtype=float).copy()
        if "rng_state" in payload:
            self._rng.setstate(_rng_state(payload["rng_state"]))
        self._imaging_since = None if payload.get("imaging_since") is None else float(payload["imaging_since"])
        if "mass_kg" in payload:
            self.spacecraft.config.propulsion.mass_kg = float(payload["mass_kg"])
        if "burn" in payload:
            self._burn = None if payload["burn"] is None else dict(payload["burn"])
        if "safe_latched" in payload:
            self._safe_latched = bool(payload["safe_latched"])
            self._load_before_safe = payload.get("load_before_safe")
        if "process_queue" in payload:
            self._process_queue = [dict(item) for item in payload["process_queue"]]
        if "looks" in payload:
            self._looks = {
                str(name): [(float(item[0]), float(item[1])) for item in samples]
                for name, samples in payload["looks"].items()
            }
        if "sar_used_sec" in payload:
            self._sar_used_sec = float(payload["sar_used_sec"])
        if "exposure_reason" in payload:
            self._exposure_reason = str(payload["exposure_reason"])
        if "symptom" in payload:
            self._symptom = str(payload["symptom"])
            self._symptom_factor = float(payload.get("symptom_factor", 1.0))
            self._hidden_cause = str(payload.get("hidden_cause", ""))
            self._recovery_sec = float(payload.get("recovery_sec", 0.0))
            self._recover_until = float(payload.get("recover_until", 0.0))
            self._symptom_until = float(payload.get("symptom_until", 0.0))
            self._false_latch = bool(payload.get("false_latch", False))
        if "message_miss_m" in payload:
            self._message_miss_m = None if payload["message_miss_m"] is None else float(payload["message_miss_m"])
        if "avoid_reason" in payload:
            self._avoid_reason = str(payload["avoid_reason"])
        if "keeping_block" in payload:
            self._keeping_block = bool(payload["keeping_block"])
        if "keeping_reason" in payload:
            self._keeping_reason = str(payload["keeping_reason"])
        if "rate_share" in payload:
            self._rate_share = float(payload["rate_share"])
        if "phase_reason" in payload:
            self._phase_reason = str(payload["phase_reason"])
        self._prev_soc = self.spacecraft.state.power.battery_soc
        if "spacecraft_state" in payload:
            self.spacecraft.state = SpacecraftState.model_validate(payload["spacecraft_state"])
        for name, value in payload.get("runtime", {}).items():
            if name in {"_comms_sessions", "_data_downlink_Mb", "_pass_capacity_bits",
                        "_imaging_requested", "_prev_soc", "_soc_rate", "last_command_ok"}:
                setattr(self, name, value)
        if "passes" in payload:
            self._passes = [PassWindow(**row) for row in payload["passes"]]
        if "encounter" in payload:
            self._encounter = _restore_encounter(payload["encounter"])
        self._refresh_environment()


def _capture_encounter(encounter):
    if encounter is None:
        return None
    import copy

    state = copy.deepcopy(vars(encounter))
    state["debris"] = asdict(encounter.debris)
    when = state["debris"]["when"]
    state["debris"]["when"] = when.isoformat() if isinstance(when, datetime) else when
    return state


def _restore_encounter(payload):
    if payload is None:
        return None
    import copy

    state = copy.deepcopy(payload)
    context = state.pop("debris")
    if isinstance(context["when"], str):
        context["when"] = datetime.fromisoformat(context["when"])
    encounter = NumericalEncounter(
        state.pop("r"), state.pop("v"), state.pop("covariance"),
        state.pop("tca_sec"), state.pop("hard_body_m"), PropagationContext(**context),
    )
    for name, value in state.items():
        if name in {"passed", "frozen_miss_m", "frozen_pc", "frozen_radial_m", "frozen_cross_m"}:
            setattr(encounter, name, value)
    return encounter


def _rng_state(state):
    """random.Random.setstate ждёт кортеж. JSON превращает его в списки."""
    if isinstance(state, list):
        version, internal, gauss = state
        return (version, tuple(internal), gauss)
    return state


def _merge_intervals(windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    points = [(start, stop) for start, stop in windows if abs(stop - start) <= 1e-9]
    spans = sorted((start, stop) for start, stop in windows if stop > start + 1e-9)
    merged: list[tuple[float, float]] = []
    for start, stop in spans:
        if merged and start <= merged[-1][1] + 1e-9:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged + points


def _covered_seconds(start: float, stop: float, windows: list[tuple[float, float]]) -> float:
    covered = 0.0
    for window_start, window_stop in windows:
        if window_stop <= window_start:
            continue
        lo = max(start, window_start)
        hi = min(stop, window_stop)
        if hi > lo:
            covered += hi - lo
    return covered


def _is_due(at_sec: float, t0: float, t1: float, *, include_zero: bool) -> bool:
    if include_zero and at_sec == 0.0 and t0 == 0.0 and t1 == 0.0:
        return True
    return t0 < at_sec <= t1
