"""
Движок симуляции: шаг по времени, пропагация орбиты, обновление подсистем, события, лог.
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

from space_env.core import Orbit, Time
from space_env.core.environment import in_eclipse
from space_env.core.spacecraft import Spacecraft, SubsystemHealth
from space_env.mission import (
    Constraints,
    EventKind,
    MissionEvent,
    MissionGoal,
    ScenarioConfig,
)
from space_env.simulation.state import SimulationState

if TYPE_CHECKING:
    pass

# Упрощённое направление на Солнце (инерциально) — для eclipse
_SUN_DIRECTION = np.array([1.0, 0.0, 0.0]) * 1.5e11  # м


class SimulationEngine:
    """
    Движок пошаговой симуляции.
    Вход: начальная орбита, КА, цели и ограничения миссии, сценарий.
    Шаг: пропагация орбиты, обновление питания/термики (упрощённо), проверка инжектов, лог.
    """

    def __init__(
        self,
        orbit_0: Orbit,
        spacecraft: Spacecraft,
        mission_goal: MissionGoal | None = None,
        constraints: Constraints | None = None,
        scenario: ScenarioConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.orbit = orbit_0
        self.spacecraft = spacecraft
        self.mission_goal = mission_goal or MissionGoal()
        self.constraints = constraints or Constraints()
        self.scenario = scenario or ScenarioConfig()
        self._rng = random.Random(seed)
        if seed is not None:
            np.random.seed(seed)

        self._time_sec = 0.0
        self._epoch = orbit_0.epoch
        self._log: list[tuple[float, SimulationState, list[MissionEvent]]] = []
        self._apply_scenario_initial()
        # Начальная запись в лог
        self._log.append((0.0, self.get_state_snapshot(), []))

    def _apply_scenario_initial(self) -> None:
        if self.scenario.initial_battery_soc is not None:
            self.spacecraft.state.power.battery_soc = self.scenario.initial_battery_soc
        if self.scenario.initial_delta_v_fraction is not None:
            total = self.spacecraft.config.propulsion.delta_v_total_m_s
            self.spacecraft.state.propulsion.delta_v_remaining_m_s = total * self.scenario.initial_delta_v_fraction

    def _current_time(self) -> Time:
        return self._epoch + self._time_sec

    def _inject_anomalies(self, events: list[MissionEvent]) -> None:
        for inj in self.scenario.anomaly_injects:
            if abs(self._time_sec - inj.at_sec) < 1.0:  # в пределах 1 с
                events.append(
                    MissionEvent(
                        kind=EventKind.ANOMALY,
                        time_sec=self._time_sec,
                        payload={"subsystem": inj.subsystem, "severity": inj.severity},
                    )
                )
                self.spacecraft.state.health[inj.subsystem] = (
                    SubsystemHealth.FAILED if inj.severity == "failed" else SubsystemHealth.DEGRADED
                )

    def _inject_conjunctions(self, events: list[MissionEvent]) -> None:
        for inj in self.scenario.conjunction_injects:
            if abs(self._time_sec - inj.at_sec) < 1.0:
                events.append(
                    MissionEvent(
                        kind=EventKind.CONJUNCTION_WARNING,
                        time_sec=self._time_sec,
                        payload={"miss_distance_km": inj.miss_distance_km, "probability": inj.probability},
                    )
                )

    def _update_power_simple(self, dt_sec: float, in_umbra: bool) -> None:
        """Упрощённое обновление SoC: заряд на свету, разряд в тени."""
        cfg = self.spacecraft.config.power
        st = self.spacecraft.state.power
        st.in_sun = not in_umbra
        consumption_W = st.consumption_W or cfg.consumption_nominal_W
        if in_umbra:
            # Только разряд
            Wh_used = consumption_W * (dt_sec / 3600.0)
            st.battery_soc = max(0.0, st.battery_soc - Wh_used / cfg.battery_capacity_Wh)
        else:
            # Заряд (упрощённо: мощность СБ минус потребление)
            power_in_W = 0.3 * cfg.solar_panel_area_m2 * 1361.0  # КПД ~0.3, солнечная постоянная
            net_W = power_in_W - consumption_W
            if net_W > 0:
                st.battery_soc = min(1.0, st.battery_soc + net_W * (dt_sec / 3600.0) / cfg.battery_capacity_Wh)
            else:
                st.battery_soc = max(0.0, st.battery_soc + net_W * (dt_sec / 3600.0) / cfg.battery_capacity_Wh)

    def step(self, dt_sec: float) -> tuple[SimulationState, list[MissionEvent]]:
        """
        Один шаг симуляции.
        Возвращает (снимок состояния, список событий за шаг).
        """
        events: list[MissionEvent] = []
        # Пропагация орбиты
        self.orbit = self.orbit.propagate(dt_sec)
        self._time_sec += dt_sec
        r = self.orbit.r
        in_umbra = in_eclipse(r, _SUN_DIRECTION)
        # Инжекты по сценарию
        self._inject_anomalies(events)
        self._inject_conjunctions(events)
        # Eclipse события (опционально, при пересечении границы можно детектировать — упрощённо пропуск)
        # Обновление питания
        self._update_power_simple(dt_sec, in_umbra)
        # Снимок состояния
        st = self.spacecraft.state
        snap = SimulationState(
            time_sec=self._time_sec,
            altitude_km=self.orbit.altitude_km(),
            in_eclipse=in_umbra,
            battery_soc=st.power.battery_soc,
            delta_v_remaining_m_s=st.propulsion.delta_v_remaining_m_s,
            temp_C=st.thermal.temp_C,
            imaging_count=0,  # заполняется при явном вызове record_imaging
            comms_sessions=0,
            data_downlink_Mb=st.comms.data_sent_this_pass_Mb,
            delta_v_used_m_s=self.spacecraft.config.propulsion.delta_v_total_m_s - st.propulsion.delta_v_remaining_m_s,
        )
        self._log.append((self._time_sec, snap, events))
        return snap, events

    def get_state_snapshot(self) -> SimulationState:
        """Текущий снимок без шага."""
        st = self.spacecraft.state
        return SimulationState(
            time_sec=self._time_sec,
            altitude_km=self.orbit.altitude_km(),
            in_eclipse=in_eclipse(self.orbit.r, _SUN_DIRECTION),
            battery_soc=st.power.battery_soc,
            delta_v_remaining_m_s=st.propulsion.delta_v_remaining_m_s,
            temp_C=st.thermal.temp_C,
            imaging_count=0,
            comms_sessions=0,
            data_downlink_Mb=st.comms.data_sent_this_pass_Mb,
            delta_v_used_m_s=self.spacecraft.config.propulsion.delta_v_total_m_s - st.propulsion.delta_v_remaining_m_s,
        )

    @property
    def time_sec(self) -> float:
        return self._time_sec

    @property
    def log(self) -> list[tuple[float, SimulationState, list[MissionEvent]]]:
        return self._log
