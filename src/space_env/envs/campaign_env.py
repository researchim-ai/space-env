"""
DayMission: принять или сломать уже посчитанные сутки.
DayShift: вести эти сутки по делам — съёмка, сдача и один импульс.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import gymnasium as gym
import numpy as np

from space_env.catalog.build import make_engine
from space_env.mission.campaign import (
    MISSION_KEYS,
    DayCampaign,
    SharedWindow,
    build_day_campaign,
    campaign_drafts,
    commit_award,
    shared_windows,
)
from space_env.mission.coast import coast_to_job
from space_env.mission.horizon import PC_THRESHOLD
from space_env.mission.outcome import classify_outcome
from space_env.mission.scenarios import GroundTarget
from space_env.mission.shift_scenario import load_shift_scenario
from space_env.rl.rewards import load_rewards
from space_env.simulation.engine import SimulationEngine


@dataclass(frozen=True)
class _Job:
    kind: str
    vehicle: str
    name: str
    start_sec: float
    end_sec: float = 0.0
    direction: str = ""


def _shift_jobs(campaign: DayCampaign) -> tuple[_Job, ...]:
    """Дела всех аппаратов на одних часах. Импульс раньше съёмки, если время совпало."""
    jobs: list[_Job] = []
    for plan in campaign.plans:
        for item in plan.required_items():
            if item.kind in {"imaging", "comms"}:
                jobs.append(_Job(item.kind, plan.vehicle, item.name, item.start_sec, item.end_sec))
    if campaign.burn_at_sec is not None:
        burn = next(item for item in campaign.conjunctions if item.required)
        jobs.append(
            _Job(
                "maneuver",
                campaign.vehicles[0],
                burn.name,
                campaign.burn_at_sec,
                campaign.burn_at_sec + 1.0,
                burn.direction or "prograde",
            )
        )
    jobs.sort(key=lambda job: (job.start_sec, 0 if job.kind == "maneuver" else 1, job.vehicle))
    return tuple(jobs)


_KIND_CODE = {"imaging": 0.5, "maneuver": 0.7, "comms": 1.0}
_ACTION_KIND = {1: "imaging", 2: "comms", 3: "maneuver"}


class DayMissionEnv(gym.Env):
    """
    Наблюдение: доля бюджета, Pc горячей пары, фрагмент ниже порога, доля аппаратов с сеансом.
    Действия: 0 принять сутки, 1 жечь по фрагменту, 2 не жечь горячую пару.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 1):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self._campaign: DayCampaign | None = None
        self._weights = load_rewards()["day_mission"]

    def _campaign_or_raise(self) -> DayCampaign:
        if self._campaign is None:
            raise RuntimeError("сначала reset")
        return self._campaign

    def _obs(self) -> np.ndarray:
        campaign = self._campaign_or_raise()
        budget = campaign.dv_budget_m_s
        spent = 0.0 if budget <= 0.0 else min(1.0, campaign.dv_spent_m_s / budget)
        hot = next(item for item in campaign.conjunctions if item.name == "HOT")
        debris_clear = not any(item.required and item.name == "debris-fragment" for item in campaign.conjunctions)
        with_pass = 0
        for plan in campaign.plans:
            if any(item.required and item.kind == "comms" for item in plan.items):
                with_pass += 1
        share = with_pass / len(campaign.plans) if campaign.plans else 0.0
        return np.array(
            [
                spent,
                float(np.clip(hot.probability / 1.0e-2, 0.0, 1.0)),
                1.0 if debris_clear else 0.0,
                share,
            ],
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._campaign = build_day_campaign()
        return self._obs(), {"lines": self._campaign.lines()[:8]}

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        choice = int(np.asarray(action).flat[0])
        campaign = self._campaign_or_raise()
        hot = next(item for item in campaign.conjunctions if item.name == "HOT")
        sentinel = next(plan for plan in campaign.plans if plan.vehicle == "sentinel-2a")
        spent_ok = campaign.dv_spent_m_s <= campaign.dv_budget_m_s + 1e-9
        debris_clear = not any(item.required and item.name == "debris-fragment" for item in campaign.conjunctions)
        has_dump = any(item.required and item.kind == "comms" for item in sentinel.items)
        has_orders = len(sentinel.targets) >= 8
        plan_ok = spent_ok and debris_clear and has_dump and has_orders
        flown = dict(campaign.flown_pc)
        coast = dict(campaign.coast_pc)
        if choice == 0:
            hot_ok = hot.required or hot.probability < PC_THRESHOLD
            flown_ok = campaign.burn_at_sec is not None and flown.get("HOT", 1.0) < PC_THRESHOLD and coast.get("HOT", 0.0) >= PC_THRESHOLD
            success = plan_ok and hot_ok and flown_ok
        elif choice == 1:
            success = False
        else:
            success = plan_ok and not hot.required
        reward = self._weights["accepted"] if success else -self._weights["rejected"]
        info = {
            "success": success,
            "dv_spent_m_s": campaign.dv_spent_m_s,
            "orders": len(sentinel.targets),
            "dumps": sum(1 for item in sentinel.required_items() if item.kind == "comms"),
            "choice": choice,
            "hot_coast_pc": coast.get("HOT", 0.0),
            "hot_flown_pc": flown.get("HOT", 0.0),
        }
        return self._obs(), float(reward), True, False, info


class DayShiftEnv(gym.Env):
    """
    Наблюдение: вид дела, в окне, доля закрытых, это импульс, доля бюджета.
    Действия: 0 ждать, 1 съёмка, 2 сдача, 3 импульс по плану.
    Дела берутся из суток, где каждое общее окно уже отдано большему буферу.
    Успех — дело каждого аппарата подтверждено его движком:
    съёмка пишет названную цель, сдача увеличивает счётчик станции, импульс списывает Δv.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 120):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)
        self._campaign: DayCampaign | None = None
        self._jobs: tuple[_Job, ...] = ()
        self._index = 0
        self._done = 0
        self._burned = False
        self._engines: dict[str, SimulationEngine] = {}
        self._weights = load_rewards()["day_shift"]

    def _obs(self) -> np.ndarray:
        campaign = self._campaign
        if campaign is None or self._index >= len(self._jobs):
            return np.zeros(5, dtype=np.float32)
        job = self._jobs[self._index]
        budget = campaign.dv_budget_m_s
        spent = 0.0 if budget <= 0.0 else min(1.0, campaign.dv_spent_m_s / budget)
        return np.array(
            [
                _KIND_CODE[job.kind],
                1.0,
                0.0 if not self._jobs else self._done / len(self._jobs),
                1.0 if job.kind == "maneuver" else 0.0,
                spent,
            ],
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._campaign = build_day_campaign()
        self._jobs = _shift_jobs(self._campaign)
        self._index = 0
        self._done = 0
        self._burned = False
        self._engines = {}
        for plan in self._campaign.plans:
            targets = [GroundTarget(name=name, lat_deg=lat, lon_deg=lon) for name, lat, lon in plan.targets]
            self._engines[plan.vehicle] = make_engine(
                plan.vehicle,
                epoch=self._campaign.epoch,
                nu_deg=180.0,
                with_stations=True,
                ground_targets=targets,
            )
        return self._obs(), {"jobs": len(self._jobs), "vehicles": list(self._engines)}

    def _engine_for(self, vehicle: str) -> SimulationEngine:
        engine = self._engines.get(vehicle)
        if engine is None:
            raise KeyError(vehicle)
        return engine

    def _coast_to(self, engine: SimulationEngine, at_sec: float) -> None:
        """Шаг 60 с, как у плана: время третьего тела не застывает на длинном скачке."""
        coast_to_job(engine, max(at_sec, engine.time_sec))

    def _engine_accepts(self, job: _Job) -> bool:
        engine = self._engine_for(job.vehicle)
        self._coast_to(engine, job.start_sec)
        if job.kind == "maneuver":
            before = engine.spacecraft.state.propulsion.delta_v_remaining_m_s
            engine.step(1.0, "maneuver", {"dv_m_s": 1.0, "direction": job.direction or "prograde"})
            spent = before - engine.spacecraft.state.propulsion.delta_v_remaining_m_s
            return bool(engine.last_command_ok and spent >= 0.5 and job.name == "HOT")
        if job.kind == "imaging":
            if job.name in engine.captured_targets():
                return True
            span = max(1.0, min(60.0, job.end_sec - max(job.start_sec, engine.time_sec)))
            engine.step(span, "imaging")
            return job.name in engine.captured_targets()
        span = max(1.0, min(60.0, job.end_sec - max(job.start_sec, engine.time_sec)))
        engine.step(span, "comms")
        return bool(engine.last_command_ok)

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._campaign is None:
            raise RuntimeError("сначала reset")
        choice = int(np.asarray(action).flat[0])
        if self._index >= len(self._jobs):
            info = {"success": False, "done": self._done, "burned": self._burned, "missed": 1}
            return self._obs(), -self._weights["missed"], True, False, info
        job = self._jobs[self._index]
        matched = _ACTION_KIND.get(choice) == job.kind
        if not matched:
            info = {
                "success": False,
                "done": self._done,
                "burned": self._burned,
                "missed": 1,
                "kind": job.kind,
                "choice": choice,
            }
            return self._obs(), -float(self._weights["missed"]), True, False, info
        if not self._engine_accepts(job):
            info = {
                "success": False,
                "done": self._done,
                "burned": self._burned,
                "missed": 1,
                "kind": job.kind,
                "name": job.name,
            }
            return self._obs(), -float(self._weights["missed"]), True, False, info
        if job.kind == "maneuver":
            self._burned = True
        self._done += 1
        self._index += 1
        finished = self._index >= len(self._jobs)
        success = finished and self._burned and self._done == len(self._jobs)
        reward = self._weights["task_done"] if success else 0.0
        images = {name: engine.get_state_snapshot().imaging_count for name, engine in self._engines.items()}
        downlink = {name: engine.get_state_snapshot().data_downlink_Mb for name, engine in self._engines.items()}
        info = {
            "success": success,
            "done": self._done,
            "jobs": len(self._jobs),
            "burned": self._burned,
            "missed": 0,
            "kind": job.kind,
            "vehicle": job.vehicle,
            "engine_images": images,
            "engine_downlink_mb": downlink,
        }
        return self._obs(), float(reward), finished, False, info


class DishAwardEnv(gym.Env):
    """
    Все физические пересечения суток, по одному окну на шаг.
    Наблюдение — доля буфера, участие в окне и кто начинает раньше, по каждому аппарату.
    Действие отдаёт зеркало аппарату с этим номером.
    Успех — каждое окно у большего буфера, движок победителя закрывает станцию,
    у каждого аппарата остаётся свой сеанс.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 24):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self._keys = MISSION_KEYS
        width = 3 * len(self._keys)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(width,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(self._keys))
        self._campaign: DayCampaign | None = None
        self._drafts = ()
        self._window: SharedWindow | None = None
        self._awards: list[SharedWindow] = []
        self._weights = load_rewards()["dish_award"]

    def _obs(self) -> np.ndarray:
        share = np.zeros(len(self._keys), dtype=np.float32)
        inside = np.zeros(len(self._keys), dtype=np.float32)
        early = np.zeros(len(self._keys), dtype=np.float32)
        window = self._window
        if window is not None:
            total = window.ready_bits[0] + window.ready_bits[1]
            earliest = min(window.starts)
            for slot, name in enumerate(window.vehicles):
                index = self._keys.index(name)
                inside[index] = 1.0
                share[index] = 0.5 if total <= 0.0 else window.ready_bits[slot] / total
                if window.starts[slot] <= earliest + 1e-9:
                    early[index] = 1.0
        return np.concatenate((share, inside, early)).astype(np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._campaign = build_day_campaign()
        self._drafts = campaign_drafts(self._campaign)
        self._awards = []
        windows = shared_windows(self._drafts)
        if not windows:
            raise RuntimeError("на этих сутках зеркала не пересекаются")
        self._window = windows[0]
        return self._obs(), {
            "station": self._window.station,
            "vehicles": list(self._window.vehicles),
            "starts": list(self._window.starts),
            "ready_bits": list(self._window.ready_bits),
            "remaining": len(windows),
        }

    def _fly_awards(self) -> tuple[bool, dict[str, float]]:
        campaign = self._campaign
        if campaign is None:
            raise RuntimeError("сначала reset")
        spans: dict[str, list[tuple[float, float, str, float]]] = {}
        for window in self._awards:
            keeper = window.keeper_index()
            spans.setdefault(window.vehicles[keeper], []).append(
                (
                    window.starts[keeper],
                    window.ends[keeper],
                    window.station,
                    float(window.ready_bits[keeper]),
                )
            )
        downlink: dict[str, float] = {}
        for vehicle, rows in spans.items():
            rows.sort()
            engine = make_engine(vehicle, epoch=campaign.epoch, nu_deg=180.0, with_stations=True)
            if vehicle == campaign.vehicles[0] and campaign.burn_at_sec is not None:
                burn = next(item for item in campaign.conjunctions if item.required)
                coast_to_job(engine, campaign.burn_at_sec)
                engine.step(1.0, "maneuver", {"dv_m_s": burn.dv_m_s, "direction": burn.direction or "prograde"})
            for start, end, station, ready in rows:
                coast_to_job(engine, max(start, engine.time_sec))
                capacity = engine.spacecraft.config.payload.memory_bits
                room = ready if capacity <= 0.0 else min(ready, max(0.0, capacity - engine.stored_bits))
                if room > engine.stored_bits:
                    engine.preload_bits(room - engine.stored_bits)
                before = engine.get_state_snapshot().data_downlink_Mb
                span = max(1.0, min(60.0, end - max(start, engine.time_sec)))
                engine.step(span, "comms")
                snap = engine.get_state_snapshot()
                if not (engine.last_command_ok and snap.data_downlink_Mb > before and snap.contact_station == station):
                    downlink[vehicle] = float(snap.data_downlink_Mb)
                    return False, downlink
            downlink[vehicle] = float(engine.get_state_snapshot().data_downlink_Mb)
        return True, downlink

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._campaign is None or self._window is None:
            raise RuntimeError("сначала reset")
        choice = int(np.asarray(action).flat[0])
        window = self._window
        keeper_name = window.vehicles[window.keeper_index()]
        keeper_choice = self._keys.index(keeper_name)
        info = {
            "success": False,
            "choice": choice,
            "keeper": keeper_name,
            "station": window.station,
            "vehicles": list(window.vehicles),
            "ready_bits": list(window.ready_bits),
            "awards": len(self._awards),
        }
        if choice != keeper_choice:
            return self._obs(), -float(self._weights["missed"]), True, False, info
        self._awards.append(window)
        self._drafts = commit_award(self._drafts, window)
        remaining = shared_windows(self._drafts)
        if remaining:
            self._window = remaining[0]
            info["awards"] = len(self._awards)
            info["remaining"] = len(remaining)
            return self._obs(), 0.0, False, False, info
        self._window = None
        opened, downlink = self._fly_awards()
        dumps = {
            draft.plan.vehicle: next((item.name for item in draft.plan.required_items() if item.kind == "comms"), "")
            for draft in self._drafts
        }
        success = bool(opened and self._awards and all(dumps.values()))
        info.update(
            {
                "success": success,
                "awards": len(self._awards),
                "remaining": 0,
                "downlink_mb": downlink,
                "dumps": dumps,
                "seen": sorted({name for item in self._awards for name in item.vehicles}),
            }
        )
        reward = self._weights["awarded"] if success else -self._weights["missed"]
        return self._obs(), float(reward), True, False, info


_COMMANDS = 4


class OperatorShiftEnv(gym.Env):
    """
    Одна смена. Пока окна одной станции пересекаются, действие отдаёт зеркало.
    Затем те же часы исполняются: съёмка, сдача и импульс по уже разданным делам.
    keys, horizon_sec, epoch, nu_deg и dv_budget_m_s задают сутки.
    until="award" кончает эпизод после раздачи окон, until="full" доводит ещё и дела.
    За каждый верный шаг есть отдельная награда, в конце — награда за всю смену.
    Аппарат из окна, но не с большим буфером, получает зеркало: смена идёт по этому плану,
    шаг штрафуется, успех суток не ставится.
    Ожидание вместо текущего дела пропускает дело и оставляет смену открытой.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        keys: tuple[str, ...] | None = None,
        horizon_sec: float | None = None,
        epoch: str | None = None,
        nu_deg: float | None = None,
        dv_budget_m_s: float | None = None,
        until: str | None = None,
        scenario: str | None = None,
        max_episode_steps: int = 1600,
    ):
        super().__init__()
        cfg = load_shift_scenario(scenario)
        chosen_until = cfg["until"] if until is None else until
        if chosen_until not in {"award", "full"}:
            raise ValueError("until должен быть award или full")
        self._until = chosen_until
        self._keys = tuple(cfg["keys"] if keys is None else keys)
        if len(self._keys) < 1:
            raise ValueError("в смене нужен хотя бы один аппарат")
        self._horizon_sec = float(cfg["horizon_sec"] if horizon_sec is None else horizon_sec)
        self._epoch = cfg["epoch"] if epoch is None else epoch
        self._nu_deg = float(cfg["nu_deg"] if nu_deg is None else nu_deg)
        self._dv_budget_m_s = float(cfg["dv_budget_m_s"] if dv_budget_m_s is None else dv_budget_m_s)
        self.max_episode_steps = max_episode_steps
        width = 1 + 3 * len(self._keys) + 5
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(width,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(_COMMANDS + len(self._keys))
        self._campaign: DayCampaign | None = None
        self._drafts = ()
        self._window: SharedWindow | None = None
        self._awards: list[SharedWindow] = []
        self._wrong_awards = 0
        self._phase = "award"
        self._jobs: tuple[_Job, ...] = ()
        self._index = 0
        self._done = 0
        self._burned = False
        self._outcomes: dict[tuple[str, str, str, int], str] = {}
        self._flight: dict[str, list[dict]] = {}
        self._engines: dict[str, SimulationEngine] = {}
        self._weights = load_rewards()["operator_shift"]

    def _obs(self) -> np.ndarray:
        count = len(self._keys)
        phase = np.zeros(1, dtype=np.float32)
        share = np.zeros(count, dtype=np.float32)
        inside = np.zeros(count, dtype=np.float32)
        early = np.zeros(count, dtype=np.float32)
        job = np.zeros(5, dtype=np.float32)
        window = self._window
        if self._phase == "award" and window is not None:
            total = window.ready_bits[0] + window.ready_bits[1]
            earliest = min(window.starts)
            for slot, name in enumerate(window.vehicles):
                index = self._keys.index(name)
                inside[index] = 1.0
                share[index] = 0.5 if total <= 0.0 else window.ready_bits[slot] / total
                if window.starts[slot] <= earliest + 1e-9:
                    early[index] = 1.0
        elif self._phase == "execute" and self._campaign is not None and self._index < len(self._jobs):
            phase[0] = 1.0
            current = self._jobs[self._index]
            budget = self._campaign.dv_budget_m_s
            spent = 0.0 if budget <= 0.0 else min(1.0, self._campaign.dv_spent_m_s / budget)
            burn_due, link_open, target_lit = self._opportunity(current)
            job[:] = (
                burn_due,
                link_open,
                target_lit,
                0.0 if not self._jobs else self._done / len(self._jobs),
                spent,
            )
        return np.concatenate((phase, share, inside, early, job)).astype(np.float32)

    def _opportunity(self, job: _Job) -> tuple[float, float, float]:
        """Окно импульса, открытая линия и цель на свету. Кода команды здесь нет."""
        campaign = self._campaign
        if campaign is None:
            return 0.0, 0.0, 0.0
        burn_due = 0.0
        if (
            not self._burned
            and campaign.burn_at_sec is not None
            and abs(job.start_sec - campaign.burn_at_sec) <= 1.0
            and dict(campaign.coast_pc).get("HOT", 0.0) >= PC_THRESHOLD
        ):
            burn_due = 1.0
        link_open = 0.0
        target_lit = 0.0
        for plan in campaign.plans:
            if plan.vehicle != job.vehicle:
                continue
            for item in plan.required_items():
                if item.end_sec + 1e-6 < job.start_sec or job.start_sec + 1e-6 < item.start_sec:
                    continue
                if item.kind == "comms":
                    link_open = 1.0
                elif item.kind == "imaging":
                    target_lit = 1.0
        if burn_due:
            return 1.0, 0.0, 0.0
        if link_open:
            return 0.0, 1.0, 0.0
        if target_lit or job.kind == "imaging":
            return 0.0, 0.0, 1.0
        return burn_due, link_open, target_lit

    def _shield(self, choice: int) -> str | None:
        """Запрет команды, которую состояние не допускает. Это не штраф награды."""
        if self._phase == "award":
            window = self._window
            if window is None:
                return None
            if choice < _COMMANDS or choice >= _COMMANDS + len(self._keys):
                return "в споре за зеркало нужна команда назначения"
            if self._keys[choice - _COMMANDS] not in window.vehicles:
                return "аппарат не в этом окне"
            return None
        if self._index >= len(self._jobs):
            return None
        if choice >= _COMMANDS:
            return "назначение зеркала вне спора"
        burn_due, link_open, target_lit = self._opportunity(self._jobs[self._index])
        if choice == 3 and burn_due < 0.5:
            return "импульс вне окна сближения"
        if choice == 2 and link_open < 0.5:
            return "сдача без открытой линии"
        if choice == 1 and target_lit < 0.5:
            return "съёмка без цели на свету"
        return None

    def _open_shift(self) -> None:
        if self._campaign is None:
            raise RuntimeError("сначала reset")
        plans = tuple(draft.plan for draft in self._drafts)
        self._campaign = replace(self._campaign, plans=plans)
        self._jobs = _shift_jobs(self._campaign)
        self._index = 0
        self._done = 0
        self._burned = False
        self._outcomes = {}
        self._phase = "execute"
        self._window = None
        self._engines = {}
        for plan in plans:
            targets = [GroundTarget(name=name, lat_deg=lat, lon_deg=lon) for name, lat, lon in plan.targets]
            self._engines[plan.vehicle] = make_engine(
                plan.vehicle,
                epoch=self._campaign.epoch,
                nu_deg=self._campaign.nu_deg,
                with_stations=True,
                ground_targets=targets,
            )
        self._flight = {}
        for name in self._engines:
            self._append_flight(name)

    def _append_flight(self, name: str) -> None:
        engine = self._engines[name]
        snap = engine.get_state_snapshot()
        row = {
            "t": float(engine.time_sec),
            "r": [float(item) for item in np.asarray(engine.orbit.r, dtype=float).reshape(3)],
            "v": [float(item) for item in np.asarray(engine.orbit.v, dtype=float).reshape(3)],
            "altitude_km": float(snap.altitude_km),
            "eclipse": bool(snap.in_eclipse),
            "contact": bool(snap.in_contact),
            "station": str(snap.contact_station or ""),
        }
        track = self._flight.setdefault(name, [])
        if track and abs(track[-1]["t"] - row["t"]) < 0.25:
            track[-1] = row
        else:
            track.append(row)

    def _advance_vehicle(self, name: str, until_sec: float, releasing: _Job | None) -> None:
        """Доводит аппарат ожиданием, не проскакивая его следующее дело."""
        engine = self._engines[name]
        cap = float(until_sec)
        for job in self._jobs[self._index :]:
            if (
                releasing is not None
                and job.vehicle == name
                and job.kind == releasing.kind
                and job.name == releasing.name
                and abs(job.start_sec - releasing.start_sec) < 0.5
            ):
                continue
            if job.vehicle == name and job.end_sec > engine.time_sec + 0.5:
                cap = min(cap, max(engine.time_sec, float(job.start_sec)))
                break
        coast_to_job(engine, max(cap, engine.time_sec), on_step=lambda: self._append_flight(name))
        self._append_flight(name)

    def _advance_fleet(self, until_sec: float, releasing: _Job | None = None) -> None:
        for name in self._engines:
            self._advance_vehicle(name, until_sec, releasing)

    def _remember_job(self, job: _Job, status: str) -> None:
        self._outcomes[(job.vehicle, job.kind, job.name, round(job.start_sec))] = status

    def _engine_accepts(self, job: _Job) -> bool:
        engine = self._engines[job.vehicle]
        coast_to_job(engine, max(job.start_sec, engine.time_sec))
        if job.kind == "maneuver":
            campaign = self._campaign
            burn = next(item for item in campaign.conjunctions if item.required) if campaign is not None else None
            dv = 1.0 if burn is None else burn.dv_m_s
            before = engine.spacecraft.state.propulsion.delta_v_remaining_m_s
            engine.step(1.0, "maneuver", {"dv_m_s": dv, "direction": job.direction or "prograde"})
            spent = before - engine.spacecraft.state.propulsion.delta_v_remaining_m_s
            return bool(engine.last_command_ok and spent >= 0.5 and job.name == "HOT")
        if job.kind == "imaging":
            if job.name in engine.captured_targets():
                return True
            span = max(1.0, min(60.0, job.end_sec - max(job.start_sec, engine.time_sec)))
            engine.step(span, "imaging")
            return job.name in engine.captured_targets()
        span = max(1.0, min(60.0, job.end_sec - max(job.start_sec, engine.time_sec)))
        engine.step(span, "comms")
        return bool(engine.last_command_ok)

    def _info(self, success: bool, missed: int, violation: str | None = None) -> dict:
        images = {name: engine.get_state_snapshot().imaging_count for name, engine in self._engines.items()}
        downlink = {name: engine.get_state_snapshot().data_downlink_Mb for name, engine in self._engines.items()}
        delivered = bool(self._engines) and all(engine.delivered_bits > 0.0 for engine in self._engines.values())
        outcome = classify_outcome(
            command_accepted=violation is None,
            operation_completed=success,
            request_delivered=delivered,
            safe=violation is None,
        )
        return {
            "success": success,
            "phase": self._phase,
            "awards": len(self._awards),
            "jobs": len(self._jobs),
            "done": self._done,
            "burned": self._burned,
            "missed": missed,
            "wrong_awards": self._wrong_awards,
            "vehicles": list(self._keys),
            "horizon_sec": self._horizon_sec,
            "until": self._until,
            "unsafe": violation is not None,
            "violations": [] if violation is None else [violation],
            "engine_images": images,
            "engine_downlink_mb": downlink,
            **outcome,
        }

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._campaign = build_day_campaign(
            self._keys,
            horizon_sec=self._horizon_sec,
            epoch=self._epoch,
            nu_deg=self._nu_deg,
            dv_budget_m_s=self._dv_budget_m_s,
        )
        self._drafts = campaign_drafts(self._campaign)
        self._awards = []
        self._wrong_awards = 0
        self._jobs = ()
        self._index = 0
        self._done = 0
        self._burned = False
        self._outcomes = {}
        self._flight = {}
        self._engines = {}
        windows = shared_windows(self._drafts)
        if windows:
            self._phase = "award"
            self._window = windows[0]
        elif self._until == "award":
            self._phase = "award"
            self._window = None
        else:
            self._window = None
            self._open_shift()
        return self._obs(), {
            "phase": self._phase,
            "vehicles": list(self._keys),
            "horizon_sec": self._horizon_sec,
            "remaining": len(windows),
            "until": self._until,
        }

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._campaign is None:
            raise RuntimeError("сначала reset")
        choice = int(np.asarray(action).flat[0])
        if self._phase == "award":
            window = self._window
            if window is None:
                if self._until == "award":
                    return self._obs(), float(self._weights["task_done"]), True, False, self._info(True, 0)
                raise RuntimeError("окно смены не собрано")
            keeper_name = window.vehicles[window.keeper_index()]
            violation = self._shield(choice)
            if violation is not None:
                return self._obs(), 0.0, True, False, self._info(False, 0, violation)
            chosen = self._keys[choice - _COMMANDS]
            correct = chosen == keeper_name
            self._awards.append(window)
            self._drafts = commit_award(self._drafts, window, holder=chosen)
            if not correct:
                self._wrong_awards += 1
            remaining = shared_windows(self._drafts)
            step_reward = float(self._weights["step_done"] if correct else -self._weights["missed"])
            missed = 0 if correct else 1
            if remaining:
                self._window = remaining[0]
                return self._obs(), step_reward, False, False, self._info(False, missed)
            if self._until == "award":
                self._window = None
                success = self._wrong_awards == 0
                reward = step_reward + (float(self._weights["task_done"]) if success else 0.0)
                return self._obs(), reward, True, False, self._info(success, 0 if success else 1)
            self._open_shift()
            if not self._jobs:
                success = self._wrong_awards == 0
                reward = step_reward + (float(self._weights["task_done"]) if success else 0.0)
                return self._obs(), reward, True, False, self._info(success, 0 if success else 1)
            return self._obs(), step_reward, False, False, self._info(False, missed)
        if self._index >= len(self._jobs):
            return self._obs(), -float(self._weights["missed"]), True, False, self._info(False, 1)
        job = self._jobs[self._index]
        violation = self._shield(choice)
        if violation is not None:
            return self._obs(), 0.0, True, False, self._info(False, 0, violation)
        if _ACTION_KIND.get(choice) != job.kind:
            self._advance_fleet(job.start_sec, releasing=job)
            self._remember_job(job, "skipped")
            self._index += 1
            finished = self._index >= len(self._jobs)
            return self._obs(), -float(self._weights["missed"]), finished, False, self._info(False, 1)
        if not self._engine_accepts(job):
            self._advance_fleet(self._engines[job.vehicle].time_sec, releasing=job)
            self._remember_job(job, "skipped")
            self._index += 1
            finished = self._index >= len(self._jobs)
            return self._obs(), -float(self._weights["missed"]), finished, False, self._info(False, 1)
        self._advance_fleet(self._engines[job.vehicle].time_sec, releasing=job)
        if job.kind == "maneuver":
            self._burned = True
        self._remember_job(job, "done")
        self._done += 1
        self._index += 1
        finished = self._index >= len(self._jobs)
        needs_burn = self._campaign.burn_at_sec is not None
        success = (
            finished
            and self._done == len(self._jobs)
            and (self._burned or not needs_burn)
            and self._wrong_awards == 0
        )
        reward = float(self._weights["step_done"])
        if success:
            reward += float(self._weights["task_done"])
        return self._obs(), reward, finished, False, self._info(success, 0)
