"""
DayMission: принять или сломать уже посчитанные сутки.
Успех — расход Δv внутри бюджета, фрагмент без импульса, у Sentinel-2A есть сдача.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from space_env.mission.campaign import DayCampaign, build_day_campaign
from space_env.mission.horizon import PC_THRESHOLD
from space_env.rl.rewards import load_rewards


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
