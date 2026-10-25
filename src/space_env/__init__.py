"""
space-env: среда для генерации данных и обучения RL/LLM агентов в космической отрасли.
"""

from space_env.core import (
    Orbit,
    Spacecraft,
    SpacecraftConfig,
    Time,
)
# Среды поднимаются раньше каталога: план миссии импортирует сборку каталога.
from space_env.envs import registered_env_ids
from space_env.envs.cards import ENV_CARDS
from space_env.catalog import make_engine
from space_env.mission import (
    build_acquisition_plan,
    build_day_campaign,
    build_conjunction_plan,
    build_relay_schedule,
    build_reroute_schedule,
    build_two_burn_phase_plan,
    execute_decay_plan,
)

__version__ = "0.84.0"
__all__ = [
    "ENV_CARDS",
    "Orbit",
    "Spacecraft",
    "SpacecraftConfig",
    "Time",
    "__version__",
    "build_acquisition_plan",
    "build_day_campaign",
    "build_conjunction_plan",
    "build_relay_schedule",
    "build_reroute_schedule",
    "build_two_burn_phase_plan",
    "execute_decay_plan",
    "make_engine",
    "registered_env_ids",
]
