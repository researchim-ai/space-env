"""
space-env: среда для генерации данных и обучения RL/LLM агентов в космической отрасли.
"""

from space_env.core import (
    Orbit,
    Spacecraft,
    SpacecraftConfig,
    Time,
)
from space_env.envs import (
    AnomalyResponseEnv,
    ConjunctionAssessmentEnv,
    MissionPlannerEnv,
    ResourceManagerEnv,
)

__version__ = "0.6.0"
__all__ = [
    "__version__",
    "AnomalyResponseEnv",
    "ConjunctionAssessmentEnv",
    "MissionPlannerEnv",
    "Orbit",
    "ResourceManagerEnv",
    "Spacecraft",
    "SpacecraftConfig",
    "Time",
]
