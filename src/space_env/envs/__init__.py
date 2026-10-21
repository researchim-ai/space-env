"""
RL-среды (Gymnasium): планирование миссий, ресурсы, аномалии, конъюнкции.
Идентификаторы: space_env/MissionPlanner-v0, ResourceManager-v0,
AnomalyResponse-v0, ConjunctionAssessment-v0.
"""
from gymnasium.envs.registration import register, registry

from space_env.envs.anomaly_response_env import AnomalyResponseEnv
from space_env.envs.conjunction_env import ConjunctionAssessmentEnv
from space_env.envs.constellation_env import ConstellationEnv
from space_env.envs.constellation_ops_env import ConstellationOpsEnv
from space_env.envs.mission_planner_env import MissionPlannerEnv
from space_env.envs.phasing_env import SentinelPhasingEnv
from space_env.envs.resource_manager_env import ResourceManagerEnv
from space_env.envs.sentinel_env import LandsatAcquisitionEnv, SentinelAcquisitionEnv
from space_env.envs.workplan_env import WorkPlanEnv

_SPECS = (
    ("space_env/MissionPlanner-v0", "space_env.envs.mission_planner_env:MissionPlannerEnv"),
    ("space_env/ResourceManager-v0", "space_env.envs.resource_manager_env:ResourceManagerEnv"),
    ("space_env/AnomalyResponse-v0", "space_env.envs.anomaly_response_env:AnomalyResponseEnv"),
    ("space_env/ConjunctionAssessment-v0", "space_env.envs.conjunction_env:ConjunctionAssessmentEnv"),
    ("space_env/Constellation-v0", "space_env.envs.constellation_env:ConstellationEnv"),
    ("space_env/ConstellationOps-v0", "space_env.envs.constellation_ops_env:ConstellationOpsEnv"),
    ("space_env/SentinelAcquisition-v0", "space_env.envs.sentinel_env:SentinelAcquisitionEnv"),
    ("space_env/LandsatAcquisition-v0", "space_env.envs.sentinel_env:LandsatAcquisitionEnv"),
    ("space_env/SentinelPhasing-v0", "space_env.envs.phasing_env:SentinelPhasingEnv"),
    ("space_env/WorkPlan-v0", "space_env.envs.workplan_env:WorkPlanEnv"),
)

for _env_id, _entry in _SPECS:
    if _env_id not in registry:
        register(id=_env_id, entry_point=_entry)

__all__ = [
    "AnomalyResponseEnv",
    "ConjunctionAssessmentEnv",
    "ConstellationEnv",
    "ConstellationOpsEnv",
    "LandsatAcquisitionEnv",
    "MissionPlannerEnv",
    "ResourceManagerEnv",
    "SentinelAcquisitionEnv",
    "SentinelPhasingEnv",
    "WorkPlanEnv",
]
