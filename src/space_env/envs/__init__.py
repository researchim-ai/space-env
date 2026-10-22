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
from space_env.envs.market_envs import OrderBookEnv, StationContentionEnv
from space_env.envs.sentinel_env import (
    CubesatAcquisitionEnv,
    LandsatAcquisitionEnv,
    SentinelAcquisitionEnv,
    SentinelPassFaultEnv,
)
from space_env.envs.shift_env import FleetShiftEnv
from space_env.envs.industry_envs import (
    DebrisAvoidEnv,
    DisposalEnv,
    PassFaultEnv,
    PriorityCoverageEnv,
    RendezvousHoldEnv,
)
from space_env.envs.horizon_envs import (
    ConjunctionQueueEnv,
    DecayWatchEnv,
    DecayWatchHighEnv,
    PhaseBurnEnv,
    RerouteEnv,
)
from space_env.envs.campaign_env import DayMissionEnv, DayShiftEnv, DishAwardEnv, OperatorShiftEnv
from space_env.envs.regime_envs import LongitudeKeepEnv, RadarPassEnv
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
    ("space_env/ConjunctionQueue-v0", "space_env.envs.horizon_envs:ConjunctionQueueEnv"),
    ("space_env/DecayWatch-v0", "space_env.envs.horizon_envs:DecayWatchEnv"),
    ("space_env/DecayWatchHigh-v0", "space_env.envs.horizon_envs:DecayWatchHighEnv"),
    ("space_env/PhaseBurn-v0", "space_env.envs.horizon_envs:PhaseBurnEnv"),
    ("space_env/Reroute-v0", "space_env.envs.horizon_envs:RerouteEnv"),
    ("space_env/FleetShift-v0", "space_env.envs.shift_env:FleetShiftEnv"),
    ("space_env/RendezvousHold-v0", "space_env.envs.industry_envs:RendezvousHoldEnv"),
    ("space_env/PriorityCoverage-v0", "space_env.envs.industry_envs:PriorityCoverageEnv"),
    ("space_env/PassFault-v0", "space_env.envs.industry_envs:PassFaultEnv"),
    ("space_env/DebrisAvoid-v0", "space_env.envs.industry_envs:DebrisAvoidEnv"),
    ("space_env/Disposal-v0", "space_env.envs.industry_envs:DisposalEnv"),
    ("space_env/StationContention-v0", "space_env.envs.market_envs:StationContentionEnv"),
    ("space_env/OrderBook-v0", "space_env.envs.market_envs:OrderBookEnv"),
    ("space_env/SentinelPassFault-v0", "space_env.envs.sentinel_env:SentinelPassFaultEnv"),
    ("space_env/CubesatAcquisition-v0", "space_env.envs.sentinel_env:CubesatAcquisitionEnv"),
    ("space_env/DayMission-v0", "space_env.envs.campaign_env:DayMissionEnv"),
    ("space_env/DayShift-v0", "space_env.envs.campaign_env:DayShiftEnv"),
    ("space_env/DishAward-v0", "space_env.envs.campaign_env:DishAwardEnv"),
    ("space_env/OperatorShift-v0", "space_env.envs.campaign_env:OperatorShiftEnv"),
    ("space_env/RadarPass-v0", "space_env.envs.regime_envs:RadarPassEnv"),
    ("space_env/LongitudeKeep-v0", "space_env.envs.regime_envs:LongitudeKeepEnv"),
)

def registered_env_ids() -> tuple[str, ...]:
    """Идентификаторы в порядке регистрации. Импорт пакета уже вносит их в Gymnasium."""
    return tuple(env_id for env_id, _entry in _SPECS)


for _env_id, _entry in _SPECS:
    if _env_id not in registry:
        register(id=_env_id, entry_point=_entry)

__all__ = [
    "AnomalyResponseEnv",
    "ConjunctionAssessmentEnv",
    "ConjunctionQueueEnv",
    "DebrisAvoidEnv",
    "DayMissionEnv",
    "DayShiftEnv",
    "DishAwardEnv",
    "OperatorShiftEnv",
    "DecayWatchEnv",
    "DisposalEnv",
    "DecayWatchHighEnv",
    "FleetShiftEnv",
    "ConstellationEnv",
    "CubesatAcquisitionEnv",
    "ConstellationOpsEnv",
    "LandsatAcquisitionEnv",
    "LongitudeKeepEnv",
    "MissionPlannerEnv",
    "OrderBookEnv",
    "PassFaultEnv",
    "PriorityCoverageEnv",
    "RendezvousHoldEnv",
    "PhaseBurnEnv",
    "RadarPassEnv",
    "RerouteEnv",
    "ResourceManagerEnv",
    "registered_env_ids",
    "SentinelAcquisitionEnv",
    "SentinelPassFaultEnv",
    "StationContentionEnv",
    "SentinelPhasingEnv",
    "WorkPlanEnv",
]
