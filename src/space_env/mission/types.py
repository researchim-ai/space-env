"""
Типы миссий: LEO, GEO, MEO и шаблоны целей.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class MissionType(str, Enum):
    LEO = "LEO"
    GEO = "GEO"
    MEO = "MEO"
    INTERPLANETARY = "interplanetary"


class MissionProfile(BaseModel):
    """Профиль миссии: тип и типичные параметры (для шаблонов)."""
    mission_type: MissionType = MissionType.LEO
    name: str = ""
    description: str = ""
    # Ориентиры для орбиты (м)
    typical_altitude_km: float = Field(default=400.0, ge=200.0, le=50000.0)
    # Цели по умолчанию задаются в MissionGoal
