"""Explicit reference physics, isolated from legacy-training mission dynamics."""
from space_env.reference.frames import FrameContext
from space_env.reference.orbit import (
    ForceModel,
    ReferenceState,
    acceleration,
    propagate,
)
from space_env.reference.time import EOPSeries, Epoch, Orientation

__all__ = ["EOPSeries", "Epoch", "ForceModel", "FrameContext", "Orientation", "ReferenceState", "acceleration", "propagate"]
