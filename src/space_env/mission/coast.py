"""Power-aware attitude management while a scheduled executor waits for its job."""
from __future__ import annotations

import math
from collections.abc import Callable

from space_env.simulation.engine import SimulationEngine


def coast_to_job(
    engine: SimulationEngine, at_sec: float, *, on_step: Callable[[], None] | None = None,
) -> None:
    """Charge during long idle spans, reserving time to return to nadir.

    The high-level shift still chooses the scheduled job. This low-level executor
    uses ordinary pointing commands, slew dynamics and battery accounting. It does
    not replenish charge directly or move a job's deadline.
    """
    if not math.isfinite(at_sec) or at_sec < engine.time_sec - .5:
        raise ValueError("время задания должно быть конечным и не в прошлом")
    cfg = engine.spacecraft.config
    # Worst-case 180-degree return, acceleration/deceleration and one second margin.
    reserve = 180.0 / cfg.slew_rate_deg_s + 1.0
    if cfg.slew_accel_deg_s2 > 0:
        reserve += 2.0 * cfg.slew_rate_deg_s / cfg.slew_accel_deg_s2
    charge_until = at_sec - reserve
    while engine.time_sec + .5 < at_sec:
        remaining = at_sec - engine.time_sec
        mode = engine.spacecraft.state.pointing
        command = "wait"
        dt = min(60.0, remaining)
        if remaining > 2.0 * reserve and mode != "sun":
            command = "point_sun"
        elif remaining <= reserve + 1e-9 and mode != "nadir":
            command = "point_nadir"
        if (mode == "sun" or command == "point_sun") and charge_until > engine.time_sec + 1e-9:
            dt = min(dt, charge_until - engine.time_sec)
        engine.step(dt, command)
        if on_step is not None:
            on_step()
