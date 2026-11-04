"""Current geometric reference pointing; snapshot goals, never forecasts."""

import numpy as np


def resolve_slew_goal(backend, command):
    """Resolve once at admission. Existing controllers subsequently hold GCRS."""
    from space_env.reference.spacecraft_bus import pointing_quaternion

    goal = command.slew
    if goal.quaternion_wxyz is not None:
        q = np.asarray(goal.quaternion_wxyz)
        return (q / np.linalg.norm(q)).tolist()
    if backend.bus is None or command.actor_id not in backend.bus.models:
        raise ValueError("reference slew requires configured spacecraft bus")
    references = (backend.acquisition_targets if goal.target_id is not None
                  else backend.mission.stations)
    key = goal.target_id if goal.target_id is not None else goal.station_id
    if key not in references:
        raise ValueError("unknown slew reference")
    return pointing_quaternion(
        backend.mission.ephemerides[command.actor_id], references[key], backend.now()
    ).tolist()
