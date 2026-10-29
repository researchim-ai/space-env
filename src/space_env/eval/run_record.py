"""
Запись исхода прогона: команды, сдача, заряд и остаток импульса.
Два одинаковых прогона дают одну запись. Запись не содержит скрытую причину отказа.
"""
from __future__ import annotations


def freeze_run(world, *, seed: int, commands: list[dict]) -> dict:
    vehicles = {}
    for name, engine in world.engines.items():
        vehicles[name] = {
            "time_sec": round(float(engine.time_sec), 6),
            "soc": round(float(engine.spacecraft.state.power.battery_soc), 9),
            "dv_remaining_m_s": round(float(engine.spacecraft.state.propulsion.delta_v_remaining_m_s), 9),
            "delivered_bits": round(float(engine.delivered_bits), 3),
            "stored_bits": round(float(engine.stored_bits), 3),
            "receipts": [
                {
                    "request_id": str(item["request_id"]),
                    "destination": str(item["destination"]),
                    "bits": round(float(item["bits"]), 3),
                    "time_sec": round(float(item["time_sec"]), 6),
                }
                for item in engine.receipts
            ],
        }
    return {
        "schema": 1,
        "seed": int(seed),
        "commands": [dict(item) for item in commands],
        "safety_events": int(world.safety_events()),
        "vehicles": vehicles,
    }


def same_outcome(left: dict, right: dict) -> bool:
    """Совпадают сдача, заряд, импульс и нарушения. Список команд сам по себе исходом не считается."""
    return left["vehicles"] == right["vehicles"] and left["safety_events"] == right["safety_events"]
