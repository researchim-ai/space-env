"""Пульт смены: один живой эпизод OperatorShift для человека, не для проигрывателя."""
from __future__ import annotations

import threading

import gymnasium as gym

import space_env.envs  # регистрация сред
from space_env.mission.campaign import shared_windows

_LOCK = threading.Lock()
_ENV = None
_FINISHED = True
_LOG: list[dict] = []

_KIND = {"imaging": "съёмка", "comms": "сдача", "maneuver": "импульс"}
_DIRECTION_RU = {
    "prograde": "по скорости",
    "retrograde": "против скорости",
    "normal": "по нормали",
    "radial": "по радиусу",
}


def _close() -> None:
    global _ENV
    if _ENV is not None:
        _ENV.close()
        _ENV = None


def _action_label(env, action: int) -> str:
    raw = env.unwrapped
    labels = ["Ждать", "Съёмка", "Сдача", "Импульс"]
    labels.extend(f"Зеркало {name}" for name in raw._keys)
    if 0 <= action < len(labels):
        return labels[action]
    return str(action)


def _remember(label: str, view: dict, phase_before: str) -> None:
    if view["unsafe"]:
        text = f"{label}: отвергнуто. {'; '.join(view['violations'])}"
    elif phase_before == "award":
        text = f"{label}: раздача, награда {view['reward']}. Чужих зеркал {view['wrong_awards']}."
    elif view["terminated"]:
        outcome = "успех" if view["success"] else "без успеха"
        text = f"{label}: смена закрыта, {outcome}. Дел {view['done']} из {view['jobs']}."
    elif view["reward"] is not None and view["reward"] < 0:
        job = view.get("job")
        follow = f" Дальше {job['vehicle']}: {job['kind_label']}." if job else ""
        text = f"{label}: дело пропущено, награда {view['reward']}.{follow}"
    else:
        job = view.get("job")
        follow = ""
        if job:
            follow = f" Дальше {job['vehicle']}: {job['kind_label']}."
        text = f"{label}: исполнено {view['done']} из {view['jobs']}.{follow}"
    _LOG.append(
        {
            "action": label,
            "reward": view["reward"],
            "success": bool(view["success"]),
            "unsafe": bool(view["unsafe"]),
            "done": int(view["done"]),
            "summary": text,
        }
    )


def _actions(env) -> list[dict]:
    raw = env.unwrapped
    labels = ["Ждать", "Съёмка", "Сдача", "Импульс"]
    labels.extend(f"Зеркало {name}" for name in raw._keys)
    rows = []
    for index, label in enumerate(labels):
        blocked = raw._shield(index)
        rows.append({"id": index, "label": label, "blocked": blocked})
    return rows


def _window(env) -> dict | None:
    raw = env.unwrapped
    window = raw._window
    if raw._phase != "award" or window is None:
        return None
    return {
        "station": window.station,
        "rows": [
            {
                "vehicle": name,
                "ready_bits": float(window.ready_bits[slot]),
                "start_sec": float(window.starts[slot]),
            }
            for slot, name in enumerate(window.vehicles)
        ],
    }


def _job(env) -> dict | None:
    raw = env.unwrapped
    if raw._phase != "execute" or raw._index >= len(raw._jobs):
        return None
    job = raw._jobs[raw._index]
    burn_due, link_open, target_lit = raw._opportunity(job)
    return {
        "kind": job.kind,
        "kind_label": _KIND.get(job.kind, job.kind),
        "vehicle": job.vehicle,
        "name": job.name,
        "start_sec": float(job.start_sec),
        "end_sec": float(job.end_sec),
        "burn_due": burn_due > 0.5,
        "link_open": link_open > 0.5,
        "target_lit": target_lit > 0.5,
    }


def _queue(env) -> list[dict]:
    """Все ещё не разобранные споры. Первый — тот, который решает текущая команда."""
    raw = env.unwrapped
    if raw._phase != "award":
        return []
    rows = []
    for index, window in enumerate(shared_windows(raw._drafts)):
        rows.append(
            {
                "current": index == 0,
                "station": window.station,
                "start_sec": float(min(window.starts)),
                "vehicles": [
                    {
                        "vehicle": name,
                        "ready_bits": float(window.ready_bits[slot]),
                        "start_sec": float(window.starts[slot]),
                    }
                    for slot, name in enumerate(window.vehicles)
                ],
            }
        )
    return rows


def _ledger(env) -> list[dict]:
    """Съёмки и сдача, которые движки уже записали. До исполнения — нули."""
    raw = env.unwrapped
    rows = []
    for name in raw._keys:
        engine = raw._engines.get(name)
        if engine is None:
            rows.append({"vehicle": name, "images": 0, "downlink_mb": 0.0, "dv_left_m_s": None})
            continue
        snap = engine.get_state_snapshot()
        rows.append(
            {
                "vehicle": name,
                "images": int(snap.imaging_count),
                "downlink_mb": float(snap.data_downlink_Mb),
                "dv_left_m_s": float(engine.spacecraft.state.propulsion.delta_v_remaining_m_s),
            }
        )
    return rows


def _burn_note(burn) -> str:
    note = burn.reason
    for english, russian in _DIRECTION_RU.items():
        note = note.replace(f"м/с {english}", f"м/с {russian}")
    return note


def _job_status(raw, vehicle: str, kind: str, name: str, start_sec: float, required: bool) -> str:
    if not required:
        return "denied"
    return raw._outcomes.get((vehicle, kind, name, round(start_sec)), "pending")


def _journal(env) -> list[dict]:
    """План открытой смены: сдача, отказ зеркала, съёмка и импульс со статусом шага."""
    raw = env.unwrapped
    rows: list[dict] = []
    for draft in raw._drafts:
        for item in draft.plan.items:
            if item.kind == "imaging" and not item.required:
                continue
            if item.kind == "comms" and not item.required and "Зеркало занято" not in item.reason:
                continue
            if item.kind not in {"imaging", "comms"}:
                continue
            rows.append(
                {
                    "vehicle": draft.plan.vehicle,
                    "kind": item.kind,
                    "name": item.name,
                    "start_sec": float(item.start_sec),
                    "end_sec": float(item.end_sec),
                    "required": bool(item.required),
                    "note": item.reason,
                    "status": _job_status(
                        raw, draft.plan.vehicle, item.kind, item.name, item.start_sec, item.required
                    ),
                }
            )
    campaign = raw._campaign
    if campaign is not None and campaign.burn_at_sec is not None and campaign.vehicles:
        burn = next((item for item in campaign.conjunctions if item.required), None)
        if burn is not None:
            start = float(campaign.burn_at_sec)
            vehicle = campaign.vehicles[0]
            rows.append(
                {
                    "vehicle": vehicle,
                    "kind": "maneuver",
                    "name": burn.name,
                    "start_sec": start,
                    "end_sec": start,
                    "required": True,
                    "note": _burn_note(burn),
                    "status": _job_status(raw, vehicle, "maneuver", burn.name, start, True),
                }
            )
    rows.sort(key=lambda row: (row["start_sec"], row["vehicle"], row["kind"]))
    return rows


def _plan(env) -> tuple[list[dict], list[dict]]:
    """Обязательные сдачи и импульс текущего плана. Съёмки — отдельным счётом."""
    raw = env.unwrapped
    current = None
    if raw._phase == "execute" and raw._index < len(raw._jobs):
        job = raw._jobs[raw._index]
        current = (job.vehicle, job.kind, round(job.start_sec))
    rows: list[dict] = []
    imaging = {name: 0 for name in raw._keys}
    for draft in raw._drafts:
        for item in draft.plan.items:
            if item.kind == "imaging" and item.required:
                imaging[draft.plan.vehicle] = imaging.get(draft.plan.vehicle, 0) + 1
                continue
            if item.kind != "comms":
                continue
            if not item.required and "Зеркало занято" not in item.reason:
                continue
            start = float(item.start_sec)
            rows.append(
                {
                    "vehicle": draft.plan.vehicle,
                    "kind": "comms",
                    "kind_label": "сдача" if item.required else "не сдавать",
                    "name": item.name,
                    "start_sec": start,
                    "end_sec": float(item.end_sec),
                    "required": bool(item.required),
                    "reason": item.reason,
                    "current": (draft.plan.vehicle, "comms", round(start)) == current,
                }
            )
    campaign = raw._campaign
    if campaign is not None and campaign.burn_at_sec is not None and campaign.vehicles:
        burn = next((item for item in campaign.conjunctions if item.required), None)
        if burn is not None:
            start = float(campaign.burn_at_sec)
            vehicle = campaign.vehicles[0]
            rows.append(
                {
                    "vehicle": vehicle,
                    "kind": "maneuver",
                    "kind_label": "импульс",
                    "name": burn.name,
                    "start_sec": start,
                    "end_sec": start,
                    "required": True,
                    "reason": burn.reason,
                    "current": (vehicle, "maneuver", round(start)) == current,
                }
            )
    rows.sort(key=lambda row: (row["start_sec"], row["vehicle"], row["kind"]))
    counts = [{"vehicle": name, "count": imaging.get(name, 0)} for name in raw._keys]
    return rows, counts


def _notes(env) -> list[str]:
    raw = env.unwrapped
    found = []
    for draft in raw._drafts:
        for item in draft.plan.items:
            if item.kind == "comms" and not item.required and "занято" in item.reason:
                found.append(item.reason)
    return found


def _burn_decision(raw) -> str:
    """Импульс ещё впереди, уже сделан или пропущен. До команды картина — дуга без него."""
    campaign = raw._campaign
    if campaign is None or campaign.burn_at_sec is None:
        return "none"
    if raw._burned:
        return "done"
    if raw._phase == "execute" and not any(job.kind == "maneuver" for job in raw._jobs[raw._index :]):
        return "skipped"
    return "pending"


def _screening(env) -> dict:
    """Pc без импульса и после него. Активная колонка — та дуга, которая сейчас на картинке."""
    raw = env.unwrapped
    campaign = raw._campaign
    decision = _burn_decision(raw)
    notes = {
        "pending": "Импульс ещё не сделан. Картина и Pc — дуга без импульса.",
        "skipped": "Импульс пропущен. Горячая пара остаётся на дуге без импульса.",
        "done": "Импульс сделан. Пара уведена, картина идёт по дуге с импульсом.",
        "none": "",
    }
    if campaign is None:
        return {"arc": "flown", "decision": "none", "rows": [], "note": ""}
    coast = dict(campaign.coast_pc)
    flown = dict(campaign.flown_pc)
    arc = "flown" if decision == "done" else "coast"
    rows = []
    for item in campaign.conjunctions:
        coast_pc = float(coast.get(item.name, item.probability))
        flown_pc = float(flown.get(item.name, item.probability))
        rows.append(
            {
                "name": item.name,
                "t_sec": float(item.start_sec),
                "required": bool(item.required),
                "coast_pc": coast_pc,
                "flown_pc": flown_pc,
                "active_pc": flown_pc if arc == "flown" else coast_pc,
                "direction": item.direction or "",
                "dv_m_s": float(item.dv_m_s),
            }
        )
    rows.sort(key=lambda row: (-row["coast_pc"], row["t_sec"], row["name"]))
    return {"arc": arc, "decision": decision, "rows": rows, "note": notes[decision]}


def _flight(env) -> dict:
    """Уже проинтегрированный полёт открытой смены. Дальше этого часа картина расчётная."""
    raw = env.unwrapped
    tracks = getattr(raw, "_flight", {}) or {}
    clocks = {name: float(rows[-1]["t"]) for name, rows in tracks.items() if rows}
    return {
        "time_sec": min(clocks.values()) if clocks else 0.0,
        "clocks": clocks,
        "vehicles": tracks,
    }


def _focus_sec(env) -> float | None:
    window = _window(env)
    if window is not None:
        return min(row["start_sec"] for row in window["rows"])
    job = _job(env)
    if job is not None:
        return float(job["start_sec"])
    return None


def _view(env, info: dict, reward: float | None, terminated: bool, truncated: bool) -> dict:
    raw = env.unwrapped
    focus = _focus_sec(env)
    plan, imaging_planned = _plan(env)
    return {
        "open": True,
        "phase": raw._phase,
        "until": raw._until,
        "vehicles": list(raw._keys),
        "window": _window(env),
        "focus_sec": None if focus is None else float(focus),
        "notes": _notes(env),
        "queue": _queue(env),
        "plan": plan,
        "journal": _journal(env),
        "flight": _flight(env),
        "imaging_planned": imaging_planned,
        "ledger": _ledger(env),
        "remaining_windows": len(shared_windows(raw._drafts)) if raw._phase == "award" else 0,
        "job": _job(env),
        "actions": _actions(env),
        "reward": None if reward is None else float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": bool(info.get("success", False)),
        "unsafe": bool(info.get("unsafe", False)),
        "violations": list(info.get("violations") or []),
        "wrong_awards": int(info.get("wrong_awards", raw._wrong_awards)),
        "awards": int(info.get("awards", len(raw._awards))),
        "done": int(info.get("done", raw._done)),
        "jobs": int(info.get("jobs", len(raw._jobs))),
        "burned": bool(info.get("burned", raw._burned)),
        "screening": _screening(env),
    }


def open_shift(seed: int = 0) -> dict:
    """Новая смена. Предыдущий эпизод закрывается."""
    global _ENV, _FINISHED
    with _LOCK:
        _close()
        _ENV = gym.make("space_env/OperatorShift-v0", max_episode_steps=2000)
        _observation, info = _ENV.reset(seed=int(seed))
        _FINISHED = False
        _LOG.clear()
        del _observation
        view = _view(_ENV, info, None, False, False)
        view["log"] = []
        return view


def step_shift(action: int) -> dict:
    """Один шаг пульта. Закрытую смену повторно не двигает."""
    global _FINISHED
    with _LOCK:
        if _ENV is None or _FINISHED:
            raise RuntimeError("смена не открыта")
        choice = int(action)
        if choice < 0 or choice >= _ENV.action_space.n:
            raise ValueError("нет такой команды")
        phase_before = _ENV.unwrapped._phase
        _observation, reward, terminated, truncated, info = _ENV.step(choice)
        del _observation
        _FINISHED = bool(terminated or truncated)
        view = _view(_ENV, info, float(reward), bool(terminated), bool(truncated))
        _remember(_action_label(_ENV, choice), view, phase_before)
        view["log"] = [dict(item) for item in _LOG]
        return view


def shift_is_open() -> bool:
    return _ENV is not None and not _FINISHED
