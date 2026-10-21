"""
Проектные соседи Walker-star и путь по ним.
Связь считается открытой, если это сосед схемы и луч не пересекает лимб.
ЭИИМ в это условие не входит: для Iridium он не опубликован.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from space_env.catalog.geometry import mean_semimajor_axis_m
from space_env.catalog.references import reference
from space_env.core.frames import elevation_deg
from space_env.core.link import earth_occults
from space_env.core.time import Time


@dataclass(frozen=True)
class ShellSlot:
    plane: int
    slot: int
    raan_deg: float
    nu_deg: float


def shell_slots(total: int, planes: int, phasing: int) -> list[ShellSlot]:
    """Узлы звезды. Долгота узла покрывает 180°, аномалия — как у Walker T/P/F."""
    if planes < 1 or total < 2 or total % planes != 0:
        raise ValueError("T должно делиться на P")
    if not 0 <= phasing < planes:
        raise ValueError("F должен быть в диапазоне 0 … P−1")
    per_plane = total // planes
    found: list[ShellSlot] = []
    for plane in range(planes):
        raan = 180.0 * plane / planes
        for slot in range(per_plane):
            nu = (360.0 * slot / per_plane + 360.0 * phasing * plane / total) % 360.0
            found.append(ShellSlot(plane, slot, raan, nu))
    return found


def _slot(plane: int, index: int, total: int, planes: int, phasing: int) -> ShellSlot:
    per_plane = total // planes
    plane_i = plane % planes
    slot_i = index % per_plane
    nu = (360.0 * slot_i / per_plane + 360.0 * phasing * plane_i / total) % 360.0
    return ShellSlot(plane_i, slot_i, 180.0 * plane_i / planes, nu)


def designed_neighbors(node: ShellSlot, total: int, planes: int, phasing: int) -> dict[str, ShellSlot]:
    """Впереди и сзади в плоскости, и тот же слот в двух соседних плоскостях, включая шов звезды."""
    return {
        "fore": _slot(node.plane, node.slot + 1, total, planes, phasing),
        "aft": _slot(node.plane, node.slot - 1, total, planes, phasing),
        "plus": _slot(node.plane + 1, node.slot, total, planes, phasing),
        "minus": _slot(node.plane - 1, node.slot, total, planes, phasing),
    }


def circular_position_m(a_m: float, inclination_deg: float, raan_deg: float, nu_deg: float) -> np.ndarray:
    """Круговая орбита, аргумент перигея ноль. Ось Z — север."""
    nu = np.deg2rad(nu_deg)
    inclination = np.deg2rad(inclination_deg)
    raan = np.deg2rad(raan_deg)
    perifocal = float(a_m) * np.array([np.cos(nu), np.sin(nu), 0.0])
    cosine = np.cos(inclination)
    sine = np.sin(inclination)
    intermediate = np.array(
        [perifocal[0], cosine * perifocal[1] - sine * perifocal[2], sine * perifocal[1] + cosine * perifocal[2]]
    )
    cosine = np.cos(raan)
    sine = np.sin(raan)
    return np.array(
        [
            cosine * intermediate[0] - sine * intermediate[1],
            sine * intermediate[0] + cosine * intermediate[1],
            intermediate[2],
        ]
    )


def link_open(left: ShellSlot, right: ShellSlot, a_m: float, inclination_deg: float) -> bool:
    start = circular_position_m(a_m, inclination_deg, left.raan_deg, left.nu_deg)
    end = circular_position_m(a_m, inclination_deg, right.raan_deg, right.nu_deg)
    return not earth_occults(start, end)


def route(
    start: ShellSlot,
    goal: ShellSlot,
    *,
    a_m: float,
    inclination_deg: float,
    total: int,
    planes: int,
    phasing: int,
) -> list[str] | None:
    """
    Кратчайшие имена переходов fore/aft/plus/minus.
    Закрытый Землёй луч в путь не входит. None — пути нет.
    """
    if (start.plane, start.slot) == (goal.plane, goal.slot):
        return []
    queue: list[tuple[ShellSlot, list[str]]] = [(start, [])]
    seen = {(start.plane, start.slot)}
    while queue:
        node, path = queue.pop(0)
        if len(path) >= total:
            continue
        for name, nxt in designed_neighbors(node, total, planes, phasing).items():
            key = (nxt.plane, nxt.slot)
            if key in seen or not link_open(node, nxt, a_m, inclination_deg):
                continue
            step = path + [name]
            if key == (goal.plane, goal.slot):
                return step
            seen.add(key)
            queue.append((nxt, step))
    return None


def slots_above(
    slots: list[ShellSlot],
    a_m: float,
    inclination_deg: float,
    epoch: Time,
    lat_deg: float,
    lon_deg: float,
    *,
    alt_m: float = 0.0,
    mask_deg: float = 5.0,
) -> list[ShellSlot]:
    """Слоты, у которых угол места над точкой не ниже маски. Радиолиния фидера сюда не входит."""
    seen: list[ShellSlot] = []
    for slot in slots:
        position = circular_position_m(a_m, inclination_deg, slot.raan_deg, slot.nu_deg)
        if elevation_deg(position, lat_deg, lon_deg, alt_m, epoch) >= mask_deg:
            seen.append(slot)
    return seen


def route_to_gateway(
    start: ShellSlot,
    gateway_lat_deg: float,
    gateway_lon_deg: float,
    *,
    a_m: float,
    inclination_deg: float,
    epoch: Time,
    total: int,
    planes: int,
    phasing: int,
    alt_m: float = 0.0,
    mask_deg: float = 5.0,
) -> tuple[ShellSlot, list[str]] | None:
    """Кратчайший путь до любого слота над шлюзом. None, если ни один слот не выше маски."""
    slots = shell_slots(total, planes, phasing)
    visible = slots_above(
        slots, a_m, inclination_deg, epoch, gateway_lat_deg, gateway_lon_deg, alt_m=alt_m, mask_deg=mask_deg
    )
    best: tuple[ShellSlot, list[str]] | None = None
    for candidate in visible:
        path = route(
            start,
            candidate,
            a_m=a_m,
            inclination_deg=inclination_deg,
            total=total,
            planes=planes,
            phasing=phasing,
        )
        if path is None:
            continue
        if best is None or len(path) < len(best[1]):
            best = (candidate, path)
    return best


def iridium_shell() -> tuple[list[ShellSlot], float, float]:
    """66/6/2 на высоте и наклонении карточки. Возвращает слоты, большую полуось и наклонение."""
    spec = reference("iridium-next")
    slots = shell_slots(66, 6, 2)
    return slots, mean_semimajor_axis_m(spec.altitude_m), spec.inclination_deg
