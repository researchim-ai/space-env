"""
Группировка: несколько полноценных аппаратов на общих часах.
Каждый ведёт свой SimulationEngine (батарея, тепло, связь, съёмка).
Между парами считается геометрия RTN, межспутниковая линия и скрининг Pc.
Буфер кадра сбрасывается на станцию или соседу.
"""
from __future__ import annotations

from dataclasses import dataclass

import math

import numpy as np

from space_env.core.cdm import ConjunctionMessage
from space_env.core.link import earth_occults, isl_margin_db, isl_rate_bps
from space_env.core.orbit import Orbit
from space_env.core.probability import bplane_probability
from space_env.core.relative import cw_stm, eci_to_rtn, mean_motion_rad_s
from space_env.core.spacecraft import Spacecraft, SpacecraftConfig
from space_env.core.time import Time
from space_env.mission.scenarios import ScenarioConfig
from space_env.simulation.engine import SimulationEngine, _covered_seconds, _merge_intervals


@dataclass(frozen=True)
class PairGeometry:
    leader: str
    follower: str
    range_m: float
    range_rate_m_s: float
    radial_m: float
    along_m: float
    cross_m: float
    earth_blocked: bool
    isl_margin_db: float


@dataclass(frozen=True)
class ScreenHit:
    leader: str
    follower: str
    range_m: float
    probability: float


@dataclass
class _StoredCdm:
    position_m2: np.ndarray
    state_si: np.ndarray | None
    radius_m: float
    tca: Time | None


class Fleet:
    """Именованные аппараты. Шаг один и тот же для всех."""

    def __init__(self, vehicles: dict[str, SimulationEngine]) -> None:
        if len(vehicles) < 2:
            raise ValueError("в группировке минимум два аппарата")
        self.vehicles = dict(vehicles)
        self._cdm: dict[tuple[str, str], _StoredCdm] = {}
        self._relay_used = {name: [] for name in self.vehicles}

    def use_reservations(self, book: dict[str, list]) -> None:
        """Книга занятости терминала живёт в мире и переживает новую обёртку Fleet."""
        for name in self.vehicles:
            book.setdefault(name, [])
        self._relay_used = book

    @classmethod
    def string_of_pearls(
        cls,
        count: int = 3,
        separation_deg: float = 10.0,
        *,
        epoch: str = "2024-01-01T00:00:00Z",
        a_m: float = 6800e3,
        inc_deg: float = 51.6,
        scenario: ScenarioConfig | None = None,
        names: list[str] | None = None,
    ) -> "Fleet":
        """Одна плоскость, истинная аномалия с шагом separation_deg. Это цепочка, не полный Walker."""
        if count < 2:
            raise ValueError("в цепочке минимум два аппарата")
        template = scenario or ScenarioConfig()
        moment = Time(epoch)
        vehicles: dict[str, SimulationEngine] = {}
        for index in range(count):
            name = names[index] if names else f"SC-{index + 1}"
            vehicles[name] = _make_vehicle(
                name, a_m, inc_deg, 0.0, separation_deg * index, moment, template
            )
        return cls(vehicles)

    @classmethod
    def walker(
        cls,
        total: int,
        planes: int,
        phasing: int,
        *,
        epoch: str = "2024-01-01T00:00:00Z",
        a_m: float = 6800e3,
        inc_deg: float = 51.6,
        scenario: ScenarioConfig | None = None,
    ) -> "Fleet":
        """
        Walker-delta T/P/F.
        T — число аппаратов, P — число плоскостей, F — параметр фазирования.
        В плоскости j долгота узла 360°·j/P, аномалия i-го аппарата
        360°·i/(T/P) + 360°·F·j/T.
        """
        if planes < 1 or total < 2 or total % planes != 0:
            raise ValueError("T должно делиться на P, аппаратов не меньше двух")
        if not 0 <= phasing < planes:
            raise ValueError("F должен быть в диапазоне 0 … P−1")
        per_plane = total // planes
        template = scenario or ScenarioConfig()
        moment = Time(epoch)
        vehicles: dict[str, SimulationEngine] = {}
        for plane in range(planes):
            raan = 360.0 * plane / planes
            for slot in range(per_plane):
                nu = 360.0 * slot / per_plane + 360.0 * phasing * plane / total
                name = f"P{plane + 1}-S{slot + 1}"
                vehicles[name] = _make_vehicle(name, a_m, inc_deg, raan, nu, moment, template)
        return cls(vehicles)

    @classmethod
    def walker_star(
        cls,
        total: int,
        planes: int,
        phasing: int,
        *,
        epoch: str = "2024-01-01T00:00:00Z",
        a_m: float = 6800e3,
        inc_deg: float = 86.4,
        scenario: ScenarioConfig | None = None,
        spacecraft: SpacecraftConfig | None = None,
    ) -> "Fleet":
        """
        Walker-star T/P/F: узлы разнесены на 180°, а не на 360°.
        Так устроена приполярная сетка Iridium: 86.4°:66/6/2.
        Аномалия та же, что у Walker-delta: 360°·i/(T/P) + 360°·F·j/T.
        """
        if planes < 1 or total < 2 or total % planes != 0:
            raise ValueError("T должно делиться на P, аппаратов не меньше двух")
        if not 0 <= phasing < planes:
            raise ValueError("F должен быть в диапазоне 0 … P−1")
        per_plane = total // planes
        template = scenario or ScenarioConfig()
        moment = Time(epoch)
        vehicles: dict[str, SimulationEngine] = {}
        for plane in range(planes):
            raan = 180.0 * plane / planes
            for slot in range(per_plane):
                nu = 360.0 * slot / per_plane + 360.0 * phasing * plane / total
                name = f"P{plane + 1}-S{slot + 1}"
                vehicles[name] = _make_vehicle(
                    name, a_m, inc_deg, raan, nu, moment, template, spacecraft
                )
        return cls(vehicles)

    def step(
        self,
        dt_sec: float,
        commands: dict[str, str | None] | None = None,
        payloads: dict[str, dict] | None = None,
    ) -> dict[str, tuple]:
        """Команда адресуется по имени. Аппарат без команды просто летит."""
        commands = commands or {}
        payloads = payloads or {}
        unknown = set(commands) - set(self.vehicles)
        if unknown:
            raise KeyError(f"нет аппарата {sorted(unknown)[0]}")
        results = {}
        for name, engine in self.vehicles.items():
            results[name] = engine.step(dt_sec, commands.get(name), payloads.get(name))
        times = {engine.time_sec for engine in self.vehicles.values()}
        if len(times) != 1:
            raise RuntimeError("часы аппаратов разъехались")
        return results

    @property
    def time_sec(self) -> float:
        return next(iter(self.vehicles.values())).time_sec

    def pairs(self) -> list[PairGeometry]:
        """Все неупорядоченные пары. Ось RTN — у первого имени в алфавитном порядке."""
        names = sorted(self.vehicles)
        found: list[PairGeometry] = []
        for i, leader in enumerate(names):
            eng_a = self.vehicles[leader]
            for follower in names[i + 1 :]:
                eng_b = self.vehicles[follower]
                dr = eng_b.orbit.r - eng_a.orbit.r
                dv = eng_b.orbit.v - eng_a.orbit.v
                distance = float(np.linalg.norm(dr))
                rate = float(np.dot(dr, dv) / distance) if distance > 1.0 else 0.0
                rtn = eci_to_rtn(dr, eng_a.orbit.r, eng_a.orbit.v)
                blocked = earth_occults(eng_a.orbit.r, eng_b.orbit.r)
                margin = -99.0 if blocked else isl_margin_db(distance)
                found.append(
                    PairGeometry(
                        leader,
                        follower,
                        distance,
                        rate,
                        float(rtn[0]),
                        float(rtn[1]),
                        float(rtn[2]),
                        blocked,
                        margin,
                    )
                )
        return found

    def use_arc_covariance(self, leader: str, follower: str, covariance_rtn_m2: np.ndarray, *, hard_body_radius_m: float = 15.0) -> None:
        """Ковариация дуги дальностей на текущую эпоху. TCA нет, матрица в скрининг идёт как есть."""
        if leader not in self.vehicles or follower not in self.vehicles:
            raise KeyError("оба имени должны быть в группировке")
        matrix = np.array(covariance_rtn_m2, dtype=float, copy=True)
        if matrix.shape != (6, 6):
            raise ValueError("ковариация дуги должна быть 6×6")
        self._cdm[tuple(sorted((leader, follower)))] = _StoredCdm(
            matrix[:3, :3].copy(),
            0.5 * (matrix + matrix.T),
            hard_body_radius_m,
            None,
        )

    def use_cdm(self, leader: str, follower: str, message: ConjunctionMessage) -> None:
        """Ковариация и радиус твёрдого тела пары берутся из разобранного CDM."""
        if leader not in self.vehicles or follower not in self.vehicles:
            raise KeyError("оба имени должны быть в группировке")
        state = message.combined_state_covariance_si()
        tca = Time(message.tca) if message.tca else None
        self._cdm[tuple(sorted((leader, follower)))] = _StoredCdm(
            np.array(message.combined_covariance_m2(), dtype=float, copy=True),
            None if state is None else np.array(state, dtype=float, copy=True),
            message.hard_body_radius_m,
            tca,
        )

    def pair_covariance(self, leader: str, follower: str, *, sigma_m: float = 200.0) -> np.ndarray:
        """
        6×6 в RTN на текущих часах лидера.
        Полный CDM с TCA и блоком скоростей переносится матрицей Клохесси–Уилтшира.
        Позиционный блок без скоростей остаётся на эпохе скрининга: нулевая скорость
        в сообщении означала бы идеальное знание, а не отсутствие данных.
        """
        stored = self._cdm.get(tuple(sorted((leader, follower))))
        if stored is None:
            return np.diag([sigma_m**2, (8.0 * sigma_m) ** 2, sigma_m**2, 0.0, 0.0, 0.0])
        if stored.state_si is not None and stored.tca is not None:
            primary = self.vehicles[leader]
            dt = (primary.now().jd - stored.tca.jd) * 86400.0
            motion = mean_motion_rad_s(float(np.linalg.norm(primary.orbit.r)))
            phi = cw_stm(motion, dt)
            mapped = phi @ stored.state_si @ phi.T
            return 0.5 * (mapped + mapped.T)
        if stored.state_si is not None:
            # Дуга дальностей уже приведена к текущей эпохе. Перенос от TCA не нужен.
            return 0.5 * (stored.state_si + stored.state_si.T)
        padded = np.zeros((6, 6))
        padded[:3, :3] = stored.position_m2
        return padded

    @property
    def buffer_mb(self) -> dict[str, float]:
        """Мегабиты в регистраторе движка. Отдельного буфера группировки нет."""
        return {name: engine.stored_bits / 1.0e6 for name, engine in self.vehicles.items()}

    @property
    def delivered_mb(self) -> float:
        """Мегабиты, сданные на Землю регистраторами аппаратов."""
        return sum(engine.delivered_bits for engine in self.vehicles.values()) / 1.0e6

    def collect(self, name: str, megabits: float) -> None:
        """Кладёт кадр в регистратор аппарата. Сброс на Землю и ретрансляция забирают его оттуда."""
        if name not in self.vehicles:
            raise KeyError(name)
        if megabits < 0.0:
            raise ValueError("объём кадра не отрицательный")
        if megabits == 0.0:
            return
        self.vehicles[name].accept_bits(megabits * 1.0e6, origin="collect", request_id=name)

    def downlink(self, name: str, dt_sec: float) -> float:
        """Сбрасывает регистратор на станцию. Повтор того же интервала биты не двигает."""
        if name not in self.vehicles:
            raise KeyError(name)
        return self.vehicles[name].transmit_downlink(dt_sec)

    def best_neighbor(self, name: str) -> str | None:
        """Сосед с наибольшим запасом межспутниковой линии. Закрытый Землёй луч не кандидат."""
        chosen: str | None = None
        best_margin = -1.0
        for pair in self.pairs():
            if name not in (pair.leader, pair.follower) or pair.earth_blocked or pair.isl_margin_db < 0.0:
                continue
            other = pair.follower if pair.leader == name else pair.leader
            if pair.isl_margin_db >= best_margin:
                best_margin = pair.isl_margin_db
                chosen = other
        return chosen

    def route(self, source: str, *, down: set[frozenset[str]] | None = None) -> str | None:
        """Следующий хоп: замкнутая линия и самая короткая очередь. down выключает ребро."""
        from space_env.simulation.dtn import next_hop

        queues = {name: engine.stored_bits for name, engine in self.vehicles.items()}
        return next_hop(self.pairs(), source, down=down, queues=queues)

    def relay(
        self,
        source: str,
        destination: str,
        dt_sec: float,
        *,
        at_sec: float | None = None,
        custody: bool = True,
    ) -> float:
        """Перекладывает буфер на соседа, если луч свободен и запас линии неотрицательный."""
        if source == destination or dt_sec <= 0.0:
            return 0.0
        pair = next(
            (item for item in self.pairs() if {item.leader, item.follower} == {source, destination}),
            None,
        )
        if pair is None or pair.earth_blocked or pair.isl_margin_db < 0.0:
            return 0.0
        source_engine = self.vehicles[source]
        now = source_engine.time_sec if at_sec is None else float(at_sec)
        free = dt_sec - _covered_seconds(now, now + dt_sec, self._relay_used[source])
        free = min(free, dt_sec - _covered_seconds(now, now + dt_sec, self._relay_used[destination]))
        if free <= 1e-9:
            return 0.0
        destination_engine = self.vehicles[destination]
        room = destination_engine._room_bits()
        offered = min(source_engine.stored_bits, isl_rate_bps(pair.range_m) * free)
        moved_bits = offered if math.isinf(room) else min(offered, room)
        if moved_bits <= 0.0:
            return 0.0
        from space_env.simulation.dtn import order_for_forward

        source_engine._products = order_for_forward(source_engine._products)
        records = source_engine.release_bits(moved_bits, destination=destination)
        if not records:
            return 0.0
        ack = float(source_engine.spacecraft.config.comms.ack_sec) if custody else 0.0
        for record in records:
            record.pop("ack_due_sec", None)
            record.pop("ack_from", None)
            record.pop("acked", None)
            if ack > 0.0:
                record["ack_due_sec"] = now + dt_sec + ack
                record["ack_from"] = source
                record["acked"] = False
        destination_engine.accept_custody(records)
        self._relay_used[source] = _merge_intervals([*self._relay_used[source], (now, now + dt_sec)])
        self._relay_used[destination] = _merge_intervals([*self._relay_used[destination], (now, now + dt_sec)])
        return sum(float(item["onboard_bits"]) for item in records) / 1.0e6

    def forward(self, source: str, path: list[str], dt_sec: float, *, down: set[frozenset[str]] | None = None) -> dict:
        """Хопы занимают соседние интервалы. Остановившийся хоп оставляет биты у последнего держателя."""
        if dt_sec <= 0.0 or not path or source not in self.vehicles:
            return {"arrived": False, "holder": source, "reason": "нет хопа"}
        holder = source
        base = self.vehicles[source].time_sec
        blocked = down or set()
        for index, hop in enumerate(path):
            if hop not in self.vehicles:
                return {"arrived": False, "holder": holder, "reason": "нет аппарата"}
            if frozenset({holder, hop}) in blocked:
                return {"arrived": False, "holder": holder, "reason": "ребро выключено"}
            moved = self.relay(holder, hop, dt_sec, at_sec=base + index * dt_sec)
            if moved <= 0.0:
                return {"arrived": False, "holder": holder, "reason": "линия не прошла"}
            holder = hop
        return {"arrived": True, "holder": holder, "reason": ""}

    def forward_live(self, source: str, path: list[str], dt_sec: float, *, down: set[frozenset[str]] | None = None) -> dict:
        """Каждый хоп занимает dt, затем орбита всех аппаратов сдвигается на этот dt."""
        if dt_sec <= 0.0 or not path or source not in self.vehicles:
            return {"arrived": False, "holder": source, "reason": "нет хопа", "time_sec": self.time_sec}
        holder = source
        blocked = down or set()
        for hop in path:
            if hop not in self.vehicles:
                return {"arrived": False, "holder": holder, "reason": "нет аппарата", "time_sec": self.time_sec}
            if frozenset({holder, hop}) in blocked:
                return {"arrived": False, "holder": holder, "reason": "ребро выключено", "time_sec": self.time_sec}
            moved = self.relay(holder, hop, dt_sec)
            if moved <= 0.0:
                return {"arrived": False, "holder": holder, "reason": "линия не прошла", "time_sec": self.time_sec}
            before = self.vehicles[holder].orbit.r.copy()
            self.step(dt_sec)
            if float(np.linalg.norm(self.vehicles[holder].orbit.r - before)) <= 1.0:
                return {"arrived": False, "holder": holder, "reason": "орбита не сдвинулась", "time_sec": self.time_sec}
            holder = hop
        return {"arrived": True, "holder": holder, "reason": "", "time_sec": self.time_sec}

    def route_live(self, source: str, dt_sec: float, *, hops: int = 1, down: set[frozenset[str]] | None = None) -> dict:
        """На каждом хопе маршрут считается заново по текущим линиям, затем орбита сдвигается."""
        if dt_sec <= 0.0 or hops < 1 or source not in self.vehicles:
            return {"arrived": False, "holder": source, "path": [], "reason": "нет хопа", "time_sec": self.time_sec}
        holder = source
        taken: list[str] = []
        for _index in range(int(hops)):
            nxt = self.route(holder, down=down)
            if nxt is None:
                return {"arrived": False, "holder": holder, "path": taken, "reason": "нет хопа", "time_sec": self.time_sec}
            moved = self.relay(holder, nxt, dt_sec)
            if moved <= 0.0:
                return {"arrived": False, "holder": holder, "path": taken, "reason": "линия не прошла", "time_sec": self.time_sec}
            self.step(dt_sec)
            taken.append(nxt)
            holder = nxt
        return {"arrived": True, "holder": holder, "path": taken, "reason": "", "time_sec": self.time_sec}

    def acknowledge(self, holder: str, product_id: str) -> bool:
        """Подтверждение опеки. После него просрочка этот продукт не возвращает."""
        for item in self.vehicles[holder]._products:
            if str(item.get("product_id")) != product_id or not item.get("ack_from"):
                continue
            item["acked"] = True
            return True
        return False

    def return_unacked(self, holder: str, dt_sec: float, *, down: set[frozenset[str]] | None = None) -> float:
        """Просроченная опека возвращается предыдущему держателю, только если линия ещё жива."""
        engine = self.vehicles[holder]
        now = engine.time_sec
        pending = [
            item
            for item in engine._products
            if item.get("ack_from")
            and not item.get("acked")
            and not item.get("delivered_to")
            and float(item.get("onboard_bits", 0.0)) > 1e-9
            and now + 1e-9 >= float(item.get("ack_due_sec", 1e30))
        ]
        if not pending:
            return 0.0
        home = str(pending[0]["ack_from"])
        if down and frozenset({holder, home}) in down:
            return 0.0
        return self.relay(holder, home, dt_sec, at_sec=now, custody=False)

    def screen(
        self,
        *,
        hard_body_radius_m: float = 15.0,
        sigma_m: float = 200.0,
        range_limit_m: float = 50_000.0,
    ) -> list[ScreenHit]:
        """
        Pc для пар ближе порога. Без CDM ковариация диагональная: σ по радиусу и нормали,
        вдоль трека в 8 раз больше. Пара с CDM скринится и за порогом дистанции.
        """
        hits: list[ScreenHit] = []
        for pair in self.pairs():
            stored = self._cdm.get(tuple(sorted((pair.leader, pair.follower))))
            if stored is None and pair.range_m > range_limit_m:
                continue
            covariance = self.pair_covariance(pair.leader, pair.follower, sigma_m=sigma_m)
            radius = stored.radius_m if stored is not None else hard_body_radius_m
            leader = self.vehicles[pair.leader]
            follower = self.vehicles[pair.follower]
            dr = eci_to_rtn(follower.orbit.r - leader.orbit.r, leader.orbit.r, leader.orbit.v)
            dv = eci_to_rtn(follower.orbit.v - leader.orbit.v, leader.orbit.r, leader.orbit.v)
            probability = bplane_probability(dr, dv, covariance, radius)
            hits.append(ScreenHit(pair.leader, pair.follower, pair.range_m, probability))
        return hits


def _make_vehicle(
    name: str,
    a_m: float,
    inc_deg: float,
    raan_deg: float,
    nu_deg: float,
    moment: Time,
    template: ScenarioConfig,
    spacecraft: SpacecraftConfig | None = None,
) -> SimulationEngine:
    orbit = Orbit.from_keplerian(
        a_m, 0.0, np.deg2rad(inc_deg), np.deg2rad(raan_deg), 0.0, np.deg2rad(nu_deg), moment
    )
    config = spacecraft.model_copy(deep=True) if spacecraft is not None else SpacecraftConfig()
    config.name = name
    craft = Spacecraft(config=config)
    craft.reset_state()
    return SimulationEngine(orbit, craft, scenario=template.model_copy(deep=True))


def one_watt_star_shell() -> dict[str, float | int]:
    """
    Оболочка 86.4°:66/6/2 на 780 км и линия 1 Вт / 6 дБи.
    Возвращает ближайшую дистанцию и число замкнутых линий. Пороги модели не подгоняются.
    """
    from space_env.core.link import isl_closure_range_m

    radius = 6_378_137.0 + 780_000.0
    fleet = Fleet.walker_star(
        66,
        6,
        2,
        a_m=radius,
        inc_deg=86.4,
        scenario=ScenarioConfig(ground_stations=[]),
    )
    pairs = fleet.pairs()
    nearest = min(pairs, key=lambda item: item.range_m)
    closed = sum(1 for item in pairs if not item.earth_blocked and item.isl_margin_db >= 0.0)
    return {
        "count": 66,
        "nearest_m": float(nearest.range_m),
        "nearest_margin_db": float(nearest.isl_margin_db),
        "closed_links": closed,
        "closure_range_m": isl_closure_range_m(),
    }
