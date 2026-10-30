"""Two-part UTC/TAI/TT, SI-second arithmetic and explicit Earth orientation data."""
from __future__ import annotations

import csv
import hashlib
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_SECONDS = 86400.


def erfa_backend():
    try:
        import erfa
    except ImportError as exc:
        raise ImportError("reference physics requires pyerfa; no legacy fallback is allowed") from exc
    return erfa


@dataclass(frozen=True)
class Epoch:
    """TAI two-part JD; constructed from explicit UTC, never legacy float-JD Time."""
    tai1: float
    tai2: float

    def __post_init__(self):
        if not math.isfinite(self.tai1) or not math.isfinite(self.tai2):
            raise ValueError("epoch parts must be finite")
        year = erfa_backend().jd2cal(*erfa_backend().taiutc(self.tai1, self.tai2))[0]
        if not 1972 <= year <= 2026:
            raise ValueError("reference leap-second applicability is 1972 through 2026")

    @classmethod
    def utc(cls, text: str) -> Epoch:
        erfa = erfa_backend()
        # UTC ISO only, including a literal leap second. Reject timezone-free text.
        if not isinstance(text, str) or not text.endswith("Z"):
            raise ValueError("reference UTC epoch must be ISO text ending in Z")
        date, clock = text[:-1].split("T")
        year, month, day = map(int, date.split("-"))
        hour, minute, second = clock.split(":")
        if not 1972 <= year <= 2026:
            raise ValueError("reference leap-second applicability is 1972 through 2026")
        with warnings.catch_warnings():
            warnings.simplefilter("error", erfa.ErfaWarning)
            a, b = erfa.dtf2d("UTC", year, month, day, int(hour), int(minute), float(second))
            ta, tb = erfa.utctai(a, b)
        return cls(float(ta), float(tb))

    @property
    def utc_jd(self) -> tuple[float, float]:
        a, b = erfa_backend().taiutc(self.tai1, self.tai2)
        return float(a), float(b)

    @property
    def tt_jd(self) -> tuple[float, float]:
        a, b = erfa_backend().taitt(self.tai1, self.tai2)
        return float(a), float(b)

    def shifted(self, seconds: float) -> Epoch:
        if not math.isfinite(seconds):
            raise ValueError("time shift must be finite")
        # Normalize the small part without collapsing a+b into a single float.
        small = self.tai2 + seconds / _SECONDS
        whole = math.floor(small)
        return Epoch(self.tai1 + whole, small - whole)

    def seconds_since(self, other: Epoch) -> float:
        return ((self.tai1 - other.tai1) + (self.tai2 - other.tai2)) * _SECONDS

    def iso(self) -> str:
        year, month, day, clock = erfa_backend().d2dtf("UTC", 9, *self.utc_jd)
        return f"{year:04d}-{month:02d}-{day:02d}T{clock['h']:02d}:{clock['m']:02d}:{clock['s']:02d}.{clock['f']:09d}Z"

    def datetime(self) -> datetime:
        # Only atmospheric models use datetime; do not silently fold a leap second.
        return datetime.fromisoformat(self.iso().replace("Z", "+00:00")).astimezone(timezone.utc)


@dataclass(frozen=True)
class Orientation:
    dut1_s: float
    xp_arcsec: float
    yp_arcsec: float

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.dut1_s, self.xp_arcsec, self.yp_arcsec)):
            raise ValueError("Earth orientation values must be finite")
        if abs(self.dut1_s) > .9 or abs(self.xp_arcsec) > 5 or abs(self.yp_arcsec) > 5:
            raise ValueError("Earth orientation values outside declared terrestrial limits")


class EOPSeries:
    """Finite, strictly ordered daily nodes; no zero/EOP extrapolation fallback.

    DUT1 interpolation uses UT1-TAI so a UTC leap does not smear into Earth rotation.
    Predictions require explicit opt-in and remain marked in metadata.
    """
    def __init__(self, rows: list[dict], *, source: str, allow_predictions: bool = False):
        if len(rows) < 2 or not source:
            raise ValueError("EOP requires at least two nodes and a source")
        self._rows = []
        self._epochs = []
        self.allow_predictions = bool(allow_predictions)
        self.source = source
        values = []
        for row in rows:
            if set(row) != {"utc", "dut1_s", "xp_arcsec", "yp_arcsec", "status"}:
                raise ValueError("unsupported EOP node fields")
            epoch = Epoch.utc(row["utc"])
            orientation = Orientation(row["dut1_s"], row["xp_arcsec"], row["yp_arcsec"])
            if row["status"] not in {"final", "prediction", "synthetic"}:
                raise ValueError("unsupported EOP node status")
            if self._epochs and epoch.seconds_since(self._epochs[-1]) <= 0:
                raise ValueError("EOP nodes must be strictly increasing")
            if self._epochs and epoch.seconds_since(self._epochs[-1]) > 86401.001:
                raise ValueError("EOP interpolation gap exceeds one UTC day")
            self._epochs.append(epoch)
            # TT-UTC from ERFA's calendar-specific leap table, not fixed37s.
            year, month, day, _clock = erfa_backend().d2dtf("UTC", 0, *epoch.utc_jd)
            delta_at = float(erfa_backend().dat(year, month, day, 0.))
            values.append((orientation.dut1_s - delta_at, orientation.xp_arcsec, orientation.yp_arcsec))
            self._rows.append(dict(row))
        self._seconds = np.array([epoch.seconds_since(self._epochs[0]) for epoch in self._epochs])
        self._values = np.array(values)

    @classmethod
    def packaged(cls, *, allow_predictions: bool = False) -> EOPSeries:
        path = Path(__file__).resolve().parents[1] / "core/eop/finals2000A.daily.csv"
        with path.open(encoding="utf-8") as stream:
            reader = csv.reader(stream, delimiter=";")
            header = next(reader)
            dut = header.index("UT1-UTC")
            nodes = [{"utc": f"{r[1]}-{r[2]}-{r[3]}T00:00:00Z",
                      "dut1_s": float(r[dut]), "xp_arcsec": float(r[header.index("x_pole")]),
                      "yp_arcsec": float(r[header.index("y_pole")]),
                      "status": "prediction" if "prediction" in (r[4], r[dut - 1]) else r[dut - 1]}
                     for r in reader if int(r[1]) <= 2026]
        return cls(nodes, source="IERS packaged finals2000A; sha256=" + hashlib.sha256(path.read_bytes()).hexdigest(),
                   allow_predictions=allow_predictions)

    def at(self, epoch: Epoch) -> Orientation:
        seconds = epoch.seconds_since(self._epochs[0])
        if not self._seconds[0] <= seconds <= self._seconds[-1]:
            raise ValueError("epoch outside explicit EOP coverage")
        i = min(int(np.searchsorted(self._seconds, seconds, side="right")) - 1, len(self._rows) - 2)
        i = max(0, i)
        if not self.allow_predictions and any(self._rows[k]["status"] == "prediction" for k in (i, i + 1)):
            raise ValueError("predicted EOP requires allow_predictions=True")
        fraction = (seconds - self._seconds[i]) / (self._seconds[i + 1] - self._seconds[i])
        ut1_tai, xp, yp = self._values[i] * (1 - fraction) + self._values[i + 1] * fraction
        year, month, day, clock = erfa_backend().d2dtf("UTC", 6, *epoch.utc_jd)
        utc_fraction = min((clock["h"] * 3600 + clock["m"] * 60 + clock["s"]) / _SECONDS, .999999999)
        delta_at = erfa_backend().dat(year, month, day, utc_fraction)
        return Orientation(float(ut1_tai + delta_at), float(xp), float(yp))

    def manifest(self) -> dict:
        return {"source": self.source, "allow_predictions": self.allow_predictions,
                "interpolation": "linear UT1-TAI/xp/yp over TAI seconds",
                "rows": [dict(row) for row in self._rows]}
