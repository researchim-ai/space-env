"""IAU 2000A, центрированный F10.7, линия оболочки 66 и срок экспоненциальной атмосферы."""
import csv
from datetime import datetime, timezone
from pathlib import Path

import pytest

from space_env.core.frames import gast_iau2000a_rad, gast_rad
from space_env.core.lifetime import circular_lifetime_years
from space_env.core.space_weather import centered_f107a, indices_for
from space_env.core.time import Time
from space_env.simulation.fleet import one_watt_star_shell

_FLUX = Path(__file__).resolve().parents[1] / "src" / "space_env" / "core" / "eop" / "f107_trailing.csv"


def test_iau2000a_matches_erfa_and_stays_out_of_the_training_frame():
    import erfa

    from space_env.core.dut1 import dut1_seconds

    moment = Time("2026-10-20T12:00:00Z")
    got = gast_iau2000a_rad(moment)
    ut1 = moment.jd + dut1_seconds(moment) / 86400.0
    tt = moment.jd + (37.0 + 32.184) / 86400.0
    reference = float(erfa.gst00a(ut1, 0.0, tt, 0.0))
    assert got == pytest.approx(reference, abs=1e-12)
    assert gast_rad(moment) == pytest.approx(got, abs=1e-12)
    from space_env.core.frames import gast_iau1980_rad

    delta = abs(got - gast_iau1980_rad(moment))
    assert 1e-8 < delta < 5e-5


def test_centered_flux_exists_only_inside_the_snapshot():
    edge = centered_f107a(Time("2026-10-20T00:00:00Z"))
    assert edge is None
    assert indices_for(Time("2026-10-20T12:00:00Z")).f107_centered is None
    moment = Time("2026-08-20T00:00:00Z")
    found = centered_f107a(moment)
    assert found is not None
    rows = []
    with _FLUX.open(encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            stamp = datetime.fromisoformat(record["date"]).replace(tzinfo=timezone.utc)
            rows.append((stamp.date().isoformat(), float(record["f107_sfu"])))
    place = next(index for index, item in enumerate(rows) if item[0] == "2026-08-20")
    window = [item[1] for item in rows[place - 40 : place + 41]]
    assert len(window) == 81
    assert found[0] == pytest.approx(sum(window) / 81.0, abs=1e-9)
    assert found[1] == 81
    october = indices_for(Time("2026-10-20T12:00:00Z"))
    assert october is not None
    assert october.f107_centered is None
    assert october.f107a == pytest.approx(113.5669, abs=0.01)


def test_the_66_shell_one_watt_link_stays_open():
    report = one_watt_star_shell()
    assert report["count"] == 66
    assert report["closed_links"] == 0
    assert report["nearest_m"] > report["closure_range_m"]
    assert report["nearest_margin_db"] < 0.0


def test_lifetime_estimate_separates_a_low_orbit_from_a_high_one():
    low = circular_lifetime_years(200_000.0, mass_kg=10.0, area_m2=1.0, drag_coefficient=2.2)
    high = circular_lifetime_years(800_000.0, mass_kg=1000.0, area_m2=0.01, drag_coefficient=2.2)
    assert low["within_25_years"] is True
    assert low["years"] < 1.0
    assert high["within_25_years"] is False
    assert high["years"] > 25.0
    assert low["model"] == "exponential_scale_height"
