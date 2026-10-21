"""Свежий TLE пишется рядом, снимок пакета остаётся от 20 октября 2026."""
from pathlib import Path

from space_env.catalog.build import read_tle
from space_env.catalog.live import parse_tle_text, save_live_tle

_SNAPSHOT = Path(__file__).resolve().parents[1] / "src" / "space_env" / "catalog" / "tle" / "sentinel-2a.tle"


def test_live_save_does_not_replace_the_october_snapshot(tmp_path):
    pinned_before = _SNAPSHOT.read_text(encoding="utf-8")
    assert "26293.94887774" in pinned_before
    line1, line2 = read_tle("sentinel-2a")

    def opener(url: str) -> str:
        assert "CATNR=40697" in url
        return "SENTINEL-2A\n" + line1.replace("26293.94887774", "26300.00000000") + "\n" + line2 + "\n"

    written = save_live_tle("sentinel-2a", tmp_path, opener)
    assert written.parent == tmp_path
    assert "26300.00000000" in written.read_text(encoding="utf-8")
    assert _SNAPSHOT.read_text(encoding="utf-8") == pinned_before
    parsed = parse_tle_text(written.read_text(encoding="utf-8"))
    assert parsed[0].startswith("1 40697")
