"""
Свежие TLE с Celestrak пишутся в отдельный каталог.
Снимок в пакете, на котором сидят тесты, эта функция не трогает.
"""
from __future__ import annotations

from pathlib import Path

from space_env.catalog.references import reference


def celestrak_url(key: str) -> str:
    spec = reference(key)
    if spec.norad_id is None:
        raise KeyError(f"у {key} нет номера NORAD")
    return f"https://celestrak.org/NORAD/elements/gp.php?CATNR={spec.norad_id}&FORMAT=TLE"


def parse_tle_text(text: str) -> tuple[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    line1 = next(line for line in lines if line.startswith("1 "))
    line2 = next(line for line in lines if line.startswith("2 "))
    return line1, line2


def fetch_tle(key: str, opener=None) -> tuple[str, str]:
    """Качает элементы. opener(url) -> str нужен, чтобы тест не ходил в сеть."""
    url = celestrak_url(key)
    if opener is None:
        import urllib.request

        with urllib.request.urlopen(url, timeout=30) as response:
            text = response.read().decode("utf-8")
    else:
        text = opener(url)
    return parse_tle_text(text)


def save_live_tle(key: str, directory: str | Path, opener=None) -> Path:
    """Пишет `{key}.tle` в directory. Каталог пакета не является этим directory."""
    line1, line2 = fetch_tle(key, opener)
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{key}.tle"
    path.write_text(f"{line1}\n{line2}\n", encoding="utf-8")
    return path
