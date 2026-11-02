"""Build portable, dependency-free project pages from checked-in content fragments."""

import re
import shutil
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PAGES = {
    "index": (
        "Моделер космических миссий для AI",
        "Космические миссии как среда для обучения AI-агентов. Физика, планирование, группировки, reinforcement learning и воспроизводимые данные.",
    ),
    "models": (
        "Физические модели и математика",
        "Орбитальная динамика, ориентация, тепловые сети, энергетика, RF и передача данных: уравнения, реализация и границы применимости space-env.",
    ),
    "learning": (
        "Агенты, RL и данные",
        "TaskSession, Gymnasium, PettingZoo, причинные наблюдения, планирование и проверяемые RL/SFT-датасеты на общем runtime.",
    ),
    "guide": (
        "Начать работу",
        "Установка space-env, запуск синтетической миссии, обучение, native replay, Mission Control и восстановление сессий.",
    ),
}
DOCS = [
    "AGENT_RUNTIME_CONTRACT.md",
    "PLANNING_CONTRACT.md",
    "FACTORY_CONTRACT.md",
    "SCALING_CONTRACT.md",
    "VERIFIED_EXPORT_CONTRACT.md",
    "OPERATOR_DURABILITY_CONTRACT.md",
]
VALIDATION = [
    "RIGID_ATTITUDE_CONTRACT.md",
    "RIGID_MISSION_CONTRACT.md",
    "THERMAL_NETWORK_CONTRACT.md",
    "RADIO_GROUND_CONTRACT.md",
    "CUSTODY_NETWORK_CONTRACT.md",
    "NAVIGATION_COVARIANCE_CONTRACT.md",
    "MOMENTUM_UNLOADING_CONTRACT.md",
    "ROTATING_THRUST_CONTRACT.md",
]
for file in DOCS:
    shutil.copyfile(REPO / "docs" / file, ROOT / "reference" / file)
for file in VALIDATION:
    shutil.copyfile(REPO / "docs/validation" / file, ROOT / "reference" / file)
shutil.copyfile(
    REPO / "docs/acceptance/B10_DURABILITY.json", ROOT / "reference/B10_DURABILITY.json"
)
# Preserve linked evidence and relative references inside the bundled contracts.
queue = [(REPO / "docs" / f, ROOT / "reference" / f) for f in DOCS]
queue += [(REPO / "docs/validation" / f, ROOT / "reference" / f) for f in VALIDATION]
seen = set()
while queue:
    source, destination = queue.pop()
    if (source, destination) in seen or source.suffix != ".md":
        continue
    seen.add((source, destination))
    for link in re.findall(r"\]\(([^)]+)\)", source.read_text()):
        url = urlsplit(link)
        if url.scheme or url.netloc or not url.path:
            continue
        origin = (source.parent / unquote(url.path)).resolve()
        target = (destination.parent / unquote(url.path)).resolve()
        if (
            origin.is_file()
            and origin.is_relative_to(REPO)
            and target.is_relative_to(ROOT)
        ):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, target)
            queue.append((origin, target))

for slug, (title, description) in PAGES.items():
    nav = "".join(
        f'<a href="{key}.html" {"aria-current=page" if key == slug else ""}>{label}</a>'
        for key, label in [
            ("index", "Обзор"),
            ("models", "Модели"),
            ("learning", "Агенты и данные"),
            ("guide", "Руководство"),
        ]
    )
    content = (ROOT / "content" / f"{slug}.html").read_text()
    html = f'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="theme-color" content="#0b1014"><title>{title} · space-env</title><meta name="description" content="{description}"><meta property="og:locale" content="ru_RU"><meta property="og:type" content="website"><meta property="og:title" content="{title} · space-env"><meta property="og:description" content="{description}"><link rel="icon" type="image/svg+xml" href="assets/mark.svg"><link rel="stylesheet" href="assets/site.css"><link rel="stylesheet" href="assets/mission.css"><script type="module" src="assets/site.js"></script></head>
<body data-page="{slug}"><noscript><style>@media(max-width:800px){{.site-header{{height:auto;flex-wrap:wrap;padding-block:20px}}.site-header nav{{display:flex;position:static;width:100%;flex-direction:column;box-shadow:none;padding:0;order:5}}.menu-toggle{{display:none}}}}</style></noscript><a class="skip" href="#main">К содержанию</a><header class="site-header"><a class="logo" href="index.html" aria-label="space-env — главная"><img src="assets/mark.svg" alt="" width="34" height="34">space<span>—</span>env</a><nav aria-label="Основная навигация" id="navigation">{nav}</nav><a class="parent-link" href="https://researchim-ai.github.io/">Researchim AI <span aria-hidden="true">↗</span></a><button class="menu-toggle" aria-expanded="false" aria-controls="navigation" aria-label="Открыть меню">☰</button></header>
<main id="main" tabindex="-1">{content}</main>
<footer class="site-footer"><div><a class="logo" href="index.html">space<span>—</span>env</a><p>Космические миссии.<br>Обучаемые решения.</p></div><div><span class="eyebrow">ПРОЕКТ</span><a href="models.html">Физические модели</a><a href="learning.html">Агенты и данные</a><a href="guide.html#contracts">Контракты реализации</a></div><div><span class="eyebrow">НАЧАТЬ</span><a href="guide.html#install">Установка</a><a href="guide.html#mission-control">Mission Control</a><a href="https://researchim-ai.github.io/">Researchim AI ↗</a></div><div class="footer-note"><span class="version">v0.134.0 · Apache 2.0</span><p>Открытый инженерный моделер. Точность и доступные механики зависят от профиля сценария. Не система управления реальными полётами.</p><a href="reference/LICENSE.txt">Лицензия</a></div><div class="footer-bottom"><span>SPACE MISSION ENVIRONMENTS FOR AI</span><span>Срез реализации · 02 ноября 2026</span><a href="#main">Наверх ↑</a></div></footer><div class="toast" role="status" aria-live="polite"></div></body></html>'''
    (ROOT / f"{slug}.html").write_text(html)
print("Built 4 portable pages and refreshed their linked reference documents.")
