# space-env project website

Static Russian-language project site. The checked-in HTML is ready to serve;
Node, a package manager and runtime CDNs are not required.

## Preview

From the repository root:

```bash
python -m http.server 8080 --directory web
```

Open `http://127.0.0.1:8080/`. The site is independent of the Python Mission Control
server; the orbital scene is an explicitly illustrative browser animation.

## GitHub Pages subsite

Copy the **contents** of `web/` to a project directory of the existing Pages site,
for example `space-env/`, resulting in:

```
https://researchim-ai.github.io/space-env/
```

This is a suggested path, not an already published URL. All internal links, scripts,
images and styles use relative paths. No root-base assumption, SPA rewrite,
external font server or CDN is needed. Existing GitHub Pages files/workflows were
not changed. Nothing has been published by this change.

The production directory needs `*.html`, `assets/`, `reference/` and `examples/`; `content/`,
`tools/` and this README can be omitted. Keep the asset attribution and license.

## Edit and rebuild

Edit the source fragments in `content/*.html`, shared markup and page metadata in
`tools/build.py`, CSS/JS in `assets/`. From the repository root run:

```bash
python web/tools/build.py
python web/tools/check.py
```

The builder refreshes published contract copies from repository docs and the
acceptance JSON. The presentation is a versioned snapshot, **v0.134.0 / 2026-11-02**;
update numerical claims, API examples and metadata deliberately for future releases.
Check the browser console and desktop/mobile layouts after visual changes.

## Scope and assets

The site distinguishes implemented engineering profiles, internal regression and
external physical qualification. It claims neither a flight-qualified universal
digital twin nor independent superiority over other simulators. Formulae explain
model structure; precise conventions and applicability are in the linked contracts.

`assets/earth.jpg` is reused from the repository's existing three.js example
texture. Its recorded upstream attribution is preserved in `assets/EARTH_SOURCE.txt`
(NASA Visible Earth / three.js r160 examples). `mark.svg`, layout, orbit drawing and
all page interactions are code-native. The project license is included in
`reference/LICENSE.txt` (Apache 2.0).

## Verification performed

- Four pages checked in Chromium at 1440, 768, 390 and 320 px under `/web/`.
- No document overflow, JavaScript page errors or failed requests.
- Keyboard scenario tabs, mobile menu, pause/reduced-motion and clipboard checked.
- All HTML local links/assets and all bundled Markdown file links resolve.
- Published Python TaskSession/BaselinePolicy snippet executed one accepted native
  step against the actual relay example; this does not rerun all physics validation.
- CSS/JS, fonts, textures and native MathML need no external runtime services.
- Public deployment and upstream website availability were not verified.

The graphite/lime visual identity uses locally bundled Lato subsets with Cyrillic
support; the font license is in `assets/fonts/LICENSE.txt`. The original project
mark and name are retained. `assets/mission-control.png` is a screenshot of the
actual local console with a synthetic fleet-demo, not a live browser backend.
