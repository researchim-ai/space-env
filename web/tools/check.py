"""Check the portable site's local references, fragment IDs and editorial boundaries."""

import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__(convert_charrefs=True)
        self.path, self.ids, self.links, self.h1 = path, set(), [], 0
        self.feed(path.read_text())

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            assert attrs["id"] not in self.ids, (
                f"Duplicate ID: {self.path}: {attrs['id']}"
            )
            self.ids.add(attrs["id"])
        self.h1 += tag == "h1"
        for key in ["href", "src"]:
            if key in attrs:
                self.links.append(attrs[key])


pages = {p.resolve(): Page(p) for p in ROOT.glob("*.html")}
count = 0
for path, page in pages.items():
    assert page.h1 == 1, f"Expected one H1: {path}"
    for link in page.links:
        url = urlsplit(link)
        if url.scheme or url.netloc:
            continue
        assert not url.path.startswith("/"), f"Root-based URL: {path}: {link}"
        target = (path.parent / unquote(url.path)).resolve() if url.path else path
        assert target.is_relative_to(ROOT), f"Link leaves site: {path}: {link}"
        assert target.is_file(), f"Missing file: {path}: {link}"
        if url.fragment and target in pages:
            assert unquote(url.fragment) in pages[target].ids, (
                f"Missing fragment: {path}: {link}"
            )
        count += 1
    text = path.read_text().lower()
    assert "стартап" not in text
    assert 'lang="ru"' in text
    assert 'name="description"' in text
json.loads((ROOT / "reference/B10_DURABILITY.json").read_text())
print(
    f"PASS: {len(pages)} pages, {count} local links/assets, unique IDs and valid fragments."
)

for path in list((ROOT / "reference").rglob("*.md")) + list(
    (ROOT / "examples").rglob("*.md")
):
    for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
        url = urlsplit(link)
        if url.scheme or url.netloc or not url.path:
            continue
        target = (path.parent / unquote(url.path)).resolve()
        assert target.is_relative_to(ROOT) and target.is_file(), (
            f"Missing Markdown reference: {path}: {link}"
        )
print("PASS: bundled Markdown references remain inside the site and resolve.")
