"""Optional Playwright acceptance: real HTTP restart, control transfer and archive import.

Requires playwright and its Chromium browser. Leaves screenshots and private test
connection files in a fresh temporary directory; never use production credentials.
"""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

root = Path(tempfile.mkdtemp(prefix="space-env-b10b-browser-"))
code = """import sys,json
from pathlib import Path
from http.server import ThreadingHTTPServer
from space_env.api import operator
from space_env.api.server import _Handler
operator.STORE=operator.OperatorStore(sys.argv[1])
s=ThreadingHTTPServer(("127.0.0.1",0),_Handler)
Path(sys.argv[2]).write_text(json.dumps({"port":s.server_port}))
s.serve_forever()
"""
(root / "server.py").write_text(code)
proc = None


def start():
    global proc
    marker = root / "port.json"
    marker.unlink(missing_ok=True)
    with (root / "server.log").open("a") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                str(root / "server.py"),
                str(root / "db.sqlite"),
                str(marker),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    for _ in range(300):
        if marker.exists():
            return json.loads(marker.read_text())["port"]
        if proc.poll() is not None:
            raise AssertionError((root / "server.log").read_text())
        time.sleep(0.1)
    raise AssertionError("server startup timeout")


errors = []
try:
    port = start()
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--use-gl=angle",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
            ],
        )
        page = browser.new_page(viewport={"width": 1500, "height": 1100})
        page.on("pageerror", lambda e: errors.append(str(e)))

        def ready(revision):
            page.wait_for_function(
                f"document.body.dataset.revision==='{revision}' && !document.getElementById('new').disabled",
                timeout=90000,
            )

        page.goto(f"http://127.0.0.1:{port}/operator")
        ready(0)
        with page.expect_download() as d:
            page.locator("#connection-save").click()
        connection = root / "connection.json"
        d.value.save_as(connection)
        page.locator("#wait").click()
        ready(1)
        before = page.locator("#time").inner_text()
        page.locator("#to-agent").click()
        page.wait_for_function(
            "document.getElementById('owner').textContent.includes('агент') && !document.getElementById('new').disabled"
        )
        assert page.locator("#wait").is_disabled()
        assert page.locator("#time").inner_text() == before
        page.locator("#role").select_option("agent")
        page.wait_for_function("!document.getElementById('wait').disabled")
        page.locator("#wait").click()
        ready(2)
        page.locator("#to-operator").click()
        page.wait_for_function(
            "document.getElementById('owner').textContent.includes('оператор') && !document.getElementById('new').disabled"
        )
        page.locator("#role").select_option("operator")
        page.wait_for_function("!document.getElementById('wait').disabled")
        before = page.locator("#time").inner_text()
        proc.terminate()
        proc.wait(timeout=10)
        port = start()
        page.goto(f"http://127.0.0.1:{port}/operator")
        page.wait_for_function(
            "document.getElementById('connection').textContent.includes('Подключите')"
        )
        page.locator("#upload").set_input_files(connection)
        ready(2)
        assert page.locator("#time").inner_text() == before
        with page.expect_download() as d:
            page.locator("#archive").click()
        archive = root / "archive.json"
        d.value.save_as(archive)
        page.wait_for_function("!document.getElementById('new').disabled")
        old_id = page.locator("#session-label").inner_text()
        page.locator("#upload").set_input_files(archive)
        page.wait_for_function(
            "(old) => document.getElementById('session-label').textContent!==old && !document.getElementById('new').disabled",
            arg=old_id,
            timeout=90000,
        )
        ready(2)
        assert not page.locator("#notice").evaluate(
            '(el)=>el.classList.contains("error")'
        )
        page.screenshot(path=str(root / "desktop.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(root / "mobile.png"), full_page=True)
        assert page.evaluate(
            "document.documentElement.scrollWidth <= window.innerWidth"
        )
        browser.close()
    assert not errors, errors
    result = {
        "ok": True,
        "restart_reconnect": True,
        "controller_transfer": True,
        "archive_restore": True,
        "mobile_no_overflow": True,
        "page_errors": errors,
    }
    (root / "receipt.json").write_text(json.dumps(result))
    print(json.dumps(result))
finally:
    if proc and proc.poll() is None:
        proc.terminate()
        proc.wait(timeout=10)
