"""Memory gates must not bill an exec child for its parent's resident peak."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from space_env.planning.runner import process_peak_rss_bytes


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux proc contract")
def test_linux_uses_executable_watermark_not_inherited_getrusage(monkeypatch):
    import resource

    monkeypatch.setattr(
        resource, "getrusage", lambda *a: SimpleNamespace(ru_maxrss=9_000_000)
    )
    monkeypatch.setattr(
        Path, "read_text", lambda *a, **k: "Name:\tpython\nVmHWM:\t123456 kB\n"
    )
    assert process_peak_rss_bytes() == 123456 * 1024


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux proc contract")
@pytest.mark.parametrize(
    "status", ["", "VmHWM: unknown kB", "VmHWM: -1 kB", "VmHWM: 1 MB"]
)
def test_missing_or_invalid_proc_watermark_is_unknown(monkeypatch, status):
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: status)
    assert process_peak_rss_bytes() is None


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux proc contract")
def test_actual_fork_exec_does_not_inherit_parent_memory_charge():
    # Outer driver deliberately touches 100 MB after imports, then execs a new
    # interpreter with the same library. No test-process global allocator state.
    driver = """
import json, subprocess, sys
from space_env.planning.runner import process_peak_rss_bytes
payload = bytearray(100_000_000)
parent = process_peak_rss_bytes()
probe = "import json,resource; from space_env.planning.runner import process_peak_rss_bytes; print(json.dumps({'peak':process_peak_rss_bytes(),'legacy':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}))"
child = subprocess.run([sys.executable, '-c', probe], check=True, capture_output=True, text=True)
print(json.dumps({'parent':parent, 'child':json.loads(child.stdout)}))
"""
    result = subprocess.run(
        [sys.executable, "-c", driver],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    row = json.loads(result.stdout)
    assert row["parent"] > row["child"]["peak"] + 50_000_000
    assert row["child"]["legacy"] > row["child"]["peak"] + 50_000_000
