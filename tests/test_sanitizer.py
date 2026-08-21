import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import check_sanitizer as cs  # noqa: E402


class _FakeProcess:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture
def exe(tmp_path):
    path = tmp_path / "sanitizer_proof"
    path.write_text("")
    return path


def test_check_detected_when_expected_and_sanitizer_reports(exe, monkeypatch, capsys):
    monkeypatch.setattr(cs.shutil, "which", lambda tool: "/usr/bin/compute-sanitizer")
    monkeypatch.setattr(cs.subprocess, "run", lambda *a, **k: _FakeProcess(1))
    rc = cs.main([str(exe), "--mode", "broken_race", "--tool", "racecheck",
                  "--expect-detected"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "match"


def test_check_mismatch_when_expected_detected_but_clean(exe, monkeypatch, capsys):
    monkeypatch.setattr(cs.shutil, "which", lambda tool: "/usr/bin/compute-sanitizer")
    monkeypatch.setattr(cs.subprocess, "run", lambda *a, **k: _FakeProcess(0))
    rc = cs.main([str(exe), "--mode", "broken_race", "--tool", "racecheck",
                  "--expect-detected"])
    assert rc == 1
    assert json.loads(capsys.readouterr().out)["verdict"] == "mismatch"


def test_check_correct_clean_passes(exe, monkeypatch, capsys):
    monkeypatch.setattr(cs.shutil, "which", lambda tool: "/usr/bin/compute-sanitizer")
    monkeypatch.setattr(cs.subprocess, "run", lambda *a, **k: _FakeProcess(0))
    rc = cs.main([str(exe), "--mode", "correct", "--tool", "memcheck"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "match"


def test_check_missing_sanitizer(exe, monkeypatch, capsys):
    monkeypatch.setattr(cs.shutil, "which", lambda tool: None)
    rc = cs.main([str(exe), "--mode", "correct", "--tool", "memcheck"])
    assert rc == 2
