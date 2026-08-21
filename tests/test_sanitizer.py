import json
import os
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


CLEAN = "========= ERROR SUMMARY: 0 errors"
DETECTED = "========= ERROR: Race reported\n========= ERROR SUMMARY: 1 errors"
# Realistic Compute Sanitizer racecheck output (mixed case banner, tool-specific
# summary label).
RACECHECK_OUTPUT = (
    "========= Error: Race reported\n"
    "    at some_location\n"
    "========= RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)"
)


@pytest.fixture
def exe(tmp_path):
    path = tmp_path / "sanitizer_proof"
    path.write_text("")
    return path


def test_check_detected_when_expected_and_sanitizer_reports(exe, monkeypatch, capsys):
    monkeypatch.setattr(cs.shutil, "which", lambda tool: "/usr/bin/compute-sanitizer")
    monkeypatch.setattr(cs.subprocess, "run",
                        lambda *a, **k: _FakeProcess(1, stdout=DETECTED))
    rc = cs.main([str(exe), "--mode", "broken_race", "--tool", "racecheck",
                  "--expect-detected"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "match"


def test_check_mismatch_when_expected_detected_but_clean(exe, monkeypatch, capsys):
    monkeypatch.setattr(cs.shutil, "which", lambda tool: "/usr/bin/compute-sanitizer")
    monkeypatch.setattr(cs.subprocess, "run",
                        lambda *a, **k: _FakeProcess(0, stdout=CLEAN))
    rc = cs.main([str(exe), "--mode", "broken_race", "--tool", "racecheck",
                  "--expect-detected"])
    assert rc == 1
    assert json.loads(capsys.readouterr().out)["verdict"] == "mismatch"


def test_check_correct_clean_passes(exe, monkeypatch, capsys):
    monkeypatch.setattr(cs.shutil, "which", lambda tool: "/usr/bin/compute-sanitizer")
    monkeypatch.setattr(cs.subprocess, "run",
                        lambda *a, **k: _FakeProcess(0, stdout=CLEAN))
    rc = cs.main([str(exe), "--mode", "correct", "--tool", "memcheck"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "match"


def test_check_racecheck_format_detected(exe, monkeypatch, capsys):
    # Realistic racecheck output uses a mixed-case banner and RACECHECK SUMMARY.
    monkeypatch.setattr(cs.shutil, "which", lambda tool: "/usr/bin/compute-sanitizer")
    monkeypatch.setattr(cs.subprocess, "run",
                        lambda *a, **k: _FakeProcess(1, stdout=RACECHECK_OUTPUT))
    rc = cs.main([str(exe), "--mode", "broken_race", "--tool", "racecheck",
                  "--expect-detected"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "match"


def test_check_crash_without_error_text_is_not_a_detection(exe, monkeypatch, capsys):
    # A nonzero exit with no ERROR text must not masquerade as a detection.
    monkeypatch.setattr(cs.shutil, "which", lambda tool: "/usr/bin/compute-sanitizer")
    monkeypatch.setattr(cs.subprocess, "run",
                        lambda *a, **k: _FakeProcess(7, stderr="segfault"))
    rc = cs.main([str(exe), "--mode", "broken_race", "--tool", "racecheck",
                  "--expect-detected"])
    assert rc == 1
    assert json.loads(capsys.readouterr().out)["detected"] is False


def test_check_crash_on_correct_baseline_is_not_a_match(exe, monkeypatch, capsys):
    # A sanitizer crash on the clean baseline must not green the run.
    monkeypatch.setattr(cs.shutil, "which", lambda tool: "/usr/bin/compute-sanitizer")
    monkeypatch.setattr(cs.subprocess, "run",
                        lambda *a, **k: _FakeProcess(7, stderr="segfault"))
    rc = cs.main([str(exe), "--mode", "correct", "--tool", "memcheck"])
    assert rc == 1
    assert json.loads(capsys.readouterr().out)["verdict"] == "mismatch"


def test_check_missing_executable(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cs.shutil, "which", lambda tool: "/usr/bin/compute-sanitizer")
    rc = cs.main([str(tmp_path / "does_not_exist"), "--mode", "correct",
                  "--tool", "memcheck"])
    assert rc == 2


def test_check_missing_sanitizer(exe, monkeypatch, capsys):
    monkeypatch.setattr(cs.shutil, "which", lambda tool: None)
    rc = cs.main([str(exe), "--mode", "correct", "--tool", "memcheck"])
    assert rc == 2
