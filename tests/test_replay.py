import json
import math
import sys
from pathlib import Path

import pytest

# Make the tools package importable without installation.
TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import replay_failure as rf  # noqa: E402

ARTIFACT = {
    "schema": "vanta.failure.v0",
    "kernel": "reduce_sum",
    "implementation": "faulty_tail",
    "input": {
        "dtype": "float32",
        "shape": [31],
        "values": [1.0] * 31,
    },
    "launch": {"block_threads": 32, "grid_blocks": 1},
    "environment": {"gpu": "Tesla T4", "compute_capability": "7.5"},
    "observed": {
        "kind": "correctness_mismatch",
        "pattern": 0,
        "expected": 31.0,
        "actual": 30.0,
    },
}

VECTOR_ADD_ARTIFACT = {
    "schema": "vanta.failure.v0",
    "kernel": "vector_add",
    "implementation": "faulty_tail",
    "input": {
        "dtype": "float32",
        "shape": [31],
        "values": [1.0] * 31,
    },
    "params": {"a": 2.0},
    "aux": {"y": [1.0] * 31},
    "launch": {"block_threads": 32, "grid_blocks": 1},
    "environment": {"gpu": "Tesla T4", "compute_capability": "7.5"},
    "observed": {
        "kind": "correctness_mismatch",
        "index": 30,
        "pattern": 0,
        "expected": 3.0,
        "actual": 0.0,
    },
}


def _write(tmp_path, data):
    p = tmp_path / "artifact.json"
    p.write_text(json.dumps(data))
    return p


def test_load_artifact_ok(tmp_path):
    assert rf.load_artifact(_write(tmp_path, ARTIFACT))["kernel"] == "reduce_sum"


def test_load_vector_add_artifact_ok(tmp_path):
    assert (rf.load_artifact(_write(tmp_path, VECTOR_ADD_ARTIFACT))["kernel"]
            == "vector_add")


def test_load_vector_add_requires_aux_length(tmp_path):
    bad = dict(VECTOR_ADD_ARTIFACT)
    bad["aux"] = {"y": [1.0]}
    with pytest.raises(ValueError):
        rf.load_artifact(_write(tmp_path, bad))


def test_load_artifact_missing_field(tmp_path):
    bad = dict(ARTIFACT)
    del bad["observed"]
    with pytest.raises(ValueError):
        rf.load_artifact(_write(tmp_path, bad))


def test_load_artifact_shape_mismatch(tmp_path):
    bad = dict(ARTIFACT)
    bad["input"] = dict(ARTIFACT["input"])
    bad["input"]["shape"] = [5]
    with pytest.raises(ValueError):
        rf.load_artifact(_write(tmp_path, bad))


def test_load_artifact_empty_values(tmp_path):
    bad = dict(ARTIFACT)
    bad["input"] = dict(ARTIFACT["input"])
    bad["input"]["values"] = []
    with pytest.raises(ValueError):
        rf.load_artifact(_write(tmp_path, bad))


def test_load_artifact_top_level_not_object(tmp_path):
    p = tmp_path / "artifact.json"
    p.write_text("null")
    with pytest.raises(ValueError):
        rf.load_artifact(p)


def test_compute_mismatch_true():
    assert rf.compute_mismatch(31.0, 30.0) is True


def test_compute_mismatch_false_exact():
    assert rf.compute_mismatch(31.0, 31.0) is False


def test_compute_mismatch_false_within_tolerance():
    # 1e-5 * max(1, 31) = 3.1e-4; delta of 1e-5 is well inside.
    assert rf.compute_mismatch(31.0, 31.0 + 1e-5) is False


def test_classify_same_failure():
    result = {"expected": 31.0, "actual": 30.0}
    assert rf.classify(ARTIFACT, result) == "same_failure"


def test_classify_resolved():
    result = {"expected": 31.0, "actual": 31.0}
    assert rf.classify(ARTIFACT, result) == "resolved"


def test_classify_different_failure():
    result = {"expected": 31.0, "actual": 29.0}
    assert rf.classify(ARTIFACT, result) == "different_failure"


def test_classify_expected_mismatch():
    result = {"expected": 32.0, "actual": 30.0}
    assert rf.classify(ARTIFACT, result) == "different_failure"


def test_build_payload_shape():
    payload = rf.build_payload(ARTIFACT)
    lines = payload.strip().split("\n")
    assert lines[0] == "faulty_tail"
    assert lines[1] == "31"
    assert lines[2] == "32"
    assert len(lines[3].split(" ")) == 31


def test_build_vector_add_payload_shape():
    payload = rf.build_payload(VECTOR_ADD_ARTIFACT)
    lines = payload.strip().split("\n")
    assert lines[0] == "faulty_tail"
    assert lines[1] == "31"
    assert lines[2] == "32"
    assert lines[3] == "2.0"
    assert len(lines[4].split(" ")) == 31
    assert len(lines[5].split(" ")) == 31


def test_classify_vector_add_same_failure():
    result = {"expected": 3.0, "actual": 0.0}
    assert rf.classify(VECTOR_ADD_ARTIFACT, result) == "same_failure"


class _FakeProcess:
    def __init__(self, returncode, stdout, stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_main_same_failure_rc0(tmp_path, monkeypatch, capsys):
    art = _write(tmp_path, ARTIFACT)
    exe = tmp_path / "fake"
    exe.write_text("")

    def fake_run(cmd, **kwargs):
        return _FakeProcess(0, '{"expected": 31.0, "actual": 30.0}')

    monkeypatch.setattr(rf.subprocess, "run", fake_run)
    rc = rf.main([str(art), "--executable", str(exe), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["classification"] == "same_failure"
    assert out["mismatch"] is True


def test_main_vector_add_same_failure_rc0(tmp_path, monkeypatch, capsys):
    art = _write(tmp_path, VECTOR_ADD_ARTIFACT)
    exe = tmp_path / "fake"
    exe.write_text("")

    def fake_run(cmd, **kwargs):
        return _FakeProcess(0, '{"expected": 3.0, "actual": 0.0}')

    monkeypatch.setattr(rf.subprocess, "run", fake_run)
    rc = rf.main([str(art), "--executable", str(exe), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["classification"] == "same_failure"


def test_main_infrastructure_failure_on_bad_json(tmp_path, monkeypatch, capsys):
    art = _write(tmp_path, ARTIFACT)
    exe = tmp_path / "fake"
    exe.write_text("")

    def fake_run(cmd, **kwargs):
        return _FakeProcess(0, "this is not json")

    monkeypatch.setattr(rf.subprocess, "run", fake_run)
    rc = rf.main([str(art), "--executable", str(exe), "--json"])
    assert rc == 1
    out = json.loads(capsys.readouterr().out)
    assert out["classification"] == "infrastructure_failure"


def test_main_infrastructure_failure_on_nonzero(tmp_path, monkeypatch, capsys):
    art = _write(tmp_path, ARTIFACT)
    exe = tmp_path / "fake"
    exe.write_text("")

    def fake_run(cmd, **kwargs):
        return _FakeProcess(1, "", "cuda crashed")

    monkeypatch.setattr(rf.subprocess, "run", fake_run)
    rc = rf.main([str(art), "--executable", str(exe), "--json"])
    assert rc == 1
    out = json.loads(capsys.readouterr().out)
    assert out["classification"] == "infrastructure_failure"


import minimize_failure as mf  # noqa: E402

import colab_entry as ce  # noqa: E402


def test_colab_entry_compile_fail_writes_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(ce.shutil, "which", lambda tool: "/usr/bin/" + tool)
    monkeypatch.setattr(ce, "compile_source", lambda *a, **k: False)
    monkeypatch.setattr(ce, "probe_environment", lambda: {"gpu": "x"})

    out = tmp_path / "out"
    rc = ce.main([
        "--source", "x.cu",
        "--artifact", "f.json",
        "--out", str(out),
    ])
    assert rc == 1
    summary = json.loads((out / "campaign_summary.json").read_text())
    assert summary["status"] == "fail"
    assert summary["compile_ok"] is False
    assert summary["correct_campaign_ok"] is False


# --- minimize_failure ------------------------------------------------------

def test_find_minimal_prefix_returns_shortest_failing_prefix():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    def still_fails(candidate):
        # failure only present once we keep at least the first three elements
        return len(candidate) >= 3

    best, tested = mf.find_minimal_prefix(values, still_fails)
    assert best == [1.0, 2.0, 3.0]
    assert tested == len(values)


def test_find_minimal_prefix_seed_must_fail():
    values = [1.0, 2.0, 3.0]

    def never_fails(_candidate):
        return False

    with pytest.raises(ValueError):
        mf.find_minimal_prefix(values, never_fails)


def test_candidate_artifact_updates_shape_and_observed():
    result = {"expected": 12.0, "actual": 9.0}
    out = mf.candidate_artifact(ARTIFACT, [2.0, 3.0, 7.0], result)
    assert out["input"]["values"] == [2.0, 3.0, 7.0]
    assert out["input"]["shape"] == [3]
    assert out["observed"]["expected"] == 12.0
    assert out["observed"]["actual"] == 9.0
    # original artifact is not mutated
    assert ARTIFACT["input"]["values"] == [1.0] * 31


def test_replay_candidate_parses_json(tmp_path, monkeypatch):
    exe = tmp_path / "fake"
    exe.write_text("")

    def fake_run(cmd, **kwargs):
        assert cmd == [str(exe), "replay"]
        return _FakeProcess(0, '{"expected": 5.0, "actual": 4.0}')

    monkeypatch.setattr(mf.subprocess, "run", fake_run)
    result = mf.replay_candidate(exe, ARTIFACT, [1.0] * 5)
    assert result["expected"] == 5.0
    assert result["actual"] == 4.0


def test_replay_candidate_raises_on_bad_output(tmp_path, monkeypatch):
    exe = tmp_path / "fake"
    exe.write_text("")

    def fake_run(cmd, **kwargs):
        return _FakeProcess(0, "not json")

    monkeypatch.setattr(mf.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError):
        mf.replay_candidate(exe, ARTIFACT, [1.0] * 5)


def test_main_rejects_non_object_artifact(tmp_path, capsys):
    art = tmp_path / "artifact.json"
    art.write_text("null")
    exe = tmp_path / "fake"
    exe.write_text("")
    rc = mf.main([str(art), "--executable", str(exe),
                  "--out", str(tmp_path / "out.json")])
    assert rc == 1
    assert "minimization failed" in capsys.readouterr().err


def test_main_minimizes_grid_overlap(tmp_path, monkeypatch):
    # Simulate the repro/grid-overlap binary replayed as "correct": single-block
    # cases (n <= block 32) sum correctly, multi-block cases (n > 32) reproduce
    # the grid-stride overlap bug.
    seed = {
        "schema": "vanta.failure.v0",
        "kernel": "reduce_sum",
        "implementation": "correct",
        "input": {"dtype": "float32", "shape": [255],
                  "values": [1.0] * 255},
        "launch": {"block_threads": 32, "grid_blocks": 8},
        "environment": {"gpu": "Tesla T4", "compute_capability": "7.5"},
        "observed": {"kind": "correctness_mismatch", "expected": 255.0,
                     "actual": 1144.0},
    }
    art = _write(tmp_path, seed)
    runner = tmp_path / "fake"
    runner.write_text("")

    def fake_run(cmd, **kwargs):
        payload = kwargs["input"]
        lines = payload.strip().split("\n")
        assert lines[0] == "correct"
        n = int(lines[1])
        expected = float(n)
        # bug surfaces only when more than one block is needed
        actual = expected if n <= 32 else expected + 100.0
        return _FakeProcess(0, json.dumps(
            {"expected": expected, "actual": actual}))

    monkeypatch.setattr(mf.subprocess, "run", fake_run)

    out = tmp_path / "min.json"
    rc = mf.main([str(art), "--executable", str(runner), "--out", str(out)])
    assert rc == 0
    result = json.loads(out.read_text())
    assert result["input"]["shape"] == [33]
    assert len(result["input"]["values"]) == 33
    assert result["minimization"]["original_length"] == 255
    assert result["minimization"]["minimized_length"] == 33
    assert result["observed"]["expected"] == 33.0
    assert result["observed"]["actual"] == 133.0
