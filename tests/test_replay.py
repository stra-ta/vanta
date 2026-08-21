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


def _write(tmp_path, data):
    p = tmp_path / "artifact.json"
    p.write_text(json.dumps(data))
    return p


def test_load_artifact_ok(tmp_path):
    assert rf.load_artifact(_write(tmp_path, ARTIFACT))["kernel"] == "reduce_sum"


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
