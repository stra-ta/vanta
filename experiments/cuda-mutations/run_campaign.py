#!/usr/bin/env python3
"""Controlled CUDA mutation campaign derived from a pinned Vanta revision."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
KERNELS = {
    "reduce_sum": "reduction_proof.cu",
    "vector_add": "vector_add_proof.cu",
    "matmul": "matmul_proof.cu",
}
TOOLS = ("memcheck", "racecheck", "synccheck")
FLAGS = ("-std=c++20", "-O0", "-lineinfo", "-arch=sm_75")
REQUIRED_IDS = tuple(f"R{i:02d}" for i in range(1, 7)) + tuple(
    f"V{i:02d}" for i in range(1, 7)) + tuple(f"M{i:02d}" for i in range(1, 7))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != "vanta.mutation-manifest.v1":
        raise ValueError("manifest schema must be vanta.mutation-manifest.v1")
    if manifest.get("study_id") != "cuda-mutations-t4-v0":
        raise ValueError("manifest study_id must be cuda-mutations-t4-v0")
    if manifest.get("repetitions") != 3:
        raise ValueError("manifest repetitions must be 3")
    mutations = manifest.get("mutations")
    if not isinstance(mutations, list) or len(mutations) != 18:
        raise ValueError("manifest must contain exactly 18 mutations")
    ids = [item.get("id") for item in mutations]
    if len(set(ids)) != len(ids) or set(ids) != set(REQUIRED_IDS):
        raise ValueError("manifest IDs must be R01-R06, V01-V06, and M01-M06")
    counts = {kernel: sum(item.get("kernel") == kernel for item in mutations)
              for kernel in KERNELS}
    if counts != {kernel: 6 for kernel in KERNELS}:
        raise ValueError("manifest must contain six mutations for each kernel")
    for item in mutations:
        if item.get("kernel") not in KERNELS:
            raise ValueError(f"unknown kernel for {item.get('id')}")
        transform = item.get("transformation")
        trigger = item.get("trigger")
        if not isinstance(transform, dict) or not isinstance(trigger, dict):
            raise ValueError(f"missing transformation or trigger for {item['id']}")
        if not all(isinstance(transform.get(key), str) and transform[key]
                   for key in ("old", "new")):
            raise ValueError(f"invalid source strings for {item['id']}")
        if not isinstance(transform.get("expected_matches"), int) or transform["expected_matches"] < 1:
            raise ValueError(f"invalid expected_matches for {item['id']}")
        if not isinstance(transform.get("occurrence"), int) or transform["occurrence"] < 1:
            raise ValueError(f"invalid occurrence for {item['id']}")
        if transform["occurrence"] > transform["expected_matches"]:
            raise ValueError(f"occurrence exceeds expected_matches for {item['id']}")
        if not isinstance(trigger.get("n"), int) or not isinstance(trigger.get("block_threads"), int):
            raise ValueError(f"invalid trigger for {item['id']}")
        if not isinstance(item.get("eligible_tools"), list) or any(
                tool not in TOOLS for tool in item["eligible_tools"]):
            raise ValueError(f"invalid eligible_tools for {item['id']}")
        if item.get("minimization") not in ("prefix", "not_applicable"):
            raise ValueError(f"invalid minimization for {item['id']}")


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest root must be an object")
    validate_manifest(manifest)
    return manifest


def apply_transformation(source: str, transformation: dict[str, Any]) -> str:
    old = transformation["old"]
    expected = transformation["expected_matches"]
    occurrence = transformation["occurrence"]
    actual = source.count(old)
    if actual != expected:
        raise ValueError(
            f"transformation expected {expected} matches for {old!r}, found {actual}")
    if occurrence > actual:
        raise ValueError(f"transformation occurrence {occurrence} is out of range")
    parts = source.split(old)
    return old.join(parts[:occurrence]) + transformation["new"] + old.join(parts[occurrence:])


def adapt_vector_replay(source: str) -> str:
    """Adapt canonical vector replay without changing its kernel or campaign modes."""
    old_guard = '''    if (implementation != "faulty_tail") {
        std::cerr << "Unsupported implementation: " << implementation << '\\n';
        return 2;
    }'''
    new_guard = '''    if (implementation != "correct" && implementation != "faulty_tail") {
        std::cerr << "Unsupported implementation: " << implementation << '\\n';
        return 2;
    }'''
    old_call = """    if (!run_case(a, x, y, block_size, /*faulty=*/true, actual)) {"""
    new_call = """    if (!run_case(a, x, y, block_size,
                  implementation == "faulty_tail", actual)) {"""
    old_output = '''    std::vector<float> expected;
    reference_vadd(a, x, y, expected);
    const int idx = n - 1;

    std::cout << std::setprecision(9) << "{\\"n\\":" << n
              << ",\\"block_threads\\":" << block_size
              << ",\\"expected\\":" << expected[idx]
              << ",\\"actual\\":" << actual[idx] << "}" << '\\n';'''
    new_output = '''    std::cout << std::setprecision(9) << "{\\"n\\":" << n
              << ",\\"block_threads\\":" << block_size << ",\\"actual\\":[";
    for (int i = 0; i < n; ++i) {
        if (i > 0) {
            std::cout << ',';
        }
        std::cout << actual[i];
    }
    std::cout << "]}" << '\\n';'''
    for old, new in ((old_guard, new_guard), (old_call, new_call),
                     (old_output, new_output)):
        if source.count(old) != 1:
            raise ValueError("canonical vector replay adapter anchor changed")
        source = source.replace(old, new, 1)
    return source


def sanitizer_detected(output: str) -> bool:
    if re.search(r"^\s*=========\s*error:", output, re.IGNORECASE | re.MULTILINE):
        return True
    for summary in re.findall(
            r"^\s*=========\s*(?:error|racecheck|synccheck|memcheck)\s+summary:\s*(.*)$",
            output, re.IGNORECASE | re.MULTILINE):
        if re.search(r"\b[1-9][0-9]*\s+(?:errors?|hazards?)\b", summary,
                     re.IGNORECASE):
            return True
    return False


def reference_reduce(values: list[float]) -> float:
    return math.fsum(float(value) for value in values)


def reference_vector(a: float, x: list[float], y: list[float]) -> list[float]:
    return [a * left + right for left, right in zip(x, y)]


def reference_matmul(a: list[float], b: list[float], n: int) -> list[float]:
    return [math.fsum(float(a[row * n + k]) * float(b[k * n + col])
                      for k in range(n))
            for row in range(n) for col in range(n)]


def tolerance_for(kernel: str, expected: float, n: int = 1) -> float:
    if kernel == "matmul":
        return max(1.0, abs(expected)) * n * 3.0 * math.ldexp(1.0, -23)
    return 1e-5 * max(1.0, abs(expected))


def values_match(kernel: str, expected: Any, actual: Any, n: int) -> bool:
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(expected) == len(actual) and
                all(values_match(kernel, left, right, n)
                    for left, right in zip(expected, actual)))
    return (isinstance(actual, (int, float)) and
            abs(float(expected) - float(actual)) <= tolerance_for(kernel, float(expected), n))


def literal_values(n: int, seed: int) -> list[float]:
    base = (3.0, -1.5, 4.0, 0.5, -2.0, 7.0, -3.5, 2.0, 5.5, -4.0, 1.25)
    return [base[(index + seed) % len(base)] for index in range(n)]


def differential_cases(kernel: str) -> list[dict[str, Any]]:
    if kernel == "reduce_sum":
        return [{"n": n, "block_threads": block, "values": literal_values(n, seed)}
                for n, block, seed in ((31, 32, 0), (33, 32, 1), (65, 32, 2),
                                        (257, 128, 3))]
    if kernel == "vector_add":
        return [{"n": n, "block_threads": block, "a": a,
                 "x": literal_values(n, x_seed), "y": literal_values(n, y_seed)}
                for n, block, a, x_seed, y_seed in ((31, 32, 1.5, 0, 4),
                                                     (33, 32, -2.0, 1, 5),
                                                     (65, 32, 0.75, 2, 6))]
    return [{"n": n, "block_threads": 16, "a": literal_values(n * n, left),
             "b": literal_values(n * n, right)}
            for n, left, right in ((16, 0, 3), (24, 1, 5))]


def payload_for(kernel: str, case: dict[str, Any]) -> str:
    if kernel == "reduce_sum":
        lines = ["correct", str(case["n"]), str(case["block_threads"]),
                 " ".join(map(str, case["values"]))]
    elif kernel == "vector_add":
        lines = ["correct", str(case["n"]), str(case["block_threads"]), str(case["a"]),
                 " ".join(map(str, case["x"])), " ".join(map(str, case["y"]))]
    else:
        lines = ["correct", str(case["n"]), str(case["block_threads"]),
                 " ".join(map(str, case["a"])), " ".join(map(str, case["b"]))]
    return "\n".join(lines) + "\n"


def expected_for(kernel: str, case: dict[str, Any]) -> Any:
    if kernel == "reduce_sum":
        return reference_reduce(case["values"])
    if kernel == "vector_add":
        return reference_vector(case["a"], case["x"], case["y"])
    return reference_matmul(case["a"], case["b"], case["n"])


class Campaign:
    def __init__(self, out: Path, build_dir: Path, timeout: float, repetitions: int):
        self.out = out
        self.build_dir = build_dir
        self.timeout = timeout
        self.repetitions = repetitions
        self.commands: list[dict[str, Any]] = []
        self.sequence = 0

    def run(self, command: list[str], label: str, stdin: str | None = None) -> dict[str, Any]:
        self.sequence += 1
        log = self.out / "logs" / f"{self.sequence:04d}-{label}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        try:
            process = subprocess.run(command, input=stdin, text=True, capture_output=True,
                                     timeout=self.timeout, cwd=ROOT, check=False)
            result = {"command": command, "returncode": process.returncode,
                      "stdout": process.stdout, "stderr": process.stderr,
                      "timeout": False}
        except subprocess.TimeoutExpired as error:
            result = {"command": command, "returncode": None,
                      "stdout": error.stdout or "", "stderr": error.stderr or "",
                      "timeout": True}
        raw = str(result["stdout"]) + str(result["stderr"])
        log.write_text(raw, encoding="utf-8")
        evidence = {key: value for key, value in result.items() if key not in ("stdout", "stderr")}
        evidence.update({"log": str(log), "raw_output_sha256": sha256_text(raw),
                         "elapsed_seconds": time.monotonic() - started})
        self.commands.append(evidence)
        result.update(evidence)
        return result

    def actual(self, binary: Path, kernel: str, case: dict[str, Any], label: str) -> dict[str, Any]:
        result = self.run([str(binary), "replay"], label, payload_for(kernel, case))
        if result["timeout"]:
            return {"verdict": "timeout", "command": result}
        if result["returncode"] != 0:
            return {"verdict": "execution_error", "command": result}
        lines = str(result["stdout"]).rstrip().splitlines()
        if not lines:
            return {"verdict": "execution_error", "command": result, "reason": "missing JSON output"}
        try:
            decoded = json.loads(lines[-1])
        except json.JSONDecodeError:
            return {"verdict": "execution_error", "command": result, "reason": "invalid final JSON"}
        if not isinstance(decoded, dict) or "actual" not in decoded:
            return {"verdict": "execution_error", "command": result, "reason": "missing actual"}
        return {"verdict": "ok", "command": result, "actual": decoded["actual"],
                "case": case}

    def replay(self, binary: Path, kernel: str, case: dict[str, Any], label: str) -> dict[str, Any]:
        observed = self.actual(binary, kernel, case, label)
        if observed["verdict"] != "ok":
            return observed
        expected = expected_for(kernel, case)
        match = values_match(kernel, expected, observed["actual"], case["n"])
        return {"verdict": "no_signal" if match else "detected",
                "command": observed["command"], "expected": expected,
                "actual": observed["actual"], "case": case}

    def compile(self, source: Path, binary: Path, label: str) -> dict[str, Any]:
        binary.parent.mkdir(parents=True, exist_ok=True)
        return self.run(["nvcc", *FLAGS, str(source), "-o", str(binary)], label)

    def sanitizer(self, binary: Path, tool: str, case: dict[str, Any], kernel: str,
                  label: str) -> dict[str, Any]:
        result = self.run(["compute-sanitizer", f"--tool={tool}", str(binary), "replay"],
                          label, payload_for(kernel, case))
        raw = str(result["stdout"]) + str(result["stderr"])
        if result["timeout"]:
            verdict = "timeout"
        elif sanitizer_detected(raw):
            verdict = "detected"
        elif re.search(r"compute-sanitizer.*(?:internal|unsupported|unable to initialize)",
                       raw, re.IGNORECASE):
            verdict = "tool_error"
        elif result["returncode"] != 0:
            verdict = "execution_error"
        else:
            verdict = "no_signal"
        return {"verdict": verdict, "command": result}


def property_check(campaign: Campaign, binary: Path, kernel: str, subject: str,
                   name: str, cases: list[dict[str, Any]],
                   relation) -> dict[str, Any]:
    """Evaluate a relation between GPU executions, not a CPU reference variant."""
    observations = [campaign.actual(binary, kernel, case, f"{subject}-meta-{name}-{index}")
                    for index, case in enumerate(cases, 1)]
    failures = [item for item in observations if item["verdict"] != "ok"]
    if failures:
        return {"property": name, "verdict": failures[0]["verdict"],
                "observations": observations}
    actuals = [item["actual"] for item in observations]
    n = max(case["n"] for case in cases)
    return {"property": name,
            "verdict": "no_signal" if relation(actuals, n) else "detected",
            "observations": observations}


def transpose(values: list[float], n: int) -> list[float]:
    return [values[col * n + row] for row in range(n) for col in range(n)]


def run_metamorphic(campaign: Campaign, binary: Path, kernel: str,
                    subject: str) -> list[dict[str, Any]]:
    if kernel == "reduce_sum":
        base = literal_values(33, 0)
        other = literal_values(17, 4)
        base_case = {"n": 33, "block_threads": 32, "values": base}
        return [
            property_check(campaign, binary, kernel, subject, "permutation", [
                base_case,
                {"n": 33, "block_threads": 32, "values": list(reversed(base))},
            ], lambda values, n: values_match(kernel, values[0], values[1], n)),
            property_check(campaign, binary, kernel, subject, "zero_padding", [
                base_case,
                {"n": 40, "block_threads": 32, "values": base + [0.0] * 7},
            ], lambda values, n: values_match(kernel, values[0], values[1], n)),
            property_check(campaign, binary, kernel, subject, "concatenation_additivity", [
                base_case,
                {"n": 17, "block_threads": 32, "values": other},
                {"n": 50, "block_threads": 32, "values": base + other},
            ], lambda values, n: values_match(kernel, values[0] + values[1], values[2], n)),
            property_check(campaign, binary, kernel, subject, "launch_invariance", [
                base_case,
                {"n": 33, "block_threads": 64, "values": base},
            ], lambda values, n: values_match(kernel, values[0], values[1], n)),
        ]
    if kernel == "vector_add":
        x, y, a = literal_values(33, 0), literal_values(33, 4), 1.5
        base_case = {"n": 33, "block_threads": 32, "a": a, "x": x, "y": y}
        shifted = [value + 2.25 for value in y]
        first, second = 16, 17
        return [
            property_check(campaign, binary, kernel, subject, "zero_y_identity", [
                {"n": 33, "block_threads": 32, "a": a, "x": x, "y": [0.0] * 33},
            ], lambda values, n: values_match(kernel, [a * value for value in x], values[0], n)),
            property_check(campaign, binary, kernel, subject, "y_shift_relation", [
                base_case,
                {"n": 33, "block_threads": 32, "a": a, "x": x, "y": shifted},
            ], lambda values, n: values_match(
                kernel, [value + 2.25 for value in values[0]], values[1], n)),
            property_check(campaign, binary, kernel, subject, "partition_consistency", [
                base_case,
                {"n": first, "block_threads": 32, "a": a,
                 "x": x[:first], "y": y[:first]},
                {"n": second, "block_threads": 32, "a": a,
                 "x": x[first:], "y": y[first:]},
            ], lambda values, n: values_match(kernel, values[0], values[1] + values[2], n)),
            property_check(campaign, binary, kernel, subject, "launch_invariance", [
                base_case,
                {"n": 33, "block_threads": 64, "a": a, "x": x, "y": y},
            ], lambda values, n: values_match(kernel, values[0], values[1], n)),
        ]
    n = 24
    matrix = literal_values(n * n, 2)
    identity = [1.0 if row == col else 0.0 for row in range(n) for col in range(n)]
    zero = [0.0] * (n * n)
    a, b = literal_values(n * n, 1), literal_values(n * n, 5)
    return [
        property_check(campaign, binary, kernel, subject, "zero", [
            {"n": n, "block_threads": 16, "a": matrix, "b": zero},
        ], lambda values, _n: values_match(kernel, zero, values[0], n)),
        property_check(campaign, binary, kernel, subject, "identity", [
            {"n": n, "block_threads": 16, "a": matrix, "b": identity},
        ], lambda values, _n: values_match(kernel, matrix, values[0], n)),
        property_check(campaign, binary, kernel, subject, "transpose_relation", [
            {"n": n, "block_threads": 16, "a": a, "b": b},
            {"n": n, "block_threads": 16, "a": transpose(b, n),
             "b": transpose(a, n)},
        ], lambda values, _n: values_match(kernel, transpose(values[0], n), values[1], n)),
    ]


def run_checks(campaign: Campaign, binary: Path, kernel: str, subject: str) -> dict[str, Any]:
    return {
        "differential": [campaign.replay(binary, kernel, case, f"{subject}-diff-{index}")
                           for index, case in enumerate(differential_cases(kernel), 1)],
        "metamorphic": run_metamorphic(campaign, binary, kernel, subject),
    }


def trigger_case(kernel: str, trigger: dict[str, int]) -> dict[str, Any]:
    for case in differential_cases(kernel):
        if case["n"] == trigger["n"] and case["block_threads"] == trigger["block_threads"]:
            return case
    raise ValueError(f"trigger {trigger} is not represented by the shared corpus")


def repeated_failure(campaign: Campaign, binary: Path, kernel: str, case: dict[str, Any],
                     subject: str) -> dict[str, Any]:
    attempts = [campaign.replay(binary, kernel, case, f"{subject}-replay-{index}")
                for index in range(1, 4)]
    stable = all(item["verdict"] == "detected" for item in attempts)
    return {"classification": "stable_same_failure" if stable else "unstable_failure",
            "attempts": attempts}


def minimize_prefix(campaign: Campaign, binary: Path, kernel: str, case: dict[str, Any],
                    subject: str) -> dict[str, Any]:
    values_key = "values" if kernel == "reduce_sum" else "x" if kernel == "vector_add" else None
    if values_key is None:
        return {"status": "not_applicable"}
    initial = campaign.replay(binary, kernel, case, f"{subject}-min-seed")
    if initial["verdict"] != "detected":
        return {"status": "not_applicable", "reason": "predicate did not remain true"}
    original = list(case[values_key])
    for length in range(1, len(original) + 1):
        candidate = dict(case)
        candidate[values_key] = original[:length]
        candidate["n"] = length
        if kernel == "vector_add":
            candidate["y"] = case["y"][:length]
        result = campaign.replay(binary, kernel, candidate, f"{subject}-min-{length}")
        if result["verdict"] == "detected":
            return {"status": "prefix", "original_length": len(original),
                    "minimized_length": length, "case": candidate}
    return {"status": "not_applicable", "reason": "predicate did not remain true"}


def summarize_verdicts(checks: dict[str, Any], sanitizer: dict[str, list[dict[str, Any]]]) -> str:
    verdicts = [item["verdict"] for group in checks.values() for item in group]
    verdicts.extend(item["verdict"] for group in sanitizer.values() for item in group)
    sanitizer_flaky = any(len({item["verdict"] for item in group}) > 1
                          for group in sanitizer.values())
    if sanitizer_flaky:
        return "flaky"
    if "detected" in verdicts:
        return "killed"
    if verdicts and all(verdict == "no_signal" for verdict in verdicts):
        return "survived"
    return "infrastructure_failure"


def detection_matrix(results: list[dict[str, Any]]) -> str:
    lines = ["# CUDA Mutation Detection Matrix", "", "| Subject | Differential | Metamorphic | memcheck | racecheck | synccheck | Status |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for result in results:
        def group_verdict(items: list[dict[str, Any]]) -> str:
            values = {item["verdict"] for item in items}
            return ", ".join(sorted(values)) if values else "not_run"
        checks = result.get("checks", {})
        sanitizer = result.get("sanitizer", {})
        lines.append("| {id} | {d} | {m} | {mem} | {race} | {sync} | {status} |".format(
            id=result["id"], d=group_verdict(checks.get("differential", [])),
            m=group_verdict(checks.get("metamorphic", [])),
            mem=group_verdict(sanitizer.get("memcheck", [])),
            race=group_verdict(sanitizer.get("racecheck", [])),
            sync=group_verdict(sanitizer.get("synccheck", [])), status=result.get("status", "not_run")))
    return "\n".join(lines) + "\n"


def canonical_source(campaign: Campaign, revision: str, kernel: str) -> str:
    result = campaign.run(["git", "show", f"{revision}:cuda/{KERNELS[kernel]}"], f"source-{kernel}")
    if result["timeout"] or result["returncode"] != 0:
        raise RuntimeError(f"could not fetch canonical {kernel} source from {revision}")
    return str(result["stdout"])


def write_subject(out: Path, kernel: str, subject: str, canonical: str, derived: str,
                  transformation: dict[str, Any] | None) -> Path:
    directory = out / "subjects" / kernel / subject
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / KERNELS[kernel]
    source.write_text(derived, encoding="utf-8")
    if transformation is not None:
        patch = "".join(difflib.unified_diff(canonical.splitlines(keepends=True),
                                               derived.splitlines(keepends=True),
                                               fromfile=f"a/{KERNELS[kernel]}",
                                               tofile=f"b/{KERNELS[kernel]}"))
        (directory / "subject.patch").write_text(patch, encoding="utf-8")
    elif canonical != derived:
        patch = "".join(difflib.unified_diff(canonical.splitlines(keepends=True),
                                               derived.splitlines(keepends=True),
                                               fromfile=f"a/{KERNELS[kernel]}",
                                               tofile=f"b/{KERNELS[kernel]}"))
        (directory / "adapter.patch").write_text(patch, encoding="utf-8")
    return source


def require_cuda() -> bool:
    missing = [tool for tool in ("nvcc", "nvidia-smi", "compute-sanitizer") if not shutil.which(tool)]
    if missing:
        print("CUDA mutation campaign requires: " + ", ".join(missing), file=sys.stderr)
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifest.json"))
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "cuda-mutation-campaign-v0")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build" / "cuda-mutations")
    parser.add_argument("--only", nargs="+", metavar="ID")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not require_cuda():
        return 2
    try:
        manifest = load_manifest(args.manifest)
        selected = manifest["mutations"]
        if args.only:
            unknown = set(args.only) - set(REQUIRED_IDS)
            if unknown:
                raise ValueError("unknown mutation IDs: " + ", ".join(sorted(unknown)))
            selected = [item for item in selected if item["id"] in set(args.only)]
        args.out.mkdir(parents=True, exist_ok=True)
        args.build_dir.mkdir(parents=True, exist_ok=True)
        campaign = Campaign(args.out, args.build_dir, args.timeout, manifest["repetitions"])
        revision_result = campaign.run(["git", "rev-parse", "HEAD"], "revision")
        if revision_result["returncode"] != 0 or revision_result["timeout"]:
            raise RuntimeError("git rev-parse HEAD failed")
        revision = str(revision_result["stdout"]).strip()
        environment = {
            "nvidia_smi": campaign.run(["nvidia-smi"], "environment-gpu"),
            "nvcc": campaign.run(["nvcc", "--version"], "environment-nvcc"),
            "compute_sanitizer": campaign.run(
                ["compute-sanitizer", "--version"], "environment-sanitizer"),
        }
        atomic_json(args.out / "environment.json", {
            "baseline_commit": revision,
            "probes": {
                name: {key: result[key] for key in (
                    "command", "returncode", "timeout", "log",
                    "raw_output_sha256", "elapsed_seconds")}
                for name, result in environment.items()
            },
            "compiler_flags": list(FLAGS),
        })
        sources = {kernel: canonical_source(campaign, revision, kernel) for kernel in KERNELS}
        adapted = dict(sources)
        adapted["vector_add"] = adapt_vector_replay(sources["vector_add"])
        adapter_hash = sha256_text(adapted["vector_add"])
        results: list[dict[str, Any]] = []
        campaign_doc: dict[str, Any] = {
            "schema": "vanta.cuda-mutation-campaign.v0", "study_id": manifest["study_id"],
            "baseline_commit": revision, "manifest_sha256": sha256_file(args.manifest),
            "compiler_flags": list(FLAGS), "vector_replay_adapter_sha256": adapter_hash,
            "environment_path": str(args.out / "environment.json"),
            "baselines": [], "mutants": results, "commands": campaign.commands,
        }
        for kernel in KERNELS:
            source = write_subject(args.out, kernel, "baseline", sources[kernel], adapted[kernel], None)
            binary = args.build_dir / f"{kernel}-baseline"
            compile_result = campaign.compile(source, binary, f"{kernel}-baseline-compile")
            baseline = {"id": f"{kernel}-baseline", "kernel": kernel,
                        "canonical_source_sha256": sha256_text(sources[kernel]),
                        "derived_source_sha256": sha256_text(adapted[kernel]),
                        "adapter_sha256": adapter_hash if kernel == "vector_add" else None,
                        "adapter_patch": (str(source.with_name("adapter.patch"))
                                          if kernel == "vector_add" else None),
                        "adapter_patch_sha256": (sha256_file(source.with_name("adapter.patch"))
                                                 if kernel == "vector_add" else None),
                        "source": str(source), "binary": str(binary),
                        "binary_sha256": None, "compile": compile_result}
            if compile_result["returncode"] != 0 or compile_result["timeout"]:
                baseline["status"] = "infrastructure_failure"
            else:
                baseline["binary_sha256"] = sha256_file(binary)
                baseline["checks"] = run_checks(campaign, binary, kernel, f"{kernel}-baseline")
                baseline["sanitizer"] = {tool: [campaign.sanitizer(
                    binary, tool, differential_cases(kernel)[0], kernel,
                    f"{kernel}-baseline-{tool}-{rep}") for rep in range(1, manifest["repetitions"] + 1)]
                    for tool in TOOLS}
                baseline["status"] = summarize_verdicts(baseline["checks"], baseline["sanitizer"])
            campaign_doc["baselines"].append(baseline)
            atomic_json(args.out / "campaign.json", campaign_doc)
            (args.out / "detection-matrix.md").write_text(detection_matrix(results), encoding="utf-8")
            if baseline["status"] != "survived":
                print(f"baseline failed: {kernel}", file=sys.stderr)
                return 1
        for mutation in selected:
            kernel = mutation["kernel"]
            base = adapted[kernel]
            # Check the preregistered anchor against HEAD before any shared replay adaptation.
            apply_transformation(sources[kernel], mutation["transformation"])
            derived = apply_transformation(base, mutation["transformation"])
            source = write_subject(args.out, kernel, mutation["id"], base, derived,
                                   mutation["transformation"])
            binary = args.build_dir / mutation["id"]
            compile_result = campaign.compile(source, binary, f"{mutation['id']}-compile")
            result = {"id": mutation["id"], "kernel": kernel, "taxon": mutation["taxon"],
                       "description": mutation["description"], "trigger": mutation["trigger"],
                       "eligible_tools": mutation["eligible_tools"], "source": str(source),
                       "binary": str(binary), "canonical_source_sha256": sha256_text(sources[kernel]),
                       "adapter_sha256": adapter_hash if kernel == "vector_add" else None,
                       "transformation_sha256": sha256_text(json.dumps(mutation["transformation"], sort_keys=True)),
                       "derived_source_sha256": sha256_text(derived),
                       "subject_patch": str(source.with_name("subject.patch")),
                       "subject_patch_sha256": sha256_file(source.with_name("subject.patch")),
                       "binary_sha256": None, "compile": compile_result}
            if compile_result["returncode"] != 0 or compile_result["timeout"]:
                result["status"] = "infrastructure_failure"
            else:
                result["binary_sha256"] = sha256_file(binary)
                result["checks"] = run_checks(campaign, binary, kernel, mutation["id"])
                first = next((item for item in result["checks"]["differential"]
                              if item["verdict"] == "detected"), None)
                if first:
                    failure_path = args.out / "mutants" / mutation["id"] / "failure.json"
                    atomic_json(failure_path, {
                        "schema": "vanta.mutation-failure.v0",
                        "mutation_id": mutation["id"],
                        "kernel": kernel,
                        "baseline_commit": revision,
                        "case": first["case"],
                        "expected": first["expected"],
                        "actual": first["actual"],
                    })
                    result["failure_artifact"] = str(failure_path)
                    result["failure_replay"] = repeated_failure(
                        campaign, binary, kernel, first["case"], mutation["id"])
                    result["minimization"] = (minimize_prefix(campaign, binary, kernel, first["case"], mutation["id"])
                                               if mutation["minimization"] == "prefix" else {"status": "not_applicable"})
                    if result["minimization"].get("status") == "prefix":
                        minimized_path = args.out / "mutants" / mutation["id"] / "minimized.json"
                        atomic_json(minimized_path, result["minimization"])
                        result["minimization_artifact"] = str(minimized_path)
                else:
                    result["minimization"] = {"status": "not_applicable"}
                trigger = trigger_case(kernel, mutation["trigger"])
                result["sanitizer"] = {tool: [dict(campaign.sanitizer(
                    binary, tool, trigger, kernel, f"{mutation['id']}-{tool}-{rep}"),
                    eligible=tool in mutation["eligible_tools"])
                    for rep in range(1, manifest["repetitions"] + 1)] for tool in TOOLS}
                result["status"] = summarize_verdicts(result["checks"], result["sanitizer"])
                if result.get("failure_replay", {}).get("classification") == "unstable_failure":
                    result["status"] = "flaky"
            results.append(result)
            atomic_json(args.out / "campaign.json", campaign_doc)
            (args.out / "detection-matrix.md").write_text(detection_matrix(results), encoding="utf-8")
        return 1 if any(item["status"] == "infrastructure_failure" for item in results) else 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"campaign error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
