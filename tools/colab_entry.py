#!/usr/bin/env python3
"""Colab entrypoint for the first Vanta reduction proof.

This script runs only on a GPU-enabled Colab runtime. It detects the
environment, compiles the single reduction source with nvcc, runs the
correct and faulty campaigns, replays the captured failure, and runs
Compute Sanitizer memcheck. Every artifact is written under `results/`.

It deliberately avoids CMake for now. When the repository version is
stable, swap the nvcc invocation for a CMake build.

Usage on Colab:
    !python3 tools/colab_entry.py --source cuda/reduction_proof.cu \
        --artifact failures/reduction-tail-v0.json --out results/first-campaign
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(command, **kwargs) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(command)}")
    return subprocess.run(command, text=True, capture_output=True, **kwargs)


def require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise RuntimeError(f"{tool} not found on PATH")
    return path


def probe_environment() -> dict:
    nvidia = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                  "--format=csv,noheader"])
    nvcc = run([require("nvcc"), "--version"])
    sanitizer = run([require("compute-sanitizer"), "--version"])
    python = run([sys.executable, "--version"])
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": nvidia.stdout.strip(),
        "nvcc": nvcc.stdout.strip(),
        "compute_sanitizer": sanitizer.stdout.strip(),
        "python": python.stdout.strip(),
    }


def compile_source(nvcc: str, source: Path, output: Path, log: Path) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    process = run([
        nvcc, "-std=c++20", "-O0", "-lineinfo", "-arch=sm_75",
        str(source), "-o", str(output),
    ])
    log.write_text(process.stdout + process.stderr)
    print(process.stderr)
    return process.returncode == 0


def run_campaign(binary: Path, mode: str, log: Path) -> bool:
    process = run([str(binary), mode])
    log.write_text(process.stdout + process.stderr)
    print(process.stdout)
    if process.stderr.strip():
        print(process.stderr, file=sys.stderr)
    return process.returncode == 0


def replay_failure(binary: Path, artifact: Path, log: Path):
    process = run([
        sys.executable, "tools/replay_failure.py", str(artifact),
        "--executable", str(binary), "--json",
    ])
    log.write_text(process.stdout + process.stderr)
    print(process.stdout)
    classification = None
    try:
        classification = json.loads(process.stdout).get("classification")
    except json.JSONDecodeError:
        pass
    return process.returncode == 0, classification


def run_memcheck(binary: Path, log: Path) -> bool:
    process = run([
        require("compute-sanitizer"), "--tool", "memcheck",
        "--error-exitcode", "1", str(binary), "correct",
    ])
    log.write_text(process.stdout + process.stderr)
    print(process.stdout)
    if process.stderr.strip():
        print(process.stderr, file=sys.stderr)
    return process.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("cuda/reduction_proof.cu"))
    parser.add_argument("--artifact", type=Path,
                        default=Path("failures/reduction-tail-v0.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/first-campaign"))
    args = parser.parse_args(argv)

    if shutil.which("nvidia-smi") is None:
        print("ERROR: nvidia-smi not found. Select a GPU runtime in Colab.",
              file=sys.stderr)
        return 2

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    binary = out / "reduction_proof"

    env = probe_environment()
    (out / "environment.json").write_text(json.dumps(env, indent=2))

    compile_ok = compile_source(require("nvcc"), args.source, binary,
                                out / "build.log")

    if not compile_ok or not binary.exists():
        summary = {
            "schema": "vanta.campaign.v0",
            "source": str(args.source),
            "artifact": str(args.artifact),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "compile_ok": compile_ok,
            "correct_campaign_ok": False,
            "faulty_campaign_detected": False,
            "replay_classification": None,
            "memcheck_ok": False,
            "status": "fail",
        }
        (out / "campaign_summary.json").write_text(
            json.dumps(summary, indent=2))
        print("Vanta first-campaign status: fail (compile)")
        return 1

    correct_ok = run_campaign(binary, "correct", out / "correct.log")
    faulty_ok = run_campaign(binary, "faulty", out / "faulty.log")
    replay_ok, replay_class = replay_failure(binary, args.artifact,
                                             out / "replay.log")
    memcheck_ok = run_memcheck(binary, out / "memcheck.log")

    status = "pass"
    if not (compile_ok and correct_ok and faulty_ok and replay_ok
            and memcheck_ok):
        status = "fail"

    summary = {
        "schema": "vanta.campaign.v0",
        "source": str(args.source),
        "artifact": str(args.artifact),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "compile_ok": compile_ok,
        "correct_campaign_ok": correct_ok,
        "faulty_campaign_detected": faulty_ok,
        "replay_classification": replay_class,
        "memcheck_ok": memcheck_ok,
        "status": status,
    }
    (out / "campaign_summary.json").write_text(json.dumps(summary, indent=2))

    print("Vanta first-campaign status:", status)
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
