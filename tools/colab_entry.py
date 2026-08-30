#!/usr/bin/env python3
"""Run the Vanta CUDA proof campaigns on a GPU-enabled Colab runtime.

Each kernel gets an isolated output directory under ``--out``. The aggregate
``campaign_summary.json`` records every build, campaign, replay, and memcheck
result. A nonzero exit status means at least one required step failed.

Usage on Colab:
    !python3 tools/colab_entry.py --out results/third-campaign

The legacy ``--source`` and ``--artifact`` options still override the
reduction proof. Use ``--kernel reduce_sum`` to run only that proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


KERNELS = {
    "reduce_sum": {
        "source": Path("cuda/reduction_proof.cu"),
        "artifact": Path("failures/reduction-tail-v0.json"),
    },
    "vector_add": {
        "source": Path("cuda/vector_add_proof.cu"),
        "artifact": Path("failures/vector_add-tail-v0.json"),
    },
    "matmul": {
        "source": Path("cuda/matmul_proof.cu"),
        "artifact": Path("failures/matmul-tile-v0.json"),
    },
}

# Sanitizer proof: each (mode, tool, expect_detected) pair. A correct kernel
# must stay clean; each broken mode must be caught by its tool.
SANITIZER_CASES = [
    ("correct", "memcheck", False),
    ("correct", "racecheck", False),
    ("correct", "synccheck", False),
    ("broken_oob", "memcheck", True),
    ("broken_race", "racecheck", True),
    ("broken_sync", "synccheck", True),
]


def run(command, **kwargs) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(command)}")
    return subprocess.run(command, text=True, capture_output=True, **kwargs)


def require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise RuntimeError(f"{tool} not found on PATH")
    return path


def probe_environment() -> dict:
    if shutil.which("nvidia-smi") is not None:
        nvidia = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                      "--format=csv,noheader"])
        gpu = nvidia.stdout.strip() or nvidia.stderr.strip()
    else:
        gpu = "nvidia-smi not found"

    def version(tool: str) -> str:
        path = shutil.which(tool)
        if path is None:
            return f"{tool} not found"
        process = run([path, "--version"])
        return process.stdout.strip() or process.stderr.strip()

    python = subprocess.run([sys.executable, "--version"],
                            text=True, capture_output=True)
    cpu = platform.processor()
    microcode = ""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith(("model name", "hardware", "processor")) and ":" in line and not cpu:
                cpu = line.split(":", 1)[1].strip()
            if line.lower().startswith("microcode") and ":" in line and not microcode:
                microcode = line.split(":", 1)[1].strip()
    commit = run(["git", "rev-parse", "HEAD"])
    status = run(["git", "status", "--porcelain", "--untracked-files=all"])
    kernel_cmdline = ""
    cmdline = Path("/proc/cmdline")
    if cmdline.exists():
        kernel_cmdline = cmdline.read_text(encoding="utf-8", errors="replace").strip()
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu,
        "nvcc": version("nvcc"),
        "compute_sanitizer": version("compute-sanitizer"),
        "python": python.stdout.strip() or python.stderr.strip(),
        "commit_sha": commit.stdout.strip(),
        "dirty_tree": bool(status.stdout),
        "status_sha256": hashlib.sha256(status.stdout.encode()).hexdigest(),
        "architecture": platform.machine(),
        "kernel": platform.release(),
        "cpu": cpu,
        "microcode": microcode,
        "kernel_cmdline": kernel_cmdline,
        "numa_nodes": sorted(path.name for path in Path("/sys/devices/system/node").glob("node[0-9]*")),
        "compiler_flags": ["-std=c++20", "-O0", "-lineinfo", "-arch=sm_75"],
        "build_type": "cuda-O0",
    }


def compile_source(nvcc: str, source: Path, output: Path, log: Path) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    process = run([
        nvcc, "-std=c++20", "-O0", "-lineinfo", "-arch=sm_75",
        str(source), "-o", str(output),
    ])
    log.write_text(process.stdout + process.stderr)
    if process.stderr.strip():
        print(process.stderr, file=sys.stderr)
    return process.returncode == 0


def run_campaign(binary: Path, mode: str, log: Path) -> bool:
    process = run([str(binary.resolve()), mode], cwd=log.parent)
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
        "--error-exitcode", "1", str(binary.resolve()), "correct",
    ])
    log.write_text(process.stdout + process.stderr)
    print(process.stdout)
    if process.stderr.strip():
        print(process.stderr, file=sys.stderr)
    return process.returncode == 0


def run_sanitizer_proof(nvcc: str, out: Path, provenance: dict) -> dict:
    """Build the sanitizer proof and run every (mode, tool) verdict check."""
    kernel_out = out / "sanitizer"
    kernel_out.mkdir(parents=True, exist_ok=True)
    binary = kernel_out / "sanitizer_proof"
    compile_ok = compile_source(
        nvcc, Path("cuda/sanitizer_proof.cu"), binary,
        kernel_out / "build.log")

    cases = []
    all_ok = compile_ok
    if compile_ok and binary.exists():
        for mode, tool, expect in SANITIZER_CASES:
            cmd = [
                sys.executable, "tools/check_sanitizer.py",
                "--executable", str(binary.resolve()),
                "--mode", mode, "--tool", tool,
            ]
            if expect:
                cmd.append("--expect-detected")
            process = run(cmd)
            (kernel_out / f"{mode}_{tool}.log").write_text(
                process.stdout + process.stderr)
            ok = process.returncode == 0
            all_ok = all_ok and ok
            cases.append({
                "mode": mode, "tool": tool,
                "expect_detected": expect,
                "returncode": process.returncode, "ok": ok,
            })
    else:
        all_ok = False

    return {
        "source": "cuda/sanitizer_proof.cu",
        "compile_ok": compile_ok,
        "cases": cases,
        "status": "pass" if all_ok else "fail",
        "provenance": provenance,
    }


def failed_kernel_result(name: str, config: dict, out: Path,
                          error: str, provenance: dict) -> dict:
    kernel_out = out / name
    kernel_out.mkdir(parents=True, exist_ok=True)
    (kernel_out / "build.log").write_text(f"FAILED: {error}\n")
    for skipped_log in ("correct.log", "faulty.log", "replay.log",
                        "memcheck.log"):
        (kernel_out / skipped_log).write_text(
            f"SKIPPED: {error}\n")
    return {
        "source": str(config["source"]),
        "artifact": str(config["artifact"]),
        "compile_ok": False,
        "correct_campaign_ok": False,
        "faulty_campaign_detected": False,
        "replay_classification": None,
        "memcheck_ok": False,
        "status": "fail",
        "error": error,
        "provenance": provenance,
    }


def run_kernel(name: str, config: dict, out: Path, provenance: dict) -> dict:
    kernel_out = out / name
    kernel_out.mkdir(parents=True, exist_ok=True)
    source = Path(config["source"])
    artifact = Path(config["artifact"])
    binary = kernel_out / name

    try:
        nvcc = require("nvcc")
    except RuntimeError as exc:
        return failed_kernel_result(name, config, out, str(exc), provenance)

    compile_ok = compile_source(nvcc, source, binary, kernel_out / "build.log")
    result = {
        "source": str(source),
        "artifact": str(artifact),
        "compile_ok": compile_ok,
        "correct_campaign_ok": False,
        "faulty_campaign_detected": False,
        "replay_classification": None,
        "memcheck_ok": False,
        "status": "fail",
        "provenance": provenance,
    }

    if not compile_ok or not binary.exists():
        for skipped_log in ("correct.log", "faulty.log", "replay.log",
                            "memcheck.log"):
            (kernel_out / skipped_log).write_text(
                "SKIPPED: compilation did not produce an executable\n")
        return result

    result["correct_campaign_ok"] = run_campaign(
        binary, "correct", kernel_out / "correct.log")
    result["faulty_campaign_detected"] = run_campaign(
        binary, "faulty", kernel_out / "faulty.log")
    replay_ok, replay_class = replay_failure(
        binary, artifact, kernel_out / "replay.log")
    result["replay_classification"] = replay_class
    try:
        result["memcheck_ok"] = run_memcheck(binary,
                                              kernel_out / "memcheck.log")
    except RuntimeError as exc:
        (kernel_out / "memcheck.log").write_text(f"FAILED: {exc}\n")
        result["memcheck_ok"] = False
        result["error"] = str(exc)
    result["status"] = (
        "pass"
        if (result["compile_ok"] and result["correct_campaign_ok"]
            and result["faulty_campaign_detected"] and replay_ok
            and result["replay_classification"] == "same_failure"
            and result["memcheck_ok"])
        else "fail"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", choices=["all", *KERNELS], default="all")
    parser.add_argument("--source", type=Path, default=None,
                        help="legacy reduction source override")
    parser.add_argument("--artifact", type=Path, default=None,
                        help="legacy reduction artifact override")
    parser.add_argument("--out", type=Path,
                        default=Path("results/second-campaign"))
    parser.add_argument("--seed", type=int, default=2026,
                        help="seed recorded in campaign provenance")
    args = parser.parse_args(argv)

    selected = list(KERNELS) if args.kernel == "all" else [args.kernel]
    configs = {name: dict(KERNELS[name]) for name in selected}
    if args.source is not None:
        configs["reduce_sum"]["source"] = args.source
    if args.artifact is not None:
        configs["reduce_sum"]["artifact"] = args.artifact

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    env = probe_environment()
    env["seed"] = args.seed
    provenance = dict(env)
    provenance["commands_path"] = str(out / "campaign_summary.json")
    provenance["command_line"] = [sys.executable, *sys.argv[1:]]
    (out / "environment.json").write_text(json.dumps(env, indent=2))

    if shutil.which("nvidia-smi") is None:
        error = "nvidia-smi not found; select a GPU runtime in Colab"
        print(f"ERROR: {error}", file=sys.stderr)
        kernel_results = {
                name: failed_kernel_result(name, config, out, error, provenance)
            for name, config in configs.items()
        }
        sanitizer_result = {
            "source": "cuda/sanitizer_proof.cu",
            "compile_ok": False, "cases": [], "status": "fail",
            "error": error, "provenance": provenance,
        }
    else:
        kernel_results = {
            name: run_kernel(name, config, out, provenance)
            for name, config in configs.items()
        }
        try:
            nvcc = require("nvcc")
            sanitizer_result = run_sanitizer_proof(nvcc, out, provenance)
        except RuntimeError as exc:
            sanitizer_result = {
                "source": "cuda/sanitizer_proof.cu",
                "compile_ok": False, "cases": [], "status": "fail",
                "error": str(exc),
                "provenance": provenance,
            }
    (out / "sanitizer_summary.json").write_text(
        json.dumps(sanitizer_result, indent=2))

    compile_ok = all(result["compile_ok"] for result in kernel_results.values())
    correct_ok = all(result["correct_campaign_ok"]
                     for result in kernel_results.values())
    faulty_ok = all(result["faulty_campaign_detected"]
                    for result in kernel_results.values())
    memcheck_ok = all(result["memcheck_ok"]
                      for result in kernel_results.values())
    sanitizer_ok = sanitizer_result["status"] == "pass"
    status = ("pass" if (all(result["status"] == "pass"
                            for result in kernel_results.values())
                         and sanitizer_ok)
              else "fail")

    summary = {
        "schema": "vanta.campaign.v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
        "command_line": [sys.executable, *sys.argv[1:]],
        "kernels": kernel_results,
        "sanitizer": sanitizer_result,
        "compile_ok": compile_ok,
        "correct_campaign_ok": correct_ok,
        "faulty_campaign_detected": faulty_ok,
        "memcheck_ok": memcheck_ok,
        "sanitizer_ok": sanitizer_ok,
        "status": status,
    }
    (out / "campaign_summary.json").write_text(json.dumps(summary, indent=2))
    print("Vanta campaign status:", status)
    if status != "pass" and shutil.which("nvidia-smi") is None:
        return 2
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
