import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "experiments" / "cuda-mutations" / "run_campaign.py"
MANIFEST_PATH = ROOT / "experiments" / "cuda-mutations" / "manifest.json"
MANIFEST_V2_PATH = ROOT / "experiments" / "cuda-mutations" / "manifest-v2.json"
MANIFEST_V3_PATH = ROOT / "experiments" / "cuda-mutations" / "manifest-v3.json"
SPEC = importlib.util.spec_from_file_location("mutation_campaign", RUNNER_PATH)
campaign = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(campaign)


def test_manifest_has_preregistered_kernel_counts_and_ids():
    manifest = campaign.load_manifest(MANIFEST_PATH)
    mutations = manifest["mutations"]
    assert manifest["schema"] == "vanta.mutation-manifest.v1"
    assert manifest["study_id"] == "cuda-mutations-t4-v1"
    assert manifest["repetitions"] == 3
    assert [item["id"] for item in mutations] == list(campaign.REQUIRED_IDS)
    assert {kernel: sum(item["kernel"] == kernel for item in mutations)
            for kernel in campaign.KERNELS} == {
                "reduce_sum": 6, "vector_add": 6, "matmul": 6}


def test_manifest_validation_rejects_missing_preregistered_id():
    manifest = campaign.load_manifest(MANIFEST_PATH)
    manifest["mutations"] = manifest["mutations"][:-1]
    with pytest.raises(ValueError, match="exactly 18 mutations"):
        campaign.validate_manifest(manifest)


def test_exact_transformation_replaces_only_declared_occurrence():
    transformed = campaign.apply_transformation(
        "before OLD middle OLD after",
        {"old": "OLD", "new": "NEW", "expected_matches": 2, "occurrence": 2},
    )
    assert transformed == "before OLD middle NEW after"


def test_manifest_transformations_anchor_to_canonical_sources_and_adapter():
    manifest = campaign.load_manifest(MANIFEST_PATH)
    for mutation in manifest["mutations"]:
        source = (ROOT / "cuda" / campaign.KERNELS[mutation["kernel"]]).read_text()
        transform = mutation["transformation"]
        assert source.count(transform["old"]) == transform["expected_matches"]
        adapted = (campaign.adapt_vector_replay(source)
                   if mutation["kernel"] == "vector_add" else source)
        derived = campaign.apply_transformation(adapted, transform)
        assert derived != adapted


def test_transformation_rejects_unexpected_match_count_and_occurrence():
    with pytest.raises(ValueError, match="expected 2 matches"):
        campaign.apply_transformation(
            "only OLD", {"old": "OLD", "new": "NEW", "expected_matches": 2, "occurrence": 1})
    with pytest.raises(ValueError, match="out of range"):
        campaign.apply_transformation(
            "OLD", {"old": "OLD", "new": "NEW", "expected_matches": 1, "occurrence": 2})


def test_vector_replay_adapter_adds_correct_mode_and_full_actual_vector():
    source = (ROOT / "cuda" / "vector_add_proof.cu").read_text()
    adapted = campaign.adapt_vector_replay(source)
    assert 'implementation != "correct" && implementation != "faulty_tail"' in adapted
    assert 'implementation == "faulty_tail", actual' in adapted
    assert '",\\"actual\\":["' in adapted
    assert "std::cout << actual[i];" in adapted
    assert "__global__ void vector_add_kernel" in adapted


def test_vector_adapter_is_recorded_as_a_separate_patch(tmp_path):
    canonical = (ROOT / "cuda" / "vector_add_proof.cu").read_text()
    adapted = campaign.adapt_vector_replay(canonical)
    source = campaign.write_subject(tmp_path, "vector_add", "baseline",
                                    canonical, adapted, None)
    patch = source.with_name("adapter.patch")
    assert patch.exists()
    assert "-    if (implementation != \"faulty_tail\")" in patch.read_text()


def test_sanitizer_parser_handles_clean_mixed_case_summary_and_crash_text():
    assert not campaign.sanitizer_detected("========= ERROR SUMMARY: 0 errors")
    assert campaign.sanitizer_detected(
        "========= Error: Race reported\n"
        "========= RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)")
    assert campaign.sanitizer_detected("========= MEMCHECK SUMMARY: 2 errors")
    assert not campaign.sanitizer_detected("segmentation fault\nprocess exited 7")


def test_timeout_output_is_normalized_before_it_reaches_json(monkeypatch, tmp_path):
    def timed_out(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["fake"], 1, output=b"partial", stderr=b"error")

    monkeypatch.setattr(campaign.subprocess, "run", timed_out)
    runner = campaign.Campaign(tmp_path, tmp_path / "build", 1.0, 3)
    result = runner.run(["fake"], "timeout-test")
    assert result["timeout"] is True
    assert result["stdout"] == "partial"
    assert result["stderr"] == "error"
    json.dumps(result)
    assert (tmp_path / "logs" / "0001-timeout-test.log").read_text() == "partialerror"


def test_invalid_barrier_mutants_run_synccheck_only():
    manifest = campaign.load_manifest(MANIFEST_PATH)
    for mutation_id in ("R05", "M06"):
        mutation = next(item for item in manifest["mutations"] if item["id"] == mutation_id)
        assert mutation["eligible_tools"] == ["synccheck"]
        assert mutation["run_tools"] == ["synccheck"]
        assert mutation["not_applicable_reason"]


def test_not_applicable_tools_do_not_make_a_clean_subject_infrastructure_failure():
    checks = {"differential": [{"verdict": "no_signal"}],
              "metamorphic": [{"verdict": "no_signal"}]}
    sanitizer = {
        "memcheck": [{"verdict": "not_applicable"}],
        "racecheck": [{"verdict": "not_applicable"}],
        "synccheck": [{"verdict": "no_signal"}],
    }
    assert campaign.summarize_verdicts(checks, sanitizer) == "survived"


def test_local_cpu_references_and_operation_aware_tolerance():
    assert campaign.reference_reduce([1.0, -2.0, 4.5]) == 3.5
    assert campaign.reference_vector(2.0, [1.0, 3.0], [4.0, -1.0]) == [6.0, 5.0]
    assert campaign.reference_matmul([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], 2) == [19.0, 22.0, 43.0, 50.0]
    assert campaign.values_match("reduce_sum", 10.0, 10.0 + 1e-6, 1)
    assert not campaign.values_match("reduce_sum", 10.0, 10.1, 1)
    assert campaign.tolerance_for("matmul", 10.0, 24) > campaign.tolerance_for("matmul", 10.0, 4)


def test_generated_markdown_has_all_detector_columns():
    matrix = campaign.detection_matrix([{
        "id": "R01",
        "checks": {"differential": [{"verdict": "detected"}],
                   "metamorphic": [{"verdict": "no_signal"}]},
        "sanitizer": {tool: [{"verdict": "no_signal"}] for tool in campaign.TOOLS},
        "status": "killed",
    }])
    assert "| Differential | Metamorphic | memcheck | racecheck | synccheck | Status |" in matrix
    assert "| R01 | detected | no_signal | no_signal | no_signal | no_signal | killed |" in matrix


def test_v2_manifest_differs_from_v1_only_in_study_id_and_replaced_mutants():
    v1 = campaign.load_manifest(MANIFEST_PATH)
    v2 = campaign.load_manifest(MANIFEST_V2_PATH)
    assert v2["study_id"] == "cuda-mutations-t4-v2"
    allowed = {"study_id"}
    replaced = {"R03", "R05"}
    for key in ("schema", "repetitions"):
        assert v2[key] == v1[key]
    assert [item["id"] for item in v2["mutations"]] == [item["id"] for item in v1["mutations"]]
    for old, new in zip(v1["mutations"], v2["mutations"]):
        if old["id"] in replaced:
            continue
        assert old == new, f"unexpected v2 change in {old['id']}"
        allowed.add(old["id"])
    assert allowed | replaced >= set(allowed)


def test_v2_manifest_transformations_anchor_to_canonical_sources():
    manifest = campaign.load_manifest(MANIFEST_V2_PATH)
    for mutation in manifest["mutations"]:
        source = (ROOT / "cuda" / campaign.KERNELS[mutation["kernel"]]).read_text()
        transform = mutation["transformation"]
        assert source.count(transform["old"]) == transform["expected_matches"]
        adapted = (campaign.adapt_vector_replay(source)
                   if mutation["kernel"] == "vector_add" else source)
        derived = campaign.apply_transformation(adapted, transform)
        assert derived != adapted


def test_v2_r03_is_a_non_equivalent_deterministic_scaling_fault():
    manifest = campaign.load_manifest(MANIFEST_V2_PATH)
    r03 = next(item for item in manifest["mutations"] if item["id"] == "R03")
    assert r03["taxon"] != "odd_reduction_bound"
    source = (ROOT / "cuda" / "reduction_proof.cu").read_text()
    derived = campaign.apply_transformation(source, r03["transformation"])
    assert derived.count("partial[blockIdx.x] = 2.0f * shared[0];") == 1
    assert r03["eligible_tools"] == []
    assert r03["minimization"] == "prefix"


def test_v2_r05_keeps_valid_barrier_and_adds_invalid_syncwarp_mask():
    manifest = campaign.load_manifest(MANIFEST_V2_PATH)
    r05 = next(item for item in manifest["mutations"] if item["id"] == "R05")
    source = (ROOT / "cuda" / "reduction_proof.cu").read_text()
    derived = campaign.apply_transformation(source, r05["transformation"])
    # The valid post-load block barrier is retained before the fault.
    assert derived.count("shared[tid] = local;\n    __syncthreads();") == 2
    assert "__syncwarp(lower_half);" in derived
    assert "(tid % warpSize) <= warpSize / 2" in derived
    assert r05["eligible_tools"] == ["synccheck"]
    assert r05["run_tools"] == ["synccheck"]
    assert r05["not_applicable_reason"]


def test_v3_manifest_differs_from_v2_only_in_study_id_and_r05():
    v2 = campaign.load_manifest(MANIFEST_V2_PATH)
    v3 = campaign.load_manifest(MANIFEST_V3_PATH)
    assert v3["study_id"] == "cuda-mutations-t4-v3"
    assert [item["id"] for item in v3["mutations"]] == [item["id"] for item in v2["mutations"]]
    for old, new in zip(v2["mutations"], v3["mutations"]):
        if old["id"] == "R05":
            continue
        assert old == new, f"unexpected v3 change in {old['id']}"


def test_v3_r05_uses_warp_aligned_divergent_barriers_and_synccheck_only():
    manifest = campaign.load_manifest(MANIFEST_V3_PATH)
    r05 = next(item for item in manifest["mutations"] if item["id"] == "R05")
    source = (ROOT / "cuda" / "reduction_proof.cu").read_text()
    derived = campaign.apply_transformation(source, r05["transformation"])
    assert "if (tid < warpSize) {" in derived
    assert "if (tid >= warpSize) {" in derived
    assert "__syncwarp" not in derived
    # The split is warp-aligned, so a single-warp launch keeps one valid barrier.
    assert r05["trigger"] == {"n": 257, "block_threads": 128}
    assert r05["eligible_tools"] == ["synccheck"]
    assert r05["run_tools"] == ["synccheck"]
    assert r05["not_applicable_reason"]
    case = campaign.trigger_case("reduce_sum", r05["trigger"])
    assert case["block_threads"] >= 64


def test_runner_still_rejects_unknown_study_ids():
    with pytest.raises(ValueError, match="study_id must be one of"):
        campaign.validate_manifest({
            "schema": "vanta.mutation-manifest.v1",
            "study_id": "cuda-mutations-t4-v9",
            "repetitions": 3,
            "mutations": [],
        })


def test_main_returns_two_when_cuda_tools_are_unavailable(monkeypatch, capsys):
    monkeypatch.setattr(campaign.shutil, "which", lambda _tool: None)
    assert campaign.main([]) == 2
    assert "requires: nvcc, nvidia-smi, compute-sanitizer" in capsys.readouterr().err
