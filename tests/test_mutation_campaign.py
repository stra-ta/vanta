import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "experiments" / "cuda-mutations" / "run_campaign.py"
MANIFEST_PATH = ROOT / "experiments" / "cuda-mutations" / "manifest.json"
SPEC = importlib.util.spec_from_file_location("mutation_campaign", RUNNER_PATH)
campaign = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(campaign)


def test_manifest_has_preregistered_kernel_counts_and_ids():
    manifest = campaign.load_manifest(MANIFEST_PATH)
    mutations = manifest["mutations"]
    assert manifest["schema"] == "vanta.mutation-manifest.v1"
    assert manifest["study_id"] == "cuda-mutations-t4-v0"
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


def test_main_returns_two_when_cuda_tools_are_unavailable(monkeypatch, capsys):
    monkeypatch.setattr(campaign.shutil, "which", lambda _tool: None)
    assert campaign.main([]) == 2
    assert "requires: nvcc, nvidia-smi, compute-sanitizer" in capsys.readouterr().err
