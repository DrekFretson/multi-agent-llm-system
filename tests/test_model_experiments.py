import json

import pytest

from src.evaluation import (
    ExperimentAnalyzer,
    ExperimentConfigError,
    ExperimentRunner,
    load_benchmark_dataset,
)


def _create_files(tmp_path):
    dataset = tmp_path / "tasks.json"
    dataset.write_text(json.dumps([{
        "task_id": "simple-1", "role": "coding", "request": "Write a function",
        "requirements": ["include tests"], "expected_properties": ["returns a value"],
        "criteria": ["correctness"],
    }]), encoding="utf-8")
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "name: test-bench\n"
        "dataset: tasks.json\n"
        "models:\n  coding: [coding, reasoning]\n"
        "generation_params:\n  coding:\n    temperature: 0\n    max_tokens: 700\n"
        "repeats: 2\nseed: 19\nlimits:\n  max_calls: 3\ncritic:\n  enabled: false\n",
        encoding="utf-8",
    )
    return config, dataset


def test_load_dataset_rejects_duplicate_task_ids(tmp_path):
    source = tmp_path / "bad.json"
    task = {"task_id": "same", "role": "coding", "request": "x",
            "expected_properties": ["y"], "criteria": ["z"]}
    source.write_text(json.dumps([task, task]), encoding="utf-8")
    with pytest.raises(ExperimentConfigError, match="unique"):
        load_benchmark_dataset(source)


def test_experiment_runs_same_task_per_model_and_repeat_and_exports_results(tmp_path):
    config, _ = _create_files(tmp_path)
    results_dir = tmp_path / "results"
    seen = []

    def run_case(**kwargs):
        seen.append(kwargs)
        collector = kwargs["metrics_collector"]
        model_id = kwargs["models"]["coding"]
        collector.record_llm_call(model_id=model_id, agent_id="coding", task_id=kwargs["task"]["task_id"],
            agent_role="coding", duration_ms=20, prompt_tokens=10, completion_tokens=5,
            total_tokens=15, finish_reason="stop", success=True)
        return {"status": "COMPLETED", "answer": "solution"}

    def quality(task, result):
        return {"quality_score": 1.0 if result["answer"] else 0.0, "tests_passed": True}

    output = ExperimentRunner(results_dir=results_dir).run(
        config, run_case, quality_evaluator=quality)
    assert len(seen) == 4
    assert {item["task"]["task_id"] for item in seen} == {"simple-1"}
    assert all(item["generation_params"]["coding"]["max_tokens"] == 700 for item in seen)
    assert len({item["metrics_collector"].run_id for item in seen}) == 4
    assert len(output["records"]) == 4
    assert output["records"][0]["api_calls"] == 1
    assert output["records"][0]["prompt_tokens"] == 10
    assert output["summary"]["recommendations"]["coding"]["recommended"] is not None
    assert (results_dir / (output["experiment_id"])).exists() is False
    paths = output["files"]
    assert json.loads(open(paths["json"], encoding="utf-8").read())
    assert len(open(paths["jsonl"], encoding="utf-8").readlines()) == 4
    assert len(open(paths["csv"], encoding="utf-8").readlines()) == 5


def test_recommendation_declines_to_choose_without_objective_quality():
    analysis = ExperimentAnalyzer().analyze([
        {"agent_role": "math", "model_id": "m1", "execution_status": "COMPLETED", "latency_ms": 5},
        {"agent_role": "math", "model_id": "m1", "execution_status": "COMPLETED", "latency_ms": 6},
        {"agent_role": "math", "model_id": "m2", "execution_status": "COMPLETED", "latency_ms": 2},
        {"agent_role": "math", "model_id": "m2", "execution_status": "FAILED", "latency_ms": 3},
    ])
    rec = analysis["recommendations"]["math"]
    assert rec["status"] == "insufficient_objective_quality_data"
    assert rec["recommended"] is None
    assert rec["fastest_observed"] == "m2"


def test_provider_failure_is_separate_from_quality_denominator():
    analysis = ExperimentAnalyzer().analyze([
        {"agent_role": "research", "model_id": "m1", "execution_status": "COMPLETED",
         "latency_ms": 4, "objective_quality": {"quality_score": 1.0}},
        {"agent_role": "research", "model_id": "m1", "execution_status": "ERROR",
         "provider_error_calls": 1, "error_type": "RateLimitError", "latency_ms": 2},
    ])
    group = analysis["groups"][0]
    assert group["runs"] == 2
    assert group["quality_sample_runs"] == 1
    assert group["success_rate"] == 1.0
    assert group["provider_error_runs"] == 1


def test_provider_failure_is_saved_and_not_scored_as_quality(tmp_path):
    config, _ = _create_files(tmp_path)
    checked = []

    class RateLimitError(Exception):
        pass

    def run_case(**kwargs):
        raise RateLimitError("rate limit")

    output = ExperimentRunner(results_dir=tmp_path / "results").run(
        config, run_case,
        quality_evaluator=lambda *_: checked.append(True) or {"quality_score": 1.0},
    )
    assert output["records"][0]["execution_status"] == "ERROR"
    assert output["records"][0]["error_type"] == "RateLimitError"
    assert output["records"][0]["provider_error_calls"] == 1
    assert checked == []
    group = next(row for row in output["summary"]["groups"] if row["model_id"] == "cohere/north-mini-code:free")
    assert group["quality_sample_runs"] == 0
    assert group["success_rate"] is None
    assert group["provider_error_runs"] == 2

