import json
from types import SimpleNamespace
from unittest.mock import Mock

from src.coordination import compare_critic_modes


def test_critic_comparison_runs_paired_modes_and_persists_metrics(tmp_path):
    orchestrator = Mock()
    orchestrator.run_adaptive.side_effect = [
        SimpleNamespace(
            status="COMPLETED", errors=[], final_answer="without review",
            execution_metrics={"review_status": "NOT_REQUIRED", "calls": 2,
                               "prompt_tokens": 10, "completion_tokens": 20,
                               "duration_ms": 100},
            execution_history=[],
        ),
        SimpleNamespace(
            status="COMPLETED", errors=[], final_answer="reviewed",
            execution_metrics={"review_status": "APPROVED", "calls": 3,
                               "prompt_tokens": 14, "completion_tokens": 30,
                               "duration_ms": 150},
            execution_history=[
                {"event": "critic_refinement"}, {"event": "critic_review_rejected"}
            ],
        ),
    ]
    output = tmp_path / "experiments" / "critic_comparison.jsonl"

    rows = compare_critic_modes(
        orchestrator,
        ["Implement a small function"],
        output_path=output,
        run_options={"max_workers": 1, "max_calls": 8},
        quality_evaluator=lambda _request, result: result.status == "COMPLETED",
    )

    assert [row["critic_enabled"] for row in rows] == [False, True]
    assert rows[0]["review_status"] == "NOT_REQUIRED"
    assert rows[1]["refinement_cycles"] == 1
    assert rows[1]["critic_rejections"] == 1
    assert rows[1]["calls"] > rows[0]["calls"]
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2
    stored = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert stored[0]["objective_quality"] is True
    assert orchestrator.run_adaptive.call_args_list[0].kwargs["critic_enabled"] is False
    assert orchestrator.run_adaptive.call_args_list[1].kwargs["critic_enabled"] is True
