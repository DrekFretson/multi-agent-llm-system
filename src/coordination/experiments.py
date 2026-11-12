"""Paired, append-only experiments comparing reviewed and unreviewed runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from src.coordination.task_store import TaskStore


def compare_critic_modes(
    orchestrator: Any,
    requests: Iterable[str],
    *,
    output_path: str | Path,
    run_options: Mapping[str, Any] | None = None,
    quality_evaluator: Callable[[str, Any], Any] | None = None,
) -> list[dict[str, Any]]:
    """Run each same request with Critic disabled and enabled, saving JSONL rows.

    ``quality_evaluator`` is optional and should return an objective external
    measure (for example, a deterministic test-suite result). Critic's score is
    retained as a review metric but is not treated as objective quality.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    options = dict(run_options or {})
    if "critic_enabled" in options or "task_store" in options:
        raise ValueError("run_options cannot override critic_enabled or task_store.")
    rows: list[dict[str, Any]] = []

    with path.open("a", encoding="utf-8") as stream:
        for request_index, request in enumerate(requests):
            if not isinstance(request, str) or not request.strip():
                raise ValueError("Each experiment request must be a non-empty string.")
            for critic_enabled in (False, True):
                row: dict[str, Any] = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "request_index": request_index,
                    "request": request,
                    "critic_enabled": critic_enabled,
                }
                try:
                    result = orchestrator.run_adaptive(
                        request,
                        task_store=TaskStore(),
                        critic_enabled=critic_enabled,
                        **options,
                    )
                except Exception as error:
                    row.update({
                        "status": "ERROR",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "review_status": "NOT_AVAILABLE",
                        "critic_rejections": 0,
                        "refinement_cycles": 0,
                        "calls": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "duration_ms": None,
                    })
                else:
                    metrics = getattr(result, "execution_metrics", {}) or {}
                    history = getattr(result, "execution_history", []) or []
                    row.update({
                        "status": getattr(result, "status", "UNKNOWN"),
                        "review_status": metrics.get("review_status", "UNKNOWN"),
                        "critic_rejections": sum(
                            1 for item in history
                            if isinstance(item, Mapping) and item.get("event") == "critic_review_rejected"
                        ),
                        "refinement_cycles": sum(
                            1 for item in history
                            if isinstance(item, Mapping) and item.get("event") == "critic_refinement"
                        ),
                        "calls": metrics.get("calls", 0),
                        "prompt_tokens": metrics.get("prompt_tokens", 0),
                        "completion_tokens": metrics.get("completion_tokens", 0),
                        "estimated_tokens": metrics.get("estimated_tokens", 0),
                        "duration_ms": metrics.get("duration_ms"),
                        "errors": getattr(result, "errors", []),
                        "final_answer": getattr(result, "final_answer", None),
                    })
                    if quality_evaluator is not None:
                        try:
                            row["objective_quality"] = quality_evaluator(request, result)
                        except Exception as error:
                            row["quality_evaluation_error"] = f"{type(error).__name__}: {error}"
                stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                stream.flush()
                rows.append(row)
    return rows


__all__ = ["compare_critic_modes"]
