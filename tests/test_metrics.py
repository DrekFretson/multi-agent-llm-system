import csv
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

from src.evaluation import MetricsCollector
from src.llm.client import LLMClient
from src.tools import SearchTool


def test_collector_aggregates_calls_and_keeps_unknowns_explicit():
    metrics = MetricsCollector()
    metrics.record_llm_call(model_id="nvidia/nemotron-3-ultra-550b-a55b:free",
        agent_id="agent-1", task_id="task-1", agent_role="math", duration_ms=12.5,
        prompt_tokens=10, completion_tokens=5, total_tokens=15, finish_reason="stop", success=True)
    metrics.record_llm_call(model_id="unregistered/model", agent_id="agent-2",
        task_id="task-2", agent_role="coding", duration_ms=3,
        prompt_tokens=4, completion_tokens=2, total_tokens=None,
        finish_reason=None, success=True)
    snapshot = metrics.snapshot()
    assert snapshot["api_calls"] == 2
    assert snapshot["prompt_tokens"] == 14
    assert snapshot["completion_tokens"] == 7
    assert snapshot["total_tokens"] is None
    assert snapshot["known_prompt_tokens_subtotal"] == 14
    assert snapshot["estimated_cost"] is None
    assert snapshot["known_cost_subtotal"] == 0.0
    assert snapshot["unknown_cost_calls"] == 1
    assert snapshot["llm_calls"][1]["cost_status"] == "unknown_price"


def test_missing_usage_is_not_reported_as_zero():
    metrics = MetricsCollector()
    call = metrics.record_llm_call(model_id="cohere/north-mini-code:free", agent_id=None,
        task_id=None, agent_role=None, duration_ms=1, prompt_tokens=None,
        completion_tokens=None, total_tokens=None, finish_reason="length", success=True)
    assert call["usage_source"] == "missing"
    assert call["prompt_tokens"] is None
    assert call["completion_tokens"] is None
    assert call["estimated_cost"] is None
    assert metrics.snapshot()["prompt_tokens"] is None


def test_parallel_records_are_not_lost():
    metrics = MetricsCollector()
    def add(index):
        metrics.record_task(task_id=str(index), agent_role="coding", status="COMPLETED",
                            duration_ms=index)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(add, range(100)))
    snapshot = metrics.snapshot()
    assert snapshot["task_count"] == 100
    assert {row["task_id"] for row in snapshot["tasks"]} == {str(i) for i in range(100)}


def test_export_writes_json_and_pandas_friendly_csv(tmp_path):
    metrics = MetricsCollector(experiment_id="experiment-1", run_id="run-1")
    metrics.record_task(task_id="task-1", agent_role="research", status="FAILED",
                        duration_ms=8, error="provider unavailable")
    metrics.record_llm_call(model_id="unknown", agent_id="research-1", task_id="task-1",
        agent_role="research", duration_ms=2, prompt_tokens=None, completion_tokens=None,
        total_tokens=None, finish_reason=None, success=False, error_type="NetworkError",
        error_message="network error")
    paths = metrics.export(tmp_path)
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert data["run_id"] == "run-1"
    assert {item["error_type"] for item in data["errors"]} == {"NetworkError", "TaskExecutionError"}
    with paths["llm_calls"].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["experiment_id"] == "experiment-1"
    assert rows[0]["cost_status"] == "unknown_price"


def test_llm_client_records_actual_call_without_prompt_or_key_in_metrics():
    client = object.__new__(LLMClient)
    client._api_key = "secret-token"
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=Mock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
            model="nvidia/nemotron-3-ultra-550b-a55b:free",
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1, total_tokens=3),
        )))))
    client._logger = Mock()
    client._request_guard = None
    client._response_observer = None
    metrics = MetricsCollector(experiment_id="exp", run_id="run")
    with metrics.activate(task_id="task", agent_role="coding"):
        response = client.generate("nvidia/nemotron-3-ultra-550b-a55b:free",
            [{"role": "user", "content": "secret-token prompt"}], agent_id="coding-1")
    snapshot_text = json.dumps(metrics.snapshot())
    assert response.content == "ok"
    assert "secret-token" not in snapshot_text
    assert "secret-token prompt" not in snapshot_text
    call = metrics.snapshot()["llm_calls"][0]
    assert call["run_id"] == "run"
    assert call["task_id"] == "task"
    assert call["agent_role"] == "coding"
    assert call["agent_id"] == "coding-1"
    assert call["request_id"]
    assert client._logger.log_event.call_args.kwargs["run_id"] == "run"
    assert call["model_id"] == "nvidia/nemotron-3-ultra-550b-a55b:free"


def test_llm_client_records_provider_failure_and_sanitizes_secret():
    client = object.__new__(LLMClient)
    client._api_key = "secret-token"
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=Mock(side_effect=RuntimeError("provider rejected secret-token")))))
    client._logger = Mock()
    client._request_guard = None
    client._response_observer = None
    metrics = MetricsCollector()
    with metrics.activate(task_id="task-fail", agent_role="research"):
        try:
            client.generate("unknown/model", [{"role": "user", "content": "question"}], agent_id="r1")
        except RuntimeError as error:
            assert str(error) == "provider rejected secret-token"
        else:
            raise AssertionError("expected the original provider exception")
    snapshot = metrics.snapshot()
    assert snapshot["api_calls"] == 1
    assert snapshot["llm_calls"][0]["success"] is False
    assert snapshot["llm_calls"][0]["error_type"] == "RuntimeError"
    assert "secret-token" not in json.dumps(snapshot)
    assert snapshot["errors"][0]["task_id"] == "task-fail"


def test_tavily_requests_are_counted_without_logging_the_query_in_metrics():
    provider = Mock()
    provider.search.return_value = {"results": [{
        "title": "Reference", "url": "https://example.org/page",
        "content": "A short excerpt", "score": 0.8,
    }]}
    tool = SearchTool(client=provider, logger=Mock(), agent_id="research-1")
    metrics = MetricsCollector()
    with metrics.activate(task_id="research-task", agent_role="research"):
        result = tool.search("private search query")
    snapshot = metrics.snapshot()
    assert len(result) == 1
    assert snapshot["api_calls"] == 1
    assert snapshot["llm_api_calls"] == 0
    assert snapshot["external_api_calls"] == 1
    call = snapshot["external_calls"][0]
    assert call["service"] == "tavily"
    assert call["task_id"] == "research-task"
    assert "private search query" not in json.dumps(snapshot)


def test_failed_tavily_request_is_counted_as_external_error():
    provider = Mock()
    provider.search.side_effect = RuntimeError("network down")
    tool = SearchTool(client=provider, logger=Mock(), agent_id="research-1")
    metrics = MetricsCollector()
    with metrics.activate(task_id="research-task", agent_role="research"):
        try:
            tool.search("query")
        except Exception:
            pass
        else:
            raise AssertionError("expected the provider error")
    snapshot = metrics.snapshot()
    assert snapshot["api_calls"] == 1
    assert snapshot["external_calls"][0]["success"] is False
    assert snapshot["external_calls"][0]["error_type"] == "SearchAPIError"
    assert snapshot["error_count"] >= 1

