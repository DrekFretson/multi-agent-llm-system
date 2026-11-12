"""Reproducible, callback-driven model benchmark execution."""

from __future__ import annotations

import csv
import itertools
import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from uuid import uuid4

import yaml

from src.evaluation.metrics import MetricsCollector
from src.llm.model_registry import ModelNotFoundError, ModelRegistry


class ExperimentError(Exception):
    """Base class for invalid experiment settings or execution failures."""


class ExperimentConfigError(ExperimentError, ValueError):
    """Raised when an experiment or benchmark dataset is invalid."""


@dataclass(frozen=True)
class BenchmarkTask:
    """One stable benchmark input and its evaluation metadata."""

    task_id: str
    roles: tuple[str, ...]
    request: str
    requirements: tuple[str, ...]
    expected_properties: tuple[str, ...]
    criteria: tuple[str, ...]
    reference_answer: str | None = None
    unit_tests: tuple[str, ...] = ()
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly task document passed unchanged to a runner."""
        return {"task_id": self.task_id, "roles": list(self.roles), "request": self.request,
                "requirements": list(self.requirements),
                "expected_properties": list(self.expected_properties),
                "criteria": list(self.criteria), "reference_answer": self.reference_answer,
                "unit_tests": list(self.unit_tests), "metadata": dict(self.metadata or {})}


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated settings loaded from an experiment YAML file."""

    name: str
    dataset: Path
    models_by_role: Mapping[str, tuple[str, ...]]
    generation_params: Mapping[str, Mapping[str, Any]]
    repeats: int
    seed: int
    limits: Mapping[str, Any]
    critic: Mapping[str, Any]


def load_experiment_config(path: str | Path, registry: ModelRegistry | None = None) -> ExperimentConfig:
    """Load a YAML experiment config and validate all model IDs locally."""
    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ExperimentConfigError(f"Could not load experiment config {source}.") from error
    allowed = {"name", "dataset", "models", "generation_params", "repeats", "seed", "limits", "critic"}
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise ExperimentConfigError("Experiment YAML must be a mapping with supported fields only.")
    name = raw.get("name")
    dataset = raw.get("dataset")
    if not isinstance(name, str) or not name.strip():
        raise ExperimentConfigError("Experiment 'name' must be a non-empty string.")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ExperimentConfigError("Experiment 'dataset' must be a non-empty path.")
    dataset_path = (source.parent / dataset).resolve()
    models = raw.get("models")
    if not isinstance(models, dict) or not models:
        raise ExperimentConfigError("Experiment 'models' must map roles to logical model IDs.")
    model_registry = registry or ModelRegistry()
    model_map: dict[str, tuple[str, ...]] = {}
    for role, ids in models.items():
        if not isinstance(role, str) or not role.strip() or not isinstance(ids, list) or not ids:
            raise ExperimentConfigError("Each model role must have a non-empty list of logical IDs.")
        if any(not isinstance(item, str) or not item.strip() for item in ids):
            raise ExperimentConfigError(f"Model list for role {role!r} must contain unique strings.")
        if len(set(ids)) != len(ids):
            raise ExperimentConfigError(f"Model list for role {role!r} must contain unique strings.")
        for logical_id in ids:
            try:
                model_registry.get_model(logical_id)
            except ModelNotFoundError as error:
                raise ExperimentConfigError(
                    f"Role {role!r} references an unregistered model logical ID {logical_id!r}."
                ) from error
        model_map[role] = tuple(ids)
    params = raw.get("generation_params", {})
    if not isinstance(params, dict) or any(not isinstance(role_params, dict) for role_params in params.values()):
        raise ExperimentConfigError("'generation_params' must map roles to parameter mappings.")
    if set(params) - model_map.keys():
        raise ExperimentConfigError("Generation parameters reference a role with no configured models.")
    repeats, seed = raw.get("repeats", 1), raw.get("seed", 0)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ExperimentConfigError("'repeats' must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ExperimentConfigError("'seed' must be an integer.")
    limits = raw.get("limits", {})
    critic = raw.get("critic", {})
    if not isinstance(limits, dict) or not isinstance(critic, dict):
        raise ExperimentConfigError("'limits' and 'critic' must be mappings.")
    for key, value in limits.items():
        if key in {"max_calls", "max_tokens", "max_total_retries"}:
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ExperimentConfigError(f"Limit {key!r} must be a non-negative integer or null.")
        elif key == "max_workers":
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ExperimentConfigError("Limit 'max_workers' must be a positive integer.")
        elif key in {"max_execution_time", "task_timeout", "max_latency_ms"}:
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0):
                raise ExperimentConfigError(f"Limit {key!r} must be a positive number or null.")
        elif key == "max_cost":
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0):
                raise ExperimentConfigError("Limit 'max_cost' must be non-negative or null.")
        else:
            raise ExperimentConfigError(f"Unsupported execution limit {key!r}.")
    if set(critic) - {"enabled", "max_cycles"}:
        raise ExperimentConfigError("'critic' supports only enabled and max_cycles.")
    if "enabled" in critic and not isinstance(critic["enabled"], bool):
        raise ExperimentConfigError("Critic 'enabled' must be a boolean.")
    if "max_cycles" in critic and (isinstance(critic["max_cycles"], bool)
            or not isinstance(critic["max_cycles"], int) or critic["max_cycles"] < 0):
        raise ExperimentConfigError("Critic 'max_cycles' must be a non-negative integer.")
    try:
        json.dumps(params, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ExperimentConfigError("Generation parameters must contain JSON-safe finite values.") from error
    return ExperimentConfig(name.strip(), dataset_path, model_map,
        {role: dict(value) for role, value in params.items()}, repeats, seed,
        dict(limits), dict(critic))


def load_benchmark_dataset(path: str | Path) -> list[BenchmarkTask]:
    """Load a JSON list of validated benchmark task records."""
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExperimentConfigError(f"Could not load benchmark dataset {source}.") from error
    if not isinstance(data, list) or not data:
        raise ExperimentConfigError("Benchmark dataset must be a non-empty JSON list.")
    tasks: list[BenchmarkTask] = []
    ids: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            raise ExperimentConfigError("Each benchmark entry must be an object.")
        task_id = item.get("task_id")
        role_value = item.get("roles", item.get("role"))
        roles = [role_value] if isinstance(role_value, str) else role_value
        request = item.get("request")
        if not isinstance(task_id, str) or not task_id.strip() or task_id in ids:
            raise ExperimentConfigError("Every task must have a unique, non-empty task_id.")
        if not isinstance(roles, list) or not roles or any(not isinstance(r, str) or not r.strip() for r in roles):
            raise ExperimentConfigError(f"Task {task_id!r} must have a role or non-empty roles list.")
        if len(set(roles)) != len(roles):
            raise ExperimentConfigError(f"Task {task_id!r} contains a duplicate role.")
        if not isinstance(request, str) or not request.strip():
            raise ExperimentConfigError(f"Task {task_id!r} needs a non-empty request.")
        def string_tuple(field: str, *, required: bool = False) -> tuple[str, ...]:
            value = item.get(field, [])
            if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
                raise ExperimentConfigError(f"Task {task_id!r}: {field} must be a list of strings.")
            if required and not value:
                raise ExperimentConfigError(f"Task {task_id!r}: {field} cannot be empty.")
            return tuple(value)
        reference = item.get("reference_answer")
        if reference is not None and not isinstance(reference, str):
            raise ExperimentConfigError(f"Task {task_id!r}: reference_answer must be a string or null.")
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ExperimentConfigError(f"Task {task_id!r}: metadata must be an object.")
        tasks.append(BenchmarkTask(task_id, tuple(roles), request.strip(),
            string_tuple("requirements"), string_tuple("expected_properties", required=True),
            string_tuple("criteria", required=True), reference,
            string_tuple("unit_tests"), metadata))
        ids.add(task_id)
    return tasks


class ExperimentRunner:
    """Run identical benchmark inputs across model choices and repeats.

    ``run_case`` is injected, receives ``task``, ``role``, ``models`` (actual
    provider IDs keyed by role), generation parameters, limits, seed and the
    active ``metrics_collector``. It may execute one agent or a multi-agent
    pipeline. No provider or agent implementation is assumed here.
    """

    def __init__(self, registry: ModelRegistry | None = None,
                 results_dir: str | Path = "experiments/results") -> None:
        self.registry = registry or ModelRegistry()
        self.results_dir = Path(results_dir)

    def run(self, config_path: str | Path,
            run_case: Callable[..., Any], *,
            quality_evaluator: Callable[[BenchmarkTask, Any], Mapping[str, Any]] | None = None,
            dataset_path: str | Path | None = None) -> dict[str, Any]:
        """Run configured benchmark combinations and persist each result promptly."""
        if not callable(run_case):
            raise TypeError("run_case must be callable.")
        config = load_experiment_config(config_path, self.registry)
        tasks = load_benchmark_dataset(dataset_path or config.dataset)
        unknown_roles = {role for task in tasks for role in task.roles} - config.models_by_role.keys()
        if unknown_roles:
            raise ExperimentConfigError("Dataset roles have no configured candidate models: "
                                        + ", ".join(sorted(unknown_roles)))
        experiment_id = str(uuid4())
        stamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        stem = _safe_name(config.name) + "_" + experiment_id
        self.results_dir.mkdir(parents=True, exist_ok=True)
        config_record = {"experiment_id": experiment_id, "name": config.name,
                         "created_at": stamp, "config_path": str(Path(config_path)),
                         "dataset_path": str(dataset_path or config.dataset),
                         "models_by_role": {key: list(value) for key, value in config.models_by_role.items()},
                         "generation_params": config.generation_params, "repeats": config.repeats,
                         "seed": config.seed, "limits": config.limits, "critic": config.critic}
        config_file = self.results_dir / f"{stem}_config.json"
        config_file.write_text(json.dumps(config_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        jsonl_path = self.results_dir / f"{stem}.jsonl"
        csv_path = self.results_dir / f"{stem}.csv"
        rows: list[dict[str, Any]] = []
        experiment_started = time.perf_counter()
        columns = ["timestamp", "experiment_id", "run_id", "task_id", "agent_role",
                   "model_id", "repeat", "seed", "generation_params", "execution_status",
                   "quality_metrics", "critic_status", "latency_ms", "api_calls",
                   "llm_api_calls", "external_api_calls",
                   "prompt_tokens", "completion_tokens", "total_tokens", "estimated_cost",
                   "known_prompt_tokens_subtotal", "known_completion_tokens_subtotal",
                   "known_cost_subtotal",
                   "error_type", "error_message", "unknown_cost_calls", "incomplete_cost_calls",
                   "missing_usage_calls", "provider_error_calls", "objective_quality"]
        for task in tasks:
            for model_choice in _model_combinations(task.roles, config.models_by_role):
                for repeat in range(config.repeats):
                    run_id = str(uuid4())
                    collector = MetricsCollector(experiment_id=experiment_id, run_id=run_id,
                                                 model_registry=self.registry)
                    role = task.roles[0] if len(task.roles) == 1 else "multi_agent"
                    model_configs = {r: self.registry.get_model(logical) for r, logical in model_choice.items()}
                    actual_models = {r: value.model_id for r, value in model_configs.items()}
                    params = {r: dict(config.generation_params.get(r, {})) for r in task.roles}
                    seed = config.seed + repeat
                    started = time.perf_counter()
                    result: Any = None
                    error_type = error_message = None
                    try:
                        with collector.activate(task_id=task.task_id, agent_role=role):
                            result = run_case(task=task.to_dict(), role=role,
                                models=actual_models,
                                model_configs=model_configs,
                                generation_params=params,
                                limits=dict(config.limits), critic=dict(config.critic),
                                seed=seed, metrics_collector=collector)
                        status = _execution_status(result)
                    except Exception as error:
                        status = "ERROR"
                        error_type, error_message = type(error).__name__, str(error)
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    partial_snapshot = collector.snapshot()
                    provider_error_calls = sum(1 for call in partial_snapshot["llm_calls"]
                        if not call["success"] and _is_provider_error(call.get("error_type")))
                    failed_calls = [call for call in partial_snapshot["llm_calls"] if not call["success"]]
                    if failed_calls:
                        error_type = error_type or failed_calls[0].get("error_type")
                        error_message = error_message or failed_calls[0].get("error_message")
                    if error_message is None and result is not None:
                        result_errors = getattr(result, "errors", None)
                        if result_errors is None and isinstance(result, Mapping):
                            result_errors = result.get("errors")
                        if result_errors:
                            error_message = str(result_errors[0])
                    if _is_provider_error(error_type):
                        provider_error_calls = max(1, provider_error_calls)
                    objective: Mapping[str, Any] | None = None
                    unscorable_run = (status == "ERROR" or
                                      status == "FAILED" and provider_error_calls > 0)
                    if quality_evaluator is not None and result is not None and not unscorable_run:
                        try:
                            measured = quality_evaluator(task, result)
                            if not isinstance(measured, Mapping):
                                raise TypeError("quality_evaluator must return a mapping of measured outcomes.")
                            objective = dict(measured)
                        except Exception as error:
                            objective = {"evaluation_error": f"{type(error).__name__}: {error}"}
                    execution_metrics = _result_metrics(result)
                    collector.set_quality(execution_status=status,
                        critic_status=execution_metrics.get("review_status"))
                    snapshot = collector.finish(status=status)
                    model_id_field: str | None = (actual_models[task.roles[0]]
                        if len(actual_models) == 1 else json.dumps(actual_models, sort_keys=True))
                    params_field = params.get(role, params) if role != "multi_agent" else params
                    row = {
                        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                        "experiment_id": experiment_id, "run_id": run_id, "task_id": task.task_id,
                        "agent_role": role, "model_id": model_id_field, "repeat": repeat + 1,
                        "seed": seed, "generation_params": params_field, "execution_status": status,
                        "quality_metrics": objective, "critic_status": execution_metrics.get("review_status"),
                        "latency_ms": round(elapsed_ms, 3), "api_calls": snapshot["api_calls"],
                        "llm_api_calls": snapshot["llm_api_calls"],
                        "external_api_calls": snapshot["external_api_calls"],
                        "prompt_tokens": snapshot["prompt_tokens"], "completion_tokens": snapshot["completion_tokens"],
                        "total_tokens": snapshot["total_tokens"], "estimated_cost": snapshot["estimated_cost"],
                        "known_prompt_tokens_subtotal": snapshot["known_prompt_tokens_subtotal"],
                        "known_completion_tokens_subtotal": snapshot["known_completion_tokens_subtotal"],
                        "known_cost_subtotal": snapshot["known_cost_subtotal"],
                        "error_type": error_type, "error_message": error_message,
                        "unknown_cost_calls": snapshot["unknown_cost_calls"],
                        "incomplete_cost_calls": snapshot["incomplete_cost_calls"],
                        "missing_usage_calls": snapshot["missing_usage_calls"],
                        "provider_error_calls": provider_error_calls,
                        "objective_quality": objective,
                    }
                    rows.append(row)
                    _append_jsonl(jsonl_path, row)
                    _append_csv(csv_path, columns, row)
        summary = ExperimentAnalyzer().analyze(rows)
        summary.update({"experiment_id": experiment_id, "name": config.name,
                        "created_at": stamp, "records": len(rows),
                        "duration_ms": round((time.perf_counter() - experiment_started) * 1000, 3),
                        "api_calls": sum(row["api_calls"] for row in rows),
                        "llm_api_calls": sum(row["llm_api_calls"] for row in rows),
                        "external_api_calls": sum(row["external_api_calls"] for row in rows),
                        "prompt_tokens": _complete_sum(rows, "prompt_tokens"),
                        "completion_tokens": _complete_sum(rows, "completion_tokens"),
                        "total_tokens": _complete_sum(rows, "total_tokens"),
                        "estimated_cost": _complete_sum(rows, "estimated_cost"),
                        "known_cost_subtotal": (sum(row["known_cost_subtotal"] for row in rows
                                                     if isinstance(row["known_cost_subtotal"], (int, float)))
                            if any(isinstance(row["known_cost_subtotal"], (int, float)) for row in rows) else None),
                        "provider_error_calls": sum(row["provider_error_calls"] for row in rows)})
        _write_json(self.results_dir / f"{stem}_results.json", rows)
        _write_json(self.results_dir / f"{stem}_summary.json", summary)
        _write_summary_csv(self.results_dir / f"{stem}_summary.csv", summary["groups"])
        return {"experiment_id": experiment_id, "config": config_record,
                "records": rows, "summary": summary,
                "files": {"config": str(config_file), "jsonl": str(jsonl_path),
                          "csv": str(csv_path),
                          "json": str(self.results_dir / f"{stem}_results.json"),
                          "summary": str(self.results_dir / f"{stem}_summary.json")}}


class ExperimentAnalyzer:
    """Aggregate model outcomes without inventing a scalar quality score."""

    def analyze(self, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        """Compare observed success, objective outcomes, latency and known costs."""
        grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
        for row in rows:
            grouped.setdefault((str(row.get("agent_role")), str(row.get("model_id"))), []).append(row)
        summaries: list[dict[str, Any]] = []
        for (role, model_id), values in sorted(grouped.items()):
            statuses = [row.get("execution_status") for row in values]
            quality_rows = [row for row in values if not (
                row.get("execution_status") == "ERROR" or
                row.get("execution_status") == "FAILED" and row.get("provider_error_calls", 0) > 0)]
            quality_statuses = [row.get("execution_status") for row in quality_rows]
            successful = sum(status in {"COMPLETED", "SUCCESS", "APPROVED"} for status in quality_statuses)
            latencies = [row["latency_ms"] for row in values if isinstance(row.get("latency_ms"), (int, float))]
            costs = [row["estimated_cost"] for row in values if isinstance(row.get("estimated_cost"), (int, float))]
            tokens = [row["total_tokens"] for row in values if isinstance(row.get("total_tokens"), int)]
            objective = [row.get("objective_quality") for row in values
                         if isinstance(row.get("objective_quality"), Mapping)
                         and row.get("objective_quality")]
            critic_values = [row.get("critic_status") for row in values if row.get("critic_status")]
            summaries.append({
                "agent_role": role, "model_id": model_id, "runs": len(values),
                "quality_sample_runs": len(quality_rows),
                "successful_runs": successful,
                "success_rate": successful / len(quality_rows) if quality_rows else None,
                "error_runs": sum(status in {"ERROR", "FAILED", "PARTIAL"} for status in statuses),
                "provider_error_runs": sum(bool(row.get("provider_error_calls")) for row in values),
                "critic_acceptance_rate": (sum(value in {"APPROVED", "NOT_REQUIRED"} for value in critic_values)
                                           / len(critic_values) if critic_values else None),
                "status_stability": max((quality_statuses.count(status) / len(quality_statuses)
                                          for status in set(quality_statuses)), default=None),
                "objective_quality_runs": len(objective),
                "objective_quality": _average_objective_mappings(objective),
                "objective_quality_score_stdev": _objective_stdev(objective, "quality_score"),
                "latency_mean_ms": _mean(latencies), "latency_median_ms": _median(latencies),
                "latency_p95_ms": _p95(latencies) if len(latencies) >= 20 else None,
                "known_usage_runs": sum(row.get("prompt_tokens") is not None
                                         and row.get("completion_tokens") is not None for row in values),
                "known_cost_runs": sum(row.get("estimated_cost") is not None for row in values),
                "average_total_tokens": _mean(tokens), "average_api_calls": _mean(
                    [row["api_calls"] for row in values if isinstance(row.get("api_calls"), int)]),
                "average_known_cost": _mean(costs), "total_known_cost": sum(costs) if costs else None,
                "unknown_cost_runs": sum(bool(row.get("unknown_cost_calls")) for row in values),
                "status_counts": {str(status): statuses.count(status) for status in sorted(set(statuses), key=str)},
            })
        return {"groups": summaries, "recommendations": _recommend(summaries)}


def _recommend(groups: list[dict[str, Any]]) -> dict[str, Any]:
    recommendations: dict[str, Any] = {}
    for role in sorted({row["agent_role"] for row in groups}):
        eligible = [row for row in groups if row["agent_role"] == role and row["runs"] >= 2]
        if not eligible:
            recommendations[role] = {"status": "insufficient_data", "runs_required": 2}
            continue
        objective = [row for row in eligible if row["objective_quality_runs"] >= 2
                     and row["objective_quality"]]
        if not objective:
            recommendations[role] = {
                "status": "insufficient_objective_quality_data",
                "fastest_observed": _best(eligible, "latency_mean_ms", min),
                "lowest_known_cost": _best(eligible, "average_known_cost", min),
                "highest_success_rate": _best(eligible, "success_rate", max),
                "recommended": None,
            }
            continue
        # A quality field is rankable only when the evaluator provides a numeric
        # value explicitly named quality_score; no derived quality is fabricated.
        quality_key = "quality_score"
        ranked = [row for row in objective
                  if isinstance(row["objective_quality"].get(quality_key), (int, float))]
        if not ranked:
            quality_key = "tests_passed"
            ranked = [row for row in objective
                      if isinstance(row["objective_quality"].get(quality_key), (int, float))]
        if not ranked:
            recommendations[role] = {"status": "no_comparable_quality_score",
                "fastest_observed": _best(eligible, "latency_mean_ms", min),
                "lowest_known_cost": _best(eligible, "average_known_cost", min),
                "highest_success_rate": _best(eligible, "success_rate", max), "recommended": None}
            continue
        winner = max(ranked, key=lambda row: (
            row["objective_quality"][quality_key], row["success_rate"] or 0,
            -(row["latency_mean_ms"] or float("inf")), -(row["average_known_cost"] or float("inf"))))
        recommendations[role] = {"status": "ranked_by_objective_quality_then_success_latency_cost",
            "quality_metric": quality_key,
            "best_quality": _best(ranked, f"objective_quality.{quality_key}", max),
            "fastest_observed": _best(eligible, "latency_mean_ms", min),
            "lowest_known_cost": _best(eligible, "average_known_cost", min),
            "highest_success_rate": _best(eligible, "success_rate", max),
            "recommended": winner["model_id"]}
    return recommendations


def _best(rows: list[dict[str, Any]], key: str, chooser: Callable[..., Any]) -> str | None:
    def value(row: Mapping[str, Any]) -> Any:
        if key.startswith("objective_quality."):
            return (row.get("objective_quality") or {}).get(key.split(".", 1)[1])
        return row.get(key)
    available = [row for row in rows if isinstance(value(row), (int, float))]
    return chooser(available, key=value)["model_id"] if available else None


def _model_combinations(roles: tuple[str, ...], candidates: Mapping[str, tuple[str, ...]]) -> Iterable[dict[str, str]]:
    for choice in itertools.product(*(candidates[role] for role in roles)):
        yield dict(zip(roles, choice))


def _execution_status(result: Any) -> str:
    status = getattr(result, "status", None)
    if status is None and isinstance(result, Mapping):
        status = result.get("status")
        if status is None and "success" in result:
            status = "COMPLETED" if result["success"] else "FAILED"
    if status is None and hasattr(result, "success"):
        status = "COMPLETED" if result.success else "FAILED"
    return str(status or "COMPLETED").upper()


def _is_provider_error(error_type: Any) -> bool:
    return error_type in {"AuthenticationError", "RateLimitError", "LLMTimeoutError",
                          "NetworkError", "LLMAPIError", "APITimeoutError",
                          "APIConnectionError", "APIError"}


def _result_metrics(result: Any) -> Mapping[str, Any]:
    value = getattr(result, "execution_metrics", None)
    return value if isinstance(value, Mapping) else {}


def _mean(values: list[int | float]) -> float | None:
    return sum(values) / len(values) if values else None


def _complete_sum(rows: list[Mapping[str, Any]], key: str) -> int | float | None:
    values = [row.get(key) for row in rows]
    if not values or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                         for value in values):
        return None
    return sum(values)


def _median(values: list[int | float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def _p95(values: list[int | float]) -> float | None:
    return sorted(values)[max(0, int(0.95 * len(values) + 0.999999) - 1)] if values else None


def _average_objective_mappings(values: list[Mapping[str, Any]]) -> dict[str, float] | None:
    keys = set.intersection(*(set(value) for value in values)) if values else set()
    numeric = {key: _mean([float(value[key]) if isinstance(value.get(key), bool) else value[key]
                           for value in values
                           if isinstance(value.get(key), (int, float))])
               for key in keys}
    return {key: value for key, value in numeric.items() if value is not None} or None


def _objective_stdev(values: list[Mapping[str, Any]], key: str) -> float | None:
    scores = [float(value[key]) for value in values
              if isinstance(value.get(key), (int, float)) and not isinstance(value.get(key), bool)]
    return statistics.pstdev(scores) if len(scores) >= 2 else None


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=str) + "\n")
        stream.flush()


def _append_csv(path: Path, columns: list[str], row: Mapping[str, Any]) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: json.dumps(value, ensure_ascii=False, default=str)
                         if isinstance(value, (dict, list)) else value for key, value in row.items()})
        stream.flush()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def _write_summary_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    columns = list(rows[0]) if rows else ["agent_role", "model_id"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, default=str)
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in value.strip())
    return safe[:60] or "experiment"


__all__ = ["BenchmarkTask", "ExperimentAnalyzer", "ExperimentConfig", "ExperimentConfigError",
           "ExperimentError", "ExperimentRunner", "load_benchmark_dataset", "load_experiment_config"]
