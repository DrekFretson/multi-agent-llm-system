"""Shared persistence and metrics plumbing for isolated lab experiment runs."""

from __future__ import annotations

import csv
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import yaml

from src.evaluation.metrics import MetricsCollector


_FILE_LOCK = threading.Lock()


def load_lab_inputs(config_path: str | Path, dataset_path: str | Path | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read one lab's YAML settings and JSON scenarios, validating basic shape."""
    config_file = Path(config_path)
    try:
        config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"Could not load lab configuration {config_file}.") from error
    if not isinstance(config, dict):
        raise ValueError("Lab configuration must be a YAML mapping.")
    scenario_file = Path(dataset_path) if dataset_path is not None else (config_file.parent / config.get("dataset", ""))
    try:
        scenarios = json.loads(scenario_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load lab scenarios {scenario_file}.") from error
    if not isinstance(scenarios, list) or not scenarios or any(not isinstance(row, dict) for row in scenarios):
        raise ValueError("Lab scenario dataset must be a non-empty list of objects.")
    identifiers = [row.get("scenario_id") for row in scenarios]
    if any(not isinstance(value, str) or not value.strip() for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Every scenario must have a unique non-empty scenario_id.")
    return config, scenarios


def new_lab_experiment(lab_id: str, config: Mapping[str, Any], report_root: str | Path | None = None) -> tuple[str, Path]:
    """Create a unique result directory beneath this lab's report tree."""
    experiment_id = str(uuid4())
    root = Path(report_root) if report_root is not None else Path(__file__).resolve().parents[2] / "reports"
    result_dir = root / lab_id / "results" / experiment_id
    result_dir.mkdir(parents=True, exist_ok=False)
    snapshot = {"experiment_id": experiment_id, "lab_id": lab_id,
        "created_at": _utc_now(), "configuration": dict(config)}
    _write_json(result_dir / "config.json", snapshot)
    return experiment_id, result_dir


def make_metrics(experiment_id: str) -> MetricsCollector:
    """Create a run collector linked to one isolated experiment ID."""
    return MetricsCollector(experiment_id=experiment_id)


class MetricsIncidentLogger:
    """Adapter for component event loggers that records security incidents."""

    def __init__(self, metrics: MetricsCollector) -> None:
        self.metrics = metrics

    def log_event(self, event_type: str, **fields: Any) -> None:
        if event_type == "security_incident":
            self.metrics.record_security_incident(
                incident_type=str(fields.get("incident_type", "unknown")),
                actor_id=fields.get("actor_id"), task_id=fields.get("task_id"),
                reason=str(fields.get("reason", "")),
                blocked=bool(fields.get("blocked", False)))


def save_lab_run(result_dir: Path, *, scenario_id: str, variant: str,
                 repeat: int, run_id: str, payload: Mapping[str, Any],
                 metrics: MetricsCollector) -> dict[str, str]:
    """Persist one raw result and component metrics without replacing prior runs."""
    safe_scenario = _safe_name(scenario_id)
    safe_variant = _safe_name(variant)
    stem = f"{safe_scenario}_{safe_variant}_r{repeat}_{run_id}"
    result_file = result_dir / f"{stem}.json"
    _write_json(result_file, payload)
    metric_paths = metrics.export(result_dir, prefix=stem)
    return {"result": str(result_file), **{key: str(value) for key, value in metric_paths.items()}}


def write_comparison(result_dir: Path, experiment_id: str,
                     rows: list[Mapping[str, Any]]) -> Path:
    """Write one experiment-local wide CSV, keeping other labs/runs untouched."""
    path = result_dir / "comparison.csv"
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    if "experiment_id" not in columns:
        columns.insert(0, "experiment_id")
    with _FILE_LOCK, path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, default=str)
                             if isinstance(value, (dict, list)) else value
                             for key, value in {"experiment_id": experiment_id, **row}.items()})
    return path


def update_lab_report(lab_id: str, *, title: str, objective: str,
                      methodology_path: str, result_dir: Path,
                      summary: str, report_root: str | Path | None = None) -> Path:
    """Update only this lab's report with results from the just-finished run."""
    root = Path(report_root) if report_root is not None else Path(__file__).resolve().parents[2] / "reports"
    report_path = root / lab_id / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    start_marker, end_marker = "<!-- AUTO_RESULTS_START -->", "<!-- AUTO_RESULTS_END -->"
    result_section = (
        f"{start_marker}\n## Последние фактические результаты\n\n{summary}\n\n"
        f"Исходные файлы запуска: [`results/{result_dir.name}/`](results/{result_dir.name}/).\n"
        f"{end_marker}"
    )
    with _FILE_LOCK:
        if report_path.exists():
            current = report_path.read_text(encoding="utf-8")
            start = current.find(start_marker)
            end = current.find(end_marker)
            if start >= 0 and end >= start:
                end += len(end_marker)
                content = current[:start] + result_section + current[end:]
            else:
                content = current.rstrip() + "\n\n" + result_section + "\n"
        else:
            content = (
                f"# {title}\n\n## Цель\n\n{objective}\n\n"
                f"## Методика\n\nПодробные правила и ограничения описаны в [`methodology.md`]({methodology_path}).\n\n"
                f"## Ограничения\n\nИзмерения получены на контролируемых локальных сценариях. Внешние API, "
                "произвольный код агентов и реальные вредоносные действия не использовались.\n\n"
                f"{result_section}\n"
            )
        report_path.write_text(content, encoding="utf-8")
    return report_path


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False, default=str)
        stream.write("\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_name(value: str) -> str:
    text = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    return text[:70] or "scenario"


__all__ = ["MetricsIncidentLogger", "load_lab_inputs", "make_metrics", "new_lab_experiment", "save_lab_run",
           "update_lab_report", "write_comparison"]
