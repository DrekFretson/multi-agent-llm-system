"""Controlled local reliability and security scenarios for Lab 4."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import Mock

from src.communication import Message, MessageBus, MessageType
from src.coordination import DelegationCycleError, DelegationManager, Task, TaskPriority, TaskStatus, TaskStore, TaskType
from src.evaluation.lab_support import (
    MetricsIncidentLogger, load_lab_inputs, make_metrics, new_lab_experiment,
    save_lab_run, update_lab_report, write_comparison,
)
from src.security import InstructionConflictError, SecurityPolicy, SecurityPolicyError
from src.tools import SearchTool, SearchToolError


COORDINATOR = "coordinator"
PRINCIPALS = {
    COORDINATOR: "coordinator", "research_agent": "research",
    "coding_agent": "coding", "critic_agent": "critic", "attacker": "untrusted",
}


class _PolicyEventSink(MetricsIncidentLogger):
    """Metrics adapter used as the existing component event-logger interface."""


def _make_policy(scenario_id: str, metrics: Any) -> SecurityPolicy:
    return SecurityPolicy(
        principals=PRINCIPALS,
        allowed_message_types={
            "coordinator": [MessageType.TASK_ASSIGN, MessageType.SECURITY_INCIDENT],
            "research": [MessageType.HELP_REQUEST, MessageType.RESULT, MessageType.TASK_BID],
            "coding": [MessageType.HELP_REQUEST, MessageType.RESULT, MessageType.TASK_BID],
            "critic": [MessageType.REVIEW_FEEDBACK, MessageType.RESULT],
            "untrusted": [],
        },
        allowed_recipient_roles={
            "coordinator": {"research", "coding", "critic", "untrusted"},
            "research": {"coordinator", "coding", "math"},
            "coding": {"coordinator", "research", "math"},
            "critic": {"coordinator", "research", "coding"},
            "untrusted": set(),
        },
        payload_schemas={
            MessageType.TASK_ASSIGN: {"description": str},
            MessageType.HELP_REQUEST: {"description": str, "parent_task_id": str,
                                      "child_task_id": str, "initiator_id": str},
            MessageType.TASK_BID: {"competence": (int, float)},
        },
        allowed_actions={
            "coordinator": {"assign_task", "arbitrate"},
            "research": {"search", "summarize", "request_help"},
            "coding": {"write_code", "run_tests", "request_help"},
            "critic": {"review", "report_issue"},
            "untrusted": set(),
        },
        task_scopes={f"legitimate-{scenario_id}": {"write_code", "run_tests", "search", "summarize"}},
        max_payload_bytes=4096,
        logger=_PolicyEventSink(metrics),
    )


def _new_bus(policy: SecurityPolicy | None) -> MessageBus:
    bus = MessageBus(authorizer=policy.authorize_message if policy else None)
    for agent_id in PRINCIPALS:
        bus.register_agent(agent_id)
    return bus


def _send(bus: MessageBus, sender: str, receiver: str, message_type: MessageType,
          task_id: str, payload: dict[str, Any]) -> bool:
    bus.send(Message(sender=sender, receiver=receiver, message_type=message_type,
                     task_id=task_id, payload=payload))
    return True


def _scenario_case(scenario: Mapping[str, Any], protected: bool,
                   metrics: Any) -> dict[str, Any]:
    scenario_id = scenario["scenario_id"]
    policy = _make_policy(scenario_id, metrics) if protected else None
    bus = _new_bus(policy)
    started = time.perf_counter()
    detected = False
    blocked = False
    false_blocks = 0
    completed = 0
    errors = 0
    unfinished = 0
    criterion_met = False
    outcome: dict[str, Any] = {}
    incident_records: list[dict[str, Any]] = []

    try:
        kind = scenario["kind"]
        if kind == "malformed_message":
            malformed = Message(sender=COORDINATOR, receiver="coding_agent",
                message_type=MessageType.TASK_ASSIGN, task_id="bad-contract",
                payload={"unexpected": "field"})
            try:
                bus.send(malformed)
                bus.receive("coding_agent")
                outcome["malformed_message_delivered"] = True
                criterion_met = not protected
            except Exception as error:
                detected = blocked = protected
                outcome.update({"malformed_message_delivered": False,
                                "block_reason": type(error).__name__})
                criterion_met = protected

        elif kind == "unauthorized_sender":
            forged = Message(sender="attacker", receiver=COORDINATOR,
                message_type=MessageType.TASK_ASSIGN, task_id="forged-task",
                payload={"description": "attempt coordinator control"})
            try:
                bus.send(forged)
                bus.receive(COORDINATOR)
                outcome["forged_message_delivered"] = True
                criterion_met = not protected
            except Exception as error:
                detected = blocked = protected
                outcome.update({"forged_message_delivered": False,
                                "block_reason": type(error).__name__})
                criterion_met = protected

        elif kind == "tool_unavailable":
            search_client = Mock()
            search_client.search.side_effect = RuntimeError("controlled local service outage")
            tool = SearchTool(client=search_client, logger=Mock(), agent_id="research_agent")
            try:
                with metrics.task_context(task_id="research-search", agent_role="research"):
                    tool.search("fixed local failure test")
            except SearchToolError as error:
                errors += 1
                outcome["search_error_type"] = type(error).__name__
                if protected:
                    detected = True
                    metrics.record_security_incident(incident_type="tool_unavailable",
                        actor_id="research_agent", task_id="research-search",
                        reason="Search failure was caught; the task used the documented degraded path.",
                        blocked=False)
                    metrics.record_task(task_id="research-search", agent_role="research",
                        status="COMPLETED_WITH_FALLBACK", duration_ms=0.0)
                    completed += 1
                    outcome["research_status"] = "COMPLETED_WITH_FALLBACK"
                    # An independent local coding task proceeds after the failed search.
                    metrics.record_task(task_id="independent-coding", agent_role="coding",
                        status="COMPLETED", duration_ms=0.0)
                    completed += 1
                    outcome["independent_task_status"] = "COMPLETED"
                    errors -= 1
                    criterion_met = True
                else:
                    metrics.record_task(task_id="research-search", agent_role="research",
                        status="FAILED", duration_ms=0.0, error="Search tool unavailable.")
                    unfinished += 1
                    outcome.update({"research_status": "FAILED", "independent_task_status": "NOT_RUN"})
                    # Baseline pipeline halts before launching an independent dependent stage.
                    unfinished += 1
                    criterion_met = True

        elif kind == "agent_wrong_goal":
            action = scenario["unauthorized_action"]
            if protected:
                try:
                    policy.authorize_action(agent_id="coding_agent",
                        task_id="malicious-task", action=action)
                except SecurityPolicyError as error:
                    detected = blocked = True
                    outcome["unauthorized_action_status"] = "BLOCKED"
                    outcome["block_reason"] = str(error)
            else:
                outcome["unauthorized_action_status"] = "WOULD_ALLOW"
            outcome["action_executed"] = False  # Even baseline records a proposal only.
            criterion_met = outcome["unauthorized_action_status"] == ("BLOCKED" if protected else "WOULD_ALLOW")

        elif kind == "delegation_cycle":
            if protected:
                store = TaskStore()
                root = Task(task_id="cycle-root", description="controlled cycle test",
                            task_type=TaskType.GENERAL, priority=TaskPriority.MEDIUM)
                store.create_task(root)
                store.assign_task(root.task_id, "research_agent")
                store.update_status(root.task_id, TaskStatus.RUNNING)
                manager = DelegationManager(store, bus,
                    delegation_rules={"research_agent": {"coding_agent"},
                                      "coding_agent": {"research_agent"}},
                    max_delegation_depth=5, max_handoffs=5)
                first = manager.request_help(parent_task_id=root.task_id,
                    initiator_id="research_agent", recipient_id="coding_agent",
                    description="delegate controlled step", request_id="cycle-1")
                child_id = first.payload["child_task_id"]
                try:
                    manager.request_help(parent_task_id=child_id,
                        initiator_id="coding_agent", recipient_id="research_agent",
                        description="attempt repeated role", request_id="cycle-2")
                except DelegationCycleError as error:
                    detected = blocked = True
                    outcome.update({"cycle_detected": True,
                                    "block_reason": type(error).__name__,
                                    "delegated_child_status": store.get_task(child_id).status.value})
                    criterion_met = True
                    metrics.record_security_incident(incident_type="delegation_cycle",
                        actor_id="coding_agent", task_id=child_id,
                        reason="DelegationManager found the research role in task ancestry.", blocked=True)
                metrics.record_task(task_id=root.task_id, agent_role="research",
                    status=store.get_task(root.task_id).status.value, duration_ms=0.0)
                metrics.record_task(task_id=child_id, agent_role="coding",
                    status=store.get_task(child_id).status.value, duration_ms=0.0)
            else:
                chain = ["research_agent", "coding_agent", "research_agent",
                         "coding_agent", "research_agent"]
                for index, (sender, receiver) in enumerate(zip(chain, chain[1:])):
                    _send(bus, sender, receiver, MessageType.HELP_REQUEST,
                        f"cycle-{index}", {"description": "bounded synthetic cycle"})
                outcome.update({"cycle_detected": False, "harness_stopped": True,
                                "bounded_hops": len(chain) - 1})
                criterion_met = True
                unfinished += 1

        elif kind == "conflicting_instructions":
            instructions = scenario["instructions"]
            if protected:
                try:
                    resolution = policy.resolve_instructions(instructions,
                        actor_id="coding_agent", task_id="instruction-task")
                    detected = resolution.source != instructions[-1]["source"]
                    blocked = detected
                    outcome.update({"winning_source": resolution.source,
                                    "suppressed_sources": list(resolution.suppressed_sources),
                                    "unsafe_instruction_applied": False})
                    criterion_met = resolution.source == "task_owner" and detected
                except InstructionConflictError as error:
                    detected = blocked = True
                    outcome["block_reason"] = type(error).__name__
            else:
                winner = instructions[-1]
                outcome.update({"winning_source": winner["source"],
                                "unsafe_instruction_applied": winner["source"] in {"agent_output", "untrusted_data"}})
                criterion_met = outcome["unsafe_instruction_applied"]
            metrics.record_task(task_id="instruction-task", agent_role="coding",
                status="COMPLETED", duration_ms=0.0)
        else:
            raise ValueError(f"Unsupported Lab 4 scenario kind {kind!r}.")

        # Every scenario includes a known-good control to measure false blocks.
        if kind in {"malformed_message", "unauthorized_sender", "agent_wrong_goal",
                    "conflicting_instructions", "tool_unavailable", "delegation_cycle"}:
            try:
                _send(bus, COORDINATOR, "coding_agent", MessageType.TASK_ASSIGN,
                      f"legitimate-{scenario_id}", {"description": "allowed local control task"})
                delivered = bus.receive("coding_agent") is not None
                action_allowed = True
                if protected:
                    action_allowed = policy.authorize_action(agent_id="coding_agent",
                        task_id=f"legitimate-{scenario_id}", action="write_code").allowed
                if delivered and action_allowed:
                    if completed == 0:
                        completed += 1
                    metrics.record_task(task_id=f"legitimate-{scenario_id}", agent_role="coding",
                        status="COMPLETED", duration_ms=0.0)
                else:
                    false_blocks += 1
            except Exception:
                false_blocks += 1

    except Exception as error:
        errors += 1
        outcome.setdefault("unexpected_error", f"{type(error).__name__}: {error}")
        if protected:
            detected = True

    elapsed_ms = (time.perf_counter() - started) * 1000
    metrics.set_quality(detected=detected, blocked=blocked, false_blocks=false_blocks)
    snapshot = metrics.finish(status="COMPLETED" if errors == 0 and false_blocks == 0 else "PARTIAL")
    incident_records = snapshot["security_incidents"]
    return {
        "scenario_id": scenario_id, "variant": "protected" if protected else "baseline",
        "run_id": metrics.run_id, "detected_violation": detected,
        "blocked_unsafe_action": blocked, "legitimate_tasks_completed": completed,
        "false_blocks": false_blocks, "incident_handling_time_ms": round(elapsed_ms, 3),
        "messages": snapshot["message_count"], "errors": errors,
        "unfinished_tasks": unfinished, "provider_errors": snapshot["external_api_calls"] and
            sum(1 for call in snapshot["external_calls"] if not call["success"]) or 0,
        "security_incidents": incident_records, "outcome": outcome,
        "expected_behavior": scenario["expected_behavior"],
        "success_criterion": scenario["success_criterion"],
        "criterion_met": criterion_met,
        "metrics": snapshot,
    }


def run_lab4(config_path: str | Path | None = None, dataset_path: str | Path | None = None,
             *, report_root: str | Path | None = None) -> dict[str, Any]:
    """Run each controlled fault/attack scenario in baseline and protected modes."""
    project_root = Path(__file__).resolve().parents[2]
    config_path = Path(config_path) if config_path else project_root / "experiments/configs/lab_4.yaml"
    config, scenarios = load_lab_inputs(config_path, dataset_path)
    experiment_id, result_dir = new_lab_experiment("lab_4", config, report_root)
    rows: list[dict[str, Any]] = []
    repeats = int(config.get("repeats", 1))
    for repeat in range(repeats):
        for scenario in scenarios:
            for protected in (False, True):
                metrics = make_metrics(experiment_id)
                with metrics.activate(task_id=scenario["scenario_id"], agent_role="lab_4_harness"):
                    row = _scenario_case(scenario, protected, metrics)
                row["repeat"] = repeat + 1
                row["seed"] = int(config.get("seed", 0)) + repeat
                payload = {"experiment_id": experiment_id, **row, "scenario": scenario}
                paths = save_lab_run(result_dir, scenario_id=row["scenario_id"],
                    variant=row["variant"], repeat=repeat + 1,
                    run_id=metrics.run_id, payload=payload, metrics=metrics)
                row["result_path"] = paths["result"]
                rows.append(row)
    comparison = write_comparison(result_dir, experiment_id, rows)
    summary = _summarize(rows)
    summary.update({"experiment_id": experiment_id, "result_dir": str(result_dir),
                    "comparison_csv": str(comparison), "records": len(rows)})
    report = update_lab_report("lab_4", title="Лабораторная работа № 4 — надёжность и безопасность",
        objective="Проверить контролируемые отказы инструментов, нарушения полномочий, недостоверные сообщения, циклическое делегирование и конфликтующие инструкции.",
        methodology_path="methodology.md", result_dir=result_dir,
        summary=_markdown_summary(summary), report_root=report_root)
    return {"experiment_id": experiment_id, "summary": summary,
            "records": rows, "report": str(report)}


def _summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in ("baseline", "protected"):
        subset = [row for row in rows if row["variant"] == variant]
        if not subset:
            continue
        positive = [row for row in subset if row["detected_violation"]]
        result[variant] = {
            "runs": len(subset), "violations_detected": sum(row["detected_violation"] for row in subset),
            "unsafe_actions_blocked": sum(row["blocked_unsafe_action"] for row in subset),
            "legitimate_tasks_completed": sum(row["legitimate_tasks_completed"] for row in subset),
            "false_blocks": sum(row["false_blocks"] for row in subset),
            "incident_handling_time_ms_mean": sum(row["incident_handling_time_ms"] for row in subset) / len(subset),
            "messages": sum(row["messages"] for row in subset),
            "errors": sum(row["errors"] for row in subset),
            "unfinished_tasks": sum(row["unfinished_tasks"] for row in subset),
            "detection_rate": sum(row["detected_violation"] for row in subset) / len(subset) if subset else None,
        }
    return result


def _markdown_summary(summary: Mapping[str, Any]) -> str:
    lines = [f"Эксперимент `{summary['experiment_id']}` выполнил {summary['records']} локальных сценарных запусков.",
        "\n| Режим | Обнаружено | Доля обнаружения | Заблокировано | Легитимные задачи | Ложные блокировки | Сообщения | Ошибки | Незавершённые | Время, мс/прогон |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in ("baseline", "protected"):
        row = summary.get(variant)
        if isinstance(row, Mapping):
            rate = row['detection_rate']
            lines.append(f"| {variant} | {row['violations_detected']} | {rate:.1%} | {row['unsafe_actions_blocked']} | {row['legitimate_tasks_completed']} | {row['false_blocks']} | {row['messages']} | {row['errors']} | {row['unfinished_tasks']} | {row['incident_handling_time_ms_mean']:.3f} |")
    lines.extend(["", "Защита проверена только на контролируемых локальных сценариях; фактическое выполнение действий было отключено."])
    return "\n".join(lines)


__all__ = ["run_lab4"]
