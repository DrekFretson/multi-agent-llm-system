"""Deterministic resource-conflict, negotiation, priority and arbitration lab."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

from src.communication import Message, MessageBus, MessageType
from src.coordination import Task, TaskPriority, TaskStatus, TaskStore, TaskType
from src.evaluation.lab_support import (
    load_lab_inputs, make_metrics, new_lab_experiment, save_lab_run,
    update_lab_report, write_comparison,
)


COORDINATOR = "lab3_coordinator"
NEGOTIATION_TYPES = {
    MessageType.NEGOTIATION_PROPOSAL.value,
    MessageType.NEGOTIATION_COUNTER.value,
    MessageType.NEGOTIATION_ACCEPT.value,
    MessageType.NEGOTIATION_REJECT.value,
    MessageType.ARBITRATION_DECISION.value,
}


class _ResourceScenario:
    def __init__(self, scenario: Mapping[str, Any], rules: Mapping[str, Any],
                 cost_per_message: float) -> None:
        self.scenario = scenario
        self.capacity = float(scenario["resource_capacity"])
        self.free = self.capacity
        self.rules = rules
        self.cost_per_message = cost_per_message
        self.allocations: list[dict[str, Any]] = []
        self.bus = MessageBus()
        self.store = TaskStore()
        self.bus.register_agent(COORDINATOR)
        self.agent_ids = sorted({claim["agent_id"] for claim in scenario["claims"]})
        for agent_id in self.agent_ids:
            self.bus.register_agent(agent_id)
        for claim in scenario["claims"]:
            self.store.create_task(Task(task_id=claim["task_id"],
                description=claim["description"], task_type=TaskType.GENERAL,
                priority=TaskPriority(claim["priority_name"])))

    def _send(self, sender: str, receiver: str, message_type: MessageType,
              claim: Mapping[str, Any], payload: dict[str, Any]) -> None:
        self.bus.send(Message(sender=sender, receiver=receiver, message_type=message_type,
            task_id=claim["task_id"], payload={"claim_id": claim["claim_id"], **payload}))

    def _receive(self, agent_id: str) -> None:
        self.bus.receive(agent_id)

    def _claim_result(self, claim: Mapping[str, Any], units: float, status: str,
                      reason: str) -> None:
        units = min(max(0.0, units), float(claim["requested_units"]))
        self.free -= units
        self.free = max(0.0, self.free)
        is_successful = units >= float(claim["minimum_units"])
        task = self.store.get_task(claim["task_id"])
        if units > 0:
            self.store.assign_task(task.task_id, claim["agent_id"])
            self.store.update_status(task.task_id, TaskStatus.RUNNING)
        if is_successful:
            self.store.update_task(task.task_id, result={"granted_units": units, "agent_id": claim["agent_id"]})
            self.store.update_status(task.task_id, TaskStatus.COMPLETED)
        elif units == 0:
            self.store.update_task(task.task_id, error=reason)
            self.store.update_status(task.task_id, TaskStatus.SKIPPED)
        else:
            self.store.update_task(task.task_id, error=reason,
                                   result={"granted_units": units, "minimum_units": claim["minimum_units"]})
            self.store.update_status(task.task_id, TaskStatus.FAILED)
        self.allocations.append({"claim_id": claim["claim_id"], "task_id": claim["task_id"],
            "agent_id": claim["agent_id"], "requested_units": claim["requested_units"],
            "minimum_units": claim["minimum_units"], "granted_units": units,
            "satisfaction": units / float(claim["requested_units"]),
            "priority": claim["priority"], "status": status,
            "task_status": self.store.get_task(claim["task_id"]).status.value,
            "reason": reason})


class BaselineResourceAllocator(_ResourceScenario):
    """First-come, first-served allocation with no proposal revision."""

    variant = "baseline"

    def run(self) -> list[dict[str, Any]]:
        for claim in self.scenario["claims"]:
            self._send(COORDINATOR, claim["agent_id"], MessageType.TASK_ASSIGN, claim,
                       {"requested_units": claim["requested_units"], "mode": "baseline"})
            self._receive(claim["agent_id"])
            requested = float(claim["requested_units"])
            if requested <= self.free:
                self._send(claim["agent_id"], COORDINATOR, MessageType.TASK_ACCEPT, claim,
                           {"requested_units": requested})
                self._receive(COORDINATOR)
                self._claim_result(claim, requested, "accepted", "Granted as requested by arrival order.")
            else:
                self._send(claim["agent_id"], COORDINATOR, MessageType.TASK_REJECT, claim,
                           {"reason": "resource capacity exhausted; no negotiation enabled"})
                self._receive(COORDINATOR)
                self._claim_result(claim, 0, "rejected", "Capacity exhausted; baseline does not negotiate.")
        return self.allocations


class NegotiationAllocator(_ResourceScenario):
    """Allow bounded counteroffers; optionally apply priority arbitration."""

    def __init__(self, *args: Any, priority_arbitration: bool, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.priority_arbitration = priority_arbitration
        self.variant = "arbitration" if priority_arbitration else "negotiation"

    def run(self) -> list[dict[str, Any]]:
        claims = list(self.scenario["claims"])
        if self.priority_arbitration:
            claims.sort(key=lambda claim: (-int(claim["priority"]), -float(claim.get("value", 0)), claim["arrival_order"]))
        unresolved: list[Mapping[str, Any]] = []
        for index, claim in enumerate(claims):
            requested = float(claim["requested_units"])
            reserve = 0.0
            if self.priority_arbitration:
                # Keep the minimum feasible for lower-priority claims and for
                # earlier non-negotiable claims awaiting the arbitration pass.
                reserve = sum(float(item["minimum_units"]) for item in claims[index + 1:])
                reserve += sum(float(item["minimum_units"]) for item in unresolved)
            available_for_claim = max(0.0, self.free - reserve)
            self._send(COORDINATOR, claim["agent_id"], MessageType.NEGOTIATION_PROPOSAL,
                       claim, {"offer_units": requested, "round": 0,
                               "priority": claim["priority"] if self.priority_arbitration else None})
            self._receive(claim["agent_id"])
            if requested <= available_for_claim:
                self._send(claim["agent_id"], COORDINATOR, MessageType.NEGOTIATION_ACCEPT,
                           claim, {"accepted_units": requested, "round": 0})
                self._receive(COORDINATOR)
                self._claim_result(claim, requested, "accepted", "The full resource request fit the remaining capacity.")
                continue

            accepted = False
            min_units = float(claim["minimum_units"])
            if claim.get("can_negotiate", False):
                concession = float(claim.get("concession_per_round", 1.0))
                for round_number in range(1, int(self.rules.get("max_rounds", 2)) + 1):
                    counter = min(available_for_claim,
                                  max(min_units, requested - concession * round_number))
                    self._send(COORDINATOR, claim["agent_id"], MessageType.NEGOTIATION_COUNTER,
                               claim, {"offer_units": counter, "round": round_number})
                    self._receive(claim["agent_id"])
                    if counter >= min_units and counter <= self.free:
                        self._send(claim["agent_id"], COORDINATOR, MessageType.NEGOTIATION_ACCEPT,
                                   claim, {"accepted_units": counter, "round": round_number})
                        self._receive(COORDINATOR)
                        self._claim_result(claim, counter, "accepted_revised",
                                           f"Accepted revised offer in round {round_number}.")
                        accepted = True
                        break
                if not accepted:
                    self._send(claim["agent_id"], COORDINATOR, MessageType.NEGOTIATION_REJECT,
                               claim, {"reason": "No counteroffer met the minimum resource requirement."})
                    self._receive(COORDINATOR)
            if not accepted:
                unresolved.append(claim)
                if not self.priority_arbitration:
                    self._claim_result(claim, 0, "unresolved", "Negotiation ended without an acceptable offer.")

        if self.priority_arbitration:
            for index, claim in enumerate(unresolved):
                requested = float(claim["requested_units"])
                minimum = float(claim["minimum_units"])
                remaining_minimum = sum(float(item["minimum_units"])
                                        for item in unresolved[index + 1:])
                protected_capacity = max(0.0, self.free - remaining_minimum)
                grant = min(requested, self.free, max(minimum, protected_capacity))
                self._send(COORDINATOR, claim["agent_id"], MessageType.ARBITRATION_DECISION,
                           claim, {"granted_units": grant, "priority": claim["priority"],
                                   "rule": "priority desc, value desc, arrival asc; grant minimum if feasible"})
                self._receive(claim["agent_id"])
                self._claim_result(claim, grant, "arbitrated" if grant >= minimum else "denied",
                    "Arbitrator applied configured priority/value/arrival rule after negotiation.")
        return self.allocations


def run_lab3(config_path: str | Path | None = None, dataset_path: str | Path | None = None,
             *, report_root: str | Path | None = None) -> dict[str, Any]:
    """Execute the baseline, negotiation, and priority-arbitration variants."""
    project_root = Path(__file__).resolve().parents[2]
    config_path = Path(config_path) if config_path else project_root / "experiments/configs/lab_3.yaml"
    config, scenarios = load_lab_inputs(config_path, dataset_path)
    experiment_id, result_dir = new_lab_experiment("lab_3", config, report_root)
    repeats = int(config.get("repeats", 1))
    cost_per_message = float(config.get("coordination_cost_per_message_units", 1.0))
    rules = config["negotiation"]
    rows: list[dict[str, Any]] = []
    variants = [("baseline", BaselineResourceAllocator, {}),
        ("negotiation", NegotiationAllocator, {"priority_arbitration": False}),
        ("arbitration", NegotiationAllocator, {"priority_arbitration": True})]
    for repeat in range(repeats):
        for scenario in scenarios:
            for variant, mechanism, options in variants:
                metrics = make_metrics(experiment_id)
                simulation = mechanism(scenario, rules, cost_per_message, **options)
                started = time.perf_counter()
                with metrics.activate(agent_role="lab_3_coordinator"):
                    allocations = simulation.run()
                    for item in allocations:
                        task = simulation.store.get_task(item["task_id"])
                        metrics.record_task(task_id=task.task_id,
                            agent_role=task.assignee, status=task.status.value,
                            duration_ms=0.0, error=task.error)
                elapsed_ms = (time.perf_counter() - started) * 1000
                event_rows = metrics.snapshot()["messages"]
                negotiation_messages = sum(row["message_type"] in NEGOTIATION_TYPES for row in event_rows)
                conflicts = _conflicts(scenario)
                satisfied_claims = {item["claim_id"] for item in allocations
                                    if item["granted_units"] >= item["minimum_units"]}
                resolved_conflicts = sum(all(claim["claim_id"] in satisfied_claims for claim in group)
                                         for group in conflicts)
                unresolved_conflicts = len(conflicts) - resolved_conflicts
                priority_violations = _priority_violations(allocations)
                fairness = _jain_index([item["satisfaction"] for item in allocations])
                distribution: dict[str, float] = {agent_id: 0.0 for agent_id in simulation.agent_ids}
                for item in allocations:
                    distribution[item["agent_id"]] += item["granted_units"]
                snapshot = metrics.finish(status="COMPLETED" if not unresolved_conflicts else "PARTIAL")
                row = {
                    "scenario_id": scenario["scenario_id"], "variant": variant,
                    "repeat": repeat + 1, "seed": int(config.get("seed", 0)) + repeat,
                    "run_id": metrics.run_id, "execution_status": snapshot["quality"]["execution_status"],
                    "conflicts": len(conflicts), "resolved_conflicts": resolved_conflicts,
                    "unresolved_conflicts": unresolved_conflicts,
                    "conflict_resolution_time_ms": round(elapsed_ms, 3),
                    "negotiation_messages": negotiation_messages,
                    "messages": snapshot["message_count"],
                    "coordination_cost_units": snapshot["message_count"] * cost_per_message,
                    "completed_tasks": sum(item["task_status"] == "COMPLETED" for item in allocations),
                    "failed_tasks": sum(item["task_status"] == "FAILED" for item in allocations),
                    "skipped_tasks": sum(item["task_status"] == "SKIPPED" for item in allocations),
                    "resource_distribution": distribution,
                    "priority_violations": priority_violations,
                    "jain_fairness_index": fairness,
                    "claims": allocations,
                }
                metrics.set_quality(conflicts=len(conflicts), resolved_conflicts=resolved_conflicts,
                                    unresolved_conflicts=unresolved_conflicts,
                                    priority_violations=priority_violations,
                                    jain_fairness_index=fairness)
                snapshot = metrics.finish(status=row["execution_status"])
                payload = {"experiment_id": experiment_id, **row,
                           "scenario": scenario, "metrics": snapshot,
                           "fairness_method": _fairness_method()}
                paths = save_lab_run(result_dir, scenario_id=row["scenario_id"], variant=variant,
                    repeat=repeat + 1, run_id=metrics.run_id, payload=payload, metrics=metrics)
                row["result_path"] = paths["result"]
                rows.append(row)
    comparison = write_comparison(result_dir, experiment_id, rows)
    summary = _summarize(rows)
    summary.update({"experiment_id": experiment_id, "result_dir": str(result_dir),
                    "comparison_csv": str(comparison), "records": len(rows)})
    report = update_lab_report("lab_3", title="Лабораторная работа № 3 — переговоры и конфликты",
        objective="Исследовать конкурирующие запросы на ограниченный ресурс, пересмотр заявок, приоритеты и арбитраж.",
        methodology_path="methodology.md", result_dir=result_dir,
        summary=_markdown_summary(summary), report_root=report_root)
    return {"experiment_id": experiment_id, "summary": summary,
            "records": rows, "report": str(report)}


def _conflicts(scenario: Mapping[str, Any]) -> list[list[Mapping[str, Any]]]:
    claims = scenario["claims"]
    if len(claims) > 1 and sum(float(row["requested_units"]) for row in claims) > float(scenario["resource_capacity"]):
        return [claims]
    return []


def _priority_violations(allocations: list[Mapping[str, Any]]) -> int:
    violations = 0
    for first in allocations:
        for second in allocations:
            if first["priority"] > second["priority"] and first["satisfaction"] < second["satisfaction"]:
                violations += 1
    return violations


def _jain_index(satisfaction: list[float]) -> float | None:
    """Jain index over request satisfaction shares; it ignores priority/need weights."""
    if not satisfaction or sum(value * value for value in satisfaction) == 0:
        return None
    return (sum(satisfaction) ** 2) / (len(satisfaction) * sum(value * value for value in satisfaction))


def _fairness_method() -> str:
    return ("Jain index over grant/request satisfaction ratios; values nearer 1 mean more equal ratios. "
            "It does not encode task priority, differing resource value, or need-based fairness.")


def _summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in sorted({row["variant"] for row in rows}):
        subset = [row for row in rows if row["variant"] == variant]
        result[variant] = {key: sum(row[key] for row in subset)
            for key in ("conflicts", "resolved_conflicts", "unresolved_conflicts", "negotiation_messages",
                        "completed_tasks", "failed_tasks", "skipped_tasks", "priority_violations", "messages")}
        fairness = [row["jain_fairness_index"] for row in subset if row["jain_fairness_index"] is not None]
        result[variant]["mean_jain_fairness_index"] = sum(fairness) / len(fairness) if fairness else None
        result[variant]["mean_conflict_resolution_time_ms"] = sum(
            row["conflict_resolution_time_ms"] for row in subset) / len(subset)
    return result


def _markdown_summary(summary: Mapping[str, Any]) -> str:
    lines = [f"Эксперимент `{summary['experiment_id']}` выполнил {summary['records']} локальных запусков.",
        "\n| Вариант | Конфликты | Разрешены | Не разрешены | Переговорные сообщения | Все сообщения | Выполнено задач | Провалено | Нарушения приоритета | Jain index | Разрешение, мс/прогон |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant, row in summary.items():
        if isinstance(row, Mapping) and "conflicts" in row:
            fairness = row["mean_jain_fairness_index"]
            fairness_text = f"{fairness:.3f}" if fairness is not None else "n/a"
            lines.append(f"| {variant} | {row['conflicts']} | {row['resolved_conflicts']} | {row['unresolved_conflicts']} | {row['negotiation_messages']} | {row['messages']} | {row['completed_tasks']} | {row['failed_tasks']} | {row['priority_violations']} | {fairness_text} | {row['mean_conflict_resolution_time_ms']:.3f} |")
    lines.extend(["", _fairness_method()])
    return "\n".join(lines)


__all__ = ["BaselineResourceAllocator", "NegotiationAllocator", "run_lab3"]
