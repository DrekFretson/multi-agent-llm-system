"""Local simulations comparing central dispatch, Contract Net, and auctions."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from src.communication import Message, MessageBus, MessageType
from src.coordination import Task, TaskPriority, TaskStatus, TaskStore, TaskType
from src.evaluation.lab_support import (
    load_lab_inputs, make_metrics, new_lab_experiment, save_lab_run,
    update_lab_report, write_comparison,
)


COORDINATOR = "lab2_coordinator"


@dataclass
class AgentCapacity:
    """Deterministic synthetic agent profile used only in the local benchmark."""

    agent_id: str
    capabilities: frozenset[str]
    capacity_units: float
    throughput: float
    cost_per_unit: float
    competence: Mapping[str, float]
    available: bool
    used_units: float = 0.0
    planned_seconds: float = 0.0

    @property
    def load(self) -> float:
        return min(1.0, self.used_units / self.capacity_units) if self.capacity_units else 1.0


@dataclass(frozen=True)
class AgentBid:
    """A structured, auditable offer for one task."""

    agent_id: str
    expected_time_seconds: float
    estimated_cost_units: float
    load: float
    competence: float


class _DispatchSimulation:
    def __init__(self, scenario: Mapping[str, Any], *, coordination_cost_per_message: float) -> None:
        self.scenario = scenario
        self.store = TaskStore()
        self.agents = {item["agent_id"]: AgentCapacity(
            agent_id=item["agent_id"], capabilities=frozenset(item["capabilities"]),
            capacity_units=float(item["capacity_units"]), throughput=float(item["throughput"]),
            cost_per_unit=float(item["cost_per_unit"]), competence=dict(item["competence"]),
            available=bool(item.get("available", True)),
        ) for item in scenario["agents"]}
        self.bus = MessageBus()
        self.bus.register_agent(COORDINATOR)
        for agent_id in self.agents:
            self.bus.register_agent(agent_id)
        self.coordination_cost_per_message = coordination_cost_per_message
        self.decisions: list[dict[str, Any]] = []
        self.unallocated = 0
        self.failed = 0
        self.completed = 0
        self.estimated_cost_units = 0.0
        self._tasks = {item["task_id"]: dict(item) for item in scenario["tasks"]}
        for item in scenario["tasks"]:
            task_type = TaskType(item.get("task_type", "GENERAL"))
            priority = TaskPriority(item.get("priority", "MEDIUM"))
            self.store.create_task(Task(task_id=item["task_id"], description=item["description"],
                                        task_type=task_type, priority=priority))

    def _bid(self, agent: AgentCapacity, task: Mapping[str, Any]) -> AgentBid | None:
        capability = task["capability"]
        work = float(task["work_units"])
        if not agent.available or capability not in agent.capabilities:
            return None
        if agent.capacity_units - agent.used_units < work:
            return None
        return AgentBid(agent.agent_id, work / agent.throughput,
                        work * agent.cost_per_unit, agent.load,
                        float(agent.competence.get(capability, 0.0)))

    def _send(self, sender: str, receiver: str, kind: MessageType,
              task_id: str, payload: dict[str, Any]) -> None:
        self.bus.send(Message(sender=sender, receiver=receiver, message_type=kind,
                              task_id=task_id, payload=payload))

    def _receive(self, agent_id: str) -> Message | None:
        return self.bus.receive(agent_id)

    def _finish_task(self, task_id: str, bid: AgentBid | None, reason: str) -> dict[str, Any]:
        task_data = self._tasks[task_id]
        if bid is None:
            self.unallocated += 1
            self.store.update_task(task_id, error=reason)
            self.store.update_status(task_id, TaskStatus.SKIPPED)
            self.decisions.append({"task_id": task_id, "selected_agent": None, "reason": reason,
                                   "expected_time_seconds": None, "estimated_cost_units": None})
            return self.decisions[-1]

        agent = self.agents[bid.agent_id]
        agent.used_units += float(task_data["work_units"])
        agent.planned_seconds += bid.expected_time_seconds
        self.estimated_cost_units += bid.estimated_cost_units
        self.store.assign_task(task_id, bid.agent_id)
        self.store.update_status(task_id, TaskStatus.RUNNING)
        should_fail = bid.agent_id in task_data.get("fail_for_agents", [])
        if should_fail:
            self.failed += 1
            message = f"Synthetic local failure injected for {bid.agent_id}."
            self.store.update_task(task_id, error=message)
            self.store.update_status(task_id, TaskStatus.FAILED)
        else:
            self.completed += 1
            self.store.update_task(task_id, result={"status": "COMPLETED", "agent_id": bid.agent_id})
            self.store.update_status(task_id, TaskStatus.COMPLETED)
        decision = {"task_id": task_id, "selected_agent": bid.agent_id, "reason": reason,
                    "expected_time_seconds": bid.expected_time_seconds,
                    "estimated_cost_units": bid.estimated_cost_units,
                    "load_before_award": bid.load, "competence": bid.competence,
                    "execution_status": "FAILED" if should_fail else "COMPLETED"}
        self.decisions.append(decision)
        return decision


class CentralizedDispatcher(_DispatchSimulation):
    """Coordinator selects the highest-fit eligible profile without bids."""

    variant = "centralized_dispatch"

    def run(self) -> list[dict[str, Any]]:
        for task_id, task in self._tasks.items():
            candidates = [(agent, self._bid(agent, task)) for agent in self.agents.values()]
            eligible = [(agent, bid) for agent, bid in candidates if bid is not None]
            if not eligible:
                self._finish_task(task_id, None, "No available agent has the capability and capacity.")
                continue
            # Explicit central rule: capability is a hard filter, then highest
            # competence, then fastest estimate, lower load and lexical ID.
            agent, bid = min(eligible, key=lambda pair: (
                -pair[1].competence, pair[1].expected_time_seconds,
                pair[1].load, pair[0].agent_id))
            self._send(COORDINATOR, agent.agent_id, MessageType.TASK_ASSIGN, task_id,
                       {"description": task["description"], "model": "central_dispatch",
                        "selection_reason": "highest competence, then time/load tie-break"})
            self._receive(agent.agent_id)
            self._send(agent.agent_id, COORDINATOR, MessageType.TASK_ACCEPT, task_id,
                       {"agent_id": agent.agent_id})
            self._receive(COORDINATOR)
            decision = self._finish_task(task_id, bid,
                "Selected by central competence/time/load rule.")
            self._send(agent.agent_id, COORDINATOR, MessageType.RESULT, task_id,
                       {"status": decision["execution_status"], "reason": decision["reason"]})
            self._receive(COORDINATOR)
        return self.decisions


class ContractNetAllocator(_DispatchSimulation):
    """Call-for-proposals protocol; winner is the fastest feasible bidder."""

    variant = "contract_net"

    def run(self) -> list[dict[str, Any]]:
        for task_id, task in self._tasks.items():
            bids: list[AgentBid] = []
            for agent in self.agents.values():
                self._send(COORDINATOR, agent.agent_id, MessageType.TASK_ASSIGN, task_id,
                           {"contract_net": True, "description": task["description"],
                            "requested_capability": task["capability"]})
                self._receive(agent.agent_id)
                bid = self._bid(agent, task)
                if bid is None:
                    self._send(agent.agent_id, COORDINATOR, MessageType.TASK_REJECT, task_id,
                               {"reason": "capability, availability, or capacity constraint"})
                else:
                    bids.append(bid)
                    self._send(agent.agent_id, COORDINATOR, MessageType.TASK_BID, task_id,
                               _bid_payload(bid))
                self._receive(COORDINATOR)
            if not bids:
                self._finish_task(task_id, None, "Contract Net received no eligible bids.")
                continue
            winner = min(bids, key=lambda bid: (
                bid.expected_time_seconds, -bid.competence, bid.load,
                bid.estimated_cost_units, bid.agent_id))
            self._send(COORDINATOR, winner.agent_id, MessageType.CONTRACT_AWARD, task_id,
                       {"selection_rule": "minimum expected time; competence/load/cost/ID tie-break",
                        "bid": _bid_payload(winner)})
            self._receive(winner.agent_id)
            self._send(winner.agent_id, COORDINATOR, MessageType.TASK_ACCEPT, task_id,
                       {"accepted": True})
            self._receive(COORDINATOR)
            decision = self._finish_task(task_id, winner,
                "Awarded to the fastest feasible bidder under Contract Net rules.")
            self._send(winner.agent_id, COORDINATOR, MessageType.RESULT, task_id,
                       {"status": decision["execution_status"]})
            self._receive(COORDINATOR)
        return self.decisions


class AuctionAllocator(_DispatchSimulation):
    """Independent weighted auction over time, cost, load and competence bids."""

    variant = "auction"

    def __init__(self, *args: Any, weights: Mapping[str, float], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        required = {"time", "cost", "load", "competence"}
        if set(weights) != required or any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("Auction weights must define non-negative time/cost/load/competence weights.")
        self.weights = dict(weights)

    def run(self) -> list[dict[str, Any]]:
        for task_id, task in self._tasks.items():
            bids: list[AgentBid] = []
            for agent in self.agents.values():
                self._send(COORDINATOR, agent.agent_id, MessageType.AUCTION_CALL, task_id,
                           {"description": task["description"], "criteria": list(self.weights)})
                self._receive(agent.agent_id)
                bid = self._bid(agent, task)
                if bid is None:
                    self._send(agent.agent_id, COORDINATOR, MessageType.TASK_REJECT, task_id,
                               {"reason": "not eligible to bid"})
                else:
                    bids.append(bid)
                    self._send(agent.agent_id, COORDINATOR, MessageType.TASK_BID, task_id,
                               _bid_payload(bid))
                self._receive(COORDINATOR)
            if not bids:
                self._finish_task(task_id, None, "Auction received no eligible bids.")
                continue
            score_map = self._score_bids(bids)
            winner = min(bids, key=lambda bid: (-score_map[bid.agent_id], bid.agent_id))
            explanation = {"selection_rule": "maximum configured weighted score",
                           "weights": self.weights, "scores": score_map,
                           "winning_bid": _bid_payload(winner)}
            self._send(COORDINATOR, winner.agent_id, MessageType.AUCTION_AWARD, task_id, explanation)
            self._receive(winner.agent_id)
            self._send(winner.agent_id, COORDINATOR, MessageType.TASK_ACCEPT, task_id,
                       {"accepted": True, "auction_score": score_map[winner.agent_id]})
            self._receive(COORDINATOR)
            decision = self._finish_task(task_id, winner,
                f"Auction score {score_map[winner.agent_id]:.6f}; weighted criteria are recorded.")
            decision["auction_score"] = score_map[winner.agent_id]
            decision["auction_scores"] = score_map
            self._send(winner.agent_id, COORDINATOR, MessageType.RESULT, task_id,
                       {"status": decision["execution_status"]})
            self._receive(COORDINATOR)
        return self.decisions

    def _score_bids(self, bids: list[AgentBid]) -> dict[str, float]:
        max_time = max((bid.expected_time_seconds for bid in bids), default=1.0) or 1.0
        max_cost = max((bid.estimated_cost_units for bid in bids), default=1.0) or 1.0
        denominator = sum(self.weights.values())
        return {bid.agent_id: (
            self.weights["competence"] * bid.competence
            - self.weights["time"] * bid.expected_time_seconds / max_time
            - self.weights["cost"] * bid.estimated_cost_units / max_cost
            - self.weights["load"] * bid.load
        ) / denominator for bid in bids}


def run_lab2(config_path: str | Path | None = None, dataset_path: str | Path | None = None,
             *, report_root: str | Path | None = None) -> dict[str, Any]:
    """Execute all Lab 2 strategies against the same local scenarios and repeats."""
    project_root = Path(__file__).resolve().parents[2]
    config_path = Path(config_path) if config_path else project_root / "experiments/configs/lab_2.yaml"
    config, scenarios = load_lab_inputs(config_path, dataset_path)
    experiment_id, result_dir = new_lab_experiment("lab_2", config, report_root)
    repeats = int(config.get("repeats", 1))
    message_cost = float(config.get("coordination_cost_per_message_units", 1.0))
    weights = config["auction"]["weights"]
    variants: list[tuple[str, Callable[..., _DispatchSimulation]]] = [
        (CentralizedDispatcher.variant, CentralizedDispatcher),
        (ContractNetAllocator.variant, ContractNetAllocator),
        (AuctionAllocator.variant, AuctionAllocator),
    ]
    rows: list[dict[str, Any]] = []
    raw_runs: list[dict[str, Any]] = []
    for repeat in range(repeats):
        for scenario in scenarios:
            for variant, mechanism in variants:
                metrics = make_metrics(experiment_id)
                simulation = mechanism(scenario,
                    coordination_cost_per_message=message_cost,
                    **({"weights": weights} if variant == "auction" else {}))
                started = time.perf_counter()
                with metrics.activate(agent_role="lab_2_coordinator"):
                    decisions = simulation.run()
                    for decision in decisions:
                        task = simulation.store.get_task(decision["task_id"])
                        metrics.record_task(task_id=task.task_id,
                            agent_role=task.assignee, status=task.status.value,
                            duration_ms=0.0, error=task.error)
                wall_ms = (time.perf_counter() - started) * 1000
                message_rows = metrics.snapshot()["messages"]
                coordination_count = sum(row["message_type"] != MessageType.RESULT.value
                                         for row in message_rows)
                snapshot = metrics.finish(status="COMPLETED" if not simulation.failed else "PARTIAL")
                modeled_makespan = max((agent.planned_seconds for agent in simulation.agents.values()), default=0.0)
                coordination_units = coordination_count * message_cost
                row = {
                    "scenario_id": scenario["scenario_id"], "variant": variant,
                    "repeat": repeat + 1, "seed": int(config.get("seed", 0)) + repeat,
                    "run_id": metrics.run_id, "execution_status": snapshot["quality"]["execution_status"],
                    "simulation_wall_time_ms": round(wall_ms, 3),
                    "estimated_makespan_seconds": round(modeled_makespan, 6),
                    "messages": snapshot["message_count"],
                    "coordination_messages": coordination_count,
                    "coordination_cost_units": coordination_units,
                    "estimated_execution_cost_units": round(simulation.estimated_cost_units, 6),
                    "completed_tasks": simulation.completed,
                    "unallocated_tasks": simulation.unallocated,
                    "failed_tasks": simulation.failed,
                    "assignments": decisions,
                }
                payload = {"experiment_id": experiment_id, **row,
                           "scenario": scenario, "metrics": snapshot}
                paths = save_lab_run(result_dir, scenario_id=row["scenario_id"],
                    variant=variant, repeat=repeat + 1, run_id=metrics.run_id,
                    payload=payload, metrics=metrics)
                row["result_path"] = paths["result"]
                rows.append(row)
                raw_runs.append(payload)
    comparison = write_comparison(result_dir, experiment_id, rows)
    summary = _summarize(rows)
    summary.update({"experiment_id": experiment_id, "result_dir": str(result_dir),
                    "comparison_csv": str(comparison), "records": len(rows)})
    report = update_lab_report("lab_2", title="Лабораторная работа № 2 — распределение задач",
        objective="Сравнить центральную диспетчеризацию, Contract Net и взвешенный аукцион на одинаковых задачах и профилях агентов.",
        methodology_path="methodology.md", result_dir=result_dir,
        summary=_markdown_summary(summary), report_root=report_root)
    return {"experiment_id": experiment_id, "summary": summary,
            "records": rows, "report": str(report)}


def _bid_payload(bid: AgentBid) -> dict[str, Any]:
    return {"agent_id": bid.agent_id, "expected_time_seconds": bid.expected_time_seconds,
            "estimated_cost_units": bid.estimated_cost_units, "load": bid.load,
            "competence": bid.competence}


def _summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for variant in sorted({row["variant"] for row in rows}):
        subset = [row for row in rows if row["variant"] == variant]
        output[variant] = {
            "runs": len(subset), "completed_tasks": sum(row["completed_tasks"] for row in subset),
            "unallocated_tasks": sum(row["unallocated_tasks"] for row in subset),
            "failed_tasks": sum(row["failed_tasks"] for row in subset),
            "messages": sum(row["messages"] for row in subset),
            "coordination_messages": sum(row["coordination_messages"] for row in subset),
            "coordination_cost_units": sum(row["coordination_cost_units"] for row in subset),
            "mean_estimated_makespan_seconds": sum(row["estimated_makespan_seconds"] for row in subset) / len(subset),
            "mean_simulation_wall_time_ms": sum(row["simulation_wall_time_ms"] for row in subset) / len(subset),
        }
    return output


def _markdown_summary(summary: Mapping[str, Any]) -> str:
    lines = [f"Эксперимент `{summary['experiment_id']}` выполнил {summary['records']} локальных запусков.",
             "\n| Механизм | Завершено | Не распределено | Провалы | Все сообщения | Сообщения координации | Условная стоимость | Оценочный makespan, с | Python wall time, мс/прогон |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant, data in summary.items():
        if not isinstance(data, Mapping) or "runs" not in data:
            continue
        lines.append(f"| {variant} | {data['completed_tasks']} | {data['unallocated_tasks']} | {data['failed_tasks']} | {data['messages']} | {data['coordination_messages']} | {data['coordination_cost_units']} | {data['mean_estimated_makespan_seconds']:.3f} | {data['mean_simulation_wall_time_ms']:.3f} |")
    lines.extend(["", "Условная стоимость рассчитана как число coordination-сообщений × заданная стоимость одной единицы сообщения; это не денежная сумма. Python wall time измеряет только локальный harness, а makespan рассчитан из throughput профилей."])
    return "\n".join(lines)


__all__ = ["AgentBid", "AgentCapacity", "AuctionAllocator", "CentralizedDispatcher",
           "ContractNetAllocator", "run_lab2"]
