"""Integration coverage for isolated, deterministic Lab 2–4 simulations."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.evaluation.lab2_dispatch import run_lab2
from src.evaluation.lab3_negotiation import run_lab3
from src.evaluation.lab4_reliability import run_lab4
from src.security import SecurityPolicy, SecurityPolicyError
from src.communication import Message, MessageBus, MessageType


PROJECT = Path(__file__).resolve().parents[1]


def test_lab2_compares_independent_mechanisms_and_isolates_outputs(tmp_path: Path) -> None:
    first = run_lab2(report_root=tmp_path)
    second = run_lab2(report_root=tmp_path)

    assert first["experiment_id"] != second["experiment_id"]
    assert len(first["records"]) == 18
    variants = {row["variant"] for row in first["records"]}
    assert variants == {"centralized_dispatch", "contract_net", "auction"}
    assert len({row["run_id"] for row in first["records"]}) == len(first["records"])
    assert all(Path(row["result_path"]).is_file() for row in first["records"])
    assert first["summary"]["contract_net"]["coordination_messages"] > first["summary"]["centralized_dispatch"]["coordination_messages"]
    assert first["summary"]["auction"]["failed_tasks"] == 0
    for run in (first, second):
        assert Path(run["summary"]["comparison_csv"]).parent.parent.parent.name == "lab_2"
        assert (tmp_path / "lab_2" / "report.md").is_file()
    assert not (tmp_path / "lab_3").exists()


def test_lab3_negotiation_arbitration_and_reproducible_allocations(tmp_path: Path) -> None:
    run = run_lab3(report_root=tmp_path)
    assert len(run["records"]) == 18
    assert {row["variant"] for row in run["records"]} == {"baseline", "negotiation", "arbitration"}
    assert run["summary"]["arbitration"]["resolved_conflicts"] > run["summary"]["baseline"]["resolved_conflicts"]
    assert run["summary"]["arbitration"]["mean_jain_fairness_index"] > run["summary"]["baseline"]["mean_jain_fairness_index"]
    assert run["summary"]["negotiation"]["negotiation_messages"] > 0
    for row in run["records"]:
        scenario = next(item for item in json.loads((PROJECT / "experiments/datasets/lab_3_scenarios.json").read_text())
                        if item["scenario_id"] == row["scenario_id"])
        assert sum(item["granted_units"] for item in row["claims"]) <= scenario["resource_capacity"]
        assert Path(row["result_path"]).is_file()
    arbitration_claims = next(row["claims"] for row in run["records"]
                              if row["scenario_id"] == "high_low_priority_conflict"
                              and row["variant"] == "arbitration" and row["repeat"] == 1)
    assert all(item["granted_units"] >= item["minimum_units"] for item in arbitration_claims)


def test_lab4_runs_controls_and_records_security_incidents(tmp_path: Path) -> None:
    run = run_lab4(report_root=tmp_path)
    assert len(run["records"]) == 36
    assert all(row["criterion_met"] for row in run["records"])
    protected = [row for row in run["records"] if row["variant"] == "protected"]
    assert sum(row["detected_violation"] for row in protected) == len(protected)
    assert sum(row["blocked_unsafe_action"] for row in protected) > 0
    assert sum(row["false_blocks"] for row in protected) == 0
    assert all(row["security_incidents"] for row in protected)
    assert all(row["outcome"].get("action_executed", False) is False for row in run["records"]
               if row["scenario_id"] == "agent_wrong_goal")
    for row in run["records"]:
        assert Path(row["result_path"]).is_file()
        assert row["metrics"]["experiment_id"] == run["experiment_id"]


def test_lab4_concurrent_runs_do_not_mix_run_ids_or_files(tmp_path: Path) -> None:
    with ThreadPoolExecutor(max_workers=2) as pool:
        runs = list(pool.map(lambda _: run_lab4(report_root=tmp_path), range(2)))
    assert runs[0]["experiment_id"] != runs[1]["experiment_id"]
    result_dirs = [Path(run["summary"]["result_dir"]) for run in runs]
    assert result_dirs[0] != result_dirs[1]
    for run, result_dir in zip(runs, result_dirs):
        assert all(row["run_id"] != run["experiment_id"] for row in run["records"])
        assert len(list(result_dir.glob("*.json"))) >= len(run["records"])
        for path in result_dir.glob("*_malformed_message_protected_*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            assert record["experiment_id"] == run["experiment_id"]


def test_message_policy_denial_happens_before_inbox_delivery() -> None:
    policy = SecurityPolicy(
        principals={"owner": "coordinator", "worker": "coding", "intruder": "untrusted"},
        allowed_message_types={"coordinator": [MessageType.TASK_ASSIGN], "coding": [], "untrusted": []},
        allowed_recipient_roles={"coordinator": {"coding"}, "coding": set(), "untrusted": set()},
        payload_schemas={MessageType.TASK_ASSIGN: {"description": str}},
        allowed_actions={"coordinator": {"assign"}, "coding": set(), "untrusted": set()},
    )
    bus = MessageBus(authorizer=policy.authorize_message)
    for agent in ("owner", "worker", "intruder"):
        bus.register_agent(agent)
    forged = Message(sender="intruder", receiver="worker", message_type=MessageType.TASK_ASSIGN,
                     task_id="task-1", payload={"description": "forged"})
    try:
        bus.send(forged)
    except Exception as error:
        assert type(error).__name__ == "MessageAuthorizationError"
    else:
        raise AssertionError("unauthorized sender should have been denied")
    assert bus.receive("worker") is None
    assert policy.incidents[0].incident_type == "message_authority"

    malformed = Message(sender="owner", receiver="worker", message_type=MessageType.TASK_ASSIGN,
                        task_id="task-2", payload={"wrong": "schema"})
    try:
        policy.authorize_message(malformed)
    except SecurityPolicyError:
        pass
    else:
        raise AssertionError("bad payload contract should have been denied")
