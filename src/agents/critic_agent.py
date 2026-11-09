"""Independent quality review agent with deterministic supporting checks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.agents.base_agent import AgentInputError, BaseAgent
from src.agents.config import AgentConfig, AgentConfigRegistry
from src.llm import LLMClient
from src.tools import CriticCheck, CriticFinding, CriticTool, CriticToolResult


CRITIC_SYSTEM_PROMPT = """\
Ты независимый Critic Agent. Оцени результат строго относительно исходной задачи.
Не соглашайся с выводами проверяемого агента только потому, что они уверенно
сформулированы, и не переписывай ответ вместо проверки.

Проверь применимые критерии: correctness, task compliance, completeness,
logical consistency, evidence and verification. Подбирай критерии и вес доказательств
по типу результата. Ищи конкретные фактические, логические, технические ошибки,
граничные случаи и существенные пропуски. Отделяй ошибки от стиля.

Используй приложенные результаты CriticTool как программно подтверждённые факты.
Не утверждай, что код выполнялся: проверки Python ограничены AST и не запускают
предоставленную программу. Не считай синтаксическую корректность доказательством
логической правильности. Для исследования различай наличие URL, соответствие URL
предоставленному набору и достоверность содержания источника. Не делай последнее
предположение без текста источника. Для математики независимо анализируй условие,
вычисления и ограничения; не доверяй одному лишь флагу verified.
Если ResearchReport указывает, что веб-поиск не выполнялся, не считай внешние
факты проверенными и назови отсутствие внешней проверки ограничением. Само
отсутствие источников не является ошибкой формата или основанием для REJECTED.

Каждое замечание должно указывать конкретную проблему и рекомендацию. Не выдумывай
замечания и не используй неподтверждённые предположения. Явно укажи ограничения,
если материала недостаточно. APPROVED означает отсутствие критических и иных
существенных неисправленных проблем; REJECTED требуется при существенном нарушении
задачи, критической/major ошибке или отсутствии обязательной части. Высокий score
не отменяет ошибку и не определяет вердикт сам по себе.

Верни только JSON без Markdown, строго с полями:
{
  "verdict": "APPROVED или REJECTED",
  "score": 0,
  "issues": [{"category":"CORRECTNESS|LOGIC|TASK_COMPLIANCE|COMPLETENESS|SECURITY|EVIDENCE|OTHER",
    "severity":"CRITICAL|MAJOR|MINOR|INFO","description":"конкретная проблема",
    "location":null,"recommendation":"действие для исправления"}],
  "strengths": ["проверяемое достоинство"],
  "summary": "обоснованный итог",
  "criteria": {"correctness":"...","task_compliance":"...","completeness":"...",
    "logical_consistency":"...","evidence_and_verification":"..."},
  "verification_limits": ["что проверить не удалось"]
}
"""

ISSUE_CATEGORIES = frozenset({
    "CORRECTNESS", "LOGIC", "TASK_COMPLIANCE", "COMPLETENESS", "SECURITY", "EVIDENCE", "OTHER"
})
ISSUE_SEVERITIES = frozenset({"CRITICAL", "MAJOR", "MINOR", "INFO"})
VERDICTS = frozenset({"APPROVED", "REJECTED"})
CRITERIA = (
    "correctness", "task_compliance", "completeness", "logical_consistency",
    "evidence_and_verification",
)


class CriticAgentError(Exception):
    """Base class for CriticAgent review errors."""


class CriticLLMError(CriticAgentError):
    """Raised when the configured LLM call fails."""


class CriticReportParseError(CriticAgentError):
    """Raised when the LLM report is malformed or violates its schema."""


@dataclass(frozen=True)
class CriticIssue:
    """One actionable issue discovered during a review."""

    category: str
    severity: str
    description: str
    location: str | None
    recommendation: str


@dataclass(frozen=True)
class CriticReport:
    """Structured verdict with LLM judgments and deterministic evidence separated."""

    verdict: str
    score: float
    issues: list[CriticIssue]
    strengths: list[str]
    summary: str
    result_type: str
    criteria: dict[str, str] = field(default_factory=dict)
    checks: list[dict[str, str]] = field(default_factory=list)
    verification_limits: list[str] = field(default_factory=list)


class CriticAgent(BaseAgent):
    """Review other agents' outputs using an LLM and independent local checks."""

    CONFIG_ID = "critic"
    MAX_REVIEW_CHARS = 60_000

    def __init__(
        self,
        client: LLMClient,
        critic_tool: CriticTool | None = None,
        config: AgentConfig | None = None,
        *,
        config_registry: AgentConfigRegistry | None = None,
        agent_id: str = "critic",
        name: str = "Critic Agent",
        system_prompt: str = CRITIC_SYSTEM_PROMPT,
    ) -> None:
        """Create the critic with shared model configuration and local tools."""
        if config is None:
            config = (config_registry or AgentConfigRegistry()).get_agent(self.CONFIG_ID)
        elif config.name != self.CONFIG_ID:
            raise ValueError(f"CriticAgent requires the agent config {self.CONFIG_ID!r}.")
        super().__init__(agent_id, name, system_prompt, client, config)
        self.critic_tool = critic_tool or CriticTool()

    def review(
        self,
        task: str,
        result: Any,
        *,
        result_type: str | None = None,
        agent_name: str | None = None,
        additional_materials: Mapping[str, Any] | None = None,
        required_elements: Sequence[str] = (),
    ) -> CriticReport:
        """Review another agent's output against the original request.

        ``additional_materials`` can carry raw source excerpts, an approved URL
        list, test expectations, or other facts necessary for independent review.
        """
        if not isinstance(task, str) or not task.strip():
            raise AgentInputError("task must be a non-empty string.")
        if result is None or result == "":
            raise AgentInputError("result must contain the output to review.")
        if additional_materials is not None and not isinstance(additional_materials, Mapping):
            raise AgentInputError("additional_materials must be a mapping or None.")
        if not isinstance(required_elements, (list, tuple)) or any(not isinstance(v, str) for v in required_elements):
            raise AgentInputError("required_elements must be a sequence of strings.")

        plain_result = CriticTool.to_plain_data(result)
        if isinstance(plain_result, dict) and isinstance(plain_result.get("content"), str):
            content = plain_result["content"]
            try:
                decoded_content = json.loads(content)
            except json.JSONDecodeError:
                pass
            else:
                plain_result = decoded_content
        materials = CriticTool.to_plain_data(dict(additional_materials or {}))
        resolved_type = result_type or self._infer_result_type(plain_result, agent_name)
        if not isinstance(resolved_type, str) or not resolved_type.strip():
            raise AgentInputError("result_type must be a non-empty string or None.")
        tool_result = self._run_checks(
            task.strip(), plain_result, resolved_type.lower(), materials, required_elements
        )
        payload = {
            "original_task": task.strip(),
            "reviewed_agent": agent_name,
            "result_type": resolved_type,
            "result": plain_result,
            "additional_materials": materials,
            "deterministic_checks": CriticTool.to_plain_data(tool_result),
            "review_guidance": {
                "criteria": list(CRITERIA),
                "required_elements": list(required_elements),
                "distinguish_tool_checks_from_judgment": True,
                "web_search_performed": (
                    plain_result.get("search_performed")
                    if isinstance(plain_result, Mapping)
                    else None
                ),
                "external_fact_checking": (
                    "not_performed"
                    if isinstance(plain_result, Mapping)
                    and plain_result.get("search_performed") is False
                    else "evaluate_from_available_evidence"
                ),
            },
        }
        request = json.dumps(payload, ensure_ascii=False, allow_nan=False, default=str)
        if len(request) > self.MAX_REVIEW_CHARS:
            raise AgentInputError(f"Review input exceeds {self.MAX_REVIEW_CHARS} characters.")
        response = super().run("Оцени следующий результат независимым образом:\n" + request)
        if not response.success:
            raise CriticLLMError(f"Critic review call failed: {response.error}")
        parsed = self._parse_report(response.content, resolved_type, tool_result)
        return parsed

    def _run_checks(
        self,
        task: str,
        result: Any,
        result_type: str,
        materials: Mapping[str, Any],
        required_elements: Sequence[str],
    ) -> CriticToolResult:
        checks: list[CriticToolResult] = []
        if required_elements:
            checks.append(self.critic_tool.check_structure(result, required_elements=required_elements))

        if result_type in {"coding", "code", "codingagent", "python"}:
            if isinstance(result, Mapping):
                checks.append(self.critic_tool.check_structure(
                    result, required_elements=required_elements or ("code", "tests")
                ))
            code = self._text_field(result, "code")
            if code is None and isinstance(result, str):
                code = result
            if code is not None and ("python" in task.lower() or result_type in {"coding", "code", "codingagent", "python"}):
                checks.append(self.critic_tool.check_python(code, task=task))
            elif code is None:
                checks.append(CriticToolResult(
                    checks=[], findings=[], limitations=["No code field was available for AST review."]
                ))
        elif result_type in {"research", "researchreport", "report"}:
            report_text = self._flatten_text(result)
            supplied = materials.get("sources", materials.get("provided_sources", []))
            if isinstance(result, Mapping):
                supplied = [*self._source_values(result.get("sources", [])), *self._source_values(supplied)]
            checks.append(self.critic_tool.check_source_urls(report_text, supplied or []))
            if isinstance(result, Mapping) and result.get("search_performed") is False:
                checks.append(CriticToolResult(
                    checks=[CriticCheck(
                        "external_fact_verification",
                        "unavailable",
                        "Internet search was not performed; external factual verification is unavailable.",
                    )],
                    findings=[],
                    limitations=[
                        "Research claims were not checked against current external sources."
                    ],
                ))
        elif result_type in {"math", "mathreport", "mathresult"}:
            if isinstance(result, Mapping):
                checks.extend(self._check_structured_math(result, materials))
            else:
                checks.append(CriticToolResult(
                    checks=[], findings=[], limitations=["Free-form math text was not independently executable or structured for substitution."]
                ))
        else:
            checks.append(CriticToolResult(
                checks=[], findings=[], limitations=[
                    f"No specialized deterministic checker is registered for result type {result_type!r}; review uses general LLM criteria only."
                ]
            ))

        return CriticToolResult(
            checks=[check for item in checks for check in item.checks],
            findings=[finding for item in checks for finding in item.findings],
            limitations=[limit for item in checks for limit in item.limitations],
        )

    def _check_structured_math(
        self, result: Mapping[str, Any], materials: Mapping[str, Any]
    ) -> list[CriticToolResult]:
        """Independently check direct math claims or supported MathReport minimizations."""
        expr = result.get("expression") or materials.get("expression")
        variable = result.get("variable") or materials.get("variable") or "x"
        point = result.get("reported_point", result.get("point"))
        value = result.get("reported_value", result.get("value"))
        if isinstance(expr, str) and self._is_number(point) and self._is_number(value):
            return [self.critic_tool.check_math_value(
                expression=expr, variable=variable, reported_point=float(point),
                reported_value=float(value), check_stationary=bool(result.get("check_stationary", True)),
            )]

        found: list[CriticToolResult] = []
        tool_results = result.get("tool_results", [])
        if isinstance(tool_results, list):
            for tool_result in tool_results:
                if (
                    not isinstance(tool_result, Mapping)
                    or not isinstance(tool_result.get("operation"), str)
                    or tool_result.get("operation") not in {"minimize", "maximize"}
                ):
                    continue
                expr = tool_result.get("expression")
                value_result = tool_result.get("result")
                variables = value_result.get("variables") if isinstance(value_result, Mapping) else None
                objective = value_result.get("objective_value") if isinstance(value_result, Mapping) else None
                if isinstance(expr, str) and isinstance(variables, Mapping) and len(variables) == 1 and self._is_number(objective):
                    variable, point = next(iter(variables.items()))
                    if isinstance(variable, str) and self._is_number(point):
                        found.append(self.critic_tool.check_math_value(
                            expression=expr, variable=variable, reported_point=float(point),
                            # The serialized MathTool result does not preserve
                            # active bounds/constraints, so a nonzero gradient
                            # alone cannot disprove a constrained optimum.
                            reported_value=float(objective), check_stationary=False,
                        ))
                elif isinstance(variables, Mapping) and len(variables) > 1:
                    found.append(CriticToolResult(
                        checks=[CriticCheck("math_substitution", "unavailable", "Multivariable optimum needs a multivariable independent checker.")],
                        findings=[], limitations=["Independent substitution for multivariable optimization is not implemented."],
                    ))
        if not found:
            return [CriticToolResult(
                checks=[], findings=[], limitations=[
                    "The structured math result did not expose a supported expression, point, and value for independent substitution."
                ]
            )]
        return found

    @classmethod
    def _parse_report(
        cls,
        content: str,
        result_type: str,
        tool_result: CriticToolResult,
    ) -> CriticReport:
        try:
            data = json.loads(content)
        except json.JSONDecodeError as error:
            raise CriticReportParseError(
                f"Critic response is invalid JSON at line {error.lineno}, column {error.colno}."
            ) from error
        fields = {"verdict", "score", "issues", "strengths", "summary", "criteria", "verification_limits"}
        if not isinstance(data, dict) or data.keys() != fields:
            raise CriticReportParseError("Critic response has missing or unexpected fields.")
        verdict = data["verdict"]
        score = data["score"]
        if not isinstance(verdict, str) or verdict not in VERDICTS:
            raise CriticReportParseError("verdict must be APPROVED or REJECTED.")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 100:
            raise CriticReportParseError("score must be a number from 0 to 100.")
        if not isinstance(data["summary"], str) or not data["summary"].strip():
            raise CriticReportParseError("summary must be a non-empty string.")
        if not isinstance(data["strengths"], list) or any(not isinstance(item, str) for item in data["strengths"]):
            raise CriticReportParseError("strengths must be a list of strings.")
        if not isinstance(data["verification_limits"], list) or any(not isinstance(item, str) for item in data["verification_limits"]):
            raise CriticReportParseError("verification_limits must be a list of strings.")
        if not isinstance(data["criteria"], dict) or set(data["criteria"]) != set(CRITERIA) or any(not isinstance(v, str) for v in data["criteria"].values()):
            raise CriticReportParseError("criteria must contain a string evaluation for each required criterion.")
        issues = [cls._parse_issue(item) for item in data["issues"]] if isinstance(data["issues"], list) else None
        if issues is None:
            raise CriticReportParseError("issues must be a list.")
        issues.extend(CriticIssue(
            category=finding.category,
            severity=finding.severity,
            description=finding.description,
            location=finding.location,
            recommendation=finding.recommendation,
        ) for finding in tool_result.findings)
        limits = list(dict.fromkeys([*data["verification_limits"], *tool_result.limitations]))
        severe_issue = any(issue.severity in {"CRITICAL", "MAJOR"} for issue in issues)
        # Deterministic serious findings always override the language model's vote.
        final_verdict = "REJECTED" if severe_issue else verdict
        checks = [{"name": c.name, "status": c.status, "details": c.details} for c in tool_result.checks]
        if tool_result.findings:
            final_verdict = "REJECTED"
        return CriticReport(
            verdict=final_verdict,
            score=float(score),
            issues=issues,
            strengths=data["strengths"],
            summary=data["summary"],
            result_type=result_type,
            criteria=dict(data["criteria"]),
            checks=checks,
            verification_limits=limits,
        )

    @staticmethod
    def _parse_issue(data: Any) -> CriticIssue:
        fields = {"category", "severity", "description", "location", "recommendation"}
        if not isinstance(data, dict) or data.keys() != fields:
            raise CriticReportParseError("Each issue must have exactly category, severity, description, location, recommendation.")
        if (
            not isinstance(data["category"], str)
            or data["category"] not in ISSUE_CATEGORIES
            or not isinstance(data["severity"], str)
            or data["severity"] not in ISSUE_SEVERITIES
        ):
            raise CriticReportParseError("Issue category or severity is not supported.")
        if not isinstance(data["description"], str) or not data["description"].strip():
            raise CriticReportParseError("Issue description must be a non-empty string.")
        if data["location"] is not None and not isinstance(data["location"], str):
            raise CriticReportParseError("Issue location must be a string or null.")
        if not isinstance(data["recommendation"], str) or not data["recommendation"].strip():
            raise CriticReportParseError("Issue recommendation must be a non-empty string.")
        return CriticIssue(**data)

    @staticmethod
    def _infer_result_type(result: Any, agent_name: str | None) -> str:
        agent = (agent_name or "").lower()
        if "coding" in agent or (isinstance(result, Mapping) and "code" in result):
            return "coding"
        if "research" in agent or (isinstance(result, Mapping) and {"summary", "sources"} <= result.keys()):
            return "research"
        if "math" in agent or (isinstance(result, Mapping) and {"expression", "point", "value"} <= result.keys()):
            return "math"
        return "general"

    @staticmethod
    def _text_field(value: Any, key: str) -> str | None:
        field_value = value.get(key) if isinstance(value, Mapping) else None
        return field_value if isinstance(field_value, str) else None

    @staticmethod
    def _source_values(sources: Any) -> list[Any]:
        if not isinstance(sources, (list, tuple)):
            return []
        return [source if isinstance(source, str) else source for source in sources]

    @classmethod
    def _flatten_text(cls, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            return "\n".join(cls._flatten_text(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return "\n".join(cls._flatten_text(item) for item in value)
        return ""

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
