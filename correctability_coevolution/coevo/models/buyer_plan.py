from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any

from json_repair import repair_json


BUYER_ACTION_PAYLOAD_FIELDS = {
    "answer_normally": ("answer",),
    "reveal_hidden_constraint": ("constraint",),
    "withhold_information": (),
    "clarify_previous_statement": ("clarification",),
    "challenge_student_assumption": ("assumption",),
    "request_alternative": ("criteria",),
    "accept_proposal": (),
    "reject_proposal": ("reason",),
    "confirm_action": ("action",),
    "execute_user_tool": ("tool_name", "arguments"),
    "ask_about_cost": (),
    "ask_about_policy": (),
    "stop": ("stop_reason",),
}
BUYER_ACTIONS = frozenset(BUYER_ACTION_PAYLOAD_FIELDS)

FAILURE_TYPES = frozenset(
    {
        "none",
        "tool_action_mismatch",
        "missing_confirmation",
        "missing_clarification",
        "constraint_tracking_failure",
        "communicate_failure",
        "policy_violation",
        "environment_state_error",
        "other",
    }
)

_TOP_LEVEL_FIELDS = {
    "diagnosis",
    "target_skill",
    "next_move",
    "payload",
    "predicted_learning_progress",
    "stop",
}
_DIAGNOSIS_FIELDS = {"failure_type", "evidence_turns"}
_FORBIDDEN_PAYLOAD_KEYS = {
    "system_prompt",
    "developer_prompt",
    "teacher_hint",
    "verifier",
    "reward",
}


@dataclass(frozen=True)
class BuyerDiagnosis:
    failure_type: str
    evidence_turns: tuple[int, ...]

    def to_dict(self) -> dict:
        return {
            "failure_type": self.failure_type,
            "evidence_turns": list(self.evidence_turns),
        }


@dataclass(frozen=True)
class BuyerPlan:
    """Strict private action plan emitted by the trainable Buyer Planner."""

    diagnosis: BuyerDiagnosis
    target_skill: str
    next_move: str
    payload: dict[str, Any]
    predicted_learning_progress: float
    stop: bool

    @classmethod
    def from_text(cls, text: str) -> "BuyerPlan":
        if not text or not text.strip():
            raise ValueError("Buyer private plan is empty")
        candidate = text.strip()
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            candidate = "\n".join(lines)
        candidate = candidate.strip()
        if not candidate.startswith("{") or not candidate.endswith("}"):
            raise ValueError("Buyer private plan must contain only one JSON object")
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            payload = repair_json(candidate, return_objects=True)
        if not isinstance(payload, dict):
            raise ValueError("Buyer private plan must be one JSON object")
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BuyerPlan":
        unknown = set(value) - _TOP_LEVEL_FIELDS
        missing = _TOP_LEVEL_FIELDS - set(value)
        if unknown:
            raise ValueError(f"Unknown Buyer plan fields: {sorted(unknown)}")
        if missing:
            raise ValueError(f"Missing Buyer plan fields: {sorted(missing)}")

        diagnosis = value["diagnosis"]
        if not isinstance(diagnosis, dict):
            raise ValueError("diagnosis must be an object")
        diagnosis_unknown = set(diagnosis) - _DIAGNOSIS_FIELDS
        diagnosis_missing = _DIAGNOSIS_FIELDS - set(diagnosis)
        if diagnosis_unknown or diagnosis_missing:
            raise ValueError(
                "diagnosis fields must be exactly failure_type and evidence_turns"
            )
        failure_type = diagnosis["failure_type"]
        if failure_type not in FAILURE_TYPES:
            raise ValueError(f"Unknown failure_type: {failure_type!r}")
        evidence_turns = diagnosis["evidence_turns"]
        if (
            not isinstance(evidence_turns, list)
            or any(type(turn) is not int or turn < 0 for turn in evidence_turns)
        ):
            raise ValueError("evidence_turns must be a list of non-negative integers")

        target_skill = value["target_skill"]
        if not isinstance(target_skill, str) or not target_skill.strip():
            raise ValueError("target_skill must be a non-empty string")
        next_move = value["next_move"]
        if next_move not in BUYER_ACTIONS:
            raise ValueError(f"Unknown next_move: {next_move!r}")
        action_payload = value["payload"]
        if not isinstance(action_payload, dict):
            raise ValueError("payload must be an object")
        forbidden = _find_forbidden_keys(action_payload)
        if forbidden:
            raise ValueError(f"Forbidden private-plan payload keys: {sorted(forbidden)}")

        predicted_gain = value["predicted_learning_progress"]
        if type(predicted_gain) not in {int, float} or not 0 <= predicted_gain <= 1:
            raise ValueError(
                "predicted_learning_progress must be a number in [0, 1]"
            )
        stop = value["stop"]
        if type(stop) is not bool:
            raise ValueError("stop must be a boolean")
        if stop != (next_move == "stop"):
            raise ValueError("stop must be true exactly when next_move is 'stop'")

        return cls(
            diagnosis=BuyerDiagnosis(failure_type, tuple(evidence_turns)),
            target_skill=target_skill.strip(),
            next_move=next_move,
            payload=deepcopy(action_payload),
            predicted_learning_progress=float(predicted_gain),
            stop=stop,
        )

    def to_dict(self) -> dict:
        return {
            "diagnosis": self.diagnosis.to_dict(),
            "target_skill": self.target_skill,
            "next_move": self.next_move,
            "payload": deepcopy(self.payload),
            "predicted_learning_progress": self.predicted_learning_progress,
            "stop": self.stop,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def planner_system_prompt(reference_prompt: str, tool_names=()) -> str:
        actions = ", ".join(sorted(BUYER_ACTIONS))
        failures = ", ".join(sorted(FAILURE_TYPES))
        tools = ", ".join(sorted(tool_names)) or "none"
        payload_contract = "; ".join(
            f'{action} -> '
            + json.dumps(
                {
                    field: (
                        {}
                        if field == "arguments"
                        else f"<{field}>"
                    )
                    for field in fields
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for action, fields in BUYER_ACTION_PAYLOAD_FIELDS.items()
        )
        return (
            f"{reference_prompt}\n\n"
            "You are the private Buyer Planner. Return exactly one JSON object and no "
            "public user prose or tool call. The object is private and will be rendered "
            "by a frozen renderer. Do not change the hidden scenario or invent facts.\n"
            "Required schema: {\"diagnosis\":{\"failure_type\":str,"
            "\"evidence_turns\":[int]},\"target_skill\":str,\"next_move\":str,"
            "\"payload\":{},\"predicted_learning_progress\":number,\"stop\":bool}.\n"
            f"Allowed next_move values: {actions}.\n"
            f"Allowed failure_type values: {failures}.\n"
            f"Available user tools: {tools}.\n"
            "Payload must use exactly the listed keys and no others: "
            f"{payload_contract}.\n"
            "target_skill names the Student capability you want to probe; next_move "
            "names the literal public user behavior to render. Do not copy "
            "target_skill into next_move unless that is literally the user's next "
            "behavior. For an initial customer request or an ordinary natural-language "
            "reply, use next_move=answer_normally and put the full utterance only in "
            "payload.answer. ask_about_cost and ask_about_policy are fixed generic "
            "follow-up actions and therefore require an empty payload. Never place the "
            "answer key under any action other than answer_normally.\n"
            "Example initial request: "
            '{"diagnosis":{"failure_type":"none","evidence_turns":[]},'
            '"target_skill":"constraint_tracking","next_move":"answer_normally",'
            '"payload":{"answer":"I need help with my reservation."},'
            '"predicted_learning_progress":0.1,"stop":false}.\n'
            "Allowed stop_reason values: task_complete, scenario_requires_stop, "
            "user_abandoned, environment_blocked."
        )


def _find_forbidden_keys(value: Any) -> set[str]:
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in _FORBIDDEN_PAYLOAD_KEYS:
                found.add(str(key))
            found.update(_find_forbidden_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_find_forbidden_keys(item))
    return found
