from dataclasses import asdict, dataclass
import json
from typing import Callable

from openai import OpenAI
from tau2.data_model.message import Message

from coevo.config import ModelEndpoint
from coevo.cutoff.boundaries import CutoffCandidate
from coevo.environment.tau2 import dump_messages
from coevo.models.hinted_teacher import TeacherHintResult


@dataclass(frozen=True)
class SelectedCutoff:
    candidate: CutoffCandidate
    reason: str

    def to_dict(self) -> dict:
        value = asdict(self)
        value["candidate"] = self.candidate.to_dict()
        return value


class TeacherCutoffSelector:
    """Ask the shared policy plus private hint to rank semantic boundaries."""

    def __init__(
        self,
        endpoint: ModelEndpoint,
        hint_provider: Callable[[list[Message]], TeacherHintResult],
        top_k: int,
        seed: int,
    ):
        self.endpoint = endpoint
        self.hint_provider = hint_provider
        self.top_k = top_k
        self.seed = seed
        self.client = OpenAI(
            base_url=endpoint.base_url.rstrip("/") + "/v1", api_key="EMPTY"
        )

    def select(
        self,
        history_before_turn: list[Message],
        student_output: str,
        candidates: list[CutoffCandidate],
    ) -> tuple[list[SelectedCutoff], TeacherHintResult | None]:
        if not candidates:
            return [], None
        hint_result = self.hint_provider(history_before_turn)
        count = min(self.top_k, len(candidates))
        ids = [candidate.candidate_id for candidate in candidates]
        candidate_rows = [candidate.to_dict() for candidate in candidates]
        prompt = {
            "history_before_student_turn": dump_messages(history_before_turn),
            "student_complete_turn": student_output,
            "private_teacher_hint": hint_result.hint,
            "candidate_boundaries": candidate_rows,
            "instruction": (
                f"Select exactly {count} candidate IDs where takeover is most likely "
                "to reveal a teacher-correctable Student failure. Rank best first."
            ),
        }
        schema = {
            "name": "teacher_cutoff_selection",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "selected": {
                        "type": "array",
                        "minItems": count,
                        "maxItems": count,
                        "items": {
                            "type": "object",
                            "properties": {
                                "candidate_id": {"type": "integer", "enum": ids},
                                "reason": {"type": "string"},
                            },
                            "required": ["candidate_id", "reason"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["selected"],
                "additionalProperties": False,
            },
        }
        response = self.client.chat.completions.create(
            model=self.endpoint.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rank semantic cutoff candidates for on-policy distillation. "
                        "You are the same policy model used by Student. Use the private "
                        "closed-model hint only for ranking. Return the requested JSON."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            response_format={"type": "json_schema", "json_schema": schema},
            temperature=0,
            seed=self.seed,
            max_tokens=512,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        payload = json.loads(response.choices[0].message.content)
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        selected = payload["selected"]
        selected_ids = [item["candidate_id"] for item in selected]
        if len(selected_ids) != count or len(set(selected_ids)) != count:
            raise ValueError(f"Teacher returned invalid cutoff IDs: {selected_ids}")
        return [
            SelectedCutoff(by_id[item["candidate_id"]], item["reason"])
            for item in selected
        ], hint_result
