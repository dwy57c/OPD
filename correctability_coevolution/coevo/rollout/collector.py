from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os

from tau2.data_model.message import AssistantMessage

from coevo.artifacts import artifact_metadata, canonical_hash
from coevo.environment import Tau2Environment
from coevo.hints import HintLevel
from coevo.environment.tau2 import dump_messages, load_messages
from coevo.models import format_teacher_system_prompt_with_hint
from coevo.rollout.views import (
    buyer_view,
    student_view,
    swift_on_policy_prompt_messages,
)
from coevo.scoring import (
    TeacherTargetBuilder,
    TeacherTargetLabeler,
    TeacherTargetRecord,
)


class NaturalDecisionCollector:
    """Collect one canonical Teacher target at each complete Student action."""

    def __init__(
        self,
        environment: Tau2Environment,
        labeler: TeacherTargetLabeler,
        max_decisions: int,
        *,
        target_builder: TeacherTargetBuilder | None = None,
    ):
        self.environment = environment
        self.labeler = labeler
        self.max_decisions = max_decisions
        self._student_system_prompt = None
        self._student_tool_schemas = None
        self._target_builder = target_builder

    def _builder(self) -> TeacherTargetBuilder:
        if self._target_builder is None:
            self._target_builder = TeacherTargetBuilder(self.environment.config)
        return self._target_builder

    def _system_prompt(self) -> str:
        if self._student_system_prompt is None:
            student = self.environment.policies.student(
                self.environment.fresh_environment()
            )
            self._student_system_prompt = student.system_prompt
            self._student_tool_schemas = [
                deepcopy(tool.openai_schema)
                for tool in (getattr(student, "tools", None) or [])
            ]
        return self._student_system_prompt

    def _tool_schemas(self) -> list[dict]:
        self._system_prompt()
        return deepcopy(self._student_tool_schemas or [])

    def _materialize_target(self, decision: dict) -> dict:
        candidate = dict(decision)
        config = getattr(self.environment, "config", None)
        level = getattr(config, "hint_level", HintLevel.L3_ORACLE)
        candidate["hint_level"] = HintLevel.parse(level).value
        history_before = load_messages(candidate["history_before"])
        teacher_action = AssistantMessage.model_validate(candidate["teacher_action"])
        visible_messages = student_view(
            self._system_prompt(), [*history_before, teacher_action]
        )
        hinted_messages = student_view(
            format_teacher_system_prompt_with_hint(
                self._system_prompt(), candidate.get("teacher_hint")
            ),
            [*history_before, teacher_action],
        )
        hint_hash = canonical_hash(candidate.get("teacher_hint") or {})
        try:
            target = self._builder().build(
                student_visible_messages=visible_messages,
                hinted_teacher_messages=hinted_messages,
                teacher_action=visible_messages[-1],
                state_hash=str(candidate["state_hash"]),
                teacher_hint_hash=hint_hash,
                tool_schemas=self._tool_schemas(),
            )
        except Exception as error:
            candidate.update(
                {
                    "student_eligible": False,
                    "student_rejection_reason": (
                        f"{type(error).__name__}: {error}"
                    ),
                    "teacher_hint_hash": hint_hash,
                }
            )
            return candidate
        candidate.update(
            {
                "student_eligible": True,
                "student_rejection_reason": None,
                "teacher_hint_hash": hint_hash,
                "teacher_target_record": target.to_dict(),
                "raw_teacher_target_hash": target.raw_teacher_target_hash,
                "teacher_target_hash": target.teacher_target_hash,
                "teacher_action_hash": target.teacher_action_hash,
                "target_token_count": sum(target.target_loss_mask),
            }
        )
        return candidate

    def collect_one(self, seed: int | None = None) -> dict:
        env = self.environment
        initial_history = env.initial_history()
        initial_size = len(initial_history)
        trajectory_seed = env.config.seed if seed is None else seed
        orchestrator = env.orchestrator(
            initial_history, "student", seed=trajectory_seed
        )
        orchestrator.initialize()
        while not orchestrator.done:
            orchestrator.step()
            if orchestrator.step_count >= env.config.branch_max_steps:
                break

        trunk = orchestrator.get_trajectory()
        message_indexes = [
            index
            for index in range(initial_size, len(trunk))
            if isinstance(trunk[index], AssistantMessage)
        ]
        score_workers = int(os.getenv("COEVO_TURN_SCORE_WORKERS", "1"))
        if score_workers > 1 and self.max_decisions == 0:
            with ThreadPoolExecutor(max_workers=score_workers) as executor:
                scored = list(
                    executor.map(
                        lambda index: self.labeler.score_decision(trunk, index),
                        message_indexes,
                    )
                )
            decisions = [self._materialize_target(value) for value in scored]
        else:
            decisions = []
            for message_index in message_indexes:
                decisions.append(
                    self._materialize_target(
                        self.labeler.score_decision(trunk, message_index)
                    )
                )
                if self.max_decisions and len(decisions) >= self.max_decisions:
                    break
        return {
            **artifact_metadata(env.config),
            "domain": env.config.domain,
            "task_split": env.config.task_split,
            "task_id": env.task.id,
            "seed": trajectory_seed,
            "trunk": dump_messages(trunk),
            "student_decisions": decisions,
            "teacher_target_cache": self._builder().cache_stats,
        }

    def student_rows(self, record: dict) -> list[dict]:
        rows = []
        for turn in record["student_decisions"]:
            if not turn.get("student_eligible"):
                continue
            target = TeacherTargetRecord.from_dict(turn["teacher_target_record"])
            rows.append(
                {
                    **artifact_metadata(self.environment.config),
                    "messages": swift_on_policy_prompt_messages(
                        deepcopy_messages(target.student_visible_messages)
                    ),
                    "tools": self._tool_schemas(),
                    "training_target": "natural_hint_on_policy_jsd",
                    "hint_level": self.environment.config.hint_level.value,
                    "teacher_target_record": target.to_dict(),
                    "state_hash": target.state_hash,
                    "teacher_action_hash": target.teacher_action_hash,
                    "raw_teacher_target_hash": target.raw_teacher_target_hash,
                    "teacher_target_hash": target.teacher_target_hash,
                    "target_token_count": sum(target.target_loss_mask),
                    "domain": self.environment.config.domain,
                    "task_split": self.environment.config.task_split,
                    "task_id": self.environment.task.id,
                }
            )
        return rows

    def buyer_row(self) -> dict:
        env = self.environment
        buyer = env.buyer_reference_policy()
        system_prompt = buyer.system_prompt
        if env.config.buyer_plan_mode == "structured":
            from coevo.models import BuyerPlan, available_tool_names

            system_prompt = BuyerPlan.planner_system_prompt(
                system_prompt, available_tool_names(buyer.tools or [])
            )
        history = env.initial_history()
        row = {
            **artifact_metadata(env.config),
            "messages": buyer_view(system_prompt, history),
            "domain": env.config.domain,
            "task_split": env.config.task_split,
            "task_id": env.task.id,
            "tau_history": dump_messages(history),
            "buyer_plan_mode": env.config.buyer_plan_mode,
            "hint_level": env.config.hint_level.value,
        }
        if buyer.tools and env.config.buyer_plan_mode == "legacy":
            row["tools"] = [tool.openai_schema for tool in buyer.tools]
        return row


def deepcopy_messages(messages) -> list[dict]:
    return deepcopy(list(messages))
