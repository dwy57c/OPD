import asyncio
import os
from dataclasses import replace
from threading import Lock

from swift.infer_engine.protocol import RolloutOutput
from swift.rollout.multi_turn import MultiTurnScheduler
from tau2.data_model.message import AssistantMessage, ToolMessage

from coevo.artifacts import canonical_hash
from coevo.config import InfraConfig
from coevo.environment import Tau2Environment
from coevo.environment.tau2 import dump_messages, load_messages
from coevo.intervention import DecisionState
from coevo.models import (
    BuyerPlan,
    BuyerRenderContext,
    FrozenRenderer,
    format_teacher_system_prompt_with_hint,
)
from coevo.rewards import transition_validity
from coevo.rollout import build_teacher_target_labeler, student_view
from coevo.scoring import StageGapScorer


def visible_buyer_content(content: str | None) -> str:
    """Remove private Qwen thinking before exposing the public Buyer action."""
    if not content:
        return ""
    lowered = content.lower()
    parts = []
    cursor = 0
    while True:
        start = lowered.find("<think>", cursor)
        if start < 0:
            parts.append(content[cursor:])
            break
        parts.append(content[cursor:start])
        end = lowered.find("</think>", start + len("<think>"))
        if end < 0:
            break
        cursor = end + len("</think>")
    visible = "".join(parts)
    while True:
        close = visible.lower().find("</think>")
        if close < 0:
            break
        visible = visible[:close] + visible[close + len("</think>") :]
    return visible.strip()


class Tau2BuyerScheduler(MultiTurnScheduler):
    """Reward progress toward frozen S_k+skill demonstrations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_config = InfraConfig.from_env()
        self.renderer = FrozenRenderer()
        self._contexts = {}
        self._contexts_lock = Lock()
        self._stage_scorer = None

    def _scorer(self) -> StageGapScorer:
        if self._stage_scorer is None:
            scorer = StageGapScorer(self.base_config)
            scorer.validate_checkpoint_pair()
            self._stage_scorer = scorer
        return self._stage_scorer

    def _context(self, data: dict):
        domain = str(data.get("domain") or self.base_config.domain)
        task_split = str(data.get("task_split") or self.base_config.task_split)
        task_id = str(data.get("task_id") or self.base_config.task_id)
        key = (domain, task_split, task_id)
        with self._contexts_lock:
            context = self._contexts.get(key)
            if context is None:
                environment = Tau2Environment(
                    replace(
                        self.base_config,
                        domain=domain,
                        task_split=task_split,
                        task_id=task_id,
                    )
                )
                context = (
                    environment,
                    build_teacher_target_labeler(environment),
                )
                self._contexts[key] = context
        return context

    @staticmethod
    def _append_buyer_response(messages: list[dict], response_choice) -> None:
        message = response_choice.message
        row = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            row["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in message.tool_calls
            ]
        messages.append(row)

    def _rollout_infos(self, data: dict) -> dict:
        scoring_errors = list(data.get("scoring_errors", []))
        validity = float(
            float(data.get("trajectory_validity", 1.0)) > 0
            and not scoring_errors
        )
        rows = list(data.get("stage_progress_decisions", []))
        decision_rewards = [float(row["decision_reward"]) for row in rows]
        raw_reward = (
            sum(decision_rewards) / len(decision_rewards)
            if decision_rewards
            else 0.0
        )

        def values(field: str) -> list:
            return [row[field] for row in rows if field in row]

        def flattened(field: str) -> list[float]:
            result = []
            for value in values(field):
                if isinstance(value, list):
                    result.extend(float(item) for item in value)
                else:
                    result.append(float(value))
            return result

        info = {
            "buyer_reward": validity * raw_reward,
            "reward_source": "stage_learning_progress",
            "trajectory_validity": validity,
            "validity": validity,
            "decision_count": int(data.get("decision_count", 0)),
            "plan_errors": list(data.get("buyer_plan_errors", [])),
            "previous_gaps": flattened("previous_gap"),
            "current_gaps": flattened("current_gap"),
            "learning_progresses": flattened("learning_progress"),
            "positive_learning_progresses": flattened(
                "positive_learning_progress"
            ),
            "decision_rewards": decision_rewards,
            "checkpoint_previous": self.base_config.previous_policy_checkpoint,
            "checkpoint_current": self.base_config.current_policy_checkpoint,
            "checkpoint_teacher_anchor": (
                self.base_config.teacher_anchor_checkpoint
            ),
            "teacher_target_hashes": values("teacher_target_hash"),
            "raw_teacher_target_hashes": values("raw_teacher_target_hash"),
            "skill_contrast_scores": flattened("skill_contrast_scores"),
            "skill_gate_values": flattened("skill_gate_values"),
            "sharpening_temperatures": flattened("sharpening_temperatures"),
            "raw_teacher_entropies": flattened("raw_teacher_entropy"),
            "sharpened_teacher_entropies": flattened(
                "sharpened_teacher_entropy"
            ),
            "scoring_errors": scoring_errors,
        }
        if os.getenv("COEVO_CAPTURE_FULL_TRACE") == "1":
            info.update(
                {
                    "domain": str(data.get("domain") or self.base_config.domain),
                    "task_split": str(
                        data.get("task_split") or self.base_config.task_split
                    ),
                    "task_id": str(data.get("task_id") or self.base_config.task_id),
                    "buyer_private_plans": list(
                        data.get("buyer_private_plans", [])
                    ),
                    "buyer_public_actions": list(
                        data.get("buyer_public_actions", [])
                    ),
                    "plan_action_consistency": list(
                        data.get("plan_action_consistency", [])
                    ),
                    "tau_history": list(data.get("tau_history", [])),
                    "stage_progress_decisions": rows,
                    "teacher_target_labels": list(
                        data.get("teacher_target_labels", [])
                    ),
                }
            )
        return info

    def _mark_truncated(self, infer_request) -> dict:
        infer_request.data_dict["trajectory_validity"] = 0.0
        return self._rollout_infos(infer_request.data_dict)

    def _decode_buyer_action(self, environment, response_choice, data: dict):
        content = visible_buyer_content(response_choice.message.content)
        if self.base_config.buyer_plan_mode == "legacy":
            return environment.buyer_message(content, response_choice.message.tool_calls)
        if response_choice.message.tool_calls:
            raise ValueError("Structured Buyer Planner must not emit public tool calls")
        plan = BuyerPlan.from_text(content)
        tool_names = (
            environment.available_user_tool_names()
            if hasattr(environment, "available_user_tool_names")
            else ()
        )
        scenario_text = (
            environment.buyer_scenario_text()
            if hasattr(environment, "buyer_scenario_text")
            else ""
        )
        public_action = self.renderer.render(
            plan,
            BuyerRenderContext(
                available_user_tools=tuple(tool_names),
                scenario_text=scenario_text,
                turn_index=len(data.get("buyer_private_plans", [])),
            ),
        )
        data.setdefault("buyer_private_plans", []).append(plan.to_dict())
        data.setdefault("buyer_public_actions", []).append(public_action.to_dict())
        data.setdefault("plan_action_consistency", []).append(1.0)
        return public_action.to_message()

    @staticmethod
    def _record_plan_error(data: dict, error: Exception) -> None:
        data["trajectory_validity"] = 0.0
        data.setdefault("buyer_plan_errors", []).append(
            {"type": type(error).__name__, "message": str(error)}
        )

    def _score_decision(self, environment, label) -> dict:
        result = label.to_dict()
        history_before = load_messages(result["history_before"])
        teacher_action = AssistantMessage.model_validate(result["teacher_action"])
        student = environment.policies.student(environment.fresh_environment())
        visible_messages = student_view(
            student.system_prompt, [*history_before, teacher_action]
        )
        hinted_messages = student_view(
            format_teacher_system_prompt_with_hint(
                student.system_prompt, result.get("teacher_hint")
            ),
            [*history_before, teacher_action],
        )
        score = self._scorer().score(
            student_visible_messages=visible_messages,
            hinted_teacher_messages=hinted_messages,
            teacher_action=visible_messages[-1],
            state_hash=str(result["state_hash"]),
            teacher_hint_hash=canonical_hash(result.get("teacher_hint") or {}),
        )
        return score.to_dict()

    def _apply_buyer_action(
        self, infer_request, response_choice, append_observation: bool
    ) -> tuple[dict, bool]:
        data = infer_request.data_dict
        if "tau_history" not in data:
            raise KeyError(
                "tau_history is missing; train with --vllm_server_pass_dataset true"
            )
        environment, labeler = self._context(data)
        history = load_messages(data["tau_history"])
        validity = float(data.get("trajectory_validity", 1.0))
        try:
            buyer_message = self._decode_buyer_action(
                environment, response_choice, data
            )
        except (TypeError, ValueError) as error:
            self._record_plan_error(data, error)
            return self._rollout_infos(data), True
        history.append(buyer_message)

        new_decisions = 0
        if not buyer_message.is_tool_call() and not buyer_message.content:
            validity = 0.0
            finished = True
        else:
            finished = environment.buyer_stopped(buyer_message)
        if not finished and buyer_message.is_tool_call():
            transition_start = len(history)
            history = environment.execute_user_tools(history)
            transition = history[transition_start:]
            validity *= transition_validity(transition)
            if append_observation:
                for observation in transition:
                    if isinstance(observation, ToolMessage):
                        infer_request.messages.append(
                            {
                                "role": "tool",
                                "content": observation.content,
                                "tool_call_id": observation.id,
                            }
                        )
        elif not finished:
            transition_start = len(history)
            history = environment.advance_student(history)
            all_indexes = [
                index
                for index in range(transition_start, len(history))
                if isinstance(history[index], AssistantMessage)
                and (history[index].content or history[index].tool_calls)
            ]
            if not all_indexes:
                validity = 0.0
                finished = True
            else:
                indexes = all_indexes
                if self.base_config.max_teacher_targets:
                    indexes = indexes[: self.base_config.max_teacher_targets]
                for message_index in indexes:
                    new_decisions += 1
                    try:
                        label = labeler.run(
                            DecisionState.from_history(history, message_index)
                        )
                        scored = self._score_decision(environment, label)
                    except Exception as error:
                        validity = 0.0
                        data.setdefault("scoring_errors", []).append(
                            {
                                "message_index": message_index,
                                "type": type(error).__name__,
                                "message": str(error),
                            }
                        )
                    else:
                        data.setdefault("stage_progress_decisions", []).append(scored)
                        data.setdefault("teacher_target_labels", []).append(
                            label.to_dict()
                        )
                if append_observation:
                    last = history[all_indexes[-1]]
                    infer_request.messages.append(
                        {"role": "user", "content": last.content or ""}
                    )

        data["tau_history"] = dump_messages(history)
        data["trajectory_validity"] = validity
        data["decision_count"] = int(data.get("decision_count", 0)) + new_decisions
        return self._rollout_infos(data), finished

    async def run(self, infer_request, request_config, **kwargs):
        """Execute and score all generated Buyer actions, including the final one."""
        if not self.max_turns or self.max_turns < 1:
            raise ValueError("tau2_buyer requires --max_turns >= 1")
        response_token_ids = []
        response_loss_mask = []
        rollout_logprobs = []
        complete_logprobs = True
        rollout_infos = self._rollout_infos(infer_request.data_dict)
        response = None
        turns_executed = 0
        rollout_finished = False
        for current_turn in range(1, self.max_turns + 1):
            response = await self.infer_engine.infer_async(
                infer_request, request_config, **kwargs
            )
            choice = response.choices[0]
            token_ids = list(choice.token_ids or [])
            response_token_ids.append(token_ids)
            response_loss_mask.append([1] * len(token_ids))
            logprobs = self._extract_logprobs_from_choice(choice)
            if len(logprobs) == len(token_ids):
                rollout_logprobs.append(logprobs)
            else:
                complete_logprobs = False
            self._append_buyer_response(infer_request.messages, choice)
            turns_executed = current_turn
            if choice.finish_reason == "length":
                rollout_infos = self._mark_truncated(infer_request)
                break
            rollout_infos, finished = await asyncio.to_thread(
                self._apply_buyer_action,
                infer_request,
                choice,
                current_turn < self.max_turns,
            )
            if finished:
                rollout_finished = True
                break
        if response is None:
            raise RuntimeError("Buyer rollout produced no response")
        if not rollout_finished:
            rollout_infos = self._mark_truncated(infer_request)
        rollout_infos["num_turns"] = turns_executed
        return RolloutOutput(
            response=response,
            messages=infer_request.messages,
            response_token_ids=response_token_ids,
            response_loss_mask=response_loss_mask,
            rollout_infos=rollout_infos,
            rollout_logprobs=rollout_logprobs if complete_logprobs else [],
        )
