import asyncio
from dataclasses import replace
from threading import Lock

from swift.infer_engine.protocol import RolloutOutput
from swift.rollout.multi_turn import MultiTurnScheduler
from tau2.data_model.message import AssistantMessage, ToolMessage

from coevo.config import InfraConfig
from coevo.environment import Tau2Environment
from coevo.environment.tau2 import dump_messages, load_messages
from coevo.intervention import DecisionState
from coevo.models import BuyerPlan, BuyerRenderContext, FrozenRenderer
from coevo.rewards import transition_validity
from coevo.rollout import build_action_branch_runner


def visible_buyer_content(content: str | None) -> str:
    """Remove Qwen-style private thinking before exposing a Buyer turn.

    Swift normally returns thinking in ``reasoning_content`` and only the final
    answer in ``content``. Some rollout backends instead return the serialized
    ``<think>...</think>`` block in ``content``. Handle both representations so
    private reasoning can remain in the sampled token sequence without entering
    the tau2 history seen by Student, Teacher continuations, or scorers.

    An unclosed thinking block is treated as private through end-of-string. This
    deliberately produces an empty visible action when the model never reaches
    its answer, allowing the existing validity gate to reject that rollout.
    """
    if not content:
        return ""

    lower_content = content.lower()
    visible_parts = []
    cursor = 0
    while True:
        think_start = lower_content.find("<think>", cursor)
        if think_start < 0:
            visible_parts.append(content[cursor:])
            break
        visible_parts.append(content[cursor:think_start])
        think_end = lower_content.find("</think>", think_start + len("<think>"))
        if think_end < 0:
            break
        cursor = think_end + len("</think>")

    # A few OpenAI-compatible backends omit the opening tag because the template
    # supplied it as a generation prefix, but still return the closing tag.
    visible = "".join(visible_parts)
    while True:
        orphan_close = visible.lower().find("</think>")
        if orphan_close < 0:
            break
        visible = visible[:orphan_close] + visible[orphan_close + len("</think>") :]
    return visible.strip()


class Tau2BuyerScheduler(MultiTurnScheduler):
    """Run every Buyer action against the matching task environment before reward."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_config = InfraConfig.from_env()
        self.renderer = FrozenRenderer()
        self._contexts = {}
        self._contexts_lock = Lock()

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
                context = (environment, build_action_branch_runner(environment))
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

    @staticmethod
    def _rollout_infos(data: dict) -> dict:
        turn_scores = list(data.get("turn_intervention_advantages", []))
        validity = float(data.get("trajectory_validity", 1.0))
        mean_advantage = sum(turn_scores) / len(turn_scores) if turn_scores else 0.0
        fast_reward = validity * mean_advantage
        if "opd_utility_gain" in data:
            raw_reward = float(data["opd_utility_gain"])
            reward_source = str(data.get("opd_utility_source", "shadow_opd"))
            reward = validity * raw_reward
        else:
            raw_reward = sum(turn_scores) / len(turn_scores) if turn_scores else 0.0
            reward_source = "intervention_advantage"
            reward = fast_reward
        return {
            "buyer_reward": reward,
            "reward_source": reward_source,
            "raw_reward": raw_reward,
            "fast_intervention_reward": fast_reward,
            "mean_intervention_advantage": mean_advantage,
            "turn_intervention_advantages": turn_scores,
            "validity": validity,
            "decision_count": int(data.get("decision_count", 0)),
            "plan_errors": list(data.get("buyer_plan_errors", [])),
        }

    def _mark_truncated(self, infer_request) -> dict:
        data = infer_request.data_dict
        data["trajectory_validity"] = 0.0
        return self._rollout_infos(data)

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

    def _apply_buyer_action(
        self, infer_request, response_choice, append_observation: bool
    ) -> tuple[dict, bool]:
        data = infer_request.data_dict
        if "tau_history" not in data:
            raise KeyError(
                "tau_history is missing; start the rollout scheduler on the server and "
                "train with --vllm_server_pass_dataset true"
            )

        environment, branch_runner = self._context(data)
        history = load_messages(data["tau_history"])
        turn_scores = list(data.get("turn_intervention_advantages", []))
        validity_score = float(data.get("trajectory_validity", 1.0))
        try:
            buyer_message = self._decode_buyer_action(
                environment, response_choice, data
            )
        except (TypeError, ValueError) as error:
            self._record_plan_error(data, error)
            return self._rollout_infos(data), True
        history.append(buyer_message)

        decision_count = 0
        if not buyer_message.is_tool_call() and not buyer_message.content:
            validity_score = 0.0
            finished = True
        else:
            finished = environment.buyer_stopped(buyer_message)
        if not finished and buyer_message.is_tool_call():
            transition_start = len(history)
            history = environment.execute_user_tools(history)
            transition = history[transition_start:]
            validity_score *= transition_validity(transition)
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
            # Student errors are learning targets, not Buyer invalidity. Only a
            # Buyer-issued user tool transition is validity-gated above.
            all_decision_indexes = [
                index
                for index in range(transition_start, len(history))
                if isinstance(history[index], AssistantMessage)
                and (history[index].content or history[index].tool_calls)
            ]
            if not all_decision_indexes:
                validity_score = 0.0
                finished = True
            else:
                decision_indexes = all_decision_indexes
                max_decisions = self.base_config.max_intervention_decisions
                if max_decisions:
                    decision_indexes = decision_indexes[:max_decisions]
                for message_index in decision_indexes:
                    result = branch_runner.run(
                        DecisionState.from_history(history, message_index)
                    )
                    result_row = result.to_dict()
                    advantage = float(result_row["intervention_advantage"])
                    turn_scores.append(advantage)
                    data.setdefault("interventions", []).append(result_row)
                    decision_count += 1
                if append_observation:
                    message_index = all_decision_indexes[-1]
                    infer_request.messages.append(
                        {"role": "user", "content": history[message_index].content}
                    )

        data["tau_history"] = dump_messages(history)
        data["turn_intervention_advantages"] = turn_scores
        data["trajectory_validity"] = validity_score
        data["decision_count"] = int(data.get("decision_count", 0)) + decision_count
        return self._rollout_infos(data), finished

    async def run(self, infer_request, request_config, **kwargs):
        """Execute and score all generated Buyer actions, including the final one."""
        if not self.max_turns or self.max_turns < 1:
            raise ValueError("tau2_buyer requires --max_turns >= 1 on swift rollout")

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
            response_choice = response.choices[0]
            token_ids = list(response_choice.token_ids or [])
            response_token_ids.append(token_ids)
            response_loss_mask.append([1] * len(token_ids))
            logprobs = self._extract_logprobs_from_choice(response_choice)
            if len(logprobs) == len(token_ids):
                rollout_logprobs.append(logprobs)
            else:
                complete_logprobs = False

            self._append_buyer_response(infer_request.messages, response_choice)
            turns_executed = current_turn
            if response_choice.finish_reason == "length":
                rollout_infos = self._mark_truncated(infer_request)
                break

            append_observation = current_turn < self.max_turns
            rollout_infos, finished = await asyncio.to_thread(
                self._apply_buyer_action,
                infer_request,
                response_choice,
                append_observation,
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
