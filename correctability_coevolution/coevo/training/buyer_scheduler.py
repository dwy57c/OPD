import asyncio
from dataclasses import replace
from threading import Lock

from swift.infer_engine.protocol import RolloutOutput
from swift.rollout.multi_turn import MultiTurnScheduler
from tau2.data_model.message import AssistantMessage, ToolMessage

from coevo.config import InfraConfig
from coevo.environment import Tau2Environment
from coevo.environment.tau2 import dump_messages, load_messages
from coevo.rewards import trajectory_buyer_reward, transition_validity
from coevo.rollout import build_cutoff_scorer


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
                context = (environment, build_cutoff_scorer(environment))
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
        turn_scores = list(data.get("turn_correctability", []))
        validity = float(data.get("trajectory_validity", 1.0))
        return {
            "correctability_reward": trajectory_buyer_reward(turn_scores, validity),
            "trajectory_correctability": (
                sum(turn_scores) / len(turn_scores) if turn_scores else 0.0
            ),
            "turn_correctability": turn_scores,
            "validity": validity,
            "cutoff_count": int(data.get("cutoff_count", 0)),
        }

    def _mark_truncated(self, infer_request) -> dict:
        data = infer_request.data_dict
        data["trajectory_validity"] = 0.0
        return self._rollout_infos(data)

    def _apply_buyer_action(
        self, infer_request, response_choice, append_observation: bool
    ) -> tuple[dict, bool]:
        data = infer_request.data_dict
        if "tau_history" not in data:
            raise KeyError(
                "tau_history is missing; start the rollout scheduler on the server and "
                "train with --vllm_server_pass_dataset true"
            )

        environment, cutoff_scorer = self._context(data)
        history = load_messages(data["tau_history"])
        turn_scores = list(data.get("turn_correctability", []))
        validity_score = float(data.get("trajectory_validity", 1.0))
        try:
            buyer_message = environment.buyer_message(
                visible_buyer_content(response_choice.message.content),
                response_choice.message.tool_calls,
            )
        except (TypeError, ValueError):
            data["trajectory_validity"] = 0.0
            return self._rollout_infos(data), True
        history.append(buyer_message)

        cutoff_count = 0
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
            transition = history[transition_start:]
            validity_score *= transition_validity(transition)
            message_index = next(
                (
                    index
                    for index in range(len(history) - 1, transition_start - 1, -1)
                    if isinstance(history[index], AssistantMessage)
                    and history[index].content
                    and not history[index].tool_calls
                ),
                None,
            )
            if message_index is None:
                validity_score = 0.0
                finished = True
            else:
                scored_turn = cutoff_scorer.score_turn(history, message_index)
                correctability = scored_turn["correctability"] if scored_turn else 0.0
                turn_scores.append(correctability)
                cutoff_count = len(scored_turn["cutoffs"]) if scored_turn else 0
                if append_observation:
                    infer_request.messages.append(
                        {"role": "user", "content": history[message_index].content}
                    )

        data["tau_history"] = dump_messages(history)
        data["turn_correctability"] = turn_scores
        data["trajectory_validity"] = validity_score
        data["cutoff_count"] = int(data.get("cutoff_count", 0)) + cutoff_count
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
