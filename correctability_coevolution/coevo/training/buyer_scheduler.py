from swift.rollout.multi_turn import MultiTurnScheduler
from tau2.data_model.message import AssistantMessage

from coevo.config import InfraConfig
from coevo.environment import Tau2Environment
from coevo.environment.tau2 import dump_messages, load_messages
from coevo.rewards import trajectory_buyer_reward, transition_validity
from coevo.rollout import build_cutoff_scorer


class Tau2BuyerScheduler(MultiTurnScheduler):
    """Swift Buyer policy interacting with a fixed Student in a τ² environment."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.environment = Tau2Environment(InfraConfig.from_env())
        self.cutoff_scorer = build_cutoff_scorer(self.environment)

    def check_finished(self, infer_request, response_choice, current_turn):
        return current_turn >= self.max_turns or response_choice.finish_reason == "length"

    def step(self, infer_request, response_choice, current_turn):
        history = load_messages(infer_request.data_dict["tau_history"])
        turn_scores = list(infer_request.data_dict.get("turn_correctability", []))
        trajectory_validity = float(
            infer_request.data_dict.get("trajectory_validity", 1.0)
        )
        buyer_message = self.environment.buyer_message(
            response_choice.message.content,
            response_choice.message.tool_calls,
        )
        history.append(buyer_message)

        cutoff_count = 0
        if buyer_message.is_tool_call():
            history = self.environment.execute_user_tools(history)
            observation = history[-1]
            trajectory_validity *= transition_validity(history[-1:])
            infer_request.messages.append(
                {
                    "role": "tool",
                    "content": observation.content,
                    "tool_call_id": observation.id,
                }
            )
        else:
            cutoff_size = len(history)
            history = self.environment.advance_student(history)
            student_message = history[-1]
            validity = transition_validity(history[cutoff_size:])
            message_index = next(
                (
                    index
                    for index in range(len(history) - 1, cutoff_size - 1, -1)
                    if isinstance(history[index], AssistantMessage)
                    and history[index].content
                    and not history[index].tool_calls
                ),
                None,
            )
            scored_turn = (
                self.cutoff_scorer.score_turn(history, message_index)
                if message_index is not None
                else None
            )
            correctability = scored_turn["correctability"] if scored_turn else 0.0
            turn_scores.append(correctability)
            trajectory_validity *= validity
            cutoff_count = len(scored_turn["cutoffs"]) if scored_turn else 0
            infer_request.messages.append(
                {"role": "user", "content": student_message.content or ""}
            )

        infer_request.data_dict["tau_history"] = dump_messages(history)
        infer_request.data_dict["turn_correctability"] = turn_scores
        infer_request.data_dict["trajectory_validity"] = trajectory_validity
        trajectory_score = sum(turn_scores) / len(turn_scores) if turn_scores else 0.0
        rollout_infos = {
            "correctability_reward": trajectory_buyer_reward(
                turn_scores, trajectory_validity
            ),
            "trajectory_correctability": trajectory_score,
            "turn_correctability": turn_scores,
            "validity": trajectory_validity,
            "cutoff_count": cutoff_count,
        }
        token_ids = response_choice.token_ids or []
        return {
            "infer_request": infer_request,
            "response_token_ids": token_ids,
            "response_loss_mask": [1] * len(token_ids),
            "rollout_infos": rollout_infos,
        }
