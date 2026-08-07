from concurrent.futures import ThreadPoolExecutor
import os

from tau2.data_model.message import AssistantMessage

from coevo.environment import Tau2Environment
from coevo.environment.tau2 import dump_messages
from coevo.intervention import ActionBranchRunner
from coevo.rollout.views import buyer_view, student_view


class NaturalDecisionCollector:
    """Collect a trunk and branch at every complete natural Student action."""

    def __init__(
        self,
        environment: Tau2Environment,
        scorer: ActionBranchRunner,
        max_decisions: int,
    ):
        self.environment = environment
        self.scorer = scorer
        self.max_decisions = max_decisions

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
            message_index
            for message_index in range(initial_size, len(trunk))
            if isinstance(trunk[message_index], AssistantMessage)
        ]
        score_workers = int(os.getenv("COEVO_TURN_SCORE_WORKERS", "1"))
        if score_workers > 1 and self.max_decisions == 0:
            # Every continuation constructs a fresh tau2 environment. Scoring
            # independent completed turns concurrently therefore preserves rollout
            # semantics while allowing vLLM to batch Buyer and policy requests.
            with ThreadPoolExecutor(max_workers=score_workers) as executor:
                scored_decisions = list(
                    executor.map(
                        lambda index: self.scorer.score_decision(trunk, index),
                        message_indexes,
                    )
                )
            student_decisions = [
                decision for decision in scored_decisions if decision is not None
            ]
        else:
            student_decisions = []
            for message_index in message_indexes:
                scored = self.scorer.score_decision(trunk, message_index)
                if scored is not None:
                    student_decisions.append(scored)
                if (
                    self.max_decisions > 0
                    and len(student_decisions) >= self.max_decisions
                ):
                    break
        return {
            "domain": env.config.domain,
            "task_split": env.config.task_split,
            "task_id": env.task.id,
            "seed": trajectory_seed,
            "trunk": dump_messages(trunk),
            "student_decisions": student_decisions,
        }

    def student_rows(self, record: dict) -> list[dict]:
        env = self.environment
        raw_environment = env.fresh_environment()
        student = env.policies.student(raw_environment)
        rows = []
        for turn in record["student_decisions"]:
            from coevo.environment.tau2 import load_messages

            history_before = load_messages(turn["history_before"])
            advantage = float(turn["intervention_advantage"])
            if advantage <= 0:
                continue
            teacher_action = AssistantMessage.model_validate(turn["teacher_action"])
            student_action = AssistantMessage.model_validate(turn["student_action"])
            repaired_messages = student_view(
                student.system_prompt,
                [*history_before, teacher_action],
            )
            original_messages = student_view(
                student.system_prompt,
                [*history_before, student_action],
            )
            rows.append(
                {
                    "messages": repaired_messages,
                    "original_branch_messages": original_messages,
                    "training_target": "repair_then_distill",
                    "intervention_advantage": advantage,
                    "student_value": turn["student_value"],
                    "teacher_value": turn["teacher_value"],
                    "state_hash": turn["state_hash"],
                    "sample_hash": turn["sample_hash"],
                    "teacher_hint": turn.get("teacher_hint"),
                    "domain": env.config.domain,
                    "task_split": env.config.task_split,
                    "task_id": env.task.id,
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
            "messages": buyer_view(system_prompt, history),
            "domain": env.config.domain,
            "task_split": env.config.task_split,
            "task_id": env.task.id,
            "tau_history": dump_messages(history),
            "buyer_plan_mode": env.config.buyer_plan_mode,
        }
        if buyer.tools and env.config.buyer_plan_mode == "legacy":
            row["tools"] = [tool.openai_schema for tool in buyer.tools]
        return row
