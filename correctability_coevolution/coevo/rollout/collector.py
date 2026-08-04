from tau2.data_model.message import AssistantMessage

from coevo.environment import Tau2Environment
from coevo.environment.tau2 import dump_messages
from coevo.rollout.cutoff_scorer import TurnCutoffScorer
from coevo.rollout.views import buyer_view, student_view


class CorrectabilityCollector:
    """Collect a complete trunk, then score cutoffs inside completed Student turns."""

    def __init__(
        self,
        environment: Tau2Environment,
        scorer: TurnCutoffScorer,
        max_turns: int,
    ):
        self.environment = environment
        self.scorer = scorer
        self.max_turns = max_turns

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
        student_turns = []
        for message_index in range(initial_size, len(trunk)):
            message = trunk[message_index]
            if not isinstance(message, AssistantMessage):
                continue
            scored = self.scorer.score_turn(trunk, message_index)
            if scored is not None:
                student_turns.append(scored)
            if len(student_turns) >= self.max_turns:
                break
        return {
            "domain": env.config.domain,
            "task_split": env.config.task_split,
            "task_id": env.task.id,
            "seed": trajectory_seed,
            "trunk": dump_messages(trunk),
            "student_turns": student_turns,
        }

    def student_rows(self, record: dict) -> list[dict]:
        env = self.environment
        raw_environment = env.fresh_environment()
        student = env.policies.student(raw_environment)
        teacher = env.policies.teacher(raw_environment, env.task)
        rows = []
        for turn in record["student_turns"]:
            from coevo.environment.tau2 import load_messages

            history_before = load_messages(turn["history_before"])
            full_history = [
                *history_before,
                AssistantMessage(role="assistant", content=turn["student_output"]),
            ]
            messages = student_view(student.system_prompt, full_history)
            last_user = next(
                message["content"]
                for message in reversed(messages[:-1])
                if message["role"] == "user"
            )
            privileged = teacher.make_agent_instructions_from_actions()
            rows.append(
                {
                    "messages": messages,
                    "teacher_prompt": (
                        f"{last_user}\n\n<privileged_resolution_steps>\n"
                        f"{privileged}\n</privileged_resolution_steps>"
                    ),
                    "correctability": turn["correctability"],
                    "cutoff_count": len(turn["cutoffs"]),
                    "domain": env.config.domain,
                    "task_split": env.config.task_split,
                    "task_id": env.task.id,
                }
            )
        return rows

    def buyer_row(self) -> dict:
        env = self.environment
        raw_environment = env.fresh_environment()
        buyer = env.policies.buyer_reference(raw_environment, env.task)
        history = env.initial_history()
        row = {
            "messages": buyer_view(buyer.system_prompt, history),
            "domain": env.config.domain,
            "task_split": env.config.task_split,
            "task_id": env.task.id,
            "tau_history": dump_messages(history),
        }
        if buyer.tools:
            row["tools"] = [tool.openai_schema for tool in buyer.tools]
        return row
