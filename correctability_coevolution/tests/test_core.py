from tau2.data_model.message import AssistantMessage, UserMessage

from coevo.rollout.collector import NaturalDecisionCollector


def test_unlimited_collector_scores_every_natural_student_action():
    history = [UserMessage(role="user", content="Start")]
    for index in range(5):
        history.extend(
            [
                AssistantMessage(role="assistant", content=f"Student action {index}."),
                UserMessage(role="user", content=f"Buyer action {index}."),
            ]
        )

    class Orchestrator:
        done = True
        step_count = 0

        def initialize(self):
            pass

        def get_trajectory(self):
            return history

    class Environment:
        config = type(
            "Config",
            (),
            {
                "seed": 42,
                "branch_max_steps": 24,
                "domain": "test",
                "task_split": "train",
            },
        )()
        task = type("Task", (), {"id": "test"})()

        def initial_history(self):
            return history[:1]

        def orchestrator(self, initial_history, policy, seed):
            return Orchestrator()

    class Scorer:
        def score_decision(self, trajectory, message_index):
            return {
                "message_index": message_index,
                "intervention_advantage": 0.25,
            }

    record = NaturalDecisionCollector(
        Environment(), Scorer(), max_decisions=0
    ).collect_one()

    assert len(record["student_decisions"]) == 5
    assert all(
        turn["intervention_advantage"] == 0.25
        for turn in record["student_decisions"]
    )
