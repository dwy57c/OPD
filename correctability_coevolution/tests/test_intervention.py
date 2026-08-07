from types import SimpleNamespace

import pytest
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage

from coevo.intervention import (
    ActionBranchRunner,
    DecisionState,
    TeacherActionGenerator,
    TeacherActionResult,
    extract_decision_states,
)
from coevo.rewards import soft_completion_score


def reward_info(action, communication, database, environment, nl):
    return {
        "reward_basis": [
            "ACTION",
            "COMMUNICATE",
            "DB",
            "ENV_ASSERTION",
            "NL_ASSERTION",
        ],
        "action_checks": [{"action_match": value} for value in action],
        "communicate_checks": [{"met": value} for value in communication],
        "db_check": {"db_match": database},
        "env_assertions": [{"met": value} for value in environment],
        "nl_assertions": [{"met": value} for value in nl],
    }


def test_soft_score_balances_categories_instead_of_atomic_check_counts():
    result = soft_completion_score(
        reward_info(
            action=[True] * 10,
            communication=[False],
            database=True,
            environment=[False],
            nl=[False],
        )
    )

    # action=1, communication=0, environment=(1+0)/2, nl=0
    assert result.score == pytest.approx(0.375)
    assert result.categories["environment"].score == 0.5


def test_decision_state_uses_complete_text_or_parallel_tool_action():
    history = [
        UserMessage(role="user", content="help"),
        AssistantMessage(role="assistant", content="complete response"),
        UserMessage(role="user", content="look it up"),
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="a",
                    name="first",
                    arguments={},
                    requestor="assistant",
                ),
                ToolCall(
                    id="b",
                    name="second",
                    arguments={"x": 1},
                    requestor="assistant",
                ),
            ],
        ),
    ]

    decisions = extract_decision_states(history)

    assert len(decisions) == 2
    assert decisions[0].student_action.content == "complete response"
    assert len(decisions[1].student_action.tool_calls) == 2
    assert decisions[1].state_hash != decisions[1].sample_hash


def test_one_teacher_action_then_paired_student_continuations():
    calls = []

    class Environment:
        config = SimpleNamespace(seed=7)

        @staticmethod
        def continue_to_terminal(history, policy, seed):
            calls.append((history[-1].content, policy, seed))
            is_teacher_action = history[-1].content == "repair"
            info = reward_info(
                action=[is_teacher_action],
                communication=[is_teacher_action],
                database=is_teacher_action,
                environment=[is_teacher_action],
                nl=[is_teacher_action],
            )
            return SimpleNamespace(reward_info=info)

    state = DecisionState.from_history(
        [
            UserMessage(role="user", content="help"),
            AssistantMessage(role="assistant", content="mistake"),
        ],
        1,
    )
    generator = TeacherActionGenerator(
        Environment(),
        action_provider=lambda decision, seed: TeacherActionResult(
            AssistantMessage(role="assistant", content="repair"), {"seed": seed}
        ),
    )
    runner = ActionBranchRunner(
        Environment(), continuations=2, teacher_generator=generator
    )

    result = runner.run(state)

    assert result.intervention_advantage == 1.0
    assert result.teacher_action.hint == {"seed": 7}
    assert calls == [
        ("mistake", "student", 7),
        ("repair", "student", 7),
        ("mistake", "student", 8),
        ("repair", "student", 8),
    ]
