import asyncio
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from swift.infer_engine.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    RequestConfig,
    RolloutInferRequest,
    UsageInfo,
)
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage

from coevo.training import buyer_scheduler as scheduler_module
from coevo.training.buyer_scheduler import Tau2BuyerScheduler, visible_buyer_content


@pytest.fixture(autouse=True)
def disable_external_hinter_for_scheduler_unit_tests(monkeypatch):
    monkeypatch.setenv("COEVO_TEACHER_HINT_MODE", "none")
    monkeypatch.setenv("COEVO_BUYER_PLAN_MODE", "legacy")


def make_response(content: str, token_id: int):
    return ChatCompletionResponse(
        model="Qwen3-4B",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop",
                token_ids=[token_id],
            )
        ],
        usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def test_visible_buyer_content_removes_private_thinking():
    assert visible_buyer_content("<think>secret plan</think>\nI need help.") == "I need help."
    assert visible_buyer_content("<THINK>secret</THINK>\nFinal answer") == "Final answer"
    assert visible_buyer_content("</think>\nVisible answer") == "Visible answer"
    assert visible_buyer_content("</ThInK>\nVisible answer") == "Visible answer"
    assert visible_buyer_content("<think>unfinished reasoning") == ""
    assert visible_buyer_content("ordinary user message") == "ordinary user message"


def test_server_scheduler_executes_final_buyer_action(monkeypatch):
    class Engine:
        def __init__(self):
            self.responses = iter(
                [make_response("first action", 11), make_response("final action", 22)]
            )
            self.requests = []

        async def infer_async(self, infer_request, request_config, **kwargs):
            self.requests.append(deepcopy(infer_request.messages))
            return next(self.responses)

    engine = Engine()
    scheduler = Tau2BuyerScheduler(infer_engine=engine, max_turns=2)
    executed = []

    def apply_action(infer_request, response_choice, append_observation):
        executed.append((response_choice.message.content, append_observation))
        if append_observation:
            infer_request.messages.append({"role": "user", "content": "student"})
        return {
            "buyer_reward": len(executed) / 10,
            "mean_intervention_advantage": len(executed) / 10,
            "turn_intervention_advantages": [len(executed) / 10],
            "validity": 1.0,
            "decision_count": len(executed),
        }, len(executed) == 2

    monkeypatch.setattr(scheduler, "_apply_buyer_action", apply_action)
    request = RolloutInferRequest(
        messages=[{"role": "user", "content": "hello"}], data_dict={}
    )
    output = asyncio.run(scheduler.run(request, RequestConfig()))

    assert executed == [("first action", True), ("final action", False)]
    assert engine.requests[1][-1] == {"role": "user", "content": "student"}
    assert output.response_token_ids == [[11], [22]]
    assert output.response_loss_mask == [[1], [1]]
    assert output.messages[-1] == {"role": "assistant", "content": "final action"}
    assert output.rollout_infos["buyer_reward"] == 0.2
    assert output.rollout_infos["num_turns"] == 2


def test_exhausted_buyer_turn_budget_is_invalid(monkeypatch):
    class Engine:
        async def infer_async(self, infer_request, request_config, **kwargs):
            return make_response("unfinished action", 11)

    scheduler = Tau2BuyerScheduler(infer_engine=Engine(), max_turns=1)
    monkeypatch.setattr(
        scheduler,
        "_apply_buyer_action",
        lambda *args, **kwargs: (
            {
                "buyer_reward": 0.5,
                "mean_intervention_advantage": 0.5,
                "turn_intervention_advantages": [0.5],
                "validity": 1.0,
                "decision_count": 1,
            },
            False,
        ),
    )
    request = RolloutInferRequest(
        messages=[{"role": "user", "content": "hello"}],
        data_dict={"turn_intervention_advantages": [0.5]},
    )

    output = asyncio.run(scheduler.run(request, RequestConfig()))

    assert output.rollout_infos["validity"] == 0.0
    assert output.rollout_infos["buyer_reward"] == 0.0
    assert output.rollout_infos["num_turns"] == 1


def test_scheduler_uses_domain_split_and_task_from_each_row(monkeypatch):
    environments = []

    class Environment:
        def __init__(self, config):
            self.config = config
            environments.append(self)

    monkeypatch.setattr(scheduler_module, "Tau2Environment", Environment)
    monkeypatch.setattr(
        scheduler_module, "build_action_branch_runner", lambda environment: object()
    )
    scheduler = Tau2BuyerScheduler(infer_engine=object(), max_turns=2)

    first = scheduler._context(
        {"domain": "airline", "task_split": "train", "task_id": "1"}
    )
    second = scheduler._context(
        {"domain": "airline", "task_split": "test", "task_id": "2"}
    )
    repeated = scheduler._context(
        {"domain": "airline", "task_split": "train", "task_id": "1"}
    )

    assert first is repeated
    assert first is not second
    assert [
        (item.config.domain, item.config.task_split, item.config.task_id)
        for item in environments
    ] == [
        ("airline", "train", "1"),
        ("airline", "test", "2"),
    ]


def test_buyer_action_applies_environment_transition_validity(monkeypatch):
    buyer_contents = []

    class Environment:
        @staticmethod
        def buyer_message(content, tool_calls):
            buyer_contents.append(content)
            return UserMessage(role="user", content=content)

        @staticmethod
        def buyer_stopped(message):
            return False

        @staticmethod
        def advance_student(history):
            return [
                *history,
                ToolMessage(
                    id="failed-call",
                    role="tool",
                    content="tool failed",
                    requestor="assistant",
                    error=True,
                ),
                AssistantMessage(role="assistant", content="Please try again."),
            ]

    class Scorer:
        @staticmethod
        def run(decision):
            return SimpleNamespace(
                to_dict=lambda: {"intervention_advantage": 0.75}
            )

    scheduler = Tau2BuyerScheduler(infer_engine=object(), max_turns=2)
    monkeypatch.setattr(scheduler, "_context", lambda data: (Environment(), Scorer()))
    request = RolloutInferRequest(
        messages=[{"role": "user", "content": "hello"}],
        data_dict={"tau_history": []},
    )
    response_choice = make_response(
        "<think>I should remain persistent.</think>\nI still need help.", 9
    ).choices[0]

    infos, finished = scheduler._apply_buyer_action(
        request, response_choice, append_observation=True
    )

    assert finished is False
    # Student tool errors are precisely the weaknesses the Buyer should expose;
    # they do not invalidate an otherwise legal Buyer action.
    assert infos["validity"] == 1.0
    assert infos["turn_intervention_advantages"] == [0.75]
    assert infos["buyer_reward"] == 0.75
    assert buyer_contents == ["I still need help."]
    assert request.data_dict["tau_history"][0]["content"] == "I still need help."
    assert request.messages[-1] == {"role": "user", "content": "Please try again."}


def test_unclosed_thinking_is_an_invalid_empty_buyer_action(monkeypatch):
    class Environment:
        @staticmethod
        def buyer_message(content, tool_calls):
            return UserMessage(role="user", content=content or None)

        @staticmethod
        def buyer_stopped(message):
            return False

        @staticmethod
        def advance_student(history):
            raise AssertionError("private thinking must not be sent to Student")

    scheduler = Tau2BuyerScheduler(infer_engine=object(), max_turns=2)
    monkeypatch.setattr(scheduler, "_context", lambda data: (Environment(), object()))
    request = RolloutInferRequest(
        messages=[{"role": "user", "content": "hello"}],
        data_dict={"tau_history": []},
    )
    response_choice = make_response("<think>unfinished reasoning", 9).choices[0]

    infos, finished = scheduler._apply_buyer_action(
        request, response_choice, append_observation=True
    )

    assert finished is True
    assert infos["validity"] == 0.0
    assert infos["buyer_reward"] == 0.0
    assert request.data_dict["tau_history"][0]["content"] is None


def test_thinking_is_hidden_without_dropping_buyer_tool_calls(monkeypatch):
    received = []
    raw_tool_calls = [object()]

    class Environment:
        @staticmethod
        def buyer_message(content, tool_calls):
            received.append((content, tool_calls))
            return UserMessage(
                role="user",
                content=content or None,
                tool_calls=[
                    ToolCall(
                        id="buyer-call",
                        name="lookup",
                        arguments={},
                        requestor="user",
                    )
                ],
            )

        @staticmethod
        def buyer_stopped(message):
            return False

        @staticmethod
        def execute_user_tools(history):
            return [
                *history,
                ToolMessage(
                    id="buyer-call",
                    role="tool",
                    content="lookup result",
                    requestor="user",
                    error=False,
                ),
            ]

    scheduler = Tau2BuyerScheduler(infer_engine=object(), max_turns=2)
    monkeypatch.setattr(scheduler, "_context", lambda data: (Environment(), object()))
    request = RolloutInferRequest(
        messages=[{"role": "user", "content": "hello"}],
        data_dict={"tau_history": []},
    )
    response_choice = SimpleNamespace(
        message=SimpleNamespace(
            content="<think>choose a user tool</think>", tool_calls=raw_tool_calls
        )
    )

    infos, finished = scheduler._apply_buyer_action(
        request, response_choice, append_observation=True
    )

    assert finished is False
    assert infos["validity"] == 1.0
    assert received == [("", raw_tool_calls)]
    assert request.messages[-1] == {
        "role": "tool",
        "content": "lookup result",
        "tool_call_id": "buyer-call",
    }


def test_truncated_buyer_output_is_invalid_and_not_executed(monkeypatch):
    class Engine:
        async def infer_async(self, infer_request, request_config, **kwargs):
            response = make_response("partial", 7)
            response.choices[0].finish_reason = "length"
            return response

    scheduler = Tau2BuyerScheduler(infer_engine=Engine(), max_turns=2)
    monkeypatch.setattr(
        scheduler,
        "_apply_buyer_action",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("not called")),
    )
    request = RolloutInferRequest(
        messages=[{"role": "user", "content": "hello"}],
        data_dict={"turn_intervention_advantages": [0.5]},
    )
    output = asyncio.run(scheduler.run(request, RequestConfig()))

    assert output.rollout_infos["validity"] == 0.0
    assert output.rollout_infos["buyer_reward"] == 0.0
    assert output.rollout_infos["num_turns"] == 1


def test_structured_plan_is_private_and_only_rendered_action_reaches_tau2(monkeypatch):
    class Environment:
        @staticmethod
        def available_user_tool_names():
            return ()

        @staticmethod
        def buyer_stopped(message):
            return False

        @staticmethod
        def advance_student(history):
            return [*history, AssistantMessage(role="assistant", content="Student")]

    class Scorer:
        @staticmethod
        def run(decision):
            return SimpleNamespace(
                to_dict=lambda: {"intervention_advantage": 0.25}
            )

    scheduler = Tau2BuyerScheduler(infer_engine=object(), max_turns=2)
    scheduler.base_config = replace(scheduler.base_config, buyer_plan_mode="structured")
    monkeypatch.setattr(scheduler, "_context", lambda data: (Environment(), Scorer()))
    request = RolloutInferRequest(
        messages=[{"role": "user", "content": "hello"}],
        data_dict={"tau_history": []},
    )
    private_plan = (
        '{"diagnosis":{"failure_type":"missing_confirmation",'
        '"evidence_turns":[1]},"target_skill":"policy_compliance",'
        '"next_move":"ask_about_cost","payload":{},'
        '"predicted_takeover_gain":0.4,"stop":false}'
    )

    infos, finished = scheduler._apply_buyer_action(
        request, make_response(private_plan, 9).choices[0], append_observation=True
    )

    assert finished is False
    assert infos["validity"] == 1.0
    assert request.data_dict["tau_history"][0]["content"] == (
        "What will the total cost be, including all fees?"
    )
    assert private_plan not in str(request.data_dict["tau_history"])
    assert request.data_dict["buyer_private_plans"][0]["next_move"] == "ask_about_cost"
