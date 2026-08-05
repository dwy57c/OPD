import asyncio
from copy import deepcopy
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
            "correctability_reward": len(executed) / 10,
            "trajectory_correctability": len(executed) / 10,
            "turn_correctability": [len(executed) / 10],
            "validity": 1.0,
            "cutoff_count": len(executed),
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
    assert output.rollout_infos["correctability_reward"] == 0.2
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
                "correctability_reward": 0.5,
                "trajectory_correctability": 0.5,
                "turn_correctability": [0.5],
                "validity": 1.0,
                "cutoff_count": 1,
            },
            False,
        ),
    )
    request = RolloutInferRequest(
        messages=[{"role": "user", "content": "hello"}],
        data_dict={"turn_correctability": [0.5]},
    )

    output = asyncio.run(scheduler.run(request, RequestConfig()))

    assert output.rollout_infos["validity"] == 0.0
    assert output.rollout_infos["correctability_reward"] == 0.0
    assert output.rollout_infos["num_turns"] == 1


def test_scheduler_uses_domain_split_and_task_from_each_row(monkeypatch):
    environments = []

    class Environment:
        def __init__(self, config):
            self.config = config
            environments.append(self)

    monkeypatch.setattr(scheduler_module, "Tau2Environment", Environment)
    monkeypatch.setattr(
        scheduler_module, "build_cutoff_scorer", lambda environment: object()
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
        def score_turn(history, message_index):
            return {"correctability": 0.75, "cutoffs": [{"id": 1}]}

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
    assert infos["validity"] == 0.0
    assert infos["turn_correctability"] == [0.75]
    assert infos["correctability_reward"] == 0.0
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
    assert infos["correctability_reward"] == 0.0
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
        data_dict={"turn_correctability": [0.5]},
    )
    output = asyncio.run(scheduler.run(request, RequestConfig()))

    assert output.rollout_infos["validity"] == 0.0
    assert output.rollout_infos["correctability_reward"] == 0.0
    assert output.rollout_infos["num_turns"] == 1
