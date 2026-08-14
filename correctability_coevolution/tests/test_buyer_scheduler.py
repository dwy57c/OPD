import asyncio
import json
from copy import deepcopy

import pytest
from swift.infer_engine.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    RequestConfig,
    RolloutInferRequest,
    UsageInfo,
)

from coevo.training import buyer_scheduler as scheduler_module
from coevo.training.buyer_scheduler import Tau2BuyerScheduler, visible_buyer_content
from coevo.training.swift_plugin import _capture_buyer_reward_groups


@pytest.fixture(autouse=True)
def scheduler_environment(monkeypatch):
    monkeypatch.setenv("COEVO_TEACHER_HINT_MODE", "none")
    monkeypatch.setenv("COEVO_BUYER_PLAN_MODE", "legacy")
    monkeypatch.setenv("COEVO_PREVIOUS_POLICY_PATH", "/previous")
    monkeypatch.setenv("COEVO_PREVIOUS_POLICY_CHECKPOINT", "/previous")
    monkeypatch.setenv("COEVO_CURRENT_POLICY_CHECKPOINT", "/current")
    monkeypatch.setenv("COEVO_TEACHER_ANCHOR", "previous")


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
    assert visible_buyer_content("<think>secret</think>\nI need help.") == "I need help."
    assert visible_buyer_content("</THINK>Visible") == "Visible"
    assert visible_buyer_content("<think>unfinished") == ""


def test_rollout_reward_is_mean_stage_progress_only():
    scheduler = Tau2BuyerScheduler(infer_engine=object(), max_turns=2)
    infos = scheduler._rollout_infos(
        {
            "trajectory_validity": 1.0,
            "stage_progress_decisions": [
                {
                    "decision_reward": 0.2,
                    "previous_gap": 0.5,
                    "current_gap": 0.3,
                    "learning_progress": 0.2,
                    "positive_learning_progress": 0.2,
                    "teacher_target_hash": "a",
                    "raw_teacher_target_hash": "ra",
                    "skill_contrast_scores": [0.1],
                    "skill_gate_values": [1.0],
                    "sharpening_temperatures": [0.7],
                    "raw_teacher_entropy": [0.5],
                    "sharpened_teacher_entropy": [0.3],
                },
                {
                    "decision_reward": 0.0,
                    "previous_gap": 0.2,
                    "current_gap": 0.3,
                    "learning_progress": -0.1,
                    "positive_learning_progress": 0.0,
                    "teacher_target_hash": "b",
                    "raw_teacher_target_hash": "rb",
                    "skill_contrast_scores": [0.0],
                    "skill_gate_values": [0.0],
                    "sharpening_temperatures": [1.0],
                    "raw_teacher_entropy": [0.5],
                    "sharpened_teacher_entropy": [0.5],
                },
            ],
        }
    )

    assert infos["buyer_reward"] == pytest.approx(0.1)
    assert infos["reward_source"] == "stage_learning_progress"
    assert infos["learning_progresses"] == [0.2, -0.1]
    assert infos["teacher_target_hashes"] == ["a", "b"]
    assert infos["checkpoint_teacher_anchor"] == "/previous"


def test_full_trace_capture_exposes_exact_plan_action_and_decision(monkeypatch):
    monkeypatch.setenv("COEVO_CAPTURE_FULL_TRACE", "1")
    scheduler = Tau2BuyerScheduler(infer_engine=object(), max_turns=2)
    decision = {
        "decision_reward": 0.2,
        "previous_gap": 0.5,
        "current_gap": 0.3,
        "learning_progress": 0.2,
        "positive_learning_progress": 0.2,
    }
    infos = scheduler._rollout_infos(
        {
            "domain": "airline",
            "task_split": "train",
            "task_id": "1",
            "buyer_private_plans": [{"next_move": "clarify_previous_statement"}],
            "buyer_public_actions": [{"content": "Please clarify."}],
            "plan_action_consistency": [1.0],
            "tau_history": [{"role": "user", "content": "Please clarify."}],
            "stage_progress_decisions": [decision],
            "teacher_target_labels": [{"teacher_action": {"role": "assistant"}}],
        }
    )

    assert infos["task_id"] == "1"
    assert infos["buyer_private_plans"][0]["next_move"] == (
        "clarify_previous_statement"
    )
    assert infos["buyer_public_actions"][0]["content"] == "Please clarify."
    assert infos["stage_progress_decisions"] == [decision]


def test_reward_trace_matches_swift_group_normalization(monkeypatch, tmp_path):
    trace_path = tmp_path / "buyer_groups.jsonl"
    monkeypatch.setenv("COEVO_BUYER_TRACE_PATH", str(trace_path))
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    _capture_buyer_reward_groups(
        [0.1, 0.3],
        [0.1, 0.3],
        [{"task_id": "1"}, {"task_id": "1"}],
        group_ids=None,
        group_size=2,
    )

    record = json.loads(trace_path.read_text(encoding="utf-8"))
    assert record["group_mean"] == pytest.approx(0.2)
    assert record["group_sample_std"] == pytest.approx(2**0.5 / 10)
    assert record["normalized_advantages"][0] == pytest.approx(
        -0.1 / (2**0.5 / 10 + 1e-4)
    )
    assert record["normalized_advantages"][1] == pytest.approx(
        0.1 / (2**0.5 / 10 + 1e-4)
    )


def test_scoring_failure_is_fail_closed_without_fallback():
    scheduler = Tau2BuyerScheduler(infer_engine=object(), max_turns=2)
    infos = scheduler._rollout_infos(
        {
            "trajectory_validity": 1.0,
            "scoring_errors": [{"message": "previous endpoint timeout"}],
            "decision_count": 1,
            "stage_progress_decisions": [
                {
                    "decision_reward": 0.4,
                    "previous_gap": 0.6,
                    "current_gap": 0.2,
                    "learning_progress": 0.4,
                    "positive_learning_progress": 0.4,
                }
            ],
        }
    )
    assert infos["buyer_reward"] == 0.0
    assert infos["trajectory_validity"] == 0.0
    assert infos["decision_count"] == 1
    assert infos["scoring_errors"]


def test_server_scheduler_executes_final_buyer_action_and_masks_only_buyer_tokens(
    monkeypatch,
):
    class Engine:
        def __init__(self):
            self.responses = iter(
                [make_response("first", 11), make_response("final", 22)]
            )
            self.requests = []

        async def infer_async(self, infer_request, request_config, **kwargs):
            self.requests.append(deepcopy(infer_request.messages))
            return next(self.responses)

    engine = Engine()
    scheduler = Tau2BuyerScheduler(infer_engine=engine, max_turns=2)
    executed = []

    def apply_action(infer_request, choice, append_observation):
        executed.append((choice.message.content, append_observation))
        if append_observation:
            infer_request.messages.append({"role": "user", "content": "student"})
        return {"buyer_reward": len(executed) / 10}, len(executed) == 2

    monkeypatch.setattr(scheduler, "_apply_buyer_action", apply_action)
    request = RolloutInferRequest(
        messages=[{"role": "user", "content": "hello"}], data_dict={}
    )
    output = asyncio.run(scheduler.run(request, RequestConfig()))

    assert executed == [("first", True), ("final", False)]
    assert output.response_token_ids == [[11], [22]]
    assert output.response_loss_mask == [[1], [1]]
    assert output.rollout_infos["buyer_reward"] == 0.2


def test_truncated_rollout_is_invalid_and_zero_reward(monkeypatch):
    class Engine:
        async def infer_async(self, infer_request, request_config, **kwargs):
            response = make_response("partial", 7)
            response.choices[0].finish_reason = "length"
            return response

    scheduler = Tau2BuyerScheduler(infer_engine=Engine(), max_turns=2)
    request = RolloutInferRequest(
        messages=[{"role": "user", "content": "hello"}], data_dict={}
    )
    output = asyncio.run(scheduler.run(request, RequestConfig()))
    assert output.rollout_infos["trajectory_validity"] == 0.0
    assert output.rollout_infos["buyer_reward"] == 0.0


def test_scheduler_context_uses_row_domain_split_and_task(monkeypatch):
    environments = []

    class Environment:
        def __init__(self, config):
            self.config = config
            environments.append(self)

    monkeypatch.setattr(scheduler_module, "Tau2Environment", Environment)
    monkeypatch.setattr(
        scheduler_module, "build_teacher_target_labeler", lambda environment: object()
    )
    scheduler = Tau2BuyerScheduler(infer_engine=object(), max_turns=2)
    first = scheduler._context(
        {"domain": "airline", "task_split": "train", "task_id": "1"}
    )
    repeated = scheduler._context(
        {"domain": "airline", "task_split": "train", "task_id": "1"}
    )
    second = scheduler._context(
        {"domain": "airline", "task_split": "test", "task_id": "2"}
    )
    assert first is repeated
    assert first is not second


def test_structured_plan_uses_learning_progress_vocabulary(monkeypatch):
    scheduler = Tau2BuyerScheduler(infer_engine=object(), max_turns=2)
    scheduler.base_config = scheduler.base_config.__class__(
        **{
            **scheduler.base_config.__dict__,
            "buyer_plan_mode": "structured",
        }
    )

    class Environment:
        @staticmethod
        def available_user_tool_names():
            return ()

        @staticmethod
        def buyer_scenario_text():
            return ""

    plan = (
        '{"diagnosis":{"failure_type":"missing_confirmation",'
        '"evidence_turns":[1]},"target_skill":"policy_compliance",'
        '"next_move":"ask_about_cost","payload":{},'
        '"predicted_learning_progress":0.4,"stop":false}'
    )
    data = {}
    message = scheduler._decode_buyer_action(
        Environment(), make_response(plan, 9).choices[0], data
    )
    assert message.content == "What will the total cost be, including all fees?"
    assert data["buyer_private_plans"][0]["predicted_learning_progress"] == 0.4
