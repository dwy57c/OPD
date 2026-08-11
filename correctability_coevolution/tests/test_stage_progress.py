import math

import pytest
import torch

from coevo.config import InfraConfig, ModelEndpoint
from coevo.rewards.stage_progress import (
    mean_forward_kl,
    score_stage_progress,
)
from coevo.scoring.stage_gap import StageGapScorer, TeacherTargetBuilder


def test_stage_reward_is_exactly_positive_learning_progress():
    identical = score_stage_progress(
        previous_gap=0.2,
        current_gap=0.2,
    )
    improved = score_stage_progress(
        previous_gap=0.3,
        current_gap=0.2,
    )
    regressed = score_stage_progress(
        previous_gap=0.1,
        current_gap=0.2,
    )
    mastered = score_stage_progress(
        previous_gap=0.04,
        current_gap=0.01,
    )

    assert identical.learning_progress == pytest.approx(0.0)
    assert identical.decision_reward == 0.0
    assert improved.decision_reward == pytest.approx(0.1)
    assert regressed.learning_progress < 0
    assert regressed.decision_reward == 0.0
    assert mastered.decision_reward == pytest.approx(0.03)


def test_forward_kl_is_normalized_by_active_target_tokens():
    one = mean_forward_kl(
        teacher_logprobs=[[math.log(0.8), math.log(0.2)]],
        teacher_token_ids=[[1, 2]],
        student_logprobs=[[math.log(0.6), math.log(0.4)]],
        student_token_ids=[[1, 2]],
        target_token_ids=[1],
    )
    two = mean_forward_kl(
        teacher_logprobs=[[math.log(0.8), math.log(0.2)]] * 2,
        teacher_token_ids=[[1, 2]] * 2,
        student_logprobs=[[math.log(0.6), math.log(0.4)]] * 2,
        student_token_ids=[[1, 2]] * 2,
        target_token_ids=[1, 2],
    )
    assert two == pytest.approx(one)


class FakeTokenizer:
    chat_template = "fake-v1"

    @staticmethod
    def apply_chat_template(messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is True
        hinted = "private_teacher_hint" in messages[0]["content"]
        prefix = [9 if hinted else 1, 2]
        return prefix if add_generation_prompt else [*prefix, 3, 4]


class FakeClient:
    def __init__(self):
        self.calls = []

    def fetch(self, endpoint, input_ids):
        self.calls.append((endpoint.model, tuple(input_ids)))
        if input_ids[0] == 9:
            probability = 0.9
        elif endpoint.model == "current":
            probability = 0.8
        else:
            probability = 0.4
        logs = torch.tensor(
            [[math.log(probability), math.log(1 - probability)]],
            dtype=torch.float32,
        ).repeat(3, 1)
        return logs, torch.tensor([[3, 4]]).repeat(3, 1)


class FakeToolTokenizer:
    chat_template = "fake-tool-v1"

    @staticmethod
    def apply_chat_template(messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is True
        prefix = [1, 2]
        if add_generation_prompt:
            return prefix
        assert kwargs["continue_final_message"] is False
        return [*prefix, 3, 4, 99, 10]

    @staticmethod
    def encode(text, *, add_special_tokens):
        assert text == "<|im_end|>\n"
        assert add_special_tokens is False
        return [99, 10]


def test_tool_call_target_keeps_macro_action_and_strips_only_turn_terminator():
    config = InfraConfig(
        policy=ModelEndpoint("current", "http://current"),
        buyer_reference=ModelEndpoint("buyer", "http://buyer"),
        teacher_hint_mode="none",
    )
    builder = TeacherTargetBuilder(
        config,
        tokenizer=FakeToolTokenizer(),
        client=FakeClient(),
    )
    tokenized = builder.tokenize_target(
        [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "look it up"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
        ]
    )

    assert tokenized.full_input_ids == (1, 2, 3, 4)
    assert tokenized.target_input_ids == (3, 4)


def test_three_view_scorer_reuses_one_sharpened_target_and_cache():
    config = InfraConfig(
        policy=ModelEndpoint("current", "http://current"),
        previous_policy=ModelEndpoint("previous", "http://previous"),
        buyer_reference=ModelEndpoint("buyer", "http://buyer"),
        teacher_hint_mode="none",
        current_policy_checkpoint="current-checkpoint",
        previous_policy_checkpoint="previous-checkpoint",
        teacher_gap_min_support_mass=0.9,
    )
    client = FakeClient()
    builder = TeacherTargetBuilder(
        config, tokenizer=FakeTokenizer(), client=client
    )
    scorer = StageGapScorer(config, target_builder=builder)
    ordinary = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "request"},
        {"role": "assistant", "content": "target"},
    ]
    hinted = [
        {"role": "system", "content": "policy private_teacher_hint"},
        ordinary[1],
        ordinary[2],
    ]
    arguments = dict(
        student_visible_messages=ordinary,
        hinted_teacher_messages=hinted,
        teacher_action=ordinary[-1],
        state_hash="state",
        teacher_hint_hash="hint-hash",
    )

    first = scorer.score(**arguments)
    second = scorer.score(**arguments)

    assert first.progress.learning_progress > 0
    assert first.progress.decision_reward > 0
    assert first.teacher_target.teacher_target_hash == (
        second.teacher_target.teacher_target_hash
    )
    assert first.teacher_target.teacher_target_hash == (
        first.to_dict()["teacher_target_hash"]
    )
    assert len(client.calls) == 3
