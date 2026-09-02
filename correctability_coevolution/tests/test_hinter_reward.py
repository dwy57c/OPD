import math
from types import SimpleNamespace

import pytest
import torch

from coevo.config import ModelEndpoint
from coevo.hinter_training import (
    BehaviorHintSample,
    BehaviorCopyingDiscriminator,
    DiscriminatorControlReport,
    DiscriminatorGate,
    HinterCompositeReward,
    HinterRewardConfig,
    TeacherForcedUsefulnessScorer,
    build_fresh_discriminator_pairs,
    build_hinter_grpo_dataset,
    score_hinter_hint,
    pairwise_copy_probability,
    pairwise_ranking_loss,
)
from coevo.scoring.stage_gap import SparseTargetView
from coevo.training.gated_gkd import truncate_rows_to_active_token_budget
from coevo.hinter_training.behavior_discriminator import score_texts
from scripts.build_copying_discriminator_dataset import load_reward_trace


def sparse(actual_logs):
    return SparseTargetView(
        target_input_ids=(3, 4),
        topk_logprobs=(
            (actual_logs[0], math.log(0.1)),
            (actual_logs[1], math.log(0.1)),
        ),
        topk_token_ids=((3, 8), (4, 9)),
        support_mass=(0.9, 0.9),
    )


def test_reward_is_usefulness_minus_copying_minus_token_length():
    result = score_hinter_hint(
        usefulness=3.0,
        copying_probability=0.75,
        hint_tokens=4,
        config=HinterRewardConfig(
            copying_weight=2.0, length_weight=0.1, max_hint_tokens=10
        ),
    )
    assert result.copying_penalty == pytest.approx(1.0)
    assert result.length_penalty == pytest.approx(0.4)
    assert result.reward == pytest.approx(1.6)
    with pytest.raises(ValueError, match="hard token cap"):
        score_hinter_hint(
            usefulness=100,
            copying_probability=0,
            hint_tokens=11,
            config=HinterRewardConfig(max_hint_tokens=10),
        )


def test_usefulness_is_two_teacher_forced_views_on_same_standard_action():
    calls = []

    class Builder:
        def score_view(self, **kwargs):
            calls.append(kwargs)
            if kwargs["information_view"] == "hinter_reward_unhinted":
                return sparse((-2.0, -3.0))
            return sparse((-1.0, -1.0))

    config = SimpleNamespace(
        policy=ModelEndpoint("student", "http://student"),
        current_policy_checkpoint="student-checkpoint",
    )
    scorer = TeacherForcedUsefulnessScorer(config, target_builder=Builder())
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "help"},
        {"role": "assistant", "content": "standard answer"},
    ]
    result = scorer.score(
        student_visible_messages=messages,
        hint="Ask for the missing lookup key.",
        state_hash="state",
    )
    assert len(calls) == 2
    assert all(call["checkpoint_id"] == "student-checkpoint" for call in calls)
    assert all(call["messages"][-1] == messages[-1] for call in calls)
    assert result.unhinted_log_probability == pytest.approx(-5.0)
    assert result.hinted_log_probability == pytest.approx(-2.0)
    assert result.log_probability_gain == pytest.approx(3.0)
    assert result.per_token_gain == pytest.approx(1.5)
    assert result.probability_trace.actual_token_logprobs == pytest.approx(
        (-1.0, -1.0)
    )


def test_discriminator_negatives_shuffle_only_hint_within_state():
    samples = [
        BehaviorHintSample("s", {"x": 1}, "hint-a", {"action": "a"}),
        BehaviorHintSample("s", {"x": 1}, "hint-b", {"action": "b"}),
    ]
    pairs = build_fresh_discriminator_pairs(samples)
    assert len(pairs) == 2
    assert pairs[0].public_state == {"x": 1}
    assert pairs[0].positive_hint == "hint-a"
    assert pairs[0].negative_hint == "hint-b"
    assert pairs[0].student_behavior == {"action": "a"}


def test_discriminator_loader_skips_degenerate_reward_trace_rows(tmp_path):
    path = tmp_path / "reward.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"state_hash":"s","degenerate_group":true,"hint":"same"}',
                '{"state_hash":"s","public_state":{},"hint":"a",'
                '"student_behavior":{"action":"a"}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    samples = load_reward_trace(path)
    assert len(samples) == 1
    assert samples[0].hint == "a"


def test_pairwise_head_uses_logistic_ranking_and_control_gates():
    loss = pairwise_ranking_loss(torch.tensor([2.0]), torch.tensor([-1.0]))
    assert loss.item() < 0.1
    assert pairwise_copy_probability(2.0, -1.0) > 0.9
    passing = DiscriminatorControlReport(
        ordinary_pair_accuracy=0.8,
        explicit_copy_accuracy=1.0,
        explicit_copy_natural_accuracy=0.9,
        useless_mean_distance_from_chance=0.05,
        ordinary_pairs=10,
        explicit_copy_pairs=4,
        explicit_copy_natural_pairs=4,
        useless_pairs=4,
    )
    DiscriminatorGate().validate(passing)
    with pytest.raises(ValueError, match="explicit-copy"):
        DiscriminatorGate().validate(
            DiscriminatorControlReport(
                ordinary_pair_accuracy=0.8,
                explicit_copy_accuracy=0.5,
                explicit_copy_natural_accuracy=0.9,
                useless_mean_distance_from_chance=0.05,
                ordinary_pairs=10,
                explicit_copy_pairs=4,
                explicit_copy_natural_pairs=4,
                useless_pairs=4,
            )
        )


def test_discriminator_scoring_is_chunked_to_avoid_oom():
    calls = []

    class Tokenizer:
        def __call__(self, values, **_kwargs):
            calls.append(len(values))
            return {"input_ids": torch.ones((len(values), 1), dtype=torch.long)}

    class Model:
        @staticmethod
        def parameters():
            return iter([torch.nn.Parameter(torch.zeros(1))])

        @staticmethod
        def __call__(**kwargs):
            size = kwargs["input_ids"].shape[0]
            return SimpleNamespace(logits=torch.zeros((size, 1)))

    scores = score_texts(
        Model(), Tokenizer(), [str(index) for index in range(65)], max_length=8192
    )
    assert len(scores) == 65
    assert calls == [32, 32, 1]
    with pytest.raises(ValueError, match="useless-hint"):
        DiscriminatorGate().validate(
            DiscriminatorControlReport(
                ordinary_pair_accuracy=0.8,
                explicit_copy_accuracy=1.0,
                explicit_copy_natural_accuracy=0.9,
                useless_mean_distance_from_chance=0.4,
                ordinary_pairs=10,
                explicit_copy_pairs=4,
                explicit_copy_natural_pairs=4,
                useless_pairs=4,
            )
        )
    with pytest.raises(ValueError, match="natural-copy"):
        DiscriminatorGate().validate(
            DiscriminatorControlReport(
                ordinary_pair_accuracy=0.8,
                explicit_copy_accuracy=1.0,
                explicit_copy_natural_accuracy=0.5,
                useless_mean_distance_from_chance=0.05,
                ordinary_pairs=10,
                explicit_copy_pairs=4,
                explicit_copy_natural_pairs=4,
                useless_pairs=4,
            )
        )


def test_copy_penalty_increases_when_true_hint_outranks_same_state_decoys():
    discriminator = object.__new__(BehaviorCopyingDiscriminator)
    discriminator.score_texts = lambda _texts: [3.0, 0.0, -1.0]
    probability = discriminator.copy_probability(
        public_state={"task": 1},
        student_behavior={"tool_call": "lookup"},
        true_hint="use lookup",
        alternative_hints=["use lookup", "ask first", "say hello"],
    )
    assert probability > 0.95


def test_grpo_reward_compares_each_behavior_to_same_state_candidate_hints():
    reward = object.__new__(HinterCompositeReward)
    reward.reward_config = HinterRewardConfig(
        copying_weight=1.0, length_weight=0.1, max_hint_tokens=20
    )

    class Utility:
        log_probability_gain = 2.0
        per_token_gain = 1.0

        @staticmethod
        def to_dict():
            return {"log_probability_gain": 2.0}

    reward.usefulness = SimpleNamespace(score=lambda **_kwargs: Utility())
    reward.student_behavior = SimpleNamespace(
        generate=lambda hint, **_kwargs: {"operation": hint}
    )
    calls = []

    def copying(**kwargs):
        calls.append(kwargs)
        return 0.75

    reward.discriminator = SimpleNamespace(copy_probability=copying)
    reward.hinter_tokenizer = SimpleNamespace(
        encode=lambda hint, add_special_tokens=False: hint.split()
    )
    reward.trace_path = ""
    reward._trace_lock = None
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "assistant", "content": "standard"},
    ]
    values = reward(
        ["short hint", "different hint"],
        state_hash=["s", "s"],
        public_state=[[{"role": "user", "content": "x"}]] * 2,
        student_visible_messages=[messages, messages],
        tools=[[], []],
    )
    assert values == pytest.approx([0.3, 0.3])
    assert calls[0]["alternative_hints"] == ["short hint", "different hint"]
    assert calls[0]["student_behavior"] == {"operation": "short hint"}


def test_explicit_copy_anchor_is_net_negative():
    config = HinterRewardConfig()
    breakdown = score_hinter_hint(
        usefulness=0.8,
        copying_probability=0.95,
        hint_tokens=80,
        config=config,
        rule_leaks=("identifier",),
    )
    assert breakdown.reward <= -config.rule_leak_floor


def test_rule_clean_but_detected_hint_loses_to_clean_hint():
    config = HinterRewardConfig()
    detected = score_hinter_hint(
        usefulness=0.3,
        copying_probability=0.95,
        hint_tokens=80,
        config=config,
    )
    clean = score_hinter_hint(
        usefulness=0.3,
        copying_probability=0.5,
        hint_tokens=80,
        config=config,
    )
    assert detected.reward < clean.reward
    assert clean.copying_penalty == 0.0


def test_degenerate_group_returns_zero_rewards():
    reward = object.__new__(HinterCompositeReward)
    reward.reward_config = HinterRewardConfig()

    class Utility:
        log_probability_gain = 1.0
        per_token_gain = 0.5

        @staticmethod
        def to_dict():
            return {"per_token_gain": 0.5}

    reward.usefulness = SimpleNamespace(score=lambda **_kwargs: Utility())
    reward.student_behavior = SimpleNamespace(
        generate=lambda hint, **_kwargs: {"operation": hint}
    )
    reward.discriminator = SimpleNamespace(
        copy_probability=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("degenerate group called discriminator")
        )
    )
    reward.hinter_tokenizer = SimpleNamespace(
        encode=lambda hint, add_special_tokens=False: hint.split()
    )
    reward.trace_path = ""
    reward._trace_lock = None
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "assistant", "content": "standard"},
    ]
    assert reward(
        ["same hint", "same hint"],
        state_hash=["s", "s"],
        public_state=[[{"role": "user", "content": "x"}]] * 2,
        student_visible_messages=[messages, messages],
        tools=[[], []],
    ) == [0.0, 0.0]


def test_reward_callback_applies_rule_floor_from_privileged_context():
    reward = object.__new__(HinterCompositeReward)
    reward.reward_config = HinterRewardConfig(rule_leak_floor=1.0)

    class Utility:
        log_probability_gain = 1.6
        per_token_gain = 0.8

        @staticmethod
        def to_dict():
            return {"per_token_gain": 0.8}

    reward.usefulness = SimpleNamespace(score=lambda **_kwargs: Utility())
    reward.student_behavior = SimpleNamespace(
        generate=lambda hint, **_kwargs: {"operation": hint}
    )
    reward.discriminator = SimpleNamespace(copy_probability=lambda **_kwargs: 0.5)
    reward.hinter_tokenizer = SimpleNamespace(
        encode=lambda hint, add_special_tokens=False: hint.split()
    )
    reward.trace_path = ""
    reward._trace_lock = None
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "assistant", "content": "standard"},
    ]
    values = reward(
        ["Use booking ABC123.", "Ask for the booking key."],
        state_hash=["s", "s"],
        public_state=[[{"role": "user", "content": "x"}]] * 2,
        student_visible_messages=[messages, messages],
        tools=[[], []],
        privileged_context=[
            {"authoritative_oracle_steps": "Use booking ABC123"},
            {"authoritative_oracle_steps": "Use booking ABC123"},
        ],
    )
    assert values[0] <= -1.0
    assert values[1] > 0


def test_grpo_dataset_uses_one_fixed_l3_standard_action_per_state():
    standard_messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "help"},
        {"role": "assistant", "content": "verified standard"},
    ]
    common = {
        "state_hash": "s",
        "public_state": [{"role": "user", "content": "help"}],
        "privileged_context": {"oracle": "private"},
        "tools": [],
        "standard_action_eligible": True,
    }
    rows = [
        {**common, "hint_level": "L2_PROCEDURAL", "student_visible_messages": [*standard_messages[:-1], {"role": "assistant", "content": "other"}]},
        {**common, "hint_level": "L3_ORACLE", "student_visible_messages": standard_messages},
    ]
    result = build_hinter_grpo_dataset(rows)
    assert len(result) == 1
    assert result[0].student_visible_messages[-1]["content"] == "verified standard"
    assert result[0].standard_source_level == "L3_ORACLE"


def test_active_token_budget_never_partially_includes_a_row():
    selected, used = truncate_rows_to_active_token_budget(
        [
            {"target_token_count": 5, "id": 1},
            {"target_token_count": 7, "id": 2},
            {"target_token_count": 4, "id": 3},
        ],
        10,
    )
    assert [row["id"] for row in selected] == [1, 3]
    assert used == 9
