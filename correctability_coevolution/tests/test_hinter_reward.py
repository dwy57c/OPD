import math
from types import SimpleNamespace

import pytest

from coevo.config import ModelEndpoint
from coevo.hinter_training import (
    HinterCompositeReward,
    HinterRewardConfig,
    TeacherForcedUsefulnessScorer,
    build_hinter_grpo_dataset,
    calibrate_copying_weight,
    score_hinter_hint,
)
from coevo.scoring.stage_gap import SparseTargetView
from coevo.training.gated_gkd import truncate_rows_to_active_token_budget


def sparse(actual_logs):
    return SparseTargetView(
        target_input_ids=(3, 4),
        topk_logprobs=(
            (actual_logs[0], math.log(0.1)),
            (actual_logs[1], math.log(0.1)),
        ),
        topk_token_ids=((3, 8), (4, 9)),
        support_mass=(math.exp(actual_logs[0]) + 0.1, math.exp(actual_logs[1]) + 0.1),
    )


def scorer(hint_only_logs, *, empty_logs=(-4.0, -4.0), clip=5.0):
    calls = []

    class Builder:
        def score_view(self, **kwargs):
            calls.append(kwargs)
            view = kwargs["information_view"]
            if view == "hinter_reward_unhinted":
                return sparse((-3.0, -3.0))
            if view.startswith("hinter_reward_hint_only"):
                return sparse(hint_only_logs)
            if view == "hinter_reward_empty_context":
                return sparse(empty_logs)
            return sparse((-1.0, -1.0))

    config = SimpleNamespace(
        policy=ModelEndpoint("student", "http://student"),
        current_policy_checkpoint="student-checkpoint",
    )
    return TeacherForcedUsefulnessScorer(
        config, target_builder=Builder(), token_clip=clip
    ), calls


def messages():
    return [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "help"},
        {"role": "assistant", "content": "standard answer"},
    ]


def test_reward_is_four_term_analytical_objective():
    result = score_hinter_hint(
        mean_lift=3.0,
        mean_copy=0.75,
        mean_dose_kl=0.3,
        hint_tokens=4,
        config=HinterRewardConfig(
            copying_weight=2.0,
            dose_weight=0.5,
            dose_bandwidth=0.1,
            length_weight=0.1,
            max_hint_tokens=10,
        ),
    )
    assert result.copying_penalty == pytest.approx(1.5)
    assert result.dose == pytest.approx(0.2)
    assert result.dose_penalty == pytest.approx(0.1)
    assert result.length_penalty == pytest.approx(0.4)
    assert result.reward == pytest.approx(1.0)
    with pytest.raises(ValueError, match="hard token cap"):
        score_hinter_hint(
            mean_lift=100,
            mean_copy=0,
            mean_dose_kl=0,
            hint_tokens=11,
            config=HinterRewardConfig(max_hint_tokens=10),
        )


def test_usefulness_uses_four_teacher_forced_views_on_same_tau_star():
    utility, calls = scorer((-4.0, -4.0))
    result = utility.score(
        student_visible_messages=messages(),
        hint="Ask for the missing lookup key.",
        state_hash="state",
    )
    assert len(calls) == 4
    assert all(call["checkpoint_id"] == "student-checkpoint" for call in calls)
    assert all(call["messages"][-1] == messages()[-1] for call in calls)
    hint_only = next(
        call for call in calls if call["information_view"].startswith("hinter_reward_hint_only")
    )
    assert [row["role"] for row in hint_only["messages"]] == ["system", "assistant"]
    empty = next(
        call for call in calls if call["information_view"] == "hinter_reward_empty_context"
    )
    assert [row["role"] for row in empty["messages"]] == ["system", "assistant"]
    assert result.unhinted_log_probability == pytest.approx(-6.0)
    assert result.hinted_log_probability == pytest.approx(-2.0)
    assert result.mean_lift == pytest.approx(2.0)
    assert result.mean_copy == pytest.approx(0.0)
    assert result.probability_trace.token_lifts == pytest.approx((2.0, 2.0))


def test_copy_answer_hint_is_positive_but_clean_l2_is_near_zero():
    copying, _ = scorer((-0.5, -0.5))
    clean, _ = scorer((-4.0, -4.0))
    copied = copying.score(
        student_visible_messages=messages(),
        hint="The exact answer is standard answer.",
        state_hash="copy",
    )
    procedural = clean.score(
        student_visible_messages=messages(),
        hint="Retrieve the missing public evidence before deciding.",
        state_hash="clean",
    )
    assert copied.mean_copy > 2.0
    assert procedural.mean_copy == pytest.approx(0.0)


def test_copy_does_not_charge_context_removal_prior():
    prior_only, _ = scorer((-0.5, -0.5), empty_logs=(-0.5, -0.5))
    result = prior_only.score(
        student_visible_messages=messages(),
        hint="Observe before acting.",
        state_hash="prior",
    )
    assert result.hint_only_log_probability > result.unhinted_log_probability
    assert result.mean_copy == pytest.approx(0.0)


def test_per_token_lift_and_copy_are_clipped():
    utility, _ = scorer((3.0, 3.0), clip=1.25)
    result = utility.score(
        student_visible_messages=messages(), hint="copy", state_hash="clip"
    )
    assert result.mean_lift == pytest.approx(1.25)
    assert result.mean_copy == pytest.approx(1.25)
    assert all(value > 1.25 for value in result.probability_trace.raw_token_lifts)
    assert all(value > 1.25 for value in result.probability_trace.raw_token_copies)


def test_l3_anchor_calibrates_lambda():
    weight = calibrate_copying_weight(
        l3_mean_lift=0.8, l3_mean_copy=0.4, target_margin=0.2
    )
    assert weight == pytest.approx(2.5)
    with pytest.raises(ValueError, match="positive analytical copy"):
        calibrate_copying_weight(l3_mean_lift=1.0, l3_mean_copy=0.0)


def test_rule_gate_still_clamps_reward_negative():
    config = HinterRewardConfig(rule_leak_floor=1.0)
    breakdown = score_hinter_hint(
        mean_lift=4.0,
        mean_copy=0.0,
        mean_dose_kl=0.0,
        hint_tokens=2,
        config=config,
        rule_leaks=("identifier",),
    )
    assert breakdown.reward <= -config.rule_leak_floor


def test_reward_callback_has_no_rollout_or_discriminator_dependency():
    reward = object.__new__(HinterCompositeReward)
    reward.reward_config = HinterRewardConfig(
        copying_weight=1.0,
        dose_weight=1.0,
        dose_bandwidth=0.1,
        length_weight=0.1,
        max_hint_tokens=20,
    )

    class Signals:
        mean_lift = 2.0
        mean_copy = 0.5
        mean_dose_kl = 0.3

        @staticmethod
        def to_dict():
            return {"mean_lift": 2.0, "mean_copy": 0.5, "mean_dose_kl": 0.3}

    reward.usefulness = SimpleNamespace(score=lambda **_kwargs: Signals())
    reward.hinter_tokenizer = SimpleNamespace(
        encode=lambda hint, add_special_tokens=False: hint.split()
    )
    reward.trace_path = ""
    reward._trace_lock = None
    visible = [
        {"role": "system", "content": "policy"},
        {"role": "assistant", "content": "standard"},
    ]
    values = reward(
        ["short hint", "different hint"],
        state_hash=["s", "s"],
        public_state=[[{"role": "user", "content": "x"}]] * 2,
        student_visible_messages=[visible, visible],
        tools=[[], []],
        privileged_context=[
            {"authoritative_oracle_steps": "private"},
            {"authoritative_oracle_steps": "private"},
        ],
    )
    assert values == pytest.approx([1.1, 1.1])


def test_grpo_dataset_uses_one_fixed_l3_standard_action_per_state():
    standard_messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "help"},
        {"role": "assistant", "content": "verified standard"},
    ]
    common = {
        "state_hash": "s",
        "public_state": [{"role": "user", "content": "help"}],
        "student_profile": {
            "unhinted_success": 0.2,
            "curriculum_band": "frontier",
        },
        "privileged_context": {
            "domain_policy": "policy",
            "authoritative_oracle_steps": "private",
        },
        "tools": [],
        "standard_action_eligible": True,
    }
    rows = [
        {
            **common,
            "hint_level": "L2_PROCEDURAL",
            "student_visible_messages": [
                *standard_messages[:-1],
                {"role": "assistant", "content": "other"},
            ],
        },
        {
            **common,
            "hint_level": "L3_ORACLE",
            "student_visible_messages": standard_messages,
        },
    ]
    result = build_hinter_grpo_dataset(rows)
    assert len(result) == 1
    assert result[0].student_visible_messages[-1]["content"] == "verified standard"
    assert result[0].standard_source_level == "L3_ORACLE"
    assert result[0].scenario_id == "s"


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
