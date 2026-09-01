import math

import pytest

from coevo.scoring.skill_contrast import (
    SkillContrastConfig,
    construct_skill_contrast_target,
)


def build(hinted, unhinted, config=None):
    return construct_skill_contrast_target(
        hinted_topk_logprobs=[[math.log(value) for value in hinted]],
        hinted_topk_token_ids=[[3, 7]],
        unhinted_topk_logprobs=[[math.log(value) for value in unhinted]],
        unhinted_topk_token_ids=[[3, 7]],
        target_token_ids=[3],
        config=config
        or SkillContrastConfig(minimum_support_mass=0.9),
    )


def test_identical_views_recover_raw_hinted_target_exactly():
    result = build([0.8, 0.2], [0.8, 0.2])

    assert result.skill_contrast_scores == pytest.approx([0.0], abs=1e-12)
    assert result.skill_gate_values == pytest.approx([0.0])
    assert result.sharpening_temperatures == pytest.approx([1.0])
    assert result.sharpened_topk_logprobs[0] == pytest.approx(
        [math.log(0.8), math.log(0.2)]
    )


def test_larger_skill_contrast_sharpens_without_changing_mode_or_order():
    config = SkillContrastConfig(
        low=0.01,
        high=0.2,
        minimum_temperature=0.5,
        minimum_support_mass=0.9,
    )
    weak = build([0.8, 0.2], [0.75, 0.25], config)
    strong = build([0.8, 0.2], [0.2, 0.8], config)

    assert strong.skill_contrast_scores[0] > weak.skill_contrast_scores[0]
    assert strong.sharpening_temperatures[0] < weak.sharpening_temperatures[0]
    assert 0.5 <= strong.sharpening_temperatures[0] <= 1.0
    assert strong.sharpened_teacher_entropy[0] <= strong.raw_teacher_entropy[0]
    assert strong.sharpened_topk_token_ids[0] == (3, 7)
    assert strong.sharpened_topk_logprobs[0][0] > (
        strong.sharpened_topk_logprobs[0][1]
    )


def test_missing_target_or_low_teacher_support_fails_closed():
    with pytest.raises(ValueError, match="support mass"):
        build(
            [0.4, 0.2],
            [0.4, 0.2],
            SkillContrastConfig(minimum_support_mass=0.95),
        )
    with pytest.raises(ValueError, match="actual target token"):
        construct_skill_contrast_target(
            hinted_topk_logprobs=[[math.log(0.95)]],
            hinted_topk_token_ids=[[7]],
            unhinted_topk_logprobs=[[math.log(0.95)]],
            unhinted_topk_token_ids=[[7]],
            target_token_ids=[3],
            config=SkillContrastConfig(minimum_support_mass=0.9),
        )


def test_disabled_sharpening_preserves_raw_hinted_distribution():
    result = build(
        [0.8, 0.2],
        [0.2, 0.8],
        SkillContrastConfig(
            low=0.0,
            high=0.05,
            minimum_temperature=0.5,
            minimum_support_mass=0.9,
            sharpen_enabled=False,
        ),
    )

    assert result.skill_gate_values[0] > 0
    assert result.sharpening_temperatures == pytest.approx([1.0])
    assert result.sharpened_topk_logprobs[0] == pytest.approx(
        [math.log(0.8), math.log(0.2)]
    )
