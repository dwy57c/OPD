import math

import pytest

from coevo.scoring.purified_target import construct_purified_target
from coevo.scoring.stage_gap import SparseTargetView


def view(first, second):
    return SparseTargetView(
        target_input_ids=(1,),
        topk_logprobs=((math.log(first), math.log(second)),),
        topk_token_ids=((1, 2),),
        support_mass=(first + second,),
    )


def test_purified_target_removes_hint_only_component():
    base = view(0.5, 0.5)
    hint_only = view(0.8, 0.2)
    same_as_hint_only = view(0.8, 0.2)
    purified = construct_purified_target(
        unhinted=base,
        hinted=same_as_hint_only,
        hint_only=hint_only,
        beta=1.0,
    )
    probabilities = [math.exp(value) for value in purified.sharpened_topk_logprobs[0]]
    assert probabilities == pytest.approx([0.5, 0.5])


def test_purified_target_keeps_state_conditioned_residual():
    result = construct_purified_target(
        unhinted=view(0.5, 0.5),
        hinted=view(0.9, 0.1),
        hint_only=view(0.5, 0.5),
        beta=1.0,
    )
    probabilities = [math.exp(value) for value in result.sharpened_topk_logprobs[0]]
    assert probabilities[0] > probabilities[1]
