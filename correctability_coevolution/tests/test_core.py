import pytest
import torch
from tau2.data_model.message import AssistantMessage, UserMessage

from coevo.cutoff import SelectedCutoff, semantic_boundaries
from coevo.rewards.buyer import buyer_reward, trajectory_buyer_reward
from coevo.rewards.correctability import CorrectabilityResult
from coevo.rollout.cutoff_scorer import TurnCutoffScorer
from coevo.training.gates import gated_example_mean


def test_correctability_formula():
    q_teacher = (1 + 1) / 3
    q_student = (0 + 1) / 3
    result = CorrectabilityResult(1, 0, 1, q_teacher, q_student, q_teacher * (1 - q_student))
    assert result.correctability == pytest.approx(4 / 9)


def test_buyer_reward_is_validity_gated():
    assert buyer_reward(4 / 9, 1.0) == pytest.approx(4 / 9)
    assert buyer_reward(4 / 9, 0.0) == 0.0
    assert trajectory_buyer_reward([0.2, 0.6], 1.0) == pytest.approx(0.4)
    assert trajectory_buyer_reward([0.2, 0.6], 0.0) == 0.0


def test_semantic_boundaries_are_inside_completed_turn():
    text = (
        "First, I checked the account and found the reservation. "
        "Next, I will inspect the requested route carefully. "
        "Finally, I will explain the available action to the user."
    )
    candidates = semantic_boundaries(text)
    assert len(candidates) == 2
    assert all(0 < candidate.char_offset < len(text) for candidate in candidates)
    assert text[: candidates[0].char_offset].endswith(" ")


def test_turn_score_means_teacher_selected_cutoffs():
    class Selector:
        def select(self, history, output, candidates):
            return [SelectedCutoff(candidate, "test") for candidate in candidates[:2]]

    class Estimator:
        def estimate(self, history):
            return CorrectabilityResult(1, 0, 1, 2 / 3, 1 / 3, 4 / 9)

    history = [
        UserMessage(role="user", content="Please inspect my reservation."),
        AssistantMessage(
            role="assistant",
            content=(
                "First, I checked the account and found the reservation. "
                "Next, I will inspect the requested route carefully. "
                "Finally, I will explain the available action to the user."
            ),
        ),
    ]
    scored = TurnCutoffScorer(Selector(), Estimator()).score_turn(history, 1)
    assert len(scored["cutoffs"]) == 2
    assert scored["correctability"] == pytest.approx(4 / 9)
    assert all(len(item["state_hash"]) == 64 for item in scored["cutoffs"])


def test_gkd_gate_is_applied_per_example():
    loss = gated_example_mean([torch.tensor(2.0), torch.tensor(4.0)], [0.0, 0.5])
    assert loss.item() == pytest.approx(1.0)
