import pytest

from coevo.audit import (
    ConditionalLeakageProbe,
    build_matched_shuffled_examples,
    roc_auc,
)


def test_pairing_uses_hidden_record_from_a_different_group():
    rows = [
        {"state_id": "a:0", "group_id": "a", "state": "s1", "action": "a1", "hidden": "h1"},
        {"state_id": "b:0", "group_id": "b", "state": "s2", "action": "a2", "hidden": "h2"},
    ]
    examples = build_matched_shuffled_examples(rows)
    negatives = [example for example in examples if not example.matched]
    assert {example.hidden for example in negatives} == {"h1", "h2"}
    assert all(
        not (example.state == "s1" and example.hidden == "h1")
        and not (example.state == "s2" and example.hidden == "h2")
        for example in negatives
    )


def test_conditional_probe_reports_advantage_over_state_only():
    rows = [
        {"state_id": "a", "group_id": "a", "state": "same", "action": "mentions h1", "hidden": "h1"},
        {"state_id": "b", "group_id": "b", "state": "same", "action": "mentions h2", "hidden": "h2"},
    ]
    examples = build_matched_shuffled_examples(rows)
    probe = ConditionalLeakageProbe(
        lambda example: float(example.hidden in example.action),
        lambda _example: 0.5,
    )
    report = probe.evaluate(examples)
    assert report.conditional_auc == pytest.approx(1.0)
    assert report.state_only_auc == pytest.approx(0.5)
    assert report.conditional_advantage == pytest.approx(0.5)
    assert roc_auc([1, 0], [0.5, 0.5]) == pytest.approx(0.5)
