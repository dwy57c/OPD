from scripts import run_coevolution


def test_stage_reward_rollback_restores_last_committed_policy_and_buyer(monkeypatch):
    calls = []

    def fake_run_best_effort(command, env, errors):
        calls.append((command, dict(env)))

    monkeypatch.setattr(run_coevolution, "run_best_effort", fake_run_best_effort)

    errors = run_coevolution.rollback_services(
        {"COEVO_PREVIOUS_POLICY_PATH": "/checkpoints/pre-update"},
        "/checkpoints/student-committed",
        "/checkpoints/buyer-committed",
    )

    assert errors == []
    assert [command[2] for command, _ in calls[:4]] == [
        "rollout",
        "policy_previous",
        "policy",
        "buyer",
    ]
    assert [command[2] for command, _ in calls[4:]] == [
        "policy",
        "buyer",
        "rollout",
    ]
    for _, env in calls:
        assert env["COEVO_CURRENT_POLICY_CHECKPOINT"] == (
            "/checkpoints/student-committed"
        )
        assert "COEVO_PREVIOUS_POLICY_CHECKPOINT" not in env
        assert env["COEVO_BUYER_CHECKPOINT"] == "/checkpoints/buyer-committed"


def test_checkpoint_pair_rejects_missing_or_identical():
    import pytest

    with pytest.raises(ValueError, match="both previous and current"):
        run_coevolution.validate_checkpoint_order("", "current")
    with pytest.raises(ValueError, match="identical"):
        run_coevolution.validate_checkpoint_order("same", "same")
    run_coevolution.validate_checkpoint_order("before", "after")
