from coevo.config import InfraConfig


def test_branch_token_budget_reaches_policy_endpoints(monkeypatch):
    monkeypatch.setenv("COEVO_TEACHER_HINT_MODE", "none")
    monkeypatch.setenv("COEVO_POLICY_PATH", "/models/current")
    monkeypatch.setenv("COEVO_PREVIOUS_POLICY_PATH", "/models/previous")
    monkeypatch.setenv("COEVO_BRANCH_MAX_TOKENS", "73")
    monkeypatch.setenv("COEVO_NL_JUDGE_MAX_TOKENS", "91")
    monkeypatch.setenv("COEVO_SKILL_GATE_EPS", "1e-6")

    config = InfraConfig.from_env()

    assert config.branch_max_tokens == 73
    assert config.policy.max_tokens == 73
    assert config.previous_policy is not None
    assert config.previous_policy.max_tokens == 73
    assert config.buyer_reference.max_tokens == 73
    assert config.nl_judge is not None
    assert config.nl_judge.max_tokens == 91
    assert config.skill_gate_eps == 1e-6
