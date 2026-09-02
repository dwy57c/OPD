from coevo.config import HintEndpoint, InfraConfig, ModelEndpoint


def test_branch_token_budget_reaches_policy_endpoints(monkeypatch):
    monkeypatch.setenv("COEVO_TEACHER_HINT_MODE", "none")
    monkeypatch.setenv("COEVO_POLICY_PATH", "/models/current")
    monkeypatch.setenv("COEVO_PREVIOUS_POLICY_PATH", "/models/previous")
    monkeypatch.setenv("COEVO_TEACHER_ANCHOR", "previous")
    monkeypatch.setenv("COEVO_BRANCH_MAX_TOKENS", "73")
    monkeypatch.setenv("COEVO_NL_JUDGE_MAX_TOKENS", "91")
    monkeypatch.setenv("COEVO_SKILL_GATE_EPS", "1e-6")

    config = InfraConfig.from_env()

    assert config.branch_max_tokens == 73
    assert config.policy.max_tokens == 73
    assert config.previous_policy is not None
    assert config.previous_policy.max_tokens == 73
    assert config.teacher is config.previous_policy
    assert config.teacher_anchor_checkpoint == "/models/previous"
    assert config.buyer_reference.max_tokens == 73
    assert config.nl_judge is not None
    assert config.nl_judge.max_tokens == 91
    assert config.skill_gate_eps == 1e-6


def test_hint_endpoint_accepts_standard_openai_env_without_exposing_key(monkeypatch):
    monkeypatch.delenv("COEVO_TEACHER_HINT_URL", raising=False)
    monkeypatch.delenv("COEVO_TEACHER_HINT_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_BASE_URL", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-secret")
    monkeypatch.setenv("COEVO_TEACHER_HINT_MODEL", "qwen3.8-max")

    endpoint = HintEndpoint.from_env(required=True)

    assert endpoint is not None
    assert endpoint.base_url == "https://gateway.example/v1"
    assert endpoint.model == "qwen3.8-max"
    assert bool(endpoint.api_key)


def test_open_hinter_mode_is_a_supported_teacher_hint_source():
    config = InfraConfig(
        policy=ModelEndpoint("student", "http://student"),
        buyer_reference=ModelEndpoint("buyer", "http://buyer"),
        teacher_hint_mode="open_hinter",
        teacher_hinter=HintEndpoint("hinter", "http://hinter/v1", "key"),
    )
    assert config.teacher_hint_mode == "open_hinter"
