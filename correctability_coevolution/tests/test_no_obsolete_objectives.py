from pathlib import Path


def test_removed_objective_identifiers_do_not_return():
    root = Path(__file__).resolve().parents[1]
    forbidden = (
        "shadow_" + "opd",
        "opd_" + "utility",
        "Utility" + "Critic",
        "Utility" + "Features",
        "shadow_" + "gain",
    )
    matches = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".md", ".sh"}:
            continue
        if path == Path(__file__):
            continue
        text = path.read_text(errors="ignore")
        matches.extend(f"{path}:{term}" for term in forbidden if term in text)
    assert not matches, "removed objective stack was reintroduced: " + ", ".join(matches)


def test_buyer_training_uses_group_reward_normalization():
    root = Path(__file__).resolve().parents[1]
    for name in ("train_buyer_smoke.sh", "train_buyer_full.sh"):
        text = (root / "scripts" / name).read_text()
        assert "--scale_rewards group" in text
        assert "--scale_rewards none" not in text
        assert "COEVO_BUYER_BETA:-0.01" in text
        assert "COEVO_BUYER_ATTN_IMPL:-flash_attn" in text
        assert '--attn_impl "$BUYER_ATTN_IMPL"' in text
        assert "--logging_nan_inf_filter false" in text
        assert "check_adapter_finite.py" in text


def test_main_reward_and_scheduler_do_not_use_extra_curriculum_terms():
    root = Path(__file__).resolve().parents[1]
    reward_text = (root / "coevo/rewards/stage_progress.py").read_text()
    scheduler_text = (root / "coevo/training/buyer_scheduler.py").read_text()
    for term in ("teacher_quality", "residual_gate", "exploration_bonus"):
        assert term not in reward_text
        assert term not in scheduler_text
    assert "build_teacher_target_labeler" in scheduler_text
    assert "build_teacher_target_validator" not in scheduler_text
