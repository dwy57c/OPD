import pytest

from coevo.hinter_training import AcceptanceRule, AlternatingHinterLoop, PassKSnapshot


def test_one_measurement_accepts_or_rolls_back_then_trains_hinter():
    events = []
    baseline = PassKSnapshot({"s": 0.75}, k=8)
    loop = AlternatingHinterLoop(
        train_student=lambda student, hinter, steps: events.append(
            ("student", student, hinter, steps)
        )
        or "student-candidate",
        measure_pass_at_k=lambda student, panel, k: events.append(
            ("pass", student, k)
        )
        or PassKSnapshot({"s": 0.50}, k=8),
        schedule_curriculum=lambda snapshot, pool: events.append(
            ("schedule", snapshot.mean)
        )
        or pool,
        train_hinter_grpo=lambda student, hinter, curriculum, steps: events.append(
            ("hinter-grpo", student, hinter, steps)
        )
        or "next-hinter-candidate",
        rollback_student=lambda candidate, previous: events.append(
            ("rollback-student", candidate, previous)
        ),
        rollback_hinter=lambda candidate, previous: events.append(
            ("rollback-hinter", candidate, previous)
        ),
        acceptance=AcceptanceRule(
            mean_tolerance=0.0,
            per_scenario_drop=0.25,
            max_regressed_fraction=0.35,
        ),
    )
    result = loop.run_round(
        round_index=3,
        student_checkpoint="student-old",
        hinter_under_test="hinter-candidate",
        fallback_hinter_checkpoint="hinter-accepted",
        scenario_pool={"s": {}},
        acceptance_baseline=baseline,
        student_steps=10,
        hinter_grpo_steps=4,
        pass_k=8,
    )
    assert [event[0] for event in events] == [
        "student",
        "pass",
        "rollback-student",
        "rollback-hinter",
        "schedule",
        "hinter-grpo",
    ]
    assert result.pass_measurements_this_round == 1
    assert result.prior_hinter_rolled_back
    assert result.student_after == "student-old"
    assert result.accepted_hinter == "hinter-accepted"
    assert result.measured_distillation_gain == -0.25
    assert result.next_hinter_candidate == "next-hinter-candidate"
    assert any(reason.startswith("mean_pass_at_k") for reason in result.rollback_reasons)
    assert ("schedule", 0.75) in events


def test_first_round_accepts_without_baseline_and_still_measures_once():
    measurements = []
    loop = AlternatingHinterLoop(
        train_student=lambda _student, _hinter, _steps: "student-new",
        measure_pass_at_k=lambda _student, _panel, k: measurements.append(k)
        or PassKSnapshot({"s": 0.5}, k),
        schedule_curriculum=lambda _snapshot, pool: pool,
        train_hinter_grpo=lambda *_args: "hinter-next",
        rollback_student=lambda *_args: None,
        rollback_hinter=lambda *_args: None,
    )
    result = loop.run_round(
        round_index=0,
        student_checkpoint="student-base",
        hinter_under_test="hinter-base",
        fallback_hinter_checkpoint="hinter-base",
        scenario_pool={"s": {}},
        acceptance_baseline=None,
        student_steps=2,
        hinter_grpo_steps=1,
    )
    assert measurements == [8]
    assert not result.prior_hinter_rolled_back
    assert result.accepted_hinter == "hinter-base"


def test_acceptance_requires_identical_pass_panel():
    with pytest.raises(ValueError, match="identical"):
        AcceptanceRule().regressions(
            PassKSnapshot({"a": 1.0}, 8), PassKSnapshot({"b": 1.0}, 8)
        )


def test_acceptance_ignores_one_noisy_scenario_but_rejects_panel_regression():
    baseline = PassKSnapshot({str(index): 0.5 for index in range(20)}, 8)
    one_drop = dict(baseline.scores)
    one_drop["0"] = 0.25
    rule = AcceptanceRule()
    assert rule.regressions(baseline, PassKSnapshot(one_drop, 8)) == ()

    broad_drop = dict(baseline.scores)
    for index in range(8):
        broad_drop[str(index)] = 0.25
    failures = rule.regressions(baseline, PassKSnapshot(broad_drop, 8))
    assert any(value.startswith("mean_pass_at_k") for value in failures)
    assert any(value.startswith("scenario_fraction") for value in failures)
