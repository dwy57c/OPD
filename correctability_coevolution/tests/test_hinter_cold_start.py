import pytest

from coevo.hinter_training import ColdStartSource, build_hinter_cold_start_dataset


def audit_row(task_id, level, hint):
    return {
        "task_id": task_id,
        "state_id": f"{task_id}:0",
        "state_hash": f"state-{task_id}",
        "hint_level": level,
        "hint": {"hint": {"plan": hint}},
        "hint_error": None,
        "analytical_signals": {"mean_copy": 0.02},
        "public_state": [{"role": "user", "content": "help"}],
        "privileged_context": {
            "domain_policy": "policy",
            "authoritative_oracle_steps": "oracle",
        },
    }


def source(checkpoint):
    return ColdStartSource(
        checkpoint,
        (
            audit_row("a", "L1_POLICY", "Observe first and confirm each result."),
            audit_row("b", "L2_PROCEDURAL", "Acquire the missing evidence first."),
        ),
        {
            "decisions": {
                "a": {"level": "L1_POLICY"},
                "b": {"level": "L2_PROCEDURAL"},
            }
        },
    )


def test_cold_start_requires_multiple_checkpoints_and_doses():
    with pytest.raises(ValueError, match="two Student checkpoints"):
        build_hinter_cold_start_dataset([source("student-a")])

    rows = build_hinter_cold_start_dataset(
        [source("student-a"), source("student-b")]
    )
    assert {row["student_checkpoint"] for row in rows} == {
        "student-a",
        "student-b",
    }
    assert {row["minimal_sufficient_level"] for row in rows} == {
        "L1_POLICY",
        "L2_PROCEDURAL",
    }
    assert all(row["messages"][-1]["content"].startswith("level: L") for row in rows)
    student_inputs = {
        row["student_checkpoint"]: row["messages"][1]["content"] for row in rows
    }
    assert student_inputs["student-a"] != student_inputs["student-b"]


def test_cold_start_drops_high_copy_rows():
    first = source("student-a")
    dirty_rows = []
    for row in source("student-b").audit_rows:
        dirty_rows.append({**row, "analytical_signals": {"mean_copy": 0.8}})
    dirty = ColdStartSource(
        "student-b", tuple(dirty_rows), source("student-b").hstar_manifest
    )
    with pytest.raises(ValueError, match="retain low-copy rows"):
        build_hinter_cold_start_dataset([first, dirty], max_mean_copy=0.1)
