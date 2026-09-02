import pytest

from scripts.collect_dosage_curriculum import select_curriculum_tasks


def test_manifest_consumer_samples_only_trainable_tasks_and_preserves_levels():
    manifest = {
        "sampling_weights": {"a": 0.7, "b": 0.2, "c": 0.1},
        "decisions": {
            "a": {"level": "L1_POLICY"},
            "b": {"level": "L3_ORACLE"},
            "c": {"level": None},
        },
    }
    selections, dropped, mass = select_curriculum_tasks(manifest, 20, seed=7)
    assert mass == pytest.approx(0.9)
    assert dropped["c"]["reason"] == "no trainable hint level"
    assert {row["task_id"] for row in selections} <= {"a", "b"}
    assert {
        row["hint_level"] for row in selections if row["task_id"] == "a"
    } == {"L1_POLICY"}
    assert {
        row["hint_level"] for row in selections if row["task_id"] == "b"
    } == {"L3_ORACLE"}
