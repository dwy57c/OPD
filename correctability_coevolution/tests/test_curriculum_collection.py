import pytest

from scripts.collect_dosage_curriculum import select_curriculum_tasks


def test_manifest_consumer_uses_hstar_only_for_weights_and_excludes_mastered():
    manifest = {
        "sampling_weights": {"a": 0.7, "b": 0.2, "c": 0.1},
        "decisions": {
            "a": {"level": "L1_POLICY"},
            "b": {"level": "L3_ORACLE"},
            "c": {"level": "L0_NONE", "band": "mastered"},
        },
    }
    selections, dropped, mass = select_curriculum_tasks(manifest, 20, seed=7)
    assert mass == pytest.approx(0.9)
    assert "excluded" in dropped["c"]["reason"]
    assert {row["task_id"] for row in selections} <= {"a", "b"}
    assert {row["hint_level"] for row in selections} == {"HINTER"}
