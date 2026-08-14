import pytest

from coevo.artifacts import dataset_fingerprint, validate_compatible_artifacts
from coevo.orchestration.collection import _validate_resume_fingerprint
from scripts.merge_collection_shards import normalize_chat_row


def contract(**overrides):
    value = {
        "schema_version": 4,
        "target_schema_version": 2,
        "round_index": 1,
        "tokenizer_id": "qwen3-tokenizer@revision",
        "tokenizer_hash": "tokenizer-hash",
        "teacher_target_version": "skill-contrast-sharpened-v2",
        "reward_name": "tau2_stage_learning_progress",
        "reward_formula_version": "previous-skill-anchor-progress-v3",
        "student_checkpoint_current": "/checkpoints/current",
        "student_checkpoint_previous": "/checkpoints/previous",
        "buyer_checkpoint": "/checkpoints/buyer",
        "student_revision_current": "current-revision",
        "student_revision_previous": "previous-revision",
        "buyer_revision": "buyer-revision",
    }
    value.update(overrides)
    return value


def test_artifact_contract_rejects_incompatible_tokenizer_or_target():
    with pytest.raises(ValueError, match="incompatible artifact contracts"):
        validate_compatible_artifacts(
            [("first", contract()), ("second", contract(tokenizer_id="other"))]
        )
    with pytest.raises(ValueError, match="incompatible artifact contracts"):
        validate_compatible_artifacts(
            [
                ("first", contract()),
                ("second", contract(teacher_target_version="other-target")),
            ]
        )


def test_artifact_fingerprint_is_deterministic_and_order_sensitive():
    first = {"b": 2, "a": 1}
    second = {"value": 3}
    assert dataset_fingerprint([first], [second]) == dataset_fingerprint(
        [{"a": 1, "b": 2}],
        [second],
    )
    assert dataset_fingerprint([first], [second]) != dataset_fingerprint(
        [second],
        [first],
    )


def test_resume_fingerprint_rejects_partial_or_modified_collection():
    records = [{"trajectory": 1}]
    student_rows = [{"student": 1}]
    buyer_rows = [{"buyer": 1}]
    summary = {
        "dataset_fingerprint": dataset_fingerprint(
            records, student_rows, buyer_rows
        )
    }

    _validate_resume_fingerprint(summary, records, student_rows, buyer_rows)
    with pytest.raises(ValueError, match="partial or modified write"):
        _validate_resume_fingerprint(
            summary,
            [*records, {"trajectory": 2}],
            student_rows,
            buyer_rows,
        )


def test_tool_action_normalization_preserves_cached_training_target():
    action = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"function": {"name": "lookup", "arguments": "{}"}}],
    }
    row = {
        "messages": [dict(action)],
    }
    normalized = normalize_chat_row(row)

    assert normalized["messages"][-1]["content"] == ""
