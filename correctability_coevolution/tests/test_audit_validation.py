import pytest

from coevo.audit import binary_agreement, validate_annotation_rows


def test_binary_agreement_reports_exact_kappa():
    report = binary_agreement([1, 1, 0, 0], [1, 0, 0, 0])
    assert report.agreement == pytest.approx(0.75)
    assert report.cohen_kappa == pytest.approx(0.5)


def test_validation_requires_paired_detector_labels():
    rows = [
        {
            "human_clarifying": True,
            "judge_clarifying": True,
            "human_lookup": False,
            "judge_lookup": False,
            "human_ungrounded": True,
            "judge_ungrounded": False,
        }
    ]
    assert set(validate_annotation_rows(rows)) == {"clarifying", "lookup", "ungrounded"}
