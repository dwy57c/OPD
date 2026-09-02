from scripts.collect_behavior_hint_samples import validate_hint_group


def test_duplicate_hint_group_is_skipped_instead_of_raising(capsys):
    assert not validate_hint_group(["same", "same"], "state")
    captured = capsys.readouterr()
    assert "skip_duplicate_hint_group" in captured.err
    assert validate_hint_group(["first", "second"], "state")
