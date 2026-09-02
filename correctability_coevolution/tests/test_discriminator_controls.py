from scripts.collect_discriminator_controls import entity_perturbed_decoy


def test_natural_copy_negative_changes_entities_but_preserves_style():
    original = (
        "The booking ABC123 is for 2027-05-03 and carries a $75 fee; confirm it."
    )
    decoy = entity_perturbed_decoy(original)
    assert decoy is not None
    assert decoy != original
    assert "The booking" in decoy
    assert "carries a" in decoy
    assert "confirm it." in decoy


def test_natural_copy_negative_falls_back_when_no_entity_exists():
    assert entity_perturbed_decoy("Ask for the booking number first.") is None
