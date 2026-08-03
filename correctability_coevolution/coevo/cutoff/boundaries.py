from dataclasses import asdict, dataclass
import re


_SENTENCE_END = re.compile(r"[.!?。！？](?:[\"'”’])?(?=\s|$)")


@dataclass(frozen=True)
class CutoffCandidate:
    candidate_id: int
    char_offset: int
    boundary_type: str
    prefix_tail: str
    suffix_head: str

    def to_dict(self) -> dict:
        return asdict(self)


def semantic_boundaries(
    text: str, min_prefix_chars: int = 24, min_suffix_chars: int = 16
) -> list[CutoffCandidate]:
    """Return replayable sentence boundaries inside one completed Student turn."""
    offsets = []
    for match in _SENTENCE_END.finditer(text):
        offset = match.end()
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset >= min_prefix_chars and len(text) - offset >= min_suffix_chars:
            offsets.append(offset)

    candidates = []
    for candidate_id, offset in enumerate(dict.fromkeys(offsets), start=1):
        candidates.append(
            CutoffCandidate(
                candidate_id=candidate_id,
                char_offset=offset,
                boundary_type="sentence",
                prefix_tail=text[max(0, offset - 80) : offset],
                suffix_head=text[offset : offset + 80],
            )
        )
    return candidates
