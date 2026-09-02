import json
from pathlib import Path


CONTROL_FIELDS = (
    "ordinary_pair_accuracy",
    "explicit_copy_accuracy",
    "explicit_copy_natural_accuracy",
    "useless_mean_distance_from_chance",
    "ordinary_pairs",
    "explicit_copy_pairs",
    "explicit_copy_natural_pairs",
    "useless_pairs",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def control_report(report: dict) -> dict:
    missing = [field for field in CONTROL_FIELDS if field not in report]
    if missing:
        raise ValueError(f"discriminator report is missing controls: {missing}")
    return {field: report[field] for field in CONTROL_FIELDS}
