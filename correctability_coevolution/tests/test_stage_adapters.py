import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run_adapter(name, *args):
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "stages" / name), *map(str, args)],
        cwd=ROOT,
        check=True,
    )


def test_real_stage_adapters_convert_fixture_artifacts(tmp_path):
    dosage = tmp_path / "dosage.json"
    dosage.write_text(
        json.dumps(
            {
                "k": 8,
                "probes": {
                    "a": {"L0_NONE": {"success_rate": 0.5}},
                    "b": {"L0_NONE": {"success_rate": 0.25}},
                },
                "sampling_weights": {"a": 0.8, "b": 0.2},
                "decisions": {
                    "a": {"level": "L1_POLICY"},
                    "b": {"level": "L2_PROCEDURAL"},
                },
            }
        ),
        encoding="utf-8",
    )
    pass_output = tmp_path / "pass.json"
    run_adapter("probe_result.py", "--manifest", dosage, "--output", pass_output)
    assert json.loads(pass_output.read_text())["scores"] == {"a": 0.5, "b": 0.25}

    schedule_output = tmp_path / "schedule.json"
    run_adapter(
        "schedule_result.py",
        "--manifest",
        dosage,
        "--output",
        schedule_output,
        "--samples",
        4,
    )
    assert set(json.loads(schedule_output.read_text())["scenario_ids"]) <= {"a", "b"}

    checkpoint_root = tmp_path / "student"
    checkpoint = checkpoint_root / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    checkpoint_output = tmp_path / "checkpoint.json"
    run_adapter(
        "checkpoint_result.py",
        "--checkpoint-root",
        checkpoint_root,
        "--output",
        checkpoint_output,
    )
    assert json.loads(checkpoint_output.read_text())["checkpoint"] == str(
        checkpoint.resolve()
    )

    report = tmp_path / "discriminator_report.json"
    control = {
        "ordinary_pair_accuracy": 0.8,
        "explicit_copy_accuracy": 1.0,
        "explicit_copy_natural_accuracy": 0.9,
        "useless_mean_distance_from_chance": 0.05,
        "ordinary_pairs": 2,
        "explicit_copy_pairs": 2,
        "explicit_copy_natural_pairs": 2,
        "useless_pairs": 2,
        "initialized_from_student": "/student",
        "fresh_score_head": True,
    }
    report.write_text(json.dumps(control), encoding="utf-8")
    ordinary = tmp_path / "ordinary.jsonl"
    ordinary.write_text('{"state_hash":"s"}\n', encoding="utf-8")
    discriminator_output = tmp_path / "discriminator.json"
    run_adapter(
        "discriminator_result.py",
        "--report",
        report,
        "--checkpoint",
        checkpoint,
        "--ordinary-pairs",
        ordinary,
        "--round-index",
        3,
        "--output",
        discriminator_output,
    )
    result = json.loads(discriminator_output.read_text())
    assert result["checkpoint"] == str(checkpoint.resolve())
    assert result["round_index"] == 3
    assert result["converged"] is True
    assert result["control_report"]["explicit_copy_natural_accuracy"] == 0.9
