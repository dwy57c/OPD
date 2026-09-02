import json
from types import SimpleNamespace

from scripts.run_alternating_rounds import SubprocessRoundBackend


def commands():
    return {
        name: ["fake", name, "{output}"]
        for name in SubprocessRoundBackend.REQUIRED_STAGES
    }


def test_subprocess_driver_materializes_callback_outputs(monkeypatch, tmp_path):
    def fake_run(command, **kwargs):
        output = kwargs["env"].get("COEVO_STAGE_OUTPUT")
        if output:
            if command[1] == "train_student":
                payload = {"checkpoint": "student-new"}
            elif command[1] == "measure_pass_at_k":
                payload = {"k": 8, "scores": {"task": 0.75}}
            else:
                payload = {}
            with open(output, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("scripts.run_alternating_rounds.subprocess.run", fake_run)
    backend = SubprocessRoundBackend(commands(), tmp_path)
    assert backend.train_student("student", "hinter", 4) == "student-new"
    snapshot = backend.measure_pass_at_k("student-new", {"task": {}}, 8)
    assert snapshot.scores == {"task": 0.75}
    assert (tmp_path / "logs" / "train_student.stdout.log").is_file()
