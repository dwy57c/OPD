import socket
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "wait_for_free_port.py"


def test_wait_for_free_port_accepts_bindable_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("", 0))
        port = handle.getsockname()[1]
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(port), "--timeout", "0.2"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0


def test_wait_for_free_port_rejects_occupied_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("", 0))
        port = handle.getsockname()[1]
        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(port), "--timeout", "0.1"],
            capture_output=True,
            text=True,
            check=False,
        )
    assert result.returncode != 0
    assert f"port {port} did not become bindable" in result.stderr
