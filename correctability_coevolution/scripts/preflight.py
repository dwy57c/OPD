#!/usr/bin/env python3
import argparse
from dataclasses import asdict, dataclass
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
TAU2_COMMIT = "17e07b1da2bbc0cadfddeea36412686e0604127b"
SWIFT_COMMIT = "c6875ef"
TAU2_SPLIT_COUNTS = {
    "airline": {"train": 30, "test": 20},
    "retail": {"train": 74, "test": 40},
    "telecom": {"train": 74, "test": 40},
}


@dataclass
class Check:
    name: str
    status: str
    detail: str


def add(checks: list[Check], name: str, ok: bool, detail: str, warn=False) -> None:
    status = "ok" if ok else ("warn" if warn else "fail")
    checks.append(Check(name, status, detail))


def tau2_source() -> Path:
    configured = os.getenv("COEVO_TAU2_SRC")
    if configured:
        return Path(configured).expanduser().resolve()
    candidates = (
        REPO_ROOT / "tau2-bench/src",
        REPO_ROOT / "third_party/tau2-bench/src",
        ROOT / "third_party/tau2-bench/src",
    )
    return next(
        (path for path in candidates if (path / "tau2").is_dir()), candidates[1]
    )


def git_revision(repository: Path) -> str | None:
    if not (repository / ".git").exists():
        return None
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repository}",
                "-C",
                str(repository),
                "rev-parse",
                "HEAD",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def check_python(checks: list[Check]) -> None:
    supported = (3, 11) <= sys.version_info[:2] <= (3, 12)
    add(
        checks,
        "python",
        supported,
        f"{sys.version.split()[0]} (this runtime: 3.11-3.12)",
    )
    tau2_official_python = (3, 12) <= sys.version_info[:2] < (3, 14)
    add(
        checks,
        "tau2-python-support",
        tau2_official_python,
        (
            f"{sys.version.split()[0]} is officially supported by tau2 v1"
            if tau2_official_python
            else (
                f"{sys.version.split()[0]} compatibility mode; tau2 v1 metadata "
                "requires >=3.12,<3.14"
            )
        ),
        warn=sys.version_info[:2] == (3, 11),
    )
    for module in (
        "requests",
        "torch",
        "openai",
        "tau2",
        "swift",
        "vllm",
        "trl",
        "wandb",
    ):
        present = importlib.util.find_spec(module) is not None
        add(checks, f"import:{module}", present, "available" if present else "missing")
    reporter = os.getenv("COEVO_REPORT_TO", "wandb")
    add(
        checks,
        "experiment-reporter",
        reporter == "wandb",
        f"report_to={reporter} (required: wandb; SwanLab is disabled)",
    )
    add(
        checks,
        "command:setsid",
        shutil.which("setsid") is not None,
        shutil.which("setsid")
        or "missing (required for isolated service process groups)",
    )

    try:
        version = importlib.metadata.version("ms_swift")
    except importlib.metadata.PackageNotFoundError:
        version = None
    add(
        checks,
        "ms-swift-version",
        version == "4.1.3",
        version or "not installed (expected 4.1.3 / commit c6875ef)",
    )

    try:
        plugin = subprocess.run(
            [sys.executable, "-c", "import coevo.training.swift_plugin"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        add(checks, "swift-plugin-import", False, "timed out after 120 seconds")
    else:
        plugin_error = (plugin.stderr or plugin.stdout).strip().splitlines()
        detail = (
            "passed"
            if plugin.returncode == 0
            else (plugin_error[-1] if plugin_error else "failed")
        )
        add(checks, "swift-plugin-import", plugin.returncode == 0, detail)


def check_upstreams(checks: list[Check]) -> None:
    source = tau2_source()
    source_present = (source / "tau2").is_dir()
    installed = importlib.util.find_spec("tau2") is not None
    add(
        checks,
        "tau2-source",
        source_present or installed,
        str(source) if source_present else "not checked out and not installed",
    )
    if source_present:
        data_dir = Path(os.getenv("TAU2_DATA_DIR", source.parent / "data"))
        required_data = tuple(
            data_dir / f"tau2/domains/{domain}/{filename}"
            for domain in TAU2_SPLIT_COUNTS
            for filename in ("tasks.json", "split_tasks.json")
        ) + (data_dir / "tau2/user_simulator/simulation_guidelines.md",)
        missing_data = [str(path) for path in required_data if not path.is_file()]
        add(
            checks,
            "tau2-data",
            not missing_data,
            str(data_dir)
            if not missing_data
            else f"missing: {', '.join(missing_data)}",
        )
        split_errors = []
        split_details = []
        if not missing_data:
            for domain, expected in TAU2_SPLIT_COUNTS.items():
                split_path = data_dir / f"tau2/domains/{domain}/split_tasks.json"
                try:
                    splits = json.loads(split_path.read_text())
                    train = splits["train"]
                    test = splits["test"]
                except (OSError, ValueError, KeyError, TypeError) as error:
                    split_errors.append(f"{domain}: {error}")
                    continue
                counts = {"train": len(train), "test": len(test)}
                if counts != expected or set(train) & set(test):
                    split_errors.append(
                        f"{domain}: counts={counts}, overlap={len(set(train) & set(test))}"
                    )
                split_details.append(
                    f"{domain}={counts['train']} train/{counts['test']} test"
                )
        add(
            checks,
            "tau2-v1-splits",
            not missing_data and not split_errors,
            "; ".join(split_errors or split_details),
        )
        revision = git_revision(source.parent) or os.getenv("COEVO_TAU2_REVISION")
        add(
            checks,
            "tau2-revision",
            revision is not None and revision.startswith(TAU2_COMMIT),
            revision or "source is not a git checkout",
            warn=revision is None,
        )

    swift_repo = REPO_ROOT / "third_party/ms-swift"
    if swift_repo.exists():
        revision = git_revision(swift_repo)
        add(
            checks,
            "ms-swift-revision",
            revision is not None and revision.startswith(SWIFT_COMMIT),
            revision or "source is not a git checkout",
            warn=revision is None,
        )


def local_model(reference: str) -> bool:
    return reference.startswith("/") or reference.startswith(".")


def stage_reward_needs_previous() -> bool:
    return bool(
        os.getenv("COEVO_PREVIOUS_POLICY_PATH")
        or os.getenv("COEVO_PREVIOUS_POLICY_CHECKPOINT")
    )


def validate_local_model(reference: str) -> tuple[bool, str]:
    model_dir = Path(reference).expanduser()
    if not model_dir.is_dir():
        return False, f"{reference}: directory is missing"
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return False, f"{reference}: config.json is missing"

    index_paths = sorted(model_dir.glob("*.safetensors.index.json"))
    weight_paths: list[Path]
    if index_paths:
        try:
            index = json.loads(index_paths[0].read_text())
            shard_names = sorted(set(index["weight_map"].values()))
        except (OSError, ValueError, KeyError, TypeError) as error:
            return False, f"{reference}: invalid weight index: {error}"
        weight_paths = [model_dir / name for name in shard_names]
        missing = [
            path.name
            for path in weight_paths
            if not path.is_file() or path.stat().st_size == 0
        ]
        if missing:
            preview = ", ".join(missing[:3])
            suffix = "..." if len(missing) > 3 else ""
            return False, f"{reference}: missing weight shards: {preview}{suffix}"
    else:
        weight_paths = sorted(model_dir.glob("*.safetensors"))
        if not weight_paths:
            return False, f"{reference}: no safetensors weights found"

    size_gib = sum(path.stat().st_size for path in weight_paths) / (1024**3)
    return True, f"{reference}: {len(weight_paths)} shard(s), {size_gib:.1f} GiB"


def check_models(checks: list[Check]) -> None:
    model_root = os.getenv("COEVO_MODEL_ROOT", "/models")
    models = {
        "shared-policy-model": os.getenv("COEVO_POLICY_PATH", f"{model_root}/policy"),
        "buyer-model": os.getenv("COEVO_BUYER_PATH", f"{model_root}/policy"),
    }
    if stage_reward_needs_previous():
        models["previous-policy-model"] = os.getenv(
            "COEVO_PREVIOUS_POLICY_PATH", ""
        )
    for name, reference in models.items():
        if local_model(reference):
            valid, detail = validate_local_model(reference)
        else:
            downloads_allowed = os.getenv("COEVO_ALLOW_DOWNLOADS") == "1"
            valid = bool(reference) and downloads_allowed
            detail = (
                f"remote model id: {reference}"
                if valid
                else "local model path required while downloads are disabled"
            )
        add(checks, name, valid, detail)


def ports() -> dict[str, int]:
    values = {
        "policy": int(os.getenv("COEVO_POLICY_PORT", "8000")),
        "buyer": int(os.getenv("COEVO_BUYER_PORT", "8002")),
        "rollout": int(os.getenv("COEVO_BUYER_ROLLOUT_PORT", "8003")),
    }
    if stage_reward_needs_previous():
        values["policy_previous"] = int(
            os.getenv("COEVO_PREVIOUS_POLICY_PORT", "8001")
        )
    return values


def check_free_ports(checks: list[Check]) -> None:
    for role, port in ports().items():
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", port))
            free = True
        except OSError as error:
            free = False
            detail = str(error)
        else:
            detail = f"127.0.0.1:{port} is free"
        finally:
            if sock is not None:
                sock.close()
        add(checks, f"port:{role}", free, detail)


def service_urls() -> dict[str, str]:
    role_ports = ports()
    values = {
        "policy": os.getenv(
            "COEVO_POLICY_URL", f"http://127.0.0.1:{role_ports['policy']}"
        ),
        "buyer": os.getenv(
            "COEVO_BUYER_URL", f"http://127.0.0.1:{role_ports['buyer']}"
        ),
        "rollout": f"http://127.0.0.1:{role_ports['rollout']}",
    }
    if "policy_previous" in role_ports:
        values["policy_previous"] = os.getenv(
            "COEVO_PREVIOUS_POLICY_URL",
            f"http://127.0.0.1:{role_ports['policy_previous']}",
        )
    return values


def check_services(checks: list[Check]) -> None:
    # Managed inference services are loopback-only.  Ignore inherited proxy
    # variables here just as wait_for_servers.py does, otherwise a healthy local
    # endpoint can be reported as unavailable through an unrelated proxy.
    opener = build_opener(ProxyHandler({}))
    for role, base_url in service_urls().items():
        url = base_url.rstrip("/") + (
            "/health/" if role == "rollout" else "/v1/models"
        )
        try:
            with opener.open(url, timeout=3) as response:
                payload = json.load(response)
            detail = (
                payload.get("status", "ready")
                if role == "rollout"
                else payload["data"][0]["id"]
            )
        except (OSError, URLError, ValueError, KeyError, IndexError) as error:
            add(checks, f"service:{role}", False, f"{url}: {error}")
        else:
            add(checks, f"service:{role}", True, f"{url}: {detail}")


def parse_gpu_ids(value: str) -> set[int]:
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def check_gpus(checks: list[Check], require_all_free: bool) -> None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        add(checks, "gpus", False, "nvidia-smi is not available")
        return
    if result.returncode != 0:
        add(checks, "gpus", False, result.stderr.strip() or "nvidia-smi failed")
        return
    inventory = {}
    for line in result.stdout.splitlines():
        index, used, total = (int(value.strip()) for value in line.split(","))
        inventory[index] = (used, total)

    service_gpu_sets = {
        "policy": parse_gpu_ids(os.getenv("COEVO_POLICY_GPUS", "0")),
        "buyer": parse_gpu_ids(
            os.getenv("COEVO_BUYER_GPUS", os.getenv("COEVO_BUYER_GPU", "1"))
        ),
        "rollout": parse_gpu_ids(
            os.getenv(
                "COEVO_BUYER_ROLLOUT_GPUS",
                os.getenv("COEVO_BUYER_ROLLOUT_GPU", "2"),
            )
        ),
    }
    if stage_reward_needs_previous():
        service_gpu_sets["policy_previous"] = parse_gpu_ids(
            os.getenv("COEVO_PREVIOUS_POLICY_GPUS", "4")
        )
    policy_train = parse_gpu_ids(os.getenv("COEVO_POLICY_TRAIN_GPUS", "3"))
    buyer_train = parse_gpu_ids(os.getenv("COEVO_BUYER_TRAIN_GPUS", "3"))
    allocated = {}
    overlap = []
    for role, gpu_ids in service_gpu_sets.items():
        for gpu_id in gpu_ids:
            if gpu_id in allocated:
                overlap.append(f"GPU {gpu_id}: {allocated[gpu_id]} and {role}")
            allocated[gpu_id] = role
    for role, gpu_ids in (
        ("policy-trainer", policy_train),
        ("buyer-trainer", buyer_train),
    ):
        for gpu_id in gpu_ids:
            if gpu_id in allocated:
                overlap.append(f"GPU {gpu_id}: {allocated[gpu_id]} and {role}")
    add(
        checks,
        "gpu-layout",
        not overlap,
        "; ".join(overlap) or "no service/trainer overlap",
    )

    requested = set().union(*service_gpu_sets.values(), policy_train, buyer_train)
    missing = sorted(requested - inventory.keys())
    add(
        checks,
        "gpu-count",
        not missing,
        f"available={len(inventory)}, requested={sorted(requested)}, missing={missing}",
    )
    allow_busy = os.getenv("COEVO_ALLOW_BUSY_GPUS") == "1"
    max_used = int(os.getenv("COEVO_MAX_USED_GPU_MIB_BEFORE_START", "2048"))
    must_be_free = requested if require_all_free else policy_train | buyer_train
    busy = {
        gpu_id: inventory[gpu_id]
        for gpu_id in must_be_free & inventory.keys()
        if inventory[gpu_id][0] > max_used
    }
    detail = ", ".join(
        f"{gpu_id}:{used}/{total} MiB" for gpu_id, (used, total) in sorted(busy.items())
    )
    add(
        checks,
        "gpu-memory",
        not busy,
        detail or f"all requested GPUs use <= {max_used} MiB",
        warn=allow_busy and bool(busy),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate co-evolution runtime prerequisites"
    )
    parser.add_argument("mode", choices=("python", "start", "services"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    checks = []
    check_python(checks)
    check_upstreams(checks)
    if args.mode == "start":
        check_models(checks)
        check_free_ports(checks)
        check_gpus(checks, require_all_free=True)
    elif args.mode == "services":
        check_models(checks)
        check_services(checks)
        check_gpus(checks, require_all_free=False)

    if args.json:
        print(json.dumps([asdict(check) for check in checks], indent=2))
    else:
        labels = {"ok": "OK", "warn": "WARN", "fail": "FAIL"}
        for check in checks:
            print(f"[{labels[check.status]}] {check.name}: {check.detail}")
    if any(check.status == "fail" for check in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
