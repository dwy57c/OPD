from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 4


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def assistant_action_payload(action: dict) -> dict:
    """Keep only OpenAI message fields that affect assistant serialization."""
    if action.get("role") != "assistant":
        raise ValueError("Teacher target must be an assistant action")
    payload = {
        "role": "assistant",
        "content": action.get("content") or "",
    }
    if action.get("tool_calls"):
        payload["tool_calls"] = action["tool_calls"]
    return payload


def assistant_action_hash(action: dict) -> str:
    return canonical_hash(assistant_action_payload(action))


def model_manifest_revision(reference: str) -> str:
    """Return a cheap, deterministic identity without hashing model weights."""
    if not reference:
        return ""
    path = Path(reference).expanduser()
    if not path.is_dir():
        return f"model-id:{reference}"
    metadata_names = (
        "config.json",
        "adapter_config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    manifest = {"path": str(path.resolve()), "metadata": {}, "weights": []}
    for name in metadata_names:
        candidate = path / name
        if candidate.is_file():
            manifest["metadata"][name] = sha256(candidate.read_bytes()).hexdigest()
    for candidate in sorted(path.glob("*.safetensors")):
        manifest["weights"].append((candidate.name, candidate.stat().st_size))
    return f"local-manifest-sha256:{canonical_hash(manifest)}"


@dataclass(frozen=True)
class ArtifactContract:
    schema_version: int
    target_schema_version: int
    round_index: int
    tokenizer_id: str
    tokenizer_hash: str
    teacher_target_version: str
    hint_level: str
    reward_name: str
    reward_formula_version: str
    student_checkpoint_current: str
    student_checkpoint_previous: str
    buyer_checkpoint: str
    student_revision_current: str
    student_revision_previous: str
    buyer_revision: str

    @classmethod
    def from_mapping(cls, value: dict, *, source: str) -> "ArtifactContract":
        required = tuple(cls.__dataclass_fields__)
        missing = [field for field in required if field not in value]
        if missing:
            raise ValueError(
                f"{source} is missing artifact contract fields: {', '.join(missing)}"
            )
        contract = cls(**{field: value[field] for field in required})
        if contract.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"{source} schema_version={contract.schema_version!r}; "
                f"expected {SCHEMA_VERSION}. Recollect or migrate before merging."
            )
        if contract.target_schema_version != 2:
            raise ValueError(
                f"{source} target_schema_version={contract.target_schema_version!r}; "
                "expected 2. Recollect or migrate before merging."
            )
        if (
            not contract.tokenizer_id
            or not contract.tokenizer_hash
            or not contract.teacher_target_version
            or not contract.hint_level
        ):
            raise ValueError(
                f"{source} must identify tokenizer and Teacher-target construction"
            )
        return contract

    def to_dict(self) -> dict:
        return {
            field: getattr(self, field) for field in self.__dataclass_fields__
        }


def artifact_metadata(config) -> dict:
    from coevo.rewards.stage_progress import REWARD_FORMULA_VERSION, REWARD_NAME

    policy_model = str(getattr(getattr(config, "policy", None), "model", "policy"))
    current_checkpoint = str(getattr(config, "current_policy_checkpoint", ""))
    previous_checkpoint = str(getattr(config, "previous_policy_checkpoint", ""))
    buyer_checkpoint = str(getattr(config, "buyer_checkpoint", ""))
    return ArtifactContract(
        schema_version=int(getattr(config, "dataset_schema_version", SCHEMA_VERSION)),
        target_schema_version=int(getattr(config, "target_schema_version", 2)),
        round_index=int(getattr(config, "round_index", 0)),
        tokenizer_id=str(getattr(config, "tokenizer_id", "") or policy_model),
        tokenizer_hash=canonical_hash(
            str(getattr(config, "tokenizer_id", "") or policy_model)
        ),
        teacher_target_version=str(
            getattr(
                config,
                "teacher_target_version",
                "hint-ladder-raw-v1",
            )
        ),
        hint_level=str(
            getattr(getattr(config, "hint_level", None), "value", None)
            or getattr(config, "hint_level", "L3_ORACLE")
        ),
        reward_name=REWARD_NAME,
        reward_formula_version=REWARD_FORMULA_VERSION,
        student_checkpoint_current=current_checkpoint,
        student_checkpoint_previous=previous_checkpoint,
        buyer_checkpoint=buyer_checkpoint,
        student_revision_current=str(
            getattr(config, "current_policy_revision", "")
            or model_manifest_revision(current_checkpoint)
        ),
        student_revision_previous=str(
            getattr(config, "previous_policy_revision", "")
            or model_manifest_revision(previous_checkpoint)
        ),
        buyer_revision=str(
            getattr(config, "buyer_revision", "")
            or model_manifest_revision(buyer_checkpoint)
        ),
    ).to_dict()


def validate_compatible_artifacts(
    values: Iterable[tuple[str, dict]],
    *,
    require_same_provenance: bool = True,
) -> ArtifactContract:
    contracts = [
        ArtifactContract.from_mapping(value, source=source)
        for source, value in values
    ]
    if not contracts:
        raise ValueError("no artifacts were provided for compatibility validation")
    expected = contracts[0]
    if require_same_provenance:
        incompatible = [contract for contract in contracts[1:] if contract != expected]
    else:
        expected_key = (
            expected.schema_version,
            expected.target_schema_version,
            expected.tokenizer_id,
            expected.tokenizer_hash,
            expected.teacher_target_version,
            expected.hint_level,
            expected.reward_name,
            expected.reward_formula_version,
        )
        incompatible = [
            contract
            for contract in contracts[1:]
            if (
                contract.schema_version,
                contract.target_schema_version,
                contract.tokenizer_id,
                contract.tokenizer_hash,
                contract.teacher_target_version,
                contract.hint_level,
                contract.reward_name,
                contract.reward_formula_version,
            )
            != expected_key
        ]
    if incompatible:
        raise ValueError(
            "refusing to combine incompatible artifact contracts; expected "
            f"{canonical_json(expected.to_dict())}, got "
            f"{canonical_json(incompatible[0].to_dict())}"
        )
    return expected


def dataset_fingerprint(*row_groups: list[dict]) -> str:
    digest = sha256()
    for rows in row_groups:
        for row in rows:
            digest.update(canonical_json(row).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()
