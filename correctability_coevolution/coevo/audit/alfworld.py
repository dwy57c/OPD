from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


_NAVIGATION = re.compile(r"\b(?:go|walk|move|navigate)\s+to\s+(.+?)(?:\.|$)", re.I)
_QUERY = re.compile(r"\b(?:look|open|examine|inspect)\b", re.I)
_PICKUP = re.compile(r"\b(?:take|pickup|pick\s+up)\b", re.I)


def _action_text(action: Any) -> str:
    if isinstance(action, str):
        return action.strip()
    if isinstance(action, Mapping):
        return str(
            action.get("action")
            or action.get("content")
            or action.get("command")
            or ""
        ).strip()
    return str(action).strip()


def _normalize_entity(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").split()).strip(" .")


@dataclass(frozen=True)
class AlfworldPrivilege:
    expert_actions: tuple[str, ...]
    goal_object_locations: Mapping[str, str]
    destination_receptacle: str | None
    unobserved_states: Mapping[str, str]

    def hint_payload(self) -> dict[str, Any]:
        return {
            "authoritative_oracle_steps": "\n".join(
                f"[Step {index}] {action}"
                for index, action in enumerate(self.expert_actions, start=1)
            ),
            "goal_object_locations": dict(self.goal_object_locations),
            "destination_receptacle": self.destination_receptacle,
            "unobserved_states": dict(self.unobserved_states),
        }


def privilege_from_agentgym_eto_record(record: Mapping[str, Any]) -> AlfworldPrivilege:
    """Normalize AgentGym/ETO expert traces plus simulator hidden state."""

    hidden = record.get("hidden_state") or {}
    locations = hidden.get("goal_object_locations") or record.get(
        "goal_object_locations"
    )
    if not isinstance(locations, Mapping) or not locations:
        raise ValueError("ALFWorld record is missing hidden goal_object_locations")
    expert = record.get("expert_actions") or record.get("expert_trajectory")
    if not isinstance(expert, list) or not expert:
        raise ValueError("ALFWorld record is missing an AgentGym/ETO expert trajectory")
    destination = hidden.get("destination_receptacle") or record.get(
        "destination_receptacle"
    )
    states = hidden.get("unobserved_states") or record.get("unobserved_states") or {}
    return AlfworldPrivilege(
        expert_actions=tuple(_action_text(action) for action in expert),
        goal_object_locations={str(key): str(value) for key, value in locations.items()},
        destination_receptacle=str(destination) if destination else None,
        unobserved_states={str(key): str(value) for key, value in states.items()},
    )


@dataclass(frozen=True)
class AlfworldBehaviorReport:
    first_navigation_target: str | None
    direct_location_hit: bool
    queried_before_pickup: bool
    pickup_index: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_alfworld_behavior(
    actions: Iterable[Any], *, true_goal_locations: Iterable[str]
) -> AlfworldBehaviorReport:
    """Domain audit for unsupported navigation and pre-pickup observation."""

    texts = [_action_text(action) for action in actions]
    first_target = None
    for text in texts:
        match = _NAVIGATION.search(text)
        if match:
            first_target = _normalize_entity(match.group(1))
            break
    normalized_truth = {_normalize_entity(value) for value in true_goal_locations}
    pickup_index = next(
        (index for index, text in enumerate(texts) if _PICKUP.search(text)), None
    )
    prefix = texts if pickup_index is None else texts[:pickup_index]
    return AlfworldBehaviorReport(
        first_navigation_target=first_target,
        direct_location_hit=first_target in normalized_truth if first_target else False,
        queried_before_pickup=any(_QUERY.search(text) for text in prefix),
        pickup_index=pickup_index,
    )


def load_agentgym_eto_split(root: Path, split: str) -> list[dict[str, Any]]:
    """Load the canonical ALFWorld train/valid_seen/valid_unseen trajectory split."""

    aliases = {"train": "train", "valid_seen": "valid_seen", "valid_unseen": "valid_unseen"}
    if split not in aliases:
        raise ValueError(f"unsupported AgentGym/ETO ALFWorld split: {split}")
    split_root = root / aliases[split]
    if not split_root.is_dir():
        raise FileNotFoundError(split_root)
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(split_root.rglob("*.json"))
    ]
