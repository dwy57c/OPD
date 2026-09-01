from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Callable, Mapping

from coevo.config import InfraConfig
from coevo.environment import Tau2Environment
from coevo.hints import HINT_LEVELS, HintLevel


@dataclass(frozen=True)
class ProbeResult:
    task_id: str
    hint_level: HintLevel
    k: int
    successes: int
    rewards: tuple[float, ...]

    @property
    def success_rate(self) -> float:
        return self.successes / self.k

    @property
    def pass_at_k(self) -> float:
        return float(self.successes > 0)

    def to_dict(self) -> dict:
        value = asdict(self)
        value["hint_level"] = self.hint_level.value
        value["success_rate"] = self.success_rate
        value["pass_at_k"] = self.pass_at_k
        value["rewards"] = list(self.rewards)
        return value


class ScenarioBand(str, Enum):
    MASTERED = "mastered"
    FRONTIER = "frontier"
    SCAFFOLDED = "scaffolded"
    OUT_OF_REACH = "out_of_reach"


@dataclass(frozen=True)
class HStarDecision:
    level: HintLevel | None
    band: ScenarioBand
    no_hint_score: float
    best_hint_score: float
    monotone: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "level": self.level.value if self.level is not None else None,
            "band": self.band.value,
            "no_hint_score": self.no_hint_score,
            "best_hint_score": self.best_hint_score,
            "monotone": self.monotone,
            "reason": self.reason,
        }


def probe_scenario(
    config: InfraConfig,
    task_id: str,
    level: HintLevel | str,
    k: int = 8,
    *,
    trial: Callable[[Tau2Environment, str, int], float] | None = None,
) -> ProbeResult:
    """Evaluate one frozen checkpoint and one dose over k independent seeds."""

    if k < 1:
        raise ValueError("k must be positive")
    parsed = HintLevel.parse(level)
    level_config = replace(config, task_id=str(task_id), hint_level=parsed)
    rewards = []
    for offset in range(k):
        environment = Tau2Environment(level_config)
        policy = "student" if parsed is HintLevel.L0_NONE else "teacher"
        seed = config.seed + offset
        if trial is not None:
            reward = float(trial(environment, policy, seed))
        else:
            orchestrator = environment.orchestrator(
                environment.initial_history(), policy, seed=seed
            )
            simulation = orchestrator.run()
            simulation.reward_info = environment.evaluate(simulation)
            reward = float(simulation.reward_info.reward)
        rewards.append(reward)
    return ProbeResult(
        task_id=str(task_id),
        hint_level=parsed,
        k=k,
        successes=sum(value > 0 for value in rewards),
        rewards=tuple(rewards),
    )


def _score(value: ProbeResult | float) -> float:
    return value.success_rate if isinstance(value, ProbeResult) else float(value)


def minimal_sufficient_level(
    results: Mapping[HintLevel | str, ProbeResult | float],
    *,
    sufficient: float = 0.5,
    near_zero: float = 0.05,
) -> HStarDecision:
    """Return the smallest sufficient dose and its four-quadrant curriculum band.

    The implementation verifies the monotonicity required by binary search. If
    empirical noise violates it, all four levels are scanned and the manifest
    records ``monotone=False`` rather than silently returning a wrong dose.
    """

    if not 0 <= near_zero < sufficient <= 1:
        raise ValueError("thresholds must satisfy 0 <= near_zero < sufficient <= 1")
    scores = {
        HintLevel.parse(level): _score(value) for level, value in results.items()
    }
    missing = [level for level in HINT_LEVELS if level not in scores]
    if missing:
        raise ValueError(f"missing probe levels: {[level.value for level in missing]}")
    ordered = [scores[level] for level in HINT_LEVELS]
    monotone = all(left <= right + 1e-12 for left, right in zip(ordered, ordered[1:]))
    no_hint = scores[HintLevel.L0_NONE]
    best_hint = max(scores[level] for level in HINT_LEVELS[1:])

    if no_hint >= sufficient:
        return HStarDecision(
            HintLevel.L0_NONE,
            ScenarioBand.MASTERED,
            no_hint,
            best_hint,
            monotone,
            "unhinted checkpoint already meets the sufficient threshold",
        )
    candidates = [level for level in HINT_LEVELS[1:] if scores[level] >= sufficient]
    if not candidates:
        return HStarDecision(
            None,
            ScenarioBand.OUT_OF_REACH,
            no_hint,
            best_hint,
            monotone,
            "no permitted hint level reaches the sufficient threshold",
        )
    level = candidates[0]
    band = ScenarioBand.FRONTIER if no_hint > near_zero else ScenarioBand.SCAFFOLDED
    return HStarDecision(
        level,
        band,
        no_hint,
        best_hint,
        monotone,
        "smallest empirically sufficient dose",
    )


def curriculum_weights(
    decisions: Mapping[str, HStarDecision],
    *,
    review_epsilon: float = 0.05,
    explore_epsilon: float = 0.05,
) -> dict[str, float]:
    """Allocate mass to frontier/scaffolded tasks plus review and exploration."""

    if review_epsilon < 0 or explore_epsilon < 0 or review_epsilon + explore_epsilon >= 1:
        raise ValueError("epsilon masses must be non-negative and sum to less than one")
    groups = {
        band: [task_id for task_id, value in decisions.items() if value.band is band]
        for band in ScenarioBand
    }
    result = {task_id: 0.0 for task_id in decisions}

    def distribute(task_ids: list[str], mass: float) -> None:
        if not task_ids:
            return
        share = mass / len(task_ids)
        for task_id in task_ids:
            result[task_id] += share

    active = groups[ScenarioBand.FRONTIER] + groups[ScenarioBand.SCAFFOLDED]
    main_mass = 1.0 - review_epsilon - explore_epsilon
    if active:
        distribute(active, main_mass)
        distribute(groups[ScenarioBand.MASTERED], review_epsilon)
        distribute(groups[ScenarioBand.OUT_OF_REACH], explore_epsilon)
    else:
        # Preserve a valid distribution during cold start or complete mastery.
        distribute(list(decisions), 1.0)
    total = sum(result.values())
    if total == 0:
        raise ValueError("cannot schedule an empty decision set")
    return {task_id: weight / total for task_id, weight in result.items()}
