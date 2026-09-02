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
    hint_error_trials: int = 0

    def __post_init__(self) -> None:
        if len(self.rewards) != self.k:
            raise ValueError("probe rewards must contain exactly k trials")
        if not 0 <= self.hint_error_trials <= self.k:
            raise ValueError("hint_error_trials must be in [0, k]")
        if not 0 <= self.successes <= self.valid_trials:
            raise ValueError("successes must count only valid probe trials")

    @property
    def valid_trials(self) -> int:
        return self.k - self.hint_error_trials

    @property
    def measured(self) -> bool:
        return self.valid_trials * 2 >= self.k

    @property
    def success_rate(self) -> float | None:
        if not self.measured:
            return None
        return self.successes / self.valid_trials

    @property
    def pass_at_k(self) -> float | None:
        return float(self.successes > 0) if self.measured else None

    def to_dict(self) -> dict:
        value = asdict(self)
        value["hint_level"] = self.hint_level.value
        value["success_rate"] = self.success_rate
        value["pass_at_k"] = self.pass_at_k
        value["valid_trials"] = self.valid_trials
        value["measured"] = self.measured
        value["rewards"] = list(self.rewards)
        return value


class ScenarioBand(str, Enum):
    MASTERED = "mastered"
    FRONTIER = "frontier"
    SCAFFOLDED = "scaffolded"
    OUT_OF_REACH = "out_of_reach"
    UNMEASURED = "unmeasured"


@dataclass(frozen=True)
class HStarDecision:
    level: HintLevel | None
    band: ScenarioBand
    no_hint_score: float
    best_hint_score: float | None
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
    valid_rewards = []
    hint_error_trials = 0
    for offset in range(k):
        environment = Tau2Environment(level_config)
        policy = "student" if parsed is HintLevel.L0_NONE else "teacher"
        seed = config.seed + offset
        trial_hint_error = False
        if trial is not None:
            reward = float(trial(environment, policy, seed))
        else:
            orchestrator = environment.orchestrator(
                environment.initial_history(), policy, seed=seed
            )
            if parsed is not HintLevel.L0_NONE:
                if not hasattr(orchestrator.agent, "refresh_hint_each_turn"):
                    raise ValueError(
                        "hinted pass@k probe requires a turn-refreshable Teacher"
                    )
                orchestrator.agent.refresh_hint_each_turn = True
            simulation = orchestrator.run()
            simulation.reward_info = environment.evaluate(simulation)
            reward = float(simulation.reward_info.reward)
            trial_hint_error = parsed is not HintLevel.L0_NONE and any(
                record.get("error")
                for record in getattr(orchestrator.agent, "hint_records", [])
            )
            if trial_hint_error:
                hint_error_trials += 1
        rewards.append(reward)
        if not trial_hint_error:
            valid_rewards.append(reward)
    return ProbeResult(
        task_id=str(task_id),
        hint_level=parsed,
        k=k,
        successes=sum(value > 0 for value in valid_rewards),
        rewards=tuple(rewards),
        hint_error_trials=hint_error_trials,
    )


def _score(value: ProbeResult | float) -> float | None:
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
    ordered = [scores[level] for level in HINT_LEVELS if scores[level] is not None]
    monotone = all(left <= right + 1e-12 for left, right in zip(ordered, ordered[1:]))
    no_hint = scores[HintLevel.L0_NONE]
    if no_hint is None:
        raise ValueError("L0 probe must be measured")
    measured_hints = [
        scores[level] for level in HINT_LEVELS[1:] if scores[level] is not None
    ]
    best_hint = max(measured_hints) if measured_hints else None

    if no_hint >= sufficient:
        return HStarDecision(
            HintLevel.L0_NONE,
            ScenarioBand.MASTERED,
            no_hint,
            best_hint,
            monotone,
            "unhinted checkpoint already meets the sufficient threshold",
        )
    candidates = [
        level
        for level in HINT_LEVELS[1:]
        if scores[level] is not None and scores[level] >= sufficient
    ]
    if not candidates:
        if any(scores[level] is None for level in HINT_LEVELS[1:]):
            return HStarDecision(
                None,
                ScenarioBand.UNMEASURED,
                no_hint,
                best_hint,
                monotone,
                "one or more hinted levels had too few valid trials",
            )
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
        fallback = (
            groups[ScenarioBand.MASTERED]
            + groups[ScenarioBand.OUT_OF_REACH]
        )
        if not fallback:
            raise ValueError("cannot schedule a curriculum with no measured task")
        distribute(fallback, 1.0)
    total = sum(result.values())
    if total == 0:
        raise ValueError("cannot schedule an empty decision set")
    return {task_id: weight / total for task_id, weight in result.items()}
