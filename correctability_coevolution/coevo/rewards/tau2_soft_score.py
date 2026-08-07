from dataclasses import asdict, dataclass
from typing import Any


def _value(item: Any, name: str, default=None):
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _basis_name(item: Any) -> str:
    value = getattr(item, "value", item)
    return str(value).upper()


@dataclass(frozen=True)
class CategoryScore:
    numerator: float
    denominator: int

    @property
    def score(self) -> float:
        return self.numerator / self.denominator if self.denominator else 0.0

    def to_dict(self) -> dict:
        return {**asdict(self), "score": self.score}


@dataclass(frozen=True)
class SoftScoreResult:
    score: float
    categories: dict[str, CategoryScore]
    reward_basis: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "categories": {
                name: category.to_dict() for name, category in self.categories.items()
            },
            "reward_basis": list(self.reward_basis),
        }


def _boolean_category(items, attribute: str) -> CategoryScore:
    checks = list(items or [])
    numerator = sum(float(bool(_value(item, attribute, False))) for item in checks)
    return CategoryScore(numerator=numerator, denominator=len(checks))


def soft_completion_score(reward_info) -> SoftScoreResult:
    """Build the category-balanced dense τ² completion score.

    Only categories enabled by ``reward_basis`` participate in the outer mean.
    Missing checks in an enabled category are fail-closed (score zero). DB and
    environment assertions form one Environment category, so numerous action
    checks cannot dominate other criterion families.
    """
    if reward_info is None:
        return SoftScoreResult(0.0, {}, ())

    basis = tuple(_basis_name(item) for item in (_value(reward_info, "reward_basis") or ()))
    enabled = set(basis)
    categories: dict[str, CategoryScore] = {}

    if "ACTION" in enabled:
        categories["action"] = _boolean_category(
            _value(reward_info, "action_checks"), "action_match"
        )

    if "COMMUNICATE" in enabled:
        categories["communication"] = _boolean_category(
            _value(reward_info, "communicate_checks"), "met"
        )

    if "DB" in enabled or "ENV_ASSERTION" in enabled:
        numerator = 0.0
        denominator = 0
        if "DB" in enabled:
            denominator += 1
            db_check = _value(reward_info, "db_check")
            if db_check is not None:
                numerator += float(bool(_value(db_check, "db_match", False)))
        if "ENV_ASSERTION" in enabled:
            checks = list(_value(reward_info, "env_assertions") or [])
            denominator += len(checks)
            numerator += sum(float(bool(_value(item, "met", False))) for item in checks)
            # An enabled but missing assertion list is malformed and fail-closed.
            if not checks:
                denominator += 1
        categories["environment"] = CategoryScore(numerator, denominator)

    if "NL_ASSERTION" in enabled:
        categories["nl_assertions"] = _boolean_category(
            _value(reward_info, "nl_assertions"), "met"
        )

    score = (
        sum(category.score for category in categories.values()) / len(categories)
        if categories
        else 0.0
    )
    return SoftScoreResult(score=score, categories=categories, reward_basis=basis)
