from dataclasses import asdict, dataclass
from typing import Callable


@dataclass(frozen=True)
class ShadowOPDConfig:
    optimizer_steps: int
    learning_rate: float
    token_budget: int
    lora_rank: int
    batch_size: int

    def __post_init__(self):
        if self.optimizer_steps < 1:
            raise ValueError("optimizer_steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.token_budget < 1:
            raise ValueError("token_budget must be positive")
        if self.lora_rank < 1:
            raise ValueError("lora_rank must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")


@dataclass(frozen=True)
class ShadowTrainStats:
    optimizer_steps: int
    effective_tokens: int


@dataclass(frozen=True)
class ShadowOPDResult:
    probe_loss_before: float
    probe_loss_after: float
    effective_tokens: int
    optimizer_steps: int
    gain_per_token: float
    config: ShadowOPDConfig

    def to_dict(self) -> dict:
        return {**asdict(self), "config": asdict(self.config)}


class ShadowOPDEvaluator:
    """Evaluate fixed-budget OPD in a disposable adapter context.

    ``adapter_factory`` must return a context manager; its ``__exit__`` owns
    deletion of the temporary LoRA even when training or probing fails.
    """

    def __init__(
        self,
        *,
        config: ShadowOPDConfig,
        adapter_factory: Callable,
        trainer: Callable,
        probe_loss: Callable,
    ):
        self.config = config
        self.adapter_factory = adapter_factory
        self.trainer = trainer
        self.probe_loss = probe_loss

    def evaluate(self, base_model, dataset, probe_set) -> ShadowOPDResult:
        if not dataset:
            raise ValueError("Shadow OPD requires non-empty positive-utility data")
        before = float(self.probe_loss(base_model, probe_set))
        with self.adapter_factory(base_model, self.config) as adapter:
            stats = self.trainer(adapter, dataset, self.config)
            if not isinstance(stats, ShadowTrainStats):
                raise TypeError("Shadow trainer must return ShadowTrainStats")
            if stats.optimizer_steps != self.config.optimizer_steps:
                raise ValueError("Shadow trainer did not use the fixed optimizer budget")
            if not 0 < stats.effective_tokens <= self.config.token_budget:
                raise ValueError("Shadow trainer violated the fixed token budget")
            after = float(self.probe_loss(adapter, probe_set))
        return ShadowOPDResult(
            probe_loss_before=before,
            probe_loss_after=after,
            effective_tokens=stats.effective_tokens,
            optimizer_steps=stats.optimizer_steps,
            gain_per_token=(before - after) / stats.effective_tokens,
            config=self.config,
        )
